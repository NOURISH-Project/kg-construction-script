#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FOODON → NAICS 2022 matching using OpenAI only,
with ontology-aligned relationship types.

Features:
- Fetch FOODON item names from Neo4j using your Cypher:
    MATCH (n)
    WHERE toUpper(n.name) CONTAINS 'FOODON'
      AND n.out_degree = 1
    RETURN n.name_text AS item
    LIMIT $limit

- For each food item, use OpenAI (NAICS 2022 knowledge) to:
    * choose one or more NAICS codes
    * assign a relation_type describing how that NAICS industry relates
      to the food item (produced_by, processed_by, sold_by, consumed_by, etc.)
    * assign matching_probability and confidence_score in [0,1]

- Output CSV:
    foodon_name, relation_type, naics_name, naics_code,
    matching_probability, confidence_score

- Optional Cypher output:
    MERGE (f:FoodItem {name: ...})
    MERGE (n:NAICS {code: ..., title: ...})
    MERGE (f)-[:RELATION {prob:..., confidence:...}]->(n)
"""

import os
import csv
import re
import json
import time
import argparse
from functools import wraps
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# ---------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# RELATION TYPE VOCABULARY (from your list)
# ---------------------------------------------------------------------

ALL_REL_TYPES = [
    # ---------------------------------------------------------
    # PRODUCTION / CULTIVATION
    # ---------------------------------------------------------
    "produced_by",
    "cultivated_by",
    "harvested_by",
    "grown_by",
    "raised_by",
    "caught_by",
    "foraged_by",
    "bred_by",
    "pollinated_by",

    # ---------------------------------------------------------
    # PROCESSING & MANUFACTURING
    # ---------------------------------------------------------
    "processed_by",
    "milled_by",
    "pressed_by",
    "fermented_by",
    "distilled_by",
    "brewed_by",
    "roasted_by",
    "ground_by",
    "baked_by",
    "churned_by",
    "slaughtered_by",
    "butchered_by",
    "smoked_by",
    "dried_by",
    "cured_by",

    # ---------------------------------------------------------
    # PACKAGING / CERTIFICATION / SAFETY
    # ---------------------------------------------------------
    "packaged_by",
    "labeled_by",
    "certified_by",
    "inspected_by",
    "graded_by",
    "tested_by",
    "analyzed_by",
    "sealed_by",

    # ---------------------------------------------------------
    # DISTRIBUTION / TRANSPORT / SALES / STORAGE
    # ---------------------------------------------------------
    "purchased_by",
    "sold_by",
    "transported_by",
    "imported_by",
    "exported_by",
    "warehoused_by",
    "stored_by",
    "refrigerated_by",
    "distributed_by",
    "delivered_by",

    # marketing/retail buckets
    "marketed_by",
    "retailed_by",
    "wholesaled_by",

    # ---------------------------------------------------------
    # PREPARATION / SERVING / CATERING
    # ---------------------------------------------------------
    "prepared_by",
    "served_by",
    "cooked_by",
    "assembled_by",
    "catered_by",

    # ---------------------------------------------------------
    # CULTURAL / CONSUMPTION / ETHNOGRAPHIC
    # ---------------------------------------------------------
    "consumed_by",
    "is_traditional_to",
    "is_staple_of",
    "is_originated_from",

    # ---------------------------------------------------------
    # ADDITIONAL SPECIALTY RELATIONS
    # ---------------------------------------------------------
    "is_ingredient_in",
    "is_substitute_for",
    "is_preserved_by",

    # ---------------------------------------------------------
    # TAG RELATIONS (non-binary)
    # ---------------------------------------------------------
    "is_EthnicElement_Of",
    "is_imported_From",
]

# For NAICS edges, restrict to relations that connect foods to businesses.
# (We exclude tag-like and ingredient-like relations.)
NAICS_REL_TYPES = [
    # production / cultivation
    "produced_by", "cultivated_by", "harvested_by", "grown_by",
    "raised_by", "caught_by", "foraged_by", "bred_by", "pollinated_by",
    # processing / manufacturing
    "processed_by", "milled_by", "pressed_by", "fermented_by", "distilled_by",
    "brewed_by", "roasted_by", "ground_by", "baked_by", "churned_by",
    "slaughtered_by", "butchered_by", "smoked_by", "dried_by", "cured_by",
    # packaging / compliance
    "packaged_by", "labeled_by", "certified_by", "inspected_by",
    "graded_by", "tested_by", "analyzed_by", "sealed_by",
    # distribution / logistics / retail
    "purchased_by", "sold_by", "transported_by", "imported_by", "exported_by",
    "warehoused_by", "stored_by", "refrigerated_by", "distributed_by",
    "delivered_by", "marketed_by", "retailed_by", "wholesaled_by",
    # preparation / foodservice / bulk consumption
    "prepared_by", "served_by", "cooked_by", "assembled_by", "catered_by",
    "consumed_by",
]

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def retry(max_tries: int = 3, base_delay: float = 0.5):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_tries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt + 1 == max_tries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Retry {fn.__name__} in {delay:.2f}s due to: {e}")
                    time.sleep(delay)
        return wrapper
    return deco


def json_relaxed_load(s: str):
    """Parse JSON possibly wrapped in ```json fences or with extra text."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL | re.IGNORECASE)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise

# ---------------------------------------------------------------------
# NEO4J: FETCH FOODON ITEMS
# ---------------------------------------------------------------------

def fetch_foodon_items(limit: int) -> List[str]:
    """
    Run your Cypher and return a de-duplicated list of FOODON item names.
    """
    q = """
        MATCH (n)
        WHERE toUpper(n.name) CONTAINS 'FOODON'
          AND n.out_degree = 1
        RETURN n.name_text AS item
        LIMIT $limit
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            rows = s.run(q, limit=limit)
            items = [r["item"] for r in rows if r["item"]]
    finally:
        driver.close()

    # sanitize + dedupe (preserve order)
    items = list(dict.fromkeys(x.strip() for x in items if x and x.strip()))
    return items

# ---------------------------------------------------------------------
# OPENAI: NAICS + RELATIONSHIP FOR ONE ITEM
# ---------------------------------------------------------------------

@retry(max_tries=3, base_delay=0.7)
def llm_match_naics_for_item(food_item: str) -> List[Dict[str, Any]]:
    """
    Ask OpenAI to return NAICS matches + relation_type for one food item.

    Returns a list of dicts:
      {
        "naics_code": "311212",
        "naics_title": "Rice Milling",
        "relation_type": "milled_by",
        "matching_probability": 0.91,
        "confidence_score": 0.87
      }
    """

    allowed_list_str = ", ".join(sorted(NAICS_REL_TYPES))

    system_msg = (
        "You are an expert in the NAICS 2022 industrial classification system "
        "and in food supply chains.\n\n"
        "TASK:\n"
        "Given a food item name, determine which NAICS 2022 codes are most relevant "
        "to the industries that interact with this item along the supply chain.\n\n"
        "For EACH NAICS code that you return, you must also choose a relationship_type "
        "that describes how that industry relates to the food item.\n\n"
        "The relationship_type MUST be one of the following exact strings:\n"
        f"{allowed_list_str}\n\n"
        "INTERPRETATION GUIDELINES:\n"
        "- Farms, fisheries, ranches: produced_by, cultivated_by, harvested_by, grown_by, "
        "  raised_by, caught_by, foraged_by, bred_by, pollinated_by.\n"
        "- Primary / secondary processing: processed_by, milled_by, pressed_by, fermented_by, "
        "  distilled_by, brewed_by, roasted_by, ground_by, baked_by, churned_by, slaughtered_by, "
        "  butchered_by, smoked_by, dried_by, cured_by.\n"
        "- Packaging & compliance: packaged_by, labeled_by, certified_by, inspected_by, "
        "  graded_by, tested_by, analyzed_by, sealed_by.\n"
        "- Distribution / logistics / retail: purchased_by, sold_by, transported_by, imported_by, "
        "  exported_by, warehoused_by, stored_by, refrigerated_by, distributed_by, delivered_by, "
        "  marketed_by, retailed_by, wholesaled_by.\n"
        "- Foodservice / consumption: prepared_by, served_by, cooked_by, assembled_by, catered_by, "
        "  consumed_by (for industries that purchase the item in bulk as input to meals).\n\n"
        "IMPORTANT:\n"
        "- Map ONLY to NAICS 2022 industries (businesses), not to regions or individual people.\n"
        "- Do NOT use tag-like relations such as is_ingredient_in, is_traditional_to, "
        "  is_staple_of, is_originated_from, is_EthnicElement_Of, or is_imported_From.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON array. Each element MUST be an object with keys:\n"
        "  naics_code           (number, e.g., '311212')\n"
        "  naics_title          (string, official NAICS title)\n"
        "  relation_type        (string, one of the allowed relation types above)\n"
        "  matching_probability (float in [0,1])\n"
        "  confidence_score     (float in [0,1])\n\n"
        "matching_probability = how likely this NAICS code with that relation_type is correct.\n"
        "confidence_score     = how confident you are, given ambiguity.\n\n"
        "Return between 0 and 8 objects. If nothing fits, return an empty array []."
    )

    user_msg = f"Food item: {food_item}\n\nReturn the JSON array as specified."

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.8,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",    "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content.strip()
    data = json_relaxed_load(content)
    if not isinstance(data, list):
        raise ValueError("NAICS matcher returned non-list JSON")

    # Enforce that relation_type is in NAICS_REL_TYPES
    cleaned: List[Dict[str, Any]] = []
    for obj in data:
        rel = str(obj.get("relation_type", "")).strip()
        if rel not in NAICS_REL_TYPES:
            continue
        cleaned.append(obj)

    return cleaned

# ---------------------------------------------------------------------
# BULK MATCHING
# ---------------------------------------------------------------------

def match_items_to_naics(
    items: List[str],
    min_prob: float = 0.4,
) -> List[Tuple[str, str, str, str, float, float]]:
    """
    For each food item, call OpenAI to get NAICS candidates with relationship_type,
    filter by matching_probability >= min_prob.

    Returns rows:
      (foodon_name, relation_type, naics_title, naics_code, matching_probability, confidence_score)
    """
    results: List[Tuple[str, str, str, str, float, float]] = []

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] Matching NAICS + relation for: {item!r}")
        try:
            matches = llm_match_naics_for_item(item)
        except Exception as e:
            print(f"  ⚠️ Error for {item!r}: {e}")
            continue

        if not matches:
            print("  → No NAICS matches returned.")
            continue

        for m in matches:
            code = str(m.get("naics_code", "")).strip()
            title = str(m.get("naics_title", "")).strip()
            rel   = str(m.get("relation_type", "")).strip()
            if not code or not title or not rel:
                continue

            try:
                prob = float(m.get("matching_probability", 0.0))
            except Exception:
                prob = 0.0
            try:
                conf = float(m.get("confidence_score", prob))
            except Exception:
                conf = prob

            # clamp values
            prob = max(0.0, min(1.0, prob))
            conf = max(0.0, min(1.0, conf))

            if prob < min_prob:
                continue

            results.append(
                (item, rel, title, code, round(prob, 4), round(conf, 4))
            )

    return results

# ---------------------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------------------

def write_csv(
    rows: List[Tuple[str, str, str, str, float, float]],
    path: str,
):
    header = [
        "foodon_name",
        "relation_type",
        "naics_name",
        "naics_code",
        "matching_probability",
        "confidence_score",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------------------------------------------------------------------
# CYPHER OUTPUT (OPTIONAL)
# ---------------------------------------------------------------------

def write_cypher(
    rows: List[Tuple[str, str, str, str, float, float]],
    path: str,
):
    """
    Emit Cypher that creates nodes/edges:
      (f:FoodItem {name: ...})
      (n:NAICS   {code: ..., title: ...})
      (f)-[:REL_TYPE {prob: ..., confidence: ...}]->(n)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Cypher for FOODON → NAICS relationships\n")
        f.write("// Relationship types drawn from your ontology relation list.\n\n")

        for foodon_name, rel_type, naics_title, naics_code, prob, conf in rows:
            rel_neo = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper())
            food_esc  = foodon_name.replace("'", "\\'")
            title_esc = naics_title.replace("'", "\\'")

            f.write(
                "MERGE (f:FoodItem {name: '%s'})\n"
                "MERGE (n:NAICS {code: '%s'})\n"
                "ON CREATE SET n.title = '%s'\n"
                "MERGE (f)-[r:%s {prob: %.4f, confidence: %.4f}]->(n)\n"
                ";\n\n"
                % (food_esc, naics_code, title_esc, rel_neo, prob, conf)
            )

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Match FOODON names to NAICS 2022 codes using OpenAI, "
            "assigning ontology-aligned relationship types."
        )
    )
    p.add_argument(
        "--item",
        help="Single food item name to match (skips Neo4j fetch).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="If no --item supplied, max FOODON items to fetch from Neo4j (default: 100).",
    )
    p.add_argument(
        "--min-prob",
        type=float,
        default=0.4,
        help="Minimum matching_probability threshold (default: 0.4).",
    )
    p.add_argument(
        "--output",
        default="foodon_naics_llm_rel.csv",
        help="Output CSV file (default: foodon_naics_llm_rel.csv).",
    )
    p.add_argument(
        "--emit-cypher",
        action="store_true",
        help="Also emit Cypher file to create FoodItem-NAICS relationships.",
    )
    p.add_argument(
        "--cypher-file",
        default="foodon_naics_relationships.cypher",
        help="Cypher output file (default: foodon_naics_relationships.cypher).",
    )
    return p.parse_args()

def main():
    args = parse_args()

    if args.item:
        items = [args.item.strip()]
    else:
        items = fetch_foodon_items(args.limit)
        if not items:
            print("No FOODON items found in Neo4j.")
            return
        print(f"Fetched {len(items)} FOODON items (sample: {items[:5]})")

    matches = match_items_to_naics(items, min_prob=args.min_prob)
    print(f"Total kept matches (prob >= {args.min_prob}): {len(matches)}")

    write_csv(matches, args.output)
    print(f"✅ Wrote CSV matches to {args.output}")

    if args.emit_cypher:
        write_cypher(matches, args.cypher_file)
        print(f"✅ Wrote Cypher relationships to {args.cypher_file}")

if __name__ == "__main__":
    main()
