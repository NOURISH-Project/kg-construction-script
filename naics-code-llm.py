#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct FOODON → NAICS 2022 matching using OpenAI only.

Options:
- If you pass --item "basmati rice", it will match that single item.
- Otherwise it pulls FOODON items from Neo4j with your Cypher:

    MATCH (n)
    WHERE toUpper(n.name) CONTAINS 'FOODON'
      AND n.out_degree = 1
    RETURN n.name_text AS item
    LIMIT $limit

Output CSV columns:
  foodon_name, naics_name, naics_code, matching_probability, confidence_score
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


load_dotenv()
# ------------ ENV ------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------ Helpers ------------

def retry(max_tries=3, base_delay=0.5):
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

# ------------ Neo4j: fetch FOODON items ------------

def fetch_foodon_items(limit: int) -> List[str]:
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

    # sanitize + dedupe
    items = list(dict.fromkeys(x.strip() for x in items if x and x.strip()))
    return items

# ------------ OpenAI: NAICS matching for ONE item ------------

@retry(max_tries=3, base_delay=0.7)
def llm_match_naics_for_item(food_item: str) -> List[Dict[str, Any]]:
    """
    Ask OpenAI (NAICS 2022 knowledge) to return best NAICS categories
    for a single food item.

    Returns a list of objects:
      {
        "naics_code": "311212",
        "naics_title": "Rice Milling",
        "matching_probability": 0.9,
        "confidence_score": 0.85
      }
    """
    system_msg = (
        "You are an expert in the NAICS 2022 industrial classification system. "
        "Given a food item name, determine which NAICS 2022 codes best describe "
        "the primary industry that PRODUCES or SELLS this item. "
        "Base your answer on the official NAICS 2022 structure (codes and titles). "
        "Return between 0 and 5 candidate codes.\n\n"
        "For each candidate, output an object with:\n"
        "  naics_code (string, e.g., '311212'),\n"
        "  naics_title (official NAICS title),\n"
        "  matching_probability (float in [0,1]),\n"
        "  confidence_score (float in [0,1]).\n\n"
        "matching_probability = how likely this NAICS code is correct.\n"
        "confidence_score = how confident you are in that probability "
        "given possible ambiguity.\n\n"
        "Return ONLY a JSON array. Example format:\n"
        "[{\"naics_code\":\"311212\",\"naics_title\":\"Rice Milling\","
        "\"matching_probability\":0.9,\"confidence_score\":0.85}]"
    )

    user_msg = f"Food item: {food_item}\n\nReturn JSON array as specified."

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content.strip()
    data = json_relaxed_load(content)
    if not isinstance(data, list):
        raise ValueError("NAICS matcher returned non-list JSON")
    return data

# ------------ Bulk matching ------------

def match_items_to_naics(
    items: List[str],
    min_prob: float = 0.4
) -> List[Tuple[str, str, str, float, float]]:
    """
    For each item, call OpenAI to get NAICS candidates,
    filter by matching_probability >= min_prob.

    Returns rows:
      (foodon_name, naics_title, naics_code, matching_probability, confidence_score)
    """
    results: List[Tuple[str, str, str, float, float]] = []

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] Matching NAICS for: {item!r}")
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
            if not code or not title:
                continue
            try:
                prob = float(m.get("matching_probability", 0.0))
            except Exception:
                prob = 0.0
            try:
                conf = float(m.get("confidence_score", prob))
            except Exception:
                conf = prob

            # clamp
            prob = max(0.0, min(1.0, prob))
            conf = max(0.0, min(1.0, conf))

            if prob < min_prob:
                continue

            results.append(
                (item, title, code, round(prob, 4), round(conf, 4))
            )

    return results

# ------------ CSV output ------------

def write_csv(
    rows: List[Tuple[str, str, str, float, float]],
    path: str
):
    header = [
        "foodon_name",
        "naics_name",
        "naics_code",
        "matching_probability",
        "confidence_score",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ------------ CLI ------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Match FOODON names to NAICS 2022 codes using OpenAI (no NAICS CSV)."
    )
    p.add_argument(
        "--item",
        help="Single food item name to match (skips Neo4j).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="If no --item given, max FOODON items to fetch from Neo4j (default: 100).",
    )
    p.add_argument(
        "--min-prob",
        type=float,
        default=0.4,
        help="Minimum matching_probability to keep a match (default: 0.4).",
    )
    p.add_argument(
        "--output",
        default="foodon_naics_llm_direct.csv",
        help="Output CSV file (default: foodon_naics_llm_direct.csv).",
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
    print(f"✅ Wrote matches to {args.output}")

if __name__ == "__main__":
    main()
