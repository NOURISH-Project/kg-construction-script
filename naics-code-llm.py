#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import unicodedata
import re
from dataclasses import dataclass
from typing import List, Tuple
from difflib import SequenceMatcher

from neo4j import GraphDatabase

# ------------- Neo4j config (reuse your env vars) -------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# ------------- Neo4j: fetch FOODON items -------------
def fetch_foodon_items(limit: int) -> List[str]:
    """
    Uses your query:

        MATCH (n)
        WHERE toUpper(n.name) CONTAINS 'FOODON'
          AND n.out_degree = 1
        RETURN n.name_text AS item
        LIMIT $limit
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

    # sanitize & dedupe
    items = list(dict.fromkeys(x.strip() for x in items if x and x.strip()))
    return items


# ------------- NAICS model + loader -------------
@dataclass
class NaicsEntry:
    code: str
    title: str
    norm_title: str
    tokens: set


def _strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _normalize(text: str) -> str:
    text = _strip_diacritics((text or "").lower())
    # remove most punctuation
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set:
    return {t for t in _normalize(text).split() if t}


def load_naics(csv_path: str) -> List[NaicsEntry]:
    """
    Expect CSV columns:
        naics_code, naics_title

    Adjust fieldnames here if your file is different.
    """
    entries: List[NaicsEntry] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Allow some flexibility in column naming
        # Try common variants
        code_field = None
        title_field = None
        lower_fields = {name.lower(): name for name in reader.fieldnames}

        for cand in ["naics_code", "code"]:
            if cand in lower_fields:
                code_field = lower_fields[cand]
                break

        for cand in ["naics_title", "title", "naics_title_2022"]:
            if cand in lower_fields:
                title_field = lower_fields[cand]
                break

        if not code_field or not title_field:
            raise ValueError(
                f"Could not find NAICS code/title columns in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            code = (row.get(code_field) or "").strip()
            title = (row.get(title_field) or "").strip()
            if not code or not title:
                continue
            norm_title = _normalize(title)
            tokens = _tokenize(title)
            entries.append(NaicsEntry(code=code, title=title,
                                      norm_title=norm_title, tokens=tokens))
    return entries


# ------------- Matching heuristic -------------
def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _string_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def match_probability(foodon_name: str, naics: NaicsEntry) -> float:
    """
    Compute a similarity-based 'probability' in [0,1].

    Combination of:
      - token Jaccard similarity
      - character-level ratio
    """
    norm_foodon = _normalize(foodon_name)
    tokens_foodon = _tokenize(foodon_name)

    jacc = _jaccard(tokens_foodon, naics.tokens)
    ratio = _string_ratio(norm_foodon, naics.norm_title)

    # Simple weighted combination; you can tune these weights.
    prob = 0.6 * jacc + 0.4 * ratio

    # Cap within [0,1]
    if prob < 0:
        prob = 0.0
    elif prob > 1:
        prob = 1.0
    return prob


# ------------- Main matching logic -------------
def match_foodon_to_naics(
    foodon_items: List[str],
    naics_entries: List[NaicsEntry],
    min_score: float = 0.45
) -> List[Tuple[str, str, str, float, float]]:
    """
    Returns list of tuples:
        (foodon_name, naics_title, naics_code, match_prob, confidence_score)

    - multiple NAICS per foodon are allowed
    - matches with prob < min_score are dropped
    """
    results = []

    for food in foodon_items:
        best_for_food = []

        for naics in naics_entries:
            prob = match_probability(food, naics)
            if prob >= min_score:
                # For now, use same value for probability & confidence.
                confidence = prob
                best_for_food.append((naics, prob, confidence))

        # Sort matches for this item by probability, descending
        best_for_food.sort(key=lambda x: x[1], reverse=True)

        for naics, prob, conf in best_for_food:
            results.append(
                (food, naics.title, naics.code, round(prob, 4), round(conf, 4))
            )

    return results


# ------------- CSV output -------------
def write_matches_to_csv(
    matches: List[Tuple[str, str, str, float, float]],
    output_path: str
):
    """
    Columns:
      foodon_name, naics_name, naics_code, matching_probability, confidence_score
    """
    header = [
        "foodon_name",
        "naics_name",
        "naics_code",
        "matching_probability",
        "confidence_score",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in matches:
            w.writerow(row)


# ------------- CLI -------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Match FOODON items in Neo4j to NAICS codes via string similarity."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max number of FOODON items to pull from Neo4j (default: 200).",
    )
    p.add_argument(
        "--naics-csv",
        required=True,
        help="Path to NAICS 2022 CSV with code/title columns.",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="Minimum matching probability threshold (default: 0.45).",
    )
    p.add_argument(
        "--output",
        default="foodon_naics_matches.csv",
        help="Output CSV path (default: foodon_naics_matches.csv).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Get FOODON items from Neo4j
    food_items = fetch_foodon_items(args.limit)
    if not food_items:
        print("No FOODON items returned from Neo4j.")
        return
    print(f"Got {len(food_items)} FOODON items (sample: {food_items[:5]})")

    # 2) Load NAICS
    naics_entries = load_naics(args.naics_csv)
    print(f"Loaded {len(naics_entries)} NAICS entries from {args.naics_csv}")

    # 3) Match
    matches = match_foodon_to_naics(food_items, naics_entries, min_score=args.min_score)
    print(f"Found {len(matches)} FOODON–NAICS matches with score >= {args.min_score}")

    # 4) Write CSV
    write_matches_to_csv(matches, args.output)
    print(f"✅ Wrote matches to {args.output}")


if __name__ == "__main__":
    main()
