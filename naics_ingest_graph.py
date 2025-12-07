#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest FOODON → NAICS relationships from CSV into an existing Neo4j graph.

Assumptions:
- FOODON nodes already exist like:
    (:FOODON:entity {name_text: "...", ...})

- NAICS nodes already exist like (example):
    (:NAICS {
        naics_code: 722511,
        code: "722511",
        level: "specific_industry_group",
        title: "Full-Service Restaurants ",
        description: "...",
        version: "2022"
    })

CSV format (from your LLM matcher):
  foodon_name,relation_type,naics_name,naics_code,matching_probability,confidence_score

For each row, we:
  MATCH (f:FOODON:entity) WHERE toLower(trim(f.name_text)) = toLower(trim($foodon_name))
  MATCH (n:NAICS)
    WHERE (exists(n.naics_code) AND toString(n.naics_code) = $naics_code)
       OR (exists(n.code)      AND toString(n.code)      = $naics_code)
  MERGE (f)-[r:REL_TYPE]->(n)
  SET r.prob = $prob, r.confidence = $conf

No nodes are created; rows with missing FOODON or NAICS nodes are logged and skipped.
"""

import os
import csv
import re
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------
# Neo4j config from env
# ---------------------------------------------------------------------
load_dotenv()
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def sanitize_rel_type(rel: str) -> str:
    """
    Convert relation_type string to a valid Neo4j relationship type:
    - uppercase
    - replace non [A-Za-z0-9_] with "_"
    """
    rel = (rel or "").strip()
    rel = rel.upper()
    rel = re.sub(r"[^A-Z0-9_]", "_", rel)
    if not rel:
        rel = "RELATED_TO"
    return rel


def load_rows_from_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def ingest_rows_into_neo4j(rows: List[Dict[str, Any]]):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            batch_size = 500
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                summary = session.execute_write(_ingest_batch, batch)
                print(
                    f"✅ Ingested rows {i+1}–{i+len(batch)} / {len(rows)} "
                    f"(rels: {summary['rels_created']}, FOODON matched: {summary['foodon_matched']}, "
                    f"NAICS matched: {summary['naics_matched']}, skipped: {summary['skipped']})"
                )
    finally:
        driver.close()


def _ingest_batch(tx, batch: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Ingest a batch inside a write transaction.
    Returns simple counters.
    """
    rels_created = 0
    foodon_matched = 0
    naics_matched = 0
    skipped = 0

    debug_limit = 10
    debug_shown = 0

    for row in batch:
        foodon_name = (row.get("foodon_name") or "").strip()
        rel_type    = (row.get("relation_type") or "").strip()
        naics_code  = (row.get("naics_code") or "").strip()

        if not foodon_name or not naics_code or not rel_type:
            skipped += 1
            continue

        rel_neo = sanitize_rel_type(rel_type)

        try:
            prob = float(row.get("matching_probability", 0.0))
        except Exception:
            prob = 0.0
        try:
            conf = float(row.get("confidence_score", prob))
        except Exception:
            conf = prob

        query = f"""
        MATCH (f:FOODON:entity)
        WHERE toLower(trim(f.name_text)) = toLower(trim($foodon_name))
        MATCH (n:NAICS)
        WHERE (n.naics_code IS NOT NULL AND toString(n.naics_code) = $naics_code)
           OR (n.code       IS NOT NULL AND toString(n.code)       = $naics_code)
        MERGE (f)-[r:{rel_neo}]->(n)
        SET r.prob = $prob, r.confidence = $conf
        RETURN count(f) AS f_cnt, count(n) AS n_cnt, count(r) AS r_cnt
        """

        result = tx.run(
            query,
            foodon_name=foodon_name,
            naics_code=naics_code,
            prob=prob,
            conf=conf,
        ).single()

        if not result:
            skipped += 1
            continue

        f_cnt = result["f_cnt"]
        n_cnt = result["n_cnt"]
        r_cnt = result["r_cnt"]

        if f_cnt > 0:
            foodon_matched += 1
        if n_cnt > 0:
            naics_matched += 1
        if r_cnt > 0:
            rels_created += 1

        if (f_cnt == 0 or n_cnt == 0) and debug_shown < debug_limit:
            print(
                f"⚠️ No match for row: FOODON='{foodon_name}', NAICS='{naics_code}' "
                f"(f_cnt={f_cnt}, n_cnt={n_cnt})"
            )
            debug_shown += 1

    return {
        "rels_created": rels_created,
        "foodon_matched": foodon_matched,
        "naics_matched": naics_matched,
        "skipped": skipped,
    }



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Ingest FOODON → NAICS relationships CSV into Neo4j, "
            "matching existing (:FOODON:entity {name_text}) and (:NAICS {naics_code/code}). "
            "Edge properties prob/confidence are taken from the score columns."
        )
    )
    p.add_argument(
        "csv_path",
        help="Path to the CSV file (e.g., foodon_naics_llm_rel.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_rows_from_csv(args.csv_path)
    if not rows:
        print("No rows found in CSV.")
        return
    print(f"Loaded {len(rows)} rows from {args.csv_path}")
    ingest_rows_into_neo4j(rows)
    print("🎉 Ingestion complete.")


if __name__ == "__main__":
    main()
