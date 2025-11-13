#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import re
from decimal import Decimal, InvalidOperation

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

# ---------- Config ----------
RELTYPE_ALLOWED = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")  # valid Cypher rel type
DEFAULT_BATCH = 500

# ---------- Helpers ----------
def sanitize_reltype(rt: str) -> str:
    """
    Normalize a relation type to a safe Cypher label:
    - Uppercase, spaces/hyphens -> underscores
    - Must match RELTYPE_ALLOWED. Return None if invalid.
    """
    if rt is None:
        return None
    norm = rt.strip().upper().replace(" ", "_").replace("-", "_")
    return norm if RELTYPE_ALLOWED.match(norm) else None

def to_decimal(x):
    if x is None or x == "":
        return None
    try:
        return Decimal(str(x))
    except InvalidOperation:
        return None

# ---------- Cypher builders ----------
def cypher_merge_edge(reltype: str, require_out_degree_1: bool) -> str:
    """
    Build a Cypher query with an injected (validated) relationship type.
    We cannot parameterize the rel type, so we inject only after validation.
    """
    where_extra = " AND n.out_degree = 1" if require_out_degree_1 else ""
    return f"""
    MATCH (n)
    WHERE toUpper(n.name) CONTAINS 'FOODON'
      AND n.name_text = $food_item
      {where_extra}
    MERGE (org:FOODORG {{name_text: $org}})
    MERGE (n)-[r:{reltype}]->(org)
    SET r.score = $score
    RETURN count(r) AS created_or_merged
    """

CONSTRAINT_CYPHER = """
CREATE CONSTRAINT foodorg_name_text_unique IF NOT EXISTS
FOR (x:FOODORG) REQUIRE x.name_text IS UNIQUE
"""

# ---------- Main logic ----------
def upsert_from_csv(uri, user, pwd, csv_path, batch_size=DEFAULT_BATCH, require_out_degree_1=False, dry_run=False):
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    created = 0
    skipped = 0
    rows_seen = 0

    logging.info("Connecting to Neo4j...")
    with driver, driver.session() as sess:
        # Ensure constraint for FOODORG
        logging.info("Ensuring FOODORG(name_text) uniqueness constraint...")
        sess.run(CONSTRAINT_CYPHER)

        # Process CSV
        logging.info("Reading CSV: %s", csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"food_item", "relation", "org_or_value", "score"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

            tx = None
            tx_ops = 0
            try:
                for row in reader:
                    rows_seen += 1
                    food_item = (row.get("food_item") or "").strip()
                    relation  = (row.get("relation") or "").strip()
                    org_value = (row.get("org_or_value") or "").strip()
                    score     = to_decimal(row.get("score"))

                    reltype = sanitize_reltype(relation)
                    if not food_item or not org_value or not reltype:
                        skipped += 1
                        logging.warning("Skipping row %d (invalid fields). food_item=%r relation=%r org=%r",
                                        rows_seen, food_item, relation, org_value)
                        continue

                    params = {
                        "food_item": food_item,
                        "org": org_value,
                        "score": float(score) if score is not None else None,
                    }

                    if dry_run:
                        logging.info("[DRY-RUN] Would MERGE (%s)-[:%s {score:%s}]->(:FOODORG {%r})",
                                     food_item, reltype, params["score"], org_value)
                        continue

                    if tx is None:
                        tx = sess.begin_transaction()

                    query = cypher_merge_edge(reltype, require_out_degree_1)
                    tx.run(query, **params)
                    tx_ops += 1

                    if tx_ops >= batch_size:
                        tx.commit()
                        created += tx_ops
                        logging.info("Committed batch of %d operations (total committed: %d)", tx_ops, created)
                        tx = None
                        tx_ops = 0

                # Final commit
                if tx is not None and tx_ops > 0:
                    tx.commit()
                    created += tx_ops
                    logging.info("Committed final batch of %d operations (total committed: %d)", tx_ops, created)

            except Neo4jError as e:
                logging.error("Neo4jError encountered: %s", e)
                if tx is not None:
                    tx.rollback()
                raise
    return rows_seen, created, skipped

def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd  = os.getenv("NEO4J_PASSWORD")

    parser = argparse.ArgumentParser(description="Insert FOODON → FOODORG edges from CSV.")
    parser.add_argument("csv", help="Path to CSV with columns: food_item,relation,org_or_value,score")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Transaction batch size (default 500)")
    parser.add_argument("--require-out-degree-1", action="store_true",
                        help="Also require n.out_degree = 1 when matching the item node")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate only; no writes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not uri or not user or not pwd:
        raise SystemExit("Missing Neo4j credentials. Ensure NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are set in .env")

    rows_seen, created, skipped = upsert_from_csv(
        uri, user, pwd, args.csv, batch_size=args.batch_size,
        require_out_degree_1=args.require_out_degree_1, dry_run=args.dry_run
    )

    logging.info("Done. Rows seen: %d | relationships created/merged (in batches): %d | rows skipped: %d",
                 rows_seen, created, skipped)

if __name__ == "__main__":
    main()
