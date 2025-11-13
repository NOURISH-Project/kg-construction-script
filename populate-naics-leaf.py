#!/usr/bin/env python3
import os
import math
from typing import List, Dict

import psycopg2
from psycopg2.extras import DictCursor
from neo4j import GraphDatabase
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# PostgreSQL configuration
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "postgres")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# Other runtime settings
ITEM_LIMIT = int(os.getenv("ITEM_LIMIT", "200"))
NAICS_VERSION = os.getenv("NAICS_VERSION", "2022")
TABLE_FQN = 'public."2022_NAICS_Keywords"'

# -----------------------------
# SQL and Cypher templates
# -----------------------------
SQL_COUNT = f"SELECT COUNT(*) FROM {TABLE_FQN};"

SQL_FETCH = f"""
SELECT DISTINCT
    TRIM(CAST(naics_code AS TEXT)) AS code,
    TRIM(CAST(naics_keywords AS TEXT)) AS keyword
FROM {TABLE_FQN}
WHERE naics_keywords IS NOT NULL AND naics_code IS NOT NULL
OFFSET %s LIMIT %s;
"""

CYPHER_CONSTRAINTS = """
CREATE CONSTRAINT naics_code_unique IF NOT EXISTS
FOR (n:NAICS) REQUIRE (n.code, n.version) IS UNIQUE;

CREATE CONSTRAINT term_unique IF NOT EXISTS
FOR (t:NAICSTerm) REQUIRE (t.code, t.title) IS UNIQUE;
"""

CYPHER_MERGE = """
UNWIND $rows AS row
WITH row
WHERE row.code IS NOT NULL AND row.title IS NOT NULL
MATCH (n:NAICS {code: row.code, version: $naics_version})
MERGE (t:NAICSTerm {code: row.code, title: row.title})
  ON CREATE SET t.level = 'leaf'
MERGE (t)-[:isSubClassOf]->(n);
"""

# -----------------------------
# Helper functions
# -----------------------------
def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_total(cur):
    cur.execute(SQL_COUNT)
    return cur.fetchone()[0]

def fetch_page(cur, offset: int, limit: int) -> List[Dict[str, str]]:
    cur.execute(SQL_FETCH, (offset, limit))
    rows = cur.fetchall()
    return [{"code": r["code"], "title": r["keyword"]} for r in rows]

def upsert_to_neo4j(driver, batch, version):
    if not batch:
        return
    with driver.session() as session:
        session.run(CYPHER_MERGE, rows=batch, naics_version=version)

# -----------------------------
# Main execution
# -----------------------------
def main():
    print("🔹 Starting NAICS keyword import...")

    with get_pg_connection() as conn, conn.cursor(cursor_factory=DictCursor) as cur:
        total = fetch_total(cur)
        print(f"Found {total} records in {TABLE_FQN}")

        with get_neo4j_driver() as driver:
            # Ensure constraints exist
            with driver.session() as s:
                for stmt in CYPHER_CONSTRAINTS.strip().split(";"):
                    if stmt.strip():
                        s.run(stmt)

            pages = math.ceil(total / ITEM_LIMIT)
            processed = 0

            for i in range(pages):
                offset = i * ITEM_LIMIT
                batch = fetch_page(cur, offset, ITEM_LIMIT)
                upsert_to_neo4j(driver, batch, NAICS_VERSION)
                processed += len(batch)
                print(f"Page {i+1}/{pages} processed: {len(batch)} rows (total {processed})")

    print("✅ Import complete.")

# -----------------------------
if __name__ == "__main__":
    main()
