#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build NAICS 2022 hierarchy in Neo4j from Postgres.

Developed by: Subhasis Dasgupta
Contact: sudasgupta@ucsd.edu
Affiliation: San Diego Supercomputer Center, UC San Diego

Description
-----------
Reads NAICS 2022 data from Postgres and constructs a NAICS ontology
in Neo4j:

- :NAICS nodes with:
    code, version, level, naics_code, title, description
- :IS_SUBCLASS_OF edges from child -> parent based on parent_code

Environment variables used (from .env)
--------------------------------------
Neo4j:
    NEO4J_URI=bolt://awesome-compute.sdsc.edu/:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=...

Postgres:
    PG_HOST=awesome-hw.sdsc.edu
    PG_PORT=5432
    PG_DB=nourish
    PG_USER=...
    PG_PASSWORD=...

Other variables in .env are ignored by this script.
"""

import os
import logging

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BATCH_SIZE = 500
NAICS_VERSION = "2022"

# Postgres query (your original logic, with clear aliases)
NAICS_SQL = """
SELECT DISTINCT
    "2022_naics" AS code,
    CASE
        WHEN length("2022_naics") = 2 THEN NULL
        WHEN length("2022_naics") = 3 THEN substring("2022_naics" from 1 for 2)
        WHEN length("2022_naics") = 4 THEN substring("2022_naics" from 1 for 3)
        WHEN length("2022_naics") = 5 THEN substring("2022_naics" from 1 for 4)
        WHEN length("2022_naics") = 6 THEN substring("2022_naics" from 1 for 5)
        ELSE NULL
    END AS parent_code,
    CASE
        WHEN length("2022_naics") = 2 THEN 'sector'
        WHEN length("2022_naics") = 3 THEN 'subsector'
        WHEN length("2022_naics") = 4 THEN 'industry_group'
        WHEN length("2022_naics") = 5 THEN 'general_industry'
        WHEN length("2022_naics") = 6 THEN 'specific_industry_group'
        ELSE 'unknown'
    END AS level,
    naics_code,
    naics_title AS specific_industry,
    description
FROM "2022_naics_descriptions" D
FULL OUTER JOIN "2022_NAICS_Keywords" K
    ON D."2022_naics" = K.naics_code::text
ORDER BY "2022_naics";
"""

# Unique constraint on NAICS nodes (code, version)
NAICS_CONSTRAINT = """
CREATE CONSTRAINT naics_code_version_unique IF NOT EXISTS
FOR (n:NAICS) REQUIRE (n.code, n.version) IS UNIQUE
"""

# Cypher to create nodes + subclass edges in batch
NAICS_MERGE_CYPHER = """
UNWIND $rows AS row
MERGE (n:NAICS {code: row.code, version: $version})
SET n.level       = row.level,
    n.naics_code  = row.naics_code,
    n.title       = row.title,
    n.description = row.description
WITH n, row
WHERE row.parent_code IS NOT NULL AND row.parent_code <> ''
MERGE (p:NAICS {code: row.parent_code, version: $version})
MERGE (n)-[:IS_SUBCLASS_OF]->(p)
"""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def get_pg_connection():
    """
    Connect to Postgres using the specific env variables:

        PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
    """
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB")
    user = os.getenv("PG_USER")
    pwd  = os.getenv("PG_PASSWORD")

    missing = [name for name, val in [
        ("PG_HOST", host),
        ("PG_DB", db),
        ("PG_USER", user),
        ("PG_PASSWORD", pwd),
    ] if not val]

    if missing:
        raise SystemExit(f"Missing Postgres environment variables: {', '.join(missing)}")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=pwd,
    )


def fetch_naics_rows(pg_conn):
    """Run the NAICS SQL and return all rows as dicts."""
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(NAICS_SQL)
        return cur.fetchall()


def batch(iterable, size):
    """Simple batching generator."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    missing_neo4j = [name for name, val in [
        ("NEO4J_URI", neo4j_uri),
        ("NEO4J_USER", neo4j_user),
        ("NEO4J_PASSWORD", neo4j_password),
    ] if not val]

    if missing_neo4j:
        raise SystemExit(f"Missing Neo4j environment variables: {', '.join(missing_neo4j)}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Connect to Postgres and fetch NAICS hierarchy data
    logging.info("Connecting to Postgres (nourish)...")
    pg_conn = get_pg_connection()
    with pg_conn:
        logging.info("Fetching NAICS rows from Postgres...")
        rows = fetch_naics_rows(pg_conn)
    logging.info("Fetched %d NAICS rows from Postgres", len(rows))

    # Transform rows to the shape Neo4j query expects
    prepared_rows = []
    for r in rows:
        code = (r.get("code") or "").strip() if r.get("code") is not None else ""
        if code == "":
            # skip malformed rows
            continue

        parent_code = (r.get("parent_code") or "").strip() if r.get("parent_code") else None

        prepared_rows.append({
            "code": code,
            "parent_code": parent_code,
            "level": r.get("level"),
            "naics_code": r.get("naics_code"),
            "title": r.get("specific_industry"),
            "description": r.get("description"),
        })

    logging.info("Prepared %d NAICS rows for Neo4j", len(prepared_rows))

    # Connect to Neo4j and create/merge nodes + edges
    logging.info("Connecting to Neo4j at %s ...", neo4j_uri)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    total_processed = 0

    with driver, driver.session() as session:
        # Ensure unique constraint
        logging.info("Ensuring NAICS (code, version) unique constraint...")
        session.run(NAICS_CONSTRAINT)

        for b in batch(prepared_rows, BATCH_SIZE):
            session.run(
                NAICS_MERGE_CYPHER,
                rows=b,
                version=NAICS_VERSION,
            )
            total_processed += len(b)
            logging.info("Processed %d / %d rows into Neo4j", total_processed, len(prepared_rows))

    logging.info("Done. Total NAICS rows processed into Neo4j: %d", total_processed)


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------
# Code Signature
# -------------------------------------------------------------------------
# Developed by: Subhasis Dasgupta
# Affiliation: San Diego Supercomputer Center, UC San Diego
# Contact: sudasgupta@ucsd.edu
# Script: NAICS 2022 hierarchy loader (Postgres -> Neo4j)
# Version: 1.0
# -------------------------------------------------------------------------
