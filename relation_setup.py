#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Food triples (fixed relations) + LLM inference for:
- ethnic market segment (Indian grocery, Mexican grocer, etc.)
- likely import origin (country/region)
- U.S. certification authority

Emits CSV triples: food_item, relation, org_or_value
Relations (fixed):
  produces_by, processes_by, packages_by, certifies_by, purchases_by, transports_by,
  sells_by, prepares_by, serves_by, is_EthnicElement_Of, is_imported_From
"""

import os, json, csv, re, time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
from functools import wraps

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ------------- load .env -------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "food_relationships.csv")
ITEM_LIMIT = int(os.getenv("ITEM_LIMIT", "200"))

# ------------- fixed relations -------------
FIXED_RELS = [
    "produces_by",
    "processes_by",
    "packages_by",
    "certifies_by",
    "purchases_by",
    "transports_by",
    "sells_by",
    "prepares_by",
    "serves_by",
    "is_EthnicElement_Of",
    "is_imported_From",
]

# ------------- stage templates (use only FIXED_RELS) -------------
TEMPLATE_BY_STAGE = {
    "raw_commodity": [
        ("produces_by",   "Farm"),
        ("processes_by",  "Mill"),
        # certification for raw commodity may be USDA/APHIS/USDA Organic; we’ll inject dynamically
    ],
    "processed_product": [
        ("produces_by",   "Mill"),
        ("packages_by",   "Packager"),
        # certifies_by injected dynamically
        ("purchases_by",  "Distributor"),
        ("transports_by", "Distributor"),
        ("sells_by",      "Wholesaler"),
        ("sells_by",      "Supermarket"),
        ("purchases_by",  "Restaurant"),
        # ethnic & import injected dynamically below
    ],
    "prepared_dish": [
        ("prepares_by",   "Restaurant"),
        ("serves_by",     "Restaurant"),
        ("prepares_by",   "Bulk Producer"),
    ],
}

# Validate template relations
for _stage, _pairs in TEMPLATE_BY_STAGE.items():
    for _rel, _ in _pairs:
        assert _rel in FIXED_RELS, f"Template uses non-allowed relation '{_rel}' in stage '{_stage}'"

# ------------- utilities -------------
def retry(max_tries=3, base_delay=0.6):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            tries = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    tries += 1
                    if tries >= max_tries:
                        raise
                    time.sleep(base_delay * (2 ** (tries - 1)))
        return wrapper
    return deco

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
    return s.strip()

def json_relaxed_load(s: str):
    s = _strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise

# ------------- OpenAI client -------------
def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

# ------------- state -------------
@dataclass
class State:
    items: List[str] = field(default_factory=list)
    item_stage: Dict[str, str] = field(default_factory=dict)
    ethnic_market: Dict[str, str] = field(default_factory=dict)   # item -> "Indian Grocery" / "Mexican Grocery" / ...
    import_origin: Dict[str, str] = field(default_factory=dict)   # item -> "India" / "Mexico" / "Domestic (USA)" / region
    us_cert_authority: Dict[str, str] = field(default_factory=dict)  # item -> "USDA Organic" / "FDA" / "Local Health Department" ...
    triples: List[List[str]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

# ------------- Neo4j -------------
def fetch_items_via_cypher() -> List[str]:
    q = f"""
    MATCH (n)
    WHERE toUpper(n.name) CONTAINS 'FOODON'
      AND n.out_degree = 1
    RETURN n.name_text AS item
    LIMIT $limit
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            return [r["item"] for r in s.run(q, limit=ITEM_LIMIT)]
    finally:
        driver.close()

# ------------- Heuristics (fallbacks) -------------
def heuristic_stage_one(item: str) -> str:
    it = (item or "").lower()
    if any(w in it for w in ["fried", "cooked", "soup", "stew", "salad", "dish", "curry", "tacos", "burrito"]):
        return "prepared_dish"
    if any(w in it for w in ["raw", "paddy", "grain", "leaf", "seed", "fruit", "vegetable", "bean", "lentil"]):
        return "raw_commodity"
    return "processed_product"

def heuristic_stage(items: List[str]) -> Dict[str, str]:
    return {it: heuristic_stage_one(it) for it in items}

ETHNIC_CUES = [
    ("Indian Grocery",   ["basmati", "poha", "idli", "dosa", "ghee", "masala", "turmeric", "urad", "chana", "moong", "hing"]),
    ("Mexican Grocery",  ["masa", "maseca", "jalapeño", "chipotle", "tortilla", "epazote", "achiote", "queso", "tomatillo"]),
    ("East Asian Grocery", ["nori", "miso", "kimchi", "udon", "matcha", "gochujang", "daikon", "katsuobushi", "mirin", "furikake"]),
    ("Middle Eastern Grocery", ["tahini", "sumac", "za'atar", "bulgur", "pita", "labneh"]),
    ("Southeast Asian Grocery", ["lemongrass", "galangal", "laksa", "belacan", "tamari", "kecap", "pho"]),
]

COUNTRY_CUES = [
    ("India",   ["basmati", "idli", "dosa", "poha", "turmeric", "hing", "toor", "arhar"]),
    ("Mexico",  ["masa", "maseca", "jalapeño", "chipotle", "tomatillo", "achiote"]),
    ("Japan",   ["nori", "matcha", "katsuobushi", "mirin", "udon"]),
    ("Korea",   ["kimchi", "gochujang"]),
    ("China",   ["doubanjiang", "shaoxing", "sichuan"]),
    ("Thailand",["galangal", "lemongrass"]),
    ("Vietnam", ["pho"]),
    ("Middle East", ["tahini", "za'atar", "bulgur"]),
]

def heuristic_ethnic(item: str) -> str:
    it = (item or "").lower()
    for label, keys in ETHNIC_CUES:
        if any(k in it for k in keys):
            return label
    return "General American Grocery"

def heuristic_origin(item: str) -> str:
    it = (item or "").lower()
    for country, keys in COUNTRY_CUES:
        if any(k in it for k in keys):
            return country
    # simple guess: raw commodities with non-native cues => imported; else domestic
    if any(k in it for k in ["basmati", "gochujang", "miso", "kimchi", "mirin", "achiote", "epazote", "tamarind"]):
        return "Imported (Likely Asia/Latin America)"
    return "Domestic (USA)"

def heuristic_us_cert(item: str, stage: str) -> str:
    it = (item or "").lower()
    if stage == "prepared_dish":
        return "Local Health Department"
    if "organic" in it:
        return "USDA Organic"
    if stage == "raw_commodity":
        return "USDA/APHIS"
    return "FDA"

# ------------- LLM calls -------------
def llm_available() -> bool:
    return get_openai_client() is not None

@retry(max_tries=3, base_delay=0.5)
def classify_stage_via_llm(items: List[str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        return heuristic_stage(items)
    sys = "Classify each food term into exactly one of: raw_commodity, processed_product, prepared_dish."
    user = "Return ONLY a JSON object mapping item->stage (no extra text). Items:\n" + json.dumps(items, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    obj = json_relaxed_load(resp.choices[0].message.content.strip())
    if not isinstance(obj, dict):
        return heuristic_stage(items)
    out = {}
    for k in items:
        v = str(obj.get(k, "")).strip().lower()
        out[k] = v if v in TEMPLATE_BY_STAGE else heuristic_stage_one(k)
    return out

@retry(max_tries=3, base_delay=0.5)
def infer_ethnic_market_via_llm(items: List[str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        return {it: heuristic_ethnic(it) for it in items}
    sys = (
        "For each food name, assign the most likely U.S. retail ethnic market segment. "
        "Choose from a concise set such as: 'Indian Grocery', 'Mexican Grocery', 'East Asian Grocery', "
        "'Southeast Asian Grocery', 'Middle Eastern Grocery', 'General American Grocery'."
    )
    user = "Return ONLY a JSON object mapping item->ethnic_segment. Items:\n" + json.dumps(items, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    obj = json_relaxed_load(resp.choices[0].message.content.strip())
    if not isinstance(obj, dict):
        return {it: heuristic_ethnic(it) for it in items}
    out = {}
    for k in items:
        out[k] = str(obj.get(k, "")).strip() or heuristic_ethnic(k)
    return out

@retry(max_tries=3, base_delay=0.5)
def infer_import_origin_via_llm(items: List[str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        return {it: heuristic_origin(it) for it in items}
    sys = (
        "For each food item, estimate the most likely country or region of origin when sold in the U.S. retail market. "
        "If domestically produced, return 'Domestic (USA)'. Otherwise a country ('India', 'Mexico') or region ('Middle East')."
    )
    user = "Return ONLY a JSON object mapping item->origin. Items:\n" + json.dumps(items, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    obj = json_relaxed_load(resp.choices[0].message.content.strip())
    if not isinstance(obj, dict):
        return {it: heuristic_origin(it) for it in items}
    out = {}
    for k in items:
        out[k] = str(obj.get(k, "")).strip() or heuristic_origin(k)
    return out

@retry(max_tries=3, base_delay=0.5)
def infer_us_cert_authority_via_llm(items: List[str], stages: Dict[str, str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        return {it: heuristic_us_cert(it, stages.get(it, "processed_product")) for it in items}
    # Provide stage as hint (FDA vs USDA/Health Dept)
    payload = [{"item": it, "stage": stages.get(it, "processed_product")} for it in items]
    sys = (
        "For each item, output the most relevant U.S. certification or oversight authority at point-of-sale: "
        "Examples: 'USDA Organic', 'USDA/APHIS', 'FDA', 'Local Health Department'. "
        "Use 'Local Health Department' for prepared dishes at restaurants."
    )
    user = "Return ONLY a JSON object mapping item->authority. Inputs:\n" + json.dumps(payload, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    obj = json_relaxed_load(resp.choices[0].message.content.strip())
    if not isinstance(obj, dict):
        return {it: heuristic_us_cert(it, stages.get(it, "processed_product")) for it in items}
    out = {}
    for k in items:
        v = str(obj.get(k, "")).strip()
        out[k] = v or heuristic_us_cert(k, stages.get(k, "processed_product"))
    return out

# ------------- synthesis -------------
def synthesize_triples(
    items: List[str],
    stages: Dict[str, str],
    ethnic: Dict[str, str],
    origin: Dict[str, str],
    cert: Dict[str, str]
) -> List[List[str]]:
    allowed = set(FIXED_RELS)
    triples: List[List[str]] = []

    for it in items:
        stage = stages.get(it, "processed_product")

        # Stage-based chain
        for rel, org in TEMPLATE_BY_STAGE.get(stage, []):
            if rel not in allowed:
                raise ValueError(f"Template used non-allowed relation: {rel}")
            triples.append([it, rel, org])

        # Ethnic market segment
        seg = (ethnic.get(it) or "").strip()
        if seg:
            triples.append([it, "is_EthnicElement_Of", seg])

        # Import origin
        src = (origin.get(it) or "").strip()
        if src:
            triples.append([it, "is_imported_From", src])

        # Certification authority (use on raw + processed; skip for prepared unless you want restaurant health dept)
        if stage in ("raw_commodity", "processed_product"):
            auth = (cert.get(it) or "").strip()
            if auth:
                triples.append([it, "certifies_by", auth])
        elif stage == "prepared_dish":
            # Optional: include health dept as certifies_by for prepared items
            auth = (cert.get(it) or "").strip()
            if auth and auth.lower() == "local health department":
                triples.append([it, "certifies_by", auth])

    return triples

# ------------- CSV -------------
def write_csv(triples: List[List[str]], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["food_item", "relation", "org_or_value"])
        for row in triples:
            w.writerow(row)

# ------------- diagnostics -------------
def diagnostics_report(
    items: List[str],
    stages: Dict[str, str],
    ethnic: Dict[str, str],
    origin: Dict[str, str],
    cert: Dict[str, str],
    triples: List[List[str]]
) -> Dict[str, Any]:
    rel_counts = Counter(t[1] for t in triples)
    stage_counts = Counter(stages.values())
    return {
        "num_items": len(items),
        "stage_counts": dict(stage_counts),
        "relation_counts": dict(rel_counts),
        "ethnic_segment_examples": {k: ethnic[k] for k in list(ethnic)[:8]},
        "import_origin_examples": {k: origin[k] for k in list(origin)[:8]},
        "cert_authority_examples": {k: cert[k] for k in list(cert)[:8]},
    }

# ------------- main -------------
def main():
    st = State()

    # 1) Items
    st.items = fetch_items_via_cypher()
    if not st.items:
        print("No items returned by Cypher.")
        return
    print(f"Items: {len(st.items)} (sample: {st.items[:5]})")

    # 2) Stage
    if llm_available():
        st.item_stage = classify_stage_via_llm(st.items)
    else:
        st.item_stage = heuristic_stage(st.items)

    # 3) Ethnic market segment
    if llm_available():
        st.ethnic_market = infer_ethnic_market_via_llm(st.items)
    else:
        st.ethnic_market = {it: heuristic_ethnic(it) for it in st.items}

    # 4) Import origin
    if llm_available():
        st.import_origin = infer_import_origin_via_llm(st.items)
    else:
        st.import_origin = {it: heuristic_origin(it) for it in st.items}

    # 5) U.S. certification authority
    if llm_available():
        st.us_cert_authority = infer_us_cert_authority_via_llm(st.items, st.item_stage)
    else:
        st.us_cert_authority = {it: heuristic_us_cert(it, st.item_stage.get(it, "processed_product")) for it in st.items}

    # 6) Synthesize triples (strict fixed relations)
    st.triples = synthesize_triples(
        st.items, st.item_stage, st.ethnic_market, st.import_origin, st.us_cert_authority
    )

    # 7) Write CSV
    write_csv(st.triples, OUTPUT_CSV)

    # 8) Diagnostics
    st.diagnostics = diagnostics_report(
        st.items, st.item_stage, st.ethnic_market, st.import_origin, st.us_cert_authority, st.triples
    )
    print(f"✅ Wrote {len(st.triples)} triples to {OUTPUT_CSV}")
    print(json.dumps(st.diagnostics, indent=2))

if __name__ == "__main__":
    main()
