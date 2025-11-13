#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Food triples (fixed relations) + optional Cypher generation via LLM.

Now with edge scoring:
- Each emitted edge includes a calibrated probability score in [0,1].
- Stage-chain edges: score = blend(heuristic prior, optional LLM confidence).
- Tag edges (ethnic/import/cert): rule-based scores.

Env vars (can be overridden via CLI):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  OPENAI_API_KEY, OPENAI_MODEL, OPENAI_VALIDATOR_MODEL
  OUTPUT_CSV, ITEM_LIMIT
  REGION, EMIT_CYPHER, CYPHER_DIR
  VALIDATE_WITH_LLM  (true/false)
"""

import os, json, csv, re, time, logging, argparse
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
from functools import wraps, lru_cache
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
import unicodedata

from relation_setup_2 import PROMPT_HEADER

# ------------- load .env -------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VALIDATOR_MODEL = os.getenv("OPENAI_VALIDATOR_MODEL", OPENAI_MODEL)

OUTPUT_CSV = os.getenv("OUTPUT_CSV", "food_relationships.csv")
ITEM_LIMIT = int(os.getenv("ITEM_LIMIT", "200"))

REGION = os.getenv("REGION", "US")
EMIT_CYPHER = os.getenv("EMIT_CYPHER", "false").lower() in ("1","true","yes")
CYPHER_DIR = os.getenv("CYPHER_DIR", "cypher_out")
VALIDATE_WITH_LLM = os.getenv("VALIDATE_WITH_LLM", "false").lower() in ("1","true","yes")

# ------------- logging -------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("food-triples")

# ------------- fixed relations (single source of truth) -------------
FIXED_RELS = {
    # Production / Cultivation
    "produced_by", "cultivated_by", "harvested_by", "raised_by", "caught_by",
    "processed_by", "graded_by", "pressed_by", "aged_by", "roasted_by", "ground_by",
    "baked_by", "confected_by", "brewed_by", "fermented_by", "slaughtered_by",
    "butchered_by", "filleted_by", "smoked_by", "dried_by",

    # Packaging / Certification
    "packaged_by", "labeled_by", "certified_by", "inspected_by",

    # Distribution / Transport / Sales
    "purchased_by", "sold_by", "transported_by", "imported_by", "exported_by",
    "warehoused_by", "distributed_by",

    # Preparation / Serving (Prepared Foods)
    "prepared_by", "served_by", "cooked_by", "assembled_by", "delivered_by",

    # Non-binary tag relations retained as-is
    "is_EthnicElement_Of", "is_imported_From",
}

# ------------- stage templates (use only FIXED_RELS) -------------
TEMPLATE_BY_STAGE = {
    "raw_commodity": [
        ("produced_by",    "Farm"),
        ("harvested_by",   "Farm"),
        ("raised_by",      "Ranch"),
        ("caught_by",      "Fishery"),
        ("cultivated_by",  "Orchard"),
        ("cultivated_by",  "Vineyard"),
        ("produced_by",    "Dairy Farm"),
        ("produced_by",    "Poultry Farm"),
        ("produced_by",    "Apiary"),
        ("cultivated_by",  "Greenhouse"),
        ("cultivated_by",  "Urban Farm"),
        ("cultivated_by",  "Hydroponic Farm"),
        # first handling
        ("processed_by",   "Mill"),
        ("processed_by",   "Grain Elevator"),
        ("graded_by",      "Inspector"),
        ("packaged_by",    "Packhouse"),
        # retail (fresh)
        ("sold_by",        "Fresh Produce Market"),
        ("sold_by",        "Farmers' Market"),
        ("sold_by",        "Produce Stand"),
        ("sold_by",        "CSA / Produce Box"),
        ("sold_by",        "Fishmonger / Seafood Market"),
        ("sold_by",        "Butcher"),
        # logistics
        ("transported_by", "Carrier"),
        ("transported_by", "Cold Chain Carrier"),
    ],

    "processed_product": [
        # primary processing & manufacturing
        ("produced_by",    "Mill"),
        ("processed_by",   "Cannery"),
        ("processed_by",   "Creamery / Dairy Processor"),
        ("processed_by",   "Meat Processor / Abattoir"),
        ("processed_by",   "Butcher / Charcuterie"),
        ("processed_by",   "Seafood Processor"),
        ("processed_by",   "Picklery / Fermenter"),
        ("processed_by",   "Tortilleria"),
        ("processed_by",   "Noodle Factory"),
        ("processed_by",   "Tofu Factory / Soy Processor"),
        ("roasted_by",     "Coffee Roastery"),
        ("ground_by",      "Spice Grinder / Masala Mill"),
        ("baked_by",       "Bakery"),
        ("confected_by",   "Confectioner / Chocolatier"),
        ("brewed_by",      "Brewery"),
        ("fermented_by",   "Winery / Cidery / Distillery"),
        ("pressed_by",     "Oil Press"),
        ("aged_by",        "Cheesemaker"),
        # packaging & compliance
        ("packaged_by",    "Packager"),
        # distribution
        ("purchased_by",   "Importer"),
        ("purchased_by",   "Distributor"),
        ("transported_by", "Distributor"),
        ("transported_by", "Cold Chain Carrier"),
        ("sold_by",        "Wholesaler"),
        # retail: general + specialty + diet-specific
        ("sold_by",        "Supermarket"),
        ("sold_by",        "Convenience Store"),
        ("sold_by",        "Co-op / Health Food Store"),
        ("sold_by",        "Ethnic Grocery"),
        ("sold_by",        "Asian Grocery"),
        ("sold_by",        "Indian Store"),
        ("sold_by",        "Mexican Market"),
        ("sold_by",        "Middle Eastern Market"),
        ("sold_by",        "European Deli"),
        ("sold_by",        "Vegan Market"),
        ("sold_by",        "Gluten-Free Specialty"),
        ("sold_by",        "Online Marketplace"),
        # food-service purchasing
        ("purchased_by",   "Restaurant"),
        ("purchased_by",   "Caterer"),
        ("purchased_by",   "Ghost Kitchen / Cloud Kitchen"),
        ("purchased_by",   "Food Truck"),
    ],

    "prepared_dish": [
        ("prepared_by",    "Restaurant"),
        ("served_by",      "Restaurant"),
        ("prepared_by",    "Bulk Producer"),
        ("prepared_by",    "Caterer"),
        ("prepared_by",    "Ghost Kitchen / Cloud Kitchen"),
        ("prepared_by",    "Food Truck"),
        ("prepared_by",    "Deli / Prepared Foods Counter"),
        ("prepared_by",    "Bakery-Café"),
        ("prepared_by",    "Pizzeria"),
        ("prepared_by",    "Taqueria"),
        ("prepared_by",    "BBQ Joint"),
        ("prepared_by",    "Butcher (Ready-to-Cook / Marinated)"),
        ("prepared_by",    "Fishmonger (Ready-to-Cook / Marinated)"),
        # dietary-focused venues
        ("prepared_by",    "Vegan Restaurant"),
        ("prepared_by",    "Vegetarian Restaurant"),
        ("prepared_by",    "Indian Vegan Restaurant"),
        ("prepared_by",    "Gluten-Free Kitchen"),
        # institutional / large-volume
        ("prepared_by",    "Commissary / Central Kitchen"),
        ("served_by",      "Cafeteria / Institutional Foodservice"),
        # retail prepared sections
        ("served_by",      "Supermarket Hot Bar"),
        ("served_by",      "Supermarket Deli"),
        # logistics (for meal kits / bulk prepared)
        ("transported_by", "Carrier"),
        ("transported_by", "Cold Chain Carrier"),
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
                except Exception as e:
                    tries += 1
                    logger.warning("Retry %s/%s in %s: %s", tries, max_tries, fn.__name__, e)
                    if tries >= max_tries:
                        raise
                    time.sleep(base_delay * (2 ** (tries - 1)))
        return wrapper
    return deco

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|cypher)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
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

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# ------------- OpenAI client -------------
def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.info("OpenAI client unavailable: %s", e)
        return None

def llm_available() -> bool:
    return get_openai_client() is not None

# ------------- state -------------
@dataclass
class State:
    items: List[str] = field(default_factory=list)
    item_stage: Dict[str, str] = field(default_factory=dict)
    ethnic_market: Dict[str, str] = field(default_factory=dict)   # item -> "Indian Grocery" / ...
    import_origin: Dict[str, str] = field(default_factory=dict)   # item -> "India" / "Domestic (USA)" / region
    us_cert_authority: Dict[str, str] = field(default_factory=dict)  # item -> "USDA Organic" / "FDA" / ...
    triples: List[List[str]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

# ------------- Neo4j -------------
def fetch_items_via_cypher(limit: int) -> List[str]:
    q_v5 = """
        MATCH (n)
        WHERE toUpper(n.name) CONTAINS 'FOODON'
        AND n.out_degree = 1
        RETURN n.name_text AS item
        LIMIT $limit
        """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            try:
                rows = s.run(q_v5, limit=limit)
                data = [r["item"] for r in rows if r["item"]]
                if data:
                    return data
            except CypherSyntaxError:
                pass
    finally:
        driver.close()

# ------------- Heuristics (fallbacks) -------------
def heuristic_stage_one(item: str) -> str:
    it = (item or "").lower().strip()
    prepared_cues = [
        "fried","baked","grilled","roasted","stir-fry","sauteed","cooked",
        "boiled","braised","broth","soup","stew","salad","curry","tacos",
        "burrito","sandwich","wrap","pizza","pasta","burger","omelette",
        "noodle","rice bowl","ramen","dumpling","casserole","stewed",
        "sautéed","sauté","fried rice","stir fry","biryani","enchilada",
        "samosa","kebab","roll","cooked meal","entrée","prepared dish"
    ]
    if any(w in it for w in prepared_cues):
        return "prepared_dish"

    raw_cues = [
        "raw","fresh","paddy","grain","leaf","seed","fruit","vegetable",
        "bean","lentil","legume","tuber","root","nut","herb","spice",
        "flower","mushroom","shoot","sprout","stem","whole","green",
        "unpolished","unprocessed","uncooked","farm","produce"
    ]
    if any(w in it for w in raw_cues):
        return "raw_commodity"

    processed_cues = [
        "powder","paste","sauce","oil","flour","noodles","snack","chips",
        "frozen","instant","canned","jar","pickled","dried","dehydrated",
        "fermented","sweetened","salted","syrup","extract","essence",
        "concentrate","mix","blend","spread","butter","jam","preserve",
        "marinated","ready to eat","packaged","processed"
    ]
    if any(w in it for w in processed_cues):
        return "processed_product"

    return "processed_product"

def heuristic_stage(items: List[str]) -> Dict[str, str]:
    return {it: heuristic_stage_one(it) for it in items}

ETHNIC_CUES = [
    ("Indian Grocery", [
        "basmati","poha","idli","dosa","ghee","masala","turmeric",
        "urad dal","chana dal","moong dal","hing","paneer","atta",
        "besan","tamarind","curry leaves","mustard seeds","cumin",
        "fennel","cardamom","clove","fenugreek","sambar powder",
        "biryani masala","kashmiri chili","garam masala","jaggery",
        "any product of Indian or South Asian origin"
    ]),
    ("Pakistani Grocery", ["nihari masala","shan masala","haleem","achar","seviyan"]),
    ("Bangladeshi Grocery", ["ilish","mustard oil","pitha","shutki"]),
    ("Nepali / Bhutanese Grocery", ["momo","thukpa","sel roti","gundruk"]),
    ("Mexican Grocery", ["masa","maseca","jalapeño","chipotle","tortilla","epazote"]),
    ("East Asian Grocery", ["nori","miso","kimchi","udon","matcha","gochujang","ramen"]),
    ("Chinese Grocery", ["doubanjiang","shaoxing","sichuan pepper","hoisin"]),
    ("Japanese Grocery", ["katsuobushi","kombu","mirin","dashi","panko","sake"]),
    ("Korean Grocery", ["gochugaru","doenjang","kimchi","tteokbokki"]),
    ("Southeast Asian Grocery", ["lemongrass","galangal","laksa","belacan","pho","fish sauce"]),
    ("Thai Grocery", ["green curry","red curry","palm sugar"]),
    ("Vietnamese Grocery", ["pho noodles","nuoc mam","rice paper"]),
    ("Indonesian Grocery", ["tempeh","sambal","rendang"]),
    ("Filipino Grocery", ["adobo","pandesal","ube","bagoong"]),
    ("Middle Eastern Grocery", ["tahini","sumac","za'atar","bulgur","labneh"]),
    ("Mediterranean Grocery", ["olive oil","feta","halloumi","olives"]),
    ("African Grocery", ["injera","berbere","teff","fufu","egusi"]),
    ("Italian Grocery", ["pasta","olive oil","parmesan","risotto"]),
    ("French Grocery", ["baguette","brie","herbes de provence","dijon"]),
    ("Eastern European Grocery", ["borscht","sauerkraut","pierogi","kielbasa"]),
    # U.S. regional
    ("American Southern Grocery", ["grits","cajun"]),
    ("Cajun / Creole Grocery", ["gumbo base","jambalaya mix","beignet"]),
    ("Tex-Mex Grocery", ["queso","enchilada","fajita"]),
    ("Hawaiian Grocery", ["spam","poke"]),
    ("New England Grocery", ["clam chowder","cranberry"]),
    ("California Grocery", ["avocado","sourdough","kombucha"]),
    ("General American Grocery", [])
]

COUNTRY_CUES = [
    ("India", ["basmati","idli","dosa","poha","turmeric","hing","toor dal","ghee","paneer","biryani","tamarind"]),
    ("China", ["doubanjiang","sichuan","hoisin","soy sauce","bok choy"]),
    ("Japan", ["nori","matcha","katsuobushi","mirin","ramen","miso","wasabi"]),
    ("Korea", ["kimchi","gochujang","doenjang","bibimbap","soju"]),
    ("Thailand", ["galangal","lemongrass","kaffir lime","fish sauce","green curry"]),
    ("Vietnam", ["pho","nuoc mam","rice paper","bun cha"]),
    ("Mexico", ["masa","jalapeño","chipotle","tomatillo","achiote"]),
    ("Italy", ["pasta","risotto","pesto","mozzarella","parmesan"]),
    ("France", ["baguette","brie","camembert","crème fraîche","truffle"]),
    ("Greece", ["feta","kalamata","tzatziki","dolma"]),
    ("Ethiopia", ["injera","berbere","teff"]),
    ("United States (General)", ["burger","barbecue","mac and cheese","hot dog","apple pie"]),
]

def heuristic_ethnic(item: str) -> str:
    it = (item or "").lower()
    for label, keys in ETHNIC_CUES:
        if any(k in it for k in keys):
            return label
    return "General American Grocery"

def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _normalize(s: str) -> str:
    return _strip_diacritics((s or "").lower().strip())

def _token_to_pattern(tok: str) -> re.Pattern:
    t = _normalize(tok)
    t = re.escape(t).replace(r"\ ", r"\s+")
    opt_plural = r"(?:s)?" if re.search(r"[a-z]$", t) else ""
    return re.compile(rf"\b{t}{opt_plural}\b", flags=re.IGNORECASE)

def _compile_table(cues: List[Tuple[str, List[str]]]):
    return [(label, [_token_to_pattern(k) for k in keys]) for label, keys in cues]

def _any_match(text_norm: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(text_norm) for p in pats)

@lru_cache(maxsize=1)
def _compiled_country():
    return _compile_table(tuple(COUNTRY_CUES))

@lru_cache(maxsize=1)
def _compiled_ethnic():
    return _compile_table(tuple(ETHNIC_CUES))

@lru_cache(maxsize=1)
def _compiled_fallback():
    _FALLBACK_BUCKETS = [
        ("Imported (Likely East Asia)", ["miso","kombu","nori","ramen","dashi","gochujang","doenjang","kimchi","mirin","tamari"]),
        ("Imported (Likely South/Southeast Asia)", ["tamarind","garam masala","hing","jaggery","lemongrass","galangal","laksa","belacan","fish sauce"]),
        ("Imported (Likely Latin America)", ["achiote","epazote","tomatillo","masa","guajillo","ancho","aji amarillo"]),
        ("Imported (Likely Middle East/Mediterranean)", ["tahini","za'atar","labneh","bulgur","sumac","harissa"]),
        ("Imported (Likely Africa)", ["berbere","teff","injera","egusi","fufu","jollof"]),
        ("Imported (Likely Europe)", ["parmesan","mozzarella","prosciutto","manchego","brie","camembert"]),
    ]
    return _compile_table(tuple(_FALLBACK_BUCKETS))

def heuristic_origin(item: str) -> str:
    it = _normalize(item)
    for country, pats in _compiled_country():
        if _any_match(it, pats):
            return country
    for label, pats in _compiled_ethnic():
        if _any_match(it, pats):
            return label
    for label, pats in _compiled_fallback():
        if _any_match(it, pats):
            return label
    return "Domestic (USA)"

def heuristic_us_cert(item: str, stage: str) -> str:
    it = (item or "").lower()
    if stage == "prepared_dish":
        return "Local Health Department"
    if "organic" in it:
        return "USDA Organic"
    if any(k in it for k in ["meat","poultry","egg product"]) or "raw milk" in it:
        return "USDA FSIS"
    if stage == "raw_commodity":
        return "USDA/APHIS"
    return "FDA"

# ------------- Heuristic plausibility + scoring -------------
_REL_PRIOR = {
    # very common
    "sold_by": 0.95, "purchased_by": 0.92, "transported_by": 0.9,
    "processed_by": 0.85, "packaged_by": 0.85,
    # action/facility
    "baked_by": 0.85, "roasted_by": 0.85, "brewed_by": 0.84, "fermented_by": 0.83,
    "ground_by": 0.83, "pressed_by": 0.82, "aged_by": 0.8, "graded_by": 0.75,
    "prepared_by": 0.9, "served_by": 0.9,
    # others
    "imported_by": 0.7, "exported_by": 0.7, "warehoused_by": 0.7, "distributed_by": 0.8,
    "certified_by": 0.9, "inspected_by": 0.7,
    "produced_by": 0.8, "cultivated_by": 0.78, "harvested_by": 0.78, "raised_by": 0.78, "caught_by": 0.75,
    "butchered_by": 0.78, "filleted_by": 0.76, "smoked_by": 0.76, "dried_by": 0.74, "labeled_by": 0.7,
}

_ORG_ACTION_HINTS = {
    "bakery": {"baked_by"},
    "coffee roastery": {"roasted_by","ground_by"},
    "roastery": {"roasted_by"},
    "brewery": {"brewed_by"},
    "winery": {"fermented_by"},
    "cidery": {"fermented_by"},
    "distillery": {"fermented_by"},
    "picklery": {"fermented_by","processed_by"},
    "fermenter": {"fermented_by","processed_by"},
    "butcher": {"butchered_by","processed_by","sold_by"},
    "seafood processor": {"processed_by","filleted_by","smoked_by","dried_by"},
    "cheesemaker": {"aged_by","processed_by"},
    "spice grinder": {"ground_by","processed_by"},
    "masala mill": {"ground_by","processed_by"},
    "mill": {"produced_by","processed_by","ground_by"},
    "packager": {"packaged_by","labeled_by"},
    "inspector": {"graded_by","inspected_by"},
    "wholesaler": {"sold_by","purchased_by"},
    "distributor": {"transported_by","distributed_by","purchased_by"},
    "supermarket": {"sold_by"},
    "restaurant": {"prepared_by","served_by","purchased_by"},
    "carrier": {"transported_by"},
    "cold chain carrier": {"transported_by"},
}

def _align_bonus(rel: str, org: str) -> float:
    r = rel.lower()
    o = _normalize(org)
    for key, rels in _ORG_ACTION_HINTS.items():
        if key in o:
            return 0.2 if r in rels else -0.25
    return 0.0

def is_relation_plausible(item: str, rel: str, org: str, stage: str) -> Tuple[bool, str]:
    """Cheap rule-based filter before the LLM validator."""
    txt = _normalize(item)
    rel_l = rel.lower()
    org_l = _normalize(org)

    is_pickle_like = any(k in txt for k in ["pickle", "pickled", "achar", "kimchi"])
    is_coffee_like = any(k in txt for k in ["coffee", "espresso", "arabica", "robusta"])
    is_tea_like = any(k in txt for k in ["tea", "matcha"])
    is_spice_like = any(k in txt for k in ["spice", "masala", "chili", "turmeric", "cumin", "cardamom", "clove", "pepper", "coriander"])
    is_raw = (stage == "raw_commodity")

    non_processing_orgs = ["importer","distributor","wholesaler","supermarket","convenience","marketplace","restaurant","caterer","ghost kitchen","food truck"]
    if rel_l in {"roasted_by","baked_by","brewed_by","fermented_by","confected_by","ground_by","pressed_by","aged_by","processed_by"}:
        if any(k in org_l for k in non_processing_orgs):
            return False, f"{org} is not a processing facility for '{rel}'."

    if is_pickle_like and rel_l in {"baked_by","roasted_by"}:
        return False, "Pickled/fermented items are not baked/roasted."

    if is_coffee_like:
        if rel_l in {"baked_by","fermented_by"}:
            return False, "Coffee is roasted/ground, not baked/fermented."
        if rel_l in {"roasted_by","ground_by"} and not any(k in org_l for k in ["roast", "grind", "coffee"]):
            return False, "Coffee should be roasted/ground by a roastery/grinder."

    if is_tea_like and rel_l in {"brewed_by","fermented_by"} and "kombucha" not in txt:
        return False, "Tea as a retail product isn't 'brewed_by' in supply chain relations."

    if is_spice_like and rel_l in {"baked_by","brewed_by"}:
        return False, "Spices are typically ground/processed, not baked/brewed."

    if is_raw and rel_l in {"baked_by","roasted_by","brewed_by","fermented_by","confected_by","aged_by"}:
        if not any(k in txt for k in ["coffee", "cocoa", "cacao"]):
            return False, f"Raw commodities generally aren't '{rel}' directly."

    return True, ""

def heuristic_edge_score(item: str, rel: str, org: str, stage: str, plaus_ok: bool) -> float:
    """Return a prior score in [0,1] for the edge given cheap rules."""
    if not plaus_ok:
        return 0.05
    base = _REL_PRIOR.get(rel.lower(), 0.6)

    # alignment bonus/penalty
    base += _align_bonus(rel, org)

    # item-specific tweaks
    t = _normalize(item)
    r = rel.lower()
    if any(k in t for k in ["coffee", "espresso"]) and r in {"roasted_by","ground_by"}:
        base += 0.15
    if any(k in t for k in ["pickle","kimchi","achar"]) and r in {"fermented_by","processed_by"}:
        base += 0.1
    if stage == "prepared_dish" and r in {"prepared_by","served_by"}:
        base += 0.1

    # light penalties for borderline combos
    if "distributor" in _normalize(org) and r in {"roasted_by","baked_by","brewed_by","fermented_by","ground_by","pressed_by"}:
        base -= 0.35

    return clamp01(base)

# ------------- LLM triple validation (batch) -------------
def _format_triple(item: str, rel: str, org: str, stage: str) -> str:
    return f"({item}) -[{rel.replace('_', ' ')}]-> ({org})  [stage={stage}]"

@retry(max_tries=3, base_delay=0.6)
def llm_validate_triples(candidates: List[Tuple[str, str, str, str]]) -> Dict[int, Dict[str, Any]]:
    """
    Validate triples with an LLM in batches.
    Input: list of (item, rel, org, stage)
    Output: dict[idx] = {"ok": bool, "reason": "<text>", "confidence": float}
    """
    client = get_openai_client()
    if not client:
        return {i: {"ok": True, "reason": "", "confidence": None} for i, _ in enumerate(candidates)}

    sys = (
        "You are a strict food-supply-chain validator.\n"
        "For each item–relation–organization triple, decide if it is plausible in real supply chains and give a confidence in [0,1].\n"
        "Output ONLY JSON list: [{\"idx\": <int>, \"ok\": true|false, \"reason\": \"...\", \"confidence\": 0.0-1.0}]"
    )

    out: Dict[int, Dict[str, Any]] = {}
    for chunk in batched(list(enumerate(candidates)), 60):
        payload = [
            {
                "idx": i,
                "item": it,
                "relation": rel,
                "organization": org,
                "stage": stage,
                "pretty": _format_triple(it, rel, org, stage),
            }
            for i, (it, rel, org, stage) in chunk
        ]
        user = "Validate these triples. Return ONLY the JSON described.\n" + json.dumps(payload, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=OPENAI_VALIDATOR_MODEL,
            temperature=0.0,
            messages=[{"role":"system","content":sys}, {"role":"user","content":user}]
        )
        try:
            arr = json_relaxed_load(resp.choices[0].message.content.strip())
            if not isinstance(arr, list):
                raise ValueError("validator returned non-list JSON")
            for obj in arr:
                i = int(obj.get("idx"))
                ok = bool(obj.get("ok"))
                reason = str(obj.get("reason", "")).strip()
                conf = obj.get("confidence", None)
                conf = None if conf is None else clamp01(conf)
                out[i] = {"ok": ok, "reason": reason, "confidence": conf}
        except Exception as e:
            for i, _ in chunk:
                out[i] = {"ok": True, "reason": f"validator parse error: {e}", "confidence": None}
    return out

# ------------- Tag-edge scoring -------------
def score_ethnic_edge(item: str, label: str) -> float:
    # Stronger if clearly matched via cues
    it = _normalize(item)
    label_norm = _normalize(label)
    if label_norm in {"general american grocery", ""}:
        return 0.55
    # presence of a direct cue increases score
    if any(k in it for _, keys in ETHNIC_CUES for k in keys):
        return 0.85
    return 0.7

def score_origin_edge(item: str, origin: str) -> float:
    it = _normalize(item)
    if origin.startswith("Imported (Likely"):
        # coarse bucket
        return 0.7 if any(k in it for _, pats in _compiled_fallback() for k in []) else 0.65
    if origin == "Domestic (USA)":
        # default modest confidence
        return 0.6
    # direct country cue
    if any(p.search(it) for country, pats in _compiled_country() if country == origin for p in pats):
        return 0.9
    return 0.75

def score_cert_edge(stage: str, authority: str) -> float:
    a = (authority or "").lower()
    if stage == "prepared_dish" and a == "local health department":
        return 0.95
    if a in {"usda organic","usda/aphis","usda fsis"}:
        return 0.9
    if a == "fda":
        return 0.85
    return 0.75

# ------------- synthesis to fixed triples -------------
def synthesize_triples(
    items: List[str],
    stages: Dict[str, str],
    ethnic: Dict[str, str],
    origin: Dict[str, str],
    cert: Dict[str, str]
) -> List[List[str]]:
    allowed = set(FIXED_RELS)
    triples: List[List[str]] = []
    skipped: List[Tuple[str,str,str,str,str]] = []  # (item, stage, rel, org, reason)

    # 1) Collect stage-chain candidates via template + heuristic gate
    stage_candidates: List[Tuple[str,str,str,str]] = []  # (item, rel, org, stage)
    plaus_flags: List[bool] = []
    heur_scores: List[float] = []

    for it in items:
        stage = stages.get(it, "processed_product")
        for rel, org in TEMPLATE_BY_STAGE.get(stage, []):
            if rel not in allowed:
                raise ValueError(f"Template used non-allowed relation: {rel}")
            ok, reason = is_relation_plausible(it, rel, org, stage)
            if ok:
                stage_candidates.append((it, rel, org, stage))
                plaus_flags.append(True)
                heur_scores.append(heuristic_edge_score(it, rel, org, stage, True))
            else:
                skipped.append((it, stage, rel, org, f"heuristic: {reason}"))

    # 2) LLM validate (optional)
    if VALIDATE_WITH_LLM and stage_candidates:
        verdicts = llm_validate_triples(stage_candidates)
    else:
        verdicts = {i: {"ok": True, "reason": "", "confidence": None} for i in range(len(stage_candidates))}

    # 3) Emit stage-chain triples allowed by both gates, with scores
    for i, (it, rel, org, stage) in enumerate(stage_candidates):
        v = verdicts.get(i, {"ok": True, "confidence": None})
        llm_ok = bool(v.get("ok", True))
        if not llm_ok:
            skipped.append((it, stage, rel, org, f"llm: {v.get('reason','')}"))
            continue

        prior = heur_scores[i]
        llm_conf = v.get("confidence", None)
        if llm_conf is None:
            score = prior
        else:
            # blend: emphasize LLM but retain prior
            score = clamp01(0.7 * llm_conf + 0.3 * prior)

        triples.append([it, rel, org, f"{score:.3f}"])

    # 4) Tags & certification (with scores)
    for it in items:
        seg = (ethnic.get(it) or "").strip()
        if seg:
            triples.append([it, "is_EthnicElement_Of", seg, f"{score_ethnic_edge(it, seg):.3f}"])

        src = (origin.get(it) or "").strip()
        if src:
            triples.append([it, "is_imported_From", src, f"{score_origin_edge(it, src):.3f}"])

        stage = stages.get(it, "processed_product")
        auth = (cert.get(it) or "").strip()
        if auth:
            if stage in ("raw_commodity", "processed_product"):
                triples.append([it, "certified_by", auth, f"{score_cert_edge(stage, auth):.3f}"])
            elif stage == "prepared_dish" and auth.lower() == "local health department":
                triples.append([it, "certified_by", auth, f"{score_cert_edge(stage, auth):.3f}"])

    if skipped:
        logger.info("Skipped %d triples (heuristic/LLM).", len(skipped))
        for (itm, stg, rel, org, why) in skipped[:50]:
            logger.debug("SKIP  %s  stage=%s  rel=%s  org=%s  reason=%s", itm, stg, rel, org, why)
        if len(skipped) > 50:
            logger.debug("... and %d more", len(skipped) - 50)

    return triples

# ------------- CSV -------------
CSV_HEADER = ["food_item", "relation", "org_or_value", "score"]

def write_csv(triples: List[List[str]], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(triples)

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

# ------------- Prompt + Cypher generation -------------
PROMPT_HEADER = ""  # keep imported value if defined externally

def build_prompt_for_item(
    food_item_name: str,
    region: str,
    item_class: str,
    processing_level: str,
    ethnic_tags,
    certification_required: bool,
    cold_chain_required: bool,
    foodon_id: str | None,
    mapping_conf: float,
    retail_allowed=None,
    retail_blocked=None,
    legality_flags=None,
    data_source="FoodOn",
    created_by_model="GPT-5",
    confidence_score=0.72,
    timestamp=None,
    # NEW:
    stage: str | None = None,
    ethnic_label: str | None = None,
    origin: str | None = None,
    cert_authority: str | None = None,
    suggested_edges: list[dict] | None = None,
):
    """
    We keep PROMPT_HEADER unchanged and append a machine-readable JSON context block
    so the model can see template edges + ethnic/origin/cert context.
    """
    ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    header = PROMPT_HEADER.format(
        food_item_name=food_item_name,
        region=region,
        cold_chain_required=str(cold_chain_required).lower(),
        processing_level=processing_level,
        item_class=item_class,
        ethnic_tags=json.dumps(ethnic_tags or [], ensure_ascii=False),
        certification_required=str(certification_required).lower(),
        retail_allowed=json.dumps(retail_allowed or [], ensure_ascii=False),
        retail_blocked=json.dumps(retail_blocked or [], ensure_ascii=False),
        legality_flags=json.dumps(legality_flags or {}, ensure_ascii=False),
        data_source=data_source,
        created_by_model=created_by_model,
        confidence_score=confidence_score,
        timestamp=ts,
        foodon_id=(foodon_id or ""),
        mapping_conf=mapping_conf,
    )

    # Append a compact JSON context block the LLM can consume directly.
    context = {
        "context_version": "v1",
        "item": food_item_name,
        "stage": stage,
        "item_class": item_class,
        "processing_level": processing_level,
        "ethnic_label": ethnic_label,
        "ethnic_tags": ethnic_tags or [],
        "origin": origin,
        "cert_authority": cert_authority,
        "region": region,
        "suggested_edges": suggested_edges or [],  # [{relation, organization, heuristic_score}]
    }

    return (
        header
        + "\n\n# Context (do not summarize; consume as input)\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\n# Output Requirements\n"
        + "Generate ONLY executable Cypher reflecting the context and suggested_edges;"
          " prefer higher scores when choosing which relations to materialize."
    )
@retry(max_tries=3, base_delay=0.8)
def generate_cypher_for_item(food_item_name: str, meta: Dict[str, Any]) -> str:
    client = get_openai_client()
    if client is None:
        return f"""// Fallback Cypher (LLM unavailable)
MERGE (f:FoodItem {{name:{json.dumps(food_item_name)}}})
ON CREATE SET f.created_at=datetime()
WITH f
MERGE (root:FSConcept {{name:'FoodSystem'}})
MERGE (f)-[:part_of]->(root)
// TODO: Expand with processing, distribution, and regulation nodes
"""
    prompt = build_prompt_for_item(food_item_name=food_item_name, **meta)
    sys = "Generate ONLY executable Cypher per the prompt's Output Requirements. Do not include explanations outside comments."
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":prompt}
        ]
    )
    return _strip_code_fences(resp.choices[0].message.content.strip())

# ------------- metadata mappers for prompt -------------
def map_stage_to_processing_level(stage: str) -> str:
    if stage == "raw_commodity":
        return "raw"
    if stage == "prepared_dish":
        return "tertiary"
    return "secondary"

def map_item_class(item: str, stage: str) -> str:
    t = (item or "").lower()
    if any(k in t for k in ["salmon","shrimp","fish","tuna","mackerel"]):
        return "seafood"
    if any(k in t for k in ["beef","pork","chicken","lamb","meat"]):
        return "meat"
    if stage == "prepared_dish":
        return "prepared_food"
    if any(k in t for k in ["juice","tea","coffee","soda","drink","beverage","milk"]):
        return "beverage"
    if any(k in t for k in ["apple","lettuce","spinach","tomato","mango","banana"]):
        return "produce"
    if any(k in t for k in ["rice","wheat","corn","grain","lentil","bean"]):
        return "staple"
    return "processed_plant_food"

def ethnicity_tags_from_label(label: str) -> list[str]:
    if not label or label == "General American Grocery":
        return []
    return [label.replace(" Grocery","")]

def certification_required_from_class(c: str) -> bool:
    return c in ("seafood","meat","prepared_food")

# ------------- CLI -------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=ITEM_LIMIT)
    p.add_argument("--output", default=OUTPUT_CSV)
    p.add_argument("--items-file", help="Optional newline-delimited items file")
    p.add_argument("--emit-cypher", action="store_true", help="Emit Cypher per item using the regulated prompt")
    p.add_argument("--region", default=REGION)
    p.add_argument("--cypher-dir", default=CYPHER_DIR)
    return p.parse_args()

# ------------- main -------------
def template_edges_for_item(item: str, stage: str):
    """Raw edges from TEMPLATE_BY_STAGE for this stage."""
    return [{"relation": rel, "organization": org} for (rel, org) in TEMPLATE_BY_STAGE.get(stage, [])]

def suggested_edges_for_item(item: str, stage: str) -> list[dict]:
    """
    Filter + score the template edges and return a compact list ready for the prompt.
    Uses the same plausibility gate + heuristic prior you already have for CSV scoring.
    """
    suggestions = []
    for rel, org in TEMPLATE_BY_STAGE.get(stage, []):
        ok, reason = is_relation_plausible(item, rel, org, stage)
        score = heuristic_edge_score(item, rel, org, stage, ok)
        if ok:
            suggestions.append({
                "relation": rel,
                "organization": org,
                "heuristic_score": round(float(score), 3)
            })
        else:
            # keep impl simple: skip implausible; if you want to show them, include with a low score & reason
            # suggestions.append({"relation": rel, "organization": org, "heuristic_score": 0.05, "why": reason})
            pass
    return suggestions

def main():
    args = parse_args()
    st = State()

    # 1) Items
    if args.items_file:
        with open(args.items_file, "r", encoding="utf-8") as f:
            st.items = [line.strip() for line in f if line.strip()]
    else:
        st.items = fetch_items_via_cypher(args.limit)

    # sanitize & dedupe
    st.items = list(dict.fromkeys(x.strip() for x in st.items if x and x.strip()))
    if not st.items:
        print("No items returned.")
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

    # 6) Synthesize triples (strict fixed relations) with validation + scores
    st.triples = synthesize_triples(
        st.items, st.item_stage, st.ethnic_market, st.import_origin, st.us_cert_authority
    )

    # 7) Write CSV
    write_csv(st.triples, args.output)

    # 8) Diagnostics
    st.diagnostics = diagnostics_report(
        st.items, st.item_stage, st.ethnic_market, st.import_origin, st.us_cert_authority, st.triples
    )
    print(f"✅ Wrote {len(st.triples)} triples to {args.output}")
    print("Stages:", st.diagnostics["stage_counts"], " Relations:", st.diagnostics["relation_counts"])
    print(json.dumps(st.diagnostics, indent=2))

    # 9) Optional: Cypher emission (regulated prompt)
    if args.emit_cypher or EMIT_CYPHER:
        outdir = Path(args.cypher_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"🔧 Emitting Cypher to: {outdir.resolve()} (region={args.region})")
        for it in st.items:
            stage = st.item_stage.get(it, "processed_product")
            item_class = map_item_class(it, stage)
            processing_level = map_stage_to_processing_level(stage)
            tags = ethnicity_tags_from_label(st.ethnic_market.get(it, ""))
            cert_req = certification_required_from_class(item_class)
            cold_chain = item_class in ("seafood","prepared_food")
            suggestions = suggested_edges_for_item(it, stage)
            origin = st.import_origin.get(it, None)
            cert_authority = st.us_cert_authority.get(it, None)
            ethnic_label = st.ethnic_market.get(it, "")
            meta = dict(
                region=args.region,
                item_class=item_class,
                processing_level=processing_level,
                ethnic_tags=tags,
                certification_required=cert_req,
                cold_chain_required=cold_chain,
                retail_allowed=[],
                retail_blocked=[],
                legality_flags={"region": args.region},
                data_source="FoodOn",
                created_by_model="GPT-5",
                confidence_score=0.98,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                foodon_id=None,
                mapping_conf=0.85,
                # NEW fields passed to the prompt:
                stage=stage,
                ethnic_label=ethnic_label,
                origin=origin,
                cert_authority=cert_authority,
                suggested_edges=suggestions,
            )
            cypher = generate_cypher_for_item(it, meta)
            fname = outdir / (re.sub(r"[^\w.-]+", "_", it) + ".cypher")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(cypher + "\n")
        print("✅ Cypher emission complete.")

# ---- LLM classifiers (unchanged logic, kept at end to avoid forward-ref issues) ----
def get_stage_default(it: str) -> str:
    return "processed_product"

@retry(max_tries=3, base_delay=0.5)
def classify_stage_via_llm(items: List[str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        logger.info("LLM unavailable; using stage heuristics.")
        return heuristic_stage(items)
    sys = "Classify each food term into exactly one of: raw_commodity, processed_product, prepared_dish."
    out: Dict[str,str] = {}
    for chunk in batched(items, 100):
        user = "Return ONLY a JSON object mapping item->stage (no extra text). Items:\n" + json.dumps(chunk, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        obj = json_relaxed_load(resp.choices[0].message.content.strip())
        if not isinstance(obj, dict):
            obj = {}
        for k in chunk:
            v = str(obj.get(k, "")).strip().lower()
            out[k] = v if v in TEMPLATE_BY_STAGE else heuristic_stage_one(k)
    return out

@retry(max_tries=3, base_delay=0.5)
def infer_ethnic_market_via_llm(items: List[str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        logger.info("LLM unavailable; using ethnic heuristics.")
        return {it: heuristic_ethnic(it) for it in items}
    sys = (
        "For each food name, assign the most likely U.S. retail ethnic market segment. "
        "Choose from a concise set such as: 'Indian Grocery', 'Mexican Grocery', 'East Asian Grocery', "
        "'Southeast Asian Grocery', 'Middle Eastern Grocery', 'General American Grocery'."
    )
    out: Dict[str,str] = {}
    for chunk in batched(items, 100):
        user = "Return ONLY a JSON object mapping item->ethnic_segment. Items:\n" + json.dumps(chunk, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        obj = json_relaxed_load(resp.choices[0].message.content.strip())
        if not isinstance(obj, dict):
            obj = {}
        for k in chunk:
            out[k] = (str(obj.get(k, "")).strip() or heuristic_ethnic(k))
    return out

@retry(max_tries=3, base_delay=0.5)
def infer_import_origin_via_llm(items: List[str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        logger.info("LLM unavailable; using origin heuristics.")
        return {it: heuristic_origin(it) for it in items}
    sys = (
        "For each food item, estimate the most likely country or region of origin when sold in the U.S. retail market. "
        "If domestically produced, return 'Domestic (USA)'. Otherwise a country ('India', 'Mexico') or region ('Middle East')."
    )
    out: Dict[str,str] = {}
    for chunk in batched(items, 100):
        user = "Return ONLY a JSON object mapping item->origin. Items:\n" + json.dumps(chunk, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        obj = json_relaxed_load(resp.choices[0].message.content.strip())
        if not isinstance(obj, dict):
            obj = {}
        for k in chunk:
            out[k] = (str(obj.get(k, "")).strip() or heuristic_origin(k))
    return out

@retry(max_tries=3, base_delay=0.5)
def infer_us_cert_authority_via_llm(items: List[str], stages: Dict[str, str]) -> Dict[str, str]:
    client = get_openai_client()
    if client is None or not items:
        logger.info("LLM unavailable; using certification heuristics.")
        return {it: heuristic_us_cert(it, stages.get(it, "processed_product")) for it in items}
    sys = (
        "For each item, output the most relevant U.S. certification or oversight authority at point-of-sale: "
        "Examples: 'USDA Organic', 'USDA/APHIS', 'USDA FSIS', 'FDA', 'Local Health Department'. "
        "Use 'Local Health Department' for prepared dishes at restaurants."
    )
    out: Dict[str,str] = {}
    for chunk in batched(items, 75):
        payload = [{"item": it, "stage": stages.get(it, "processed_product")} for it in chunk]
        user = "Return ONLY a JSON object mapping item->authority. Inputs:\n" + json.dumps(payload, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        obj = json_relaxed_load(resp.choices[0].message.content.strip())
        if not isinstance(obj, dict):
            obj = {}
        for k in chunk:
            v = str(obj.get(k, "")).strip()
            out[k] = v or heuristic_us_cert(k, stages.get(k, "processed_product"))
    return out

if __name__ == "__main__":
    main()