#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Food triples (fixed relations) + optional Cypher generation via LLM.

Features:
- Pull items from Neo4j (or a text file).
- Infer stage/ethnic/import/cert using LLM if available; fall back to heuristics.
- Emit CSV of strict fixed relations.
- (Optional) Emit per-item Cypher files using a regulated, ontology-anchored prompt.

Env vars (can be overridden via CLI):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  OPENAI_API_KEY, OPENAI_MODEL
  OUTPUT_CSV, ITEM_LIMIT
  REGION, EMIT_CYPHER, CYPHER_DIR
"""

import os, json, csv, re, time, logging, argparse
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple


# ------------- load .env -------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OUTPUT_CSV = os.getenv("OUTPUT_CSV", "food_relationships.csv")
ITEM_LIMIT = int(os.getenv("ITEM_LIMIT", "200"))

REGION = os.getenv("REGION", "US")
EMIT_CYPHER = os.getenv("EMIT_CYPHER", "false").lower() in ("1","true","yes")
CYPHER_DIR = os.getenv("CYPHER_DIR", "cypher_out")

# ------------- logging -------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("food-triples")

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
# TEMPLATE_BY_STAGE = {
#     "raw_commodity": [
#         ("produces_by",   "Farm"),
#         ("processes_by",  "Mill"),
#         # certification injected dynamically
#     ],
#     "processed_product": [
#         ("produces_by",   "Mill"),
#         ("packages_by",   "Packager"),
#         # certifies_by injected dynamically
#         ("purchases_by",  "Distributor"),
#         ("transports_by", "Distributor"),
#         ("sells_by",      "Wholesaler"),
#         ("sells_by",      "Supermarket"),
#         ("purchases_by",  "Restaurant"),
#         # ethnic & import injected dynamically
#     ],
#     "prepared_dish": [
#         ("prepares_by",   "Restaurant"),
#         ("serves_by",     "Restaurant"),
#         ("prepares_by",   "Bulk Producer"),
#     ],
# }

TEMPLATE_BY_STAGE = {
    # ---------------------------------------------------------------------
    # Unprocessed agricultural inputs
    # ---------------------------------------------------------------------
    "raw_commodity": [
        # primary production
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
        ("graded_by",      "Inspector"),     # optional QA/regulatory
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
        # certifications injected dynamically
    ],

    # ---------------------------------------------------------------------
    # Manufactured / packaged foods, ingredients, and specialty goods
    # ---------------------------------------------------------------------
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
        # ethnic & import tags injected dynamically
    ],

    # ---------------------------------------------------------------------
    # Ready-to-eat / prepared foods
    # ---------------------------------------------------------------------
    "prepared_dish": [
        # back-of-house prep & service
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
        # certifications injected dynamically
    ],
}
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

    # Domain-specific variants (optional flexibility)
    "supplied_by", "sourced_by",

    # Non-binary tag relations retained as-is
    "is_EthnicElement_Of", "is_imported_From",
}

for _stage, _pairs in TEMPLATE_BY_STAGE.items():
    for _rel, _ in _pairs:
        assert _rel in FIXED_RELS, f"Template uses non-allowed relation '{_rel}' in stage '{_stage}'"


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
    # Primary (Neo4j 5+): use NOT EXISTS { (n)-->() }
    q_v5 = """
    MATCH (n)
    WHERE toUpper(n.name) CONTAINS 'FOODON'
    AND n.out_degree = 1
    RETURN n.name_text AS item
    LIMIT $limit
    """
    # Fallback (works on older versions too): OPTIONAL MATCH + COUNT
    # q_fallback = """
    # MATCH (n)
    # WHERE toUpper(coalesce(n.name_text, n.name, "")) CONTAINS 'FOODON'
    # OPTIONAL MATCH (n)-[r]->()
    # WITH n, count(r) AS outdeg
    # WHERE outdeg = 0
    # RETURN coalesce(n.name_text, n.name) AS item
    # LIMIT $limit
    # """

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            try:
                rows = s.run(q_v5, limit=limit)
                return [r["item"] for r in rows if r["item"]]
            except CypherSyntaxError:
                print("Cypher syntax error",
                      file=sys.stderr,
                      flush=True)
    finally:
        driver.close()

# ------------- Heuristics (fallbacks) -------------
def heuristic_stage_one(item: str) -> str:
    """
    Classify a food-related item into one of three broad categories:
    - prepared_dish: cooked or ready-to-eat meals
    - raw_commodity: unprocessed ingredients or agricultural products
    - processed_product: packaged, preserved, or industrially processed items
    """

    it = (item or "").lower().strip()

    # --- Prepared / Cooked Dishes ---
    prepared_cues = [
        "fried", "baked", "grilled", "roasted", "stir-fry", "sauteed", "cooked",
        "boiled", "braised", "broth", "soup", "stew", "salad", "curry", "tacos",
        "burrito", "sandwich", "wrap", "pizza", "pasta", "burger", "omelette",
        "noodle", "rice bowl", "ramen", "dumpling", "casserole", "stewed",
        "sautéed", "sauté", "fried rice", "stir fry", "biryani", "enchilada",
        "samosa", "kebab", "roll", "cooked meal", "entrée", "prepared dish"
    ]
    if any(w in it for w in prepared_cues):
        return "prepared_dish"

    # --- Raw / Agricultural Commodities ---
    raw_cues = [
        "raw", "fresh", "paddy", "grain", "leaf", "seed", "fruit", "vegetable",
        "bean", "lentil", "legume", "tuber", "root", "nut", "herb", "spice",
        "flower", "mushroom", "shoot", "sprout", "stem", "whole", "green",
        "unpolished", "unprocessed", "uncooked", "farm", "produce"
    ]
    if any(w in it for w in raw_cues):
        return "raw_commodity"

    # --- Processed / Manufactured Food Products ---
    processed_cues = [
        "powder", "paste", "sauce", "oil", "flour", "noodles", "snack", "chips",
        "frozen", "instant", "canned", "jar", "pickled", "dried", "dehydrated",
        "fermented", "sweetened", "salted", "syrup", "extract", "essence",
        "concentrate", "mix", "blend", "spread", "butter", "jam", "preserve",
        "marinated", "ready to eat", "packaged", "instant", "processed"
    ]
    if any(w in it for w in processed_cues):
        return "processed_product"

    # --- Fallback Classification ---
    return "processed_product"


def heuristic_stage(items: List[str]) -> Dict[str, str]:
    return {it: heuristic_stage_one(it) for it in items}

ETHNIC_CUES = [
    # --- South Asian / Indian Subcontinent ---
    ("Indian Grocery", [
        "basmati", "poha", "idli", "dosa", "ghee", "masala", "turmeric",
        "urad dal", "chana dal", "moong dal", "hing", "paneer", "atta",
        "besan", "tamarind", "curry leaves", "mustard seeds", "cumin",
        "fennel", "cardamom", "clove", "fenugreek", "sambar powder",
        "biryani masala", "kashmiri chili", "garam masala", "jaggery",
        "any product of Indian or South Asian origin"
    ]),
    ("Pakistani Grocery", [
        "basmati", "nihari masala", "shan masala", "haleem mix", "achar",
        "korma", "rooh afza", "atta", "seviyan", "green cardamom",
        "any product of Pakistani origin"
    ]),
    ("Bangladeshi Grocery", [
        "ilish", "mustard oil", "pitha", "chitol", "shutki", "chili pickle",
        "daal", "any product of Bangladeshi origin"
    ]),
    ("Nepali / Bhutanese Grocery", [
        "momo wrapper", "thukpa noodles", "sel roti mix", "gundruk",
        "any product of Nepali or Bhutanese origin"
    ]),

    # --- Latin American / Hispanic ---
    ("Mexican Grocery", [
        "masa", "maseca", "jalapeño", "chipotle", "tortilla", "epazote",
        "achiote", "queso fresco", "tomatillo", "ancho chili", "guajillo",
        "avocado leaves", "refried beans", "salsa verde", "any product of Mexican or Hispanic origin"
    ]),
    ("Central American Grocery", [
        "plantain", "yuca", "pupusa", "black beans", "quesillo", "salsa roja",
        "any product of Central American origin"
    ]),
    ("Caribbean Grocery", [
        "jerk seasoning", "allspice", "scotch bonnet", "plantain", "ackee",
        "saltfish", "cassava", "rum", "callaloo", "any product of Caribbean origin"
    ]),

    # --- East Asian ---
    ("East Asian Grocery", [
        "nori", "miso", "kimchi", "udon", "matcha", "gochujang", "daikon",
        "katsuobushi", "mirin", "furikake", "soy sauce", "tofu", "rice vinegar",
        "sesame oil", "wasabi", "ponzu", "kombu", "ramen", "shoyu",
        "any product of Chinese, Japanese, or Korean origin"
    ]),
    ("Chinese Grocery", [
        "doubanjiang", "shaoxing wine", "sichuan pepper", "hoisin",
        "chili oil", "black bean paste", "soy sauce", "five spice",
        "lotus root", "bok choy", "any product of Chinese origin"
    ]),
    ("Japanese Grocery", [
        "miso", "nori", "katsuobushi", "kombu", "sake", "mirin",
        "ramen", "udon", "dashi", "wasabi", "matcha", "panko",
        "any product of Japanese origin"
    ]),
    ("Korean Grocery", [
        "kimchi", "gochujang", "doenjang", "gim", "tteokbokki",
        "soju", "bibimbap sauce", "gochugaru", "any product of Korean origin"
    ]),

    # --- Southeast Asian ---
    ("Southeast Asian Grocery", [
        "lemongrass", "galangal", "laksa", "belacan", "tamari", "kecap manis",
        "pho", "fish sauce", "kaffir lime", "coconut milk", "tamarind paste",
        "bird’s eye chili", "any product of Southeast Asian origin"
    ]),
    ("Thai Grocery", [
        "green curry paste", "red curry paste", "lemongrass", "galangal",
        "kaffir lime", "fish sauce", "coconut milk", "palm sugar", "any product of Thai origin"
    ]),
    ("Vietnamese Grocery", [
        "pho noodles", "nuoc mam", "rice paper", "lemongrass", "fish sauce",
        "banh trang", "any product of Vietnamese origin"
    ]),
    ("Indonesian Grocery", [
        "tempeh", "sambal", "rendang", "candlenut", "palm sugar",
        "any product of Indonesian origin"
    ]),
    ("Filipino Grocery", [
        "adobo sauce", "pandesal", "ube", "halo-halo mix", "bagoong",
        "any product of Filipino origin"
    ]),

    # --- Middle Eastern / Mediterranean ---
    ("Middle Eastern Grocery", [
        "tahini", "sumac", "za'atar", "bulgur", "pita", "labneh",
        "rose water", "date syrup", "couscous", "halva", "any product of Middle Eastern origin"
    ]),
    ("Mediterranean Grocery", [
        "olive oil", "feta", "halloumi", "olives", "oregano", "tahini",
        "grape leaves", "bulgur", "za'atar", "hummus", "any product of Mediterranean origin"
    ]),

    # --- African ---
    ("African Grocery", [
        "injera", "berbere", "teff", "fufu", "egusi", "palm oil",
        "cassava", "jollof mix", "any product of African origin"
    ]),
    ("North African Grocery", [
        "harissa", "ras el hanout", "couscous", "preserved lemon",
        "any product of Moroccan or Tunisian origin"
    ]),

    # --- European ---
    ("Italian Grocery", [
        "pasta", "olive oil", "parmesan", "risotto", "mozzarella", "balsamic vinegar",
        "san marzano", "polenta", "any product of Italian origin"
    ]),
    ("Greek Grocery", [
        "feta", "olive oil", "tzatziki", "kalamata olives", "dolma",
        "ouzo", "any product of Greek origin"
    ]),
    ("French Grocery", [
        "baguette", "brie", "camembert", "herbes de provence",
        "mustard de dijon", "truffle oil", "any product of French origin"
    ]),
    ("Eastern European Grocery", [
        "borscht", "sauerkraut", "pierogi", "kielbasa", "rye bread",
        "any product of Polish, Russian, or Ukrainian origin"
    ]),

    # --- American Regional / Ethnic ---
    ("American Southern Grocery", [
        "grits", "cornbread mix", "collard greens", "cajun seasoning",
        "sweet tea", "pecan pie filling", "hot sauce", "any product of Southern origin"
    ]),
    ("Cajun / Creole Grocery", [
        "gumbo base", "jambalaya mix", "andouille sausage", "file powder",
        "cajun seasoning", "dirty rice", "beignet mix", "any product of Cajun or Creole origin"
    ]),
    ("Tex-Mex Grocery", [
        "queso", "enchilada sauce", "fajita seasoning", "refried beans",
        "tortilla chips", "pico de gallo", "chili powder", "any product of Tex-Mex origin"
    ]),
    ("Hawaiian Grocery", [
        "spam", "macadamia nuts", "poke seasoning", "teriyaki sauce",
        "hawaiian sea salt", "poi", "any product of Hawaiian origin"
    ]),
    ("New England Grocery", [
        "clam chowder", "maple syrup", "cranberry", "baked beans",
        "any product of New England origin"
    ]),
    ("California Grocery", [
        "avocado", "sourdough", "kombucha", "almond butter", "quinoa",
        "acai bowl", "any product of Californian origin"
    ])
]


COUNTRY_CUES = [
    # --- South Asia ---
    ("India", [
        "basmati", "idli", "dosa", "poha", "turmeric", "hing", "toor dal", "arhar dal",
        "ghee", "garam masala", "paneer", "biryani", "tamarind", "sambar", "rasam",
        "roti", "naan", "any food of Indian origin"
    ]),
    ("Pakistan", [
        "biryani", "nihari", "haleem", "chapli kebab", "paratha", "lassi", "korma",
        "karahi", "any food of Pakistani origin"
    ]),
    ("Bangladesh", [
        "ilish", "bhuna", "tehari", "pitha", "shutki", "any food of Bangladeshi origin"
    ]),
    ("Sri Lanka", [
        "hoppers", "kottu roti", "pol sambol", "malu", "any food of Sri Lankan origin"
    ]),
    ("Nepal", [
        "momo", "thukpa", "sel roti", "gundruk", "any food of Nepali origin"
    ]),

    # --- East Asia ---
    ("China", [
        "doubanjiang", "shaoxing wine", "sichuan pepper", "hoisin", "soy sauce",
        "five spice", "bok choy", "tofu", "dumpling", "any food of Chinese origin"
    ]),
    ("Japan", [
        "nori", "matcha", "katsuobushi", "mirin", "udon", "ramen", "miso", "wasabi",
        "dashi", "sake", "teriyaki", "any food of Japanese origin"
    ]),
    ("Korea", [
        "kimchi", "gochujang", "doenjang", "bibimbap", "bulgogi", "soju",
        "gimbap", "any food of Korean origin"
    ]),
    ("Taiwan", [
        "bubble tea", "stinky tofu", "lu rou fan", "oyster omelet", "any food of Taiwanese origin"
    ]),

    # --- Southeast Asia ---
    ("Thailand", [
        "galangal", "lemongrass", "kaffir lime", "fish sauce", "coconut milk",
        "pad thai", "green curry", "red curry", "any food of Thai origin"
    ]),
    ("Vietnam", [
        "pho", "banh mi", "nuoc mam", "lemongrass", "rice paper", "bun cha",
        "any food of Vietnamese origin"
    ]),
    ("Indonesia", [
        "tempeh", "sambal", "rendang", "satay", "nasi goreng", "any food of Indonesian origin"
    ]),
    ("Malaysia", [
        "laksa", "roti canai", "nasi lemak", "kaya", "any food of Malaysian origin"
    ]),
    ("Philippines", [
        "adobo", "sinigang", "lumpia", "pandesal", "halo-halo", "any food of Filipino origin"
    ]),
    ("Singapore", [
        "laksa", "chili crab", "hainanese chicken rice", "any food of Singaporean origin"
    ]),

    # --- Middle East & Mediterranean ---
    ("Middle East", [
        "tahini", "za'atar", "bulgur", "sumac", "labneh", "pita", "shawarma",
        "hummus", "kebabs", "falafel", "any food of Middle Eastern origin"
    ]),
    ("Mediterranean", [
        "olive oil", "feta", "halloumi", "oregano", "tahini", "bulgur", "za'atar",
        "tzatziki", "any food of Mediterranean origin"
    ]),
    ("Turkey", [
        "baklava", "dolma", "borek", "menemen", "kebab", "pide", "ayran",
        "any food of Turkish origin"
    ]),
    ("Lebanon", [
        "tabbouleh", "hummus", "shawarma", "labneh", "kibbeh", "any food of Lebanese origin"
    ]),
    ("Israel", [
        "hummus", "falafel", "sabich", "shakshuka", "any food of Israeli origin"
    ]),

    # --- Europe ---
    ("Italy", [
        "pasta", "risotto", "pesto", "balsamic", "mozzarella", "parmesan",
        "lasagna", "gnocchi", "any food of Italian origin"
    ]),
    ("France", [
        "baguette", "brie", "camembert", "crème fraîche", "truffle", "bouillabaisse",
        "croissant", "ratatouille", "any food of French origin"
    ]),
    ("Spain", [
        "paella", "chorizo", "tapas", "gazpacho", "manchego", "olive oil",
        "saffron", "any food of Spanish origin"
    ]),
    ("Greece", [
        "feta", "kalamata", "tzatziki", "moussaka", "dolma", "spanakopita",
        "any food of Greek origin"
    ]),
    ("Germany", [
        "bratwurst", "sauerkraut", "pretzel", "schnitzel", "spaetzle", "any food of German origin"
    ]),
    ("Poland", [
        "pierogi", "borscht", "kielbasa", "bigosh", "any food of Polish origin"
    ]),
    ("United Kingdom", [
        "fish and chips", "shepherd’s pie", "marmite", "crumpet", "pudding",
        "any food of British origin"
    ]),

    # --- Africa ---
    ("Ethiopia", [
        "injera", "berbere", "teff", "shiro", "doro wat", "any food of Ethiopian origin"
    ]),
    ("Morocco", [
        "tagine", "couscous", "ras el hanout", "harira", "any food of Moroccan origin"
    ]),
    ("Nigeria", [
        "jollof rice", "egusi", "suya", "fufu", "any food of Nigerian origin"
    ]),
    ("South Africa", [
        "biltong", "bobotie", "boerewors", "chakalaka", "any food of South African origin"
    ]),

    # --- The Americas ---
    ("Mexico", [
        "masa", "maseca", "jalapeño", "chipotle", "tomatillo", "achiote", "epazote",
        "quesadilla", "taco", "enchilada", "salsa", "any food of Mexican or Hispanic origin"
    ]),
    ("Brazil", [
        "feijoada", "pão de queijo", "farofa", "brigadeiro", "any food of Brazilian origin"
    ]),
    ("Peru", [
        "ceviche", "aji amarillo", "lomo saltado", "quinoa", "any food of Peruvian origin"
    ]),
    ("Argentina", [
        "chimichurri", "asado", "empanada", "dulce de leche", "any food of Argentine origin"
    ]),
    ("Caribbean", [
        "jerk", "plantain", "ackee", "saltfish", "rum", "callaloo", "any food of Caribbean origin"
    ]),

    # --- United States & Regional Cuisines ---
    ("United States (General)", [
        "burger", "cornbread", "barbecue", "mac and cheese", "buffalo wings",
        "hot dog", "clam chowder", "apple pie", "any food of American origin"
    ]),
    ("Southern United States", [
        "fried chicken", "collard greens", "grits", "cornbread", "sweet tea",
        "biscuits and gravy", "hush puppies", "pecan pie", "any food of Southern origin"
    ]),
    ("Cajun / Creole", [
        "gumbo", "jambalaya", "étouffée", "andouille", "dirty rice", "beignet",
        "any food of Cajun or Creole origin"
    ]),
    ("Tex-Mex", [
        "chili con carne", "queso", "enchilada", "fajita", "nachos",
        "taco", "any food of Tex-Mex origin"
    ]),
    ("New England", [
        "clam chowder", "lobster roll", "baked beans", "cranberry",
        "cod", "maple syrup", "any food of New England origin"
    ]),
    ("Midwest", [
        "deep-dish pizza", "casserole", "cheese curd", "bratwurst",
        "butter burger", "hotdish", "any food of Midwestern origin"
    ]),
    ("Pacific Northwest", [
        "salmon", "dungeness crab", "huckleberry", "hazelnut",
        "craft coffee", "any food of Pacific Northwest origin"
    ]),
    ("Southwest", [
        "green chili", "posole", "fry bread", "sopapilla", "navajo taco",
        "any food of Southwestern origin"
    ]),
    ("Hawaiian", [
        "poke", "spam musubi", "loco moco", "kalua pork", "poi",
        "macadamia nut", "any food of Hawaiian origin"
    ]),
    ("Appalachian", [
        "apple butter", "cornbread", "pinto beans", "ramps", "wild greens",
        "sorghum", "any food of Appalachian origin"
    ]),
    ("California", [
        "avocado toast", "fish taco", "acai bowl", "sourdough",
        "farm-to-table", "kombucha", "any food of Californian origin"
    ]),
    ("New York", [
        "bagel", "cheesecake", "pizza", "pastrami", "hot dog",
        "black and white cookie", "any food of New York origin"
    ]),
    ("Chicago", [
        "deep-dish pizza", "italian beef", "hot dog", "gyros",
        "maxwell street polish", "any food of Chicago origin"
    ]),
    ("Louisiana", [
        "gumbo", "po'boy", "red beans and rice", "beignet",
        "muffuletta", "any food of Louisiana origin"
    ]),
    ("Texas", [
        "brisket", "barbecue", "chili", "fajita", "pecan pie", "any food of Texas origin"
    ]),
    ("Florida", [
        "key lime pie", "cuban sandwich", "stone crab", "gator bites",
        "conch fritter", "any food of Floridian origin"
    ]),
    ("Alaska", [
        "salmon", "halibut", "reindeer sausage", "king crab",
        "any food of Alaskan origin"
    ])
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
    """
    Build a robust token pattern:
    - accent/diacritic insensitive (callers normalize)
    - word boundaries to avoid spurious matches (ramen vs cameraman)
    - whitespace-insensitive within tokens (e.g., 'mac and cheese')
    - optional plural 's' when token ends with a letter
    """
    t = _normalize(tok)
    t = re.escape(t).replace(r"\ ", r"\s+")
    opt_plural = r"(?:s)?" if re.search(r"[a-z]$", t) else ""
    return re.compile(rf"\b{t}{opt_plural}\b", flags=re.IGNORECASE)

def _compile_table(cues: List[Tuple[str, List[str]]]):
    return [(label, [_token_to_pattern(k) for k in keys]) for label, keys in cues]

def _any_match(text_norm: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(text_norm) for p in pats)

# U.S. regional labels to prioritize over generic/foreign hits if present
_US_PRIORITY = {
    "united states (general)",
    "southern united states",
    "cajun / creole",
    "tex-mex",
    "new england",
    "midwest",
    "pacific northwest",
    "southwest",
    "hawaiian",
    "appalachian",
    "california",
    "new york",
    "chicago",
    "louisiana",
    "texas",
    "florida",
    "alaska",
}

# Fallback buckets if nothing hits in your cue lists
_FALLBACK_BUCKETS = [
    ("Imported (Likely East Asia)", [
        "miso", "kombu", "nori", "shoyu", "ramen", "dashi", "katsuobushi",
        "gochujang", "doenjang", "kimchi", "mirin", "tamari"
    ]),
    ("Imported (Likely South/Southeast Asia)", [
        "tamarind", "garam masala", "hing", "jaggery", "lemongrass",
        "galangal", "laksa", "belacan", "kecap manis", "fish sauce"
    ]),
    ("Imported (Likely Latin America)", [
        "achiote", "epazote", "tomatillo", "masa", "maseca", "guajillo", "ancho",
        "aji amarillo"
    ]),
    ("Imported (Likely Middle East/Mediterranean)", [
        "tahini", "za'atar", "labneh", "bulgur", "sumac", "harissa", "preserved lemon"
    ]),
    ("Imported (Likely Africa)", [
        "berbere", "teff", "injera", "egusi", "fufu", "yassa", "jollof"
    ]),
    ("Imported (Likely Europe)", [
        "parmesan", "mozzarella", "prosciutto", "manchego", "brie", "camembert"
    ]),
]

# ------------------ cached compiles ------------------

@lru_cache(maxsize=1)
def _compiled_country():
    # expects COUNTRY_CUES in global scope
    return _compile_table(tuple(COUNTRY_CUES))  # tuple for cache stability

@lru_cache(maxsize=1)
def _compiled_ethnic():
    # expects ETHNIC_CUES in global scope
    return _compile_table(tuple(ETHNIC_CUES))

@lru_cache(maxsize=1)
def _compiled_fallback():
    return _compile_table(tuple(_FALLBACK_BUCKETS))

# ------------------ main heuristic ------------------

def heuristic_origin(item: str) -> str:
    """
    Return a best-guess origin label for an item using:
    1) U.S. regional/ethnic (ETHNIC_CUES) with priority for U.S. regions
    2) COUNTRY_CUES (broad/global)
    3) ETHNIC_CUES (non-U.S. ethnic categories)
    4) Fallback continent-ish buckets
    5) Default 'Domestic (USA)'
    """
    it = _normalize(item)

    # 1) Direct U.S. regional priority (from ETHNIC_CUES)
    hits_us = []
    for label, pats in _compiled_ethnic():
        if label.lower() in _US_PRIORITY and _any_match(it, pats):
            hits_us.append(label)
    if hits_us:
        # Return the first (you can change tie-break logic if needed)
        return hits_us[0]

    # 2) Country-level matches
    for country, pats in _compiled_country():
        if _any_match(it, pats):
            return country

    # 3) Non-U.S. ethnic categories (general ethnic groceries)
    for label, pats in _compiled_ethnic():
        # skip U.S. labels here—they already had priority
        if label.lower() in _US_PRIORITY:
            continue
        if _any_match(it, pats):
            return label

    # 4) Fallback buckets (coarse, continent-ish guesses)
    for label, pats in _compiled_fallback():
        if _any_match(it, pats):
            return label

    # 5) Default
    return "Domestic (USA)"
def heuristic_us_cert(item: str, stage: str) -> str:
    it = (item or "").lower()
    if stage == "prepared_dish":
        return "Local Health Department"
    if "organic" in it:
        return "USDA Organic"
    if any(k in it for k in ["meat", "poultry", "egg product"]) or "raw milk" in it:
        return "USDA FSIS"
    if stage == "raw_commodity":
        return "USDA/APHIS"
    return "FDA"

# ------------- LLM calls (classification) -------------
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

        # Certification authority
        if stage in ("raw_commodity", "processed_product"):
            auth = (cert.get(it) or "").strip()
            if auth:
                triples.append([it, "certifies_by", auth])
        elif stage == "prepared_dish":
            auth = (cert.get(it) or "").strip()
            if auth and auth.lower() == "local health department":
                triples.append([it, "certifies_by", auth])

    return triples

# ------------- CSV -------------
CSV_HEADER = ["food_item", "relation", "org_or_value"]

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
PROMPT_HEADER ="" ""
# ------------- Prompt + Cypher generation -------------
# PROMPT_HEADER = """You are a Neo4j + Food System Ontology expert operating in a regulated, multi-ontology environment.
#
# ───────────────────────────────────────────────────────────────
# 🎯 GOAL
# ───────────────────────────────────────────────────────────────
# Given a food item f, generate Cypher code that builds *only plausible and legally valid* relationships between
# the food item and all relevant business entities in the Food System Ontology.
#
# Each emitted Cypher block must:
# 1. Use MATCH to anchor existing ontology nodes (FSConcept hierarchy);
# 2. Use MERGE for instance creation (FoodItem, FoodBusiness, Organization, Regulation, etc.);
# 3. Guarantee idempotency, semantic correctness, and full ontology connectivity (each subclass chain must reach FoodSystem).
#
# ───────────────────────────────────────────────────────────────
# 📥 INPUT
# ───────────────────────────────────────────────────────────────
# food_item_name: {food_item_name}
# region: {region}
# relationship_set:
#   [grown_by, processed_by, manufactured_by, packaged_by,
#    transported_by, supplied_by, stored_by, sold_by,
#    certified_by, is_menu_item_of]
#
# ontology_top_classes:
#   Root → FoodSystem →
#     {{ FoodProduction, FoodProcessing, FoodDistribution,
#       FoodConsumption, FoodWasteManagement, FoodGovernance,
#       FoodProductionInput, FoodChainInfrastructure }}
#
# optional_metadata:
#   - cold_chain_required: {cold_chain_required}
#   - processing_level: {processing_level}
#   - class: {item_class}
#   - ethnic_tags: {ethnic_tags}
#   - certification_required: {certification_required}
#   - retail_channels_allowed: {retail_allowed}
#   - retail_channels_blocked: {retail_blocked}
#   - legality_flags: {legality_flags}
#   - data_source: {data_source}
#   - created_by_model: {created_by_model}
#   - confidence_score: {confidence_score}
#   - timestamp: {timestamp}
#
# ───────────────────────────────────────────────────────────────
# 🧮 INFERENCE RULES
# ───────────────────────────────────────────────────────────────
# 1. **Ontology Anchoring**
#    MATCH all required FSConcept nodes:
#    FoodProduction, FoodProcessing, FoodDistribution,
#    FoodGovernance, Restaurant, GroceryStore, etc.
#    If a required FSConcept node is missing, MERGE it
#    with property `status:'inferred'`.
#
# 2. **Relationship Applicability**
#    Apply these default filters:
#    • wildlife_restricted (region="US"): GroceryStore/RetailDistribution
#       Restaurant, StreetFood.
#    • processed_plant_food:  all processing + packaging;  cold_chain.
#    • staple: full production→retail chain.
#    • seafood:  cold_chain,  certification.
#    • prepared_food:  restaurants only.
#    • beverage: manufacturer + distributor + retail.
#    • produce: farm + wholesaler + grocery + restaurant.
#    Skip illegal or implausible edges with a comment:
#    `// OMITTED: <reason>`.
#
# 3. **Ethnic Inference**
#    Derive ethnic business subtypes from `ethnic_tags`
#    or linguistic cues in `food_item_name`:
#    "samosa" → IndianRestaurant
#    "tacos" → MexicanRestaurant
#    "bamboo shoot" → ThaiRestaurant, ChineseRestaurant.
#
# 4. **Regulatory Integration**
#    If certification_required or regional rule applies:
#    MERGE (f)-[:regulated_under]->(:Regulation {{name:'<applicable law>'}})
#    MERGE (f)-[:certified_by]->(:Organization {{name:'<cert body>'}})
#    Optionally tag each regulation node with `jurisdiction`, `citation`, and `regulation_status`.
#
# 5. **External Ontology Mapping**
#    MERGE (f)-[:mapped_to]->(:ExternalConcept
#          {{source:'FoodOn', id:'{foodon_id}', match_confidence:{mapping_conf}}})
#    Add additional mappings to AGROVOC/NALT if available.
#
# 6. **Sustainability / Waste Hooks**
#    Optionally:
#    (f)-[:produces_waste]->(:FoodWasteType {{name:'ProcessingResidue'}})
#    (f)-[:recycled_into]->(:UpcycledProduct {{name:'Compost'}})
#
# 7. **Provenance Recording**
#    Every generated graph includes:
#    (f)-[:inferred_by]->(:Model
#          {{name:'{created_by_model}', prompt_version:'v6',
#           confidence:{confidence_score}, timestamp:'{timestamp}'}})
#
# 8. **Node Property Discipline**
#    Each MERGE for business nodes should include:
#      name, role, region, supply_chain_stage, scale
#    Example:
#      MERGE (riceMill:FoodBusiness
#             {{name:'RiceMill', role:'Processor',
#              region:'{region}', supply_chain_stage:'processing', scale:'industrial'}})
#
# 9. **Uniqueness Constraints**
#    (add once per database)
#    CREATE CONSTRAINT IF NOT EXISTS FOR (n:FoodItem) REQUIRE n.name IS UNIQUE;
#    CREATE CONSTRAINT IF NOT EXISTS FOR (n:FSConcept) REQUIRE n.name IS UNIQUE;
#
# 10. **Lifecycle and Temporal Attributes**
#     Set timestamps on creation:
#     ON CREATE SET f.created_at=datetime().
#
# ───────────────────────────────────────────────────────────────
# 📤 OUTPUT REQUIREMENTS
# ───────────────────────────────────────────────────────────────
# - Only executable Cypher code.
# - Use headers:
#   // --- FOOD PRODUCTION ---
#   // --- FOOD PROCESSING ---
#   etc.
# - Include explanatory comments for each relationship block.
# - Optionally append a JSON summary block at the end.
# """

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
):
    ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return PROMPT_HEADER.format(
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

@retry(max_tries=3, base_delay=0.8)
def generate_cypher_for_item(food_item_name: str, meta: Dict[str, Any]) -> str:
    client = get_openai_client()
    if client is None:
        # deterministic fallback skeleton
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

    # 6) Synthesize triples (strict fixed relations)
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
                foodon_id=None,          # optionally look up from Neo4j if available
                mapping_conf=0.85,
            )
            cypher = generate_cypher_for_item(it, meta)
            fname = outdir / (re.sub(r"[^\w.-]+", "_", it) + ".cypher")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(cypher + "\n")
        print("✅ Cypher emission complete.")

if __name__ == "__main__":
    main()
