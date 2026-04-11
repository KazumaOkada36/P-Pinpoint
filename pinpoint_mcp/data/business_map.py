"""
Maps business types to BEA CAGDP9 LineCodes and defines scoring weights per priority.

LineCode reference (top-level, non-rollup):
  1  = All industry total
  3  = Agriculture (NAICS 11)
  6  = Mining (NAICS 21)
  10 = Utilities (NAICS 22)
  11 = Construction (NAICS 23)
  12 = Manufacturing (NAICS 31-33)
  34 = Wholesale trade (NAICS 42)
  35 = Retail trade (NAICS 44-45)
  36 = Transportation & warehousing (NAICS 48-49)
  45 = Information (NAICS 51)
  50 = Finance, insurance, real estate (NAICS 52-53)
  59 = Professional & business services (NAICS 54-56)
  68 = Education & health care (NAICS 61-62)
  75 = Arts, entertainment, accommodation & food (NAICS 71-72)
  82 = Other services (NAICS 81)
  83 = Government (NAICS 92)

Sub-linecodes used for precision:
  76 = Arts, entertainment & recreation only (NAICS 71)
  79 = Accommodation & food services only (NAICS 72)
  70 = Health care & social assistance (NAICS 62)
  69 = Educational services (NAICS 61)
  60 = Professional, scientific & technical (NAICS 54)
  51 = Finance & insurance (NAICS 52)
  56 = Real estate (NAICS 53)
"""

# ── Business type → industry linecode mapping ─────────────────────────────────
# primary: the main industry the business operates in
# complementary: nearby industries that bring in customers/support demand
#   weights on complementary are relative importance (sum doesn't need to = 1)
# avoid_primary: if primary LC is high, indicates market saturation (penalize)
BUSINESS_PROFILES = {
    "cafe": {
        "primary": 79,           # Accommodation & food services
        "complementary": {
            59: 0.30,            # Professional offices = daytime foot traffic
            45: 0.25,            # Information/tech workers = coffee buyers
            69: 0.20,            # Universities/colleges = student demand
            35: 0.15,            # Retail density = high foot traffic areas
            68: 0.10,            # Healthcare workers = early-morning demand
        },
        "avoid_primary": False,  # More food services = more food culture, not saturation
        "aliases": ["coffee shop", "coffee", "bubble tea", "boba", "tea shop", "bakery", "cafe"],
    },
    "restaurant": {
        "primary": 79,
        "complementary": {
            59: 0.25,
            35: 0.25,
            76: 0.20,            # Entertainment venues = evening diners
            45: 0.15,
            68: 0.15,
        },
        "avoid_primary": False,
        "aliases": ["restaurant", "dining", "food", "eatery", "bistro", "bar", "pub"],
    },
    "retail": {
        "primary": 35,           # Retail trade
        "complementary": {
            79: 0.30,            # Food & accommodation = traffic generators
            76: 0.20,            # Entertainment = same shopping destinations
            59: 0.20,            # Professional workers = discretionary income
            68: 0.15,
            45: 0.15,
        },
        "avoid_primary": True,   # High retail share = saturated market, penalize
        "aliases": ["retail", "boutique", "store", "shop", "clothing", "apparel", "goods"],
    },
    "gym": {
        "primary": 76,           # Arts, entertainment & recreation
        "complementary": {
            59: 0.30,            # Professional workers = gym-goers
            68: 0.25,            # Health care affinity = health-conscious
            35: 0.20,            # Retail = suburban density
            45: 0.15,
            79: 0.10,
        },
        "avoid_primary": True,   # High gym/recreation density = more competition
        "aliases": ["gym", "fitness", "yoga", "crossfit", "wellness", "health club", "studio"],
    },
    "tech": {
        "primary": 45,           # Information (NAICS 51)
        "complementary": {
            60: 0.35,            # Professional/scientific/tech services
            69: 0.25,            # Universities (talent pipeline)
            59: 0.20,
            51: 0.20,            # Finance & insurance (venture capital proximity)
        },
        "avoid_primary": False,
        "aliases": ["tech", "software", "startup", "saas", "app", "technology", "ai", "data"],
    },
    "healthcare": {
        "primary": 70,           # Health care & social assistance
        "complementary": {
            69: 0.30,            # Universities/medical schools
            59: 0.25,
            68: 0.25,
            83: 0.20,            # Government payers/insurance
        },
        "avoid_primary": False,
        "aliases": ["healthcare", "medical", "clinic", "dental", "pharmacy", "therapy", "doctor"],
    },
    "finance": {
        "primary": 51,           # Finance & insurance
        "complementary": {
            59: 0.35,
            45: 0.25,
            60: 0.25,
            56: 0.15,            # Real estate (co-investment)
        },
        "avoid_primary": False,
        "aliases": ["finance", "financial", "insurance", "banking", "investment", "wealth"],
    },
    "real_estate": {
        "primary": 56,           # Real estate
        "complementary": {
            11: 0.25,            # Construction activity
            59: 0.25,
            51: 0.25,
            35: 0.25,
        },
        "avoid_primary": False,
        "aliases": ["real estate", "realty", "property", "housing", "brokerage"],
    },
    "manufacturing": {
        "primary": 12,           # Manufacturing
        "complementary": {
            34: 0.30,            # Wholesale trade (supply chain)
            36: 0.30,            # Transportation (logistics)
            6: 0.20,             # Mining/raw materials
            59: 0.20,
        },
        "avoid_primary": False,
        "aliases": ["manufacturing", "factory", "production", "industrial", "fabrication"],
    },
    "education": {
        "primary": 69,           # Educational services
        "complementary": {
            59: 0.25,
            45: 0.25,
            79: 0.25,
            82: 0.25,
        },
        "avoid_primary": False,
        "aliases": ["education", "school", "tutoring", "training", "academy", "learning"],
    },
}

# ── Priority name → scoring weight profile ────────────────────────────────────
# Weights for: [industry_health, growth, market_size, complementary, stability]
PRIORITY_WEIGHTS = {
    "default": {
        "industry_health": 0.30,
        "growth":          0.25,
        "market_size":     0.20,
        "complementary":   0.15,
        "stability":       0.10,
    },
    "foot traffic": {
        "industry_health": 0.20,
        "growth":          0.20,
        "market_size":     0.35,   # bigger market = more people walking around
        "complementary":   0.20,
        "stability":       0.05,
    },
    "affordable rent": {
        "industry_health": 0.20,
        "growth":          0.15,
        "market_size":     0.05,   # smaller markets tend to have lower rents
        "complementary":   0.15,
        "stability":       0.45,   # stable = predictable costs
    },
    "growth potential": {
        "industry_health": 0.20,
        "growth":          0.45,
        "market_size":     0.15,
        "complementary":   0.15,
        "stability":       0.05,
    },
    "low competition": {
        "industry_health": 0.15,   # lower primary share = less saturation
        "growth":          0.20,
        "market_size":     0.15,
        "complementary":   0.40,   # strong demand drivers without a saturated primary
        "stability":       0.10,
    },
    "stability": {
        "industry_health": 0.20,
        "growth":          0.10,
        "market_size":     0.20,
        "complementary":   0.15,
        "stability":       0.35,
    },
    "target customer proximity": {
        "industry_health": 0.25,
        "growth":          0.15,
        "market_size":     0.25,
        "complementary":   0.30,   # complementary industries = customer base
        "stability":       0.05,
    },
}

# Keyword normalization for priority matching
PRIORITY_KEYWORDS = {
    "foot traffic":               ["foot traffic", "traffic", "busy", "people", "pedestrian", "walk"],
    "affordable rent":            ["affordable", "cheap", "low cost", "rent", "budget", "inexpensive"],
    "growth potential":           ["growth", "growing", "expanding", "emerging", "upside", "future"],
    "low competition":            ["competition", "competitive", "less competition", "untapped", "underserved"],
    "stability":                  ["stable", "stability", "safe", "established", "reliable", "consistent"],
    "target customer proximity":  ["customer", "demographic", "target", "proximity", "audience", "clientele"],
}


def normalize_business_type(raw: str) -> str:
    """Map a free-text business type string to a canonical key."""
    raw_lower = raw.lower().strip()
    for key, profile in BUSINESS_PROFILES.items():
        for alias in profile["aliases"]:
            if alias in raw_lower or raw_lower in alias:
                return key
    # Fallback: return the raw type, engine will use default profile
    return "cafe"  # safest default (food service = broadly applicable)


def normalize_priority(raw: str) -> str:
    """Map a free-text priority to a canonical weight key."""
    raw_lower = raw.lower().strip()
    for key, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in raw_lower for kw in keywords):
            return key
    return "default"


def get_weights(priorities: list[str]) -> dict[str, float]:
    """
    Blend weight profiles based on user priorities.
    If multiple priorities, average their weight vectors.
    """
    if not priorities:
        return PRIORITY_WEIGHTS["default"]

    normalized = [normalize_priority(p) for p in priorities]
    valid = [p for p in normalized if p in PRIORITY_WEIGHTS]

    if not valid:
        return PRIORITY_WEIGHTS["default"]

    # Average the weight dicts
    keys = list(PRIORITY_WEIGHTS["default"].keys())
    blended = {k: 0.0 for k in keys}
    for pname in valid:
        profile = PRIORITY_WEIGHTS[pname]
        for k in keys:
            blended[k] += profile[k]
    n = len(valid)
    blended = {k: v / n for k, v in blended.items()}

    # Re-normalize to sum to 1.0
    total = sum(blended.values())
    return {k: v / total for k, v in blended.items()}
