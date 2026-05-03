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

occ_group: 2-digit BLS OCC major-group code for the primary workforce this
  business hires. Used to look up OEWS median wage data for talent_supply scoring.
  11=Mgmt  13=Business/financial  15=Computer/math  25=Education
  29=Healthcare  35=Food prep  39=Personal care  41=Sales  51=Production
"""

# ── Business type → industry linecode mapping ─────────────────────────────────
BUSINESS_PROFILES = {
    "cafe": {
        "primary": 79,
        "complementary": {
            59: 0.30,
            45: 0.25,
            69: 0.20,
            35: 0.15,
            68: 0.10,
        },
        "avoid_primary": False,
        "occ_group": "35",   # Food preparation & serving
        "aliases": ["coffee shop", "coffee", "bubble tea", "boba", "tea shop", "bakery", "cafe"],
    },
    "restaurant": {
        "primary": 79,
        "complementary": {
            59: 0.25,
            35: 0.25,
            76: 0.20,
            45: 0.15,
            68: 0.15,
        },
        "avoid_primary": False,
        "occ_group": "35",   # Food preparation & serving
        "aliases": ["restaurant", "dining", "food", "eatery", "bistro", "bar", "pub"],
    },
    "retail": {
        "primary": 35,
        "complementary": {
            79: 0.30,
            76: 0.20,
            59: 0.20,
            68: 0.15,
            45: 0.15,
        },
        "avoid_primary": True,
        "occ_group": "41",   # Sales & related
        "aliases": ["retail", "boutique", "store", "shop", "clothing", "apparel", "goods"],
    },
    "gym": {
        "primary": 76,
        "complementary": {
            59: 0.30,
            68: 0.25,
            35: 0.20,
            45: 0.15,
            79: 0.10,
        },
        "avoid_primary": True,
        "occ_group": "39",   # Personal care & service
        "aliases": ["gym", "fitness", "yoga", "crossfit", "wellness", "health club", "studio"],
    },
    "tech": {
        "primary": 45,
        "complementary": {
            60: 0.35,
            69: 0.25,
            59: 0.20,
            51: 0.20,
        },
        "avoid_primary": False,
        "occ_group": "15",   # Computer & mathematical
        "aliases": ["tech", "software", "startup", "saas", "app", "technology", "ai", "data"],
    },
    "healthcare": {
        "primary": 70,
        "complementary": {
            69: 0.30,
            59: 0.25,
            68: 0.25,
            83: 0.20,
        },
        "avoid_primary": False,
        "occ_group": "29",   # Healthcare practitioners & technical
        "aliases": ["healthcare", "medical", "clinic", "dental", "pharmacy", "therapy", "doctor"],
    },
    "finance": {
        "primary": 51,
        "complementary": {
            59: 0.35,
            45: 0.25,
            60: 0.25,
            56: 0.15,
        },
        "avoid_primary": False,
        "occ_group": "13",   # Business & financial operations
        "aliases": ["finance", "financial", "insurance", "banking", "investment", "wealth"],
    },
    "real_estate": {
        "primary": 56,
        "complementary": {
            11: 0.25,
            59: 0.25,
            51: 0.25,
            35: 0.25,
        },
        "avoid_primary": False,
        "occ_group": "13",   # Business & financial operations
        "aliases": ["real estate", "realty", "property", "housing", "brokerage"],
    },
    "manufacturing": {
        "primary": 12,
        "complementary": {
            34: 0.30,
            36: 0.30,
            6:  0.20,
            59: 0.20,
        },
        "avoid_primary": False,
        "occ_group": "51",   # Production
        "aliases": ["manufacturing", "factory", "production", "industrial", "fabrication"],
    },
    "education": {
        "primary": 69,
        "complementary": {
            59: 0.25,
            45: 0.25,
            79: 0.25,
            82: 0.25,
        },
        "avoid_primary": False,
        "occ_group": "25",   # Education, training & library
        "aliases": ["education", "school", "tutoring", "training", "academy", "learning"],
    },
}

# ── Priority name → scoring weight profile ────────────────────────────────────
# Eight dimensions (five original + three new from expanded datasets):
#   industry_health    BEA CAGDP9 — primary industry GDP share, LQ, CAGR
#   growth             BEA CAGDP9 — total economy 5yr CAGR + COVID recovery
#   market_size        BEA CAGDP9 — log-normalized total GDP
#   complementary      BEA CAGDP9 — weighted complementary industry presence
#   stability          BEA CAGDP9 — diversity score + low volatility
#   competitive_density  CBP — establishment count in primary NAICS per county
#   talent_supply      LODES + OEWS — skilled workforce density + wage affordability
#   tax_climate        Tax Foundation — state corporate income tax rate (inverted)
PRIORITY_WEIGHTS: dict[str, dict[str, float]] = {
    "default": {
        "industry_health":    0.25,
        "growth":             0.20,
        "market_size":        0.15,
        "complementary":      0.12,
        "stability":          0.08,
        "competitive_density": 0.10,
        "talent_supply":      0.07,
        "tax_climate":        0.03,
    },
    "foot traffic": {
        "industry_health":    0.15,
        "growth":             0.15,
        "market_size":        0.28,
        "complementary":      0.16,
        "stability":          0.03,
        "competitive_density": 0.14,
        "talent_supply":      0.07,
        "tax_climate":        0.02,
    },
    "affordable rent": {
        "industry_health":    0.15,
        "growth":             0.12,
        "market_size":        0.08,
        "complementary":      0.08,
        "stability":          0.35,
        "competitive_density": 0.05,
        "talent_supply":      0.02,
        "tax_climate":        0.15,
    },
    "growth potential": {
        "industry_health":    0.18,
        "growth":             0.38,
        "market_size":        0.10,
        "complementary":      0.08,
        "stability":          0.03,
        "competitive_density": 0.10,
        "talent_supply":      0.12,
        "tax_climate":        0.01,
    },
    "low competition": {
        "industry_health":    0.08,
        "growth":             0.18,
        "market_size":        0.12,
        "complementary":      0.25,
        "stability":          0.05,
        "competitive_density": 0.28,
        "talent_supply":      0.03,
        "tax_climate":        0.01,
    },
    "stability": {
        "industry_health":    0.15,
        "growth":             0.10,
        "market_size":        0.15,
        "complementary":      0.10,
        "stability":          0.30,
        "competitive_density": 0.03,
        "talent_supply":      0.02,
        "tax_climate":        0.15,
    },
    "target customer proximity": {
        "industry_health":    0.18,
        "growth":             0.10,
        "market_size":        0.22,
        "complementary":      0.25,
        "stability":          0.04,
        "competitive_density": 0.07,
        "talent_supply":      0.12,
        "tax_climate":        0.02,
    },
    "talent access": {
        "industry_health":    0.18,
        "growth":             0.15,
        "market_size":        0.10,
        "complementary":      0.10,
        "stability":          0.05,
        "competitive_density": 0.06,
        "talent_supply":      0.34,
        "tax_climate":        0.02,
    },
    "tax efficiency": {
        "industry_health":    0.18,
        "growth":             0.15,
        "market_size":        0.10,
        "complementary":      0.08,
        "stability":          0.12,
        "competitive_density": 0.08,
        "talent_supply":      0.05,
        "tax_climate":        0.24,
    },
}

# Keyword normalization for priority matching
PRIORITY_KEYWORDS = {
    "foot traffic":               ["foot traffic", "traffic", "busy", "people", "pedestrian", "walk", "footfall"],
    "affordable rent":            ["affordable", "cheap", "low cost", "rent", "budget", "inexpensive", "low rent"],
    "growth potential":           ["growth", "growing", "expanding", "emerging", "upside", "future", "growth potential"],
    "low competition":            ["competition", "competitive", "less competition", "untapped", "underserved", "low competition"],
    "stability":                  ["stable", "stability", "safe", "established", "reliable", "consistent"],
    "target customer proximity":  ["customer", "demographic", "target", "proximity", "audience", "clientele"],
    "talent access":              ["talent", "workforce", "hiring", "hire", "employees", "staff", "engineers", "skilled", "labor pool", "talent access", "talent pool"],
    "tax efficiency":             ["tax", "taxes", "corporate tax", "tax rate", "tax climate", "tax burden"],
}


def normalize_business_type(raw: str) -> str:
    raw_lower = raw.lower().strip()
    for key, profile in BUSINESS_PROFILES.items():
        for alias in profile["aliases"]:
            if alias in raw_lower or raw_lower in alias:
                return key
    return "cafe"


def normalize_priority(raw: str) -> str | None:
    raw_lower = raw.lower().strip()
    # Exact key match first
    if raw_lower in PRIORITY_WEIGHTS:
        return raw_lower
    for key, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in raw_lower for kw in keywords):
            return key
    return None


def get_weights(priorities: list[str]) -> dict[str, float]:
    """Blend weight profiles based on user priorities; re-normalize to 1.0."""
    if not priorities:
        return PRIORITY_WEIGHTS["default"]

    valid = [normalize_priority(p) for p in priorities]
    valid = [p for p in valid if p is not None and p in PRIORITY_WEIGHTS and p != "default"]
    if not valid:
        return PRIORITY_WEIGHTS["default"]

    keys    = list(PRIORITY_WEIGHTS["default"].keys())
    blended = {k: 0.0 for k in keys}
    for pname in valid:
        for k in keys:
            blended[k] += PRIORITY_WEIGHTS[pname][k]
    n       = len(valid)
    blended = {k: v / n for k, v in blended.items()}

    total = sum(blended.values())
    return {k: v / total for k, v in blended.items()}
