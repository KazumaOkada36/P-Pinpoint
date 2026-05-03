import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("secrets.env")

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "business_type": {"type": ["string", "null"]},
        "region": {"type": ["string", "null"]},
        "employee_count": {"type": ["integer", "null"]},
        "valuation_usd": {"type": ["number", "null"]},
        "monthly_rent_budget_usd": {"type": ["number", "null"]},
        "priorities_known": {"type": "boolean"},
        "user_priorities": {
            "type": "array",
            "items": {"type": "string"}
        },
        "suggested_priorities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3
        },
        "follow_up_question": {"type": "string"}
    },
    "required": [
        "business_type",
        "region",
        "employee_count",
        "valuation_usd",
        "monthly_rent_budget_usd",
        "priorities_known",
        "user_priorities",
        "suggested_priorities",
        "follow_up_question"
    ]
}

def parse_business_message(message: str):
    try:
        response = _get_client().responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are Pinpoint AI's intake parser. "
                        "Extract the user's business details into structured JSON. "
                        "Set priorities_known to TRUE and populate user_priorities if the user mentions ANYTHING about: "
                        "cost, rent, budget, competition, foot traffic, customers, growth, talent, employees, tax, stability, or location preferences. "
                        "Most business descriptions imply priorities — infer them confidently. "
                        "Only set priorities_known to false if the message is completely ambiguous with no business context at all. "
                        "Always suggest exactly 3 priorities regardless. "
                        "Normalize shorthand: SoCal→Southern California, NYC→New York City, 700k→700000. "
                        "Do not recommend locations."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "business_intake",
                    "strict": True,
                    "schema": SCHEMA
                }
            }
        )

        return json.loads(response.output_text)

    except Exception as e:
        print("Parser error:", e)
        return _keyword_fallback(message)


def _keyword_fallback(message: str) -> dict:
    """Simple keyword parser used when GPT is unavailable."""
    t = message.lower()

    import re
    def _wm(text, words):
        return any(re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', text) for w in words)

    def _parse_num(digits, suffix):
        n = float(digits.replace(',', ''))
        s = (suffix or '').lower()
        if s.startswith('b'):   n *= 1_000_000_000
        elif s.startswith('m'): n *= 1_000_000
        elif s.startswith('k'): n *= 1_000
        return n

    NUM = r'(\d[\d,]*(?:\.\d+)?)\s*([kmb](?:illion)?)?'

    def _extract_employee_count(text):
        m = re.search(NUM + r'\s*(?:employees?|engineers?|staff|workers?|people|team\s+members?)', text)
        if not m: m = re.search(r'(?:team|staff|workforce|headcount)\s+(?:of\s+)?' + NUM, text)
        if not m: m = re.search(r'hiring\s+' + NUM, text)
        if m:
            return int(_parse_num(m.group(1), m.group(2)))
        return None

    def _extract_valuation(text):
        m = re.search(r'(?:valu(?:ation|ed\s+at)|worth)\s+(?:of\s+|:\s*)?[\$€]?' + NUM, text)
        if m:
            return _parse_num(m.group(1), m.group(2))
        return None

    def _extract_rent(text):
        m = re.search(r'(?:rent|budget|lease)\s+(?:of\s+|around\s+|~\s*)?[\$€]?' + NUM, text)
        if not m: m = re.search(r'[\$€]' + NUM + r'\s*(?:per\s+month|/month|monthly)', text)
        if not m: m = re.search(r'monthly\s+(?:rent|budget|cost)\s+(?:of\s+)?[\$€]?' + NUM, text)
        if m:
            return _parse_num(m.group(1), m.group(2))
        return None

    # Business type — check more specific types first to avoid substring conflicts
    btype = None
    for key, words in [
        ("healthcare",   ["health", "clinic", "medical", "hospital", "therapy", "pharma", "biotech", "gene"]),
        ("finance",      ["finance", "bank", "investment", "insurance", "fintech"]),
        ("real_estate",  ["real estate", "property", "realty", "housing"]),
        ("manufacturing",["manufacturing", "factory", "production", "warehouse"]),
        ("education",    ["school", "education", "tutoring", "university", "training"]),
        ("cafe",         ["coffee", "cafe", "café", "boba", "bubble tea", "bakery"]),
        ("restaurant",   ["restaurant", "food", "dining", "pizza", "fast casual", "bistro"]),
        ("gym",          ["gym", "fitness", "yoga", "studio", "workout", "crossfit"]),
        ("retail",       ["retail", "boutique", "clothing", "store", "shop", "fashion"]),
        ("tech",         ["tech", "startup", "software", "saas"]),
    ]:
        if _wm(t, words):
            btype = key
            break

    # Region — more specific matches before broad ones
    region = None
    for r, words in [
        ("Southern California", ["socal", "southern california", "los angeles", "san diego", "orange county"]),
        ("Northern California", ["norcal", "northern california", "bay area", "san francisco", "silicon valley", "san jose"]),
        ("California",          ["california"]),
        ("Texas",               ["texas", "austin", "houston", "dallas", "san antonio"]),
        ("New York",            ["new york", "nyc", "manhattan", "brooklyn", "queens"]),
        ("Florida",             ["florida", "miami", "orlando", "tampa"]),
        ("Northeast",           ["northeast", "new england", "boston", "philadelphia"]),
        ("Southeast",           ["southeast", "atlanta", "charlotte", "nashville"]),
        ("Midwest",             ["midwest", "chicago", "detroit", "minneapolis"]),
        ("Pacific Northwest",   ["pacific northwest", "seattle", "portland"]),
        ("Southwest",           ["southwest", "phoenix", "denver", "las vegas"]),
    ]:
        if _wm(t, words):
            region = r
            break

    # Numeric field extraction
    employee_count        = _extract_employee_count(t)
    valuation_usd         = _extract_valuation(t)
    monthly_rent_budget_usd = _extract_rent(t)

    # Priorities from keywords
    priorities = []
    if any(w in t for w in ["talent", "hire", "employee", "team", "workforce", "engineer"]):
        priorities.append("talent access")
    if any(w in t for w in ["growth", "growing", "emerging", "expand"]):
        priorities.append("growth potential")
    if any(w in t for w in ["cheap", "affordable", "budget", "cost", "rent", "low cost"]):
        priorities.append("affordable rent")
    if any(w in t for w in ["traffic", "busy", "footfall", "customers", "foot traffic"]):
        priorities.append("foot traffic")
    if any(w in t for w in ["competition", "competitor", "market", "untapped"]):
        priorities.append("low competition")
    if any(w in t for w in ["stable", "stability", "safe", "reliable"]):
        priorities.append("stability")

    defaults = ["growth potential", "talent access", "low competition"]
    seen = set()
    suggested = []
    for p in priorities + defaults:
        if p not in seen:
            seen.add(p)
            suggested.append(p)
        if len(suggested) == 3:
            break

    return {
        "business_type": btype,
        "region": region,
        "employee_count": employee_count,
        "valuation_usd": valuation_usd,
        "monthly_rent_budget_usd": monthly_rent_budget_usd,
        "priorities_known": len(priorities) > 0,
        "user_priorities": priorities,
        "suggested_priorities": suggested,
        "follow_up_question": "What matters most for your location — growth potential, talent access, or low competition?"
    }
