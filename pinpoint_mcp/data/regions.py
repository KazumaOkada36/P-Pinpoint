"""
Maps user-supplied region strings to sets of US state abbreviations.
County-level filtering then uses the state_abbr field in the feature matrix.

Supports:
  - Full state names:      "California", "New York"
  - State abbreviations:   "CA", "NY"
  - Metro / region names:  "Southern California", "Bay Area", "NYC", "Texas Triangle"
  - Broad regions:         "West Coast", "Midwest", "Southeast"
  - Nationwide:            None or ""
"""

from functools import lru_cache

# ── State name / abbreviation lookup ─────────────────────────────────────────
STATE_NAME_TO_ABBR: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

ALL_STATE_ABBRS = set(STATE_NAME_TO_ABBR.values())

# ── Metro / multi-state region definitions ────────────────────────────────────
# Each entry maps keyword(s) to a set of state abbreviations.
# Order matters — more specific entries should come first.
REGION_DEFINITIONS: list[tuple[list[str], set[str]]] = [
    # California metros (single state, but allow county-level drill-down)
    (["southern california", "socal", "so cal", "la", "los angeles", "san diego", "orange county"],
     {"CA"}),
    (["bay area", "silicon valley", "san francisco", "sf bay"],
     {"CA"}),
    (["northern california", "norcal", "sacramento"],
     {"CA"}),

    # Northeast metros
    (["new york city", "nyc", "new york metro", "tri-state"],
     {"NY", "NJ", "CT"}),
    (["boston", "new england"],
     {"MA", "CT", "RI", "NH", "VT", "ME"}),
    (["philadelphia", "philly"],
     {"PA", "NJ", "DE"}),
    (["washington dc", "dmv", "dc metro", "beltway"],
     {"DC", "MD", "VA"}),

    # Southeast
    (["miami", "south florida"],
     {"FL"}),
    (["atlanta", "georgia"],
     {"GA"}),
    (["charlotte", "raleigh", "research triangle", "carolinas"],
     {"NC", "SC"}),
    (["nashville", "memphis", "tennessee"],
     {"TN"}),
    (["southeast", "deep south"],
     {"FL", "GA", "AL", "MS", "SC", "NC", "TN", "AR", "LA"}),

    # Texas
    (["houston", "dallas", "austin", "san antonio", "texas triangle", "dfw"],
     {"TX"}),

    # Midwest
    (["chicago", "chicagoland"],
     {"IL", "IN", "WI"}),
    (["detroit", "michigan"],
     {"MI"}),
    (["minneapolis", "twin cities", "minnesota"],
     {"MN"}),
    (["midwest", "great lakes", "rust belt"],
     {"IL", "IN", "MI", "OH", "WI", "MN", "IA", "MO", "KS", "NE", "ND", "SD"}),

    # Mountain West / Southwest
    (["denver", "colorado", "rocky mountains"],
     {"CO"}),
    (["phoenix", "tucson", "arizona"],
     {"AZ"}),
    (["las vegas", "nevada"],
     {"NV"}),
    (["salt lake", "utah"],
     {"UT"}),
    (["mountain west", "southwest"],
     {"CO", "UT", "NV", "AZ", "NM", "WY", "MT", "ID"}),

    # Pacific Northwest
    (["seattle", "portland", "pacific northwest", "pnw"],
     {"WA", "OR"}),

    # Broad coast / national regions
    (["west coast"],
     {"CA", "OR", "WA"}),
    (["east coast", "eastern seaboard"],
     {"ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "DC", "VA", "NC", "SC", "GA", "FL"}),
    (["sun belt"],
     {"FL", "GA", "AL", "MS", "LA", "TX", "AZ", "NV", "CA"}),
    (["great plains"],
     {"ND", "SD", "NE", "KS", "OK", "TX"}),
]


@lru_cache(maxsize=256)
def resolve_region(region_str: str | None) -> set[str] | None:
    """
    Resolve a region string to a set of state abbreviations.
    Returns None to indicate nationwide (no filter).
    """
    if not region_str or region_str.strip().lower() in ("nationwide", "anywhere", "all", "us", "usa", "united states"):
        return None

    r = region_str.strip().lower()

    # Direct state abbreviation match (e.g., "CA", "NY")
    if r.upper() in ALL_STATE_ABBRS:
        return {r.upper()}

    # Full state name match
    if r in STATE_NAME_TO_ABBR:
        return {STATE_NAME_TO_ABBR[r]}

    # Metro / region keyword match
    for keywords, states in REGION_DEFINITIONS:
        if any(kw in r for kw in keywords):
            return states

    # Partial state name match
    for name, abbr in STATE_NAME_TO_ABBR.items():
        if name in r or r in name:
            return {abbr}

    # No match — treat as nationwide
    return None


def get_region_label(region_str: str | None) -> str:
    """Human-readable label for the resolved region."""
    if not region_str:
        return "Nationwide"
    states = resolve_region(region_str)
    if states is None:
        return "Nationwide"
    if len(states) == 1:
        abbr = next(iter(states))
        # Reverse lookup full state name
        for name, a in STATE_NAME_TO_ABBR.items():
            if a == abbr:
                return name.title()
        return abbr
    return f"{region_str.title()} ({', '.join(sorted(states))})"
