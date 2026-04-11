"""
Pinpoint MCP Server — Business Location Recommendation Engine

Exposes five tools to Claude / any MCP client:

  recommend_locations     — ranked county recommendations for a business type
  analyze_county          — deep-dive profile of a specific county
  get_industry_trends     — GDP time-series for an industry in a state/county
  compare_locations       — side-by-side score comparison of specific counties
  list_supported_regions  — enumerate the regions and states available

Run with:
  python -m pinpoint_mcp.server
or via MCP CLI:
  mcp run pinpoint_mcp/server.py

Add to Claude Desktop config (claude_desktop_config.json):
  {
    "mcpServers": {
      "pinpoint": {
        "command": "python",
        "args": ["-m", "pinpoint_mcp.server"],
        "cwd": "/path/to/P-Pinpoint-main"
      }
    }
  }
"""

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    class FastMCP:  # pragma: no cover - fallback for local dev without mcp installed
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def tool(self):
            def decorator(func):
                return func
            return decorator

        def run(self):
            raise RuntimeError(
                "The 'mcp' package is not installed. Install pinpoint_mcp/requirements.txt "
                "to run the MCP transport."
            )

from .data.loader import load_raw, YEARS, get_timeseries
from .data.features import compute_feature_matrix
from .data.business_map import normalize_business_type, normalize_priority, BUSINESS_PROFILES
from .data.regions import resolve_region, get_region_label, STATE_NAME_TO_ABBR
from .engine.ranker import rank_locations
from .engine.scorer import score_county
from .engine.explainer import explain_score

mcp = FastMCP(
    name="Pinpoint Recommendation Engine",
    description=(
        "AI-powered business location advisor. "
        "Uses BEA county-level GDP data (2001–2024) across all 50 US states "
        "to score and rank counties for any business type."
    ),
)


# ── Tool 1: Recommend Locations ───────────────────────────────────────────────

@mcp.tool()
def recommend_locations(
    business_type: str,
    region: str = "",
    priorities: list[str] = [],
    top_n: int = 10,
) -> dict:
    """
    Return the top-N US counties for a given business type, filtered by region
    and weighted by the user's stated priorities.

    Args:
        business_type: Type of business, e.g. "cafe", "restaurant", "retail",
                       "gym", "tech startup", "healthcare clinic".
        region:        Optional geographic filter, e.g. "California",
                       "Southern California", "Texas", "NYC", "Midwest".
                       Leave empty for nationwide ranking.
        priorities:    List of what matters most, e.g.
                       ["foot traffic", "affordable rent", "growth potential"].
                       Accepted values: foot traffic, affordable rent,
                       growth potential, low competition, stability,
                       target customer proximity.
        top_n:         How many results to return (default 10, max 50).

    Returns:
        {
          business_type, region_label, priorities,
          total_counties_scored,
          results: [
            { rank, county, state, geoname, geofips, score, breakdown,
              key_metrics, explanation, lat, lng },
            ...
          ]
        }
    """
    top_n = min(max(1, top_n), 50)
    return rank_locations(
        business_type_raw=business_type,
        region_str=region or None,
        priorities=priorities,
        top_n=top_n,
    )


# ── Tool 2: Analyze County ────────────────────────────────────────────────────

@mcp.tool()
def analyze_county(
    county_name: str,
    state: str,
    business_type: str = "cafe",
    priorities: list[str] = [],
) -> dict:
    """
    Deep-dive analysis of a specific county for a given business type.

    Args:
        county_name:   County name, e.g. "Santa Clara", "Los Angeles", "Cook".
        state:         Two-letter state abbreviation, e.g. "CA", "IL".
        business_type: Business type to evaluate against (default "cafe").
        priorities:    Optional priority list for score weighting.

    Returns:
        Full county profile including all industry metrics, score breakdown,
        and a narrative explanation.
    """
    matrix = compute_feature_matrix()
    state_upper = state.strip().upper()

    # Find matching county (case-insensitive partial match)
    target = county_name.strip().lower()
    matches = [
        (fips, feat)
        for fips, feat in matrix.items()
        if feat["state"] == state_upper
        and feat.get("county", "").lower().startswith(target)
    ]

    if not matches:
        return {
            "error": f"No county found matching '{county_name}' in {state}. "
                     f"Check spelling or try a broader name.",
            "hint": f"Available states: {', '.join(sorted(set(f['state'] for f in matrix.values())))}",
        }

    geofips, geo_feat = matches[0]
    btype = normalize_business_type(business_type)
    score_result = score_county(geo_feat, btype, priorities)

    # Build full industry breakdown
    industries_out = {}
    for lc, ind in geo_feat["industry"].items():
        industries_out[lc] = {
            "description":     ind.get("description", ""),
            "gdp_2024_millions": round(ind["gdp_2024"] / 1000, 1) if ind.get("gdp_2024") else None,
            "share_of_total":  f"{ind['share'] * 100:.1f}%" if ind.get("share") else None,
            "location_quotient": round(ind["lq"], 2) if ind.get("lq") else None,
            "cagr_5yr":        f"{ind['cagr_5yr'] * 100:.1f}%" if ind.get("cagr_5yr") is not None else None,
            "covid_recovery":  f"{ind['covid_recovery']:.2f}x" if ind.get("covid_recovery") else None,
        }

    explanation = explain_score(
        score_result["breakdown"],
        score_result["weights"],
        geo_feat,
        btype,
        priorities,
    )

    return {
        "geofips":     geofips,
        "geoname":     geo_feat["geoname"],
        "county":      geo_feat["county"],
        "state":       geo_feat["state"],
        "score":       score_result["total"],
        "breakdown":   score_result["breakdown"],
        "weights":     score_result["weights"],
        "explanation": explanation,
        "total_gdp_2024_billions": round(geo_feat["total"]["gdp_2024"] / 1_000_000, 2)
            if geo_feat["total"].get("gdp_2024") else None,
        "total_gdp_cagr_5yr":    f"{geo_feat['total']['cagr_5yr'] * 100:.1f}%"
            if geo_feat["total"].get("cagr_5yr") is not None else None,
        "covid_recovery":        f"{geo_feat['total']['covid_recovery']:.2f}x"
            if geo_feat["total"].get("covid_recovery") else None,
        "diversity_score":       round(geo_feat.get("diversity_score", 0) * 100, 1),
        "industries":            industries_out,
    }


# ── Tool 3: Get Industry Trends ────────────────────────────────────────────────

@mcp.tool()
def get_industry_trends(
    industry: str,
    state: str,
    county: str = "",
) -> dict:
    """
    Return annual GDP time-series (2001–2024) for a specific industry
    in a given state or county.

    Args:
        industry: Industry name or linecode, e.g. "retail trade", "information",
                  "accommodation and food services", "manufacturing", or "79".
        state:    Two-letter state abbreviation, e.g. "CA".
        county:   Optional county name for county-level data. If blank,
                  returns state-level aggregate.

    Returns:
        { industry_label, geoname, years: [...], values: [...], units,
          cagr_5yr, cagr_full, covid_recovery }
    """
    lc = _resolve_linecode(industry)
    if lc is None:
        return {
            "error": f"Unknown industry '{industry}'. "
                     "Use a name like 'retail trade', 'information', "
                     "'accommodation and food services', or a numeric linecode (1-92)."
        }

    raw = load_raw()
    state_upper = state.strip().upper()

    if county:
        target = county.strip().lower()
        matches = [
            (fips, geo)
            for fips, geo in raw.items()
            if geo["state_abbr"] == state_upper
            and geo["is_county"]
            and (geo.get("county") or "").lower().startswith(target)
        ]
        if not matches:
            return {"error": f"County '{county}' not found in {state}."}
        geofips, geo = matches[0]
    else:
        # State-level row
        matches = [
            (fips, geo)
            for fips, geo in raw.items()
            if geo["state_abbr"] == state_upper and not geo["is_county"]
        ]
        if not matches:
            return {"error": f"State '{state}' not found."}
        geofips, geo = matches[0]

    ind = geo["industries"].get(lc)
    if not ind:
        return {
            "error": f"No data for industry linecode {lc} in {geo['geoname']}. "
                     "Some industries are suppressed for certain counties."
        }

    vals = ind["values"]
    years_out = YEARS
    values_out = vals  # May contain None for suppressed years

    # Compute summary stats
    v2019 = vals[YEARS.index(2019)]
    v2024 = vals[YEARS.index(2024)]
    v2001 = vals[YEARS.index(2001)]

    def _cagr(start, end, n):
        if start and end and start > 0 and end > 0:
            return round(((end / start) ** (1.0 / n) - 1) * 100, 2)
        return None

    return {
        "industry_label":  ind["description"],
        "linecode":        lc,
        "geoname":         geo["geoname"],
        "units":           "Thousands of chained 2017 dollars",
        "years":           years_out,
        "values":          values_out,
        "cagr_5yr_pct":    _cagr(v2019, v2024, 5),
        "cagr_full_pct":   _cagr(v2001, v2024, 23),
        "covid_recovery":  round(v2024 / v2019, 3) if (v2019 and v2024 and v2019 > 0) else None,
        "gdp_2024_millions": round(v2024 / 1000, 1) if v2024 else None,
    }


# ── Tool 4: Compare Locations ─────────────────────────────────────────────────

@mcp.tool()
def compare_locations(
    locations: list[dict],
    business_type: str,
    priorities: list[str] = [],
) -> dict:
    """
    Score and compare a list of specific counties side by side.

    Args:
        locations:     List of {"county": "Santa Clara", "state": "CA"} dicts.
        business_type: Business type key for scoring.
        priorities:    Priority list for weight adjustment.

    Returns:
        Ranked comparison table with per-county scores and breakdowns.
    """
    if not locations:
        return {"error": "Provide at least one location to compare."}

    matrix = compute_feature_matrix()
    btype = normalize_business_type(business_type)
    results = []

    for loc in locations:
        cname = loc.get("county", "").strip().lower()
        sname = loc.get("state", "").strip().upper()

        matches = [
            (fips, feat)
            for fips, feat in matrix.items()
            if feat["state"] == sname
            and (feat.get("county") or "").lower().startswith(cname)
        ]

        if not matches:
            results.append({
                "county":  loc.get("county"),
                "state":   loc.get("state"),
                "error":   "Not found in dataset.",
            })
            continue

        geofips, geo_feat = matches[0]
        score_result = score_county(geo_feat, btype, priorities)
        explanation = explain_score(
            score_result["breakdown"],
            score_result["weights"],
            geo_feat,
            btype,
            priorities,
        )
        results.append({
            "county":      geo_feat["county"],
            "state":       geo_feat["state"],
            "geoname":     geo_feat["geoname"],
            "geofips":     geofips,
            "score":       score_result["total"],
            "breakdown":   score_result["breakdown"],
            "explanation": explanation,
        })

    results.sort(key=lambda x: x.get("score", -1), reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return {
        "business_type": btype,
        "priorities":    priorities,
        "comparison":    results,
    }


# ── Tool 5: List Supported Regions ────────────────────────────────────────────

@mcp.tool()
def list_supported_regions() -> dict:
    """
    Return available states, metros, and region aliases.

    Use this to discover valid region values for recommend_locations.
    """
    matrix = compute_feature_matrix()

    # Count counties per state
    state_counts: dict[str, int] = {}
    for feat in matrix.values():
        s = feat["state"]
        state_counts[s] = state_counts.get(s, 0) + 1

    states_list = [
        {"abbr": abbr, "name": name.title(), "county_count": state_counts.get(abbr, 0)}
        for name, abbr in STATE_NAME_TO_ABBR.items()
        if abbr in state_counts
    ]
    states_list.sort(key=lambda x: x["name"])

    metros = [
        "Southern California", "Bay Area", "NYC", "Boston", "Chicago",
        "Dallas", "Houston", "Austin", "Miami", "Atlanta", "Denver",
        "Seattle", "Portland", "Phoenix", "Washington DC",
        "Nashville", "Charlotte", "Philadelphia",
    ]

    broad_regions = [
        "West Coast", "East Coast", "Midwest", "Southeast",
        "Sun Belt", "Mountain West", "Pacific Northwest", "New England",
        "Great Plains",
    ]

    return {
        "total_counties_in_dataset": len(matrix),
        "states": states_list,
        "supported_metros": metros,
        "broad_regions": broad_regions,
        "business_types": list(BUSINESS_PROFILES.keys()),
        "priority_options": [
            "foot traffic",
            "affordable rent",
            "growth potential",
            "low competition",
            "stability",
            "target customer proximity",
        ],
    }


# ── Linecode resolver ─────────────────────────────────────────────────────────

_INDUSTRY_KEYWORD_MAP = {
    "agriculture":             3,
    "farming":                 3,
    "mining":                  6,
    "utilities":               10,
    "construction":            11,
    "manufacturing":           12,
    "wholesale":               34,
    "wholesale trade":         34,
    "retail":                  35,
    "retail trade":            35,
    "transportation":          36,
    "information":             45,
    "tech":                    45,
    "technology":              45,
    "finance":                 51,
    "finance and insurance":   51,
    "insurance":               51,
    "real estate":             56,
    "professional":            59,
    "professional services":   59,
    "business services":       59,
    "education":               69,
    "educational services":    69,
    "healthcare":              70,
    "health care":             70,
    "social assistance":       70,
    "arts":                    76,
    "entertainment":           76,
    "recreation":              76,
    "accommodation":           79,
    "food services":           79,
    "restaurant":              79,
    "accommodation and food":  79,
    "accommodation and food services": 79,
    "other services":          82,
    "government":              83,
    "all industries":          1,
    "total":                   1,
    "private industries":      2,
}


def _resolve_linecode(industry: str) -> int | None:
    """Resolve an industry name or numeric string to a BEA linecode."""
    s = industry.strip()
    # Try direct numeric
    try:
        return int(s)
    except ValueError:
        pass
    # Try keyword map
    return _INDUSTRY_KEYWORD_MAP.get(s.lower())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
