"""
Ranks all counties in scope and returns the top-N results.

Accepts:
  business_type  canonical business key (from business_map.normalize_business_type)
  region_str     user-supplied region (resolved via regions.resolve_region)
  priorities     list of free-text priority strings
  top_n          how many results to return

Returns a list of location dicts ready for the Flask /recommend endpoint
and the MCP tool output.
"""

from ..data.features import compute_feature_matrix
from ..data.regions import resolve_region, get_region_label
from ..data.business_map import normalize_business_type, normalize_priority
from .scorer import score_county
from .explainer import explain_score


def rank_locations(
    business_type_raw: str,
    region_str: str | None,
    priorities: list[str],
    top_n: int = 10,
) -> dict:
    """
    Main entry point for the recommendation engine.

    Returns:
    {
      "business_type": "cafe",
      "region_label":  "California",
      "priorities":    ["foot traffic", "growth potential"],
      "results": [
        {
          "rank": 1,
          "county": "Santa Clara",
          "state": "CA",
          "geoname": "Santa Clara, CA",
          "geofips": "06085",
          "score": 82.4,
          "breakdown": { ... },
          "key_metrics": { ... },
          "explanation": "...",
        },
        ...
      ]
    }
    """
    business_type = normalize_business_type(business_type_raw)
    state_filter  = resolve_region(region_str)
    region_label  = get_region_label(region_str)

    matrix = compute_feature_matrix()

    # Filter counties by region
    candidates = {
        fips: feat
        for fips, feat in matrix.items()
        if state_filter is None or feat["state"] in state_filter
    }

    if not candidates:
        return {
            "business_type": business_type,
            "region_label":  region_label,
            "priorities":    priorities,
            "results":       [],
            "warning":       f"No county data found for region: {region_str}",
        }

    # Score all candidates
    scored = []
    for geofips, geo_feat in candidates.items():
        result = score_county(geo_feat, business_type, priorities)
        scored.append((geofips, geo_feat, result))

    # Sort by score descending
    scored.sort(key=lambda x: x[2]["total"], reverse=True)
    top = scored[:top_n]

    # Build output
    results = []
    for rank, (geofips, geo_feat, score_result) in enumerate(top, start=1):
        total = geo_feat["total"]
        primary_lc = _get_primary_lc(business_type)
        primary_ind = geo_feat["industry"].get(primary_lc, {})

        key_metrics = {
            "total_gdp_2024_billions": _fmt_billions(total.get("gdp_2024")),
            "total_gdp_cagr_5yr_pct":  _fmt_pct(total.get("cagr_5yr")),
            "covid_recovery_ratio":    _fmt_ratio(total.get("covid_recovery")),
            "economic_diversity":      _fmt_pct(geo_feat.get("diversity_score")),
            "primary_industry_share":  _fmt_pct(primary_ind.get("share")),
            "primary_industry_lq":     _fmt_ratio(primary_ind.get("lq")),
            "primary_industry_cagr_5yr": _fmt_pct(primary_ind.get("cagr_5yr")),
        }

        explanation = explain_score(
            score_result["breakdown"],
            score_result["weights"],
            geo_feat,
            business_type,
            priorities,
        )

        # Approximate lat/lng for map display (county centroid lookup)
        lat, lng = _county_latlon(geofips, geo_feat)

        results.append({
            "rank":        rank,
            "county":      geo_feat["county"],
            "state":       geo_feat["state"],
            "geoname":     geo_feat["geoname"],
            "geofips":     geofips,
            "score":       score_result["total"],
            "breakdown":   score_result["breakdown"],
            "weights":     score_result["weights"],
            "key_metrics": key_metrics,
            "explanation": explanation,
            "lat":         lat,
            "lng":         lng,
        })

    return {
        "business_type":  business_type,
        "region_label":   region_label,
        "priorities":     priorities,
        "total_counties_scored": len(candidates),
        "results":        results,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_primary_lc(business_type: str) -> int:
    from ..data.business_map import BUSINESS_PROFILES
    return BUSINESS_PROFILES.get(business_type, BUSINESS_PROFILES["cafe"])["primary"]


def _fmt_billions(v) -> str | None:
    if v is None:
        return None
    return f"{v / 1_000_000:.2f}B"


def _fmt_pct(v) -> str | None:
    if v is None:
        return None
    return f"{v * 100:.1f}%"


def _fmt_ratio(v) -> str | None:
    if v is None:
        return None
    return f"{v:.2f}x"


def _county_latlon(geofips: str, geo_feat: dict) -> tuple[float | None, float | None]:
    """
    Return approximate county centroid for map display.
    Uses a small hardcoded table for the most common counties;
    returns None, None for unknown counties (frontend can geocode).
    This table can be replaced with a full county centroid CSV later.
    """
    # Indexed by GeoFIPS: (lat, lng)
    CENTROIDS: dict[str, tuple[float, float]] = {
        # California
        "06037": (34.0522, -118.2437),  # Los Angeles
        "06073": (32.7157, -117.1611),  # San Diego
        "06059": (33.7175, -117.8311),  # Orange
        "06085": (37.3382, -121.8863),  # Santa Clara
        "06001": (37.6017, -121.7195),  # Alameda
        "06013": (37.9161, -122.0820),  # Contra Costa
        "06075": (37.7749, -122.4194),  # San Francisco
        "06083": (34.4208, -119.6982),  # Santa Barbara
        "06065": (33.9806, -116.6194),  # Riverside
        "06071": (34.1083, -117.2898),  # San Bernardino
        # New York
        "36061": (40.7831, -73.9712),   # New York (Manhattan)
        "36047": (40.6501, -73.9496),   # Kings (Brooklyn)
        "36081": (40.7282, -73.7949),   # Queens
        "36005": (40.8448, -73.8648),   # Bronx
        "36059": (40.7282, -73.5950),   # Nassau
        "36103": (40.9849, -72.8720),   # Suffolk
        # Texas
        "48201": (29.7604, -95.3698),   # Harris (Houston)
        "48113": (32.7767, -96.7970),   # Dallas
        "48029": (29.4241, -98.4936),   # Bexar (San Antonio)
        "48453": (30.2672, -97.7431),   # Travis (Austin)
        "48141": (32.7555, -97.3308),   # Tarrant (Fort Worth)
        # Florida
        "12086": (25.7617, -80.1918),   # Miami-Dade
        "12011": (28.5383, -81.3792),   # Orange (Orlando)
        "12057": (27.9506, -82.4572),   # Hillsborough (Tampa)
        "12031": (30.3322, -81.6557),   # Duval (Jacksonville)
        # Illinois
        "17031": (41.8781, -87.6298),   # Cook (Chicago)
        # Washington
        "53033": (47.6062, -122.3321),  # King (Seattle)
        # Massachusetts
        "25025": (42.3601, -71.0589),   # Suffolk (Boston)
        # Colorado
        "08031": (39.7392, -104.9903),  # Denver
        # Georgia
        "13121": (33.7490, -84.3880),   # Fulton (Atlanta)
        # Arizona
        "04013": (33.4484, -112.0740),  # Maricopa (Phoenix)
        # Nevada
        "32003": (36.1699, -115.1398),  # Clark (Las Vegas)
        # Oregon
        "41051": (45.5234, -122.6762),  # Multnomah (Portland)
        # Pennsylvania
        "42101": (39.9526, -75.1652),   # Philadelphia
        # Ohio
        "39035": (41.4993, -81.6944),   # Cuyahoga (Cleveland)
        "39049": (40.0008, -82.9291),   # Franklin (Columbus)
        # Michigan
        "26163": (42.3314, -83.0458),   # Wayne (Detroit)
        # Minnesota
        "27053": (44.9778, -93.2650),   # Hennepin (Minneapolis)
        # Maryland
        "24510": (39.2904, -76.6122),   # Baltimore City
        # Virginia
        "51059": (38.8462, -77.3064),   # Fairfax
        # North Carolina
        "37119": (35.2271, -80.8431),   # Mecklenburg (Charlotte)
        "37183": (35.7796, -78.6382),   # Wake (Raleigh)
        # Tennessee
        "47037": (36.1627, -86.7816),   # Davidson (Nashville)
        # Missouri
        "29189": (38.6270, -90.1994),   # St. Louis
        # Louisiana
        "22071": (29.9511, -90.0715),   # Orleans (New Orleans)
        # Utah
        "49035": (40.7608, -111.8910),  # Salt Lake
        # New Mexico
        "35001": (35.0853, -106.6056),  # Bernalillo (Albuquerque)
    }
    return CENTROIDS.get(geofips, (None, None))
