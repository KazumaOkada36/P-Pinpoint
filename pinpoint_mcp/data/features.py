"""
Pre-computes a feature matrix from the loaded BEA data.

Features per (geofips × linecode):
  - gdp_2024         raw GDP value (thousands chained 2017$)
  - cagr_5yr         5-year CAGR (2019→2024)
  - cagr_full        full-period CAGR (2001→2024)
  - covid_recovery   2024/2019 ratio (>1 = recovered, <1 = still depressed)

Features per geofips (county-level aggregates):
  - total_gdp_2024
  - total_cagr_5yr
  - total_covid_recovery
  - diversity_score      1 − HHI across top-level industries (higher = more diverse)
  - volatility           std-dev of annual growth rates for total GDP
  - industry_shares      {linecode: share_of_total_2024}
  - location_quotients   {linecode: county_share / national_share}
"""

import math
from functools import lru_cache
from typing import Optional

from .loader import load_raw, YEARS, TOP_LEVEL_LINECODES, RELEVANT_LINECODES

YEAR_2001_IDX = 0
YEAR_2019_IDX = YEARS.index(2019)
YEAR_2020_IDX = YEARS.index(2020)
YEAR_2024_IDX = YEARS.index(2024)


# ── Low-level math helpers ────────────────────────────────────────────────────

def _cagr(start: Optional[float], end: Optional[float], n: int) -> Optional[float]:
    if start is None or end is None or start <= 0 or end <= 0 or n <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0


def _annual_growth_rates(values: list[Optional[float]]) -> list[float]:
    rates = []
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if a and b and a > 0:
            rates.append((b - a) / a)
    return rates


def _mean(lst: list[float]) -> Optional[float]:
    return sum(lst) / len(lst) if lst else None


def _std(lst: list[float]) -> float:
    if len(lst) < 2:
        return 0.0
    m = sum(lst) / len(lst)
    variance = sum((x - m) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)


# ── Per-industry feature extraction ──────────────────────────────────────────

def _industry_features(ind: dict) -> dict:
    vals = ind["values"]
    v2001 = vals[YEAR_2001_IDX]
    v2019 = vals[YEAR_2019_IDX]
    v2024 = vals[YEAR_2024_IDX]
    return {
        "gdp_2024":       v2024,
        "gdp_2019":       v2019,
        "gdp_2001":       v2001,
        "cagr_5yr":       _cagr(v2019, v2024, 5),
        "cagr_full":      _cagr(v2001, v2024, 23),
        "covid_recovery": (v2024 / v2019) if (v2019 and v2019 > 0 and v2024 is not None) else None,
    }


# ── County-level feature matrix ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def compute_feature_matrix() -> dict:
    """
    Returns a dict keyed by geofips (county rows only):

    {
      "06001": {
        "geoname":   "Alameda, CA",
        "county":    "Alameda",
        "state":     "CA",
        "total": {
          "gdp_2024": ...,
          "cagr_5yr": ...,
          "covid_recovery": ...,
          "volatility": ...,         # std-dev of annual growth rates
          "log_gdp_2024": ...,
        },
        "diversity_score": 0.87,     # 1 - HHI (higher = more diverse)
        "industry": {
          linecode (int): {
            "gdp_2024": ...,
            "cagr_5yr": ...,
            "cagr_full": ...,
            "covid_recovery": ...,
            "share": ...,            # fraction of total GDP
            "lq": ...,               # location quotient vs national avg
          }
        }
      }
    }
    """
    raw = load_raw()

    # ── Pass 1: extract per-industry features for all counties ────────────────
    county_matrix: dict = {}
    for geofips, geo in raw.items():
        if not geo["is_county"]:
            continue

        industries = geo["industries"]
        total_ind = industries.get(1)
        if not total_ind:
            continue

        total_feat = _industry_features(total_ind)
        total_vals = total_ind["values"]
        total_feat["log_gdp_2024"] = (
            math.log(total_feat["gdp_2024"]) if total_feat["gdp_2024"] and total_feat["gdp_2024"] > 0 else None
        )
        total_feat["volatility"] = _std(_annual_growth_rates(total_vals))

        ind_features: dict = {}
        for lc, ind in industries.items():
            if lc == 1:
                continue
            ind_features[lc] = {
                "description": ind.get("description", ""),
                **_industry_features(ind),
            }

        county_matrix[geofips] = {
            "geoname": geo["geoname"],
            "county":  geo["county"],
            "state":   geo["state_abbr"],
            "total":   total_feat,
            "industry": ind_features,
        }

    # ── Pass 2: compute national share per linecode (for LQ) ──────────────────
    national_gdp: dict[int, float] = {}
    national_total = 0.0
    for geo_feat in county_matrix.values():
        t = geo_feat["total"]["gdp_2024"]
        if t:
            national_total += t
        for lc, ind in geo_feat["industry"].items():
            v = ind["gdp_2024"]
            if v:
                national_gdp[lc] = national_gdp.get(lc, 0.0) + v

    national_shares: dict[int, float] = {
        lc: v / national_total for lc, v in national_gdp.items() if national_total > 0
    }

    # ── Pass 3: compute per-county industry share and LQ ─────────────────────
    for geofips, geo_feat in county_matrix.items():
        total_gdp = geo_feat["total"]["gdp_2024"] or 0.0
        hhi = 0.0

        for lc, ind in geo_feat["industry"].items():
            v = ind["gdp_2024"]
            share = (v / total_gdp) if (v and total_gdp > 0) else None
            nat_share = national_shares.get(lc)
            lq = (share / nat_share) if (share and nat_share and nat_share > 0) else None

            ind["share"] = share
            ind["lq"] = lq

            # HHI uses only top-level codes
            if lc in TOP_LEVEL_LINECODES and share:
                hhi += share ** 2

        geo_feat["diversity_score"] = max(0.0, 1.0 - hhi)

    # ── Pass 4: normalize log_gdp across all counties ─────────────────────────
    log_gdps = [
        gf["total"]["log_gdp_2024"]
        for gf in county_matrix.values()
        if gf["total"]["log_gdp_2024"] is not None
    ]
    min_log = min(log_gdps) if log_gdps else 0.0
    max_log = max(log_gdps) if log_gdps else 1.0
    log_range = max_log - min_log or 1.0

    for geo_feat in county_matrix.values():
        lg = geo_feat["total"]["log_gdp_2024"]
        geo_feat["total"]["market_size_norm"] = (lg - min_log) / log_range if lg is not None else 0.5

    return county_matrix


def get_county_features(geofips: str) -> Optional[dict]:
    return compute_feature_matrix().get(geofips)
