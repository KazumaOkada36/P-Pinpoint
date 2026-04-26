"""
Pre-computes a feature matrix from all loaded datasets.

BEA CAGDP9 features per (geofips × linecode):
  gdp_2024, cagr_5yr, cagr_full, covid_recovery, share, lq

BEA aggregate per geofips:
  total.gdp_2024, total.cagr_5yr, total.covid_recovery
  total.volatility, total.log_gdp_2024, total.market_size_norm
  diversity_score

LODES WAC (new) — daytime workforce per geofips:
  lodes.total_jobs, lodes.high_wage_share, lodes.mid_wage_share,
  lodes.low_wage_share, lodes.daytime_jobs_norm (log-normalized 0-1)
  lodes.industry_jobs      {lc: int}
  lodes.industry_job_share {lc: float}

CBP (new) — competitive landscape per geofips:
  cbp.total_establishments
  cbp.industry_establishments  {lc: int}
  cbp.industry_est_per_1k_jobs {lc: float}   density metric

Tax Foundation (new) — state-level cost signal:
  tax.top_rate    float  (e.g. 0.065)
  tax.rate_score  float  (0-1, higher = lower tax = better for business)

OEWS (new) — occupation wage baseline per state:
  oews.{occ_group}.median_hourly  float
  oews.{occ_group}.total_emp      int
"""

import math
from functools import lru_cache
from typing import Optional

from .loader      import load_raw, YEARS, TOP_LEVEL_LINECODES, RELEVANT_LINECODES
from .loader_lodes import load_lodes
from .loader_cbp   import load_cbp
from .loader_tax   import load_tax
from .loader_oews  import load_oews

YEAR_2001_IDX = 0
YEAR_2019_IDX = YEARS.index(2019)
YEAR_2020_IDX = YEARS.index(2020)
YEAR_2024_IDX = YEARS.index(2024)

# Top corporate rate in the U.S. used for normalization (~NJ at 11.5%)
_MAX_CORP_RATE = 0.115


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
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


# ── Per-industry feature extraction ──────────────────────────────────────────

def _industry_features(ind: dict) -> dict:
    vals  = ind["values"]
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
    Returns a dict keyed by geofips (county rows only).

    Each entry:
    {
      "geoname":        str,
      "county":         str,
      "state":          str,
      "total":          { gdp_2024, cagr_5yr, covid_recovery, volatility,
                          log_gdp_2024, market_size_norm },
      "diversity_score": float,
      "industry":       { lc: { gdp_2024, cagr_5yr, cagr_full,
                                covid_recovery, share, lq } },
      "lodes":          { total_jobs, high_wage_share, mid_wage_share,
                          low_wage_share, daytime_jobs_norm,
                          industry_jobs, industry_job_share },
      "cbp":            { total_establishments,
                          industry_establishments,
                          industry_est_per_1k_jobs },
      "tax":            { top_rate, rate_score },
      "oews":           { occ_group: { median_hourly, total_emp } },
    }
    """
    raw = load_raw()

    # ── Pass 1: BEA per-industry features for all counties ───────────────────
    county_matrix: dict = {}
    for geofips, geo in raw.items():
        if not geo["is_county"]:
            continue

        industries = geo["industries"]
        total_ind  = industries.get(1)
        if not total_ind:
            continue

        total_feat      = _industry_features(total_ind)
        total_vals      = total_ind["values"]
        total_feat["log_gdp_2024"] = (
            math.log(total_feat["gdp_2024"])
            if total_feat["gdp_2024"] and total_feat["gdp_2024"] > 0 else None
        )
        total_feat["volatility"] = _std(_annual_growth_rates(total_vals))

        ind_features: dict = {}
        for lc, ind in industries.items():
            if lc == 1:
                continue
            ind_features[lc] = {"description": ind.get("description", ""), **_industry_features(ind)}

        county_matrix[geofips] = {
            "geoname":  geo["geoname"],
            "county":   geo["county"],
            "state":    geo["state_abbr"],
            "total":    total_feat,
            "industry": ind_features,
        }

    # ── Pass 2: national share per linecode (for LQ) ─────────────────────────
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

    # ── Pass 3: per-county industry share, LQ, HHI diversity ─────────────────
    for geofips, geo_feat in county_matrix.items():
        total_gdp = geo_feat["total"]["gdp_2024"] or 0.0
        hhi = 0.0
        for lc, ind in geo_feat["industry"].items():
            v         = ind["gdp_2024"]
            share     = (v / total_gdp) if (v and total_gdp > 0) else None
            nat_share = national_shares.get(lc)
            lq        = (share / nat_share) if (share and nat_share and nat_share > 0) else None
            ind["share"] = share
            ind["lq"]    = lq
            if lc in TOP_LEVEL_LINECODES and share:
                hhi += share ** 2
        geo_feat["diversity_score"] = max(0.0, 1.0 - hhi)

    # ── Pass 4: normalize log_gdp (market_size_norm) ─────────────────────────
    log_gdps = [
        gf["total"]["log_gdp_2024"]
        for gf in county_matrix.values()
        if gf["total"]["log_gdp_2024"] is not None
    ]
    min_log   = min(log_gdps) if log_gdps else 0.0
    max_log   = max(log_gdps) if log_gdps else 1.0
    log_range = max_log - min_log or 1.0

    for geo_feat in county_matrix.values():
        lg = geo_feat["total"]["log_gdp_2024"]
        geo_feat["total"]["market_size_norm"] = (lg - min_log) / log_range if lg is not None else 0.5

    # ── Pass 5: LODES — join daytime workforce data ───────────────────────────
    lodes = load_lodes()

    all_jobs = [d["total_jobs"] for d in lodes.values() if d["total_jobs"] > 0]
    log_jobs_min = math.log(min(all_jobs)) if all_jobs else 0.0
    log_jobs_max = math.log(max(all_jobs)) if all_jobs else 1.0
    log_jobs_rng = log_jobs_max - log_jobs_min or 1.0

    for geofips, geo_feat in county_matrix.items():
        ld = lodes.get(geofips)
        if ld:
            log_j = math.log(ld["total_jobs"]) if ld["total_jobs"] > 0 else None
            norm  = (log_j - log_jobs_min) / log_jobs_rng if log_j is not None else 0.5
            geo_feat["lodes"] = {
                **ld,
                "daytime_jobs_norm": norm,
            }
        else:
            geo_feat["lodes"] = {
                "total_jobs":         0,
                "high_wage_share":    0.0,
                "mid_wage_share":     0.0,
                "low_wage_share":     0.0,
                "daytime_jobs_norm":  0.3,   # below-average default
                "industry_jobs":      {},
                "industry_job_share": {},
            }

    # ── Pass 6: CBP — join establishment data + compute density ──────────────
    cbp = load_cbp()

    for geofips, geo_feat in county_matrix.items():
        cd  = cbp.get(geofips)
        lod = geo_feat["lodes"]
        total_jobs = lod["total_jobs"] or 1  # avoid div/0

        if cd:
            est_density: dict[int, float] = {
                lc: (est / total_jobs * 1000)  # establishments per 1,000 jobs
                for lc, est in cd["industry_establishments"].items()
            }
            geo_feat["cbp"] = {
                "total_establishments":    cd["total_establishments"],
                "industry_establishments": cd["industry_establishments"],
                "industry_est_per_1k_jobs": est_density,
            }
        else:
            geo_feat["cbp"] = {
                "total_establishments":     0,
                "industry_establishments":  {},
                "industry_est_per_1k_jobs": {},
            }

    # ── Pass 7: Tax Foundation — state-level corporate rate ──────────────────
    tax = load_tax()

    for geo_feat in county_matrix.values():
        state    = geo_feat["state"]
        td       = tax.get(state)
        top_rate = td["top_rate"] if td else 0.06  # neutral fallback: 6%
        rate_score = max(0.0, 1.0 - top_rate / _MAX_CORP_RATE)
        geo_feat["tax"] = {
            "top_rate":   top_rate,
            "rate_score": rate_score,
        }

    # ── Pass 8: OEWS — state-level occupation wage data ──────────────────────
    oews = load_oews()

    for geo_feat in county_matrix.values():
        state = geo_feat["state"]
        geo_feat["oews"] = oews.get(state, {})

    return county_matrix


def get_county_features(geofips: str) -> Optional[dict]:
    return compute_feature_matrix().get(geofips)
