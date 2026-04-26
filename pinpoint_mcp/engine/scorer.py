"""
Core scoring engine.

Given a county's feature dict and a business profile, computes a 0-100 score
broken into eight sub-dimensions:

  industry_health    — BEA: primary industry GDP share, LQ, 5yr CAGR
  growth             — BEA: total economy 5yr CAGR + COVID recovery
  market_size        — BEA: log-normalized total GDP
  complementary      — BEA: weighted complementary industry shares
  stability          — BEA: economic diversity + low volatility
  competitive_density — CBP: establishment count in primary NAICS
  talent_supply      — LODES + OEWS: skilled workforce density + wage affordability
  tax_climate        — Tax Foundation: state corporate rate (inverted)

Each sub-score is [0, 1] before weights are applied.
Final score = weighted sum × 100.
"""

from typing import Optional
from ..data.business_map import BUSINESS_PROFILES, get_weights


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Original five sub-scores (BEA data) ───────────────────────────────────────

def _industry_health_score(geo_feat: dict, primary_lc: int, avoid_saturation: bool) -> float:
    ind   = geo_feat["industry"].get(primary_lc)
    if not ind:
        return 0.3

    share = ind.get("share") or 0.0
    lq    = ind.get("lq")    or 1.0
    cagr  = ind.get("cagr_5yr") or 0.0

    share_score = _clamp(share / 0.20)
    lq_score    = _clamp(lq / 2.0)
    cagr_score  = _clamp((cagr - (-0.05)) / 0.15)

    raw = 0.40 * share_score + 0.35 * lq_score + 0.25 * cagr_score

    if avoid_saturation and lq > 1.8:
        raw = max(0.0, raw - _clamp((lq - 1.8) / 1.2) * 0.25)

    return raw


def _growth_score(geo_feat: dict) -> float:
    total    = geo_feat["total"]
    cagr     = total.get("cagr_5yr")    or 0.0
    recovery = total.get("covid_recovery") or 1.0

    cagr_score     = _clamp((cagr     - (-0.02)) / 0.10)
    recovery_score = _clamp((recovery - 0.70)    / 0.60)
    return 0.60 * cagr_score + 0.40 * recovery_score


def _market_size_score(geo_feat: dict) -> float:
    return geo_feat["total"].get("market_size_norm", 0.5)


def _complementary_score(geo_feat: dict, complementary: dict[int, float]) -> float:
    if not complementary:
        return 0.5

    total_weight = sum(complementary.values())
    if total_weight == 0:
        return 0.5

    score = 0.0
    for lc, w in complementary.items():
        ind    = geo_feat["industry"].get(lc)
        share  = (ind.get("share") or 0.0) if ind else 0.0
        score += (w / total_weight) * _clamp(share / 0.15)
    return score


def _stability_score(geo_feat: dict) -> float:
    diversity  = geo_feat.get("diversity_score", 0.5)
    volatility = geo_feat["total"].get("volatility", 0.03)
    vol_score  = _clamp(1.0 - volatility / 0.08)
    return 0.60 * diversity + 0.40 * vol_score


# ── Three new sub-scores (CBP + LODES + OEWS + Tax) ──────────────────────────

def _competitive_density_score(
    geo_feat: dict,
    primary_lc: int,
    avoid_saturation: bool,
) -> float:
    """
    Score the competitive landscape for the primary industry.

    Reads CBP establishment counts.  Density = establishments per 1,000 jobs.

    avoid_saturation=True  (retail, gym):
      Fewer competitors is better.  High density → lower score.
      Target: < 2 est/1k jobs is wide-open; ≥ 10 est/1k is saturated.

    avoid_saturation=False (tech, finance, healthcare, etc.):
      Agglomeration matters.  Some density signals a healthy ecosystem.
      Target: 1–5 est/1k is healthy; 0 = nascent market (modest score).
    """
    cbp      = geo_feat.get("cbp", {})
    density  = cbp.get("industry_est_per_1k_jobs", {}).get(primary_lc)

    if density is None:
        return 0.5  # no data — neutral

    if avoid_saturation:
        # Lower density = more room to operate = higher score
        # 0 est/1k → 1.0; ≥ 10 est/1k → 0.0
        return _clamp(1.0 - density / 10.0)
    else:
        # Moderate density is good (ecosystem exists); very high is fine too
        # 0 → 0.3 (nascent), 3 → 0.8, ≥ 6 → 1.0
        return _clamp(0.3 + (density / 6.0) * 0.7)


def _talent_supply_score(
    geo_feat: dict,
    business_type: str,
    occ_group: str,
) -> float:
    """
    Score talent availability and affordability.

    Two signals (weighted equally):
      density    — LODES high_wage_share: fraction of workers earning >$3,333/mo
                   Proxy for skilled-workforce concentration in the county.
      affordability — OEWS median hourly wage for the primary occupation group.
                   Lower wage = more affordable to hire.
                   Capped at $50/hr as "very expensive"; $10/hr = very cheap.

    Consumer-facing businesses (cafe, restaurant, retail, gym) value
    affordability more; knowledge businesses (tech, finance) value density more.
    """
    lodes = geo_feat.get("lodes", {})
    oews  = geo_feat.get("oews",  {})

    # ── Density signal (from LODES) ───────────────────────────────────────────
    high_wage_share = lodes.get("high_wage_share", 0.0)
    # National median high-wage share is ~30 %; cap normalisation at 60 %
    density_score = _clamp(high_wage_share / 0.60)

    # ── Affordability signal (from OEWS) ─────────────────────────────────────
    occ_data = oews.get(occ_group)
    if occ_data and occ_data.get("median_hourly"):
        wage = occ_data["median_hourly"]
        # $10/hr → 1.0 (very affordable), $50/hr → 0.0 (very expensive)
        affordability_score = _clamp(1.0 - (wage - 10.0) / 40.0)
    else:
        affordability_score = 0.5  # no data — neutral

    # Weight: consumer-facing businesses care more about affordability
    consumer_facing = business_type in {"cafe", "restaurant", "retail", "gym"}
    if consumer_facing:
        return 0.35 * density_score + 0.65 * affordability_score
    else:
        return 0.65 * density_score + 0.35 * affordability_score


def _tax_climate_score(geo_feat: dict) -> float:
    """
    Score the state's corporate tax environment.
    Reads the pre-computed rate_score (0-1, higher = lower tax = better).
    Falls back to 0.5 if tax data is unavailable.
    """
    return geo_feat.get("tax", {}).get("rate_score", 0.5)


# ── Top-level scorer ──────────────────────────────────────────────────────────

def score_county(geo_feat: dict, business_type: str, priorities: list[str]) -> dict:
    """
    Score a county for a given business type and priority set.

    Returns:
      total      — final score 0-100
      breakdown  — sub-scores (each 0-100, before weighting)
      weights    — weight applied to each dimension
    """
    profile = BUSINESS_PROFILES.get(business_type) or BUSINESS_PROFILES["cafe"]
    weights = get_weights(priorities)

    primary_lc    = profile["primary"]
    complementary = profile["complementary"]
    avoid_sat     = profile.get("avoid_primary", False)
    occ_group     = profile.get("occ_group", "35")

    ih = _industry_health_score(geo_feat, primary_lc, avoid_sat)
    gr = _growth_score(geo_feat)
    ms = _market_size_score(geo_feat)
    cp = _complementary_score(geo_feat, complementary)
    st = _stability_score(geo_feat)
    cd = _competitive_density_score(geo_feat, primary_lc, avoid_sat)
    ts = _talent_supply_score(geo_feat, business_type, occ_group)
    tc = _tax_climate_score(geo_feat)

    raw_total = (
        weights["industry_health"]    * ih
        + weights["growth"]           * gr
        + weights["market_size"]      * ms
        + weights["complementary"]    * cp
        + weights["stability"]        * st
        + weights["competitive_density"] * cd
        + weights["talent_supply"]    * ts
        + weights["tax_climate"]      * tc
    )

    final = round(_clamp(raw_total) * 100, 1)

    return {
        "total": final,
        "breakdown": {
            "industry_health":    round(ih * 100, 1),
            "growth":             round(gr * 100, 1),
            "market_size":        round(ms * 100, 1),
            "complementary":      round(cp * 100, 1),
            "stability":          round(st * 100, 1),
            "competitive_density": round(cd * 100, 1),
            "talent_supply":      round(ts * 100, 1),
            "tax_climate":        round(tc * 100, 1),
        },
        "weights": {k: round(v, 3) for k, v in weights.items()},
    }
