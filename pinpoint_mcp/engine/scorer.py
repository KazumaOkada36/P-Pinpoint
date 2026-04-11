"""
Core scoring engine.

Given a county's feature dict and a business profile, computes a 0-100 score
broken down into five sub-dimensions:

  industry_health  — How well is the target industry doing here?
                     (share of total GDP, location quotient, 5yr CAGR)

  growth           — Is the overall economy growing?
                     (total GDP 5yr CAGR, COVID recovery ratio)

  market_size      — How large is the economic base?
                     (log-normalized total GDP, already pre-computed)

  complementary    — Are the demand-driving industries present?
                     (weighted sum of complementary industry shares)

  stability        — Is the economy stable and diversified?
                     (diversity score, inverse of volatility)

Each sub-score is [0, 1] before weights are applied.
Final score = weighted sum × 100.
"""

from typing import Optional
from ..data.business_map import BUSINESS_PROFILES, get_weights


# ── Sub-score computation ─────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _industry_health_score(
    geo_feat: dict,
    primary_lc: int,
    avoid_saturation: bool,
) -> float:
    """
    Score how healthy the primary industry is in this county.

    Components:
      share   — industry's % of total county GDP (higher = more established)
      lq      — location quotient vs national avg (>1 = specialized)
      cagr    — 5-year growth rate of the primary industry
    """
    ind = geo_feat["industry"].get(primary_lc)
    if not ind:
        return 0.3  # no data = neutral-low score

    share = ind.get("share") or 0.0
    lq    = ind.get("lq") or 1.0
    cagr  = ind.get("cagr_5yr") or 0.0

    # Normalize share: 0% → 0, ≥20% → 1.0
    share_score = _clamp(share / 0.20)

    # Normalize LQ: 0 → 0, 1.0 (national avg) → 0.5, ≥2.0 → 1.0
    lq_score = _clamp((lq - 0.0) / 2.0)

    # Normalize CAGR: −5% → 0, +10% → 1.0
    cagr_score = _clamp((cagr - (-0.05)) / 0.15)

    raw = 0.40 * share_score + 0.35 * lq_score + 0.25 * cagr_score

    # If market is saturated, penalize (e.g., retail in a retail-heavy county)
    if avoid_saturation and lq > 1.8:
        saturation_penalty = _clamp((lq - 1.8) / 1.2) * 0.25
        raw = max(0.0, raw - saturation_penalty)

    return raw


def _growth_score(geo_feat: dict) -> float:
    """
    Score overall economic growth trajectory.

    Components:
      cagr_5yr        — total GDP compound growth 2019→2024
      covid_recovery  — 2024 GDP / 2019 GDP ratio
    """
    total = geo_feat["total"]

    cagr = total.get("cagr_5yr") or 0.0
    recovery = total.get("covid_recovery") or 1.0

    # Normalize CAGR: −2% → 0, +8% → 1.0
    cagr_score = _clamp((cagr - (-0.02)) / 0.10)

    # Normalize recovery: 0.70 → 0, 1.30 → 1.0
    recovery_score = _clamp((recovery - 0.70) / 0.60)

    return 0.60 * cagr_score + 0.40 * recovery_score


def _market_size_score(geo_feat: dict) -> float:
    """Already pre-normalized to [0,1] in features.py."""
    return geo_feat["total"].get("market_size_norm", 0.5)


def _complementary_score(geo_feat: dict, complementary: dict[int, float]) -> float:
    """
    Weighted presence of demand-generating complementary industries.
    Each complementary LC has a relative weight (already sums to ~1 in business_map).
    Industry share is used as the presence signal.
    """
    if not complementary:
        return 0.5

    total_weight = sum(complementary.values())
    if total_weight == 0:
        return 0.5

    score = 0.0
    for lc, w in complementary.items():
        ind = geo_feat["industry"].get(lc)
        share = (ind.get("share") or 0.0) if ind else 0.0
        # Normalize share contribution: 0% → 0, ≥15% → 1.0
        presence = _clamp(share / 0.15)
        score += (w / total_weight) * presence

    return score


def _stability_score(geo_feat: dict) -> float:
    """
    Score economic stability and resilience.

    Components:
      diversity_score  — 1 - HHI (higher = more diverse = less sector risk)
      volatility       — std-dev of annual growth rates (lower = more stable)
    """
    diversity  = geo_feat.get("diversity_score", 0.5)
    volatility = geo_feat["total"].get("volatility", 0.03)

    # Normalize volatility: ≥8% std-dev → 0 (very volatile), 0% → 1.0
    vol_score = _clamp(1.0 - volatility / 0.08)

    return 0.60 * diversity + 0.40 * vol_score


# ── Top-level scorer ──────────────────────────────────────────────────────────

def score_county(geo_feat: dict, business_type: str, priorities: list[str]) -> dict:
    """
    Score a county for a given business type and priority set.

    Returns a dict with:
      total          — final score 0-100
      breakdown      — sub-scores (each 0-100 before weighting)
      weights        — weight applied to each dimension
    """
    profile = BUSINESS_PROFILES.get(business_type) or BUSINESS_PROFILES["cafe"]
    weights = get_weights(priorities)

    primary_lc    = profile["primary"]
    complementary = profile["complementary"]
    avoid_sat     = profile.get("avoid_primary", False)

    ih = _industry_health_score(geo_feat, primary_lc, avoid_sat)
    gr = _growth_score(geo_feat)
    ms = _market_size_score(geo_feat)
    cp = _complementary_score(geo_feat, complementary)
    st = _stability_score(geo_feat)

    raw_total = (
        weights["industry_health"] * ih
        + weights["growth"]        * gr
        + weights["market_size"]   * ms
        + weights["complementary"] * cp
        + weights["stability"]     * st
    )

    final = round(_clamp(raw_total) * 100, 1)

    return {
        "total": final,
        "breakdown": {
            "industry_health": round(ih * 100, 1),
            "growth":          round(gr * 100, 1),
            "market_size":     round(ms * 100, 1),
            "complementary":   round(cp * 100, 1),
            "stability":       round(st * 100, 1),
        },
        "weights": {k: round(v, 3) for k, v in weights.items()},
    }
