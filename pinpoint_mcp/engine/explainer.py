"""
Generates human-readable explanations for why a county scored well (or poorly).

This is the SHAP-style attribution layer: each dimension's contribution
to the final score is computed and rendered into a narrative sentence.
"""

from ..data.business_map import BUSINESS_PROFILES


def explain_score(
    breakdown: dict[str, float],
    weights: dict[str, float],
    geo_feat: dict,
    business_type: str,
    priorities: list[str],
) -> str:
    """
    Build a 2-3 sentence explanation of the score.

    breakdown keys: industry_health, growth, market_size, complementary, stability
    Each value is 0-100 before the weight is applied.
    """
    profile = BUSINESS_PROFILES.get(business_type, BUSINESS_PROFILES["cafe"])
    primary_lc = profile["primary"]
    primary_ind = geo_feat["industry"].get(primary_lc, {})

    county_name = geo_feat.get("county") or geo_feat.get("geoname", "This county")
    state = geo_feat.get("state", "")

    # ── Identify the top contributing dimension ────────────────────────────────
    contributions = {
        k: (breakdown[k] / 100.0) * weights[k]
        for k in breakdown
    }
    top_dim = max(contributions, key=contributions.get)
    weak_dim = min(contributions, key=contributions.get)

    # ── Build sentences by dimension ──────────────────────────────────────────
    sentences = []

    # 1. Lead with the strongest signal
    if top_dim == "industry_health":
        lq = primary_ind.get("lq")
        cagr = primary_ind.get("cagr_5yr")
        share = primary_ind.get("share")
        parts = []
        if lq and lq > 1.2:
            parts.append(f"location quotient of {lq:.1f}x the national average")
        if cagr is not None:
            dir_word = "growing" if cagr >= 0 else "contracting"
            parts.append(f"{dir_word} at {abs(cagr * 100):.1f}% annually")
        if share:
            parts.append(f"representing {share * 100:.1f}% of local GDP")
        detail = ", ".join(parts) if parts else "strong local presence"
        biz_label = _business_label(business_type)
        sentences.append(
            f"{county_name}'s {biz_label} sector is a standout — {detail}."
        )

    elif top_dim == "growth":
        cagr = geo_feat["total"].get("cagr_5yr")
        recovery = geo_feat["total"].get("covid_recovery")
        if cagr is not None and recovery is not None:
            sentences.append(
                f"{county_name}'s economy grew at {cagr * 100:.1f}% CAGR over the last 5 years "
                f"and recovered to {recovery:.2f}x its pre-COVID output."
            )
        elif cagr is not None:
            sentences.append(
                f"{county_name} shows strong economic momentum with {cagr * 100:.1f}% annual GDP growth."
            )

    elif top_dim == "market_size":
        gdp = geo_feat["total"].get("gdp_2024")
        if gdp:
            sentences.append(
                f"{county_name} is a large economic market "
                f"(${gdp / 1_000_000:.1f}B GDP), providing a broad customer base."
            )
        else:
            sentences.append(f"{county_name} has a sizeable economic base that supports new business entry.")

    elif top_dim == "complementary":
        sentences.append(
            f"{county_name}'s mix of complementary industries creates strong natural demand "
            f"for a {_business_label(business_type)} business."
        )

    elif top_dim == "stability":
        div = geo_feat.get("diversity_score", 0)
        sentences.append(
            f"{county_name} has a well-diversified economy "
            f"(diversity score: {div * 100:.0f}/100), reducing sector-specific risk."
        )

    # 2. Add a secondary insight if there's a meaningful secondary driver
    dims_sorted = sorted(contributions, key=contributions.get, reverse=True)
    if len(dims_sorted) > 1:
        second_dim = dims_sorted[1]
        if second_dim != top_dim and contributions[second_dim] > 0.10:
            sentences.append(_dim_sentence(second_dim, breakdown[second_dim], geo_feat, county_name, business_type))

    # 3. Caveat for weak dimension if notably low
    if breakdown[weak_dim] < 35 and weak_dim != top_dim:
        sentences.append(_weak_sentence(weak_dim, geo_feat, county_name))

    return " ".join(s for s in sentences if s)


def _business_label(business_type: str) -> str:
    labels = {
        "cafe":          "food & beverage",
        "restaurant":    "restaurant",
        "retail":        "retail",
        "gym":           "fitness & recreation",
        "tech":          "technology",
        "healthcare":    "healthcare",
        "finance":       "financial services",
        "real_estate":   "real estate",
        "manufacturing": "manufacturing",
        "education":     "education",
    }
    return labels.get(business_type, business_type)


def _dim_sentence(dim: str, score: float, geo_feat: dict, county: str, business_type: str) -> str:
    if dim == "growth":
        cagr = geo_feat["total"].get("cagr_5yr")
        if cagr is not None:
            return f"Overall economic growth is solid at {cagr * 100:.1f}% annually."
    if dim == "market_size":
        gdp = geo_feat["total"].get("gdp_2024")
        if gdp:
            return f"The local economy (${gdp / 1_000_000:.1f}B GDP) offers meaningful scale."
    if dim == "complementary":
        return f"Complementary industries provide a ready-made customer base for {_business_label(business_type)}."
    if dim == "stability":
        div = geo_feat.get("diversity_score", 0)
        return f"Economic diversity (score: {div * 100:.0f}/100) provides a resilient business environment."
    if dim == "industry_health":
        return f"The target industry has a healthy presence in {county}."
    return ""


def _weak_sentence(dim: str, geo_feat: dict, county: str) -> str:
    if dim == "stability":
        vol = geo_feat["total"].get("volatility", 0)
        return (
            f"Note: {county}'s economy is relatively concentrated "
            f"(volatility: {vol * 100:.1f}%), which may mean higher risk."
        )
    if dim == "market_size":
        return f"Note: {county} is a smaller market, which may limit total addressable customers."
    if dim == "growth":
        cagr = geo_feat["total"].get("cagr_5yr")
        if cagr is not None and cagr < 0:
            return f"Note: The local economy contracted {abs(cagr * 100):.1f}% annually — verify current conditions."
    if dim == "complementary":
        return f"Note: Complementary demand drivers are limited here — brand differentiation will matter more."
    if dim == "industry_health":
        return f"Note: The target industry is less established in {county} — early-mover advantage possible."
    return ""
