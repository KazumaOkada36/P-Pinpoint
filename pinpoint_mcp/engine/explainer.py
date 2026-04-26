"""
Generates human-readable explanations for why a county scored well (or poorly).

Covers all eight scoring dimensions:
  industry_health, growth, market_size, complementary, stability,
  competitive_density, talent_supply, tax_climate
"""

from ..data.business_map import BUSINESS_PROFILES


def explain_score(
    breakdown: dict[str, float],
    weights: dict[str, float],
    geo_feat: dict,
    business_type: str,
    priorities: list[str],
) -> str:
    profile    = BUSINESS_PROFILES.get(business_type, BUSINESS_PROFILES["cafe"])
    primary_lc = profile["primary"]
    primary_ind = geo_feat["industry"].get(primary_lc, {})

    county_name = geo_feat.get("county") or geo_feat.get("geoname", "This county")

    # Weighted contribution per dimension
    contributions = {
        k: (breakdown.get(k, 0) / 100.0) * weights.get(k, 0)
        for k in breakdown
    }
    top_dim  = max(contributions, key=contributions.get)
    weak_dim = min(contributions, key=contributions.get)

    sentences = []

    # ── Lead sentence: strongest signal ───────────────────────────────────────
    if top_dim == "industry_health":
        lq    = primary_ind.get("lq")
        cagr  = primary_ind.get("cagr_5yr")
        share = primary_ind.get("share")
        parts = []
        if lq and lq > 1.2:
            parts.append(f"location quotient of {lq:.1f}x the national average")
        if cagr is not None:
            dir_word = "growing" if cagr >= 0 else "contracting"
            parts.append(f"{dir_word} at {abs(cagr * 100):.1f}% annually")
        if share:
            parts.append(f"representing {share * 100:.1f}% of local GDP")
        detail    = ", ".join(parts) if parts else "strong local presence"
        biz_label = _business_label(business_type)
        sentences.append(f"{county_name}'s {biz_label} sector is a standout — {detail}.")

    elif top_dim == "growth":
        cagr     = geo_feat["total"].get("cagr_5yr")
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
        lodes = geo_feat.get("lodes", {})
        jobs  = lodes.get("total_jobs", 0)
        if jobs > 0:
            sentences.append(
                f"{county_name} is a large market with {jobs:,} daytime workers "
                f"and ${gdp / 1_000_000:.1f}B in GDP — broad customer base."
            )
        elif gdp:
            sentences.append(
                f"{county_name} is a large economic market "
                f"(${gdp / 1_000_000:.1f}B GDP), providing a broad customer base."
            )

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

    elif top_dim == "competitive_density":
        cbp     = geo_feat.get("cbp", {})
        primary_lc = profile["primary"]
        est     = cbp.get("industry_establishments", {}).get(primary_lc)
        density = cbp.get("industry_est_per_1k_jobs", {}).get(primary_lc)
        avoid   = profile.get("avoid_primary", False)
        if est is not None and avoid:
            sentences.append(
                f"With only {est:,} competing establishments in {county_name}'s primary sector "
                f"({density:.1f} per 1,000 jobs), there is meaningful whitespace to capture."
            )
        elif est is not None:
            sentences.append(
                f"{county_name} has {est:,} businesses in the primary sector, "
                f"signaling a healthy industry ecosystem."
            )
        else:
            sentences.append(
                f"{county_name} shows a favorable competitive landscape for this business type."
            )

    elif top_dim == "talent_supply":
        lodes = geo_feat.get("lodes", {})
        hw    = lodes.get("high_wage_share", 0.0)
        oews  = geo_feat.get("oews", {})
        occ   = profile.get("occ_group", "35")
        wage_data = oews.get(occ)
        if hw > 0.0 and wage_data:
            wage = wage_data.get("median_hourly", 0)
            sentences.append(
                f"{county_name} has a strong talent base — {hw * 100:.0f}% high-wage workers "
                f"and median hourly wages of ${wage:.0f} for this occupation."
            )
        elif hw > 0.0:
            sentences.append(
                f"{county_name} has a skilled workforce concentration "
                f"({hw * 100:.0f}% high-wage workers), easing hiring."
            )
        else:
            sentences.append(
                f"{county_name} offers favorable workforce conditions for this business type."
            )

    elif top_dim == "tax_climate":
        rate = geo_feat.get("tax", {}).get("top_rate", 0)
        if rate == 0:
            sentences.append(
                f"{county_name}'s state has no corporate income tax — "
                f"a significant cost advantage for incorporation."
            )
        else:
            sentences.append(
                f"{county_name}'s state has a {rate * 100:.1f}% top corporate tax rate, "
                f"among the more business-friendly in the country."
            )

    # ── Secondary insight ─────────────────────────────────────────────────────
    dims_sorted = sorted(contributions, key=contributions.get, reverse=True)
    if len(dims_sorted) > 1:
        second_dim = dims_sorted[1]
        if second_dim != top_dim and contributions[second_dim] > 0.08:
            s = _dim_sentence(second_dim, breakdown.get(second_dim, 0), geo_feat, county_name, business_type, profile)
            if s:
                sentences.append(s)

    # ── Caveat for notably weak dimension ─────────────────────────────────────
    if breakdown.get(weak_dim, 100) < 35 and weak_dim != top_dim:
        s = _weak_sentence(weak_dim, geo_feat, county_name)
        if s:
            sentences.append(s)

    return " ".join(s for s in sentences if s)


# ── Secondary / caveat sentence builders ─────────────────────────────────────

def _dim_sentence(dim: str, score: float, geo_feat: dict, county: str, business_type: str, profile: dict) -> str:
    if dim == "growth":
        cagr = geo_feat["total"].get("cagr_5yr")
        if cagr is not None:
            return f"Overall economic growth is solid at {cagr * 100:.1f}% annually."
    if dim == "market_size":
        lodes = geo_feat.get("lodes", {})
        jobs  = lodes.get("total_jobs", 0)
        if jobs > 0:
            return f"The county has {jobs:,} daytime workers, providing meaningful customer scale."
        gdp = geo_feat["total"].get("gdp_2024")
        if gdp:
            return f"The local economy (${gdp / 1_000_000:.1f}B GDP) offers meaningful scale."
    if dim == "complementary":
        return f"Complementary industries provide a ready-made customer base for {_business_label(business_type)}."
    if dim == "stability":
        div = geo_feat.get("diversity_score", 0)
        return f"Economic diversity ({div * 100:.0f}/100) provides a resilient business environment."
    if dim == "industry_health":
        return f"The target industry has a healthy presence in {county}."
    if dim == "competitive_density":
        cbp     = geo_feat.get("cbp", {})
        primary_lc = profile["primary"]
        est     = cbp.get("industry_establishments", {}).get(primary_lc)
        if est is not None:
            return f"Competitive landscape: {est:,} businesses in this sector."
    if dim == "talent_supply":
        hw = geo_feat.get("lodes", {}).get("high_wage_share", 0)
        if hw > 0:
            return f"Skilled workforce density ({hw * 100:.0f}% high-wage workers) supports hiring."
    if dim == "tax_climate":
        rate = geo_feat.get("tax", {}).get("top_rate", 0)
        return f"State corporate tax rate of {rate * 100:.1f}% keeps overhead predictable."
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
    if dim == "competitive_density":
        return f"Note: High competitive density in {county} — strong differentiation will be needed."
    if dim == "talent_supply":
        return f"Note: Talent supply is limited in {county} — remote hiring or relocation incentives may help."
    if dim == "tax_climate":
        rate = geo_feat.get("tax", {}).get("top_rate", 0)
        if rate > 0.08:
            return f"Note: {county}'s state has a {rate * 100:.1f}% corporate rate — factor into incorporation cost."
    return ""


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
