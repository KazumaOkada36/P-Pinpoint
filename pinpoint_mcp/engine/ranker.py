"""
Ranks all counties in scope and returns the top-N results.
"""

from ..data.features    import compute_feature_matrix
from ..data.regions     import resolve_region, get_region_label
from ..data.business_map import normalize_business_type, normalize_priority, BUSINESS_PROFILES
from .scorer            import score_county
from .explainer         import explain_score


def rank_locations(
    business_type_raw: str,
    region_str: str | None,
    priorities: list[str],
    top_n: int = 10,
) -> dict:
    business_type = normalize_business_type(business_type_raw)
    state_filter  = resolve_region(region_str)
    region_label  = get_region_label(region_str)

    matrix = compute_feature_matrix()

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

    scored = []
    for geofips, geo_feat in candidates.items():
        result = score_county(geo_feat, business_type, priorities)
        scored.append((geofips, geo_feat, result))

    scored.sort(key=lambda x: x[2]["total"], reverse=True)
    top = scored[:top_n]

    results = []
    for rank, (geofips, geo_feat, score_result) in enumerate(top, start=1):
        total       = geo_feat["total"]
        primary_lc  = _get_primary_lc(business_type)
        primary_ind = geo_feat["industry"].get(primary_lc, {})

        # ── LODES metrics ─────────────────────────────────────────────────────
        lodes       = geo_feat.get("lodes", {})
        lodes_jobs  = lodes.get("total_jobs", 0)
        lodes_hw    = lodes.get("high_wage_share", 0.0)
        ind_job_sh  = lodes.get("industry_job_share", {}).get(primary_lc)

        # ── CBP metrics ───────────────────────────────────────────────────────
        cbp         = geo_feat.get("cbp", {})
        ind_est     = cbp.get("industry_establishments", {}).get(primary_lc)
        est_density = cbp.get("industry_est_per_1k_jobs", {}).get(primary_lc)

        # ── Tax metrics ───────────────────────────────────────────────────────
        tax_rate    = geo_feat.get("tax", {}).get("top_rate")

        key_metrics = {
            # BEA signals
            "total_gdp_2024_billions":   _fmt_billions(total.get("gdp_2024")),
            "total_gdp_cagr_5yr_pct":    _fmt_pct(total.get("cagr_5yr")),
            "covid_recovery_ratio":      _fmt_ratio(total.get("covid_recovery")),
            "economic_diversity":        _fmt_pct(geo_feat.get("diversity_score")),
            "primary_industry_share":    _fmt_pct(primary_ind.get("share")),
            "primary_industry_lq":       _fmt_ratio(primary_ind.get("lq")),
            "primary_industry_cagr_5yr": _fmt_pct(primary_ind.get("cagr_5yr")),
            # LODES signals
            "daytime_jobs":              _fmt_count(lodes_jobs) if lodes_jobs else None,
            "high_wage_job_share":       _fmt_pct(lodes_hw) if lodes_hw else None,
            "primary_industry_job_share": _fmt_pct(ind_job_sh) if ind_job_sh is not None else None,
            # CBP signals
            "competitor_establishments": str(ind_est) if ind_est is not None else None,
            "est_per_1k_jobs":           f"{est_density:.1f}" if est_density is not None else None,
            # Tax signal
            "state_corporate_tax_rate":  _fmt_pct(tax_rate) if tax_rate is not None else None,
        }

        explanation = explain_score(
            score_result["breakdown"],
            score_result["weights"],
            geo_feat,
            business_type,
            priorities,
        )

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
        "business_type":         business_type,
        "region_label":          region_label,
        "priorities":            priorities,
        "total_counties_scored": len(candidates),
        "results":               results,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_primary_lc(business_type: str) -> int:
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


def _fmt_count(v) -> str | None:
    if v is None:
        return None
    return f"{v:,}"


def _load_centroids() -> dict[str, tuple[float, float]]:
    import json
    from pathlib import Path
    gaz = Path(__file__).parent.parent.parent / "Data_Set" / "clean_data" / "county_centroids.json"
    if not gaz.exists():
        return {}
    with open(gaz) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}

_CENTROIDS: dict[str, tuple[float, float]] = _load_centroids()


def _county_latlon(geofips: str, geo_feat: dict) -> tuple[float | None, float | None]:
    return _CENTROIDS.get(geofips, (None, None))


def _county_latlon_LEGACY(geofips: str, geo_feat: dict) -> tuple[float | None, float | None]:
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
        "06081": (37.5630, -122.3255),  # San Mateo
        "06067": (38.4747, -121.3542),  # Sacramento
        "06077": (37.9358, -121.2908),  # San Joaquin
        "06019": (36.7378, -119.7871),  # Fresno
        "06111": (34.3705, -119.1391),  # Ventura
        "06029": (35.3733, -119.0187),  # Kern
        "06055": (38.3166, -122.0426),  # Napa
        "06097": (38.5025, -122.8281),  # Sonoma
        "06095": (38.2975, -121.9496),  # Solano
        # New York
        "36061": (40.7831, -73.9712),   # New York (Manhattan)
        "36047": (40.6501, -73.9496),   # Kings (Brooklyn)
        "36081": (40.7282, -73.7949),   # Queens
        "36005": (40.8448, -73.8648),   # Bronx
        "36059": (40.7282, -73.5950),   # Nassau
        "36103": (40.9849, -72.8720),   # Suffolk
        "36087": (41.1220, -73.7949),   # Rockland
        "36119": (41.1220, -73.8500),   # Westchester
        "36029": (42.9634, -78.7320),   # Erie (Buffalo)
        "36055": (43.1566, -77.6088),   # Monroe (Rochester)
        # Texas
        "48201": (29.7604, -95.3698),   # Harris (Houston)
        "48113": (32.7767, -96.7970),   # Dallas
        "48029": (29.4241, -98.4936),   # Bexar (San Antonio)
        "48453": (30.2672, -97.7431),   # Travis (Austin)
        "48141": (32.7555, -97.3308),   # Tarrant (Fort Worth)
        "48085": (33.2148, -96.9642),   # Collin (Plano)
        "48121": (33.1584, -97.0641),   # Denton
        "48339": (31.5454, -97.1467),   # McLennan (Waco)
        "48439": (32.3513, -97.2890),   # Johnson
        "48215": (29.8174, -94.1385),   # Jefferson (Beaumont)
        "48027": (30.0686, -94.5857),   # Bell
        "48157": (26.3017, -98.1633),   # Hidalgo (McAllen)
        "48375": (31.9686, -102.0779),  # Midland
        "48303": (33.5779, -101.8552),  # Lubbock
        "48113": (32.7767, -96.7970),   # Dallas
        # Florida
        "12086": (25.7617, -80.1918),   # Miami-Dade
        "12011": (28.5383, -81.3792),   # Orange (Orlando)
        "12057": (27.9506, -82.4572),   # Hillsborough (Tampa)
        "12031": (30.3322, -81.6557),   # Duval (Jacksonville)
        "12099": (26.6406, -80.0533),   # Palm Beach
        "12097": (26.1224, -80.1373),   # Broward (Fort Lauderdale)
        "12071": (26.6406, -81.8723),   # Lee (Fort Myers)
        "12021": (28.4122, -81.2843),   # Brevard
        "12009": (28.7103, -82.5146),   # Citrus
        "12103": (27.4799, -82.5712),   # Sarasota
        # Illinois
        "17031": (41.8781, -87.6298),   # Cook (Chicago)
        "17043": (41.8125, -88.0937),   # DuPage
        "17197": (42.3251, -88.0076),   # Will
        "17089": (42.2711, -87.8545),   # Kane
        "17097": (42.3251, -88.4399),   # Lake
        # Washington
        "53033": (47.6062, -122.3321),  # King (Seattle)
        "53053": (47.2529, -122.4443),  # Pierce (Tacoma)
        "53061": (47.9790, -122.2021),  # Snohomish
        "53011": (47.6588, -117.4260),  # Spokane
        # Massachusetts
        "25025": (42.3601, -71.0589),   # Suffolk (Boston)
        "25017": (42.3601, -71.5500),   # Middlesex
        "25021": (42.1015, -71.5500),   # Norfolk
        "25023": (41.7015, -70.9843),   # Plymouth
        "25009": (42.5195, -71.1920),   # Essex
        # Colorado
        "08031": (39.7392, -104.9903),  # Denver
        "08041": (39.5501, -104.8697),  # El Paso (Colorado Springs)
        "08059": (39.7555, -105.2211),  # Jefferson
        "08005": (39.9716, -104.8197),  # Arapahoe
        "08001": (39.8856, -104.6753),  # Adams
        "08013": (40.0274, -105.2519),  # Boulder
        "08035": (39.6515, -104.9903),  # Douglas
        # Georgia
        "13121": (33.7490, -84.3880),   # Fulton (Atlanta)
        "13089": (33.7490, -84.2320),   # DeKalb
        "13067": (33.9801, -84.5799),   # Cobb
        "13135": (33.9801, -84.0039),   # Gwinnett
        "13063": (33.5190, -84.3568),   # Clayton
        # Arizona
        "04013": (33.4484, -112.0740),  # Maricopa (Phoenix)
        "04019": (32.1545, -110.8782),  # Pima (Tucson)
        "04021": (34.5400, -112.4685),  # Yavapai
        "04025": (34.8697, -111.7609),  # Yuma
        # Nevada
        "32003": (36.1699, -115.1398),  # Clark (Las Vegas)
        "32031": (39.5296, -119.8138),  # Washoe (Reno)
        # Oregon
        "41051": (45.5234, -122.6762),  # Multnomah (Portland)
        "41067": (45.4871, -122.6765),  # Washington
        "41005": (45.4652, -122.7069),  # Clackamas
        "41029": (44.9429, -123.0351),  # Jackson
        # Pennsylvania
        "42101": (39.9526, -75.1652),   # Philadelphia
        "42003": (40.4406, -79.9959),   # Allegheny (Pittsburgh)
        "42091": (40.1215, -75.3521),   # Montgomery
        "42017": (40.0594, -75.4791),   # Bucks
        "42045": (39.9526, -75.6082),   # Delaware
        # Ohio
        "39035": (41.4993, -81.6944),   # Cuyahoga (Cleveland)
        "39049": (40.0008, -82.9291),   # Franklin (Columbus)
        "39061": (39.1612, -84.4569),   # Hamilton (Cincinnati)
        "39113": (41.6638, -83.5552),   # Lucas (Toledo)
        "39153": (39.3265, -82.9820),   # Summit (Akron)
        # Michigan
        "26163": (42.3314, -83.0458),   # Wayne (Detroit)
        "26125": (42.6064, -83.1458),   # Oakland
        "26099": (43.0125, -83.6875),   # Macomb
        "26065": (42.9634, -85.6681),   # Kent (Grand Rapids)
        # Minnesota
        "27053": (44.9778, -93.2650),   # Hennepin (Minneapolis)
        "27123": (44.9778, -93.0900),   # Ramsey (St Paul)
        "27037": (44.8735, -93.3730),   # Dakota
        "27163": (44.9778, -93.5380),   # Washington
        # Maryland
        "24510": (39.2904, -76.6122),   # Baltimore City
        "24005": (39.4524, -76.6122),   # Baltimore County
        "24031": (39.1015, -77.0750),   # Montgomery
        "24033": (38.8462, -76.8813),   # Prince George's
        "24003": (38.9784, -76.5350),   # Anne Arundel
        # Virginia
        "51059": (38.8462, -77.3064),   # Fairfax
        "51013": (38.8781, -77.1015),   # Arlington
        "51107": (38.7401, -77.1515),   # Loudoun
        "51760": (37.5385, -77.4330),   # Richmond City
        "51710": (36.8468, -76.2851),   # Norfolk
        # North Carolina
        "37119": (35.2271, -80.8431),   # Mecklenburg (Charlotte)
        "37183": (35.7796, -78.6382),   # Wake (Raleigh)
        "37081": (36.0726, -79.7920),   # Guilford (Greensboro)
        "37067": (36.0726, -78.8986),   # Durham
        "37063": (35.9940, -78.5382),   # Durham
        # Tennessee
        "47037": (36.1627, -86.7816),   # Davidson (Nashville)
        "47157": (35.1495, -90.0490),   # Shelby (Memphis)
        "47093": (35.9606, -83.9207),   # Knox (Knoxville)
        "47065": (35.2271, -85.8597),   # Hamilton (Chattanooga)
        # Missouri
        "29189": (38.6270, -90.1994),   # St. Louis City
        "29510": (38.6270, -90.3499),   # St. Louis County
        "29095": (39.0997, -94.5786),   # Jackson (Kansas City)
        # Louisiana
        "22071": (29.9511, -90.0715),   # Orleans (New Orleans)
        "22033": (30.4215, -91.0651),   # East Baton Rouge
        "22051": (30.1658, -90.1146),   # Jefferson
        # Utah
        "49035": (40.7608, -111.8910),  # Salt Lake
        "49049": (40.2969, -111.6946),  # Utah
        "49011": (41.7160, -111.8338),  # Cache (Logan)
        "49057": (41.2230, -112.0391),  # Weber (Ogden)
        # New Mexico
        "35001": (35.0853, -106.6056),  # Bernalillo (Albuquerque)
        "35049": (35.6870, -105.9378),  # Santa Fe
        # Indiana
        "18097": (39.7684, -86.1581),   # Marion (Indianapolis)
        "18003": (41.1306, -85.1286),   # Allen (Fort Wayne)
        "18089": (38.2014, -85.7384),   # Lake
        # Wisconsin
        "55079": (43.0389, -87.9065),   # Milwaukee
        "55025": (43.0731, -89.4012),   # Dane (Madison)
        "55087": (44.3192, -88.0198),   # Outagamie
        # South Carolina
        "45079": (34.0007, -81.0348),   # Richland (Columbia)
        "45019": (32.7765, -79.9311),   # Charleston
        "45045": (34.9496, -82.4588),   # Greenville
        # Kentucky
        "21111": (38.2542, -85.7594),   # Jefferson (Louisville)
        "21067": (38.0406, -84.5037),   # Fayette (Lexington)
        # Alabama
        "01073": (33.5207, -86.8025),   # Jefferson (Birmingham)
        "01101": (32.3792, -86.3077),   # Montgomery
        "01117": (34.7304, -86.5861),   # Madison (Huntsville)
        # Oklahoma
        "40109": (35.4676, -97.5164),   # Oklahoma
        "40143": (36.1540, -95.9928),   # Tulsa
        # Connecticut
        "09009": (41.3083, -72.9279),   # New Haven
        "09003": (41.7658, -72.6734),   # Hartford
        "09001": (41.1865, -73.1950),   # Fairfield
        # New Jersey
        "34017": (40.7440, -74.0324),   # Hudson (Jersey City)
        "34013": (40.6548, -74.3008),   # Essex (Newark)
        "34039": (40.4774, -74.2591),   # Union
        "34023": (40.6169, -74.4304),   # Middlesex
        "34003": (39.9320, -74.9360),   # Burlington
        # Kansas
        "20091": (37.6872, -97.3301),   # Sedgwick (Wichita)
        "20209": (39.0473, -94.7233),   # Wyandotte (Kansas City KS)
        # Iowa
        "19153": (41.6005, -93.6091),   # Polk (Des Moines)
        "19013": (41.6835, -91.5301),   # Black Hawk (Cedar Rapids)
        # Nebraska
        "31055": (41.2565, -95.9345),   # Douglas (Omaha)
        "31109": (40.8136, -96.7026),   # Lancaster (Lincoln)
        # Arkansas
        "05119": (34.7465, -92.2896),   # Pulaski (Little Rock)
        "05007": (36.3484, -94.2088),   # Benton (Fayetteville)
        # Mississippi
        "28049": (32.3254, -90.1820),   # Hinds (Jackson)
        "28033": (34.3688, -88.7037),   # DeSoto
        # Idaho
        "16001": (43.6150, -116.2023),  # Ada (Boise)
        # Montana
        "30049": (47.5024, -111.3008),  # Lewis and Clark (Helena)
        "30111": (45.7833, -108.5007),  # Yellowstone (Billings)
        # Wyoming
        "56021": (41.1400, -104.8202),  # Laramie (Cheyenne)
        # New Hampshire
        "33011": (42.9956, -71.4548),   # Hillsborough (Manchester)
        "33015": (43.2081, -71.5376),   # Merrimack (Concord)
        # Hawaii
        "15003": (21.3069, -157.8583),  # Honolulu
        # Alaska
        "02020": (61.2181, -149.9003),  # Anchorage
        # Rhode Island
        "44007": (41.8240, -71.4128),   # Providence
        # Delaware
        "10003": (39.7391, -75.5398),   # New Castle (Wilmington)
        # DC
        "11001": (38.9072, -77.0369),   # District of Columbia
    }
    return CENTROIDS.get(geofips, (None, None))

