"""
Loads and aggregates LODES WAC (Workplace Area Characteristics) data to county level.

Source: US Census LEHD LODES 8 — Workplace Area Characteristics
Files:  *_wac_S000_JT00_2021.csv.gz (one per state, already in clean_data/)

Each row is one census block. We group by the first 5 digits of w_geocode
(county FIPS) and sum all job columns.

Key columns used:
  C000       total jobs at workplace
  CE01/CE02/CE03  jobs by wage tier (low/mid/high)
  CNS01–CNS20     jobs by 2-digit NAICS sector

Returns a dict keyed by 5-digit county FIPS string:
  {
    "total_jobs":       int,
    "high_wage_share":  float,   # CE03 / C000
    "mid_wage_share":   float,
    "low_wage_share":   float,
    "industry_jobs":    {linecode: int},   # mapped to BEA linecodes
    "industry_job_share": {linecode: float},
  }
"""

from pathlib import Path
from functools import lru_cache

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "Data_Set" / "clean_data"

# LODES CNS column → BEA LineCode used throughout the scoring engine
CNS_TO_LC: dict[str, int] = {
    "CNS01": 3,    # Agriculture (NAICS 11)
    "CNS02": 6,    # Mining (NAICS 21)
    "CNS03": 10,   # Utilities (NAICS 22)
    "CNS04": 11,   # Construction (NAICS 23)
    "CNS05": 12,   # Manufacturing (NAICS 31-33)
    "CNS06": 34,   # Wholesale trade (NAICS 42)
    "CNS07": 35,   # Retail trade (NAICS 44-45)
    "CNS08": 36,   # Transportation & warehousing (NAICS 48-49)
    "CNS09": 45,   # Information (NAICS 51)
    "CNS10": 51,   # Finance & insurance (NAICS 52)
    "CNS11": 56,   # Real estate (NAICS 53)
    "CNS12": 60,   # Professional/scientific/technical (NAICS 54)
    "CNS14": 59,   # Admin & support (NAICS 56) — maps to lc 59 (prof & business)
    "CNS15": 69,   # Educational services (NAICS 61)
    "CNS16": 70,   # Health care & social assistance (NAICS 62)
    "CNS17": 76,   # Arts, entertainment & recreation (NAICS 71)
    "CNS18": 79,   # Accommodation & food services (NAICS 72)
    "CNS19": 82,   # Other services (NAICS 81)
    "CNS20": 83,   # Public administration (NAICS 92)
    # CNS13 (Management of companies, NAICS 55) skipped — no clean BEA lc
}

_WAGE_COLS = ["CE01", "CE02", "CE03"]
_USECOLS   = ["w_geocode", "C000"] + _WAGE_COLS + list(CNS_TO_LC.keys())


@lru_cache(maxsize=1)
def load_lodes() -> dict:
    """
    Read all state WAC files, aggregate to county, return feature dict.
    First call takes ~15–30 s; subsequent calls are instant (cached).
    """
    chunks: list[pd.DataFrame] = []

    for gz_path in sorted(DATA_DIR.glob("*_wac_S000_JT00_2021.csv.gz")):
        df = pd.read_csv(
            gz_path,
            usecols=_USECOLS,
            dtype={"w_geocode": str},
        )
        df["county_fips"] = df["w_geocode"].str[:5]
        chunks.append(df.drop(columns=["w_geocode"]))

    if not chunks:
        return {}

    combined = pd.concat(chunks, ignore_index=True)
    agg = combined.groupby("county_fips", sort=False).sum()

    result: dict = {}
    for fips, row in agg.iterrows():
        total = int(row["C000"])
        if total <= 0:
            continue

        # Map CNS columns to BEA linecodes; multiple CNS can share a lc
        industry_jobs: dict[int, int] = {}
        for cns, lc in CNS_TO_LC.items():
            jobs = int(row.get(cns, 0))
            if jobs > 0:
                industry_jobs[lc] = industry_jobs.get(lc, 0) + jobs

        result[fips] = {
            "total_jobs":         total,
            "high_wage_share":    int(row["CE03"]) / total,
            "mid_wage_share":     int(row["CE02"]) / total,
            "low_wage_share":     int(row["CE01"]) / total,
            "industry_jobs":      industry_jobs,
            "industry_job_share": {lc: v / total for lc, v in industry_jobs.items()},
        }

    return result
