"""
Loads Census County Business Patterns (CBP) data.

Source: Census Bureau CBP 2023 — County File
File:   cbp23co.txt.gz  (already in clean_data/)

Key columns:
  fipstate  2-digit state FIPS (zero-padded string)
  fipscty   3-digit county FIPS (zero-padded string)
  naics     6-char NAICS code padded with dashes/slashes
              "------" = all industries total
              "72----" = 2-digit NAICS 72 (accommodation & food)
              "721///" = 3-digit, etc.
  est       establishment count (may be flagged: N, G, J, … → 0)

Returns dict keyed by 5-digit county FIPS:
  {
    "total_establishments": int,
    "industry_establishments": {linecode: int},   # BEA linecodes
  }
"""

import re
import gzip
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "Data_Set" / "clean_data"

# 2-digit NAICS prefix → BEA LineCode (same mapping as LODES where possible)
_NAICS2_TO_LC: dict[str, int] = {
    "11": 3,    # Agriculture
    "21": 6,    # Mining
    "22": 10,   # Utilities
    "23": 11,   # Construction
    "31": 12,   "32": 12, "33": 12,   # Manufacturing
    "42": 34,   # Wholesale
    "44": 35,   "45": 35,             # Retail
    "48": 36,   "49": 36,             # Transportation
    "51": 45,   # Information
    "52": 51,   # Finance & insurance
    "53": 56,   # Real estate
    "54": 60,   # Professional/sci/tech
    "56": 59,   # Admin & support (lc 59 approx)
    "61": 69,   # Educational
    "62": 70,   # Health care
    "71": 76,   # Arts & entertainment
    "72": 79,   # Accommodation & food
    "81": 82,   # Other services
    "92": 83,   # Government
}

_TOTAL_CODE   = "------"
_TWO_DIGIT_RE = re.compile(r"^\d{2}----$")


def _parse_est(val) -> int:
    """Return int establishment count; non-numeric flags → 0."""
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return 0


@lru_cache(maxsize=1)
def load_cbp() -> dict:
    """Load CBP county file, return dict keyed by 5-digit county FIPS."""
    # Support both compressed and raw versions
    gz_path  = DATA_DIR / "cbp23co.txt.gz"
    raw_path = DATA_DIR / "cbp23co.txt"

    if gz_path.exists():
        path, open_fn = gz_path, gzip.open
    elif raw_path.exists():
        path, open_fn = raw_path, open
    else:
        return {}

    with open_fn(path, "rt", newline="") as fh:
        df = pd.read_csv(fh, dtype=str, low_memory=False)

    # Build 5-digit FIPS
    df["fips"] = df["fipstate"].str.strip().str.zfill(2) + df["fipscty"].str.strip().str.zfill(3)
    df["est_int"] = df["est"].apply(_parse_est)
    df["naics"]   = df["naics"].str.strip()

    total_mask    = df["naics"] == _TOTAL_CODE
    two_digit_mask = df["naics"].str.match(r"^\d{2}----$")

    # ── Aggregate total establishments ───────────────────────────────────────
    total_agg = (
        df[total_mask]
        .groupby("fips")["est_int"]
        .sum()
        .to_dict()
    )

    # ── Aggregate by 2-digit NAICS → map to BEA linecode ─────────────────────
    two_digit = df[two_digit_mask].copy()
    two_digit["naics2"] = two_digit["naics"].str[:2]
    two_digit["lc"]     = two_digit["naics2"].map(_NAICS2_TO_LC)
    two_digit = two_digit.dropna(subset=["lc"])
    two_digit["lc"] = two_digit["lc"].astype(int)

    ind_agg = (
        two_digit
        .groupby(["fips", "lc"])["est_int"]
        .sum()
        .reset_index()
    )

    # Build per-county dict
    ind_by_fips: dict[str, dict[int, int]] = defaultdict(dict)
    for _, row in ind_agg.iterrows():
        lc  = int(row["lc"])
        est = int(row["est_int"])
        if est > 0:
            ind_by_fips[row["fips"]][lc] = ind_by_fips[row["fips"]].get(lc, 0) + est

    all_fips = set(total_agg) | set(ind_by_fips)
    return {
        fips: {
            "total_establishments":    total_agg.get(fips, 0),
            "industry_establishments": ind_by_fips.get(fips, {}),
        }
        for fips in all_fips
        if total_agg.get(fips, 0) > 0
    }
