"""
Loads BLS OEWS (Occupational Employment & Wage Statistics) data.

Source: BLS OEWS 2024 — Nonmetropolitan Areas
File:   BOS_M2024_dl.xlsx

This file covers non-metropolitan areas only (AREA_TYPE=6). We aggregate
by PRIM_STATE + occupation major group to get state-level wage baselines.
Used as a talent cost signal — lower wages for a relevant occupation in a
state = more affordable to hire the workforce your business needs.

Returns dict keyed by 2-letter state abbreviation:
  {
    "13": {"median_hourly": float, "total_emp": int},   # Business & financial
    "15": {"median_hourly": float, "total_emp": int},   # Computer & math
    "25": {"median_hourly": float, "total_emp": int},   # Education
    "29": {"median_hourly": float, "total_emp": int},   # Healthcare practitioners
    "35": {"median_hourly": float, "total_emp": int},   # Food preparation
    "39": {"median_hourly": float, "total_emp": int},   # Personal care
    "41": {"median_hourly": float, "total_emp": int},   # Sales
    "51": {"median_hourly": float, "total_emp": int},   # Production
    ...
  }
"""

from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent.parent / "Data_Set" / "clean_data"

# OCC_CODE major-group prefixes we care about (one per business type + extras)
RELEVANT_OCC_GROUPS = {
    "11",   # Management
    "13",   # Business & financial operations
    "15",   # Computer & mathematical
    "19",   # Life/physical/social science
    "25",   # Education, training & library
    "29",   # Healthcare practitioners & technical
    "31",   # Healthcare support
    "35",   # Food preparation & serving
    "39",   # Personal care & service
    "41",   # Sales & related
    "47",   # Construction & extraction
    "51",   # Production
    "53",   # Transportation & material moving
}


def _to_float(val) -> float | None:
    try:
        s = str(val).strip()
        if s in ("*", "#", "**", "", "nan", "None"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _to_int(val) -> int | None:
    try:
        s = str(val).strip().replace(",", "")
        if s in ("*", "#", "", "nan", "None"):
            return None
        return int(float(s))
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=1)
def load_oews() -> dict:
    """
    Return state → occupation group → wage summary dict.
    If the file is missing or openpyxl is not installed, returns {}.
    """
    oews_path = DATA_DIR / "BOS_M2024_dl.xlsx"
    if not oews_path.exists():
        return {}

    try:
        import pandas as pd
    except ImportError:
        return {}

    df = pd.read_excel(oews_path, dtype=str)

    # Keep only major-group occupation rows (O_GROUP == "major")
    df = df[df["O_GROUP"] == "major"].copy()

    # Extract 2-digit OCC prefix
    df["occ_group"] = df["OCC_CODE"].str[:2]
    df = df[df["occ_group"].isin(RELEVANT_OCC_GROUPS)]

    result: dict = {}
    for (state, occ_group), grp in df.groupby(["PRIM_STATE", "occ_group"]):
        state     = str(state).strip()
        occ_group = str(occ_group).strip()

        wages = [w for row in grp["H_MEDIAN"] if (w := _to_float(row)) is not None]
        emps  = [e for row in grp["TOT_EMP"]  if (e := _to_int(row))   is not None]

        if not wages:
            continue

        # Employment-weighted average where possible; simple average otherwise
        if emps and len(emps) == len(wages):
            total_emp = sum(emps)
            avg_wage  = sum(w * e for w, e in zip(wages, emps)) / total_emp if total_emp > 0 else sum(wages) / len(wages)
        else:
            avg_wage  = sum(wages) / len(wages)
            total_emp = sum(emps) if emps else 0

        result.setdefault(state, {})[occ_group] = {
            "median_hourly": round(avg_wage, 2),
            "total_emp":     total_emp,
        }

    return result
