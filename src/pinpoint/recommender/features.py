"""Feature engineering — turns CAGDP9 CSVs into a state feature matrix."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CAGDP9 industry line-code → semantic feature name                           #
# --------------------------------------------------------------------------- #

# Maps LineCode → feature name for industry GDP share features
INDUSTRY_FEATURES: dict[int, str] = {
    1: "total_gdp",
    3: "agriculture",
    6: "mining_energy",
    10: "utilities",
    11: "construction",
    12: "manufacturing",
    34: "wholesale",
    35: "retail",
    36: "transportation_logistics",
    45: "information_tech",
    50: "finance",
    54: "real_estate",
    59: "professional_services",
    69: "management",
    70: "admin_services",
    76: "education",
    77: "healthcare",
    82: "arts_entertainment",
    83: "hospitality",
    88: "government",
}

# Priority keyword → feature weights (must sum to 1.0)
PRIORITY_WEIGHTS: dict[str, dict[str, float]] = {
    "talent": {
        "information_tech": 0.35,
        "professional_services": 0.30,
        "education": 0.20,
        "management": 0.15,
    },
    "tech_ecosystem": {
        "information_tech": 0.50,
        "professional_services": 0.30,
        "management": 0.20,
    },
    "cost_efficiency": {
        "total_gdp_inv": 0.60,      # inverse = cheaper/smaller economy → lower cost
        "hospitality_inv": 0.40,    # inverse proxy for cost of living
    },
    "gdp_growth": {
        "gdp_growth_rate": 1.0,
    },
    "manufacturing": {
        "manufacturing": 0.60,
        "wholesale": 0.20,
        "transportation_logistics": 0.20,
    },
    "finance": {
        "finance": 0.70,
        "real_estate": 0.20,
        "management": 0.10,
    },
    "healthcare": {
        "healthcare": 0.70,
        "education": 0.20,
        "professional_services": 0.10,
    },
    "logistics": {
        "transportation_logistics": 0.50,
        "wholesale": 0.30,
        "manufacturing": 0.20,
    },
    "energy": {
        "mining_energy": 0.60,
        "utilities": 0.40,
    },
}

# US state code → full name
STATE_NAMES: dict[str, str] = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts",
    "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
    "MO": "Missouri", "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina",
    "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington",
    "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming",
}


def _state_code_from_path(path: Path) -> Optional[str]:
    """Extract 2-letter state code from filename like CAGDP9_CA_2001_2024.csv."""
    m = re.search(r"CAGDP9_([A-Z]{2})_", path.name)
    return m.group(1) if m else None


def _state_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return the state-level row (GeoFIPS ends in '000', e.g. '06000')."""
    fips = df["GeoFIPS"].astype(str).str.strip().str.replace('"', "")
    return df[fips.str.endswith("000")]


def _gdp_value(df: pd.DataFrame, line_code: int, year: str = "2024") -> float:
    """Extract a single numeric GDP value for a given line code and year."""
    row = df[df["LineCode"] == line_code]
    if row.empty or year not in row.columns:
        return 0.0
    val = row[year].iloc[0]
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def build_feature_matrix(
    data_dir: Path,
    custom_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load all CAGDP9 CSVs and return a DataFrame where:
      - index = state abbreviation
      - columns = feature names (industry shares, growth rate, etc.)
    """
    csv_files = sorted(data_dir.glob("CAGDP9_*.csv"))
    # Skip US-aggregate file
    csv_files = [f for f in csv_files if "_US_" not in f.name]

    rows: list[dict] = []

    for csv_path in csv_files:
        code = _state_code_from_path(csv_path)
        if not code or code == "US":
            continue

        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception:
            continue

        df.columns = [c.strip() for c in df.columns]

        # Make sure LineCode is numeric
        df["LineCode"] = pd.to_numeric(df["LineCode"], errors="coerce")

        state_df = _state_row(df)
        if state_df.empty:
            # Fall back to first row per LineCode
            state_df = df

        row: dict = {"state": code, "state_name": STATE_NAMES.get(code, code)}

        # Extract raw GDP by industry (2024)
        total_gdp = _gdp_value(state_df, 1, "2024")
        row["total_gdp"] = total_gdp

        for line_code, feat_name in INDUSTRY_FEATURES.items():
            if line_code == 1:
                continue
            val = _gdp_value(state_df, line_code, "2024")
            # Store as share of total GDP (avoids size bias)
            row[feat_name] = val / total_gdp if total_gdp > 0 else 0.0

        # GDP growth rate 2020 → 2024 (annualised)
        gdp_2020 = _gdp_value(state_df, 1, "2020")
        gdp_2024 = _gdp_value(state_df, 1, "2024")
        if gdp_2020 > 0 and gdp_2024 > 0:
            row["gdp_growth_rate"] = (gdp_2024 / gdp_2020) ** (1 / 4) - 1
        else:
            row["gdp_growth_rate"] = 0.0

        rows.append(row)

    # Load any custom data CSVs and merge
    if custom_data_dir and custom_data_dir.exists():
        for custom_path in custom_data_dir.glob("*.csv"):
            try:
                cdf = pd.read_csv(custom_path, dtype=str)
                # Expect columns: state, <feature_name>, ...
                if "state" in cdf.columns:
                    for _, crow in cdf.iterrows():
                        # Merge into existing row or append
                        existing = next((r for r in rows if r["state"] == crow["state"]), None)
                        if existing:
                            for col in cdf.columns:
                                if col != "state":
                                    try:
                                        existing[col] = float(crow[col])
                                    except (ValueError, TypeError):
                                        pass
            except Exception:
                pass

    feature_df = pd.DataFrame(rows).set_index("state")

    # Add inverse features for cost proxies
    if "total_gdp" in feature_df.columns:
        max_gdp = feature_df["total_gdp"].max()
        feature_df["total_gdp_inv"] = 1 - (feature_df["total_gdp"] / max_gdp).clip(0, 1)

    if "hospitality" in feature_df.columns:
        max_hosp = feature_df["hospitality"].max()
        feature_df["hospitality_inv"] = 1 - (feature_df["hospitality"] / max_hosp).clip(0, 1)

    return feature_df


def build_query_vector(
    priorities: list[str],
    feature_columns: list[str],
) -> np.ndarray:
    """
    Convert a list of priority strings into a weighted feature vector
    aligned with feature_columns.
    """
    weights = np.zeros(len(feature_columns))
    col_idx = {c: i for i, c in enumerate(feature_columns)}

    n_priorities = len(priorities) if priorities else 1
    per_priority_weight = 1.0 / n_priorities

    for priority in priorities:
        pw = PRIORITY_WEIGHTS.get(priority, {})
        if not pw:
            continue
        for feat, w in pw.items():
            if feat in col_idx:
                weights[col_idx[feat]] += w * per_priority_weight

    # Fallback: if no weights set, use total_gdp_growth as default
    if weights.sum() == 0 and "gdp_growth_rate" in col_idx:
        weights[col_idx["gdp_growth_rate"]] = 1.0

    return weights
