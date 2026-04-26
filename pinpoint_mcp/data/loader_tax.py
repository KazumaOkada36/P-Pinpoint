"""
Loads Tax Foundation 2024 state corporate income tax rate data.

Source: Tax Foundation — 2024 State Corporate Income Tax Rates & Brackets
File:   2024 State Corporate Income Tax Rates  Brackets.csv

States have one or more rows (graduated brackets). We take the top
marginal rate (max across all bracket rows for that state).

Returns dict keyed by 2-letter state abbreviation:
  {
    "top_rate": float,   # e.g. 0.065 for 6.5%
    "has_tax":  bool,
  }
"""

import csv
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent.parent / "Data_Set" / "clean_data"

_STATE_TO_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC", "Washington, D.C.": "DC",
}

# States with no traditional corporate income tax → automatic top score
_NO_INCOME_TAX_STATES = {"NV", "SD", "TX", "WY", "WA", "OH"}


def _parse_rate(raw: str) -> float:
    s = raw.strip().lower().replace("%", "").strip()
    if not s or s in ("none", "n/a", "-", "–", "no tax"):
        return 0.0
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


@lru_cache(maxsize=1)
def load_tax() -> dict:
    """Return dict keyed by state abbreviation with top marginal corporate rate."""
    tax_path = DATA_DIR / "2024 State Corporate Income Tax Rates  Brackets.csv"
    if not tax_path.exists():
        return {}

    # Collect all rate rows per state, then take the max (top marginal)
    state_rates: dict[str, list[float]] = {}

    import re
    _footnote_re = re.compile(r"\s*\([a-z]\)\s*$", re.IGNORECASE)

    with open(tax_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_name = row.get("State", "").strip()
            rate_str = row.get("Rates", "").strip()
            if not raw_name:
                continue
            # Strip trailing footnotes like "(a)", "(b)", etc.
            state_name = _footnote_re.sub("", raw_name).strip()
            abbr = _STATE_TO_ABBR.get(state_name)
            if not abbr:
                continue
            rate = _parse_rate(rate_str)
            state_rates.setdefault(abbr, []).append(rate)

    result = {
        abbr: {
            "top_rate": max(rates),
            "has_tax":  max(rates) > 0.0,
        }
        for abbr, rates in state_rates.items()
    }

    # States with no corporate income tax get a perfect score
    for abbr in _NO_INCOME_TAX_STATES:
        if abbr not in result:
            result[abbr] = {"top_rate": 0.0, "has_tax": False}

    return result
