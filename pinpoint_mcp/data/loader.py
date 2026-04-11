"""
Loads all BEA CAGDP9 CSV files into a unified, cached data store.

Each CSV covers one state and contains:
- State-level rows (GeoFIPS ending in "000")
- County-level rows (GeoFIPS ending in 001-999)
- 34 industry LineCodes per geography
- Year columns 2001-2024 (values in thousands of chained 2017 dollars)
- "(D)" or other markers for suppressed/unavailable data → treated as NaN
"""

import csv
import io
from pathlib import Path
from functools import lru_cache
from typing import Optional

DATA_DIR = Path(__file__).parent.parent.parent / "Data_Set" / "clean_data"
YEARS = list(range(2001, 2025))
YEAR_STRS = [str(y) for y in YEARS]

# Top-level, non-rollup LineCodes used in scoring
TOP_LEVEL_LINECODES = {1, 3, 6, 10, 11, 12, 34, 35, 36, 45, 50, 59, 68, 75, 82, 83}

# All linecodes we care about (top-level + key sub-categories)
RELEVANT_LINECODES = TOP_LEVEL_LINECODES | {51, 56, 60, 69, 70, 76, 79}


def _parse_value(raw: str) -> Optional[float]:
    """Convert a BEA cell value to float, returning None for suppressed/missing."""
    s = raw.strip()
    if s in ("(D)", "(L)", "(NA)", "(NM)", "(N)", "N/A", "", "nan"):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _is_county(geofips: str) -> bool:
    return not geofips.endswith("000")


def _read_csv_text(csv_path: Path) -> str:
    """
    Read a CSV file with a small encoding fallback chain.

    Some BEA source files include Windows-1252 characters in county names,
    so strict UTF-8 decoding is not always safe.
    """
    raw = csv_path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


@lru_cache(maxsize=1)
def load_raw() -> dict:
    """
    Load all CSVs and return a nested dict:

    raw[geofips] = {
        "geoname":    str,          # "Alameda, CA" or "California"
        "state_abbr": str,          # "CA"
        "county":     str | None,   # "Alameda" (None for state rows)
        "is_county":  bool,
        "industries": {
            linecode (int): {
                "description": str,
                "values": [float | None] * 24   # indexed by YEARS
            }
        }
    }
    """
    raw: dict = {}

    for csv_path in sorted(DATA_DIR.glob("CAGDP9_*.csv")):
        state_abbr = csv_path.stem.split("_")[1]

        with io.StringIO(_read_csv_text(csv_path), newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                geofips_raw = row.get("GeoFIPS", "")
                linecode_raw = row.get("LineCode", "")

                if not geofips_raw or not linecode_raw:
                    continue

                geofips = geofips_raw.strip().strip('"')
                try:
                    linecode = int(linecode_raw.strip())
                except ValueError:
                    continue

                if linecode not in RELEVANT_LINECODES:
                    continue

                geoname = row.get("GeoName", "").strip().strip('"')
                description = row.get("Description", "").strip()

                values = [_parse_value(row.get(y, "")) for y in YEAR_STRS]

                if geofips not in raw:
                    is_county = _is_county(geofips)
                    county_name = None
                    if is_county:
                        # "Alameda, CA" → "Alameda"
                        county_name = geoname.rsplit(", ", 1)[0] if ", " in geoname else geoname

                    raw[geofips] = {
                        "geoname":    geoname,
                        "state_abbr": state_abbr,
                        "county":     county_name,
                        "is_county":  is_county,
                        "industries": {},
                    }

                raw[geofips]["industries"][linecode] = {
                    "description": description,
                    "values":      values,
                }

    return raw


def get_value(geofips: str, linecode: int, year: int) -> Optional[float]:
    """Safe accessor for a single value."""
    raw = load_raw()
    geo = raw.get(geofips)
    if not geo:
        return None
    ind = geo["industries"].get(linecode)
    if not ind:
        return None
    idx = YEARS.index(year) if year in YEARS else None
    if idx is None:
        return None
    return ind["values"][idx]


def get_timeseries(geofips: str, linecode: int) -> list[Optional[float]]:
    """Return the full 2001-2024 value list for a geography × industry."""
    raw = load_raw()
    geo = raw.get(geofips)
    if not geo:
        return [None] * len(YEARS)
    ind = geo["industries"].get(linecode)
    return ind["values"] if ind else [None] * len(YEARS)


def list_counties() -> list[str]:
    """Return all county GeoFIPS codes."""
    return [k for k, v in load_raw().items() if v["is_county"]]


def list_states() -> list[str]:
    """Return all state GeoFIPS codes."""
    return [k for k, v in load_raw().items() if not v["is_county"]]
