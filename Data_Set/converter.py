"""
Dataset to CSV Converter
Supports: JSON, TSV, Excel (.xlsx/.xls), XML, SQLite (.db/.sqlite), Parquet
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Readers – each returns a list of dicts (rows)
# ---------------------------------------------------------------------------

def read_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Support top-level list or a dict that wraps a list
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
        # Single-object JSON → wrap in list
        return [data]

    raise ValueError(f"Unsupported JSON structure in {path}")


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader]


def read_excel(path: Path) -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl is required for Excel files. Run: pip install openpyxl")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    result = []
    for row in rows[1:]:
        result.append({headers[i]: (val if val is not None else "") for i, val in enumerate(row)})
    return result


def read_xml(path: Path) -> list[dict]:
    tree = ET.parse(path)
    root = tree.getroot()

    # The children of the root are treated as rows; their sub-elements as columns
    children = list(root)
    if not children:
        raise ValueError("XML root has no child elements to convert.")

    records = []
    for child in children:
        row: dict = {}
        # Attributes of the child element
        row.update(child.attrib)
        # Sub-elements
        for elem in child:
            row[elem.tag] = elem.text or ""
        records.append(row)
    return records


def read_sqlite(path: Path, table: str | None) -> list[dict]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if table is None:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        if not tables:
            raise ValueError("No tables found in the SQLite database.")
        if len(tables) > 1:
            sys.exit(
                f"Multiple tables found: {tables}\n"
                "Use --table <name> to specify which one to convert."
            )
        table = tables[0]

    cur.execute(f"SELECT * FROM [{table}]")  # noqa: S608
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def read_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required for Parquet files. Run: pip install pandas pyarrow")
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def read_csv_input(path: Path) -> list[dict]:
    """Pass-through reader for already-CSV files (useful for re-formatting)."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], out_path: Path, delimiter: str = ",") -> None:
    if not records:
        print("Warning: no records to write – output file will be empty.")
        out_path.write_text("")
        return

    # Collect all unique keys preserving insertion order
    fieldnames: list[str] = list(dict.fromkeys(k for row in records for k in row))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"Converted {len(records)} records → {out_path}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    ".json": "json",
    ".tsv":  "tsv",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".xml":  "xml",
    ".db":   "sqlite",
    ".sqlite": "sqlite",
    ".parquet": "parquet",
    ".csv":  "csv",
}


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    fmt = EXTENSION_MAP.get(ext)
    if fmt is None:
        sys.exit(
            f"Cannot detect format for extension '{ext}'.\n"
            f"Supported: {', '.join(EXTENSION_MAP)}"
        )
    return fmt


def load_file(path: Path, fmt: str, table: str | None) -> list[dict]:
    loaders = {
        "json":    lambda: read_json(path),
        "tsv":     lambda: read_tsv(path),
        "excel":   lambda: read_excel(path),
        "xml":     lambda: read_xml(path),
        "sqlite":  lambda: read_sqlite(path, table),
        "parquet": lambda: read_parquet(path),
        "csv":     lambda: read_csv_input(path),
    }
    loader = loaders.get(fmt)
    if loader is None:
        sys.exit(f"Unsupported format: {fmt}")
    return loader()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="converter",
        description="Convert common dataset formats to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported input formats:
  .json        JSON (array of objects, or object wrapping an array)
  .tsv         Tab-separated values
  .xlsx/.xls   Excel workbook (active sheet)
  .xml         XML (root's direct children become rows)
  .db/.sqlite  SQLite database table
  .parquet     Apache Parquet (requires pandas + pyarrow)
  .csv         CSV re-format / delimiter change

Examples:
  python converter.py data.json
  python converter.py data.tsv -o output.csv
  python converter.py data.xlsx --delimiter ";"
  python converter.py data.db --table users
  python converter.py data.xml -o result.csv
        """,
    )
    parser.add_argument("input", type=Path, help="Path to the input dataset file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV path (default: same name as input with .csv extension).",
    )
    parser.add_argument(
        "-f", "--format", choices=list(set(EXTENSION_MAP.values())), default=None,
        help="Force input format (auto-detected from extension by default).",
    )
    parser.add_argument(
        "--table", default=None,
        help="SQLite table name to export (required when DB has multiple tables).",
    )
    parser.add_argument(
        "--delimiter", default=",",
        help="Output CSV delimiter (default: comma).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    fmt = args.format or detect_format(input_path)

    output_path: Path = args.output or input_path.with_suffix(".csv")
    if output_path.resolve() == input_path.resolve():
        output_path = input_path.with_name(input_path.stem + "_converted.csv")

    print(f"Reading  : {input_path}  [{fmt}]")
    records = load_file(input_path, fmt, args.table)
    print(f"Records  : {len(records)}")

    write_csv(records, output_path, delimiter=args.delimiter)


if __name__ == "__main__":
    main()
