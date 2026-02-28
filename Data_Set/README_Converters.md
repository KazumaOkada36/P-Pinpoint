# P-Pinpoint – Dataset to CSV Converter

Convert common dataset file formats into CSV with a single command.

## Supported Input Formats

| Format | Extensions |
|--------|-----------|
| JSON | `.json` |
| Tab-separated values | `.tsv` |
| Excel | `.xlsx`, `.xls` |
| XML | `.xml` |
| SQLite database | `.db`, `.sqlite` |
| Apache Parquet | `.parquet` |
| CSV (re-format) | `.csv` |

## Setup

```bash
pip install -r requirements.txt
```

> **Minimum Python version:** 3.10+

## Usage

```
python converter.py <input_file> [options]
```

### Options

| Flag | Description |
|------|-------------|
| `-o`, `--output` | Output CSV path (default: same name, `.csv` extension) |
| `-f`, `--format` | Force input format (auto-detected from extension) |
| `--table` | SQLite table name (required when DB has multiple tables) |
| `--delimiter` | Output CSV delimiter character (default: `,`) |

## Examples

```bash
# JSON → CSV
python converter.py data.json

# TSV → CSV with custom output path
python converter.py data.tsv -o output/result.csv

# Excel → semicolon-delimited CSV
python converter.py report.xlsx --delimiter ";"

# SQLite table → CSV
python converter.py database.db --table users

# XML → CSV
python converter.py data.xml -o data.csv

# Parquet → CSV
python converter.py dataset.parquet
```

## JSON Structure Support

The converter handles three common JSON layouts:

```json
// 1. Top-level array
[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

// 2. Object wrapping an array
{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

// 3. Single object (wrapped in a one-row CSV)
{"id": 1, "name": "Alice"}
```

## XML Structure

The converter treats the **direct children** of the root element as rows,
and their sub-elements (and attributes) as columns.

```xml
<people>
  <person id="1"><name>Alice</name><age>30</age></person>
  <person id="2"><name>Bob</name><age>25</age></person>
</people>
```
