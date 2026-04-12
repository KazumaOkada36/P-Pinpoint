# P-Pinpoint — Claude Code Guide

## Project Overview

**P-Pinpoint** is a business location recommendation platform. Users describe their company, and the system uses AI to parse requirements and recommend optimal US cities/regions based on economic data.

## Architecture

```
P-Pinpoint/
├── p-pinpoint.html       # Standalone frontend (React-like UI, Leaflet maps)
├── main.py               # Flask backend (port 5000)
├── openai_parser.py      # GPT-4o structured parsing module
├── Data_Set/
│   ├── clean_data/       # 51 CSV files — CAGDP9 economic data per US state (2001–2024)
│   ├── converter.py      # Multi-format dataset converter
│   └── README_Converters.md
├── package.json          # NPM: leaflet, react-leaflet
├── sample.json           # Test/sample data
└── secrets.env           # (gitignored) OpenAI API key
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Standalone HTML/CSS/JS + Leaflet.js v1.9.4 |
| Backend | Python / Flask + Flask-CORS |
| AI/ML | OpenAI GPT-4o (structured JSON parsing) |
| Data | Pandas, NumPy, US Census CAGDP9 CSVs |
| Config | python-dotenv (`secrets.env`) |

## Backend API

Flask server runs on `localhost:5000` in debug mode.

- **POST `/parse-business`** — Accepts `{ message: string }`, returns structured JSON with `business_type`, `region`, `employee_count`, `valuation_usd`, `monthly_rent_budget_usd`, `user_priorities`, `suggested_priorities`, `follow_up_question`
- **POST `/recommend`** — Accepts user profile, returns ranked location recommendations (currently placeholder)

## Running the Project

```bash
# Backend
pip install flask flask-cors openai python-dotenv pandas numpy
python main.py

# Frontend — open directly in browser
open p-pinpoint.html
```

`secrets.env` must contain `OPENAI_API_KEY=sk-...` for the parser to work.

## Development Status

- [x] Frontend UI (chat sidebar, map, results dashboard)
- [x] Flask backend skeleton
- [x] OpenAI GPT-4o parser with strict JSON schema
- [x] 51-state economic dataset (CAGDP9 2001–2024)
- [ ] ML recommendation model (placeholder only — hardcoded cities)
- [ ] End-to-end integration
- [ ] Deployment

## Key Notes

- The frontend is a **single standalone HTML file** — no build step required.
- The ML recommendation logic in `/recommend` is a **stub** returning hardcoded results (Irvine, Pasadena, Costa Mesa). Real scoring logic needs to be implemented using the CSV data.
- `config.js` and `archive/` are gitignored.
- Data pipeline supports JSON, TSV, Excel, XML, SQLite, Parquet → CSV via `converter.py`.
