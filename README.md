# P-Pinpoint

> AI-powered business location recommender for the US market.

Describe your company in plain English. Pinpoint parses your requirements using Claude, scores all 50 US states against real economic data, and returns ranked recommendations with explanations — all from a terminal CLI.

---

## What it does

1. **You describe your business** — size, industry, priorities (talent, cost, growth, etc.)
2. **Claude parses your input** — extracts a structured profile via tool use
3. **The recommender scores every state** — using GDP data from the US Bureau of Economic Analysis
4. **You get ranked results** — with the key economic signals driving each recommendation

---

## Quick start

**Prerequisites:** Python 3.10+, an [Anthropic API key](https://console.anthropic.com/)

```bash
# 1. Clone and install
git clone https://github.com/KazumaOkada36/P-Pinpoint
cd P-Pinpoint
pip install -e .

# 2. Run the setup wizard
pinpoint setup
```

The wizard will ask for your API key, let you pick a model, and train the recommender on the bundled dataset.

---

## CLI usage

### Interactive REPL

```bash
pinpoint
```

Just type and talk. Claude handles the rest.

```
> We're a 60-person SaaS startup, Series B. We need strong engineering talent
  and a fast-growing tech ecosystem. Open to anywhere on the West Coast.

  #1 Washington (0.83)   — top information tech GDP share, strong professional services
  #2 California (0.81)   — largest tech ecosystem, highest professional services GDP
  #3 Colorado  (0.74)    — fastest GDP growth rate, growing tech sector

  Would you like to explore any of these in more detail, or adjust your priorities?

> What if cost efficiency matters more?
```

**Slash commands:**

| Command | Description |
|---|---|
| `/model` | Switch between Opus, Sonnet, Haiku |
| `/clear` | Reset the conversation |
| `/train` | Retrain the recommender |
| `/import <path>` | Load your own CSV data |
| `/top <n>` | Change number of results shown |
| `/config` | Show current settings |
| `/help` | List all commands |
| `/exit` | Quit |

---

### One-shot (no REPL)

```bash
# Recommend by priority keywords
pinpoint recommend --priorities talent,tech_ecosystem,gdp_growth

# More options
pinpoint recommend --priorities manufacturing,logistics --top 10

# Available priorities:
# talent, tech_ecosystem, cost_efficiency, gdp_growth,
# manufacturing, finance, healthcare, logistics, energy
```

---

### Other commands

```bash
pinpoint train                        # Retrain the recommender
pinpoint import-data my_data.csv      # Add custom data, then retrain
pinpoint doctor                       # System health check
pinpoint config show                  # View current config
pinpoint config set model claude-opus-4-6
```

---

## Training setup

### Data source

The recommender is trained on **CAGDP9** — the Bureau of Economic Analysis's *Real GDP by County and Metropolitan Area* dataset. The bundled data covers all 50 US states + DC from 2001 to 2024, split into one CSV per state:

```
Data_Set/clean_data/
  CAGDP9_CA_2001_2024.csv
  CAGDP9_TX_2001_2024.csv
  ... (51 files total)
```

Each file contains GDP broken down by industry sector (NAICS codes), in thousands of chained 2017 dollars.

### Feature engineering

For each state, the following features are extracted from the 2024 data:

| Feature | Description |
|---|---|
| `information_tech` | GDP share: information sector |
| `professional_services` | GDP share: professional & technical services |
| `finance` | GDP share: finance & insurance |
| `manufacturing` | GDP share: manufacturing |
| `healthcare` | GDP share: health care & social assistance |
| `transportation_logistics` | GDP share: transportation & warehousing |
| `mining_energy` | GDP share: mining, quarrying, oil & gas |
| `education` | GDP share: educational services |
| `construction` | GDP share: construction |
| `retail` | GDP share: retail trade |
| `gdp_growth_rate` | Annualized GDP growth, 2020–2024 |
| `total_gdp_inv` | Inverse of total GDP (proxy for lower cost / smaller market) |
| + 11 more | Other NAICS sectors |

All features are normalized to [0, 1] with MinMaxScaler before scoring.

### How recommendations are scored

Each user priority maps to a weighted combination of features:

| Priority | Feature weights |
|---|---|
| `talent` | information_tech (35%), professional_services (30%), education (20%), management (15%) |
| `tech_ecosystem` | information_tech (50%), professional_services (30%), management (20%) |
| `gdp_growth` | gdp_growth_rate (100%) |
| `cost_efficiency` | total_gdp_inv (60%), hospitality_inv (40%) |
| `manufacturing` | manufacturing (60%), wholesale (20%), logistics (20%) |
| `finance` | finance (70%), real_estate (20%), management (10%) |
| `healthcare` | healthcare (70%), education (20%), professional_services (10%) |
| `logistics` | transportation_logistics (50%), wholesale (30%), manufacturing (20%) |
| `energy` | mining_energy (60%), utilities (40%) |

When multiple priorities are given, their weight vectors are averaged. States are then ranked by **cosine similarity** between the query vector and their normalized feature vectors.

### Retraining on your own data

You can supplement the BEA data with any custom CSV that has a `state` column (2-letter code) and numeric feature columns:

```csv
state,talent_score,infrastructure_score,cost_index
CA,0.92,0.88,0.25
TX,0.78,0.82,0.61
NY,0.85,0.79,0.20
...
```

```bash
pinpoint import-data my_scores.csv
pinpoint train
```

Custom features are merged into the existing feature matrix before fitting. The model is saved to `~/.pinpoint/recommender.pkl`.

---

## Configuration

Config is stored at `~/.pinpoint/config.json`. You can also use environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # API key
```

| Key | Default | Description |
|---|---|---|
| `api_key` | — | Anthropic API key |
| `model` | `claude-sonnet-4-6` | Claude model to use |
| `top_n` | `5` | Default number of results |
| `data_dir` | bundled | Path to CAGDP9 CSV directory |

---

## Project structure

```
P-Pinpoint/
├── src/pinpoint/
│   ├── cli.py                  # Typer CLI — all commands
│   ├── agent/
│   │   ├── config.py           # Config file management
│   │   ├── runner.py           # Claude API + tool-use loop
│   │   └── system_prompt.py    # Claude's instructions
│   ├── ui/
│   │   ├── terminal.py         # Interactive REPL (prompt_toolkit)
│   │   └── status.py           # Animated thinking spinner
│   └── recommender/
│       ├── features.py         # CSV -> feature matrix, priority weights
│       ├── model.py            # Cosine similarity ranker (scikit-learn)
│       └── trainer.py          # Train, save, load model
├── Data_Set/
│   ├── clean_data/             # 51 CAGDP9 CSV files (BEA, 2001-2024)
│   └── converter.py            # Multi-format data converter
├── main.py                     # Flask web backend (REST API)
├── p-pinpoint.html             # Standalone frontend (Leaflet map)
├── pyproject.toml              # Python package config
└── CLI_USAGE.md                # Quick command reference
```

---

## Web interface

The project also includes a standalone web frontend (`p-pinpoint.html`) backed by a Flask server (`main.py`). To run it:

```bash
pip install flask flask-cors openai python-dotenv
python main.py
# Then open p-pinpoint.html in your browser
```

The web interface uses OpenAI GPT-4o for parsing (requires `OPENAI_API_KEY` in `secrets.env`). The CLI uses Claude.
