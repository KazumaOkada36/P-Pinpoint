# Pinpoint CLI

AI-powered business location recommender. Plug in your Claude API key and describe your business — Pinpoint scores all 50 US states and explains why each one fits.

## Install

```bash
pip install -e .
```

## First-time setup

```bash
pinpoint setup
```

Walks you through:
1. Anthropic API key
2. Default model (Opus / Sonnet / Haiku)
3. Training the recommender on bundled US economic data

## Usage

### Interactive REPL (recommended)

```bash
pinpoint
```

Describe your business in plain English. Claude parses your requirements, scores all states, and explains the top picks. Follow up with questions or refine your priorities.

```
> We're a 40-person fintech startup, Series A, looking for a state with strong
  financial infrastructure and available engineering talent.

  #1 New York (0.81)  — dominant finance GDP, high professional services
  #2 Massachusetts (0.76) — strong finance + tech overlap
  ...

> What about cost? We're budget-conscious.
```

**Slash commands inside the REPL:**

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/model` | Switch Claude model |
| `/clear` | Reset conversation |
| `/train` | Retrain the recommender |
| `/import <path>` | Load custom CSV data |
| `/top <n>` | Change number of results |
| `/exit` | Quit |

---

### One-shot recommendation (no REPL)

```bash
pinpoint recommend --priorities talent,tech_ecosystem,gdp_growth
pinpoint recommend --priorities manufacturing,logistics --top 10
```

Available priorities: `talent`, `tech_ecosystem`, `cost_efficiency`, `gdp_growth`, `manufacturing`, `finance`, `healthcare`, `logistics`, `energy`

---

### Retrain on your own data

```bash
# Import a custom CSV (must have a 'state' column + numeric feature columns)
pinpoint import-data my_scores.csv

# Retrain to apply it
pinpoint train
```

**Custom CSV format:**
```
state,talent_score,infrastructure_score,cost_index
CA,0.9,0.85,0.3
TX,0.75,0.8,0.6
...
```

---

### Other commands

```bash
pinpoint doctor          # Check API key, model, recommender, and data status
pinpoint config show     # View current config
pinpoint config set model claude-opus-4-6
```

## Config

Stored at `~/.pinpoint/config.json`. You can also set your API key via environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
