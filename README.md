# Measurement Instrument Assistant (Multi-Branch Agent)

AI-powered assistant for exploring measurement instruments with five routed branches:

- **Find Instrument** — BM25 two-pass search + LLM scoring (6–8 results)
- **Instrument Details** — project usage lookup by project number or instrument name
- **Compare** — full-row instrument comparison via LLM
- **Why** — explains previous recommendations from session context
- **Handbook** — general usage guidance fallback

## Setup

```bash
cd "Outcome Repo Agent"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .env
cp .env/env.example .env/.env
```

## Run

```bash
python -m streamlit run frontend/app.py
```

## Data Files

| File | Purpose |
|------|---------|
| `data/measurement_instruments.xlsx` | Main instrument catalogue |
| `data/project_usage.xlsx` | Project → instrument usage records |
| `handbook/user_guidelines.md` | Handbook fallback content |

## Architecture

```
Ask → Router (LLM classify) → Agent branch → Session update → Response
```

See `backend/orchestrator.py` for the central dispatch logic.
