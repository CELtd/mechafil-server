# MechaFil Server

FastAPI service to run Filecoin economic forecasts using historical data and return results via HTTP.


## Prerequisites

- Python 3.10+
- pip
- JAX (CPU example): `pip install -U "jax[cpu]"` (see JAX docs for GPU wheels)


## Install

Using Poetry (recommended):

```
cd mechafil-server
poetry install
```

This installs the server and all dependencies declared in `pyproject.toml` (including FastAPI, Uvicorn, JAX, matplotlib, mechafil-jax, and pystarboard).


## Configure Data Access (Spacescope)

The server fetches historical data at startup via Spacescope/Starboard (through `pystarboard`). You need a Spacescope API token.

Set credentials via environment variables (the server loads `.env` from the repo root or from this folder, and also `.test-env` from the repo root):

- `SPACESCOPE_TOKEN` — bearer token string, e.g. `Bearer YOUR_TOKEN_HERE`
- or `SPACESCOPE_AUTH_FILE` — path to a JSON file with `{ "auth_key": "Bearer YOUR_TOKEN_HERE" }`

Examples:

```
# .env at repo root
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
# or
SPACESCOPE_AUTH_FILE=./auths/spacescope_auth.json

# mechafil-server/.env
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
```


## Historical Data

- On first startup, the server fetches and caches historical data under `mechafil-server/data/` (this may take a few minutes).
- `current_date` is set to “yesterday.” Historical arrays are harmonized to a consistent length to avoid broadcasting issues.
- Plots always start at 2020‑10‑15; forecast‑only series (e.g., scheduled expirations) are aligned to `current_date`.


## Run

From the `mechafil-server` folder:

```
# Start with Poetry
poetry run mechafil-server

# Or run Uvicorn explicitly
poetry run uvicorn mechafil_server.main:app --reload --host 0.0.0.0 --port 8000
```

Docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


## API

Historical Data
- `GET /historical-data` — Summary of loaded historical data.
- `GET /historical-data/full` — Full historical arrays and smoothed values.
- `GET /sim-dates` — Effective dates and historical lengths.

Simulation
- `POST /simulate` — Run a forecast (optional body: `rbp`, `rr`, `fpr`, `lock_target`, `forecast_length_days`, `sector_duration_days`).
- `GET /latest-simulation` — Last `{ setup, results }` JSON.
- `GET /latest-simulation/plots` — Base64 PNG plots (JSON; param: `max_plots`).
- `GET /latest-simulation/plots/html` — HTML plots with historical (blue) vs forecast (red).


## Examples

### Simulation Parameters

The `/simulate` endpoint accepts these optional parameters:
- `rbp`: Raw byte power onboarding (PIB/day) - float or array
- `rr`: Renewal rate (0..1) - float or array  
- `fpr`: FIL+ rate (0..1) - float or array
- `lock_target`: Target lock ratio - float or array
- `forecast_length_days`: Forecast length in days - integer
- `sector_duration_days`: Average sector duration in days - integer

All parameters are optional. Defaults are calculated from historical data or configuration.

### Basic Examples

**Minimal request (all defaults):**
```bash
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{}'
```

**1-year forecast:**
```bash
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{"forecast_length_days": 365}'
```

**Complete parameter set:**
```bash
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "rbp": 3.38,
    "rr": 0.83,
    "fpr": 0.86,
    "lock_target": 0.3,
    "forecast_length_days": 365,
    "sector_duration_days": 540
  }'
```

### Advanced Examples

**Long-term forecast (10 years):**
```bash
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "forecast_length_days": 3650,
    "lock_target": 0.25
  }'
```

**Time-varying parameters using arrays:**
```bash
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "rbp": [3.0, 3.5, 4.0],
    "rr": [0.8, 0.85, 0.9],
    "fpr": [0.8, 0.85, 0.9],
    "forecast_length_days": 3
  }'
```

**Different sector duration scenarios:**
```bash
# Shorter sectors (1 year)
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "sector_duration_days": 365,
    "forecast_length_days": 1000
  }'

# Longer sectors (2 years)
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "sector_duration_days": 730,
    "forecast_length_days": 1000
  }'
```

**Scenario analysis:**
```bash
# Conservative scenario
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "rbp": 2.0,
    "rr": 0.7,
    "fpr": 0.5,
    "lock_target": 0.4
  }'

# Aggressive growth scenario  
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "rbp": 5.0,
    "rr": 0.95,
    "fpr": 0.95,
    "lock_target": 0.2
  }'
```


## Troubleshooting

- Missing auth: ensure `.env` contains `SPACESCOPE_TOKEN` or `SPACESCOPE_AUTH_FILE` and restart.
- First run is slow: initial data fetch can take minutes.

## Security

- Do not commit real tokens. `.gitignore` excludes `.env` and data caches.
- Restrict CORS for production.
