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


## Example

```
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{"forecast_length_days": 365, "lock_target": 0.3}'
```

Then open: http://localhost:8000/latest-simulation/plots/html


## Troubleshooting

- Missing auth: ensure `.env` contains `SPACESCOPE_TOKEN` or `SPACESCOPE_AUTH_FILE` and restart.
- First run is slow: initial data fetch can take minutes.
- Length mismatch: delete `mechafil-server/data/historical_data.pkl` and `historical_data_meta.json`, then restart.
- NaN/Inf in JSON: the server sanitizes them to `null`; gaps may appear in plots.
- Matplotlib missing: `pip install matplotlib`.


## Security

- Do not commit real tokens. `.gitignore` excludes `.env` and data caches.
- Restrict CORS for production.
