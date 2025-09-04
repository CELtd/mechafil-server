# MechaFil Server and Simulator

A Filecoin economic simulation stack:

- `mechafil-jax/`: High‑performance JAX implementation of the Filecoin economic model (power, minting, vesting, supply, pledge).
- `pystarboard/`: Data connectors for Spacescope/Starboard APIs used to build the historical dataset consumed by the simulator.
- `mechafil-server/`: FastAPI service that loads historical data at startup, runs forecast simulations on POST requests, and exposes endpoints to retrieve results and plots.
- `fip100/`: Scenario tooling, dashboards, and a duplicate of the simulator used in research. The server uses a couple of utility functions from here for historical metric smoothing.


## Prerequisites

- Python 3.10+ (tested on 3.12)
- pip (or Conda/Mamba if preferred)
- JAX. For CPU-only installs: `pip install -U "jax[cpu]"` (see https://github.com/google/jax#installation for GPU/TPU wheels)

Optional (recommended):
- Conda: See `mechafil-jax/setup_env.sh` and `mechafil-jax/environment.yaml` for a ready-to-use environment.


## Install

Create and activate a virtual environment, then install local packages and runtime deps:

```
python -m venv .venv
source .venv/bin/activate

# Install local libraries in editable mode
pip install -e ./pystarboard
pip install -e ./mechafil-jax
pip install -e ./mechafil-server

# Runtime dependencies for the server
pip install fastapi "uvicorn[standard]" python-dotenv matplotlib
# Install JAX (CPU example)
pip install -U "jax[cpu]"
```

If you prefer Conda:

```
cd mechafil-jax
./setup_env.sh
conda activate cel
# Then also install the server runtime deps
pip install -e ../pystarboard -e ../mechafil-jax -e ../mechafil-server
pip install fastapi "uvicorn[standard]" python-dotenv matplotlib
```


## Configure Data Access (Spacescope)

The server fetches historical data at startup via Spacescope endpoints (through `pystarboard`). You need a Spacescope API token.

Set credentials via environment variables (the server loads from several common locations):

- `SPACESCOPE_TOKEN` — bearer token string, e.g. `Bearer YOUR_TOKEN_HERE`
- or `SPACESCOPE_AUTH_FILE` — path to a JSON file with `{ "auth_key": "Bearer YOUR_TOKEN_HERE" }`

You can place `.env` either at the repo root or inside `mechafil-server/`. The server also attempts to load `.test-env` at the repo root.

Examples:

```
# .env at repo root (local development)
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
# or
SPACESCOPE_AUTH_FILE=./auths/spacescope_auth.json

# mechafil-server/.env (alternative location)
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
```

Notes:
- The simulator accepts either a bearer token string or an auth file path. If both are provided, the token takes precedence.
- A sample `.env.example` and `.test-env` are included; copy `.env.example` to `.env` and fill your values.


## How Historical Data Is Built

At server startup, `SimulationRunner`:

- Computes `current_date` as “yesterday” and derives `start_date` from the length of the historical arrays.
- Calls `mechafil_jax.data.get_simulation_data(token_or_file, start_date, current_date, end_date)` which:
  - Pulls power, onboarding, supply, known expirations and pledge schedules via `pystarboard` (Spacescope API).
  - Constructs the `offline_data` dictionary expected by the simulator (initial conditions + historical arrays).
- Uses `fip100/scenario_generator/utils.py` to compute 180‑day history slices and 30‑day medians for three smoothed parameters used as defaults in forecasts: daily onboarded RB power (`rbp`), renewal rate (`rr`), and FIL+ rate (`fpr`).
- Caches the tuple:
  - `(offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, hist_rbp, hist_rr, hist_fpr)` to `mechafil-server/data/historical_data.pkl`
  - Writes `mechafil-server/data/historical_data_meta.json` with the derived dates.
- On subsequent runs, the server loads the cache and re-anchors the dates to “yesterday” to keep the forecast aligned.

Fetching all historical data may take a few minutes on first startup.


## Run the Server

From the repo root (with your venv active):

```
uvicorn mechafil_server.main:app --reload --host 0.0.0.0 --port 8000 --app-dir mechafil-server
```

Open docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


## API Endpoints

Health and Info
- `GET /health` — Returns status and JAX backend.
- `GET /` — Basic info and quick links.

Historical Data
- `GET /historical-data` — Summary of the loaded historical dataset (sizes, keys, a few sample values).
- `GET /historical-data/full` — Full historical arrays and smoothed values (JSON). Arrays are converted to lists.
- `GET /sim-dates` — The effective `start_date`/`current_date` and historical lengths used by the server.

Run a Simulation
- `POST /simulate` — Runs a forecast using the cached historical data (`offline_data`).
  - JSON body (all optional):
    - `rbp`: float or [float] — daily onboarded RB power (PIB/day). Constant or vector of length `forecast_length_days`.
    - `rr`: float or [float] — renewal rate (0..1). Constant or vector.
    - `fpr`: float or [float] — FIL+ rate (0..1). Constant or vector.
    - `lock_target`: float or [float] — target lock ratio (default 0.3).
    - `forecast_length_days`: int — forecast horizon in days (default 3650; max 3650).
    - `sector_duration_days`: int — sector duration (default 540).
  - Validation:
    - `forecast_length_days` must be ≤ 3650 (10‑year window).
    - `start_date`/`current_date` are fixed to the server’s dates, matching history.
  - Response: JSON object of simulation outputs (all arrays serialized as lists).
  - Side effect: Saves the full `{ setup, results }` to `mechafil-server/data/latest_simulation.json`.

Results Retrieval
- `GET /latest-simulation` — Returns the last saved `{ setup, results }` JSON.
- `GET /latest-simulation/plots` — Returns base64‑encoded PNG plots for each numeric result series (JSON). Query: `max_plots` (default 24).
- `GET /latest-simulation/plots/html` — Renders an HTML gallery of plots (easier to view in a browser). Query: `max_plots`.


## Example Workflow

1) Start the server (first run will build the historical cache):

```
uvicorn mechafil_server.main:app --reload --host 0.0.0.0 --port 8000 --app-dir mechafil-server
```

2) Run an example simulation (1‑year forecast, default parameters):

```
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{"forecast_length_days": 365, "lock_target": 0.3}'
```

3) Inspect the saved setup and results:

```
curl http://localhost:8000/latest-simulation | jq '.setup | {start_date, current_date, end_date, forecast_length_days, sector_duration_days, lock_target}'
```

4) Open the plots in a browser:

- HTML gallery: http://localhost:8000/latest-simulation/plots/html
- JSON (base64 images): http://localhost:8000/latest-simulation/plots


## Troubleshooting

- Historical data not found (404) or not yet loaded (503):
  - Wait a bit after startup; initial fetch can take minutes.
  - Ensure your Spacescope token is valid and set in `mechafil_server/simulation.py`.
- Length mismatch errors when simulating:
  - Delete `mechafil-server/data/historical_data.pkl` and `historical_data_meta.json`, then restart the server to rebuild the cache.
- Matplotlib errors when requesting plots:
  - Install `matplotlib` in your environment.
- JAX device/backend issues:
  - See `/health` for the active JAX backend. Install a CPU or GPU wheel per JAX docs.


## Repository Layout

```
.
├── mechafil-jax/                 # JAX simulator package
├── mechafil-server/              # FastAPI server package
│   └── data/                     # Historical cache + latest simulation
├── pystarboard/                  # Data connectors (Spacescope/Starboard)
├── fip100/                       # Scenario tooling and dashboards
└── README.md                     # This file
```


## Security Notes

- The current server includes a placeholder bearer token reference. Replace it with your own token before using in production.
- Restrict CORS in `mechafil_server.main` for production deployments.
- Avoid committing secrets; see `.gitignore` below.
