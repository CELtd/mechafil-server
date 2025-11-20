# Mechafil Server

A microservices architecture for mechafil-jax simulations with serverless deployment patterns.

## Overview

MechaFil Server is a web service that wraps the [mechafil-jax](https://github.com/CELtd/mechafil-jax) simulation engine, providing:

- **Historical Data API**: Access to processed Filecoin network metrics (raw byte power, renewal rates, FIL+ rates)
- **Simulation API**: Run economic forecasts with customizable parameters
- **Real-time Processing**: Uses live data from Spacescope for up-to-date simulations
- **Caching**: Intelligent caching for performance optimization
- **Production Testing**: Comprehensive test suite validating API responses against offline simulations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   mechafil-jax   │    │   Spacescope    │
│   Web Server    │──▶│   Simulation     │───▶│   Data Source   │
│                 │    │   Engine         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Data Cache    │
│   (DiskCache)   │
└─────────────────┘
```

### Key Components

- **`services/api/main.py`**: FastAPI application with endpoint definitions
- **`services/api/data.py`**: Shared-cache reader that feeds the simulation engine  
- **`services/api/models.py`**: Pydantic models for request/response validation
- **`shared/config.py`**: Configuration management shared by the API and cache updater
- **`services/cache_updater/`**: Long-running or one-shot job that populates the cache volume from Spacescope
- **`tests/`**: Production-grade test suite with API validation


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

The cache-updater service fetches historical data via Spacescope/Starboard (through `pystarboard`). Provide credentials so it can authenticate before writing to the shared cache volume.

Set credentials via environment variables (both services load `.env` from the repo root or from this folder, and also `.test-env` from the repo root):

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


## Historical Data & Automatic Refresh

### Initial Data Loading
- Prime the shared cache by running the cache updater (`poetry run cache-updater`, `docker-compose run --rm --entrypoint="" cache-updater python -m services.cache_updater.main --once`, or the Fly.io job described below). It fetches data from Spacescope and writes DiskCache entries into the shared volume (defaults to `/data/shared-cache` or `./shared-cache` when running locally).
- When the API starts it reads the newest cache entry from that shared volume (`USE_SHARED_CACHE=true`). If no cache data exists yet the API fails fast so you know the updater must run first.

### Automated Daily Refresh
The **cache-updater service** automatically refreshes historical data daily at a configurable time:

- **Default**: Refresh every day at `02:00 UTC`
- **Configuration**: Set `RELOAD_TRIGGER=HH:MM` in your `.env` file (e.g., `RELOAD_TRIGGER=03:30` for 3:30 AM UTC)
- **Process**: The cache updater fetches fresh data on schedule and writes it into the shared cache volume. The API always uses whatever snapshot is already on disk; restart the API (or let Fly auto-stop/start it) to pick up the most recent snapshot.
- **Resilience**: If the updater fails, the previously cached snapshot remains in place until the next successful run.

### Testing Mode
For development and testing, enable frequent refresh cycles on the **cache-updater** service:

```bash
# Refresh every 2 minutes instead of daily
RELOAD_TEST_MODE=true
```

This runs inside the updater's asyncio scheduler and does not involve the API container.


## Run

From the `mechafil-server` folder:

```
# Start with Poetry (shared cache must already be populated)
poetry run mechafil-api

# Or run Uvicorn explicitly
poetry run uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000
```

Refresh the cache whenever you need new data. Locally you can point the updater at any writable directory (e.g. `./shared-cache`) so you don't need a Docker volume:

```
USE_SHARED_CACHE=true \
SHARED_CACHE_DIR=./shared-cache \
poetry run python -m services.cache_updater.main --once
```

Once populated, the API reads the snapshot from `SHARED_CACHE_DIR` at startup—restart the service whenever you want it to pick up freshly written data.

The server will start on `http://localhost:8000`.

**Note**: Build the documentation first (see [Documentation](#documentation) section below) to enable the Read the Docs interface at the root URL.

**API Documentation Access:**
- **Homepage**: http://localhost:8000 (redirects to Read the Docs if built, otherwise Swagger UI)
- **Read the Docs**: http://localhost:8000/documentation/ (after building docs)
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## Deployment Patterns

### Docker Compose (local or single VM)

Use the provided `docker-compose.yml` to replicate the serverless-style split between cache updater and API:

```
# Keep the cache fresh in the background
docker-compose up cache-updater -d

# Bring the API online on demand (press Ctrl+C to stop)
docker-compose up api

# Or run the updater once (e.g., cron job)
docker-compose run --rm --entrypoint="" cache-updater python -m services.cache_updater.main --once
```

The compose file mounts the `shared_cache` volume at `/data/shared-cache` so both services read/write the same DiskCache directory.

You can keep `api` running continuously (leave `docker-compose up api` attached or add `-d`) or treat it as on-demand infrastructure: stop it when idle and start it only when you need to serve traffic. Every time the container boots it loads the latest snapshot from the mounted volume before answering requests.

### Fly.io Deployment

You can mirror the same pattern on Fly.io: one persistent volume, a scheduled cache updater, and an auto-starting API Machine.

1. **Create a shared cache volume** (pick the region where both apps will run):
   ```bash
   fly volumes create shared_cache --region fra --size 10
   ```

2. **Deploy the cache updater app** (long-running or invoked on a schedule):
   - `fly launch --name mechafil-cache-updater --no-deploy`
   - In `fly.toml`, point to `docker/cache-updater.Dockerfile`, mount the volume, and pass env vars:
     ```toml
     [build]
     dockerfile = "docker/cache-updater.Dockerfile"

     [mounts]
     source="shared_cache"
     destination="/data/shared-cache"

     [env]
     USE_SHARED_CACHE="true"
     SHARED_CACHE_DIR="/data/shared-cache"
     SPACESCOPE_TOKEN="Bearer ..."
     ```
   - `fly deploy`
   - To mimic a Lambda-style job, trigger it via Machines or GitHub Actions:
     ```bash
     fly machine run \
       --app mechafil-cache-updater \
       --mount shared_cache:/data/shared-cache \
       --entrypoint "" \
       -- python -m services.cache_updater.main --once
     ```

3. **Deploy the API app** (auto-starts on incoming requests):
   - `fly launch --name mechafil-api --no-deploy`
   - Configure `docker/api.Dockerfile`, mount the same volume, and set env vars:
     ```toml
     [build]
     dockerfile = "docker/api.Dockerfile"

     [mounts]
     source="shared_cache"
     destination="/data/shared-cache"

     [env]
     USE_SHARED_CACHE="true"
     SHARED_CACHE_DIR="/data/shared-cache"
     ```
   - Enable serverless-style behavior with Machines:
     ```toml
     [http_service]
     internal_port = 8000
     auto_stop_machines = "stop"
     auto_start_machines = true
     ```
   - `fly deploy`

With this setup the updater keeps the Fly volume fresh (either continuously or via scheduled runs) and the API stays "cold" until Fly routes a request, similar to API Gateway + Lambda backed by EFS.

If you prefer an always-on API, simply omit the `auto_stop_machines` option (or scale to a Nomad app). Either way, startup loads the shared volume snapshot so requests always hit the most recent cache contents without any extra work.


## API

Core Endpoints
- `GET /` — Redirects to Read the Docs documentation (or Swagger UI if docs not built)
- `GET /health` — Health check endpoint with server status and JAX backend info
- `GET /documentation/` — Read the Docs HTML documentation (if built)

Historical Data
- `GET /historical-data` — Historical network data downsampled to Mondays

Simulation
- `POST /simulate` — Run a forecast with weekly averaged results (optional body: `rbp`, `rr`, `fpr`, `lock_target`, `forecast_length_days`, `sector_duration_days`, `output`).

## Examples

### Simulation Parameters

The `/simulate` endpoint accepts these optional parameters:
- `rbp`: Raw byte power onboarding (PIB/day) - float or array
- `rr`: Renewal rate (0..1) - float or array
- `fpr`: FIL+ rate (0..1) - float or array
- `lock_target`: Target lock ratio - float or array
- `forecast_length_days`: Forecast length in days - integer
- `sector_duration_days`: Average sector duration in days - integer
- `output`: Specific output field(s) to return - string or array of strings (if not specified, returns all fields)

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

**Get only specific output fields:**
```bash
# Single field
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "forecast_length_days": 365,
    "output": "available_supply"
  }'

# Multiple fields
curl -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "forecast_length_days": 365,
    "output": ["available_supply", "network_RBP_EIB", "circ_supply"]
  }'
```

### Advanced Examples

**Long-term forecast (10 years):**
```bash
curl -X POST http://localhost:8000/simulate/full \
  -H 'Content-Type: application/json' \
  -d '{
    "forecast_length_days": 3650,
    "lock_target": 0.25
  }'
```

**Time-varying parameters using arrays:**
```bash
curl -X POST http://localhost:8000/simulate/full \
  -H 'Content-Type: application/json' \
  -d '{
    "rbp": [3.0, 3.5, 4.0],
    "rr": [0.8, 0.85, 0.9],
    "fpr": [0.8, 0.85, 0.9],
    "forecast_length_days": 3
  }'
```

## Testing

MechaFil Server features a comprehensive testing strategy that validates API responses against offline simulations with mathematical precision.

### Quick Start

Install test dependencies:
```bash
poetry install --with test
```

Run all tests:
```bash
poetry run pytest tests/ -v
```

### Testing Philosophy

Our tests ensure **API responses are identical to offline simulations** run with the same parameters. This validates:
- API correctness and reliability
- Consistency between web service and direct simulation usage
- Mathematical accuracy of results

### Detailed Testing Information

For comprehensive testing methodology, architecture, test types, and detailed examples, see [`tests/README.md`](tests/README.md).

## Documentation

Complete API documentation is available in multiple formats:

### Read the Docs (Comprehensive)

After building the documentation, it's accessible at:
- **http://localhost:8000** (root redirects here automatically)
- **http://localhost:8000/documentation/**

The documentation includes:
- Complete API endpoint reference
- Request/response models
- Configuration guide
- Deployment guides (Docker, Kubernetes, Cloud)
- Code examples (Python, JavaScript, curl)
- Advanced usage patterns

**Build the documentation locally:**

```bash
# Install documentation dependencies
poetry install --with docs

# Build HTML documentation
cd docs
poetry run make html

# Start the server - docs will be available at http://localhost:8000
cd ..
poetry run mechafil-server
```

Once built, the documentation is automatically served at:
- **http://localhost:8000** (root redirects here)
- **http://localhost:8000/documentation/**

**Auto-rebuild docs while editing:**

```bash
# This will auto-rebuild on changes and serve at http://127.0.0.1:8000
poetry run sphinx-autobuild docs/source docs/build/html
```

### Interactive API Docs (Alternative)

If you prefer the interactive Swagger UI or ReDoc:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

The documentation source is in `docs/` and is ready for deployment to Read the Docs.

## Security

- Do not commit real tokens. `.gitignore` excludes `.env` and data caches.
- Restrict CORS for production.
