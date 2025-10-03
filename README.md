# MechaFil Server

A production-ready FastAPI service that provides HTTP endpoints for running Filecoin economic forecasts using real historical blockchain data and sophisticated simulation models.

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

- **`mechafil_server/main.py`**: FastAPI application with endpoint definitions
- **`mechafil_server/data.py`**: Data processing and historical metrics calculation  
- **`mechafil_server/models.py`**: Pydantic models for request/response validation
- **`mechafil_server/config.py`**: Configuration management and constants
- **`mechafil_server/scheduler.py`**: Background scheduler for automated daily data refresh
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


## Historical Data & Automatic Refresh

### Initial Data Loading
- On first startup, the server fetches and caches historical data under `mechafil-server/data/` (this may take a few minutes).
- `current_date` is set to "yesterday".

### Automated Daily Refresh
The server automatically refreshes historical data daily at a configurable time:

- **Default**: Data refreshes every day at `02:00 UTC`
- **Configuration**: Set `RELOAD_TRIGGER=HH:MM` in your `.env` file (e.g., `RELOAD_TRIGGER=03:30` for 3:30 AM UTC)
- **Process**: The scheduler clears the cache and fetches fresh data from Spacescope, exactly like startup
- **Resilience**: If refresh fails, the server continues running with existing cached data

### Testing Mode
For development and testing, enable frequent refresh cycles:

```bash
# Refresh every 2 minutes instead of daily
RELOAD_TEST_MODE=true
```

The scheduler runs as a background asyncio task and handles errors gracefully without interrupting server operations.


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

Core Endpoints
- `GET /` — Root endpoint with server information and available endpoints.
- `GET /health` — Health check endpoint with server status and JAX backend info.

Historical Data
- `GET /historical-data` — Summary of loaded historical data.

Simulation
- `POST /simulate` — Run a forecast with weekly averaged results (optional body: `rbp`, `rr`, `fpr`, `lock_target`, `forecast_length_days`, `sector_duration_days`).

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

### Online Documentation

- **Interactive API Docs**:
  - Swagger UI: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc

### Read the Docs

Comprehensive documentation including:
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

# View the documentation
python -m http.server 8080 -d build/html
# Then open http://localhost:8080
```

**Auto-rebuild on changes:**

```bash
poetry run sphinx-autobuild docs/source docs/build/html
# Opens at http://127.0.0.1:8000
```

The documentation source is in `docs/` and is ready for deployment to Read the Docs.

## Security

- Do not commit real tokens. `.gitignore` excludes `.env` and data caches.
- Restrict CORS for production.
