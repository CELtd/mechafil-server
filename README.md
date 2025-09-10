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


## Historical Data

- On first startup, the server fetches and caches historical data under `mechafil-server/data/` (this may take a few minutes).
- `current_date` is set to “yesterday.” Historical arrays are harmonized to a consistent length to avoid broadcasting issues.


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


## Testing

MechaFil Server features a comprehensive testing strategy that ensures API responses match offline simulation results with mathematical precision.

### Testing Philosophy

Our tests validate the core promise of the API: **API responses must be identical to offline simulations run with the same parameters**. This ensures:

- API correctness and reliability
- Consistency between web service and direct simulation usage  
- Mathematical accuracy of results
- Regression detection for simulation logic changes

### Test Architecture

```
tests/
├── test-simulation.py           # Offline simulation script
├── test-data-fetching.py        # Offline data fetching script  
├── integration/
│   ├── test_api_validation.py   # API vs offline comparison tests
│   └── test_endpoints.py        # API functionality tests
└── utils/
    └── simulation_helpers.py    # Test utilities and comparison logic
```

### Test Types

1. **API Validation Tests**: Compare API endpoints with offline simulations
   - Historical data endpoint vs offline data fetching
   - Simulation endpoint vs offline simulation with identical parameters
   - Parameter variation testing (lock_target, forecast_length, etc.)

2. **Endpoint Functionality Tests**: Verify API behavior
   - Response structure validation
   - Error handling verification
   - Parameter acceptance testing

3. **Integration Tests**: Full system validation
   - Server startup/shutdown automation
   - Real database and cache integration
   - End-to-end request/response cycles

### Quick Start

Install test dependencies:
```bash
poetry install --with test
```

Run all tests:
```bash
poetry run pytest tests/ -v
```

Run specific test categories:
```bash
# API validation tests (compares API with offline simulations)
poetry run pytest tests/integration/test_api_validation.py -v

# Basic endpoint functionality tests
poetry run pytest tests/integration/test_endpoints.py -v
```

Generate coverage report:
```bash
poetry run pytest tests/ --cov=mechafil_server --cov-report=html
```

### How Tests Work

1. **Start Real Server**: Tests automatically start FastAPI server on port 8001
2. **Make API Calls**: HTTP requests to actual running server endpoints
3. **Run Offline Simulation**: Execute standalone simulation scripts with same parameters
4. **Compare Results**: Mathematical comparison with floating-point tolerance
5. **Assert Equality**: Tests fail if API ≠ offline simulation results

Example test flow:
```python
# 1. API call
api_response = client.post("/simulate", json={"lock_target": 0.1})

# 2. Offline simulation  
offline_result = run_offline_simulation(params={"lock_target": 0.1})

# 3. Mathematical comparison
assert api_response.json() == offline_result
```

For detailed testing methodology, see `tests/README.md`.

### Legacy Tests

Original shell script tests in `test/` directory are preserved for reference. The new pytest framework in `tests/` provides better automation, error reporting, and CI/CD integration.

## Troubleshooting

- Missing auth: ensure `.env` contains `SPACESCOPE_TOKEN` or `SPACESCOPE_AUTH_FILE` and restart.
- First run is slow: initial data fetch can take minutes.
- Test failures: ensure server can start on port 8001 for testing.

## Security

- Do not commit real tokens. `.gitignore` excludes `.env` and data caches.
- Restrict CORS for production.
