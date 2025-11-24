# Mechafil Server

A serverless microservices architecture for mechafil-jax simulations with shared cache deployment on Fly.io.

## Overview

MechaFil Server is a web service that wraps the [mechafil-jax](https://github.com/CELtd/mechafil-jax) simulation engine, providing:

- **Historical Data API**: Access to processed Filecoin network metrics (raw byte power, renewal rates, FIL+ rates)
- **Simulation API**: Run economic forecasts with customizable parameters
- **Serverless Architecture**: Auto-start/auto-stop machines that scale to zero when idle
- **Shared Cache**: Single persistent volume shared between API and cache updater services
- **Automated Updates**: GitHub Actions + admin endpoint for cache refresh
- **Read the Docs**: Comprehensive documentation served alongside the API

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fly.io Deployment                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Single Machine (mechafil-api)           │  │
│  │                                                      │  │
│  │  ┌────────────────┐    ┌──────────────────┐        │  │
│  │  │  FastAPI       │    │  Cache Updater   │        │  │
│  │  │  (API Service) │    │  (Admin Endpoint)│        │  │
│  │  └────────┬───────┘    └────────┬─────────┘        │  │
│  │           │                     │                   │  │
│  │           └─────────┬───────────┘                   │  │
│  │                     │                               │  │
│  │           ┌─────────▼──────────┐                    │  │
│  │           │  Shared Cache      │                    │  │
│  │           │  Volume (3GB)      │                    │  │
│  │           │  /data/shared-cache│                    │  │
│  │           └────────────────────┘                    │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Machine Auto-Stop: Stops when idle (~2-3 min)             │
│  Machine Auto-Start: Starts on incoming HTTP request       │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Daily at 1:00 UTC
                            ▼
                ┌────────────────────────┐
                │  GitHub Actions        │
                │  (Trigger Cache Update)│
                └────────────────────────┘
                            │
                            │ POST /admin/update-cache
                            │ (Wakes machine, runs updater)
                            ▼
               ┌─────────────────────────┐
               │   Spacescope API        │
               │   (Historical Data)     │
               └─────────────────────────┘
```

### Key Components

- **`services/api/main.py`**: FastAPI application with endpoints and admin trigger
- **`services/api/data.py`**: Shared-cache reader that feeds the simulation engine
- **`services/api/models.py`**: Pydantic models for request/response validation
- **`services/cache_updater/main.py`**: Cache updater service (callable via admin endpoint)
- **`services/cache_updater/scheduler.py`**: Optional background scheduler for continuous runs
- **`shared/config.py`**: Configuration management shared by both services
- **`tests/`**: Production-grade test suite with API validation
- **`docker/Dockerfile`**: Unified Docker image containing both services

### Deployment Model

The deployment uses a **unified image with dual-purpose execution**:

1. **Single Machine**: Both API and cache updater code deployed in one container
2. **Shared Volume**: 3GB persistent volume at `/data/shared-cache` shared between services
3. **Serverless Operation**: Machine auto-stops after ~2-3 minutes of inactivity
4. **Two Execution Modes**:
   - **API Mode** (default): Serves HTTP requests via FastAPI
   - **Update Mode**: Admin endpoint `/admin/update-cache` imports and runs cache updater logic

**Cache Update Strategy**:
- GitHub Actions calls `/admin/update-cache` daily at 1:00 UTC
- Endpoint wakes machine (if stopped), runs cache updater, reloads data
- Machine then auto-stops after inactivity period
- Alternative: Keep machine running for continuous scheduler-based updates

## Prerequisites

- Python 3.11+
- Poetry (recommended) or pip
- JAX (CPU): `pip install -U "jax[cpu]"` (see JAX docs for GPU wheels)
- For deployment: Fly.io CLI (`flyctl`)

## Installation

Using Poetry (recommended):

```bash
cd mechafil-server
poetry install
```

This installs both services and all dependencies declared in `pyproject.toml` (including FastAPI, Uvicorn, JAX, matplotlib, mechafil-jax, and pystarboard).

## Configuration

### Environment Variables

Set credentials via `.env` file in the repo root or `mechafil-server/` folder:

```bash
# Required: Spacescope API authentication
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
# or
SPACESCOPE_AUTH_FILE=./auths/spacescope_auth.json

# Cache configuration
USE_SHARED_CACHE=true
SHARED_CACHE_DIR=/data/shared-cache

# API settings
HOST=0.0.0.0
PORT=8000

# Cache updater settings (for background scheduler mode)
RELOAD_TRIGGER=01:00  # Daily refresh at 1:00 AM UTC
RELOAD_TEST_MODE=false  # Set to true for 2-minute test cycles

# Logging
LOG_LEVEL=INFO

# CORS (for production, restrict to your domain)
CORS_ORIGINS=*
```

### Spacescope Authentication

The cache updater fetches historical data via Spacescope/Starboard. Provide credentials using either:

- `SPACESCOPE_TOKEN` — bearer token string, e.g. `Bearer YOUR_TOKEN_HERE`
- `SPACESCOPE_AUTH_FILE` — path to JSON file with `{ "auth_key": "Bearer YOUR_TOKEN_HERE" }`

## Local Development

### Option 1: Running Services Separately

**Start API service:**
```bash
# Ensure cache is populated first (see below)
poetry run mechafil-api

# Or with hot reload
poetry run uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Populate cache (run once or when you need fresh data):**
```bash
USE_SHARED_CACHE=true \
SHARED_CACHE_DIR=./shared-cache \
poetry run python -m services.cache_updater.main --once
```

**Run cache updater as background service (optional):**
```bash
# Runs scheduler that refreshes cache daily at RELOAD_TRIGGER time
USE_SHARED_CACHE=true \
SHARED_CACHE_DIR=./shared-cache \
poetry run cache-updater
```

### Option 2: Using Docker

Build and run locally:
```bash
# Build the unified image
docker build -f docker/Dockerfile -t mechafil-server .

# Run API service
docker run -p 8000:8000 \
  -v $(pwd)/shared-cache:/data/shared-cache \
  -e USE_SHARED_CACHE=true \
  -e SHARED_CACHE_DIR=/data/shared-cache \
  -e SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" \
  mechafil-server

# Run cache updater once
docker run --rm \
  -v $(pwd)/shared-cache:/data/shared-cache \
  -e USE_SHARED_CACHE=true \
  -e SHARED_CACHE_DIR=/data/shared-cache \
  -e SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" \
  --entrypoint python \
  mechafil-server -m services.cache_updater.main --once
```

### Accessing the API

Once running, the server is available at `http://localhost:8000`:

- **Homepage**: http://localhost:8000 (redirects to docs)
- **Read the Docs**: http://localhost:8000/documentation/ (after building docs)
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Historical Data**: http://localhost:8000/historical-data

## Building Documentation

The server includes comprehensive Sphinx documentation that's automatically served at the root URL:

```bash
# Install documentation dependencies
poetry install --with docs

# Build HTML documentation
cd docs
poetry run make html
cd ..

# Start the server - docs will be available at http://localhost:8000
poetry run mechafil-api
```

**Auto-rebuild docs while editing:**
```bash
poetry run sphinx-autobuild docs/source docs/build/html
```

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for comprehensive deployment guide covering:

- Fly.io serverless deployment with shared volume
- GitHub Actions for automated cache updates
- Admin endpoint configuration
- Monitoring and troubleshooting
- Alternative deployment patterns (Docker Compose, Kubernetes)

**Quick Fly.io Deployment:**

```bash
# 1. Install Fly.io CLI
curl -L https://fly.io/install.sh | sh

# 2. Login
flyctl auth login

# 3. Create shared cache volume
flyctl volumes create shared_cache --region fra --size 3

# 4. Deploy (uses fly.toml configuration)
flyctl deploy

# 5. Verify
flyctl status
curl https://mechafil-api.fly.dev/health
```

The machine will auto-stop when idle and auto-start on incoming requests.

## API Endpoints

### Core Endpoints

- `GET /` — Redirects to Read the Docs documentation (or Swagger UI if docs not built)
- `GET /health` — Health check endpoint with server status and JAX backend info
- `GET /documentation/` — Read the Docs HTML documentation (if built)

### Data Endpoints

- `GET /historical-data` — Historical network data downsampled to Mondays

### Simulation Endpoints

- `POST /simulate` — Run a forecast with Monday-averaged results
  - Optional parameters: `rbp`, `rr`, `fpr`, `lock_target`, `forecast_length_days`, `sector_duration_days`, `output`
- `POST /simulate/full` — Run a forecast with daily results (all data points)

### Admin Endpoints

- `POST /admin/update-cache` — Trigger cache update from GitHub Actions or external automation

## API Examples

### Simulation Parameters

The `/simulate` endpoint accepts these optional parameters:
- `rbp`: Raw byte power onboarding (PIB/day) - float or array
- `rr`: Renewal rate (0..1) - float or array
- `fpr`: FIL+ rate (0..1) - float or array
- `lock_target`: Target lock ratio - float or array
- `forecast_length_days`: Forecast length in days - integer
- `sector_duration_days`: Average sector duration in days - integer
- `output`: Specific output field(s) to return - string or array of strings

All parameters are optional. Defaults are calculated from historical data.

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

```bash
# Install test dependencies
poetry install --with test

# Run all tests
poetry run pytest tests/ -v

# Run specific test categories
poetry run pytest tests/integration/ -v  # Integration tests only
poetry run pytest tests/ -m "not slow" -v  # Skip slow tests
```

### Testing Philosophy

Our tests ensure **API responses are identical to offline simulations** run with the same parameters. This validates:
- API correctness and reliability
- Consistency between web service and direct simulation usage
- Mathematical accuracy of results

For comprehensive testing methodology, architecture, and examples, see [`tests/README.md`](tests/README.md).

## Project Structure

```
mechafil-server/
├── services/
│   ├── api/                    # API service
│   │   ├── main.py             # FastAPI application and endpoints
│   │   ├── data.py             # Shared cache reader
│   │   ├── models.py           # Pydantic request/response models
│   │   ├── results.py          # Result processing and formatting
│   │   ├── config.py           # API-specific configuration
│   │   └── build_docs.py       # Documentation builder utility
│   └── cache_updater/          # Cache updater service
│       ├── main.py             # Updater entry point (supports --once mode)
│       ├── data.py             # Cache writer (fetches from Spacescope)
│       ├── scheduler.py        # Background scheduler (optional)
│       └── config.py           # Cache updater-specific configuration
├── shared/
│   └── config.py               # Shared configuration for both services
├── docs/                       # Sphinx documentation
│   ├── source/                 # Documentation source files
│   └── build/html/             # Built HTML documentation
├── tests/                      # Test suite
│   ├── integration/            # Integration tests
│   └── README.md               # Testing documentation
├── docker/
│   └── Dockerfile              # Unified Docker image for both services
├── .github/workflows/
│   └── update-cache-daily.yml  # GitHub Actions for cache updates
├── fly.toml                    # Fly.io deployment configuration
├── pyproject.toml              # Poetry dependencies and scripts
├── README.md                   # This file
└── DEPLOYMENT.md               # Deployment guide
```

## Cache Update Strategies

### Strategy 1: Serverless with GitHub Actions (Current Deployment)

**How it works:**
- Machine auto-stops after ~2-3 minutes of inactivity
- GitHub Actions calls `/admin/update-cache` daily at 1:00 UTC
- Endpoint wakes machine, runs updater, reloads data
- Machine auto-stops again after update completes

**Advantages:**
- Minimal resource usage (machine only runs when needed)
- No cost when idle
- Simple automation via GitHub Actions

**Configuration:**
```yaml
# .github/workflows/update-cache-daily.yml
on:
  schedule:
    - cron: '0 1 * * *'  # Daily at 1:00 AM UTC
```

### Strategy 2: Continuous Background Scheduler

**How it works:**
- Keep machine running continuously
- Built-in scheduler refreshes cache at configured time
- Uses `services.cache_updater.scheduler` for automated updates

**Advantages:**
- Self-contained (no external triggers needed)
- More reliable (no dependency on GitHub Actions)
- Can handle more frequent updates

**Configuration:**
```bash
# In fly.toml, disable auto-stop
[http_service]
  auto_stop_machines = false  # Keep machine running
  min_machines_running = 1    # Always have 1 machine

# Set refresh time via environment variable
[env]
  RELOAD_TRIGGER = "02:00"  # Refresh at 2:00 AM UTC
```

### Strategy 3: Hybrid Approach

**How it works:**
- Machine auto-stops when idle (serverless)
- Cache updater runs in scheduler mode when machine is awake
- GitHub Actions as backup trigger

**Advantages:**
- Best of both worlds
- Automatic updates when machine happens to be running
- External trigger ensures updates happen even if machine stays stopped

## Security

- **Secrets Management**: Do not commit real tokens. `.gitignore` excludes `.env` files and data caches.
- **CORS**: Restrict `CORS_ORIGINS` for production deployments.
- **Admin Endpoint**: Consider adding authentication to `/admin/update-cache` if exposed publicly.
- **Volume Access**: Shared cache volume is only accessible within your Fly.io organization.

## Troubleshooting

### Cache Not Loading

```bash
# Check if cache directory exists and has data
flyctl ssh console --app mechafil-api
ls -lah /data/shared-cache

# Manually trigger cache update
curl -X POST https://mechafil-api.fly.dev/admin/update-cache
```

### Machine Not Stopping

```bash
# Check for health checks (should be none for true serverless)
flyctl status --app mechafil-api

# Verify fly.toml configuration
grep -A 5 "http_service" fly.toml
# Should show: auto_stop_machines = "stop", min_machines_running = 0
```

### GitHub Actions Failing

```bash
# Check workflow runs
# Visit: https://github.com/YOUR_ORG/YOUR_REPO/actions

# Test admin endpoint manually
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# Check machine logs
flyctl logs --app mechafil-api
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `poetry run pytest tests/ -v`
5. Submit a pull request

## License

[Add your license here]

## Related Projects

- [mechafil-jax](https://github.com/CELtd/mechafil-jax) - JAX-based Filecoin simulation engine
- [pystarboard](https://github.com/CELtd/pystarboard) - Python client for Spacescope API
- [mcp-server-mechafil](../mcp-server-mechafil) - Model Context Protocol server for Claude.ai integration
