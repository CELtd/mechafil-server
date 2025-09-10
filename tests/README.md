# MechaFil Server Testing

This document explains how we test the MechaFil Server to ensure API endpoints produce mathematically identical results to offline simulations.

## What Are We Testing?

The core principle is simple: **every API endpoint must return exactly the same data as running the equivalent standalone script offline**. This ensures:

- The web API is mathematically accurate
- No data processing errors exist between API and simulation logic
- Changes to the simulation engine are immediately detected
- Users get identical results whether they use the API or run simulations directly

## How We Test

### Testing Strategy

1. **Start Real Server**: Launch FastAPI server on port 8001 with real data connections
2. **Make API Calls**: Send HTTP requests to actual running server endpoints  
3. **Run Offline Scripts**: Execute standalone Python scripts with identical parameters
4. **Compare Results**: Perform mathematical comparison of JSON outputs
5. **Assert Equality**: Tests fail if any differences are found

### Key Testing Scripts

#### `test-data-fetching.py`
**Purpose**: Fetches and processes historical data exactly like the `/historical-data/full` API endpoint

**What it does**:
- Connects to Spacescope using same authentication as the server
- Fetches historical raw byte power, renewal rates, and FIL+ rates
- Processes and smooths the data using identical algorithms
- Outputs results to `spacescope_results.json` in API-compatible format

**Used by**: `test_historical_data_endpoint_matches_offline` - compares API historical data with offline results

#### `test-simulation.py`  
**Purpose**: Runs simulations exactly like the `/simulate` API endpoint

**What it does**:
- Accepts same command-line parameters as API (`--lock-target`, `--forecast-length-days`, etc.)
- Loads historical data and applies same trimming/processing logic
- Runs mechafil-jax simulation with identical parameters
- Outputs results to `offline_simulation.json` in API-compatible format

**Used by**: All simulation tests - compares API simulation results with offline results

### Comparison Logic

The `compare_simulation_results()` function performs deep mathematical comparison:
- **Floating Point Precision**: Uses tolerance of `1e-10` for numerical differences
- **Array Comparison**: Element-by-element comparison of numerical arrays  
- **Nested Objects**: Recursively compares complex JSON structures
- **Type Validation**: Ensures matching data types throughout

## Test Coverage

### API Validation Tests (`test_api_validation.py`)

These tests compare API endpoints with offline scripts to ensure mathematical accuracy:

**Historical Data Tests**:
- `test_historical_data_endpoint_matches_offline`: Compares `/historical-data/full` API response with `test-data-fetching.py` output

**Simulation Tests**:
- `test_default_simulation_endpoint`: Tests `/simulate` with default parameters vs `test-simulation.py`
- `test_lock_target_simulation`: Tests custom lock_target parameter (e.g., 0.1 instead of 0.3)
- `test_forecast_length_simulation`: Tests custom forecast length (e.g., 365 days vs default)
- `test_sector_duration_simulation`: Tests custom sector duration (e.g., 365 days vs default 540)
- `test_simulation_parameter_scenarios`: Tests multiple parameter combinations in a single test

### Endpoint Functionality Tests (`test_endpoints.py`)

These tests verify API behavior without mathematical comparison:

**Core Endpoints**:
- `test_health_endpoint`: Verifies `/health` returns status and JAX backend info
- `test_root_endpoint`: Verifies `/` returns server info and endpoint list

**Data Endpoints**:
- `test_historical_data_summary_endpoint`: Verifies `/historical-data` returns proper summary structure
- `test_historical_data_full_endpoint`: Verifies `/historical-data/full` response structure

**Simulation Endpoints**:
- `test_simulate_endpoint_minimal_request`: Tests `/simulate` with empty JSON `{}`
- `test_simulate_endpoint_with_parameters`: Tests `/simulate` with custom parameters
- `test_simulate_endpoint_parameter_validation`: Tests parameter type validation and error handling

## Example Test Flow

```python
def test_lock_target_simulation(self, api_client, offline_simulation_scripts, tmp_path):
    # 1. Define test parameters
    params = {"lock_target": 0.1, "forecast_length_days": 365}
    
    # 2. Call API endpoint
    response = api_client.post("/simulate", json=params)
    assert response.status_code == 200
    api_data = response.json()
    
    # 3. Run offline script with same parameters  
    offline_data = run_offline_simulation_with_params(
        "test-simulation.py", tmp_path, params, api_data=api_data
    )
    
    # 4. Mathematical comparison
    assert compare_simulation_results(api_data, offline_data)
```

## Running Tests

### Quick Start

```bash
# Install dependencies
poetry install --with test

# Run all tests
poetry run pytest tests/ -v

# Run only API validation tests (slower, but comprehensive)
poetry run pytest tests/integration/test_api_validation.py -v

# Run only endpoint functionality tests (faster)
poetry run pytest tests/integration/test_endpoints.py -v
```

### Test Performance
- **Endpoint Tests**: ~5-10 seconds (no data fetching)
- **API Validation Tests**: ~30-60 seconds per test (includes real data fetching) 
- **Full Test Suite**: ~2-3 minutes total

## Why This Approach Works

1. **Real Integration**: Tests use actual server, database, and external APIs
2. **Mathematical Precision**: Catches computational errors that unit tests miss
3. **Regression Detection**: Any change to simulation logic immediately breaks tests
4. **Production Confidence**: If tests pass, the API works exactly like offline simulations
5. **Automated**: No manual verification needed - tests either pass or fail definitively

## Adding New Tests

To add a new simulation parameter test:

```python
def test_new_parameter(self, api_client, offline_simulation_scripts, tmp_path):
    params = {"new_param": 42}
    
    # API call
    response = api_client.post("/simulate", json=params)
    api_data = response.json()
    
    # Offline simulation
    offline_data = run_offline_simulation_with_params(
        "test-simulation.py", tmp_path, params, api_data=api_data
    )
    
    # Comparison
    assert compare_simulation_results(api_data, offline_data)
```

The offline scripts automatically handle new parameters via command-line arguments, making test expansion straightforward.