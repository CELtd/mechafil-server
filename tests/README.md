# MechaFil Server Testing

A comprehensive testing framework that validates API responses against offline simulations with mathematical precision. This ensures the FastAPI server produces identical results to standalone simulation scripts.

## Testing Philosophy

**Core Principle**: API responses must be mathematically identical to offline simulations run with the same parameters.

This approach provides:
- **Mathematical Accuracy**: Ensures computational correctness
- **Regression Detection**: Catches simulation logic changes
- **API Reliability**: Validates web service consistency
- **Production Confidence**: Real integration testing

## Test Architecture

```
tests/
├── conftest.py                  # Test fixtures and server management
├── test-simulation.py          # Standalone simulation script
├── test-data-fetching.py       # Standalone data fetching script
├── integration/
│   ├── test_api_validation.py  # API vs offline validation tests
│   └── test_endpoints.py       # API functionality tests
├── utils/
│   └── simulation_helpers.py   # Test utilities and comparison logic
└── cache_directory/            # Test data cache
```

## Test Categories

### 1. API Validation Tests (`test_api_validation.py`)

**Purpose**: Compare API endpoints with offline simulation results

**Method**: 
1. Make API call with specific parameters
2. Run offline script with identical parameters  
3. Compare JSON outputs for exact equality

**Tests**:
- `test_historical_data_endpoint_matches_offline`: `/historical-data/full` vs `test-data-fetching.py`
- `test_default_simulation_endpoint`: `/simulate` with defaults vs `test-simulation.py`
- `test_lock_target_simulation`: `/simulate` with custom lock_target
- `test_forecast_length_simulation`: `/simulate` with custom forecast length
- `test_sector_duration_simulation`: `/simulate` with custom sector duration
- `test_simulation_parameter_scenarios`: Multiple parameter combinations

### 2. Endpoint Functionality Tests (`test_endpoints.py`)

**Purpose**: Verify API behavior and structure

**Method**:
1. Make API calls to various endpoints
2. Validate response structure and status codes
3. Test error handling and edge cases

**Tests**:
- `test_health_endpoint`: Health check functionality
- `test_root_endpoint`: API information endpoint
- `test_historical_data_summary_endpoint`: Data summary structure
- `test_simulate_endpoint_minimal_request`: Default parameter handling
- `test_simulate_endpoint_with_parameters`: Custom parameter acceptance

## How Tests Work

### Test Execution Flow

1. **Server Setup**: `live_server` fixture starts FastAPI server on port 8001
2. **API Request**: Test makes HTTP request to running server
3. **Offline Simulation**: Test runs standalone script with same parameters
4. **Result Comparison**: Mathematical comparison of JSON outputs
5. **Server Cleanup**: Fixture automatically stops server after tests

### Example Test Flow

```python
def test_lock_target_simulation(api_client, tmp_path):
    # 1. Define parameters
    params = {"lock_target": 0.1, "forecast_length_days": 365}
    
    # 2. Make API call to running server
    response = api_client.post("/simulate", json=params)
    api_data = response.json()
    
    # 3. Run offline simulation with same parameters
    offline_data = run_offline_simulation_with_params(
        "test-simulation.py", tmp_path, params, api_data=api_data
    )
    
    # 4. Mathematical comparison
    assert compare_simulation_results(api_data, offline_data)
```

### Simulation Scripts

#### `test-simulation.py`
- **Purpose**: Runs the same simulation logic as the API server
- **Parameters**: Accepts command-line args (--lock-target, --forecast-length-days, etc.)
- **Output**: Creates `offline_simulation.json` with results matching API format
- **Usage**: Used by all simulation validation tests

#### `test-data-fetching.py`  
- **Purpose**: Fetches and processes historical data like `/historical-data/full` endpoint
- **Parameters**: No command-line parameters (uses current date)
- **Output**: Creates `spacescope_results.json` with data matching API format
- **Usage**: Used by historical data validation test

### Comparison Logic

The `compare_simulation_results()` function performs deep comparison with:
- **Floating Point Tolerance**: Handles computational precision differences
- **Array Comparison**: Element-by-element comparison of numerical arrays
- **Nested Structure**: Recursive comparison of complex JSON objects
- **Type Checking**: Ensures matching data types

```python
def compare_simulation_results(api_result, offline_result, tolerance=1e-10):
    # Handles floating point precision, arrays, nested objects
    # Returns True if results match within tolerance
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
poetry install --with test
```

### Quick Start

Run all tests:
```bash
poetry run pytest tests/ -v
```

Run specific test types:
```bash
# API validation tests (API vs offline simulations)  
poetry run pytest tests/integration/test_api_validation.py -v

# Endpoint functionality tests
poetry run pytest tests/integration/test_endpoints.py -v

# Single test with detailed output
poetry run pytest tests/integration/test_api_validation.py::TestAPIValidation::test_lock_target_simulation -v -s
```

Generate coverage report:
```bash
poetry run pytest tests/ --cov=mechafil_server --cov-report=html
```

### Test Performance

- **Endpoint Tests**: ~5-10 seconds (fast)
- **API Validation Tests**: ~30-60 seconds per test (includes data fetching)
- **Full Suite**: ~60-120 seconds total

### Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
```

## Migration from Shell Scripts

### Before (Shell Scripts)
```bash
# Manual server startup required
poetry run mechafil-server &

# Run shell script
cd test/lock_target/
./test.sh

# Manual comparison and interpretation
```

### After (pytest)
```bash
# Automated end-to-end testing
poetry run pytest tests/integration/test_api_validation.py::TestAPIValidation::test_lock_target_simulation -v
```

### Migration Map
| Original Shell Script | New pytest Test |
|---|---|
| `test/historical_data/test.sh` | `test_historical_data_endpoint_matches_offline` |
| `test/default_simulation/test.sh` | `test_default_simulation_endpoint` |
| `test/lock_target/test.sh` | `test_lock_target_simulation` |
| `test/forecast_len/test.sh` | `test_forecast_length_simulation` |
| `test/sector_duration/test.sh` | `test_sector_duration_simulation` |

## Key Benefits

1. **Automated Server Management**: No manual startup/shutdown
2. **Mathematical Precision**: Exact floating-point comparison
3. **Better Error Messages**: Clear assertion failures with context
4. **Test Isolation**: Clean temporary directories per test
5. **CI/CD Ready**: Works in automated environments
6. **Coverage Reporting**: Built-in code coverage analysis
7. **Parameterized Testing**: Easy to add new scenarios

## Adding New Tests

### API Validation Test
```python
def test_new_scenario(self, api_client, offline_simulation_scripts, tmp_path):
    params = {"new_parameter": 42}
    
    # API call
    response = api_client.post("/simulate", json=params)
    assert response.status_code == 200
    api_data = response.json()
    
    # Offline simulation
    script_name = offline_simulation_scripts["default_simulation"]  
    offline_data = run_offline_simulation_with_params(
        script_name, tmp_path, params, api_data=api_data
    )
    
    # Mathematical comparison
    assert compare_simulation_results(api_data, offline_data)
```

### Endpoint Functionality Test
```python
def test_new_endpoint_behavior(self, api_client):
    response = api_client.get("/new-endpoint")
    assert response.status_code == 200
    
    data = response.json()
    assert "expected_field" in data
    assert isinstance(data["expected_field"], list)
```

## Troubleshooting

### Common Issues

**Test Server Startup Fails**:
- Check port 8001 is available
- Verify environment variables are set
- Ensure dependencies are installed

**Simulation Comparison Fails**:
- Check floating-point tolerance in `compare_simulation_results()`
- Verify offline script produces expected output format
- Check for environment differences (data sources, dates)

**Slow Test Performance**:
- Data fetching tests are inherently slower (real API calls)
- Use `-k` to run specific tests during development
- Consider caching data between test runs