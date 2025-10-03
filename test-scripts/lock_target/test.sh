#!/bin/bash

echo 'Fetching data from API'
# Fetch API results
curl -s -X POST http://localhost:8000/simulate/full     -H "Content-Type: application/json"     -d '{"forecast_length_days": 365, "sector_duration_days": 365, "lock_target": 0.1}'   | python -c "import json, sys; print(json.dumps(json.load(sys.stdin), indent=2))" > api_results.json

echo 'Simulating'
# Run the Python test with matching parameters from API request
poetry run python test-results.py --forecast-length-days 365 --sector-duration-days 365 --lock-target 0.1

