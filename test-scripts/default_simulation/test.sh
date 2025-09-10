#!/bin/bash

echo 'Fetching data from API'
# Fetch API results
curl -s -X POST http://localhost:8000/simulate     -H "Content-Type: application/json"     -d '{}'   | python -c "import json, sys; print(json.dumps(json.load(sys.stdin), indent=2))" > api_results.json

echo 'Simulating'
# Run the Python test
poetry run python test-results.py
