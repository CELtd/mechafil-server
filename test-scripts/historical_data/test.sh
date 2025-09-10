#!/bin/bash

echo 'Fetching data from API'
# Fetch API results
curl -s http://localhost:8000/historical-data/full    | python -c "import json, sys; print(json.dumps(json.load(sys.stdin), indent=2))" > api_results.json

echo 'Simulating'
# Run the Python test
poetry run python test-historical-data.py