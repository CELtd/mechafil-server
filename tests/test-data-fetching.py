#!/usr/bin/env python3
"""Standalone data fetching script for testing API historical data responses.

This script fetches and processes historical data in the same way as the FastAPI server,
allowing for direct comparison of results.
"""

import numpy as np
import jax.numpy as jnp
from datetime import date, timedelta
from diskcache import Cache
import mechafil_jax.data as data
import sys, os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import mechafil_server.data as u
from jax import config
config.update("jax_enable_x64", True)
import json

# Load environment variables
env_paths = [project_root / ".env", project_root / ".test-env"]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)


def fetch_historical_data():
    """Fetch and process historical data, same as API server."""
    
    ###########################
    # Helper Functions
    ###########################
    def get_offline_data(start_date, current_date, end_date):
        # Use cache in tests directory
        cache = Cache(project_root / "tests" / "cache_directory")
        cache_key = f"offline_data_{start_date}{current_date}{end_date}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        PUBLIC_AUTH_TOKEN = os.getenv("SPACESCOPE_TOKEN")
        offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

        _, hist_rbp = u.get_historical_daily_onboarded_power(current_date - timedelta(days=180), current_date)
        _, hist_rr = u.get_historical_renewal_rate(current_date - timedelta(days=180), current_date)
        _, hist_fpr = u.get_historical_filplus_rate(current_date - timedelta(days=180), current_date)

        smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
        smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
        smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

        result = (
            offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr,
            smoothed_last_historical_fpr, hist_rbp, hist_rr, hist_fpr
        )
        cache.set(cache_key, result)
        return result

    ###########################
    # Main Logic
    ###########################
    print("Loading historical data ...")
    current_date = date.today() - timedelta(days=1)
    start_date = date(2022, 10, 10)
    forecast_length_days = 10 * 365
    end_date = current_date + timedelta(days=forecast_length_days)

    offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, hist_rbp, hist_rr, hist_fpr = get_offline_data(
        start_date, current_date, end_date
    )

    print(f"Current smoothed last historical RBP: {smoothed_rbp}")
    print("Processing historical data ...")

    # Build results dict to match the FastAPI /historical-data/full endpoint
    results_to_save = {
        "message": "Complete historical data",
        "smoothed_metrics": {
            "raw_byte_power": float(smoothed_rbp),
            "renewal_rate": float(smoothed_rr),
            "filplus_rate": float(smoothed_fpr),
        },
        "historical_arrays": {
            "raw_byte_power": [float(x) for x in hist_rbp],
            "renewal_rate": [float(x) for x in hist_rr],
            "filplus_rate": [float(x) for x in hist_fpr],
        },
        "offline_data": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in offline_data.items()
        },
    }

    # Custom converter for JSON
    def convert(o):
        if isinstance(o, (np.ndarray, jnp.ndarray)):
            return o.tolist()
        if isinstance(o, (np.generic,)):  # NumPy scalar types
            return o.item()
        if isinstance(o, (date,)):
            return o.isoformat()
        return str(o)  # fallback: convert to string

    # Save to JSON file
    output_file = "spacescope_results.json"
    with open(output_file, "w") as f:
        json.dump(results_to_save, f, default=convert, indent=2)

    print(f"Results saved to {output_file}")

    # Compare with API output if available
    try:
        with open("api_results.json") as f:
            api_data = json.load(f)
        
        with open(output_file) as f:
            offline_data = json.load(f)
        
        print("Match with API:", api_data == offline_data)
    except FileNotFoundError:
        print("No api_results.json found from API call, skipping comparison")

    return results_to_save


if __name__ == "__main__":
    fetch_historical_data()