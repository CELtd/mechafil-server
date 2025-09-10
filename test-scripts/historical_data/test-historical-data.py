import pickle
import numpy as np
import jax.numpy as jnp
from datetime import date, timedelta, datetime
from diskcache import Cache

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import mechafil_server.data as u
from jax import config
config.update("jax_enable_x64", True)
import json


def run_simulation():
    ###########################
    # Helper Functions
    ###########################
    def get_offline_data(start_date, current_date, end_date):
        cache = Cache("../cache_directory")
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

    sector_duration_days = 540
    lock_target = 0.3

    print(f"Current smoothed last historical RBP: {smoothed_rbp}")
    print("Running simulations ...")

    rbp = jnp.ones(forecast_length_days) * smoothed_rbp
    rr = jnp.ones(forecast_length_days) * smoothed_rr
    fpr = jnp.ones(forecast_length_days) * smoothed_fpr

    _ = sim.run_sim(
        rbp, rr, fpr, lock_target, start_date, current_date,
        forecast_length_days, sector_duration_days, offline_data,
        use_available_supply=False
    )

    # Build results dict to match the FastAPI endpoint
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
    with open("spacescope_results.json", "w") as f:
        json.dump(results_to_save, f, default=convert, indent=2)

    print("Results saved to spacescope_results.json")

    with open("api_results.json") as f:
        api_data = json.load(f)
    
    with open("spacescope_results.json") as f:
        offline_data = json.load(f)
    
    print(api_data == offline_data)


if __name__ == "__main__":
    run_simulation()
