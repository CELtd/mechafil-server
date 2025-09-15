import numpy as np
import jax.numpy as jnp
from datetime import date, timedelta
from diskcache import Cache
import mechafil_jax.data as data
import mechafil_jax.sim as sim
import sys, os
import argparse
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import mechafil_server.data as u
from jax import config
config.update("jax_enable_x64", True)
import json

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))


def run_simulation(forecast_length_days=10*365, sector_duration_days=540, lock_target=0.3):
    ###########################
    # Helper Functions
    ###########################
    def get_offline_data(start_date, current_date, end_date):
        cache = Cache("./cache_directory")
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
    current_date = date.today() - timedelta(days=2)
    start_date = date(2022, 10, 10)
    end_date = current_date + timedelta(days=forecast_length_days)
    
    offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, hist_rbp, hist_rr, hist_fpr = get_offline_data(
        start_date, current_date, end_date
    )

    rbp = jnp.ones(forecast_length_days) * smoothed_rbp
    rr = jnp.ones(forecast_length_days) * smoothed_rr
    fpr = jnp.ones(forecast_length_days) * smoothed_fpr
    
    results = sim.run_sim(
        rbp, rr, fpr, lock_target, start_date, current_date,
        forecast_length_days, sector_duration_days, offline_data,
        use_available_supply=False
    )

    # Build results dict to match the FastAPI endpoint
    results_to_save = {
        "input": {
            "forecast_length_days": forecast_length_days,
        },
        "smoothed_metrics": {
            "raw_byte_power": float(smoothed_rbp),
            "renewal_rate": float(smoothed_rr),
            "filplus_rate": float(smoothed_fpr),
        },
        "simulation_output": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in results.items()
        },
    }

    # Save to JSON file
    with open("offline_simulation.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print("Offline simulation saved to offline_simulation.json")

    # Compare with API output if available
    try:
        with open("api_results.json") as f:
            api_data = json.load(f)
        print("Match with API:", api_data == results_to_save)
    except FileNotFoundError:
        print("No api_results.json found from API call, skipping comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation with customizable parameters')
    parser.add_argument('--forecast-length-days', type=int, default=10*365, 
                       help='Forecast length in days (default: 3650)')
    parser.add_argument('--sector-duration-days', type=int, default=540,
                       help='Sector duration in days (default: 540)')
    parser.add_argument('--lock-target', type=float, default=0.3,
                       help='Lock target (default: 0.3)')
    
    args = parser.parse_args()
    
    run_simulation(
        forecast_length_days=args.forecast_length_days,
        sector_duration_days=args.sector_duration_days,
        lock_target=args.lock_target
    )
