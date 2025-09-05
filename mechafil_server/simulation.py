"""Integration layer for running mechafil-jax simulations."""

import datetime
import json
import os
import time
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Union, Any, Tuple
from datetime import date, timedelta
import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd

from mechafil_jax.sim import run_sim
from mechafil_jax.data import get_simulation_data
import mechafil_jax.data as data


logger = logging.getLogger(__name__)


class SimulationRunner:
    """Handles the execution of mechafil-jax simulations."""
    
    def __init__(self):
        """Initialize the simulation runner."""
        self.historical_data = None
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.historical_data_file = self.data_dir / "historical_data.pkl"
        self.historical_meta_file = self.data_dir / "historical_data_meta.json"
        # Dates used for loading historical data; set during load
        self.start_date: date | None = None
        self.current_date: date | None = None
        
    def get_offline_data(self, start_date: date, current_date: date, end_date: date) -> Tuple:
        """Get offline data including historical metrics."""
        logger.info(f"Loading offline data from {start_date} to {current_date} (forecast to {end_date})")
        
        # Load Spacescope auth from environment
        token = os.getenv('SPACESCOPE_TOKEN')
        auth_file = os.getenv('SPACESCOPE_AUTH_FILE')
        bearer_or_file = None
        if token:
            bearer_or_file = token
        elif auth_file and os.path.exists(auth_file):
            bearer_or_file = auth_file
        else:
            raise RuntimeError(
                "Missing Spacescope auth. Set SPACESCOPE_TOKEN or SPACESCOPE_AUTH_FILE in the environment."
            )

        offline_data = data.get_simulation_data(bearer_or_file, start_date, current_date, end_date)

        # Harmonize historical array lengths to avoid broadcasting errors downstream
        # Base historical arrays that must share the same length L
        base_hist_keys = [
            "historical_raw_power_eib",
            "historical_qa_power_eib",
            "historical_onboarded_rb_power_pib",
            "historical_onboarded_qa_power_pib",
            "historical_renewed_qa_power_pib",
            "historical_renewed_rb_power_pib",
            "burnt_fil_vec",
        ]
        lengths = {}
        for k in base_hist_keys:
            v = offline_data.get(k)
            if v is not None:
                try:
                    lengths[k] = len(v)
                except Exception:
                    pass
        if lengths:
            target_len = min(lengths.values())
            # If any array is longer than target_len, trim it
            for k, ln in lengths.items():
                if ln != target_len:
                    try:
                        offline_data[k] = offline_data[k][:target_len]
                        logger.warning(f"Trimmed {k} from {ln} to {target_len} to harmonize historical lengths")
                    except Exception as _e:
                        logger.warning(f"Failed trimming {k}: {_e}")
            # historical_renewal_rate should be length target_len-1
            rr = offline_data.get("historical_renewal_rate")
            if rr is not None:
                try:
                    rr_len = len(rr)
                    desired_rr_len = max(0, target_len - 1)
                    if rr_len != desired_rr_len:
                        offline_data["historical_renewal_rate"] = rr[:desired_rr_len]
                        logger.warning(
                            f"Trimmed historical_renewal_rate from {rr_len} to {desired_rr_len} to match historical length {target_len}"
                        )
                except Exception as _e:
                    logger.warning(f"Failed trimming historical_renewal_rate: {_e}")

        # Derive recent historical windows and smoothed medians directly from offline_data
        rbp_hist = np.asarray(offline_data.get("historical_onboarded_rb_power_pib", []))
        rr_hist = np.asarray(offline_data.get("historical_renewal_rate", []))
        qa_on_hist = np.asarray(offline_data.get("historical_onboarded_qa_power_pib", []))

        # Estimate FIL+ rate from QA/RB onboarding with m=10 (default)
        m = 10.0
        with np.errstate(divide='ignore', invalid='ignore'):
            fpr_est = (qa_on_hist / np.where(rbp_hist == 0, np.nan, rbp_hist) - 1.0) / (m - 1.0)
        fpr_est = np.clip(fpr_est, 0.0, 1.0)

        def tail(arr, n):
            arr = np.asarray(arr)
            return arr[-n:] if arr.size else np.array([])

        hist_rbp = tail(rbp_hist, 180)
        hist_rr = tail(rr_hist, 180)
        hist_fpr = tail(fpr_est, 180)

        def median_last(arr, n=30, default_val=0.0):
            arr = np.asarray(arr)
            if arr.size == 0:
                return float(default_val)
            window = arr[-n:] if arr.size >= n else arr
            return float(np.nanmedian(window))

        smoothed_last_historical_rbp = median_last(hist_rbp, 30, 0.0)
        smoothed_last_historical_rr = median_last(hist_rr, 30, 0.5)
        smoothed_last_historical_fpr = median_last(hist_fpr, 30, 0.0)

        result = (
            offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr,
            smoothed_last_historical_fpr, hist_rbp, hist_rr, hist_fpr
        )

        return result 

    def load_historical_data(self) -> None:
        """Load historical data from October 15, 2020 to now."""
        logger.info("Loading historical data...")
        
        if self.historical_data_file.exists():
            logger.info("Found existing historical data file, checking if it's current...")
            
            # Check if data is stale (from a different date than today)
            data_is_stale = False
            if self.historical_meta_file.exists():
                try:
                    meta = json.load(open(self.historical_meta_file, 'r'))
                    # Get the date when the data was loaded (stored in meta)
                    data_load_date = meta.get('load_date')
                    if data_load_date:
                        loaded_date = date.fromisoformat(data_load_date)
                        today = date.today()
                        if loaded_date != today:
                            logger.info(f"Historical data is stale: loaded on {loaded_date}, today is {today}")
                            data_is_stale = True
                        else:
                            logger.info(f"Historical data is current: loaded today ({today})")
                    else:
                        logger.info("No load_date in meta file, assuming data is stale")
                        data_is_stale = True
                except Exception as me:
                    logger.warning(f"Failed to parse meta file, assuming stale: {me}")
                    data_is_stale = True
            else:
                logger.info("No meta file found, assuming data is stale")
                data_is_stale = True
            
            if not data_is_stale:
                logger.info("Loading existing historical data...")
                try:
                    with open(self.historical_data_file, 'rb') as f:
                        self.historical_data = pickle.load(f)
                    # Always derive dates from offline_data length anchored to 'yesterday'
                    # to avoid stale meta storing an old current_date before FIP-81 activation.
                    loaded_start_date = None
                    loaded_current_date = None
                    if self.historical_meta_file.exists():
                        try:
                            meta = json.load(open(self.historical_meta_file, 'r'))
                            loaded_start_date = date.fromisoformat(meta.get('start_date'))
                            loaded_current_date = date.fromisoformat(meta.get('current_date'))
                            logger.info(f"Historical data meta loaded: start_date={loaded_start_date}, current_date={loaded_current_date}")
                        except Exception as me:
                            logger.warning(f"Failed to parse meta file: {me}")
                    # Derive dates from offline_data length anchored at 'yesterday'
                    try:
                        offline_data = self.historical_data[0]
                        hist_len = len(offline_data["historical_raw_power_eib"])  # inclusive length
                        # Choose current_date as yesterday to be stable and derive start_date by length
                        derived_current = date.today() - timedelta(days=1)
                        derived_start = derived_current - timedelta(days=max(hist_len - 1, 0))
                    except Exception as de:
                        logger.warning(f"Failed to derive dates from offline data: {de}")
                        derived_start = None
                        derived_current = None

                    # Always prefer derived (anchored to yesterday) to avoid stale meta
                    self.start_date = derived_start or loaded_start_date
                    self.current_date = derived_current or loaded_current_date

                    # Write meta reflecting derived dates
                    if self.start_date and self.current_date:
                        try:
                            with open(self.historical_meta_file, 'w') as mf:
                                json.dump({
                                    'start_date': self.start_date.isoformat(),
                                    'current_date': self.current_date.isoformat(),
                                    'load_date': date.today().isoformat(),
                                }, mf)
                        except Exception as me2:
                            logger.warning(f"Failed to write meta file: {me2}")

                    logger.info("Historical data loaded successfully from file")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing historical data: {e}")
                    logger.info("Will fetch fresh data...")
            else:
                logger.info("Historical data is stale, fetching fresh data...")
        
        # Setup dates
        current_date = date.today() - timedelta(days= 1)
        start_date = date(2020, 10, 15) # Startup Spacescope data
        end_date = current_date + timedelta(days=365 * 10) # End date = 10 years from current date
        
        logger.info(f"Fetching historical data from {start_date} to {current_date}...")
        logger.info("This may take a few minutes...")
        
        try:
            logger.info("Calling get_offline_data...")
            historical_data = self.get_offline_data(start_date, current_date, end_date)
            
            # Save the data
            logger.info("Saving historical data to file...")
            with open(self.historical_data_file, 'wb') as f:
                pickle.dump(historical_data, f)
            
            self.historical_data = historical_data
            # Persist the dates used so the API can reference them
            self.start_date = start_date
            self.current_date = current_date
            logger.info("Historical data loaded and saved successfully!")
            # Save meta alongside data
            try:
                with open(self.historical_meta_file, 'w') as mf:
                    json.dump({
                        'start_date': self.start_date.isoformat(),
                        'current_date': self.current_date.isoformat(),
                        'load_date': date.today().isoformat(),
                    }, mf)
            except Exception as me:
                logger.warning(f"Failed to write historical data meta: {me}")
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            logger.exception("Full traceback:")
            raise
            
    def get_historical_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded historical data."""
        if self.historical_data is None:
            return {"error": "No historical data loaded"}
            
        try:
            offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, hist_rbp, hist_rr, hist_fpr = self.historical_data
            
            summary = {
                "data_loaded": True,
                "load_timestamp": datetime.datetime.now().isoformat(),
                "smoothed_metrics": {
                    "raw_byte_power": float(smoothed_rbp),
                    "renewal_rate": float(smoothed_rr),
                    "filplus_rate": float(smoothed_fpr)
                },
                "historical_arrays": {
                    "raw_byte_power_length": len(hist_rbp),
                    "renewal_rate_length": len(hist_rr),
                    "filplus_rate_length": len(hist_fpr)
                },
                "offline_data_keys": list(offline_data.keys()) if isinstance(offline_data, dict) else "Not a dictionary"
            }
            
            # Add some sample values if arrays exist
            if len(hist_rbp) > 0:
                summary["sample_values"] = {
                    "recent_rbp": [float(x) for x in hist_rbp[-5:]],
                    "recent_rr": [float(x) for x in hist_rr[-5:]],
                    "recent_fpr": [float(x) for x in hist_fpr[-5:]]
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating historical data summary: {e}")
            return {"error": f"Failed to generate summary: {str(e)}"}
            
    def get_historical_data(self) -> Union[Tuple, None]:
        """Get the loaded historical data."""
        return self.historical_data
    
