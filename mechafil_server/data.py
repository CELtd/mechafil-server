"""Integration layer for running mechafil-jax simulations."""

import datetime
import json
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Union, Any, Tuple
from datetime import date, timedelta
from diskcache import Cache


import jax.numpy as jnp
import numpy as np
import pandas as pd

from mechafil_jax.data import get_simulation_data
import pystarboard.data
from .config import settings

logger = logging.getLogger(__name__)


class Data:
    """Handles the execution of mechafil-jax simulations."""

    def __init__(self):
        """Initialize the simulation runner."""
        self.historical_data: Dict[str, Any] | None = None

        # Dates used for loading historical data
        self.start_date: date | None = None
        self.current_date: date | None = None

        # Smoothed historical metrics
        self.smoothed_hist_rbp: float = 0.0
        self.smoothed_hist_rr: float = 0.0
        self.smoothed_hist_fpr: float = 0.0


    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def get_offline_data(self, start_date: date, current_date: date, end_date: date) -> Dict[str, Any]:
        """Fetch offline data and compute smoothed historical metrics."""
        logger.info(f"Loading offline data from {start_date} to {current_date} (forecast to {end_date})")

        # Get Spacescope auth from config
        bearer_or_file = settings.get_spacescope_auth()

        offline_data = get_simulation_data(bearer_or_file, start_date, current_date, end_date)

        _, hist_rbp = get_historical_daily_onboarded_power(current_date - timedelta(days=180), current_date)
        _, hist_rr = get_historical_renewal_rate(current_date - timedelta(days=180), current_date)
        _, hist_fpr = get_historical_filplus_rate(current_date - timedelta(days=180), current_date)

        smoothed_rbp = float(np.median(hist_rbp[-30:]))
        smoothed_rr = float(np.median(hist_rr[-30:]))
        smoothed_fpr = float(np.median(hist_fpr[-30:]))

        return {
            "offline_data": offline_data,
            "hist_rbp": hist_rbp,
            "hist_rr": hist_rr,
            "hist_fpr": hist_fpr,
            "smoothed_rbp": smoothed_rbp,
            "smoothed_rr": smoothed_rr,
            "smoothed_fpr": smoothed_fpr,
        }

    # ------------------------------------------------------------------
    # Data loading/saving
    # ------------------------------------------------------------------

    def load_historical_data(self) -> None:
        """Load historical data from cache if fresh, otherwise fetch new."""

        logger.info("Loading historical data...")
        
        # Setup dates
        current_date = date.today() - timedelta(days=1)
        start_date = settings.STARTUP_DATE
        end_date = current_date + timedelta(days=settings.WINDOW_DAYS)
        
        # Load cache
        cache = Cache(settings.CACHE_DIR)
        cache_key = f"offline_data_{start_date}{current_date}{end_date}"
        cached_result = cache.get(cache_key)

        if cached_result is not None:
            try:
                # Save into self fields
                self.historical_data = cached_result
                self.start_date = start_date
                self.current_date = current_date
                self.smoothed_hist_rbp = cached_result["smoothed_rbp"]
                self.smoothed_hist_rr = cached_result["smoothed_rr"]
                self.smoothed_hist_fpr = cached_result["smoothed_fpr"]
                logger.info("Historical data loaded from cache successfully!")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load existing historical data: {e}")
                logger.info("Will fetch fresh data...")

        logger.info(f"Fetching historical data from {start_date} to {current_date}...")

        try:
            data_dict = self.get_offline_data(start_date, current_date, end_date)

            # Save into self fields
            self.historical_data = data_dict
            self.start_date = start_date
            self.current_date = current_date
            self.smoothed_hist_rbp = data_dict["smoothed_rbp"]
            self.smoothed_hist_rr = data_dict["smoothed_rr"]
            self.smoothed_hist_fpr = data_dict["smoothed_fpr"]

            # Save everything to cache
            cache.set(cache_key, data_dict) 

            logger.info("Historical data loaded and saved successfully!")

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            logger.exception("Full traceback:")
            raise
            
    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_historical_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded historical data."""
        if not self.historical_data:
            return {"error": "No historical data loaded"}

        try:
            hist_rbp = self.historical_data["hist_rbp"]
            hist_rr = self.historical_data["hist_rr"]
            hist_fpr = self.historical_data["hist_fpr"]
            offline_data = self.historical_data["offline_data"]

            summary = {
                "data_loaded": True,
                "load_timestamp": datetime.datetime.now().isoformat(),
                "smoothed_metrics": {
                    "raw_byte_power": float(self.smoothed_hist_rbp),
                    "renewal_rate": float(self.smoothed_hist_rr),
                    "filplus_rate": float(self.smoothed_hist_fpr),
                },
                "historical_arrays": {
                    "raw_byte_power_length": len(hist_rbp),
                    "renewal_rate_length": len(hist_rr),
                    "filplus_rate_length": len(hist_fpr),
                },
                "offline_data_keys": list(offline_data.keys()) if isinstance(offline_data, dict) else "Not a dictionary",
            }

            if len(hist_rbp) > 0:
                summary["sample_values"] = {
                    "recent_rbp": [float(x) for x in hist_rbp[-5:]],
                    "recent_rr": [float(x) for x in hist_rr[-5:]],
                    "recent_fpr": [float(x) for x in hist_fpr[-5:]],
                }

            return summary

        except Exception as e:
            logger.error(f"Error generating historical data summary: {e}")
            return {"error": f"Failed to generate summary: {str(e)}"}

    def get_historical_data(self) -> Union[Dict[str, Any], None]:
        """Get the loaded historical data (raw + smoothed + dates)."""
        if not self.historical_data:
            return None
        return {
            **self.historical_data,
            "smoothed_rbp": self.smoothed_hist_rbp,
            "smoothed_rr": self.smoothed_hist_rr,
            "smoothed_fpr": self.smoothed_hist_fpr,
            "start_date": self.start_date,
            "current_date": self.current_date,
        }


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

PIB = 2**50


def sanity_check_date(date_in: datetime.date, err_msg=None):
    today = datetime.datetime.now().date()
    if date_in > today:
        raise ValueError(err_msg or f"Supplied date {date_in} is after today {today}")


def err_check_train_data(y_train: jnp.array):
    if len(jnp.where(y_train == 0)[0]) > 3:
        raise ValueError("Starboard data may not be fully populated (too many 0 values).")


def make_forecast_date_vec(forecast_start_date: datetime.date, forecast_length: int):
    return [forecast_start_date + datetime.timedelta(days=int(x)) for x in np.arange(forecast_length)]


def get_historical_daily_onboarded_power(start_date: datetime.date, end_date: datetime.date):
    sanity_check_date(start_date, "Specified start_date is after today!")
    sanity_check_date(end_date, "Specified end_date is after today!")

    df = pystarboard.data.query_daily_power_onboarded(start_date, end_date)
    return pd.to_datetime(df.date), df["day_onboarded_rb_power_pib"].values


def get_historical_renewal_rate(start_date: datetime.date, end_date: datetime.date):
    df = pystarboard.data.query_sector_expirations(start_date, end_date)
    t_vec = pd.to_datetime(df.date)
    renewal_rate = df["extended_rb"] / df["total_rb"]
    return t_vec, renewal_rate.values


def get_historical_expirations(start_date: datetime.date, end_date: datetime.date):
    df = pystarboard.data.query_sector_expirations(start_date, end_date)
    return df["date"], df["expired_rb"].values


def get_historical_extensions(start_date: datetime.date, end_date: datetime.date):
    df = pystarboard.data.query_sector_expirations(start_date, end_date)
    return pd.to_datetime(df.date), df["extended_rb"].values


def get_historical_deals_onboard(start_date: datetime.date, end_date: datetime.date):
    url_template = "https://api.spacescope.io/v2/deals/deal_size?end_date=%s&start_date=%s"
    df = pystarboard.data.spacescope_obj.spacescope_query(start_date, end_date, url_template)
    df["date"] = pd.to_datetime(df["stat_date"])
    deals_onboard_vec = (
        df["daily_activated_regular_deal_size"].astype(float).values
        + df["daily_activated_verified_deal_size"].astype(float).values
    )
    deals_onboard_vec /= PIB
    return df["date"], deals_onboard_vec


def get_historical_filplus_rate(start_date: datetime.date, end_date: datetime.date):
    t_rbp, rbp = get_historical_daily_onboarded_power(start_date, end_date)
    t_deals, deals = get_historical_deals_onboard(start_date, end_date)

    start_aligned = pd.to_datetime(max(t_rbp.values[0], t_deals.values[0]))
    end_aligned = pd.to_datetime(min(t_rbp.values[-1], t_deals.values[-1]))

    i_start_rbp = np.where(start_aligned == t_rbp.values)[0][0]
    i_end_rbp = np.where(end_aligned == t_rbp.values)[0][0]
    rbp_aligned = rbp[i_start_rbp:i_end_rbp + 1]

    i_start_deals = np.where(start_aligned == t_deals.values)[0][0]
    i_end_deals = np.where(end_aligned == t_deals.values)[0][0]
    deals_aligned = deals[i_start_deals:i_end_deals + 1]

    t_aligned = t_deals[i_start_deals:i_end_deals + 1]
    return t_aligned, deals_aligned / rbp_aligned
