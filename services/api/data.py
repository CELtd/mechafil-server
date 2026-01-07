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

    def try_shared_cache(self, start_date: date, current_date: date, end_date: date) -> Dict[str, Any] | None:
        """Try to load data from shared cache volume.
        
        Returns:
            Dict containing the cached data if successful, None if failed or not configured
        """
        if not settings.USE_SHARED_CACHE or not settings.SHARED_CACHE_DIR:
            return None
            
        try:
            logger.info(f"Trying shared cache at {settings.SHARED_CACHE_DIR}")
            
            # Use DiskCache to access the shared cache directory
            shared_cache = Cache(settings.SHARED_CACHE_DIR)
            
            # Try to get cached data using the same key format
            end_date = current_date + timedelta(days=settings.WINDOW_DAYS)
            cache_key = f"offline_data_{start_date}{current_date}{end_date}"
            
            cached_data = shared_cache.get(cache_key)
            if cached_data is not None:
                logger.info("Successfully retrieved data from shared cache")
                return cached_data
            else:
                logger.info("Data not found in shared cache")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to access shared cache: {e}")
            return None

    # ------------------------------------------------------------------
    # Data loading/saving
    # ------------------------------------------------------------------

    def load_historical_data(self) -> None:
        """Load historical data from shared cache only. No fallback to local data fetching."""

        logger.info("Loading historical data from shared cache...")
    
        # Check if shared cache is configured
        if not settings.USE_SHARED_CACHE or not settings.SHARED_CACHE_DIR:
            raise RuntimeError(
                "Shared cache is not configured. This service requires USE_SHARED_CACHE=true "
                "and SHARED_CACHE_DIR to be set. Make sure the cache-updater service is running."
            )
    
        # Load cache data using fixed key
        from diskcache import Cache
        cache = Cache(settings.SHARED_CACHE_DIR)

        # Use fixed cache key that matches cache_updater
        cache_key = "offline_data_latest"
        logger.info(f"Loading cache entry: {cache_key}")

        cached_result = cache.get(cache_key)

        if cached_result is None:
            raise RuntimeError(
                f"No cache data found for key '{cache_key}' in shared cache. "
                "Make sure the cache-updater service has run and populated the cache."
            )

        try:
            # Save into self fields
            self.historical_data = cached_result
            self.smoothed_hist_rbp = cached_result["smoothed_rbp"]
            self.smoothed_hist_rr = cached_result["smoothed_rr"]
            self.smoothed_hist_fpr = cached_result["smoothed_fpr"]

            self.start_date = _parse_cached_date(cached_result.get("data_start_date")) or settings.STARTUP_DATE
            self.current_date = _parse_cached_date(cached_result.get("data_end_date")) or (date.today() - timedelta(days=1))

            logger.info(f"Historical data loaded from shared cache successfully!")
            logger.info(f"Data period: {self.start_date} to {self.current_date}")
            return

        except Exception as e:
            logger.error(f"Failed to load cache data from key {cache_key}: {e}")
            raise RuntimeError(f"Failed to load historical data from shared cache: {e}")
 
            
    def refresh_historical_data(self) -> None:
        """Refresh historical data by reloading from shared cache.
        
        This method simply reloads data from the shared cache. The cache-updater 
        service is responsible for refreshing the actual data.
        """
        logger.info("Refreshing historical data from shared cache...")
        self.load_historical_data()

            
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


    def trim_data_for_simulation(self, forecast_length: int) -> Dict[str, Any]:
        """Trim simulation data vectors to match the forecast time horizon.
        
        This method adjusts the data vectors to align with simulation requirements:
        - Expire vectors are limited to the forecast period only
        - Pledge release vector spans both historical + forecast periods
        
        The trimming ensures that:
        1. `rb_known_scheduled_expire_vec` and `qa_known_scheduled_expire_vec`
           contain only future expiration data (forecast period)
        2. `known_scheduled_pledge_release_full_vec` contains historical +
           forecast pledge release data (full simulation period)
           
        Args:
            forecast_length: Number of days to forecast into the future
            
        Returns:
            Dict containing trimmed simulation data with proper vector sizes
            
        Raises:
            ValueError: If historical_data is not loaded or forecast_length is invalid
        """
        if not self.historical_data:
            raise ValueError("Historical data must be loaded before trimming")
            
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")
            
        if not self.start_date or not self.current_date:
            raise ValueError("Start date and current date must be set")

        hist_data = self.historical_data['offline_data']
        new_data = hist_data.copy()
        
        # Calculate the historical period length
        historical_days = int((self.current_date - self.start_date).days)
        pledge_release_length = historical_days + forecast_length
        
        # Trim expire vectors to forecast period only
        #new_data['rb_known_scheduled_expire_vec'] = hist_data['rb_known_scheduled_expire_vec'][historical_days:historical_days+forecast_length]
        #new_data['qa_known_scheduled_expire_vec'] = hist_data['qa_known_scheduled_expire_vec'][historical_days:historical_days+forecast_length]
        new_data['rb_known_scheduled_expire_vec'] = hist_data['rb_known_scheduled_expire_vec'][:forecast_length]
        new_data['qa_known_scheduled_expire_vec'] = hist_data['qa_known_scheduled_expire_vec'][:forecast_length]
        
        # Trim pledge release vector to historical + forecast period
        new_data['known_scheduled_pledge_release_full_vec'] = hist_data['known_scheduled_pledge_release_full_vec'][:pledge_release_length]

        return new_data


def _parse_cached_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            return None
    return None

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
