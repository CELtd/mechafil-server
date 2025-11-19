"""Background scheduler for daily data refresh."""

import asyncio
import logging
from datetime import datetime, time
from typing import Callable, Optional
from .config import settings

logger = logging.getLogger(__name__)


class DataRefreshScheduler:
    """Scheduler for daily data refresh at a specific time."""
    
    def __init__(self, refresh_callback: Callable[[], None]):
        """Initialize the scheduler with a refresh callback function.
        
        Args:
            refresh_callback: Function to call when data needs to be refreshed
        """
        self.refresh_callback = refresh_callback
        self.task: Optional[asyncio.Task] = None
        self.running = False
        
    def parse_time_string(self, time_str: str) -> time:
        """Parse time string in HH:MM format to time object.
        
        Args:
            time_str: Time string in format "HH:MM" (e.g., "02:00")
            
        Returns:
            time object
            
        Raises:
            ValueError: If time string format is invalid
        """
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                raise ValueError(f"Invalid time values: {hour}:{minute}")
            return time(hour, minute)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid time format '{time_str}'. Expected HH:MM format") from e
    
    def seconds_until_next_refresh(self) -> float:
        """Calculate seconds until next scheduled refresh time.
        
        Returns:
            Number of seconds until the next refresh time
        """
        # Test mode: refresh every 2 minutes
        if settings.RELOAD_TEST_MODE:
            return 120.0
            
        now = datetime.now()
        refresh_time = self.parse_time_string(settings.RELOAD_TRIGGER)
        
        # Create datetime for today at refresh time
        today_refresh = datetime.combine(now.date(), refresh_time)
        
        # If refresh time has passed today, schedule for tomorrow
        if today_refresh <= now:
            today_refresh = today_refresh.replace(day=today_refresh.day + 1)
        
        delta = today_refresh - now
        return delta.total_seconds()
    
    async def _schedule_loop(self):
        """Main scheduling loop that runs in the background."""
        if settings.RELOAD_TEST_MODE:
            logger.info("Data refresh scheduler started in TEST MODE. Refreshing every 2 minutes")
        else:
            logger.info(f"Data refresh scheduler started. Next refresh at {settings.RELOAD_TRIGGER} UTC")
        
        while self.running:
            try:
                # Calculate time until next refresh
                seconds_to_wait = self.seconds_until_next_refresh()
                logger.info(f"Next data refresh in {seconds_to_wait:.0f} seconds")
                
                # Wait until refresh time
                await asyncio.sleep(seconds_to_wait)
                
                if self.running:  # Check if we're still running after sleep
                    logger.info("Triggering scheduled data refresh...")
                    try:
                        # Call the refresh callback
                        self.refresh_callback()
                        logger.info("Scheduled data refresh completed successfully")
                    except Exception as e:
                        logger.error(f"Error during scheduled data refresh: {e}")
                        logger.exception("Full traceback:")
                
            except asyncio.CancelledError:
                logger.info("Data refresh scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                logger.exception("Full traceback:")
                # Wait a bit before retrying to avoid rapid error loops
                await asyncio.sleep(60)
    
    def start(self):
        """Start the background scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._schedule_loop())
        logger.info("Data refresh scheduler started")
    
    def stop(self):
        """Stop the background scheduler."""
        if not self.running:
            return
        
        logger.info("Stopping data refresh scheduler...")
        self.running = False
        
        if self.task and not self.task.done():
            self.task.cancel()
            self.task = None
        
        logger.info("Data refresh scheduler stopped")
    
    async def stop_async(self):
        """Stop the background scheduler asynchronously."""
        if not self.running:
            return
        
        logger.info("Stopping data refresh scheduler...")
        self.running = False
        
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                # Wait for the task to be cancelled
                await self.task
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled
            except Exception as e:
                logger.warning(f"Error during task cancellation: {e}")
            finally:
                self.task = None
        
        logger.info("Data refresh scheduler stopped")