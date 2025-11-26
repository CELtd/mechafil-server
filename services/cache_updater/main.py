"""Cache updater service - supports both scheduled runs and background service."""

import asyncio
import logging
import signal
import sys
from typing import Optional

from .config import settings
from .data import Data
from .scheduler import CacheUpdateScheduler

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
data_handler: Optional[Data] = None
scheduler: Optional[CacheUpdateScheduler] = None
shutdown_event = asyncio.Event()


async def run_once():
    """Run cache update once and exit (Lambda-like mode)."""
    logger.info("🚀 Starting single cache update run...")
    logger.info(f"Cache directory: {settings.SHARED_CACHE_DIR}")
    
    # Ensure cache directory exists
    settings.SHARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize data handler
        data_handler = Data()
        
        # Perform single cache refresh
        data_handler.refresh_historical_data()
        logger.info("✅ Cache update completed successfully!")
        
        return 0  # Success exit code
        
    except Exception as e:
        logger.error(f"❌ Cache update failed: {e}")
        logger.exception("Full traceback:")
        return 1  # Error exit code


async def run_service():
    """Run as background service with scheduler (traditional mode)."""
    global data_handler, scheduler
    
    logger.info("Starting Cache Updater Service...")
    logger.info(f"Cache directory: {settings.SHARED_CACHE_DIR}")
    
    # Ensure cache directory exists
    settings.SHARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize data handler
        data_handler = Data()
        
        # Load initial historical data
        data_handler.load_historical_data()
        logger.info("Historical data loaded successfully")
        
        # Initialize and start scheduler
        scheduler = CacheUpdateScheduler(data_handler.refresh_historical_data)
        scheduler.start()
        logger.info(f"Cache update scheduler started. Daily refresh at {settings.RELOAD_TRIGGER} UTC")
        
    except Exception as e:
        logger.error(f"Failed to initialize cache updater: {e}")
        logger.exception("Full traceback:")
        raise


async def shutdown():
    """Clean shutdown of the service."""
    global data_handler, scheduler
    
    logger.info("Shutting down Cache Updater Service...")
    
    # Stop scheduler
    if scheduler:
        await scheduler.stop_async()
        logger.info("Cache update scheduler stopped")
    
    logger.info("Cache Updater Service shutdown complete")


async def main():
    """Main entry point - supports both modes."""
    
    # Check for run mode
    run_mode = "service"  # default
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            run_mode = "once"
        elif sys.argv[1] == "--service":
            run_mode = "service"
        else:
            print("Usage: python -m cache_updater.main [--once|--service]")
            print("  --once:    Run cache update once and exit (Lambda-like)")
            print("  --service: Run as background service with scheduler (default)")
            return 1
    
    # Lambda-like mode: run once and exit
    if run_mode == "once":
        exit_code = await run_once()
        return exit_code
    
    # Service mode: run with scheduler
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await run_service()
        logger.info("Cache Updater Service is running...")
        await shutdown_event.wait()
    except Exception as e:
        logger.error(f"Service error: {e}")
        logger.exception("Full traceback:")
    finally:
        await shutdown()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)