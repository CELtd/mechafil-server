"""Shared configuration management for Mechafil Server services."""

import os
from datetime import date
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Shared application settings for all Mechafil services."""
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # API settings
    SPACESCOPE_TOKEN: Optional[str] = os.getenv("SPACESCOPE_TOKEN")
    SPACESCOPE_AUTH_FILE: str = os.getenv("SPACESCOPE_AUTH_FILE", ".spacescope_auth")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Shared cache directory (for Fly.io volume)
    USE_SHARED_CACHE: bool = os.getenv("USE_SHARED_CACHE", "false").lower() == "true"
    SHARED_CACHE_DIR: Path = Path(os.getenv("SHARED_CACHE_DIR", "/data/shared-cache"))
    
    # Data constants and defaults
    STARTUP_DATE: date = date(2022, 10, 10)
    WINDOW_DAYS: int = 3650  # 10 years forecast
    MAX_HISTORICAL_DATA_FETCHING_RETRIES: int = 5
    
    # Cache updater specific settings
    RELOAD_TRIGGER: str = os.getenv("RELOAD_TRIGGER", "01:00")  # Daily refresh at 1:00 UTC
    RELOAD_TEST_MODE: bool = os.getenv("RELOAD_TEST_MODE", "false").lower() == "true"
    
    def get_spacescope_auth(self):
        """Get Spacescope authentication token or file path."""
        if self.SPACESCOPE_TOKEN:
            return self.SPACESCOPE_TOKEN
        
        # Check for auth file
        auth_file = Path(self.SPACESCOPE_AUTH_FILE)
        if auth_file.exists():
            return str(auth_file.absolute())
        
        # If no token or file, raise error
        raise RuntimeError(
            f"Spacescope authentication not found. Set SPACESCOPE_TOKEN environment variable "
            f"or create {self.SPACESCOPE_AUTH_FILE} file."
        )


class APISettings(Settings):
    """API service specific settings."""
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"


class CacheUpdaterSettings(Settings):
    """Cache updater service specific settings."""
    PORT: int = int(os.getenv("PORT", "8001"))


# Create service-specific setting instances
api_settings = APISettings()
cache_updater_settings = CacheUpdaterSettings()

# Default settings (backwards compatibility)
settings = api_settings