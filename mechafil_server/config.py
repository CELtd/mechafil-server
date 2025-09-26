"""Configuration management for Mechafil Server."""

import os
from datetime import date
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings."""
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    
    # API settings
    SPACESCOPE_TOKEN: Optional[str] = os.getenv("SPACESCOPE_TOKEN")
    SPACESCOPE_AUTH_FILE: str = os.getenv("SPACESCOPE_AUTH_FILE", ".spacescope_auth")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Cache 
    CACHE_DIR: Path = Path(__file__).parent.parent / '.cache'

    # Data constants and defaults
    STARTUP_DATE: date = date(2025, 1, 1)
    MAX_HISTORICAL_DATA_FETCHING_RETRIES = 10
    WINDOW_DAYS: int = 10 * 365
    SECTOR_DURATION_DAYS: int = 540
    LOCK_TARGET: float = 0.3
    
    # Daily data refresh settings
    RELOAD_TRIGGER: str = os.getenv("RELOAD_TRIGGER", "02:00")
    
    # Testing: set to True to refresh every 2 minutes instead of daily
    RELOAD_TEST_MODE: bool = os.getenv("RELOAD_TEST_MODE", "false").lower() == "true"
    
    # Historical data averaging settings
    USE_WEEKLY_AVERAGING: bool = os.getenv("USE_WEEKLY_AVERAGING", "true").lower() == "true"
    
    @property
    def has_spacescope_auth(self) -> bool:
        """Check if spacescope authentication is configured."""
        return (
            self.SPACESCOPE_TOKEN is not None or 
            os.path.exists(self.SPACESCOPE_AUTH_FILE)
        )
    
    def get_spacescope_auth(self) -> str:
        """Get Spacescope authentication token or file path."""
        token = self.SPACESCOPE_TOKEN
        auth_file = self.SPACESCOPE_AUTH_FILE
        bearer_or_file = token or (auth_file if auth_file and os.path.exists(auth_file) else None)
        
        if not bearer_or_file:
            raise RuntimeError(
                "Missing Spacescope auth. Set SPACESCOPE_TOKEN or SPACESCOPE_AUTH_FILE in the environment."
            )
        
        return bearer_or_file


# Global settings instance
settings = Settings()