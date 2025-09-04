"""Configuration management for Mechafil Server."""

import os
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
    
    @property
    def has_spacescope_auth(self) -> bool:
        """Check if spacescope authentication is configured."""
        return (
            self.SPACESCOPE_TOKEN is not None or 
            os.path.exists(self.SPACESCOPE_AUTH_FILE)
        )


# Global settings instance
settings = Settings()