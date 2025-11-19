"""Cache updater service configuration (using shared config)."""

# Import shared configuration
from shared.config import cache_updater_settings as settings

# Export for backwards compatibility  
__all__ = ["settings"]