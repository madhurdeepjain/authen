"""
Core module containing shared schemas, configuration, and utilities.
"""

from authen.core.cache import (
    CacheManager,
    MemoryCache,
    SQLiteCache,
    clear_global_cache,
    get_cache,
)
from authen.core.config import Config
from authen.core.schemas import (
    Affiliation,
    Author,
    ReferenceData,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "Config",
    "Affiliation",
    "Author",
    "ReferenceData",
    "ValidationResult",
    "ValidationStatus",
    "CacheManager",
    "MemoryCache",
    "SQLiteCache",
    "get_cache",
    "clear_global_cache",
]
