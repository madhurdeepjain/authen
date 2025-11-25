"""
Core module containing shared schemas, configuration, and utilities.
"""

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
]
