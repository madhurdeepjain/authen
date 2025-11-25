"""Authen - Academic Reference Validation Pipeline.

A modular package for extracting, parsing, validating,
and exporting academic references.
"""

from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

# Load .env from project root (parent of src/authen)
_env_path = _Path(__file__).parent.parent.parent / ".env"
_load_dotenv(_env_path)

from authen.core.config import Config  # noqa: E402
from authen.core.schemas import (  # noqa: E402
    Affiliation,
    Author,
    ReferenceData,
    ValidationResult,
    ValidationStatus,
)
from authen.pipeline.orchestrator import Pipeline  # noqa: E402

__version__ = "0.1.0"
__all__ = [
    "Config",
    "Pipeline",
    "Affiliation",
    "Author",
    "ReferenceData",
    "ValidationResult",
    "ValidationStatus",
]
