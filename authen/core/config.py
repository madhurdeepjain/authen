"""Shared configuration helpers for Authen."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMP_DIR = PACKAGE_ROOT / ".temp"
DEFAULT_LOG_DIR = PACKAGE_ROOT / ".logs"
DEFAULT_LLM_CACHE_DIR = PACKAGE_ROOT / ".cache" / "llm_cache"
DEFAULT_VALIDATION_CACHE_DIR = PACKAGE_ROOT / ".cache" / "validation_cache"

_log_dir: Optional[Path] = None
_llm_cache_dir: Optional[Path] = None
_validation_cache_dir: Optional[Path] = None


def set_log_dir(log_dir: Path) -> None:
    """Set the directory for log files and ensure it exists."""
    global _log_dir
    _log_dir = log_dir
    _log_dir.mkdir(parents=True, exist_ok=True)


def set_llm_cache_dir(llm_cache_dir: Path) -> None:
    """Set the directory for LLM cache responses."""
    global _llm_cache_dir
    _llm_cache_dir = llm_cache_dir
    _llm_cache_dir.mkdir(parents=True, exist_ok=True)


def set_validation_cache_dir(validation_cache_dir: Path) -> None:
    """Set the directory for validation cache responses."""
    global _validation_cache_dir
    _validation_cache_dir = validation_cache_dir
    _validation_cache_dir.mkdir(parents=True, exist_ok=True)


def get_log_dir() -> Path:
    """Return the configured log directory (or default)."""
    return _log_dir or DEFAULT_LOG_DIR


def get_llm_cache_dir() -> Path:
    """Return the configured LLM cache directory (or default)."""
    return _llm_cache_dir or DEFAULT_LLM_CACHE_DIR


def get_validation_cache_dir() -> Path:
    """Return the configured validation cache directory (or default)."""
    return _validation_cache_dir or DEFAULT_VALIDATION_CACHE_DIR
