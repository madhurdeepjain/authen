"""Logging helpers for Authen."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import get_log_dir

_global_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "authen",
    log_level: int = logging.INFO,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Instantiate a configured logger for the package."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    actual_log_dir = log_dir or get_log_dir()
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = actual_log_dir / f"authen_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "authen") -> logging.Logger:
    """Return the lazily-instantiated package logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger(name)
    return _global_logger
