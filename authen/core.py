"""Core utilities for Authen: logging and provider configurations."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import ollama

# Package-level constants (can be overridden by applications)
PACKAGE_ROOT = Path(__file__).parent.parent
DEFAULT_TEMP_DIR = PACKAGE_ROOT / ".temp"
DEFAULT_LOG_DIR = PACKAGE_ROOT / ".logs"
DEFAULT_LLM_CACHE_DIR = PACKAGE_ROOT / ".cache" / "llm_cache"
DEFAULT_VALIDATION_CACHE_DIR = PACKAGE_ROOT / ".cache" / "validation_cache"

# Global package state
_global_logger: Optional[logging.Logger] = None
_log_dir: Optional[Path] = None
_llm_cache_dir: Optional[Path] = None
_validation_cache_dir: Optional[Path] = None


def set_log_dir(log_dir: Path) -> None:
    """Set the directory for authen log files."""
    global _log_dir
    _log_dir = log_dir
    _log_dir.mkdir(parents=True, exist_ok=True)


def set_llm_cache_dir(llm_cache_dir: Path) -> None:
    """Set the directory for LLM response cache files."""
    global _llm_cache_dir
    _llm_cache_dir = llm_cache_dir
    _llm_cache_dir.mkdir(parents=True, exist_ok=True)


def set_validation_cache_dir(validation_cache_dir: Path) -> None:
    """Set the directory for validation cache files."""
    global _validation_cache_dir
    _validation_cache_dir = validation_cache_dir
    _validation_cache_dir.mkdir(parents=True, exist_ok=True)


def get_log_dir() -> Path:
    """Get the current log directory."""
    if _log_dir is None:
        return DEFAULT_LOG_DIR
    return _log_dir


def get_llm_cache_dir() -> Path:
    """Get the current LLM cache directory."""
    if _llm_cache_dir is None:
        return DEFAULT_LLM_CACHE_DIR
    return _llm_cache_dir


def get_validation_cache_dir() -> Path:
    """Get the current validation cache directory."""
    if _validation_cache_dir is None:
        return DEFAULT_VALIDATION_CACHE_DIR
    return _validation_cache_dir


def setup_logger(
    name: str = "authen", log_level: int = logging.INFO, log_dir: Optional[Path] = None
) -> logging.Logger:
    """Set up and return a configured logger instance."""
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

    actual_log_dir = log_dir or DEFAULT_LOG_DIR
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = actual_log_dir / f"authen_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "authen") -> logging.Logger:
    """Get or create a logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger(name, log_dir=get_log_dir())
    return _global_logger


# LLM Provider Configurations


def get_openai_models() -> List[str]:
    """Get list of available OpenAI models."""
    return [
        "gpt-5.1",
        "gpt-5-pro",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o3-deep-research",
        "o3-pro",
        "o3",
        "o4-mini",
        "o1",
        "o1-mini",
    ]


def get_anthropic_models() -> List[str]:
    """Get list of available Anthropic models."""
    return [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]


def get_google_models() -> List[str]:
    """Get list of available Google models."""
    return [
        "gemini-3-pro",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]


def get_grok_models() -> List[str]:
    """Get list of available xAI Grok models."""
    return [
        "grok-4",
        "grok-4-fast",
        "grok-3",
        "grok-3-mini",
    ]


def get_ollama_models() -> List[str]:
    """Get list of available local Ollama models."""
    try:
        response = ollama.list()
        # Check if response is an object with 'models' attribute
        if hasattr(response, "models"):
            return [m.model for m in response.models]
        # Fallback for older versions or dict response
        elif isinstance(response, dict) and "models" in response:
            return [m["name"] for m in response["models"]]
        return []
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []


def supports_web_search(provider: str, model_name: str) -> bool:
    """Check if a provider/model combination supports web search."""
    # OpenAI: Most models support web_search_preview, but not all
    if provider == "openai":
        unsupported = [
            "o1",
            "o1-mini",
            "o3",
            "o3-pro",
            "o3-deep-research",
            "o4-mini",
        ]
        return model_name not in unsupported

    # Anthropic: Claude models with web_search_20250305
    if provider == "anthropic":
        return True  # Most Claude models support it

    # Google: Gemini models with google_search
    if provider == "google":
        return True  # Most Gemini models support it

    # Grok: All models support search_parameters
    if provider == "grok":
        return True

    # Ollama: Local models typically don't support web search
    if provider == "ollama":
        return False

    return False


@dataclass(frozen=True)
class ProviderConfig:
    """UI metadata for configuring LLM providers."""

    env_var: Optional[str]
    api_label: Optional[str]
    model_label: str
    fetch_models: Callable[[], List[str]]
    default_model: str
    custom_placeholder: str
    empty_hint: Optional[str] = None


PROVIDER_CONFIG: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        env_var="OPENAI_API_KEY",
        api_label="OpenAI API Key",
        model_label="Model",
        fetch_models=get_openai_models,
        default_model="gpt-4o",
        custom_placeholder="gpt-4o",
    ),
    "anthropic": ProviderConfig(
        env_var="ANTHROPIC_API_KEY",
        api_label="Anthropic API Key",
        model_label="Model",
        fetch_models=get_anthropic_models,
        default_model="claude-3-opus-20240229",
        custom_placeholder="claude-3-opus-20240229",
    ),
    "google": ProviderConfig(
        env_var="GOOGLE_API_KEY",
        api_label="Google API Key",
        model_label="Model",
        fetch_models=get_google_models,
        default_model="gemini-2.5-flash",
        custom_placeholder="gemini-2.5-flash",
    ),
    "grok": ProviderConfig(
        env_var="XAI_API_KEY",
        api_label="xAI API Key",
        model_label="Model",
        fetch_models=get_grok_models,
        default_model="grok-4",
        custom_placeholder="grok-4",
    ),
    "ollama": ProviderConfig(
        env_var=None,
        api_label=None,
        model_label="Local Model",
        fetch_models=get_ollama_models,
        default_model="llama3",
        custom_placeholder="llama3",
        empty_hint="No Ollama models detected. Ensure Ollama is running and pull at least one model.",
    ),
}

PROVIDER_ORDER = ["openai", "anthropic", "google", "grok", "ollama"]
