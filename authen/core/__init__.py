"""Core package for shared Authen infrastructure."""

from dotenv import load_dotenv

from .config import (
    DEFAULT_LLM_CACHE_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_TEMP_DIR,
    DEFAULT_VALIDATION_CACHE_DIR,
    PACKAGE_ROOT,
    get_llm_cache_dir,
    get_log_dir,
    get_validation_cache_dir,
    set_llm_cache_dir,
    set_log_dir,
    set_validation_cache_dir,
)
from .logging import get_logger, setup_logger
from .providers import (
    PROVIDER_CONFIG,
    PROVIDER_ORDER,
    ProviderConfig,
    get_anthropic_models,
    get_google_models,
    get_grok_models,
    get_ollama_models,
    get_openai_models,
    supports_web_search,
)

# Load environment variables from the project root .env so downstream modules
# (UI, pipeline, validators) can rely on os.environ lookups without extra setup.
_ENV_PATH = PACKAGE_ROOT.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)

__all__ = [
    "DEFAULT_LLM_CACHE_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_TEMP_DIR",
    "DEFAULT_VALIDATION_CACHE_DIR",
    "PACKAGE_ROOT",
    "get_llm_cache_dir",
    "get_log_dir",
    "get_validation_cache_dir",
    "set_llm_cache_dir",
    "set_log_dir",
    "set_validation_cache_dir",
    "get_logger",
    "setup_logger",
    "PROVIDER_CONFIG",
    "PROVIDER_ORDER",
    "ProviderConfig",
    "get_anthropic_models",
    "get_google_models",
    "get_grok_models",
    "get_ollama_models",
    "get_openai_models",
    "supports_web_search",
]
