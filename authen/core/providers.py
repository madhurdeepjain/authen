"""LLM provider metadata for Authen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import ollama


def get_openai_models() -> List[str]:
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
    return [
        "gemini-3-pro",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]


def get_grok_models() -> List[str]:
    return ["grok-4", "grok-4-fast", "grok-3", "grok-3-mini"]


def get_ollama_models() -> List[str]:
    try:
        response = ollama.list()
        if hasattr(response, "models"):
            return [m.model for m in response.models]
        if isinstance(response, dict) and "models" in response:
            return [m["name"] for m in response["models"]]
        return []
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"Error fetching Ollama models: {exc}")
        return []


def supports_web_search(provider: str, model_name: str) -> bool:
    if provider == "openai":
        unsupported = {"o1", "o1-mini", "o3", "o3-pro", "o3-deep-research", "o4-mini"}
        return model_name not in unsupported
    if provider in {"anthropic", "google", "grok"}:
        return True
    return False


@dataclass(frozen=True)
class ProviderConfig:
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
