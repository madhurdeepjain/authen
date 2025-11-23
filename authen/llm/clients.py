"""LLM client factory functions."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

from authen.core import get_logger, supports_web_search

logger = get_logger(__name__)


class MissingAPIKeyError(ValueError):
    """Raised when an API key is required but missing."""


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise MissingAPIKeyError(f"{var_name} not found in environment variables")
    return value


def build_llm_client(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    enable_web_search: bool = False,
) -> Any:
    """Return a configured LangChain chat model for the requested provider."""
    provider = provider.lower()

    if provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")
        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        if enable_web_search:
            if not supports_web_search(provider, model_name):
                logger.warning(
                    "Model %s may not support web search. Continuing without native tool support.",
                    model_name,
                )
            else:
                return llm.bind_tools([{"type": "web_search_preview"}])
        return llm

    if provider == "anthropic":
        api_key = _require_env("ANTHROPIC_API_KEY")
        llm = ChatAnthropic(model=model_name, temperature=temperature, api_key=api_key)
        if enable_web_search:
            return llm.bind_tools(
                [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ]
            )
        return llm

    if provider == "google":
        api_key = _require_env("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
        if enable_web_search:
            return llm.bind_tools([{"google_search": {}}])
        return llm

    if provider == "grok":
        api_key = _require_env("XAI_API_KEY")
        kwargs = {"temperature": temperature, "xai_api_key": api_key}
        if enable_web_search:
            kwargs["search_parameters"] = {"mode": "auto"}
        return ChatXAI(model=model_name, **kwargs)

    if provider == "ollama":
        if enable_web_search:
            logger.warning("Ollama models do not support web search; ignoring flag.")
        return ChatOllama(model=model_name, temperature=temperature)

    raise ValueError(f"Unsupported provider: {provider}")
