"""
LLM-based reference parsing subpackage.

This subpackage provides:
- Multiple LLM provider support (OpenAI, Anthropic, Ollama, Google Gemini)
- Structured output parsing for reliable reference extraction
- Chunked processing for large documents
- Standalone usage as a reusable component
"""

from authen.llm.parser import ReferenceParser
from authen.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)

__all__ = [
    "ReferenceParser",
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "GoogleProvider",
    "get_provider",
]
