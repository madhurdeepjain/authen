"""
LLM provider implementations with structured output support.

Each provider supports:
- Structured output for reliable parsing
- Async operation
- Web search capabilities (where available)
"""

import json
import os
from abc import ABC, abstractmethod

import structlog

from authen.core.config import LLMProvider
from authen.llm.schemas import ReferenceListOutput

logger = structlog.get_logger()


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

    @abstractmethod
    async def parse_references(
        self,
        text: str,
        system_prompt: str,
    ) -> ReferenceListOutput:
        """
        Parse references from text using structured output.

        Args:
            text: Text containing references
            system_prompt: System prompt for the LLM

        Returns:
            Parsed reference list
        """
        pass

    @abstractmethod
    async def search_and_validate(
        self,
        query: str,
    ) -> dict:
        """
        Use web search to find information about a reference.

        Args:
            query: Search query

        Returns:
            Search results
        """
        pass

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider with structured output support."""

    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        api_key: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=self.api_key)

    async def parse_references(
        self,
        text: str,
        system_prompt: str,
    ) -> ReferenceListOutput:
        """Parse references using OpenAI's structured output."""
        logger.info("parsing_with_openai", model=self.model, text_len=len(text))

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format=ReferenceListOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if response.choices[0].message.parsed:
            return response.choices[0].message.parsed
        else:
            # Fallback to manual parsing
            content = response.choices[0].message.content
            if content:
                return ReferenceListOutput.model_validate_json(content)
            raise ValueError("Failed to parse response from OpenAI")

    async def search_and_validate(self, query: str) -> dict:
        """Use OpenAI with web search for validation."""
        # Note: OpenAI doesn't have native web search, but we can use
        # function calling with a search tool or just return empty
        logger.info("openai_search_not_available", query=query)
        return {"results": [], "message": "Web search not available for OpenAI"}


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with structured output support."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        api_key: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens, api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=self.api_key)

    async def parse_references(
        self,
        text: str,
        system_prompt: str,
    ) -> ReferenceListOutput:
        """Parse references using Anthropic Claude with native structured output."""
        logger.info("parsing_with_anthropic", model=self.model, text_len=len(text))

        # Create JSON schema for structured output
        schema = ReferenceListOutput.model_json_schema()

        # Use tool_use for native structured output
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=[
                {
                    "name": "extract_references",
                    "description": "Extract academic references from text",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "extract_references"},
            messages=[
                {
                    "role": "user",
                    "content": f"Parse these references:\n\n{text}",
                }
            ],
            temperature=self.temperature,
        )

        # Extract structured output from tool use
        try:
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_references":
                    return ReferenceListOutput.model_validate(block.input)
            raise ValueError("No tool_use block found in response")
        except Exception as e:
            logger.error("anthropic_parse_error", error=str(e))
            raise ValueError(f"Failed to parse Anthropic response: {e}")

    async def search_and_validate(self, query: str) -> dict:
        """Web search not natively available for Anthropic."""
        logger.info("anthropic_search_not_available", query=query)
        return {"results": [], "message": "Web search not available for Anthropic"}


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM inference."""

    def __init__(
        self,
        model: str = "gemma3:27b",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens, api_key)
        self.base_url = base_url

    async def parse_references(
        self,
        text: str,
        system_prompt: str,
    ) -> ReferenceListOutput:
        """Parse references using Ollama."""
        import httpx

        logger.info("parsing_with_ollama", model=self.model, text_len=len(text))

        schema = ReferenceListOutput.model_json_schema()

        prompt = f"""{system_prompt}

Parse the following references and return them as a JSON object matching this schema:

{json.dumps(schema, indent=2)}

References text:
{text}

Return ONLY the JSON object, no other text."""

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()

        content = result.get("response", "")

        try:
            return ReferenceListOutput.model_validate_json(content)
        except Exception as e:
            logger.error("ollama_parse_error", error=str(e), content=content[:500])
            raise ValueError(f"Failed to parse Ollama response: {e}")

    async def search_and_validate(self, query: str) -> dict:
        """Web search not available for local models."""
        return {"results": [], "message": "Web search not available for Ollama"}


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider with structured output support."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        api_key: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens, api_key)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key not provided")

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    async def parse_references(
        self,
        text: str,
        system_prompt: str,
    ) -> ReferenceListOutput:
        """Parse references using Google Gemini with JSON output."""
        import asyncio

        logger.info("parsing_with_google", model=self.model, text_len=len(text))

        # Include schema in prompt for better results with complex schemas
        schema = ReferenceListOutput.model_json_schema()
        prompt = f"""{system_prompt}

Output the results as JSON matching this schema:
{json.dumps(schema, indent=2)}

Text to parse:
{text}"""

        # Google genai uses sync API, run in executor for async compatibility
        def _generate():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    # "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                },
            )

            if not response.candidates:
                block_reason = getattr(response, "prompt_feedback", None)
                raise ValueError(f"No response candidates. Blocked: {block_reason}")

            candidate = response.candidates[0]
            if candidate.finish_reason:
                reason = getattr(candidate.finish_reason, "name", None)
                if reason == "SAFETY":
                    raise ValueError("Response blocked by safety filters")
                if reason == "MAX_TOKENS":
                    logger.warning("google_response_truncated")

            if not response.text:
                raise ValueError("Empty response from Google API")

            return response.text

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _generate)

        try:
            return ReferenceListOutput.model_validate_json(content)
        except Exception as e:
            content_preview = content[-500:] if content else "None"
            logger.error("google_parse_error", error=str(e), content=content_preview)
            raise ValueError(f"Failed to parse Google response: {e}")

    async def search_and_validate(self, query: str) -> dict:
        """
        Use Google Search grounding for validation.

        Google Gemini supports grounding with Google Search.
        """
        import asyncio

        logger.info("google_search", query=query)

        def _search():
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"Search for academic information about: {query}",
                config={
                    "tools": [{"google_search": {}}],
                },
            )
            return response.text

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _search)
            return {"results": [result], "message": "Search completed"}
        except Exception as e:
            logger.warning("google_search_error", error=str(e))
            return {"results": [], "message": f"Search failed: {e}"}


def get_provider(
    provider: LLMProvider | str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    api_key: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Factory function to get the appropriate LLM provider.

    Args:
        provider: Provider type or name
        model: Model name (uses defaults if not specified)
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        api_key: API key (uses environment if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLM provider instance
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    default_models = {
        LLMProvider.OPENAI: "gpt-5",
        LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
        LLMProvider.OLLAMA: "gemma3:27b",
        LLMProvider.GOOGLE: "gemini-2.5-flash",
    }

    model = model or default_models.get(provider, "gemini-2.5-flash")

    provider_classes = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.OLLAMA: OllamaProvider,
        LLMProvider.GOOGLE: GoogleProvider,
    }

    provider_class = provider_classes.get(provider)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider}")

    return provider_class(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs,
    )
