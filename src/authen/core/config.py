"""
Configuration management for the authen package.

Supports loading from environment variables, .env files, and programmatic configuration.
"""

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GOOGLE = "google"  # Google Gemini models


class Config(BaseSettings):
    """
    Main configuration for the authen pipeline.

    Configuration can be set via:
    - Environment variables
    - .env file in project root
    - Programmatic initialization
    """

    # LLM Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GOOGLE,
        description="LLM provider to use for reference parsing",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Model name/ID to use",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM sampling (0 for deterministic)",
    )
    llm_max_tokens: int = Field(
        default=16384,
        description="Maximum tokens for LLM response",
    )

    # API Keys (loaded from environment)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )
    google_api_key: str | None = Field(
        default=None,
        description="Google AI API key",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    # OpenAlex Configuration
    openalex_email: str = Field(
        default="user@example.com",
        description="Email for OpenAlex polite pool (10 req/sec)",
    )
    openalex_rate_limit: int = Field(
        default=10,
        description="Requests per second for OpenAlex API",
    )
    openalex_max_retries: int = Field(
        default=5,
        description="Maximum retries for failed API requests",
    )
    openalex_timeout: int = Field(
        default=30,
        description="Timeout in seconds for API requests",
    )

    # Validation Settings
    validation_title_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum title similarity for validation",
    )
    validation_author_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum author similarity for validation",
    )
    validation_batch_size: int = Field(
        default=50,
        description="Batch size for DOI lookups (max 50)",
    )

    # PDF Processing
    pdf_chunk_size: int = Field(
        default=50000,
        description="Maximum characters per chunk for LLM processing",
    )
    pdf_chunk_overlap: int = Field(
        default=1000,
        description="Overlap between chunks to avoid splitting references",
    )

    # Export Settings
    export_include_raw_openalex: bool = Field(
        default=False,
        description="Include raw OpenAlex response in export",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_api_key(self) -> str | None:
        """Get the appropriate API key for the configured provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key or os.getenv("OPENAI_API_KEY")
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        elif self.llm_provider == LLMProvider.GOOGLE:
            return self.google_api_key or os.getenv("GOOGLE_API_KEY")
        return None

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check API keys
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.get_api_key():
                issues.append("OpenAI API key not configured")
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.get_api_key():
                issues.append("Anthropic API key not configured")
        elif self.llm_provider == LLMProvider.GOOGLE:
            if not self.get_api_key():
                issues.append("Google API key not configured")

        # Check email for OpenAlex
        if self.openalex_email == "user@example.com":
            issues.append(
                "OpenAlex email not configured - using default will limit "
                "rate to 1 req/sec"
            )

        return issues


class PDFConfig(BaseModel):
    """Configuration specific to PDF extraction."""

    chunk_size: int = Field(default=50000, description="Max chars per chunk")
    chunk_overlap: int = Field(default=1000, description="Overlap between chunks")
    extract_images: bool = Field(default=False, description="Extract text from images")
    use_ocr: bool = Field(default=False, description="Use OCR for scanned PDFs")


class LLMConfig(BaseModel):
    """Configuration specific to LLM processing."""

    provider: LLMProvider = Field(default=LLMProvider.GOOGLE)
    model: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=16384)
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)


class ValidationConfig(BaseModel):
    """Configuration specific to OpenAlex validation."""

    email: str = Field(default="user@example.com")
    rate_limit: int = Field(default=10)
    max_retries: int = Field(default=5)
    timeout: int = Field(default=30)
    title_threshold: float = Field(default=0.85)
    author_threshold: float = Field(default=0.7)
    batch_size: int = Field(default=50)


def load_config(
    config_path: str | Path | None = None,
    **overrides,
) -> Config:
    """
    Load configuration from file and/or environment.

    Args:
        config_path: Optional path to .env file
        **overrides: Configuration values to override

    Returns:
        Configured Config instance
    """
    if config_path:
        os.environ["ENV_FILE"] = str(config_path)

    config = Config(**overrides)

    # Log any validation issues
    issues = config.validate_config()
    if issues:
        import structlog

        logger = structlog.get_logger()
        for issue in issues:
            logger.warning("config_issue", issue=issue)

    return config
