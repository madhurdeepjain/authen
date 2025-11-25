"""
Command-line interface for authen.

Provides commands for:
- Processing PDFs and text
- Validating existing reference lists
- Exporting results
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

import authen  # noqa: F401 - triggers .env loading
from authen.core.config import Config, LLMProvider

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="authen",
        description="Academic reference validation pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a PDF or text file",
    )
    process_parser.add_argument(
        "input",
        help="Input PDF file or text file",
    )
    process_parser.add_argument(
        "-o",
        "--output",
        help="Output file path",
        default="references.xlsx",
    )
    process_parser.add_argument(
        "--format",
        choices=["excel", "json", "csv"],
        default="excel",
        help="Output format",
    )
    process_parser.add_argument(
        "--provider",
        choices=["google", "openai", "anthropic", "ollama"],
        default="google",
        help="LLM provider",
    )
    process_parser.add_argument(
        "--model",
        help="LLM model name",
    )
    process_parser.add_argument(
        "--email",
        help="Email for OpenAlex polite pool",
    )
    process_parser.add_argument(
        "--api-key",
        help="API key for LLM provider",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate existing references JSON",
    )
    validate_parser.add_argument(
        "input",
        help="Input JSON file with references",
    )
    validate_parser.add_argument(
        "-o",
        "--output",
        help="Output file path",
        default="validated.xlsx",
    )
    validate_parser.add_argument(
        "--email",
        help="Email for OpenAlex polite pool",
    )

    # Parse command
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse references without validation",
    )
    parse_parser.add_argument(
        "input",
        help="Input PDF or text file",
    )
    parse_parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file",
        default="references.json",
    )
    parse_parser.add_argument(
        "--provider",
        choices=["google", "openai", "anthropic", "ollama"],
        default="google",
        help="LLM provider",
    )
    parse_parser.add_argument(
        "--model",
        help="LLM model name",
    )

    # Cache command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage the response cache",
    )
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")

    cache_stats_parser = cache_subparsers.add_parser(
        "stats",
        help="Show cache statistics",
    )
    cache_stats_parser.add_argument(
        "--db-path",
        help="Path to cache database",
        default=".authen_cache.db",
    )

    cache_clear_parser = cache_subparsers.add_parser(
        "clear",
        help="Clear cached data",
    )
    cache_clear_parser.add_argument(
        "--type",
        choices=["all", "llm", "openalex"],
        default="all",
        help="Type of cache to clear",
    )
    cache_clear_parser.add_argument(
        "--db-path",
        help="Path to cache database",
        default=".authen_cache.db",
    )

    cache_cleanup_parser = cache_subparsers.add_parser(
        "cleanup",
        help="Remove expired cache entries",
    )
    cache_cleanup_parser.add_argument(
        "--db-path",
        help="Path to cache database",
        default=".authen_cache.db",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == "process":
            asyncio.run(cmd_process(args))
        elif args.command == "validate":
            asyncio.run(cmd_validate(args))
        elif args.command == "parse":
            asyncio.run(cmd_parse(args))
        elif args.command == "cache":
            cmd_cache(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        logger.error("command_failed", error=str(e))
        sys.exit(1)


async def cmd_process(args):
    """Process a PDF or text file."""
    from authen.pipeline.orchestrator import Pipeline

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Build config
    config = Config(
        llm_provider=LLMProvider(args.provider),
        llm_model=args.model or get_default_model(args.provider),
        openalex_email=args.email or "user@example.com",
    )

    if args.api_key:
        if args.provider == "openai":
            config.openai_api_key = args.api_key
        elif args.provider == "anthropic":
            config.anthropic_api_key = args.api_key
        elif args.provider == "google":
            config.google_api_key = args.api_key

    # Process
    pipeline = Pipeline(config)

    if input_path.suffix.lower() == ".pdf":
        result = await pipeline.process(input_path)
    else:
        # Assume text file
        text = input_path.read_text()
        result = await pipeline.process_text(text)

    # Export
    output_path = args.output
    if args.format == "excel" and not output_path.endswith(".xlsx"):
        output_path += ".xlsx"

    pipeline.export(result, output_path, format=args.format)

    print(f"\n✅ Processed {result.total_references} references")
    print(f"   Validated: {result.validated_count}")
    print(f"   Partial:   {result.partial_match_count}")
    print(f"   Not found: {result.not_found_count}")
    print(f"   Errors:    {result.error_count}")
    print(f"\n📁 Output: {output_path}")


async def cmd_validate(args):
    """Validate existing references."""
    from authen.core.schemas import ReferenceData
    from authen.export.excel import export_to_excel
    from authen.validation import OpenAlexValidator

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load references
    with open(input_path) as f:
        data = json.load(f)

    # Handle both list of references and pipeline result format
    if isinstance(data, list):
        references = [ReferenceData(**r) for r in data]
    elif "references" in data:
        references = [ReferenceData(**r) for r in data["references"]]
    else:
        raise ValueError("Invalid input format")

    print(f"Loaded {len(references)} references")

    # Validate
    validator = OpenAlexValidator(
        email=args.email or "user@example.com",
    )

    try:
        results = await validator.validate(references)
    finally:
        await validator.close()

    # Export
    export_to_excel(results, args.output)

    validated = sum(1 for r in results if r.status.value == "validated")
    print(f"\n✅ Validated {validated}/{len(references)} references")
    print(f"📁 Output: {args.output}")


async def cmd_parse(args):
    """Parse references without validation."""
    from authen.llm import ReferenceParser, get_provider
    from authen.pdf import PDFExtractor

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get text
    if input_path.suffix.lower() == ".pdf":
        extractor = PDFExtractor()
        result = extractor.extract(input_path)
        text = result.text
    else:
        text = input_path.read_text()

    # Parse
    provider = get_provider(
        args.provider,
        model=args.model or get_default_model(args.provider),
    )
    parser = ReferenceParser(provider)

    result = await parser.parse(text)

    # Export to JSON
    output = {
        "references": [r.model_dump() for r in result.references],
        "model": result.model_used,
        "count": len(result.references),
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Parsed {len(result.references)} references")
    print(f"📁 Output: {args.output}")


def get_default_model(provider: str) -> str:
    """Get default model for provider."""
    defaults = {
        "openai": "gpt-5",
        "anthropic": "claude-sonnet-4-5-20250929",
        "google": "gemini-2.5-flash",
        "ollama": "gemma3:27b",
    }
    return defaults.get(provider, "gemini-2.5-flash")


def cmd_cache(args):
    """Manage the response cache."""
    from authen.core.cache import SQLiteCache

    db_path = getattr(args, "db_path", ".authen_cache.db")

    if args.cache_command == "stats":
        cache = SQLiteCache(db_path=db_path)
        try:
            stats = cache.get_stats()
            print("\n📊 Cache Statistics")
            print(f"   Database: {db_path}")
            print(f"   Total entries: {stats.get('total_entries', 0) or 0}")
            print(f"   LLM entries: {stats.get('llm_entries', 0) or 0}")
            print(f"   OpenAlex entries: {stats.get('openalex_entries', 0) or 0}")
            print(f"   Expired entries: {stats.get('expired_entries', 0) or 0}")
        finally:
            cache.close()

    elif args.cache_command == "clear":
        cache = SQLiteCache(db_path=db_path)
        try:
            cache_type = args.type if args.type != "all" else None
            cache.clear(cache_type=cache_type)
            type_str = f"{args.type} " if args.type != "all" else ""
            print(f"✅ Cleared {type_str}cache")
        finally:
            cache.close()

    elif args.cache_command == "cleanup":
        cache = SQLiteCache(db_path=db_path)
        try:
            count = cache.cleanup_expired()
            print(f"✅ Removed {count} expired entries")
        finally:
            cache.close()

    else:
        print("Usage: authen cache [stats|clear|cleanup]")
        sys.exit(1)


if __name__ == "__main__":
    main()
