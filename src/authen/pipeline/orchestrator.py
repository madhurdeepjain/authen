"""
Pipeline orchestrator for the complete reference validation workflow.

Coordinates:
1. PDF extraction
2. LLM reference parsing
3. OpenAlex validation
4. Export to Excel/JSON
"""

import time
from pathlib import Path

import structlog

from authen.core.config import Config
from authen.core.schemas import (
    ExtractionResult,
    ParseResult,
    PipelineResult,
    ReferenceData,
    ValidationResult,
)
from authen.export.excel import ExcelExporter
from authen.llm import ReferenceParser, get_provider
from authen.pdf import PDFExtractor
from authen.validation import OpenAlexValidator

logger = structlog.get_logger()


class Pipeline:
    """
    Main pipeline orchestrator for reference validation.

    Usage:
        config = Config(...)
        pipeline = Pipeline(config)

        # Process a PDF
        result = await pipeline.process("paper.pdf")

        # Export results
        pipeline.export(result, "references.xlsx")
    """

    def __init__(self, config: Config):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize components
        self.pdf_extractor = PDFExtractor(
            chunk_size=config.pdf_chunk_size,
            chunk_overlap=config.pdf_chunk_overlap,
        )

        self.llm_provider = get_provider(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            api_key=config.get_api_key(),
        )

        self.reference_parser = ReferenceParser(
            provider=self.llm_provider,
            chunk_size=config.pdf_chunk_size,
        )

        self.validator = OpenAlexValidator(
            email=config.openalex_email,
            rate_limit=config.openalex_rate_limit,
            title_threshold=config.validation_title_threshold,
            author_threshold=config.validation_author_threshold,
            max_retries=config.openalex_max_retries,
            timeout=config.openalex_timeout,
        )

        self.exporter = ExcelExporter(
            include_raw_openalex=config.export_include_raw_openalex,
        )

    async def process(
        self,
        source: str | Path,
        extract_references_only: bool = True,
    ) -> PipelineResult:
        """
        Process a PDF file through the complete pipeline.

        Args:
            source: Path to PDF file
            extract_references_only: Try to extract just references section

        Returns:
            Complete pipeline result
        """
        start_time = time.time()
        source = Path(source)

        logger.info("pipeline_start", source=str(source))

        result = PipelineResult(source_file=str(source))

        try:
            # Step 1: Extract text from PDF
            logger.info("step_1_extraction")
            extraction = self.pdf_extractor.extract(source)
            result.extraction = extraction

            # Try to get just the references section
            text = extraction.text
            if extract_references_only:
                refs_section = self.pdf_extractor.extract_references_section(text)
                if refs_section:
                    text = refs_section
                    logger.info(
                        "using_references_section",
                        full_length=len(extraction.text),
                        refs_length=len(refs_section),
                    )

            # Step 2: Parse references with LLM
            logger.info("step_2_parsing")
            parse_result = await self.reference_parser.parse(text)
            result.parse_result = parse_result

            if not parse_result.references:
                logger.warning("no_references_found")
                result.total_references = 0
                return result

            # Step 3: Validate with OpenAlex
            logger.info("step_3_validation", count=len(parse_result.references))
            validation_results = await self.validator.validate(parse_result.references)
            result.validation_results = validation_results

            # Compute statistics
            result.compute_stats()

        except Exception as e:
            logger.error("pipeline_error", error=str(e))
            raise

        finally:
            # Cleanup
            await self.validator.close()

        result.total_processing_time_seconds = time.time() - start_time

        logger.info(
            "pipeline_complete",
            total=result.total_references,
            validated=result.validated_count,
            time_seconds=round(result.total_processing_time_seconds, 2),
        )

        return result

    async def process_text(self, text: str) -> PipelineResult:
        """
        Process text containing references through the pipeline.

        Args:
            text: Text containing references

        Returns:
            Pipeline result
        """
        start_time = time.time()

        logger.info("pipeline_text_start", text_length=len(text))

        result = PipelineResult()

        try:
            # Create extraction result for text input
            result.extraction = ExtractionResult(
                text=text,
                source_file="text_input",
            )

            # Step 1: Parse references with LLM
            logger.info("step_1_parsing")
            parse_result = await self.reference_parser.parse(text)
            result.parse_result = parse_result

            if not parse_result.references:
                logger.warning("no_references_found")
                result.total_references = 0
                return result

            # Step 2: Validate with OpenAlex
            logger.info("step_2_validation", count=len(parse_result.references))
            validation_results = await self.validator.validate(parse_result.references)
            result.validation_results = validation_results

            # Compute statistics
            result.compute_stats()

        except Exception as e:
            logger.error("pipeline_error", error=str(e))
            raise

        finally:
            await self.validator.close()

        result.total_processing_time_seconds = time.time() - start_time

        logger.info(
            "pipeline_complete",
            total=result.total_references,
            validated=result.validated_count,
            time_seconds=round(result.total_processing_time_seconds, 2),
        )

        return result

    async def parse_only(
        self,
        source: str | Path | None = None,
        text: str | None = None,
    ) -> ParseResult:
        """
        Run only the parsing step (no validation).

        Args:
            source: Path to PDF file
            text: Text to parse (if no source)

        Returns:
            Parse result with references
        """
        if source:
            extraction = self.pdf_extractor.extract(source)
            text = extraction.text

        if not text:
            raise ValueError("Either source or text must be provided")

        return await self.reference_parser.parse(text)

    async def validate_only(
        self,
        references: list[ReferenceData],
    ) -> list[ValidationResult]:
        """
        Run only the validation step.

        Args:
            references: References to validate

        Returns:
            Validation results
        """
        try:
            return await self.validator.validate(references)
        finally:
            await self.validator.close()

    def export(
        self,
        result: PipelineResult | list[ValidationResult],
        output_path: str | Path,
        format: str = "excel",
    ) -> Path:
        """
        Export results to file.

        Args:
            result: Pipeline or validation results
            output_path: Output file path
            format: Export format (excel, json, csv)

        Returns:
            Path to created file
        """
        output_path = Path(output_path)

        if format == "excel":
            return self.exporter.export(result, output_path)
        elif format == "json":
            from authen.export.excel import export_to_json

            return export_to_json(result, output_path)
        elif format == "csv":
            from authen.export.excel import export_to_csv

            return export_to_csv(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


async def process_pdf(
    pdf_path: str | Path,
    config: Config | None = None,
    output_path: str | Path | None = None,
    **config_kwargs,
) -> PipelineResult:
    """
    Convenience function to process a PDF through the pipeline.

    Args:
        pdf_path: Path to PDF file
        config: Optional configuration
        output_path: Optional output path for export
        **config_kwargs: Configuration overrides

    Returns:
        Pipeline result
    """
    if config is None:
        config = Config(**config_kwargs)

    pipeline = Pipeline(config)
    result = await pipeline.process(pdf_path)

    if output_path:
        pipeline.export(result, output_path)

    return result


async def process_text(
    text: str,
    config: Config | None = None,
    output_path: str | Path | None = None,
    **config_kwargs,
) -> PipelineResult:
    """
    Convenience function to process text through the pipeline.

    Args:
        text: Text containing references
        config: Optional configuration
        output_path: Optional output path for export
        **config_kwargs: Configuration overrides

    Returns:
        Pipeline result
    """
    if config is None:
        config = Config(**config_kwargs)

    pipeline = Pipeline(config)
    result = await pipeline.process_text(text)

    if output_path:
        pipeline.export(result, output_path)

    return result
