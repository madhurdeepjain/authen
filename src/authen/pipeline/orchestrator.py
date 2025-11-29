"""
Pipeline orchestrator for the complete reference validation workflow.

Coordinates:
1. PDF extraction
2. LLM reference parsing
3. OpenAlex validation
4. Export to Excel/JSON

Supports two execution modes:
1. Sequential: Process all chunks, then all validations
2. Streaming: Parse chunks in parallel, validate as references arrive
"""

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import structlog

from authen.core.cache import CacheManager, SQLiteCache
from authen.core.config import Config
from authen.core.schemas import (
    ExtractionResult,
    ParseResult,
    PipelineEvent,
    PipelineEventType,
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

    By default, uses streaming parallel processing:
    - Chunks are parsed by LLM in parallel (up to max_concurrent_chunks)
    - Validation starts as soon as first references are available
    - Deduplication happens on-the-fly

    Usage:
        config = Config(...)
        pipeline = Pipeline(config)

        # Process a PDF (parallel by default)
        result = await pipeline.process("paper.pdf")

        # Force sequential processing (for debugging)
        result = await pipeline.process("paper.pdf", streaming=False)

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

        # Initialize cache if enabled
        self._cache: CacheManager | None = None
        if config.cache_enabled:
            db_path = config.cache_db_path or ".authen_cache.db"
            backend = SQLiteCache(
                db_path=db_path,
                default_ttl=config.cache_openalex_ttl,
            )
            self._cache = CacheManager(
                backend=backend,
                enabled=True,
                llm_ttl=config.cache_llm_ttl,
                openalex_ttl=config.cache_openalex_ttl,
            )
            logger.info(
                "cache_initialized",
                db_path=db_path,
                llm_ttl=config.cache_llm_ttl,
                openalex_ttl=config.cache_openalex_ttl,
            )

        # Initialize components
        self.pdf_extractor = PDFExtractor(
            chunk_size=config.pdf_chunk_size,
            chunk_overlap=config.pdf_chunk_overlap,
            x_tolerance=config.pdf_x_tolerance,
        )

        self.llm_provider = get_provider(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            api_key=config.get_api_key(),
            cache=self._cache,
            enable_cache=config.cache_enabled,
        )

        self.reference_parser = ReferenceParser(
            provider=self.llm_provider,
            chunk_size=config.pdf_chunk_size,
            max_concurrent_chunks=config.max_concurrent_chunks,
        )

        self.validator = OpenAlexValidator(
            email=config.openalex_email,
            rate_limit=config.openalex_rate_limit,
            title_threshold=config.validation_title_threshold,
            author_threshold=config.validation_author_threshold,
            max_retries=config.openalex_max_retries,
            timeout=config.openalex_timeout,
            cache=self._cache,
            enable_cache=config.cache_enabled,
        )

        self.exporter = ExcelExporter(
            include_raw_openalex=config.export_include_raw_openalex,
        )

    @property
    def cache(self) -> CacheManager | None:
        """Get the cache manager."""
        return self._cache

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return {"enabled": False}

    async def process(
        self,
        source: str | Path,
        extract_references_only: bool = True,
        streaming: bool = True,
    ) -> PipelineResult:
        """
        Process a PDF file through the complete pipeline.

        By default uses streaming parallel processing for better performance.

        Args:
            source: Path to PDF file
            extract_references_only: Try to extract just references section
            streaming: Use parallel streaming mode (default: True)

        Returns:
            Complete pipeline result
        """
        if streaming:
            return await self._process_streaming(source, extract_references_only)
        return await self._process_sequential(source, extract_references_only)

    async def _process_sequential(
        self,
        source: str | Path,
        extract_references_only: bool = True,
    ) -> PipelineResult:
        """
        Process a PDF file sequentially (parse all, then validate all).

        Used internally when streaming=False is passed to process().
        """
        start_time = time.time()
        source = Path(source)

        logger.info("pipeline_sequential_start", source=str(source))

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
            "pipeline_sequential_complete",
            total=result.total_references,
            validated=result.validated_count,
            time_seconds=round(result.total_processing_time_seconds, 2),
        )

        return result

    async def _process_streaming(
        self,
        source: str | Path,
        extract_references_only: bool = True,
    ) -> PipelineResult:
        """
        Process a PDF file with streaming parallel pipeline.

        Used internally as the default mode for process().
        """
        start_time = time.time()
        source = Path(source)
        max_concurrent = self.config.max_concurrent_validations

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

            # Step 2 & 3: Stream parse and validate in parallel
            logger.info("step_2_3_streaming_parse_and_validate")

            # Create async generator from parser
            reference_stream = self.reference_parser.parse_streaming(text)

            # Stream to validator and collect results
            validation_results: list[ValidationResult] = []
            parsed_references: list[ReferenceData] = []

            # We need to collect parsed refs while streaming to validator
            async def tracked_stream() -> AsyncIterator[ReferenceData]:
                async for ref in reference_stream:
                    parsed_references.append(ref)
                    yield ref

            async for validation_result in self.validator.validate_streaming(
                tracked_stream(),
                max_concurrent=max_concurrent,
            ):
                validation_results.append(validation_result)

            # Build parse result from collected references
            result.parse_result = ParseResult(
                references=parsed_references,
                source_text=text[:1000],
                model_used=self.llm_provider.model,
                processing_time_seconds=time.time() - start_time,
            )

            result.validation_results = validation_results

            # Compute statistics
            result.compute_stats()

            if not parsed_references:
                logger.warning("no_references_found")
                result.total_references = 0

        except Exception as e:
            logger.error("pipeline_streaming_error", error=str(e))
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

    async def process_text(
        self,
        text: str,
        streaming: bool = True,
    ) -> PipelineResult:
        """
        Process text containing references through the pipeline.

        By default uses streaming parallel processing.

        Args:
            text: Text containing references
            streaming: Use parallel streaming mode (default: True)

        Returns:
            Pipeline result
        """
        if streaming:
            return await self._process_text_streaming(text)
        return await self._process_text_sequential(text)

    async def _process_text_sequential(self, text: str) -> PipelineResult:
        """Process text sequentially (parse all, then validate all)."""
        start_time = time.time()

        logger.info("pipeline_text_sequential_start", text_length=len(text))

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
            "pipeline_text_sequential_complete",
            total=result.total_references,
            validated=result.validated_count,
            time_seconds=round(result.total_processing_time_seconds, 2),
        )

        return result

    async def _process_text_streaming(self, text: str) -> PipelineResult:
        """Process text with streaming parallel pipeline."""
        start_time = time.time()
        max_concurrent = self.config.max_concurrent_validations

        logger.info("pipeline_text_start", text_length=len(text))

        result = PipelineResult()

        try:
            # Create extraction result for text input
            result.extraction = ExtractionResult(
                text=text,
                source_file="text_input",
            )

            # Stream parse and validate in parallel
            logger.info("streaming_parse_and_validate")

            reference_stream = self.reference_parser.parse_streaming(text)

            validation_results: list[ValidationResult] = []
            parsed_references: list[ReferenceData] = []

            async def tracked_stream() -> AsyncIterator[ReferenceData]:
                async for ref in reference_stream:
                    parsed_references.append(ref)
                    yield ref

            async for validation_result in self.validator.validate_streaming(
                tracked_stream(),
                max_concurrent=max_concurrent,
            ):
                validation_results.append(validation_result)

            result.parse_result = ParseResult(
                references=parsed_references,
                source_text=text[:1000],
                model_used=self.llm_provider.model,
                processing_time_seconds=time.time() - start_time,
            )

            result.validation_results = validation_results
            result.compute_stats()

            if not parsed_references:
                logger.warning("no_references_found")

        except Exception as e:
            logger.error("pipeline_text_streaming_error", error=str(e))
            raise

        finally:
            await self.validator.close()

        result.total_processing_time_seconds = time.time() - start_time

        logger.info(
            "pipeline_text_complete",
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
            from authen.export.json import export_to_json

            return export_to_json(result, output_path)
        elif format == "csv":
            from authen.export.csv import export_to_csv

            return export_to_csv(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def process_events(self, pdf_path: str) -> AsyncIterator[PipelineEvent]:
        """
        Process a PDF and yield events for progress tracking.

        Args:
            pdf_path: Path to the PDF file

        Yields:
            PipelineEvent objects
        """
        queue = asyncio.Queue()

        async def producer():
            start_time = time.time()
            max_concurrent = self.config.max_concurrent_validations

            logger.info("pipeline_events_start", source=pdf_path)

            result = PipelineResult()

            try:
                logger.info("step_1_extraction")
                extraction_result = self.pdf_extractor.extract(pdf_path)
                result.extraction = extraction_result
                text = extraction_result.text

                await queue.put(
                    PipelineEvent(
                        type=PipelineEventType.EXTRACTION_COMPLETE,
                        data=extraction_result,
                        message=f"Extracted {len(text)} characters",
                    )
                )

                logger.info("step_2_3_streaming_parse_and_validate")

                async def on_chunk_progress(current: int, total: int):
                    await queue.put(
                        PipelineEvent(
                            type=PipelineEventType.CHUNK_PROCESSED,
                            data={"current": current, "total": total},
                            message=f"Processed chunk {current}/{total}",
                        )
                    )

                reference_stream = self.reference_parser.parse_streaming(
                    text, on_chunk_complete=on_chunk_progress
                )

                validation_results: list[ValidationResult] = []
                parsed_references: list[ReferenceData] = []

                async def tracked_stream() -> AsyncIterator[ReferenceData]:
                    async for ref in reference_stream:
                        parsed_references.append(ref)
                        await queue.put(
                            PipelineEvent(
                                type=PipelineEventType.REFERENCE_FOUND,
                                data=ref,
                                message=f"Found reference: {ref.title[:50] if ref.title else 'Unknown'}...",
                            )
                        )
                        yield ref

                async for validation_result in self.validator.validate_streaming(
                    tracked_stream(),
                    max_concurrent=max_concurrent,
                ):
                    validation_results.append(validation_result)
                    await queue.put(
                        PipelineEvent(
                            type=PipelineEventType.VALIDATION_COMPLETE,
                            data=validation_result,
                            message=f"Validated: {validation_result.get_best_reference().title[:50] if validation_result.get_best_reference().title else 'Unknown'}...",
                        )
                    )

                # Build final result
                result.parse_result = ParseResult(
                    references=parsed_references,
                    source_text=text[:1000],
                    model_used=self.llm_provider.model,
                    processing_time_seconds=time.time() - start_time,
                )
                result.validation_results = validation_results
                result.compute_stats()
                result.total_processing_time_seconds = time.time() - start_time

                await queue.put(
                    PipelineEvent(
                        type=PipelineEventType.COMPLETED,
                        data=result,
                        message=f"Completed! Processed {result.total_references} references.",
                    )
                )

            except Exception as e:
                logger.error("pipeline_events_error", error=str(e))
                await queue.put(
                    PipelineEvent(
                        type=PipelineEventType.ERROR,
                        message=str(e),
                    )
                )
            finally:
                await self.validator.close()
                await queue.put(None)  # Sentinel

        # Start producer
        asyncio.create_task(producer())

        # Consume queue
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    async def process_text_events(
        self,
        text: str,
    ) -> AsyncIterator[PipelineEvent]:
        """
        Process text and yield events for progress tracking.

        Args:
            text: Text containing references

        Yields:
            PipelineEvent objects
        """
        queue = asyncio.Queue()

        async def producer():
            start_time = time.time()
            max_concurrent = self.config.max_concurrent_validations

            logger.info("pipeline_text_events_start", text_length=len(text))

            result = PipelineResult()

            try:
                result.extraction = ExtractionResult(
                    text=text,
                    source_file="text_input",
                )

                await queue.put(
                    PipelineEvent(
                        type=PipelineEventType.EXTRACTION_COMPLETE,
                        data=text,
                        message="Text received",
                    )
                )

                logger.info("streaming_parse_and_validate")

                async def on_chunk_progress(current: int, total: int):
                    await queue.put(
                        PipelineEvent(
                            type=PipelineEventType.CHUNK_PROCESSED,
                            data={"current": current, "total": total},
                            message=f"Processed chunk {current}/{total}",
                        )
                    )

                reference_stream = self.reference_parser.parse_streaming(
                    text, on_chunk_complete=on_chunk_progress
                )

                validation_results: list[ValidationResult] = []
                parsed_references: list[ReferenceData] = []

                async def tracked_stream() -> AsyncIterator[ReferenceData]:
                    async for ref in reference_stream:
                        parsed_references.append(ref)
                        await queue.put(
                            PipelineEvent(
                                type=PipelineEventType.REFERENCE_FOUND,
                                data=ref,
                                message=f"Found reference: {ref.title[:50] if ref.title else 'Unknown'}...",
                            )
                        )
                        yield ref

                async for validation_result in self.validator.validate_streaming(
                    tracked_stream(),
                    max_concurrent=max_concurrent,
                ):
                    validation_results.append(validation_result)
                    await queue.put(
                        PipelineEvent(
                            type=PipelineEventType.VALIDATION_COMPLETE,
                            data=validation_result,
                            message=f"Validated: {validation_result.get_best_reference().title[:50] if validation_result.get_best_reference().title else 'Unknown'}...",
                        )
                    )

                result.parse_result = ParseResult(
                    references=parsed_references,
                    source_text=text[:1000],
                    model_used=self.llm_provider.model,
                    processing_time_seconds=time.time() - start_time,
                )
                result.validation_results = validation_results
                result.compute_stats()
                result.total_processing_time_seconds = time.time() - start_time

                await queue.put(
                    PipelineEvent(
                        type=PipelineEventType.COMPLETED,
                        data=result,
                        message=f"Completed! Processed {result.total_references} references.",
                    )
                )

            except Exception as e:
                logger.error("pipeline_text_events_error", error=str(e))
                await queue.put(
                    PipelineEvent(
                        type=PipelineEventType.ERROR,
                        message=str(e),
                    )
                )
            finally:
                await self.validator.close()
                await queue.put(None)  # Sentinel

        # Start producer
        asyncio.create_task(producer())

        # Consume queue
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event


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
