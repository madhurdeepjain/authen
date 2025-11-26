"""
Reference parser using LLMs with structured output.

Handles parsing of reference text into structured data,
including support for chunked processing of large documents.

Supports two modes:
1. Sequential: Parse all chunks, deduplicate, return result
2. Streaming: Parse chunks in parallel, yield references as they become available
"""

import asyncio
import time
from collections.abc import AsyncIterator

import structlog

from authen.core.schemas import (
    Affiliation,
    Author,
    ParseResult,
    ReferenceData,
)
from authen.llm.providers import BaseLLMProvider
from authen.llm.schemas import LLMReference, ReferenceListOutput

logger = structlog.get_logger()


SYSTEM_PROMPT = """You are an expert academic reference parser.
Your task is to extract structured bibliographic information from reference text.

INSTRUCTIONS:
1. Parse each reference carefully, extracting all available fields
2. For author names, split into first_name and last_name when possible
3. Extract DOIs in their standard format (10.xxxx/yyyy)
4. Recognize common reference formats: APA, MLA, Chicago, IEEE, Vancouver
5. If information is ambiguous or unclear, make reasonable inferences
6. Preserve the original reference number if present
7. Include the raw_text for each reference

WORK TYPES:
- article: Journal article
- book: Complete book
- book-chapter: Chapter in a book
- conference-paper: Conference proceeding
- thesis: PhD or Master's thesis
- preprint: Unpublished preprint
- dataset: Research dataset
- report: Technical report
- other: Other types

Return a structured JSON with all extracted references."""


class ReferenceParser:
    """
    Parse references from text using LLM structured output.

    This class is designed to be used standalone - just instantiate
    with a provider and call parse() with your text.

    Supports both sequential and streaming modes:
    - parse(): Sequential mode, returns all references after processing
    - parse_streaming(): Streaming mode, yields references as chunks complete
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        chunk_size: int = 30000,
        chunk_overlap: int = 500,
        max_concurrent_chunks: int = 3,
    ):
        """
        Initialize the reference parser.

        Args:
            provider: LLM provider instance
            chunk_size: Maximum characters per chunk for LLM processing
            chunk_overlap: Overlap between chunks
            max_concurrent_chunks: Maximum chunks to process in parallel
        """
        self.provider = provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_concurrent_chunks = max_concurrent_chunks
        self.system_prompt = SYSTEM_PROMPT

    async def parse(
        self,
        text: str,
        source_text: str | None = None,
    ) -> ParseResult:
        """
        Parse references from text.

        Args:
            text: Text containing references to parse
            source_text: Optional original source text for tracking

        Returns:
            ParseResult with extracted references
        """
        start_time = time.time()
        logger.info(
            "starting_reference_parsing",
            text_len=len(text),
            provider=self.provider.provider_name,
        )

        # Check if we need to chunk
        if len(text) > self.chunk_size:
            references = await self._parse_chunked(text)
        else:
            result = await self.provider.parse_references(text, self.system_prompt)
            references = self._convert_references(result)

        # Deduplicate references
        references = self._deduplicate_references(references)

        processing_time = time.time() - start_time

        logger.info(
            "parsing_complete",
            references_found=len(references),
            time_seconds=round(processing_time, 2),
        )

        return ParseResult(
            references=references,
            source_text=source_text or text[:1000],  # Store first 1000 chars
            model_used=self.provider.model,
            processing_time_seconds=processing_time,
        )

    async def parse_streaming(
        self,
        text: str,
        on_chunk_complete: callable = None,
    ) -> AsyncIterator[ReferenceData]:
        """
        Parse references from text in streaming mode.

        Chunks are processed in parallel (up to max_concurrent_chunks),
        and references are yielded as soon as each chunk completes.
        Deduplication happens on-the-fly.

        Args:
            text: Text containing references to parse

        Yields:
            ReferenceData objects as they become available
        """
        logger.info(
            "starting_streaming_parse",
            text_len=len(text),
            provider=self.provider.provider_name,
        )

        # Track seen references for streaming deduplication
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()

        if len(text) <= self.chunk_size:
            # Single chunk - just parse and yield
            result = await self.provider.parse_references(text, self.system_prompt)
            references = self._convert_references(result)
            
            # Report progress for single chunk
            if on_chunk_complete:
                if asyncio.iscoroutinefunction(on_chunk_complete):
                    await on_chunk_complete(1, 1)
                else:
                    on_chunk_complete(1, 1)

            for ref in references:
                if self._is_duplicate(ref, seen_dois, seen_titles):
                    continue
                self._mark_seen(ref, seen_dois, seen_titles)
                yield ref
            return

        # Multiple chunks - process in parallel with semaphore
        chunks = self._create_chunks(text)
        logger.info("streaming_parse_chunks", chunk_count=len(chunks))

        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)
        output_queue: asyncio.Queue[ReferenceData | None] = asyncio.Queue()

        # Track completed chunks for progress reporting
        completed_chunks = 0
        progress_lock = asyncio.Lock()

        async def process_chunk(chunk_num: int, chunk: str) -> None:
            """Process a single chunk and put results in queue."""
            nonlocal completed_chunks
            async with semaphore:
                logger.info("parsing_chunk", chunk_num=chunk_num + 1, total=len(chunks))
                try:
                    result = await self.provider.parse_references(
                        chunk, self.system_prompt
                    )
                    chunk_refs = self._convert_references(result)
                    for ref in chunk_refs:
                        await output_queue.put(ref)
                except Exception as e:
                    logger.error(
                        "chunk_parsing_error",
                        chunk_num=chunk_num + 1,
                        error=str(e),
                    )
                finally:
                    if on_chunk_complete:
                        async with progress_lock:
                            completed_chunks += 1
                            current_progress = completed_chunks
                        
                        if asyncio.iscoroutinefunction(on_chunk_complete):
                            await on_chunk_complete(current_progress, len(chunks))
                        else:
                            on_chunk_complete(current_progress, len(chunks))

        async def run_all_chunks() -> None:
            """Run all chunk processing tasks and signal completion."""
            tasks = [
                asyncio.create_task(process_chunk(i, chunk))
                for i, chunk in enumerate(chunks)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            await output_queue.put(None)  # Signal completion

        # Start chunk processing in background
        chunk_task = asyncio.create_task(run_all_chunks())

        # Yield references as they arrive, with deduplication
        try:
            while True:
                ref = await output_queue.get()
                if ref is None:
                    break
                if self._is_duplicate(ref, seen_dois, seen_titles):
                    continue
                self._mark_seen(ref, seen_dois, seen_titles)
                yield ref
        finally:
            # Ensure chunk task completes
            await chunk_task

        logger.info(
            "streaming_parse_complete",
            total_yielded=len(seen_dois) + len(seen_titles),
        )

    def _is_duplicate(
        self,
        ref: ReferenceData,
        seen_dois: set[str],
        seen_titles: set[str],
    ) -> bool:
        """Check if a reference is a duplicate."""
        if ref.doi and ref.doi in seen_dois:
            return True
        if ref.title:
            norm_title = ref.title.lower().strip()
            if norm_title in seen_titles:
                return True
        return False

    def _mark_seen(
        self,
        ref: ReferenceData,
        seen_dois: set[str],
        seen_titles: set[str],
    ) -> None:
        """Mark a reference as seen for deduplication."""
        if ref.doi:
            seen_dois.add(ref.doi)
        if ref.title:
            seen_titles.add(ref.title.lower().strip())

    async def _parse_chunked(self, text: str) -> list[ReferenceData]:
        """Parse text in chunks with parallel processing and combine results."""
        chunks = self._create_chunks(text)
        logger.info("parsing_in_chunks", chunk_count=len(chunks))

        all_references: list[ReferenceData] = []
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)

        async def process_chunk(chunk_num: int, chunk: str) -> list[ReferenceData]:
            """Process a single chunk with semaphore."""
            async with semaphore:
                logger.info("parsing_chunk", chunk_num=chunk_num + 1, total=len(chunks))
                try:
                    result = await self.provider.parse_references(
                        chunk, self.system_prompt
                    )
                    return self._convert_references(result)
                except Exception as e:
                    logger.error(
                        "chunk_parsing_error",
                        chunk_num=chunk_num + 1,
                        error=str(e),
                    )
                    return []

        # Process all chunks in parallel (limited by semaphore)
        tasks = [
            asyncio.create_task(process_chunk(i, chunk))
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks)

        for chunk_refs in results:
            all_references.extend(chunk_refs)

        return all_references

    def _create_chunks(self, text: str) -> list[str]:
        """Split text into chunks for processing."""
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + self.chunk_size, len(text))

            # If this is the last chunk, just take it
            if end_pos >= len(text):
                chunk = text[current_pos:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find a good break point
            for sep in ["\n\n", "\n", ". "]:
                break_pos = text.rfind(sep, current_pos, end_pos)
                if break_pos > current_pos + self.chunk_size // 2:
                    end_pos = break_pos + len(sep)
                    break

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Ensure we always advance
            next_pos = end_pos - self.chunk_overlap
            if next_pos <= current_pos:
                next_pos = end_pos
            current_pos = next_pos

        return chunks

    def _convert_references(self, result: ReferenceListOutput) -> list[ReferenceData]:
        """Convert LLM output to core schemas."""
        references = []

        for llm_ref in result.references:
            ref = self._convert_single_reference(llm_ref)
            if ref:
                references.append(ref)

        return references

    def _convert_single_reference(self, llm_ref: LLMReference) -> ReferenceData | None:
        """Convert a single LLM reference to core schema."""
        try:
            authors = []
            for llm_author in llm_ref.authors:
                affiliations = [
                    Affiliation(
                        name=aff.name,
                        department=aff.department,
                        country=aff.country,
                        city=aff.city,
                    )
                    for aff in llm_author.affiliations
                ]

                author = Author(
                    first_name=llm_author.first_name,
                    last_name=llm_author.last_name,
                    full_name=llm_author.full_name,
                    affiliations=affiliations,
                    emails=[llm_author.email] if llm_author.email else [],
                )
                authors.append(author)

            return ReferenceData(
                raw_text=llm_ref.raw_text,
                reference_number=llm_ref.reference_number,
                title=llm_ref.title,
                authors=authors,
                year=llm_ref.year,
                publication=llm_ref.publication,
                publisher=llm_ref.publisher,
                volume=llm_ref.volume,
                issue=llm_ref.issue,
                pages=llm_ref.pages,
                doi=self._normalize_doi(llm_ref.doi),
                isbn=llm_ref.isbn,
                pmid=llm_ref.pmid,
                arxiv_id=llm_ref.arxiv_id,
                url=llm_ref.url,
                work_type=llm_ref.work_type,
            )
        except Exception as e:
            logger.warning(
                "reference_conversion_error",
                error=str(e),
                title=llm_ref.title,
            )
            return None

    def _normalize_doi(self, doi: str | None) -> str | None:
        """Normalize DOI format."""
        if not doi:
            return None

        # Remove common prefixes
        doi = doi.strip()
        prefixes = [
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
            "doi:",
            "DOI:",
        ]
        for prefix in prefixes:
            if doi.startswith(prefix):
                doi = doi[len(prefix) :]
                break

        # Validate DOI format (starts with 10.)
        if doi.startswith("10."):
            return doi

        return None

    def _deduplicate_references(
        self, references: list[ReferenceData]
    ) -> list[ReferenceData]:
        """Remove duplicate references based on DOI or title similarity."""
        seen_dois = set()
        seen_titles = set()
        unique_refs = []

        for ref in references:
            # Check by DOI first
            if ref.doi:
                if ref.doi in seen_dois:
                    continue
                seen_dois.add(ref.doi)
                unique_refs.append(ref)
                continue

            # Check by normalized title
            if ref.title:
                norm_title = ref.title.lower().strip()
                if norm_title in seen_titles:
                    continue
                seen_titles.add(norm_title)

            unique_refs.append(ref)

        return unique_refs


async def parse_text(
    text: str,
    provider: BaseLLMProvider | None = None,
    provider_name: str = "openai",
    model: str | None = None,
    **kwargs,
) -> list[ReferenceData]:
    """
    Convenience function to parse references from text.

    Args:
        text: Text containing references
        provider: LLM provider instance (optional)
        provider_name: Provider name if not passing provider instance
        model: Model name (optional)
        **kwargs: Additional provider arguments

    Returns:
        List of parsed references
    """
    if provider is None:
        from authen.llm.providers import get_provider

        provider = get_provider(provider_name, model=model, **kwargs)

    parser = ReferenceParser(provider)
    result = await parser.parse(text)
    return result.references
