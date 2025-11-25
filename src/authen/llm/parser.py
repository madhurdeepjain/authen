"""
Reference parser using LLMs with structured output.

Handles parsing of reference text into structured data,
including support for chunked processing of large documents.
"""

import time

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
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        chunk_size: int = 30000,
        chunk_overlap: int = 500,
    ):
        """
        Initialize the reference parser.

        Args:
            provider: LLM provider instance
            chunk_size: Maximum characters per chunk for LLM processing
            chunk_overlap: Overlap between chunks
        """
        self.provider = provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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

    async def _parse_chunked(self, text: str) -> list[ReferenceData]:
        """Parse text in chunks and combine results."""
        chunks = self._create_chunks(text)
        logger.info("parsing_in_chunks", chunk_count=len(chunks))

        all_references = []

        for i, chunk in enumerate(chunks):
            logger.info("parsing_chunk", chunk_num=i + 1, total=len(chunks))
            try:
                result = await self.provider.parse_references(chunk, self.system_prompt)
                chunk_refs = self._convert_references(result)
                all_references.extend(chunk_refs)
            except Exception as e:
                logger.error(
                    "chunk_parsing_error",
                    chunk_num=i + 1,
                    error=str(e),
                )
                # Continue with other chunks
                continue

        return all_references

    def _create_chunks(self, text: str) -> list[str]:
        """Split text into chunks for processing."""
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + self.chunk_size, len(text))

            if end_pos < len(text):
                # Find a good break point
                for sep in ["\n\n", "\n", ". "]:
                    break_pos = text.rfind(sep, current_pos, end_pos)
                    if break_pos > current_pos + self.chunk_size // 2:
                        end_pos = break_pos + len(sep)
                        break

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos - self.chunk_overlap

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
