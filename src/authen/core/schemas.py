"""
Core data schemas for the authen reference validation pipeline.

These schemas define the data structures used throughout the pipeline,
from initial reference extraction through validation and export.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Affiliation(BaseModel):
    """Structured affiliation information for an author."""

    name: str | None = Field(default=None, description="Institution/university name")
    department: str | None = Field(default=None, description="Department or division")
    country: str | None = Field(default=None, description="Country of the institution")
    city: str | None = Field(default=None, description="City of the institution")
    ror_id: str | None = Field(
        default=None, description="Research Organization Registry (ROR) ID"
    )
    openalex_id: str | None = Field(default=None, description="OpenAlex institution ID")

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Author(BaseModel):
    """Author information with structured affiliation details."""

    first_name: str | None = Field(default=None, description="Author first name")
    last_name: str | None = Field(default=None, description="Author last name")
    full_name: str | None = Field(
        default=None, description="Full display name if first/last not available"
    )
    title: str | None = Field(
        default=None, description="Academic or professional title"
    )
    country: str | None = Field(
        default=None, description="Country associated with the author"
    )
    affiliations: list[Affiliation] = Field(
        default_factory=list,
        description="List of affiliations (can have multiple)",
    )
    emails: list[str] = Field(default_factory=list, description="Email addresses")
    address: str | None = Field(default=None, description="Physical address")
    orcid: str | None = Field(default=None, description="ORCID identifier")
    openalex_id: str | None = Field(default=None, description="OpenAlex author ID")

    @property
    def display_name(self) -> str:
        """Get the best available display name."""
        if self.full_name:
            return self.full_name
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) if parts else "Unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = self.model_dump()
        data["affiliations"] = [
            aff.to_dict() for aff in self.affiliations if aff is not None
        ]
        return {k: v for k, v in data.items() if v is not None and v != []}


class ReferenceData(BaseModel):
    """
    Complete reference data structure.

    This schema represents a parsed academic reference with all available
    metadata fields. Used both for LLM extraction output and as input
    for validation.
    """

    # Source tracking
    raw_text: str | None = Field(
        default=None, description="Original raw reference text from source"
    )
    reference_number: int | None = Field(
        default=None, description="Reference number in the original document"
    )

    # Core bibliographic data
    title: str | None = Field(default=None, description="Paper/work title")
    authors: list[Author] = Field(default_factory=list, description="Authors list")
    year: str | None = Field(default=None, description="Publication year")

    # Publication venue
    publication: str | None = Field(
        default=None, description="Venue/journal/conference name"
    )
    publisher: str | None = Field(default=None, description="Publisher name")
    volume: str | None = Field(default=None, description="Volume number")
    issue: str | None = Field(default=None, description="Issue number")
    pages: str | None = Field(default=None, description="Page range (e.g., 123-145)")

    # Identifiers
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    isbn: str | None = Field(default=None, description="ISBN for books")
    pmid: str | None = Field(default=None, description="PubMed ID")
    arxiv_id: str | None = Field(default=None, description="arXiv identifier")
    url: str | None = Field(default=None, description="Direct URL to the reference")

    # Work type
    work_type: str | None = Field(
        default=None,
        description="Type of work (article, book, conference-paper, etc.)",
    )

    # OpenAlex specific
    openalex_id: str | None = Field(
        default=None, description="OpenAlex work ID after validation"
    )
    cited_by_count: int | None = Field(
        default=None, description="Citation count from OpenAlex"
    )
    is_open_access: bool | None = Field(
        default=None, description="Whether the work is open access"
    )

    def get_search_title(self) -> str | None:
        """Get a cleaned title suitable for search queries."""
        if not self.title:
            return None
        # Remove common prefixes and clean up
        title = self.title.strip()
        # Remove quotes if present
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        return title

    def has_identifier(self) -> bool:
        """Check if the reference has any unique identifier."""
        return any([self.doi, self.pmid, self.arxiv_id, self.isbn])

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        data = self.model_dump()
        data["authors"] = [author.to_dict() for author in self.authors]
        return data


class ValidationStatus(str, Enum):
    """Status of reference validation against OpenAlex."""

    VALIDATED = "validated"  # Found and matched in OpenAlex
    PARTIAL_MATCH = "partial_match"  # Found but some fields don't match
    NOT_FOUND = "not_found"  # Not found in OpenAlex
    ERROR = "error"  # Error during validation
    PENDING = "pending"  # Not yet validated


class ValidationResult(BaseModel):
    """
    Result of validating a reference against OpenAlex.

    Contains the original parsed reference, the validated/enriched reference,
    validation status, and match confidence scores.
    """

    # Original data
    original: ReferenceData = Field(description="Original parsed reference")

    # Validated/enriched data
    validated: ReferenceData | None = Field(
        default=None, description="Reference enriched with OpenAlex data"
    )

    # Validation metadata
    status: ValidationStatus = Field(
        default=ValidationStatus.PENDING, description="Validation status"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall match confidence (0-1)",
    )
    title_similarity: float | None = Field(
        default=None, description="Title similarity score (0-1)"
    )
    author_similarity: float | None = Field(
        default=None, description="Author name similarity score (0-1)"
    )

    # Match details
    match_method: str | None = Field(
        default=None,
        description="How the match was found (doi, title_search, etc.)",
    )
    openalex_url: str | None = Field(
        default=None, description="URL to the OpenAlex work page"
    )

    # Error information
    error_message: str | None = Field(
        default=None, description="Error message if validation failed"
    )

    # Additional OpenAlex metadata
    openalex_raw: dict | None = Field(
        default=None, description="Raw OpenAlex API response for debugging"
    )

    def is_valid(self) -> bool:
        """Check if the reference was successfully validated."""
        return self.status in [
            ValidationStatus.VALIDATED,
            ValidationStatus.PARTIAL_MATCH,
        ]

    def get_best_reference(self) -> ReferenceData:
        """Get the best available reference data (validated if available)."""
        if self.validated and self.is_valid():
            return self.validated
        return self.original


class ExtractionResult(BaseModel):
    """Result of extracting text from a PDF or document."""

    text: str = Field(description="Extracted text content")
    source_file: str = Field(description="Source file path")
    page_count: int | None = Field(default=None, description="Number of pages")
    chunks: list[str] = Field(
        default_factory=list,
        description="Text chunks if document was split for processing",
    )
    metadata: dict = Field(
        default_factory=dict, description="Additional extraction metadata"
    )


class ParseResult(BaseModel):
    """Result of parsing references from text using LLM."""

    references: list[ReferenceData] = Field(
        default_factory=list, description="Parsed references"
    )
    source_text: str | None = Field(
        default=None, description="Original text that was parsed"
    )
    model_used: str | None = Field(
        default=None, description="LLM model used for parsing"
    )
    token_count: int | None = Field(
        default=None, description="Approximate tokens processed"
    )
    processing_time_seconds: float | None = Field(
        default=None, description="Time taken to parse"
    )


class PipelineResult(BaseModel):
    """Complete result from the full pipeline."""

    # Source info
    source_file: str | None = Field(default=None, description="Source file path")

    # Results at each stage
    extraction: ExtractionResult | None = Field(
        default=None, description="PDF extraction result"
    )
    parse_result: ParseResult | None = Field(
        default=None, description="LLM parsing result"
    )
    validation_results: list[ValidationResult] = Field(
        default_factory=list, description="Validation results for each reference"
    )

    # Summary statistics
    total_references: int = Field(default=0, description="Total references found")
    validated_count: int = Field(
        default=0, description="Successfully validated references"
    )
    partial_match_count: int = Field(
        default=0, description="Partially matched references"
    )
    not_found_count: int = Field(
        default=0, description="References not found in OpenAlex"
    )
    error_count: int = Field(default=0, description="References with validation errors")

    # Timing
    total_processing_time_seconds: float | None = Field(
        default=None, description="Total pipeline processing time"
    )

    def compute_stats(self) -> None:
        """Compute summary statistics from validation results."""
        self.total_references = len(self.validation_results)
        self.validated_count = sum(
            1 for r in self.validation_results if r.status == ValidationStatus.VALIDATED
        )
        self.partial_match_count = sum(
            1
            for r in self.validation_results
            if r.status == ValidationStatus.PARTIAL_MATCH
        )
        self.not_found_count = sum(
            1 for r in self.validation_results if r.status == ValidationStatus.NOT_FOUND
        )
        self.error_count = sum(
            1 for r in self.validation_results if r.status == ValidationStatus.ERROR
        )


class PipelineEventType(str, Enum):
    """Types of events emitted during pipeline processing."""

    EXTRACTION_COMPLETE = "extraction_complete"
    CHUNK_PROCESSED = "chunk_processed"
    REFERENCE_FOUND = "reference_found"
    VALIDATION_COMPLETE = "validation_complete"
    COMPLETED = "completed"
    ERROR = "error"


class PipelineEvent(BaseModel):
    """Event emitted during pipeline processing."""

    type: PipelineEventType = Field(description="Type of event")
    data: dict | ReferenceData | ValidationResult | PipelineResult | ExtractionResult | str | None = Field(
        default=None, description="Event data payload"
    )
    message: str | None = Field(default=None, description="Human-readable message")
