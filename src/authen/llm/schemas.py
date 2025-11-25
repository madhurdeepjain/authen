"""
Pydantic schemas for LLM structured output.

These schemas are designed to work with structured output features
from various LLM providers (OpenAI, Anthropic, etc.)
"""

from pydantic import BaseModel, Field


class LLMAffiliation(BaseModel):
    """Affiliation as extracted by LLM."""

    name: str | None = Field(default=None, description="Institution name")
    department: str | None = Field(default=None, description="Department")
    country: str | None = Field(default=None, description="Country")
    city: str | None = Field(default=None, description="City")


class LLMAuthor(BaseModel):
    """Author as extracted by LLM."""

    first_name: str | None = Field(default=None, description="First name")
    last_name: str | None = Field(default=None, description="Last name")
    full_name: str | None = Field(default=None, description="Full name if can't split")
    affiliations: list[LLMAffiliation] = Field(
        default_factory=list, description="Author affiliations"
    )
    email: str | None = Field(default=None, description="Email if present")


class LLMReference(BaseModel):
    """Single reference as extracted by LLM."""

    # Raw text
    raw_text: str | None = Field(default=None, description="Original reference text")
    reference_number: int | None = Field(
        default=None, description="Reference number in document"
    )

    # Core fields
    title: str | None = Field(default=None, description="Title of the work")
    authors: list[LLMAuthor] = Field(
        default_factory=list, description="List of authors"
    )
    year: str | None = Field(default=None, description="Publication year")

    # Publication details
    publication: str | None = Field(default=None, description="Journal/venue name")
    publisher: str | None = Field(default=None, description="Publisher")
    volume: str | None = Field(default=None, description="Volume number")
    issue: str | None = Field(default=None, description="Issue number")
    pages: str | None = Field(default=None, description="Page range")

    # Identifiers
    doi: str | None = Field(default=None, description="DOI")
    isbn: str | None = Field(default=None, description="ISBN")
    pmid: str | None = Field(default=None, description="PubMed ID")
    arxiv_id: str | None = Field(default=None, description="arXiv ID")
    url: str | None = Field(default=None, description="URL")

    # Work type
    work_type: str | None = Field(
        default=None,
        description="Type: article, book, conference-paper, thesis, etc.",
    )


class ReferenceListOutput(BaseModel):
    """
    Output schema for reference extraction.

    This is the structured output format that LLMs will return.
    """

    references: list[LLMReference] = Field(
        default_factory=list,
        description="List of extracted references",
    )
    total_count: int = Field(
        default=0,
        description="Total number of references found",
    )
    parsing_notes: str | None = Field(
        default=None,
        description="Any notes about parsing issues or ambiguities",
    )
