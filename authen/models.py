"""Data models for Authen."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Affiliation(BaseModel):
    """Detailed affiliation information for an author."""

    name: Optional[str] = Field(
        description="Name of the institution, university, or organization", default=None
    )
    department: Optional[str] = Field(
        description="Department or division within the institution", default=None
    )
    country: Optional[str] = Field(
        description="Country where the institution is located", default=None
    )
    city: Optional[str] = Field(
        description="City where the institution is located", default=None
    )


class Author(BaseModel):
    """Author information including affiliation and contact details."""

    first_name: Optional[str] = Field(
        description="First name of the author", default=None
    )
    last_name: Optional[str] = Field(
        description="Last name of the author", default=None
    )
    title: Optional[str] = Field(
        description="Academic or professional title (e.g., Prof., Dr.)",
        default=None,
    )
    country: Optional[str] = Field(
        description="Country of the author (if different from affiliation country)",
        default=None,
    )
    affiliation: Optional[Affiliation] = Field(
        description="Detailed affiliation information including name, department, country, and city",
        default=None,
    )
    emails: Optional[List[str]] = Field(
        description="List of email addresses if available", default=None
    )
    address: Optional[str] = Field(
        description="Physical address or location if available", default=None
    )

    @field_validator("emails", mode="before")
    @classmethod
    def clean_emails(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            # Filter out None values and non-strings
            return [e for e in v if e is not None and isinstance(e, str)]
        return v


class ReferenceData(BaseModel):
    """Structured data for a single academic reference."""

    raw_text: Optional[str] = Field(
        description="The original reference text", default=None
    )
    title: Optional[str] = Field(
        description="Title of the academic paper", default=None
    )
    authors: List[Author] = Field(
        description="List of authors and their details", default_factory=list
    )
    year: Optional[str] = Field(description="Year of publication", default=None)
    publication: Optional[str] = Field(
        description="Journal or conference name", default=None
    )
    publisher: Optional[str] = Field(
        description="Publisher or organization name", default=None
    )
    volume: Optional[str] = Field(description="Volume number", default=None)
    issue: Optional[str] = Field(description="Issue number", default=None)
    pages: Optional[str] = Field(description="Page range", default=None)
    doi: Optional[str] = Field(description="Digital Object Identifier", default=None)
    isbn: Optional[str] = Field(description="ISBN", default=None)
    url: Optional[str] = Field(description="URL to the paper", default=None)
    search_context: Optional[str] = Field(
        description="Search results for validation", default=None
    )


class ReferenceList(BaseModel):
    """Container for multiple references."""

    references: List[ReferenceData] = Field(
        description="List of extracted reference data"
    )

    @field_validator("references", mode="before")
    @classmethod
    def handle_empty_dict(cls, v):
        if isinstance(v, dict) and not v:
            return []
        return v

    @field_validator("references", mode="after")
    @classmethod
    def validate_references_list(cls, v):
        """Validate that references is a list and handle various error cases."""
        if v is None:
            return []
        if isinstance(v, dict):
            # If it's a dict, try to extract references key
            if "references" in v:
                refs = v["references"]
                if isinstance(refs, list):
                    return refs
                elif isinstance(refs, dict):
                    # Single reference wrapped in dict
                    return [refs]
                else:
                    return []
            # If dict doesn't have references key, treat as single reference
            return [v]
        if isinstance(v, list):
            # Filter out None values and ensure all items are dicts
            return [
                ref
                for ref in v
                if ref is not None and isinstance(ref, (dict, ReferenceData))
            ]
        return []


@dataclass
class AcademicValidationResult:
    """
    Container describing validation logs and any metadata recovered from
    academic APIs that can enrich the extracted references.
    """

    logs: List[str] = field(default_factory=list)
    reference_metadata: Dict[str, Any] = field(default_factory=dict)
    author_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def merge_reference_metadata(self, new_data: Optional[Dict[str, Any]]):
        """Merge new reference metadata without overwriting existing data."""
        if not new_data:
            return
        # Authors are handled separately to avoid losing partial data
        if "authors" in new_data and new_data["authors"]:
            if not self.reference_metadata.get("authors"):
                self.reference_metadata["authors"] = new_data["authors"]
        for key, value in new_data.items():
            if key == "authors":
                continue
            if value and not self.reference_metadata.get(key):
                self.reference_metadata[key] = value

    def merge_author_metadata(self, normalized_name: str, data: Dict[str, Any]):
        """Merge author-specific metadata."""
        if not normalized_name or not data:
            return
        existing = self.author_metadata.get(normalized_name, {})
        for key, value in data.items():
            if value and not existing.get(key):
                existing[key] = value
        self.author_metadata[normalized_name] = existing