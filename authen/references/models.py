"""Reference-related data models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Affiliation(BaseModel):
    name: Optional[str] = Field(default=None, description="Institution/university name")
    department: Optional[str] = Field(
        default=None, description="Department or division"
    )
    country: Optional[str] = Field(
        default=None, description="Country of the institution"
    )
    city: Optional[str] = Field(default=None, description="City of the institution")


class Author(BaseModel):
    first_name: Optional[str] = Field(default=None, description="Author first name")
    last_name: Optional[str] = Field(default=None, description="Author last name")
    title: Optional[str] = Field(
        default=None, description="Academic or professional title"
    )
    country: Optional[str] = Field(
        default=None, description="Country associated with the author"
    )
    affiliation: Optional[Affiliation] = Field(
        default=None,
        description="Structured affiliation details (institution, country, city)",
    )
    emails: Optional[List[str]] = Field(default=None, description="Email addresses")
    address: Optional[str] = Field(default=None, description="Physical address")

    @field_validator("emails", mode="before")
    @classmethod
    def clean_emails(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [email for email in value if email and isinstance(email, str)]
        return value


class ReferenceData(BaseModel):
    raw_text: Optional[str] = Field(default=None, description="Raw reference text")
    title: Optional[str] = Field(default=None, description="Paper title")
    authors: List[Author] = Field(default_factory=list, description="Authors list")
    year: Optional[str] = Field(default=None, description="Publication year")
    publication: Optional[str] = Field(default=None, description="Venue/journal")
    publisher: Optional[str] = Field(default=None, description="Publisher")
    volume: Optional[str] = Field(default=None, description="Volume number")
    issue: Optional[str] = Field(default=None, description="Issue number")
    pages: Optional[str] = Field(default=None, description="Page range")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    isbn: Optional[str] = Field(default=None, description="ISBN")
    url: Optional[str] = Field(default=None, description="Reference URL")
    search_context: Optional[str] = Field(
        default=None, description="Validation context logs"
    )


class ReferenceList(BaseModel):
    references: List[ReferenceData] = Field(description="Extracted references")

    @field_validator("references", mode="before")
    @classmethod
    def handle_empty_dict(cls, value):
        if isinstance(value, dict) and not value:
            return []
        return value

    @field_validator("references", mode="after")
    @classmethod
    def normalize_references(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            if "references" in value:
                nested = value["references"]
                if isinstance(nested, list):
                    return nested
                if isinstance(nested, dict):
                    return [nested]
                return []
            return [value]
        if isinstance(value, list):
            return [ref for ref in value if ref is not None]
        return []
