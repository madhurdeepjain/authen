"""Excel export helpers for structured reference data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd

from authen.references.models import ReferenceData, Author
from authen.references.enrichment import (
    format_affiliation,
    format_author_name,
    get_author_country,
)


def _serialize_author(author: Author) -> dict[str, Any]:
    """Serialize author objects to JSON-friendly dicts."""
    return {
        "first_name": author.first_name,
        "last_name": author.last_name,
        "title": author.title,
        "country": author.country,
        "affiliation": author.affiliation.model_dump() if author.affiliation else None,
        "emails": author.emails,
        "address": author.address,
    }


def _author_summary(author: Author) -> str:
    """Return a concise textual summary for an author."""
    identity = format_author_name(author)
    details: List[str] = []
    affiliation = format_affiliation(author)
    if affiliation:
        details.append(affiliation)
    country = get_author_country(author)
    if country and country not in affiliation:
        details.append(country)
    summary = identity
    if details:
        summary = f"{identity} [{', '.join(details)}]"
    if author.emails:
        summary = f"{summary} <{', '.join(author.emails)}>"
    return summary


def _author_slot(author: Author | None) -> dict[str, Any]:
    """Build a dict for either the first or last author slot."""
    if not author:
        return {
            "Name": None,
            "Title": None,
            "Affiliation": None,
            "Affiliation JSON": None,
            "Country": None,
            "Email": None,
        }
    affiliation = format_affiliation(author)
    affiliation_json = (
        json.dumps(author.affiliation.model_dump(), ensure_ascii=False)
        if author.affiliation
        else None
    )
    email_value = ", ".join(author.emails) if author.emails else None
    return {
        "Name": format_author_name(author),
        "Title": author.title,
        "Affiliation": affiliation or None,
        "Affiliation JSON": affiliation_json,
        "Country": get_author_country(author) or None,
        "Email": email_value,
    }


@dataclass(slots=True)
class ReferenceRowBuilder:
    """Build flattened dict rows that mirror the Excel schema."""

    def build_many(self, references: Iterable[ReferenceData]) -> List[dict[str, Any]]:
        return [self.build(reference) for reference in references]

    def build(self, reference: ReferenceData) -> dict[str, Any]:
        authors: Sequence[Author] = reference.authors or []
        first_author = authors[0] if authors else None
        last_author = authors[-1] if authors else None

        row: dict[str, Any] = {
            "Raw Reference Text": reference.raw_text,
            "Paper Title": reference.title,
            "Publication Year": reference.year,
            "Publication Venue": reference.publication,
            "Publisher": reference.publisher,
            "Volume": reference.volume,
            "Issue": reference.issue,
            "Pages": reference.pages,
            "DOI": reference.doi,
            "ISBN": reference.isbn,
            "URL": reference.url,
            "Search Context": reference.search_context,
            "Structured Reference JSON": json.dumps(
                reference.model_dump(), ensure_ascii=False
            ),
            "Authors Count": len(authors),
            "Authors (JSON)": json.dumps(
                [_serialize_author(author) for author in authors], ensure_ascii=False
            ),
            "All Authors (Detailed)": "; ".join(
                _author_summary(author) for author in authors
            ),
        }

        self._apply_author_slot(row, _author_slot(first_author), "First")
        if last_author is first_author:
            self._apply_author_slot(row, _author_slot(last_author), "Last")
        else:
            self._apply_author_slot(row, _author_slot(last_author), "Last")

        return row

    @staticmethod
    def _apply_author_slot(
        row: dict[str, Any], slot: dict[str, Any], prefix: str
    ) -> None:
        for label, value in slot.items():
            row[f"{prefix} Author {label}"] = value


class ReferenceExcelExporter:
    """Write reference collections to Excel workbooks."""

    def __init__(self, row_builder: ReferenceRowBuilder | None = None) -> None:
        self.row_builder = row_builder or ReferenceRowBuilder()

    def export(
        self, references: Sequence[ReferenceData], output_path: str | Path
    ) -> Path:
        rows = self.row_builder.build_many(references)
        dataframe = pd.DataFrame(rows)
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_excel(target_path, index=False)
        return target_path


def export_references_to_excel(
    references: Sequence[ReferenceData], output_path: str | Path
) -> Path:
    """Convenience wrapper mirroring the legacy API surface."""
    exporter = ReferenceExcelExporter()
    return exporter.export(references, output_path)
