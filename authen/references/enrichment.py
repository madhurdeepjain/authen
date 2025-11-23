"""Enrichment helpers for reference + author data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .models import Affiliation, Author, ReferenceData


def normalize_author_key(
    first: Optional[str],
    last: Optional[str],
    fallback: Optional[str] = None,
) -> str:
    parts = []
    if first:
        parts.append(first.strip())
    if last:
        parts.append(last.strip())
    if parts:
        return " ".join(parts).lower()
    if fallback:
        return fallback.strip().lower()
    return ""


def split_name(full_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not full_name:
        return None, None
    parts = full_name.strip().split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def apply_author_payload(author: Author, payload: Dict[str, Any]) -> None:
    if not payload:
        return

    for attr in ["first_name", "last_name", "title", "country", "address"]:
        value = payload.get(attr)
        if value and not getattr(author, attr):
            setattr(author, attr, value)

    affiliation_payload = payload.get("affiliation")
    if affiliation_payload and not author.affiliation:
        if isinstance(affiliation_payload, Affiliation):
            author.affiliation = affiliation_payload
        elif isinstance(affiliation_payload, dict):
            try:
                author.affiliation = Affiliation(**affiliation_payload)
            except TypeError:
                pass

    emails = payload.get("emails")
    if emails:
        if not author.emails:
            author.emails = []
        for email in emails:
            if email and email not in author.emails:
                author.emails.append(email)


def format_author_name(author: Author) -> str:
    full = f"{author.first_name or ''} {author.last_name or ''}".strip()
    if author.title:
        return f"{author.title} {full}".strip()
    return full


def format_affiliation(author: Author) -> str:
    if author.affiliation:
        aff = author.affiliation
        parts = [
            part for part in [aff.name, aff.department, aff.city, aff.country] if part
        ]
        if parts:
            return ", ".join(parts)
    return ""


def get_author_country(author: Author) -> str:
    if author.affiliation and author.affiliation.country:
        return author.affiliation.country
    return author.country or ""


def apply_reference_enrichment(
    reference: ReferenceData, metadata: Dict[str, Any]
) -> None:
    if not metadata:
        return

    for attr in [
        "title",
        "year",
        "publication",
        "publisher",
        "volume",
        "issue",
        "pages",
        "doi",
        "isbn",
        "url",
    ]:
        value = metadata.get(attr)
        if value and not getattr(reference, attr):
            setattr(reference, attr, value)

    authors_meta = metadata.get("authors") or []
    if not authors_meta:
        return

    existing = {
        normalize_author_key(author.first_name, author.last_name): author
        for author in reference.authors
        if normalize_author_key(author.first_name, author.last_name)
    }

    for payload in authors_meta:
        key = normalize_author_key(payload.get("first_name"), payload.get("last_name"))
        target = existing.get(key)
        if not target:
            target = Author(
                first_name=payload.get("first_name"),
                last_name=payload.get("last_name"),
            )
            reference.authors.append(target)
            if key:
                existing[key] = target
        apply_author_payload(target, payload)


def apply_author_enrichment(
    reference: ReferenceData, metadata_map: Dict[str, Dict[str, Any]]
) -> None:
    if not metadata_map:
        return
    for author in reference.authors:
        key = normalize_author_key(author.first_name, author.last_name)
        payload = metadata_map.get(key)
        if payload:
            apply_author_payload(author, payload)


def enrich_authors_with_emails(reference: ReferenceData, emails: List[str]) -> None:
    if not emails:
        return
    for email in emails:
        target = next(
            (author for author in reference.authors if not author.emails), None
        )
        if target:
            target.emails = target.emails or []
            if email not in target.emails:
                target.emails.append(email)
