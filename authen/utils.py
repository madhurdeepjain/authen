"""Utility functions for Authen."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .core import get_logger
from .models import Affiliation, Author, ReferenceData

logger = get_logger(__name__)


# Cache utilities


class CacheManager:
    """Manages caching for LLM responses and validation results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, content: str) -> str:
        """Generate a cache key for the given content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached response if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Cache hit for key {cache_key[:16]}...")
                return data
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None

    def save_cached_response(self, cache_key: str, data: Any):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached response for key {cache_key[:16]}...")
        except Exception as e:
            logger.warning(f"Error saving cache file {cache_file}: {e}")


# Text utilities


def clean_json_output(ai_message):
    """
    Clean and extract JSON content from LLM response messages.
    Handles various message formats and removes markdown code blocks.
    """
    # Handle list of messages (e.g., when tool calls are used)
    if isinstance(ai_message, list):
        for msg in reversed(ai_message):
            if hasattr(msg, "content") and msg.content:
                ai_message = msg
                break
        else:
            # Fall back to joining textual entries if they exist
            if all(isinstance(item, str) for item in ai_message):
                return "\n".join(ai_message)
            ai_message = ai_message[-1] if ai_message else ""

    # Extract text content
    if isinstance(ai_message, str):
        text = ai_message
    elif hasattr(ai_message, "content"):
        text = ai_message.content if ai_message.content else ""
    else:
        text = str(ai_message)

    # Normalize list/tuple content (sometimes providers return tool output arrays)
    if isinstance(text, (list, tuple)):
        text = "\n".join([str(item) for item in text if item is not None])

    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Remove "Updated Data:" prefix if present
    if "Updated Data:" in text:
        text = text.split("Updated Data:", 1)[1]

    # Remove markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text using regex."""
    if not text:
        return []
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(email_pattern, text)))


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split `text` into overlapping chunks while guaranteeing forward progress."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    normalized_overlap = max(0, min(overlap, chunk_size - 1))
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end >= text_length:
            break
        start = end - normalized_overlap

    return chunks


# Author utilities


def normalize_author_key(
    first: Optional[str], last: Optional[str], fallback: Optional[str] = None
) -> str:
    """
    Create a normalized key for author identification.
    Combines first and last name in lowercase, or uses fallback.
    """
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
    """Split a full name into first and last name."""
    if not full_name:
        return None, None
    parts = full_name.strip().split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def apply_author_payload(author: Author, payload: Dict[str, Any]) -> None:
    """
    Apply metadata from a payload dict to an Author object.
    Only updates fields that are currently None/empty.
    """
    if not payload:
        return

    # Update simple fields if they're empty
    for attr in ["first_name", "last_name", "title", "country", "address"]:
        value = payload.get(attr)
        if value and not getattr(author, attr):
            setattr(author, attr, value)

    # Handle affiliation
    affiliation_payload = payload.get("affiliation")
    if affiliation_payload and not author.affiliation:
        if isinstance(affiliation_payload, Affiliation):
            author.affiliation = affiliation_payload
        elif isinstance(affiliation_payload, dict):
            try:
                author.affiliation = Affiliation(**affiliation_payload)
            except TypeError:
                pass

    # Handle emails (merge, don't overwrite)
    emails = payload.get("emails")
    if emails:
        if not author.emails:
            author.emails = []
        for email in emails:
            if email and email not in author.emails:
                author.emails.append(email)


def format_author_name(author: Author) -> str:
    """Format author name with title."""
    fname = author.first_name or ""
    lname = author.last_name or ""
    full = f"{fname} {lname}".strip()
    if author.title:
        return f"{author.title} {full}".strip()
    return full


def format_affiliation(author: Author) -> str:
    """Format author affiliation as a string."""
    if author.affiliation:
        aff = author.affiliation
        parts = [
            part for part in [aff.name, aff.department, aff.city, aff.country] if part
        ]
        if parts:
            return ", ".join(parts)
    return ""


def get_author_country(author: Author) -> str:
    """Get author country from affiliation or author.country field."""
    if author.affiliation and author.affiliation.country:
        return author.affiliation.country
    return author.country or ""


# Enrichment utilities


def apply_reference_enrichment(
    reference: ReferenceData, metadata: Dict[str, Any]
) -> None:
    """
    Apply metadata enrichment to a reference.
    Only updates fields that are currently None/empty.
    """
    if not metadata:
        return

    # Update simple reference fields if they're empty
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

    # Build map of existing authors
    existing = {
        normalize_author_key(author.first_name, author.last_name): author
        for author in reference.authors
        if normalize_author_key(author.first_name, author.last_name)
    }

    # Merge or add authors from metadata
    for payload in authors_meta:
        key = normalize_author_key(payload.get("first_name"), payload.get("last_name"))
        target = existing.get(key)
        if not target:
            # Create new author
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
    """
    Apply author-specific metadata enrichment to reference authors.
    metadata_map keys should be normalized author names.
    """
    if not metadata_map:
        return
    for author in reference.authors:
        key = normalize_author_key(author.first_name, author.last_name)
        payload = metadata_map.get(key)
        if payload:
            apply_author_payload(author, payload)


def enrich_authors_with_emails(reference: ReferenceData, emails: List[str]) -> None:
    """
    Add discovered emails to authors who don't have any.
    Distributes emails among authors without email addresses.
    """
    if not emails:
        return

    for email in emails:
        # Find first author without email
        target = next((a for a in reference.authors if not a.emails), None)
        if target:
            target.emails = target.emails or []
            if email not in target.emails:
                target.emails.append(email)
