"""
OpenAlex validation subpackage.

This subpackage provides:
- Async validation of references against OpenAlex API
- Rate-limited API access (10 req/sec with polite pool)
- Batch DOI lookups for efficiency
- Fuzzy matching for title/author validation
- Reference enrichment with OpenAlex data
"""

from authen.validation.openalex import OpenAlexClient, OpenAlexValidator

__all__ = ["OpenAlexClient", "OpenAlexValidator"]
