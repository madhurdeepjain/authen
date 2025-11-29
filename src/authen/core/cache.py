"""
Caching mechanism for LLM responses and OpenAlex verification results.

Supports:
- In-memory caching with TTL
- Disk-based persistent caching (SQLite)
- Content-based hashing for cache keys
"""

import hashlib
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def generate_cache_key(*args: Any) -> str:
    """
    Generate a deterministic cache key from input arguments.

    Uses SHA-256 hash of the JSON-serialized arguments.
    """
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def generate_text_hash(text: str) -> str:
    """Generate a hash for text content."""
    # Normalize whitespace and case for more cache hits
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> dict | None:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        """Set a value in cache with optional TTL (seconds)."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the cache backend."""
        pass


class SQLiteCache(CacheBackend):
    """
    Persistent SQLite-based cache.

    Good for caching across sessions and for larger datasets.
    """

    def __init__(
        self,
        db_path: str | Path = ".db/cache.db",
        default_ttl: int = 86400 * 7,  # 1 week
    ):
        """
        Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
            default_ttl: Default TTL in seconds (1 week)
        """
        self.db_path = Path(db_path)
        self.default_ttl = default_ttl
        self._local = threading.local()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                cache_type TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            """
        )
        conn.commit()

    def get(self, key: str) -> dict | None:
        """Get a value from cache if not expired."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT value, expires_at FROM cache
            WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """,
            (key, time.time()),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            logger.warning("cache_json_decode_error", key=key)
            return None

    def set(
        self,
        key: str,
        value: dict,
        ttl: int | None = None,
        cache_type: str | None = None,
    ) -> None:
        """Set a value in cache with TTL."""
        conn = self._get_connection()
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None

        conn.execute(
            """
            INSERT OR REPLACE INTO cache
                (key, value, created_at, expires_at, cache_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, json.dumps(value), time.time(), expires_at, cache_type),
        )
        conn.commit()

    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        conn = self._get_connection()
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()

    def clear(self, cache_type: str | None = None) -> None:
        """Clear cached values, optionally filtered by type."""
        conn = self._get_connection()
        if cache_type:
            conn.execute("DELETE FROM cache WHERE cache_type = ?", (cache_type,))
        else:
            conn.execute("DELETE FROM cache")
        conn.commit()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of deleted items."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )
        conn.commit()
        return cursor.rowcount

    def get_stats(self) -> dict:
        """Get cache statistics."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total_entries,
                SUM(CASE WHEN cache_type = 'llm' THEN 1 ELSE 0 END)
                    as llm_entries,
                SUM(CASE WHEN cache_type = 'openalex' THEN 1 ELSE 0 END)
                    as openalex_entries,
                SUM(CASE WHEN expires_at IS NOT NULL AND expires_at < ?
                    THEN 1 ELSE 0 END) as expired_entries
            FROM cache
            """,
            (time.time(),),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection


class CacheManager:
    """
    High-level cache manager for authen components.

    Provides specialized caching methods for LLM responses and OpenAlex results.
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        enabled: bool = True,
        llm_ttl: int = 86400 * 7,  # 1 week for LLM results
        openalex_ttl: int = 86400 * 30,  # 30 days for OpenAlex results
    ):
        """
        Initialize cache manager.

        Args:
            backend: Cache backend to use (defaults to SQLite)
            enabled: Whether caching is enabled
            llm_ttl: TTL for LLM response cache
            openalex_ttl: TTL for OpenAlex cache
        """
        self.enabled = enabled
        self.llm_ttl = llm_ttl
        self.openalex_ttl = openalex_ttl

        if backend is None:
            backend = SQLiteCache()
        self.backend = backend

        self._stats = {
            "llm_hits": 0,
            "llm_misses": 0,
            "openalex_hits": 0,
            "openalex_misses": 0,
        }

    def get_llm_response(
        self,
        text: str,
        model: str,
        prompt_hash: str | None = None,
    ) -> dict | None:
        """
        Get cached LLM response for text.

        Args:
            text: Input text that was sent to LLM
            model: Model identifier
            prompt_hash: Optional hash of system prompt

        Returns:
            Cached response dict or None
        """
        if not self.enabled:
            return None

        text_hash = generate_text_hash(text)
        key = generate_cache_key("llm", model, text_hash, prompt_hash)

        result = self.backend.get(key)
        if result:
            self._stats["llm_hits"] += 1
            logger.debug("llm_cache_hit", model=model, text_len=len(text))
        else:
            self._stats["llm_misses"] += 1

        return result

    def set_llm_response(
        self,
        text: str,
        model: str,
        response: dict,
        prompt_hash: str | None = None,
    ) -> None:
        """
        Cache LLM response.

        Args:
            text: Input text that was sent to LLM
            model: Model identifier
            response: Response to cache (must be JSON-serializable)
            prompt_hash: Optional hash of system prompt
        """
        if not self.enabled:
            return

        text_hash = generate_text_hash(text)
        key = generate_cache_key("llm", model, text_hash, prompt_hash)

        if isinstance(self.backend, SQLiteCache):
            self.backend.set(key, response, ttl=self.llm_ttl, cache_type="llm")
        else:
            self.backend.set(key, response, ttl=self.llm_ttl)

        logger.debug("llm_cache_set", model=model, text_len=len(text))

    def get_openalex_result(
        self,
        identifier: str,
        identifier_type: str = "doi",
    ) -> dict | None:
        """
        Get cached OpenAlex validation result.

        Args:
            identifier: DOI, title hash, or other identifier
            identifier_type: Type of identifier (doi, title, title_author)

        Returns:
            Cached result dict or None
        """
        if not self.enabled:
            return None

        key = generate_cache_key("openalex", identifier_type, identifier)

        result = self.backend.get(key)
        if result:
            self._stats["openalex_hits"] += 1
            logger.debug(
                "openalex_cache_hit",
                identifier_type=identifier_type,
                identifier=identifier[:50],
            )
        else:
            self._stats["openalex_misses"] += 1

        return result

    def set_openalex_result(
        self,
        identifier: str,
        identifier_type: str,
        result: dict,
    ) -> None:
        """
        Cache OpenAlex validation result.

        Args:
            identifier: DOI, title hash, or other identifier
            identifier_type: Type of identifier (doi, title, title_author)
            result: Result to cache
        """
        if not self.enabled:
            return

        key = generate_cache_key("openalex", identifier_type, identifier)

        if isinstance(self.backend, SQLiteCache):
            self.backend.set(key, result, ttl=self.openalex_ttl, cache_type="openalex")
        else:
            self.backend.set(key, result, ttl=self.openalex_ttl)

        logger.debug(
            "openalex_cache_set",
            identifier_type=identifier_type,
            identifier=identifier[:50],
        )

    def get_validation_result(
        self,
        reference_key: str,
    ) -> dict | None:
        """
        Get cached validation result for a reference.

        Args:
            reference_key: Key generated by generate_reference_key

        Returns:
            Cached validation result dict or None
        """
        if not self.enabled:
            return None

        key = generate_cache_key("validation_result", reference_key)

        result = self.backend.get(key)
        if result:
            self._stats["openalex_hits"] += 1
            logger.debug("validation_cache_hit", key=reference_key[:20])
        else:
            self._stats["openalex_misses"] += 1

        return result

    def set_validation_result(
        self,
        reference_key: str,
        result: dict,
    ) -> None:
        """
        Cache validation result for a reference.

        Args:
            reference_key: Key generated by generate_reference_key
            result: Validation result to cache
        """
        if not self.enabled:
            return

        key = generate_cache_key("validation_result", reference_key)

        if isinstance(self.backend, SQLiteCache):
            self.backend.set(key, result, ttl=self.openalex_ttl, cache_type="openalex")
        else:
            self.backend.set(key, result, ttl=self.openalex_ttl)

        logger.debug("validation_cache_set", key=reference_key[:20])

    def generate_reference_key(
        self,
        title: str | None,
        authors: list | None = None,
        year: str | None = None,
    ) -> str:
        """
        Generate a cache key for a reference based on title and authors.

        Normalizes the input for better cache hit rates.
        """
        # Normalize title
        title_norm = ""
        if title:
            title_norm = " ".join(title.lower().split())

        # Normalize authors (just use last names)
        author_norm = ""
        if authors:
            last_names = []
            for author in authors[:3]:  # Use first 3 authors
                if hasattr(author, "last_name") and author.last_name:
                    last_names.append(author.last_name.lower())
                elif hasattr(author, "display_name"):
                    # Extract last name from display name
                    parts = author.display_name.split()
                    if parts:
                        last_names.append(parts[-1].lower())
            author_norm = ",".join(sorted(last_names))

        return generate_cache_key(title_norm, author_norm, year)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self._stats.copy()

        # Calculate hit rates
        total_llm = stats["llm_hits"] + stats["llm_misses"]
        total_openalex = stats["openalex_hits"] + stats["openalex_misses"]

        stats["llm_hit_rate"] = stats["llm_hits"] / total_llm if total_llm > 0 else 0.0
        stats["openalex_hit_rate"] = (
            stats["openalex_hits"] / total_openalex if total_openalex > 0 else 0.0
        )

        # Add backend stats if available
        if isinstance(self.backend, SQLiteCache):
            stats["backend"] = self.backend.get_stats()

        return stats

    def clear_cache(self, cache_type: str | None = None) -> None:
        """Clear cache, optionally by type."""
        if isinstance(self.backend, SQLiteCache):
            self.backend.clear(cache_type)
        else:
            self.backend.clear()

        logger.info("cache_cleared", cache_type=cache_type or "all")

    def close(self) -> None:
        """Close the cache manager."""
        self.backend.close()


# Global cache instance (lazy initialization)
_global_cache: CacheManager | None = None


def get_cache(
    enabled: bool = True,
    db_path: str | Path | None = None,
) -> CacheManager:
    """
    Get or create the global cache manager.

    Args:
        enabled: Whether caching should be enabled
        db_path: Optional custom database path

    Returns:
        CacheManager instance
    """
    global _global_cache

    if _global_cache is None:
        if db_path:
            backend = SQLiteCache(db_path=db_path)
        else:
            backend = SQLiteCache()

        _global_cache = CacheManager(backend=backend, enabled=enabled)

    return _global_cache


def clear_global_cache() -> None:
    """Clear and reset the global cache."""
    global _global_cache

    if _global_cache:
        _global_cache.close()
        _global_cache = None
