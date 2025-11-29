"""
Tests for the caching mechanism.
"""

import os
import tempfile

import pytest

from authen.core.cache import (
    CacheManager,
    SQLiteCache,
    generate_cache_key,
    generate_text_hash,
)


class TestCacheKeyGeneration:
    """Tests for cache key generation functions."""

    def test_generate_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        key1 = generate_cache_key("text", "model", "prompt")
        key2 = generate_cache_key("text", "model", "prompt")
        assert key1 == key2

    def test_generate_cache_key_different_inputs(self):
        """Test that different inputs produce different keys."""
        key1 = generate_cache_key("text1", "model")
        key2 = generate_cache_key("text2", "model")
        assert key1 != key2

    def test_generate_text_hash_normalized(self):
        """Test that text hash normalizes whitespace."""
        hash1 = generate_text_hash("hello  world")
        hash2 = generate_text_hash("hello world")
        assert hash1 == hash2

    def test_generate_text_hash_case_insensitive(self):
        """Test that text hash is case insensitive."""
        hash1 = generate_text_hash("Hello World")
        hash2 = generate_text_hash("hello world")
        assert hash1 == hash2


class TestSQLiteCache:
    """Tests for SQLite cache backend."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_get_set(self, temp_db):
        """Test basic get/set operations."""
        cache = SQLiteCache(db_path=temp_db)
        try:
            cache.set("key1", {"data": "value"})
            result = cache.get("key1")
            assert result == {"data": "value"}
        finally:
            cache.close()

    def test_get_missing_key(self, temp_db):
        """Test getting a non-existent key."""
        cache = SQLiteCache(db_path=temp_db)
        try:
            result = cache.get("nonexistent")
            assert result is None
        finally:
            cache.close()

    def test_delete(self, temp_db):
        """Test deleting a key."""
        cache = SQLiteCache(db_path=temp_db)
        try:
            cache.set("key1", {"data": "value"})
            cache.delete("key1")
            result = cache.get("key1")
            assert result is None
        finally:
            cache.close()

    def test_clear_all(self, temp_db):
        """Test clearing all keys."""
        cache = SQLiteCache(db_path=temp_db)
        try:
            cache.set("key1", {"data": "value1"}, cache_type="llm")
            cache.set("key2", {"data": "value2"}, cache_type="openalex")
            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None
        finally:
            cache.close()

    def test_clear_by_type(self, temp_db):
        """Test clearing keys by type."""
        cache = SQLiteCache(db_path=temp_db)
        try:
            cache.set("key1", {"data": "value1"}, cache_type="llm")
            cache.set("key2", {"data": "value2"}, cache_type="openalex")
            cache.clear(cache_type="llm")
            assert cache.get("key1") is None
            assert cache.get("key2") is not None
        finally:
            cache.close()

    def test_get_stats(self, temp_db):
        """Test getting cache statistics."""
        cache = SQLiteCache(db_path=temp_db)
        try:
            cache.set("key1", {"data": "value1"}, cache_type="llm")
            cache.set("key2", {"data": "value2"}, cache_type="openalex")
            stats = cache.get_stats()
            assert stats["total_entries"] == 2
            assert stats["llm_entries"] == 1
            assert stats["openalex_entries"] == 1
        finally:
            cache.close()


class TestCacheManager:
    """Tests for the high-level cache manager."""

    @pytest.fixture
    def manager(self):
        """Create a cache manager with memory backend."""
        backend = SQLiteCache()
        return CacheManager(backend=backend)

    def test_llm_cache_hit(self, manager):
        """Test LLM response caching."""
        manager.set_llm_response("test text", "gpt-4", {"references": []})
        result = manager.get_llm_response("test text", "gpt-4")
        assert result == {"references": []}

    def test_llm_cache_miss(self, manager):
        """Test LLM cache miss."""
        result = manager.get_llm_response("unknown text", "gpt-4")
        assert result is None

    def test_llm_cache_different_models(self, manager):
        """Test that different models have separate cache entries."""
        manager.set_llm_response("text", "gpt-4", {"model": "gpt-4"})
        manager.set_llm_response("text", "gpt-5", {"model": "gpt-5"})

        result4 = manager.get_llm_response("text", "gpt-4")
        result5 = manager.get_llm_response("text", "gpt-5")

        assert result4["model"] == "gpt-4"
        assert result5["model"] == "gpt-5"

    def test_openalex_cache_doi(self, manager):
        """Test OpenAlex DOI caching."""
        manager.set_openalex_result("10.1234/test", "doi", {"id": "W123"})
        result = manager.get_openalex_result("10.1234/test", "doi")
        assert result == {"id": "W123"}

    def test_openalex_cache_title(self, manager):
        """Test OpenAlex title search caching."""
        manager.set_openalex_result("test_key", "title_search", {"results": []})
        result = manager.get_openalex_result("test_key", "title_search")
        assert result == {"results": []}

    def test_cache_disabled(self):
        """Test that caching is disabled when enabled=False."""
        backend = SQLiteCache()
        manager = CacheManager(backend=backend, enabled=False)

        manager.set_llm_response("text", "model", {"data": "value"})
        result = manager.get_llm_response("text", "model")
        assert result is None

    def test_get_stats(self, manager):
        """Test getting cache statistics."""
        manager.set_llm_response("text1", "model", {"data": "1"})
        manager.get_llm_response("text1", "model")  # Hit
        manager.get_llm_response("text2", "model")  # Miss

        manager.set_openalex_result("doi1", "doi", {"id": "W1"})
        manager.get_openalex_result("doi1", "doi")  # Hit

        stats = manager.get_stats()
        assert stats["llm_hits"] == 1
        assert stats["llm_misses"] == 1
        assert stats["openalex_hits"] == 1
        assert stats["openalex_misses"] == 0

    def test_generate_reference_key(self, manager):
        """Test generating reference keys."""
        from authen.core.schemas import Author

        authors = [
            Author(first_name="John", last_name="Smith"),
            Author(first_name="Jane", last_name="Doe"),
        ]

        key1 = manager.generate_reference_key("Test Title", authors, "2023")
        key2 = manager.generate_reference_key("Test Title", authors, "2023")
        assert key1 == key2

        # Different title should produce different key
        key3 = manager.generate_reference_key("Different Title", authors, "2023")
        assert key1 != key3
