"""Caching helpers for reference processing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from authen.core import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Simple file-based cache for LLM and validation responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                logger.info(f"Cache hit for key {cache_key[:16]}...")
                return data
            except Exception as exc:
                logger.warning(f"Error reading cache file {cache_file}: {exc}")
        return None

    def save_cached_response(self, cache_key: str, data: Any) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            logger.info(f"Cached response for key {cache_key[:16]}...")
        except Exception as exc:
            logger.warning(f"Error saving cache file {cache_file}: {exc}")
