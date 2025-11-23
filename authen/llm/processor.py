"""Reference extraction pipeline using LLMs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from authen.core import get_logger, get_llm_cache_dir
from authen.llm.clients import build_llm_client
from authen.llm.prompts import build_reference_prompt
from authen.llm.utils import clean_json_output
from authen.pdf.preprocess import chunk_text
from authen.references.cache import CacheManager
from authen.references.models import ReferenceData, ReferenceList

logger = get_logger(__name__)


class ReferenceExtractor:
    """High-level orchestrator for chunked, cached LLM extraction."""

    def __init__(
        self,
        provider: str = "google",
        model_name: str = "gemini-3-pro",
        temperature: float = 0.0,
        enable_web_search: bool = False,
        academic_domains: Optional[List[str]] = None,
        event_logger: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.enable_web_search = enable_web_search
        self.academic_domains = academic_domains or []
        self.event_logger = event_logger
        self.progress_callback = progress_callback

        cache_dir = cache_dir or get_llm_cache_dir()
        self.cache = CacheManager(cache_dir)
        self.llm = build_llm_client(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            enable_web_search=self.enable_web_search,
        )
        self.result_parser = PydanticOutputParser(pydantic_object=ReferenceList)
        self.prompt = build_reference_prompt(self.academic_domains)

    def _log_event(self, message: str) -> None:
        if self.event_logger:
            try:
                self.event_logger(message)
                return
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Event logger callback failed", exc_info=True)
        logger.debug(message)

    def _emit_progress(
        self, event: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.progress_callback:
            return
        body = {"event": event}
        if payload:
            body.update(payload)
        try:
            self.progress_callback(body)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Progress callback failed", exc_info=True)

    async def extract_references(
        self,
        text: str,
        chunk_size: int = 15_000,
        overlap: int = 2_000,
    ) -> AsyncIterator[ReferenceData]:
        if not text or not text.strip():
            return

        self._log_event(
            "Starting extraction | provider=%s model=%s chars=%s chunk_size=%s overlap=%s"
            % (self.provider, self.model_name, len(text), chunk_size, overlap)
        )

        chunks = (
            [text]
            if len(text) <= chunk_size
            else chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        )

        if len(chunks) == 1:
            self._log_event("Processing single chunk (no splitting needed).")

        seen_keys: Set[str] = set()
        total_chunks = len(chunks)
        self._emit_progress("chunks_initialized", {"total": total_chunks})

        semaphore = asyncio.Semaphore(3)

        async def process_chunk(index: int, chunk: str):
            async with semaphore:
                chunk_label = (
                    "full" if total_chunks == 1 else f"{index + 1}/{total_chunks}"
                )
                self._log_event(
                    f"[Chunk {chunk_label}] size={len(chunk)} characters. Beginning parse."
                )
                self._emit_progress(
                    "chunk_started", {"index": index + 1, "total": total_chunks}
                )
                references = await self._extract_references_chunk(chunk, chunk_label)
                self._emit_progress(
                    "chunk_completed",
                    {
                        "index": index + 1,
                        "total": total_chunks,
                        "references": len(references),
                    },
                )
                return references

        tasks = [
            asyncio.create_task(process_chunk(i, chunk))
            for i, chunk in enumerate(chunks)
        ]

        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Chunk processing failed: %s", exc)
                continue
            for reference in result:
                key = self._reference_key(reference)
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    yield reference

        self._log_event(
            f"Finished extraction. Total references after dedupe: {len(seen_keys)}"
        )
        self._emit_progress("extraction_finished", {"total": total_chunks})

    async def _extract_references_chunk(
        self, text: str, chunk_label: str = "full"
    ) -> List[ReferenceData]:
        cache_key = self.cache.get_cache_key(
            f"{self.provider}:{self.model_name}:{text}"
        )
        cached = self.cache.get_cached_response(cache_key)
        if cached is not None:
            references = [ReferenceData(**payload) for payload in cached]
            self._log_event(
                f"[Chunk {chunk_label}] Cache hit. Reusing {len(references)} references."
            )
            return references

        chain = (
            self.prompt
            | self.llm
            | RunnableLambda(self._handle_llm_response)
            | self.result_parser
        )

        try:
            self._log_event(f"[Chunk {chunk_label}] Invoking LLM for structured parse.")
            result = await chain.ainvoke(
                {
                    "text": text,
                    "format_instructions": self.result_parser.get_format_instructions(),
                }
            )
            references = result.references if hasattr(result, "references") else []

            payload = [reference.model_dump() for reference in references]
            self.cache.save_cached_response(cache_key, payload)
            self._log_event(
                f"[Chunk {chunk_label}] Parsed {len(references)} references via LLM."
            )
            return references
        except Exception as exc:
            logger.error(
                "Error extracting references from chunk: %s", exc, exc_info=True
            )
            self._log_event(
                f"[Chunk {chunk_label}] Failed to parse references. See logs for details."
            )
            return []

    def _handle_llm_response(self, ai_message):
        messages = ai_message if isinstance(ai_message, list) else [ai_message]
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                continue
            logger.info("Tool calls detected: %s", len(tool_calls))
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                else:
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_args = getattr(tool_call, "args", {})
                logger.debug("Tool call: %s - %s", tool_name, tool_args)

        return clean_json_output(ai_message)

    @staticmethod
    def _reference_key(reference: ReferenceData) -> str:
        source = reference.title or reference.raw_text or ""
        return source.strip().lower()
