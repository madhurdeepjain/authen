"""Composable pipeline that wires PDF ingestion, LLM extraction, validation, and export."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from authen.export.excel import export_references_to_excel
from authen.llm import ReferenceExtractor
from authen.pdf.reader import extract_text_from_pdf
from authen.references.enrichment import (
    apply_author_enrichment,
    apply_reference_enrichment,
    enrich_authors_with_emails,
)
from authen.references.models import ReferenceData
from authen.references.text import extract_emails
from authen.validation import AcademicValidator

StatusHook = Callable[[str], None]
ProgressHook = Callable[[Dict[str, int]], None]
ValidationLogHook = Callable[[str], None]
ReferenceHook = Callable[[ReferenceData, int], None]
ValidationStateHook = Callable[[int, str], None]
ValidationSnapshotHook = Callable[[Dict[str, Any]], None]
CancelCallback = Callable[[], bool]


@dataclass(slots=True)
class PipelineHooks:
    """Optional callbacks for consumers that need UI updates."""

    on_status: Optional[StatusHook] = None
    on_progress: Optional[ProgressHook] = None
    on_validation_log: Optional[ValidationLogHook] = None
    on_reference: Optional[ReferenceHook] = None
    on_validation_state: Optional[ValidationStateHook] = None
    on_validation_snapshot: Optional[ValidationSnapshotHook] = None
    should_cancel: Optional[CancelCallback] = None


@dataclass(slots=True)
class PipelineConfig:
    """Dial settings for the orchestrated pipeline."""

    provider: str = "google"
    model_name: str = "gemini-3-pro"
    temperature: float = 0.0
    enable_web_search: bool = True
    enable_validation: bool = True
    academic_domains: Optional[List[str]] = None
    chunk_size: int = 4000
    chunk_overlap: int = 400


class AuthenPipeline:
    """High-level facade for PDF ➜ LLM ➜ validation ➜ export."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        hooks: Optional[PipelineHooks] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.hooks = hooks or PipelineHooks()
        self._chunk_total = 0
        self._chunk_completed = 0
        self._validation_completed = 0

    def run_from_pdf(self, pdf_path: str | Path) -> List[ReferenceData]:
        """Synchronous helper: extract text from PDF and run pipeline."""
        text = extract_text_from_pdf(pdf_path)
        return self.run_from_text(text, source_label=str(pdf_path))

    def run_from_text(
        self,
        text: str,
        source_label: str = "Manual Text",
    ) -> List[ReferenceData]:
        """Synchronous helper around the async pipeline."""
        return asyncio.run(self.run_from_text_async(text, source_label=source_label))

    async def run_from_text_async(
        self,
        text: str,
        source_label: str = "Manual Text",
    ) -> List[ReferenceData]:
        if not text or not text.strip():
            return []
        self._emit_status(
            f"Starting pipeline run from {source_label} | {len(text)} chars"
        )
        references = await self._extract_references_async(text)
        if not references:
            self._emit_status("No references detected during extraction.")
            return []
        if self._should_cancel():
            self._emit_status("Pipeline run cancelled before validation stage.")
            return references
        if self.config.enable_validation:
            await self._validate_and_enrich_async(references)
        self._emit_status(
            f"Pipeline completed with {len(references)} references after dedupe"
        )
        return references

    def export_to_excel(
        self,
        references: Sequence[ReferenceData],
        output_path: str | Path,
    ) -> Path:
        """Write references to Excel using the shared exporter."""
        return export_references_to_excel(references, output_path)

    async def _extract_references_async(self, text: str) -> List[ReferenceData]:
        extractor = ReferenceExtractor(
            provider=self.config.provider,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            enable_web_search=self.config.enable_web_search,
            academic_domains=self.config.academic_domains or None,
            event_logger=self._emit_status,
            progress_callback=self._handle_llm_progress,
        )
        references: List[ReferenceData] = []
        async for reference in extractor.extract_references(
            text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        ):
            if self._should_cancel():
                self._emit_status("Cancellation requested. Stopping extraction loop.")
                break
            references.append(reference)
            self._emit_reference(reference, len(references) - 1)
            if not self.config.enable_validation:
                self._emit_validation_state(len(references) - 1, "Skipped")
        return references

    async def _validate_and_enrich_async(
        self,
        references: Iterable[ReferenceData],
    ) -> None:
        references_list = (
            references if isinstance(references, list) else list(references)
        )
        if not references_list:
            return

        validator = AcademicValidator(log_callback=self._handle_validation_log)
        self._emit_progress({"validation_total": len(references_list)})

        for idx, reference in enumerate(references_list):
            if self._should_cancel():
                self._emit_status("Cancellation requested. Halting validation loop.")
                break
            self._emit_validation_state(idx, "Validating")
            authors_payload = [author.model_dump() for author in reference.authors]
            result = await validator.validate_reference_details(
                title=reference.title,
                authors=authors_payload,
                doi=reference.doi,
                log_prefix=f"[Ref {idx + 1}] ",
            )
            if not result:
                self._validation_completed += 1
                self._emit_progress(
                    {
                        "validation_completed": self._validation_completed,
                        "index": idx + 1,
                    }
                )
                self._emit_validation_state(idx, "Failed")
                continue
            if result.logs:
                reference.search_context = "\n".join(result.logs)
            apply_reference_enrichment(reference, result.reference_metadata)
            apply_author_enrichment(reference, result.author_metadata)

            found_emails = (
                extract_emails(reference.search_context)
                if reference.search_context
                else []
            )
            if found_emails:
                enrich_authors_with_emails(reference, found_emails)

            snapshot = {
                "index": idx + 1,
                "title": reference.title,
                "reference_metadata": result.reference_metadata,
                "author_metadata": result.author_metadata,
                "logs": result.logs,
                "emails": found_emails,
            }
            self._emit_validation_snapshot(snapshot)

            self._validation_completed += 1
            self._emit_progress(
                {
                    "validation_completed": self._validation_completed,
                    "index": idx + 1,
                }
            )
            self._emit_validation_state(idx, "Validated")

    def _emit_status(self, message: str) -> None:
        if not message:
            return
        if self.hooks.on_status:
            try:
                self.hooks.on_status(message)
                return
            except Exception:
                pass
        print(message)

    def _emit_progress(self, payload: Dict[str, int]) -> None:
        if self.hooks.on_progress:
            try:
                self.hooks.on_progress(payload)
            except Exception:
                pass

    def _handle_validation_log(self, message: str) -> None:
        if self.hooks.on_validation_log:
            try:
                self.hooks.on_validation_log(message)
                return
            except Exception:
                pass
        self._emit_status(message)

    def _handle_llm_progress(self, event: Dict[str, int]) -> None:
        if event.get("event") == "chunks_initialized":
            self._chunk_total = event.get("total", 0)
            self._chunk_completed = 0
            self._emit_progress({"chunks_total": self._chunk_total})
        elif event.get("event") == "chunk_completed":
            self._chunk_completed += 1
            self._emit_progress(
                {
                    "chunks_total": self._chunk_total,
                    "chunks_completed": self._chunk_completed,
                    "chunk_references": event.get("references", 0),
                }
            )

    def _emit_reference(self, reference: ReferenceData, index: int) -> None:
        if self.hooks.on_reference:
            try:
                self.hooks.on_reference(reference, index)
            except Exception:
                pass

    def _emit_validation_state(self, index: int, state: str) -> None:
        if self.hooks.on_validation_state:
            try:
                self.hooks.on_validation_state(index, state)
            except Exception:
                pass

    def _emit_validation_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if self.hooks.on_validation_snapshot:
            try:
                self.hooks.on_validation_snapshot(snapshot)
            except Exception:
                pass

    def _should_cancel(self) -> bool:
        if not self.hooks.should_cancel:
            return False
        try:
            return bool(self.hooks.should_cancel())
        except Exception:
            return False


__all__ = [
    "PipelineConfig",
    "PipelineHooks",
    "AuthenPipeline",
]
