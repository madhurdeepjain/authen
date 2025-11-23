"""Validation-specific data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AcademicValidationResult:
    logs: List[str] = field(default_factory=list)
    reference_metadata: Dict[str, Any] = field(default_factory=dict)
    author_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def merge_reference_metadata(self, new_data: Optional[Dict[str, Any]]):
        if not new_data:
            return
        if "authors" in new_data and new_data["authors"]:
            if not self.reference_metadata.get("authors"):
                self.reference_metadata["authors"] = new_data["authors"]
        for key, value in new_data.items():
            if key == "authors":
                continue
            if value and not self.reference_metadata.get(key):
                self.reference_metadata[key] = value

    def merge_author_metadata(self, normalized_name: str, data: Dict[str, Any]):
        if not normalized_name or not data:
            return
        existing = self.author_metadata.get(normalized_name, {})
        for key, value in data.items():
            if value and not existing.get(key):
                existing[key] = value
        self.author_metadata[normalized_name] = existing
