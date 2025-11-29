"""JSON export utilities."""

import json
from pathlib import Path

import structlog

from authen.core.schemas import PipelineResult, ValidationResult

logger = structlog.get_logger()


def export_to_json(
    results: list[ValidationResult] | PipelineResult,
    output_path: str | Path,
) -> Path:
    """
    Export validation results to JSON.

    Args:
        results: Validation results or pipeline result
        output_path: Output file path

    Returns:
        Path to created file
    """
    output_path = Path(output_path)

    if isinstance(results, PipelineResult):
        data = results.model_dump()
    else:
        data = [r.model_dump() for r in results]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info("exported_to_json", path=str(output_path))
    return output_path
