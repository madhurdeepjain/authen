"""CSV export utilities."""

from pathlib import Path

import pandas as pd
import structlog

from authen.core.schemas import PipelineResult, ValidationResult

logger = structlog.get_logger()


def export_to_csv(
    results: list[ValidationResult] | PipelineResult,
    output_path: str | Path,
) -> Path:
    """
    Export validation results to CSV.

    Args:
        results: Validation results or pipeline result
        output_path: Output file path

    Returns:
        Path to created file
    """
    output_path = Path(output_path)

    if isinstance(results, PipelineResult):
        validation_results = results.validation_results
    else:
        validation_results = results

    rows = []
    for i, result in enumerate(validation_results):
        ref = result.get_best_reference()
        row = {
            "reference_number": ref.reference_number or i + 1,
            "title": ref.title,
            "authors": ", ".join(a.display_name for a in ref.authors),
            "year": ref.year,
            "publication": ref.publication,
            "doi": ref.doi,
            "validation_status": result.status.value,
            "confidence": result.confidence,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger.info("exported_to_csv", path=str(output_path))
    return output_path
