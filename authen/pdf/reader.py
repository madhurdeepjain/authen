"""PDF reading helpers."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from authen.core import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    text_content = ""
    path = Path(pdf_path)
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        return text_content
    except Exception as exc:
        logger.error(f"Error extracting text from PDF {path}: {exc}")
        raise
