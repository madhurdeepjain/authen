"""
PDF text extraction with support for large documents.

Provides text extraction with chunking capabilities
for processing large PDFs with LLMs.
"""

import re
from pathlib import Path

import pdfplumber
import structlog

from authen.core.schemas import ExtractionResult

logger = structlog.get_logger()


class PDFExtractor:
    """
    Extract text from PDF documents with chunking support.

    Uses pdfplumber for extraction and handles large documents
    by splitting into manageable chunks for LLM processing.
    """

    def __init__(
        self,
        chunk_size: int = 50000,
        chunk_overlap: int = 1000,
    ):
        """
        Initialize the PDF extractor.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract(self, file_path: str | Path) -> ExtractionResult:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            ExtractionResult with extracted text and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info("extracting_pdf", file=str(file_path))

        text_parts = []
        page_count = 0
        metadata = {}

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            metadata = pdf.metadata or {}

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        text = "\n".join(text_parts)

        # Clean up text
        text = self._clean_text(text)

        # Create chunks if needed
        chunks = self._create_chunks(text) if len(text) > self.chunk_size else []

        logger.info(
            "extraction_complete",
            pages=page_count,
            chars=len(text),
            chunks=len(chunks) if chunks else 1,
        )

        return ExtractionResult(
            text=text,
            source_file=str(file_path),
            page_count=page_count,
            chunks=chunks,
            metadata=metadata,
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Fix common OCR/extraction issues
        text = text.replace("\x00", "")  # Null bytes
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # Control chars

        # Fix hyphenation at line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        return text.strip()

    def _create_chunks(self, text: str) -> list[str]:
        """
        Split text into chunks for LLM processing.

        Tries to split at paragraph boundaries to avoid breaking references.
        """
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Calculate end position
            end_pos = min(current_pos + self.chunk_size, len(text))

            if end_pos < len(text):
                # Try to find a good break point (paragraph or sentence)
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", current_pos, end_pos)
                if paragraph_break > current_pos + self.chunk_size // 2:
                    end_pos = paragraph_break + 2

                else:
                    # Look for sentence break
                    sentence_breaks = [
                        text.rfind(". ", current_pos, end_pos),
                        text.rfind(".\n", current_pos, end_pos),
                    ]
                    best_break = max(sentence_breaks)
                    if best_break > current_pos + self.chunk_size // 2:
                        end_pos = best_break + 1

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Move position with overlap
            current_pos = end_pos - self.chunk_overlap

        return chunks

    def extract_references_section(self, text: str) -> str | None:
        """
        Try to extract just the references section from the text.

        Returns None if no references section is found.
        """
        # Common reference section headers
        patterns = [
            r"\n\s*references\s*\n",
            r"\n\s*bibliography\s*\n",
            r"\n\s*works cited\s*\n",
            r"\n\s*literature cited\s*\n",
            r"\n\s*citations\s*\n",
        ]

        text_lower = text.lower()
        ref_start = -1

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                ref_start = match.start()
                break

        if ref_start == -1:
            return None

        # Find the end of references (next major section or end of document)
        # Look for section headers that typically come after references
        end_patterns = [
            r"\n\s*appendix",
            r"\n\s*supplementary",
            r"\n\s*acknowledgments",
            r"\n\s*author contributions",
            r"\n\s*funding",
            r"\n\s*conflicts? of interest",
        ]

        ref_end = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text_lower[ref_start:], re.IGNORECASE)
            if match:
                potential_end = ref_start + match.start()
                if potential_end < ref_end:
                    ref_end = potential_end

        references_text = text[ref_start:ref_end].strip()

        logger.info(
            "extracted_references_section",
            start=ref_start,
            end=ref_end,
            chars=len(references_text),
        )

        return references_text
