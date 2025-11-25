"""
PDF extraction subpackage for extracting text from PDF documents.

This subpackage handles:
- Text extraction from PDFs using multiple backends
- Chunking large documents for LLM processing
- Handling various PDF formats and encodings
"""

from authen.pdf.extractor import PDFExtractor

__all__ = ["PDFExtractor"]
