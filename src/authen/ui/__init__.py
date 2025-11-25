"""
Streamlit UI subpackage for the reference validation pipeline.

Provides a user-friendly web interface for:
- Uploading PDFs or pasting text
- Configuring LLM and validation settings
- Viewing and exporting results
"""

from authen.ui.app import run

__all__ = ["run"]
