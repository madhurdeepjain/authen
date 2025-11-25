"""
Pipeline subpackage for orchestrating the reference validation workflow.

Provides:
- Full pipeline from PDF to validated Excel
- Modular execution of individual steps
- Async processing with progress tracking
"""

from authen.pipeline.orchestrator import Pipeline

__all__ = ["Pipeline"]
