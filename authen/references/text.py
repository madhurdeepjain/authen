"""Generic text helpers for references."""

from __future__ import annotations

import re
from typing import List


def extract_emails(text: str) -> List[str]:
    if not text:
        return []
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(email_pattern, text)))
