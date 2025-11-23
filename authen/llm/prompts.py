"""Prompt templates for reference extraction."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate


def build_reference_prompt(
    academic_domains: Optional[List[str]] = None,
) -> ChatPromptTemplate:
    domain_restriction = ""
    if academic_domains:
        domain_list = " OR ".join(f"site:{domain}" for domain in academic_domains)
        domain_restriction = (
            "\n\nWhen using web search, prioritize results from these academic domains: "
            f"{domain_list}"
        )

    system_message = (
        "You are an expert academic librarian. Identify and extract all individual references from the provided text. "
        "For each reference, extract structured data fields such as raw_text, title, authors (with nested affiliation details), publication metadata, and identifiers."
        "\n\nIf search tools are available, use them to verify the paper, recover missing affiliation metadata, gather contact details, and ensure names are complete."
        f"{domain_restriction}"
        "\n\nIf a reference seems truncated, do not include it. Output ONLY valid JSON per the supplied format instructions."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message + "\n{format_instructions}"),
            ("user", "{text}"),
        ]
    )
