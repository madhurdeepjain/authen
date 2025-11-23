"""Helper utilities for LLM integrations."""

from __future__ import annotations

from typing import Any, Iterable, Union

from langchain_core.messages import BaseMessage


def _pick_last_message(messages: Iterable[Any]) -> Any:
    picked = None
    for message in messages:
        picked = message
    return picked


def clean_json_output(ai_message: Union[str, BaseMessage, list[Any]]) -> str:
    if isinstance(ai_message, list):
        textual_items = [item for item in ai_message if isinstance(item, str)]
        if textual_items:
            ai_message = "\n".join(textual_items)
        else:
            ai_message = next(
                (
                    msg
                    for msg in reversed(ai_message)
                    if hasattr(msg, "content") and getattr(msg, "content")
                ),
                _pick_last_message(ai_message),
            )

    if isinstance(ai_message, str):
        text = ai_message
    elif hasattr(ai_message, "content"):
        text = ai_message.content or ""
    else:
        text = str(ai_message)

    if isinstance(text, (list, tuple)):
        text = "\n".join(str(item) for item in text if item is not None)

    if not isinstance(text, str):
        text = str(text)

    if "Updated Data:" in text:
        text = text.split("Updated Data:", 1)[1]

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()
