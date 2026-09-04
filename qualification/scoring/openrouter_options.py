"""Small OpenRouter request options used by company scoring."""

from __future__ import annotations

import os


def include_reasoning_default() -> bool:
    return str(os.getenv("QUALIFICATION_LLM_INCLUDE_REASONING", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reasoning_request_unsupported(status: int, message: str) -> bool:
    if int(status) not in {400, 404, 422}:
        return False
    text = str(message or "").lower()
    return "reasoning" in text or "include_reasoning" in text
