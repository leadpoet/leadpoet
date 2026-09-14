"""Small, dependency-free contract for client-facing Arena intent prose.

Writing rules follow Tyche's reviewed Intent Details contract. Shape validation
does not establish factual support; the Arena verifier checks that separately.
"""

from __future__ import annotations

import re
import unicodedata


INTENT_DETAILS_MAX_LENGTH = 2_000


def validate_intent_details_text(value: str) -> str:
    """Return one plain paragraph, preserving its wording and punctuation."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("intent_details must be a non-empty paragraph")
    if len(value) > INTENT_DETAILS_MAX_LENGTH:
        raise ValueError("intent_details exceeds 2000 characters")
    if re.search(r"\n[ \t\r]*\n", value):
        raise ValueError("intent_details must be one paragraph")
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        and character not in "\r\n\t"
        for character in value
    ):
        raise ValueError("intent_details contains unsupported control characters")
    if re.search(r"(?:^|\n)\s*(?:#{1,6}\s|[-*•]\s|\d+[.)]\s|>)", value):
        raise ValueError("intent_details must be prose, not headings or a list")
    if "```" in value:
        raise ValueError("intent_details must be prose, not a code block")
    return " ".join(value.split())
