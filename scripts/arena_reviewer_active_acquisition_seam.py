"""Bind active acquisition targets on the existing visible-context repair.

Only the reviewed acquisition pattern changes. Frozen prompts, models,
dependencies, other guards and the submitted-company context repair stay fixed.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "completed_active_acquisition_target_v1"
ORIGINAL_SHA256 = "d1dbdd48d89ec52218c0e82270771f88924d40b4a4216fab98ba444a174cd08c"
REPAIRED_SHA256 = "fdd1e4dc6ea7c147687a0314250a4b94f934b6777e17692b8cd3912bcfa8e591"
_BEFORE = (
    "_BOUND_ACQUISITION_SUBJECT_PATTERNS = (\n"
    "    # A completed merger must name the same surviving target twice and state\n"
    "    # its wholly owned subsidiary result. A buyer or another named party does\n"
    "    # not inherit that target's stage. The shared uncertainty guards still run.\n"
    "    _COMPLETED_MERGER_SURVIVOR_RE,\n"
)
_ADDITION = r'''    # Active past-tense proof binds the acquired object, never the buyer.
    # Keep the full object so assets, products and longer names cannot become
    # aliases for the investigated company. Shared completion guards still run.
    re.compile(
        r"\bacquired\s+(?P<subject>[a-z0-9&.'’+ -]{2,120}?)"
        r"(?=\s+(?:on|for|after|from|in)\b|[,;.!?\n]|$)"
        r"(?!\s*,\s*(?:an?|the)\s+(?:[\w-]+\s+){0,3}"
        r"(?:product|assets?|brands?)\s+of\b)",
        re.I,
    ),
'''


def convert_active_acquisition_seam(source: bytes) -> bytes:
    """Reject unknown or partial input and verify the exact reviewed delta."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise ValueError("unrecognized active acquisition source")
    text = source.decode("utf-8")
    if text.count(_BEFORE) != 1:
        raise ValueError("frozen acquisition target anchor differs")
    result = text.replace(_BEFORE, _BEFORE + _ADDITION, 1).encode("utf-8")
    compile(result, "frozen_active_acquisition.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise ValueError("active acquisition postimage differs")
    return result
