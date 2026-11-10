"""Apply only reviewed completed-merger proof to exact frozen810.

The grader, other evidence gates and unknown source generations stay unchanged.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "completed_merger_survivor_target_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
FROZEN_IMAGE_DIGEST = "sha256:810de86a736380557fa8d4166b39ce28295ffd5b0471ffd5a3f3e4cd7d12e1b5"
ORIGINAL_SHA256 = '89c87b4dcfcabc5dc592d4425489c5bbdb6854eef97b038b167a0fb13cd26c3c'
REPAIRED_SHA256 = '32a9a14713fa71e595ba6a9b5f5764f5d7fdae21829d5cf7d3573bb0ba7a1fda'

_OLD_0 = r'''_ACQUIRED_STAGE_PROOF_PATTERNS = (
'''
_NEW_0 = r'''_COMPLETED_MERGER_SURVIVOR_RE = re.compile(
    r"\b(?:has\s+been\s+)?merged\s+with\s+and\s+into\s+"
    r"(?P<subject>[a-z0-9&.'’+ -]{2,120}?)"
    r"(?:\s+\([^()\r\n]{1,120}\))?\s*,\s*with\s+(?P=subject)\s+"
    r"(?:surviving\s+the\s+merger|continuing)\s+as\s+"
    r"(?:the\s+surviving\s+corporation\s+and\s+as\s+)?"
    r"(?:a\s+)?(?:direct\s+)?wholly[- ]owned\s+subsidiary\s+of\b",
    re.I,
)
_ACQUIRED_STAGE_PROOF_PATTERNS = (
    _COMPLETED_MERGER_SURVIVOR_RE,
'''

_OLD_1 = r'''_BOUND_ACQUISITION_SUBJECT_PATTERNS = (
'''
_NEW_1 = r'''_BOUND_ACQUISITION_SUBJECT_PATTERNS = (
    # A completed merger must name the same surviving target twice and state
    # its wholly owned subsidiary result. A buyer or another named party does
    # not inherit that target's stage. The shared uncertainty guards still run.
    _COMPLETED_MERGER_SURVIVOR_RE,
'''

_OLD_2 = r'''    for pattern in patterns:
        for match in pattern.finditer(text):
'''
_NEW_2 = r'''    for pattern in patterns:
        for match in pattern.finditer(text):
            if pattern is _COMPLETED_MERGER_SURVIVOR_RE and re.search(
                r"\b(?:if|unless|conditional(?:ly)?|subject\s+to)\b", text, re.I,
            ):
                # This full merger predicate can exceed the ordinary context
                # window. Conditions anywhere in its bounded quote stay binding.
                continue
'''

_OLD_3 = r'''    for pattern in _BOUND_ACQUISITION_SUBJECT_PATTERNS:
        for match in pattern.finditer(quote):
'''
_NEW_3 = r'''    for pattern in _BOUND_ACQUISITION_SUBJECT_PATTERNS:
        for match in pattern.finditer(quote):
            if pattern is _COMPLETED_MERGER_SURVIVOR_RE and re.search(
                r"\b(?:if|unless|conditional(?:ly)?|subject\s+to)\b", quote, re.I,
            ):
                continue
'''


class CompletedMergerSeamError(ValueError):
    """The source differs from the exact reviewed frozen generation."""


def convert_completed_merger_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise CompletedMergerSeamError("unrecognized or partial frozen completed-merger source")
    text = source.decode("utf-8")
    for old, new in (
        (_OLD_0, _NEW_0), (_OLD_1, _NEW_1), (_OLD_2, _NEW_2), (_OLD_3, _NEW_3),
    ):
        if text.count(old) != 1:
            raise CompletedMergerSeamError("frozen completed-merger source anchor differs")
        text = text.replace(old, new, 1)
    result = text.encode("utf-8")
    compile(result, "frozen810_completed_merger.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise CompletedMergerSeamError("completed-merger correction differs from reviewed source")
    return result
