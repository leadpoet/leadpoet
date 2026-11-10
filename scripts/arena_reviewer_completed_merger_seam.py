"""Apply only reviewed completed-merger proof to exact frozen810.

The grader, other evidence gates and unknown source generations stay unchanged.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "completed_merger_survivor_target_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
FROZEN_IMAGE_DIGEST = "sha256:810de86a736380557fa8d4166b39ce28295ffd5b0471ffd5a3f3e4cd7d12e1b5"
ORIGINAL_SHA256 = '89c87b4dcfcabc5dc592d4425489c5bbdb6854eef97b038b167a0fb13cd26c3c'
REPAIRED_SHA256 = 'bdbe6ac74622982b50cacbe8f336dd0cf09b1c7352fe1dfdc6b0b2c14b2c82e1'

_OLD_0 = r'''_ACQUIRED_STAGE_PROOF_PATTERNS = (
'''
_NEW_0 = r'''_COMPLETED_MERGER_SURVIVOR_RE = re.compile(
    r"\b(?:has\s+been\s+)?merged\s+with\s+and\s+into\s+"
    r"(?P<subject>[a-z0-9&.'’+ -]{2,120}?)"
    r'(?:\s+\((?:the\s+)?[“"]?merger[”"]?\))?\s*,\s*with\s+(?P=subject)\s+'
    r"(?:surviving\s+the\s+merger|continuing)\s+as\s+"
    r"(?:the\s+surviving\s+corporation\s+and\s+as\s+)?"
    r"(?:a\s+)?(?:direct\s+)?wholly[- ]owned\s+subsidiary\s+of\b",
    re.I,
)


def _completed_merger_match_has_uncertainty(text: str, match: re.Match) -> bool:
    """Apply a conservative uncertainty fence to this new, bounded proof form."""

    context = re.sub(r"\s+", " ", text)
    prefix = re.sub(r"\s+", " ", text[:match.start()])
    return bool(
        re.search(
            r"\b(?:if|unless|conditional(?:ly)?|subject\s+to|not|never|no|without|"
            r"unconfirmed|rumou?red|plans?|planned|planning|proposed|future|seeks?|"
            r"seeking|expects?|expected|targets?|targeted|pending|might|could|would|will|"
            r"should|shall|must|anticipates?|anticipated|intends?|intended|scheduled|"
            r"failed|denied|cancelled|canceled|aborted|whether|possible|possibly|potentially)\b",
            context, re.I,
        )
        or _has_stage_proof_uncertainty(prefix)
        or _STAGE_PROOF_HISTORICAL_RE.search(prefix)
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
            if pattern is _COMPLETED_MERGER_SURVIVOR_RE and (
                _completed_merger_match_has_uncertainty(text, match)
            ):
                continue
'''

_OLD_3 = r'''    for pattern in _BOUND_ACQUISITION_SUBJECT_PATTERNS:
        for match in pattern.finditer(quote):
'''
_NEW_3 = r'''    for pattern in _BOUND_ACQUISITION_SUBJECT_PATTERNS:
        for match in pattern.finditer(quote):
            if pattern is _COMPLETED_MERGER_SURVIVOR_RE and (
                _completed_merger_match_has_uncertainty(quote, match)
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
