"""Default / legacy prompt builder.

Used for any evidence_type that doesn't have its own per-type module:
HIRING, FUNDING, generic ``None``, and any future not-yet-specialised
value.

This refactor preserves byte-identical output for the existing pre-
refactor prompt — which always included PART D, even for non-
SOCIAL_POSTING signals (PART D self-gates on the LLM side).  So the
default builder also passes PART D as an extra part.

Follow-up PR will drop ``PART_D`` from the default builder once each
non-social evidence type has its own dedicated module + test suite.
"""
from typing import Any, Dict

from . import _common
from . import social_posting


def build_verification_prompt(row: Dict[str, Any]) -> str:
    # PART D is included here ONLY to maintain byte-identical output with
    # the pre-refactor prompt during this refactor PR.  Follow-up PR drops
    # this once each evidence type has its own builder.
    return _common.build_verification_prompt(
        row, extra_parts=[social_posting.PART_D_BLOCK],
    )


def build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
) -> str:
    return _common.build_final_judge_prompt(
        row, contents, source_name,
        extra_parts=[social_posting.PART_D_BLOCK],
    )
