"""Shared prompt builder for evidence types without specialised builders.

Used for HIRING, FUNDING, missing, and other evidence types. PART D includes
its own applicability condition inside the prompt.
"""
from typing import Any, Dict

from . import _common
from . import social_posting


def build_verification_prompt(row: Dict[str, Any]) -> str:
    # PART D contains its own applicability condition.
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
