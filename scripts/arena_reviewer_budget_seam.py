"""Consume the existing pending final correction at its next bounded turn.

Apply after the reviewer admission interface. Only this reviewed loop predicate
changes; research, validators, counters, limits and all other bytes stay frozen.
"""

from __future__ import annotations

import ast


CORRECTION_ID = "pending_submit_only_final_correction_v1"


class BudgetSeamError(ValueError):
    """The grading source does not contain the reviewed correction shape."""


_OLD = """            for _turn in range(MAX_REASONING_TURNS + 1):
                correction_turn = _turn == MAX_REASONING_TURNS
                if correction_turn and not final_correction_pending:
"""
_NEW = """            for _turn in range(MAX_REASONING_TURNS + 1):
                correction_turn = (
                    _turn == MAX_REASONING_TURNS or final_correction_pending
                )
                if correction_turn and not final_correction_pending:
"""


def convert_budget_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise BudgetSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise BudgetSeamError("grading source must use LF newlines")
    nodes = [node for node in module.body
             if isinstance(node, ast.AsyncFunctionDef)
             and node.name == "investigate_company_evidence"]
    if len(nodes) != 1:
        raise BudgetSeamError("investigator function is missing or ambiguous")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    if (
        "turn_admission = admission.next_turn(" not in body
        or "correction_turn and final_correction_pending" not in body
    ):
        raise BudgetSeamError("reviewer admission interface is missing")
    counts = body.count(_OLD), body.count(_NEW)
    if counts == (0, 1):
        return source
    if counts != (1, 0):
        raise BudgetSeamError("unrecognized or partial pending-correction block")
    lines[node.lineno - 1:node.end_lineno] = [body.replace(_OLD, _NEW, 1)]
    result = "".join(lines)
    compile(result, "frozen_budget_correction.py", "exec")
    return result.encode("utf-8")
