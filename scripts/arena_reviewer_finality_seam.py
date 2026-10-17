"""Apply only the explicitly approved PR483 terminal-negative correction.

The input is extracted from the frozen grading image. No current grading
module is copied. Every byte outside the one reviewed predicate block stays
unchanged, including affirmative evidence and qualification rules.
"""

from __future__ import annotations

import ast


CORRECTION_ID = "completed_negative_empty_observed_linkedin_v1"
PREDICATE = "_has_explicitly_unproven_fit_dimensions"


class FinalitySeamError(ValueError):
    """The frozen predicate does not have the approved PR483 shape."""


_ORIGINAL_GUARD = '''                    and all(
                        isinstance(identity_receipt.get(field), str)
                        and bool(identity_receipt[field].strip())
                        for field in (
                            "submitted_name", "submitted_domain",
                            "observed_name", "observed_domain",
                            "observed_linkedin_slug",
                        )
                    )
'''
_CORRECTED_GUARD = '''                    # An explicit empty observed slug can also exhaust a
                    # negative identity investigation. It proves no match.
                    and isinstance(
                        identity_receipt.get("observed_linkedin_slug"), str
                    )
                    and (
                        identity_receipt["observed_linkedin_slug"] == ""
                        or bool(identity_receipt["observed_linkedin_slug"].strip())
                    )
                    and all(
                        isinstance(identity_receipt.get(field), str)
                        and bool(identity_receipt[field].strip())
                        for field in (
                            "submitted_name", "submitted_domain",
                            "observed_name", "observed_domain",
                        )
                    )
'''


def convert_finality_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FinalitySeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise FinalitySeamError("grading source must use LF newlines")
    predicates = [node for node in module.body
                  if isinstance(node, ast.FunctionDef) and node.name == PREDICATE]
    if len(predicates) != 1:
        raise FinalitySeamError("finality predicate is missing or ambiguous")
    node = predicates[0]
    lines = text.splitlines(keepends=True)
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    old_count = original.count(_ORIGINAL_GUARD)
    new_count = original.count(_CORRECTED_GUARD)
    if old_count == 1 and new_count == 0:
        corrected = original.replace(_ORIGINAL_GUARD, _CORRECTED_GUARD, 1)
    elif old_count == 0 and new_count == 1:
        return source
    else:
        raise FinalitySeamError("unrecognized or partial finality predicate")
    lines[node.lineno - 1:node.end_lineno] = [corrected]
    result = "".join(lines)
    compile(result, "lead_scorer.py", "exec")
    return result.encode("utf-8")
