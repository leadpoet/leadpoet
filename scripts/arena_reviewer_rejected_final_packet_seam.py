"""Keep rejected final submissions incomplete in an otherwise frozen grader.

Only the five reviewed PR671 lines are inserted. Validators, valid negative
findings, prior repairs, provider failure precedence and retry limits stay frozen.
"""
from __future__ import annotations

import ast

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "rejected_final_packet_incomplete_v1"


class RejectedFinalPacketSeamError(ValueError):
    """The frozen source does not have the exact reviewed completion shape."""


GUARD = '''                        # Validation can downgrade rejected positive proof to
                        # UNPROVEN for repair. A rejected final packet is still
                        # incomplete; that projection is not a valid negative.
                        if rejected and not failure_reason:
                            failure_reason = MALFORMED_RESPONSE_FAILURE_REASON
'''
OLD = '''                        failure_reason = (
                            required_stage_discovery_failure_reason
                            if required_stage_incomplete
                            else PROVIDER_ERROR_FAILURE_REASON
                            if unresolved_source
                            else ""
                        )
                        # A complete validated judgment may finish UNPROVEN
                        # within bounded research. Missing submissions and
                        # required discovery failures remain incomplete;
                        # source transport faults remain in fetch_outcomes.
                        if failure_reason:
                            _record_failure(diagnostic, failure_reason)
                        return {
                            "claims": claims,
                            "_completed_submit": not failure_reason,
'''
NEW = OLD.replace(
    "                        # A complete validated judgment",
    GUARD + "                        # A complete validated judgment", 1,
)


def _guard_conditions(node: ast.AsyncFunctionDef) -> list[ast.expr]:
    return [
        item.test for item in ast.walk(node)
        if isinstance(item, ast.If)
        and {"rejected", "failure_reason"}.issubset({
            value.id for value in ast.walk(item.test)
            if isinstance(value, ast.Name)
        })
    ]


def convert_rejected_final_packet_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RejectedFinalPacketSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise RejectedFinalPacketSeamError("grading source must use LF newlines")
    nodes = [node for node in module.body if isinstance(node, ast.AsyncFunctionDef)
             and node.name == "investigate_company_evidence"]
    if (len(nodes) != 1
        or _named_bindings(module, "investigate_company_evidence") != nodes):
        raise RejectedFinalPacketSeamError("investigator binding is missing or ambiguous")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    counts = body.count(OLD), body.count(NEW)
    conditions = _guard_conditions(node)
    expected = ast.parse("rejected and not failure_reason", mode="eval").body
    if counts == (0, 1):
        guard_line = node.lineno + body[:body.index(NEW) + NEW.index(
            "                        if rejected and not failure_reason:"
        )].count("\n")
        if (len(conditions) != 1 or ast.dump(conditions[0]) != ast.dump(expected)
            or conditions[0].lineno != guard_line):
            raise RejectedFinalPacketSeamError("final rejection guard is not executable or unique")
        return source
    if counts != (1, 0) or conditions:
        raise RejectedFinalPacketSeamError("unrecognized or partial final completion block")
    lines[node.lineno - 1:node.end_lineno] = [body.replace(OLD, NEW, 1)]
    result = "".join(lines).encode("utf-8")
    # Reparse the candidate so a matching string literal cannot pose as code.
    convert_rejected_final_packet_seam(result)
    return result
