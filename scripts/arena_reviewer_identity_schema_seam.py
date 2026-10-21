"""Reach the frozen grader's existing identity repair after semantic review.

Only the three reviewed control-flow deltas from 43a29507f are applied. No
grading implementation, prompt, model, evidence rule or dependency is copied.
"""
from __future__ import annotations

import ast

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "completed_semantic_identity_schema_repair_v1"


class IdentitySchemaSeamError(ValueError):
    """The source does not have the exact reviewed identity-repair shape."""


_INIT_OLD = '''    claims: Mapping[str, Any] = {}
    if investigation_targets:
'''
_INIT_NEW = '''    claims: Mapping[str, Any] = {}
    repair_missing_identity = False
    if investigation_targets:
'''
_GATE_OLD = '''        if (
            positive_semantic_review
            and not _positive_semantic_finding_resolved(
                claims.get("industry")
                if isinstance(claims.get("industry"), Mapping)
                else None
            )
        ):
'''
_GATE_NEW = '''        unresolved_positive_semantics = (
            positive_semantic_review
            and not _positive_semantic_finding_resolved(
                claims.get("industry")
                if isinstance(claims.get("industry"), Mapping)
                else None
            )
        )
        missing_dimensions = _incomplete_company_reverify_dimensions(
            result, icp_attribute=icp_attribute, icp_stage=icp_stage,
        )
        identity_fields = ("observed_company_name", "observed_company_website")
        if company.company_linkedin:
            identity_fields += ("observed_company_linkedin",)
        receipt = result.details.get("investigation_receipt")
        receipt = receipt if isinstance(receipt, Mapping) else {}
        repair_missing_identity = (
            unresolved_positive_semantics
            and receipt.get("completed_submitted_findings") is True
            and receipt.get("failure_reason") == ""
            and "identity" in missing_dimensions
            and "rebrand" not in investigation_targets
            and any(
                not isinstance(verdict.get(field), str) or not verdict[field].strip()
                for field in identity_fields
            )
        )
        # Keep semantic finality, but do not skip an independent schema gap.
        if unresolved_positive_semantics and not repair_missing_identity:
'''
_REPAIR_OLD = '''    repaired_result = _reverify_decision(
        repaired_verdict,
        icp_attribute,
'''
_REPAIR_NEW = '''    if repair_missing_identity:
        # Filling identity must not reopen the already unproven semantic claim.
        repaired_verdict = _clear_reopened_positive_semantics(
            repaired_verdict, active_attribute=False,
        )
    repaired_result = _reverify_decision(
        repaired_verdict,
        icp_attribute,
'''
DELTAS = ((_INIT_OLD, _INIT_NEW), (_GATE_OLD, _GATE_NEW),
          (_REPAIR_OLD, _REPAIR_NEW))


def convert_identity_schema_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise IdentitySchemaSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise IdentitySchemaSeamError("grading source must use LF newlines")
    nodes = [node for node in module.body if isinstance(node, ast.AsyncFunctionDef)
             and node.name == "_llm_reverify_company"]
    if len(nodes) != 1 or _named_bindings(module, "_llm_reverify_company") != nodes:
        raise IdentitySchemaSeamError("identity verifier binding is missing or ambiguous")
    lines = text.splitlines(keepends=True)
    node = nodes[0]
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    corrected = original
    stripped = original
    states = set()
    for old, new in DELTAS:
        counts = original.replace(new, "").count(old), original.count(new)
        if counts == (1, 0):
            states.add("original")
            corrected = corrected.replace(old, new, 1)
            stripped = stripped.replace(old, "", 1)
        elif counts == (0, 1):
            states.add("corrected")
            stripped = stripped.replace(new, "", 1)
        else:
            raise IdentitySchemaSeamError("unrecognized identity correction block")
    if len(states) != 1:
        raise IdentitySchemaSeamError("partially corrected identity source")
    # Added locals must occur only in the exact reviewed blocks. In particular,
    # a second assignment cannot silently disable repair or reopen semantics.
    if any(name in stripped for name in ("repair_missing_identity",
                                        "unresolved_positive_semantics")):
        raise IdentitySchemaSeamError("unexpected identity correction binding")
    if states == {"corrected"}:
        return source
    lines[node.lineno - 1:node.end_lineno] = [corrected]
    result = "".join(lines)
    compile(result, "frozen_identity_schema.py", "exec")
    return result.encode("utf-8")
