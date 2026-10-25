"""Host-created whole-product receipts over admitted evidence.

Language interpretation and entailment remain model judgments.
"""

from __future__ import annotations

import hashlib

from collections.abc import Callable, Mapping

MAX_BINDINGS = 3

WHOLE_SCHEMA = "whole_product_coverage_v1"

PRIVATE_KEY = "_validated_product_coverage"

class CoverageError(ValueError):
    pass

class ValidatedWholeProductCoverage(dict):
    """Host-created whole-criterion receipt, never provider JSON authority."""

def validate_whole_coverage(
    raw_status: object, *, criterion: str, whole_status: str,
    bindings: list[dict[str, str]],
    citation_rejection: Callable[[str, str], str | None],
    rejected_citation_sink: list[dict] | None = None,
) -> ValidatedWholeProductCoverage:
    """Bind one native semantic verdict to at most three admitted page quotes."""
    if (not criterion.strip() or not isinstance(raw_status, str)
            or raw_status not in {"VERIFIED", "CONTRADICTED", "UNPROVEN"}
            or raw_status != whole_status):
        raise CoverageError("missing or conflicting whole-product verdict")
    if not isinstance(bindings, list) or len(bindings) > MAX_BINDINGS:
        raise CoverageError("invalid whole-product evidence count")
    if raw_status == "VERIFIED" and not bindings:
        raise CoverageError("whole-product verdict lacks bound evidence")
    validated = []
    for binding in bindings:
        if (not isinstance(binding, Mapping) or set(binding) != {"url", "quote"}
                or not isinstance(binding["url"], str)
                or not isinstance(binding["quote"], str)):
            raise CoverageError("invalid whole-product evidence")
        rejection = citation_rejection(binding["url"], binding["quote"])
        if rejection:
            if rejected_citation_sink is not None:
                rejected_citation_sink.append({**dict(binding), "reason": rejection,
                                               "submitted_status": raw_status})
            raise CoverageError("whole-product citation failed source validation")
        validated.append(dict(binding))
    return ValidatedWholeProductCoverage(
        schema=WHOLE_SCHEMA, criterion_sha256=criterion_hash(criterion),
        status=raw_status, whole_criterion_status=whole_status, evidence=validated,
    )

def criterion_hash(criterion: str) -> str:
    return hashlib.sha256(criterion.encode("utf-8")).hexdigest()

def coverage_matches(value: object, criterion: str) -> bool:
    return (isinstance(value, ValidatedWholeProductCoverage)
            and value.get("schema") == WHOLE_SCHEMA
            and value.get("criterion_sha256") == criterion_hash(criterion)
            and value.get("whole_criterion_status") == "VERIFIED"
            and value.get("status") == "VERIFIED"
            and isinstance(value.get("evidence"), list)
            and 1 <= len(value["evidence"]) <= MAX_BINDINGS)
