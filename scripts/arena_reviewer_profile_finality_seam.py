"""Reuse the frozen optional-profile finality rule in semantic review.

Only the exact reviewed function shape is accepted. The helper is extracted
from its existing predicate; prompts, evidence predicates and other bytes stay
in the frozen module. No current grading module is copied.
"""
from __future__ import annotations

import ast
import hashlib

CORRECTION_ID = "completed_malformed_profile_semantic_review_v1"
FUNCTION = "_llm_reverify_company"
HELPER = "_completed_malformed_profile_review"

class ProfileFinalitySeamError(ValueError):
    """The frozen source differs from the reviewed control-flow correction."""

ORIGINAL_FUNCTION_SHA256 = 'f5d2eea7a29078956e67897b8b430cb90b5c81bdf6ef2a33bdc730f059a1c87b'
CORRECTED_FUNCTION_SHA256 = '661ac249974f5244cc655a5a99718d329d9a6eaeb7e08de4ed62dace991299c8'
HELPER_SHA256 = 'a50931e4934a6096cfdc55892ffab907d7ab5a3b6a56103f970a6e43b93251c1'

_INLINE_OLD = '''        review_claims = receipt.get("claims")
        review_claims = review_claims if isinstance(review_claims, Mapping) else {}
        review_usage = receipt.get("usage")
        review_usage = review_usage if isinstance(review_usage, Mapping) else {}
        review_fetches = review_usage.get("fetch_outcomes")
        review_fetches = review_fetches if isinstance(review_fetches, list) else []
        current_profile = current_profile_cache.get("evidence")
        # A malformed optional structured profile supplies no evidence. A later
        # independently completed search can exhaust every disputed fact without
        # recovering that profile. This cannot clear a transport/provider fault,
        # an unfinished search, or a positive finding's missing proof.
        malformed_profile_exhausted = bool(
            refresh_outcome == "retryable_failure"
            and current_profile_cache.get("structured_failure") is True
            and current_profile_cache.get("structured_failure_reason")
            == MALFORMED_RESPONSE_FAILURE_REASON
            and current_profile_cache.get(VERIFIER_FAILURE_DETAIL_KEY)
            == MALFORMED_RESPONSE_FAILURE_REASON
            and isinstance(current_profile, Mapping)
            and current_profile.get("outcome")
            == CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE
            and "employee_size" in incomplete
            and receipt.get("gate") == "company_evidence_investigation"
            and receipt.get("completed_submitted_findings") is True
            and receipt.get("failure_reason") == ""
            and all(
                isinstance(outcome, Mapping)
                and isinstance(outcome.get("url_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", outcome["url_sha256"])
                for outcome in review_fetches
            )
            and any(
                outcome.get("ok") is True
                and type(outcome.get("loaded_text_length")) is int
                and outcome["loaded_text_length"] > 0
                for outcome in review_fetches
            )
            and all(
                target is not None
                and target in (receipt.get("targets") or ())
                and isinstance(review_claims.get(target), Mapping)
                and review_claims[target].get("target") == target
                and review_claims[target].get("status") == "UNPROVEN"
                and review_claims[target].get("evidence_url") == ""
                and review_claims[target].get("evidence_quote") == ""
                for target in (
                    {"identity": "rebrand", "employee_size": "headcount",
                     "stage": "stage", "industry": "industry",
                     "required_attribute": "required_attribute",
                     "geography": "geography"}.get(dimension)
                    for dimension in incomplete
                )
            )
        )
'''

_INLINE_NEW = '''        malformed_profile_exhausted = _completed_malformed_profile_review(
            current_profile_cache, receipt, incomplete,
        )
'''

_EARLY_OLD = '''        if (
            positive_semantic_review
            and not _positive_semantic_finding_resolved(
                claims.get("industry")
                if isinstance(claims.get("industry"), Mapping)
                else None
            )
        ):
            if not _has_explicitly_unproven_fit_dimensions(
                verdict,
                _incomplete_company_reverify_dimensions(
                    result, icp_attribute=icp_attribute, icp_stage=icp_stage,
                ),
                icp=icp,
                linkedin_refresh_outcome=str(
                    current_profile_cache.get("refresh_outcome") or ""
                ),
                identity_receipt=result.details.get("identity_receipt"),
                investigation_receipt=result.details.get("investigation_receipt"),
            ):
                # An unresolved semantic judgment cannot terminalize a
                # separate unfinished source or provider obligation.
                return result
            return company_fit_unavailable(
                result.reason,
                details={
                    **result.details,
                    "failure_class": (
                        INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS
                    ),
                },
            )
'''

_EARLY_NEW = '''        if (
            positive_semantic_review
            and not _positive_semantic_finding_resolved(
                claims.get("industry")
                if isinstance(claims.get("industry"), Mapping)
                else None
            )
        ):
            semantic_incomplete = _incomplete_company_reverify_dimensions(
                result, icp_attribute=icp_attribute, icp_stage=icp_stage,
            )
            semantic_receipt = result.details.get("investigation_receipt")
            semantic_receipt = (
                semantic_receipt if isinstance(semantic_receipt, Mapping) else {}
            )
            malformed_profile_exhausted = _completed_malformed_profile_review(
                current_profile_cache, semantic_receipt, semantic_incomplete,
            )
            if not _has_explicitly_unproven_fit_dimensions(
                verdict,
                semantic_incomplete,
                icp=icp,
                linkedin_refresh_outcome=(
                    CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE
                    if malformed_profile_exhausted else str(
                        current_profile_cache.get("refresh_outcome") or ""
                    )
                ),
                identity_receipt=result.details.get("identity_receipt"),
                investigation_receipt=result.details.get("investigation_receipt"),
            ):
                # An unresolved semantic judgment cannot terminalize a
                # separate unfinished source or provider obligation.
                return result
            return company_fit_unavailable(
                result.reason,
                details={
                    **result.details,
                    "failure_class": (
                        INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS
                    ),
                    **({"investigation_receipt": {
                        **semantic_receipt,
                        "prior_linkedin_refresh_failure_reason": (
                            MALFORMED_RESPONSE_FAILURE_REASON
                        ),
                    }} if malformed_profile_exhausted else {}),
                },
            )
'''

_HELPER_HEADER = '''def _completed_malformed_profile_review(
    current_profile_cache: Mapping[str, Any],
    receipt: Mapping[str, Any],
    incomplete: Sequence[str],
) -> bool:
    """Apply the existing completed-search rule for a malformed optional profile."""
    refresh_outcome = str(current_profile_cache.get("refresh_outcome") or "")
'''

_HELPER = _HELPER_HEADER + "".join(
    line[4:] if line.startswith("    ") else line
    for line in _INLINE_OLD.splitlines(keepends=True)
).replace("    malformed_profile_exhausted = bool(", "    return bool(") + "\n\n"
DELTAS = ((_EARLY_OLD, _EARLY_NEW), (_INLINE_OLD, _INLINE_NEW))


def convert_profile_finality_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProfileFinalitySeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise ProfileFinalitySeamError("grading source must use LF newlines")
    functions = [n for n in module.body if isinstance(n, ast.AsyncFunctionDef) and n.name == FUNCTION]
    helpers = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == HELPER]
    helper_bindings = [
        n for n in ast.walk(module)
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == HELPER
        ) or (
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            and n.id == HELPER
        ) or (
            isinstance(n, ast.alias) and (n.asname or n.name.split(".")[0]) == HELPER
        ) or (isinstance(n, ast.ExceptHandler) and n.name == HELPER)
    ]
    if helper_bindings != helpers:
        raise ProfileFinalitySeamError("correction helper binding is ambiguous")
    if len(functions) != 1 or len(helpers) > 1:
        raise ProfileFinalitySeamError("correction function is missing or ambiguous")
    node = functions[0]
    function = ast.get_source_segment(text, node)
    assert function is not None
    digest = hashlib.sha256(function.encode()).hexdigest()
    if helpers:
        helper = ast.get_source_segment(text, helpers[0])
        if (helper is None or hashlib.sha256(helper.encode()).hexdigest() != HELPER_SHA256
            or digest != CORRECTED_FUNCTION_SHA256):
            raise ProfileFinalitySeamError("partially corrected grading source")
        return source
    if digest != ORIGINAL_FUNCTION_SHA256:
        raise ProfileFinalitySeamError("unrecognized frozen correction function")
    corrected = function
    for old, new in DELTAS:
        if corrected.count(old) != 1 or new in corrected:
            raise ProfileFinalitySeamError("unrecognized or partial correction block")
        corrected = corrected.replace(old, new, 1)
    if (hashlib.sha256(corrected.encode()).hexdigest() != CORRECTED_FUNCTION_SHA256
        or hashlib.sha256(_HELPER.rstrip("\n").encode()).hexdigest() != HELPER_SHA256):
        raise ProfileFinalitySeamError("correction bytes differ from reviewed source")
    lines = text.splitlines(keepends=True)
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    assert original == function + "\n"
    lines[node.lineno - 1:node.end_lineno] = [_HELPER + corrected + "\n"]
    result = "".join(lines)
    compile(result, "frozen_profile_finality.py", "exec")
    return result.encode("utf-8")
