"""Move the frozen grader's existing completed-investigation rules before early exits.

No current grading implementation is imported. Every delta is exact and scoped
inside the existing verifier; unknown or partially applied source fails closed.
"""
from __future__ import annotations
import ast

CORRECTION_ID = "completed_investigation_source_finality_v1"

class InvestigationFinalitySeamError(ValueError):
    pass

SETTLEMENT_BLOCK = '''        refresh_outcome = str(current_profile_cache.get("refresh_outcome") or "")
        receipt = result.details.get("investigation_receipt")
        receipt = receipt if isinstance(receipt, Mapping) else {}
        review_claims = receipt.get("claims")
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
        if (
            result.decision == COMPANY_FIT_UNAVAILABLE
            and _has_explicitly_unproven_fit_dimensions(
                verdict,
                incomplete,
                icp=icp,
                linkedin_refresh_outcome=(
                    CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE
                    if malformed_profile_exhausted else refresh_outcome
                ),
                identity_receipt=(
                    result.details.get("identity_receipt")
                    if isinstance(result.details, Mapping)
                    else None
                ),
                investigation_receipt=(
                    result.details.get("investigation_receipt")
                    if isinstance(result.details, Mapping)
                    else None
                ),
            )
        ):
            return company_fit_unavailable(
                result.reason,
                details={
                    **result.details,
                    "failure_class": (
                        INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS
                    ),
                    **({"investigation_receipt": {
                        **receipt,
                        "prior_linkedin_refresh_failure_reason": (
                            MALFORMED_RESPONSE_FAILURE_REASON
                        ),
                    }} if malformed_profile_exhausted else {}),
                },
            )
        if "employee_size" in incomplete and refresh_outcome == "retryable_failure":
            return _with_verifier_failure_reason(
                company_fit_unavailable(
                    result.reason,
                    details={**result.details,
                             "failure_class": EMPLOYEE_SIZE_VERIFICATION_FAILURE_CLASS},
                ),
                current_profile_cache.get(VERIFIER_FAILURE_DETAIL_KEY),
            )
        return result
'''

_HELPER_HEADER = '''def _settle_completed_company_investigation(
    result: CompanyFitDecisionResult,
    verdict: Mapping[str, Any],
    *,
    investigation_targets: tuple[str, ...],
    icp: ICPPrompt,
    icp_attribute: str,
    icp_stage: str,
    current_profile_cache: Mapping[str, Any],
) -> Optional[CompanyFitDecisionResult]:
    """Reuse bounded-investigation finality before semantic early returns.

    Coverage only selects the existing settlement rules. It does not prove a
    finding, clear a source fault, or bypass identity or research requirements.
    """
    incomplete = _incomplete_company_reverify_dimensions(
        result, icp_attribute=icp_attribute, icp_stage=icp_stage,
    )
    investigated_dimensions = {
        {"stage": "stage", "headcount": "employee_size", "rebrand": "identity",
         "industry": "industry", "required_attribute": "required_attribute",
         "geography": "geography"}[target]
        for target in investigation_targets
    }
    if not incomplete or not investigation_targets or not set(incomplete).issubset(
        investigated_dimensions
    ):
        return None
'''
# Move the exact frozen settlement block; never substitute current grader code.
HELPER = _HELPER_HEADER + "".join(
    line[4:] if line.startswith("    ") else line
    for line in SETTLEMENT_BLOCK.splitlines(keepends=True)
) + "\n\n"

_SETTLEMENT_CALL = '''        return _settle_completed_company_investigation(
            result, verdict, investigation_targets=investigation_targets,
            icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
            current_profile_cache=current_profile_cache,
        )
'''

_EARLY_1_OLD = '''            if not _has_explicitly_unproven_fit_dimensions(
                verdict,
'''

_EARLY_1_NEW = '''            settled = _settle_completed_company_investigation(
                result, verdict, investigation_targets=investigation_targets,
                icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
                current_profile_cache=current_profile_cache,
            )
            if settled is not None:
                return settled
            if not _has_explicitly_unproven_fit_dimensions(
                verdict,
'''

_EARLY_2_OLD = '''                if not _has_explicitly_unproven_fit_dimensions(
                    repaired_verdict,
'''

_EARLY_2_NEW = '''                settled = _settle_completed_company_investigation(
                    repaired_result, repaired_verdict,
                    investigation_targets=post_repair_investigation_targets,
                    icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
                    current_profile_cache=current_profile_cache,
                )
                if settled is not None:
                    return settled
                if not _has_explicitly_unproven_fit_dimensions(
                    repaired_verdict,
'''

DELTAS = (
    (SETTLEMENT_BLOCK, _SETTLEMENT_CALL),
    (_EARLY_1_OLD, _EARLY_1_NEW),
    (_EARLY_2_OLD, _EARLY_2_NEW),
)


def convert_investigation_finality_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise InvestigationFinalitySeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise InvestigationFinalitySeamError("grading source must use LF newlines")
    nodes = [n for n in module.body if isinstance(n, ast.AsyncFunctionDef)
             and n.name == "_llm_reverify_company"]
    helpers = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "_settle_completed_company_investigation"]
    helper_bindings = [
        n for n in ast.walk(module)
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == "_settle_completed_company_investigation"
        ) or (
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            and n.id == "_settle_completed_company_investigation"
        ) or (
            isinstance(n, ast.alias)
            and (n.asname or n.name.split(".")[0])
            == "_settle_completed_company_investigation"
        ) or (
            isinstance(n, ast.ExceptHandler)
            and n.name == "_settle_completed_company_investigation"
        )
    ]
    if helper_bindings != helpers:
        raise InvestigationFinalitySeamError("settlement helper binding is ambiguous")
    if len(nodes) != 1 or len(helpers) > 1:
        raise InvestigationFinalitySeamError("correction function is missing or ambiguous")
    lines = text.splitlines(keepends=True)
    node = nodes[0]
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    states = set()
    corrected = original
    for old, new in DELTAS:
        old, new = old.rstrip("\n"), new.rstrip("\n")
        counts = original.replace(new, "").count(old), original.count(new)
        if counts == (1, 0):
            states.add("original")
            corrected = corrected.replace(old.rstrip("\n"), new.rstrip("\n"), 1)
        elif counts == (0, 1):
            states.add("corrected")
        else:
            raise InvestigationFinalitySeamError("unrecognized correction block")
    if len(states) != 1:
        raise InvestigationFinalitySeamError("partially corrected grading source")
    if states == {"corrected"}:
        if len(helpers) != 1:
            raise InvestigationFinalitySeamError("corrected source is missing its exact helper")
        actual = "".join(lines[helpers[0].lineno - 1:helpers[0].end_lineno])
        if actual.rstrip("\n") != HELPER.rstrip("\n"):
            raise InvestigationFinalitySeamError("unrecognized settlement helper")
        return source
    if helpers:
        raise InvestigationFinalitySeamError("partial settlement helper")
    lines[node.lineno - 1:node.end_lineno] = [HELPER + corrected]
    result = "".join(lines)
    compile(result, "frozen_investigation_finality.py", "exec")
    return result.encode("utf-8")
