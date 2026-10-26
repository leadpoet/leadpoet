"""Arena scoring: the plain scorer policy, scoring assignments, judge
execution, and per-run score calculation.

The competition scorer is imported lazily inside
the worker only after the policy has been applied to the process environment,
because it reads credentials and behavior knobs at import time.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from lab_arena import contracts, verify
from lab_arena.contracts import ArenaContractError

SCORING_ADAPTER_VERSION_V1 = "qualification_style_v1"
# Every OpenRouter model the Arena judge calls, by the judge's own role
# for it. The broker refuses any other model on a scoring run, so this mapping
# must cover the judge image exactly; ``test_lab_arena_judge_routes`` scans
# the judge's source for model literals against it.
DEFAULT_JUDGE_MODELS = {
    "company_fit_reverification": "perplexity/sonar",
    "intent_signal_judge": "openai/gpt-6-luna",
    "intent_details_review": "anthropic/claude-sonnet-5.5",
    "intent_three_stage_stage3": "perplexity/sonar-pro",
    "intent_fetched_evidence": "openai/gpt-6-luna",
    "role_batch_check": "google/gemini-3.5-flash-lite",
    "industry_evidence_review": "openai/gpt-6.1-sol",
}
CREDENTIAL_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "SCRAPINGDOG_API_KEY",
    "EXA_API_KEY",
    "DEEPLINE_API_KEY",
)
MAX_JUDGE_RETRIES = 3
FAILURE_REASON_CODES = frozenset(
    {
        "source_blocked",
        "malformed_response",
        "provider_error",
        "unexpected_verifier_error",
        "admission_budget_interrupted",
        "unknown",
    }
)
FAILURE_REASON_DETAIL_KEY = "failure_reason_code"


def _validated_failure_reason(value: Any) -> str:
    """Project arbitrary diagnostic input onto the fixed public enum."""

    return value if isinstance(value, str) and value in FAILURE_REASON_CODES else ""


class ScoringError(RuntimeError):
    """Scoring infrastructure failed; the round cancels if the window closes."""

    def __init__(self, message: str, *, failure_reason: str = "") -> None:
        super().__init__(message)
        self.failure_reason = _validated_failure_reason(failure_reason)


class ScorerPolicyConflict(ScoringError):
    """The process environment conflicts with the signed scorer policy."""



# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def build_scorer_policy(
    *,
    judge_models: Mapping[str, str] = DEFAULT_JUDGE_MODELS,
    provider_profile: str = "lab_arena",
    scoring_adapter_version: str = SCORING_ADAPTER_VERSION_V1,
    company_quality: bool = False,
    company_judgment_cache: bool = False,
    intent_details: bool = False,
    provider_observation_handoff: bool = False,
    normalize_intent_scale: bool = False,
    fit_fp_receipt_authority: bool = False,
) -> Dict[str, Any]:
    """Return the plain scorer settings used for every participant."""

    return contracts.validate_scorer_policy({
        "schema_version": contracts.SCORER_POLICY_SCHEMA_VERSION,
        "scoring_adapter_version": scoring_adapter_version,
        **({"company_quality_policy": "company_quality_v1"} if company_quality else {}),
        **({"company_judgment_policy": "company_judgment_v1"} if company_judgment_cache else {}),
        **({"intent_details_policy": "intent_details_v1"} if intent_details else {}),
        "fp_penalty_points": 10.0,
        "fp_unverified_primary_penalty_points": 10.0,
        "fp_penalty_icp_floor": 0.0,
        "company_cap_rule": "icp_max_companies",
        "max_scored_companies": 0,
        "judge_models": dict(judge_models),
        "provider_profile": provider_profile,
        "pre_slice_rule": "first_n_model_order",
        "employee_bucket_rule": "lab_relaxed_buckets",
        "env_bindings": {
            **({
                contracts.SCORE_NORMALIZATION_BINDING:
                    contracts.AVAILABLE_INTENT_CAP_NORMALIZATION,
            } if normalize_intent_scale else {}),
            **({
                contracts.PROVIDER_OBSERVATION_HANDOFF_BINDING:
                    contracts.AUTHENTICATED_PROVIDER_OBSERVATION_HANDOFF,
            } if provider_observation_handoff else {}),
            **({
                contracts.FIT_FP_CLASSIFICATION_BINDING:
                    contracts.FIT_RECEIPT_AUTHORITY,
            } if fit_fp_receipt_authority else {}),
        },
    })


def apply_policy_to_environment(
    policy: Mapping[str, Any],
    *,
    environ: MutableMapping[str, str],
    credentials: Mapping[str, str],
) -> str:
    """Bind the policy into ``environ`` before the evaluator is imported.

    Refuses to start when any bound variable already holds a conflicting
    value or when a credential is missing. Returns the applied scoring
    adapter version.
    """

    validated = contracts.validate_scorer_policy(policy)
    bindings = dict(validated["env_bindings"])
    # A new image can serve an older round. Preserve its frozen reviewer.
    models = validated["judge_models"]
    evidence_model = models.get("intent_fetched_evidence") or models.get(
        "intent_three_stage_stage3"
    )
    model_bindings = {
        "ARENA_INTENT_EVIDENCE_MODEL": evidence_model,
        "ARENA_INTENT_DETAILS_REVIEW_MODEL": models.get("intent_details_review"),
        "ARENA_ROLE_BATCH_MODEL": models.get("role_batch_check"),
    }
    # Absence in a frozen older policy explicitly disables the new review.
    # An ambient setting must not change an older round's judging behavior.
    industry_model = models.get("industry_evidence_review", "")
    if bindings.get("ARENA_INDUSTRY_EVIDENCE_MODEL", industry_model) != industry_model:
        raise ScorerPolicyConflict("industry reviewer conflicts with the frozen model policy")
    bindings["ARENA_INDUSTRY_EVIDENCE_MODEL"] = industry_model
    for name, model in model_bindings.items():
        if model:
            if bindings.get(name, model) != model:
                raise ScorerPolicyConflict("reviewer conflicts with the frozen model policy")
            bindings[name] = model
    for name, value in bindings.items():
        existing = environ.get(name)
        if existing is not None and existing != value:
            raise ScorerPolicyConflict("environment %s conflicts with the scorer policy" % name)
    for name in CREDENTIAL_ENV_NAMES:
        secret = credentials.get(name)
        if not secret:
            raise ScorerPolicyConflict("scoring credential %s is missing" % name)
        existing = environ.get(name)
        if existing is not None and existing != secret:
            raise ScorerPolicyConflict("environment %s conflicts with the Arena scoring credential" % name)
    for name, value in bindings.items():
        environ[name] = value
    for name in CREDENTIAL_ENV_NAMES:
        environ[name] = credentials[name]
    return str(validated["scoring_adapter_version"])


# ---------------------------------------------------------------------------
# Scoring plan (section 12.1 as revised: one work item per accepted assignment)
# ---------------------------------------------------------------------------


def build_scoring_plan(
    *,
    round_id: str,
    stage: int,
    runs: Sequence[Mapping[str, Any]],
    execution_sequence_policy: Optional[str] = None,
    configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Plan from the frozen stage result set.

    ``runs`` are every attempt row of the stage. An assignment with an
    accepted attempt contributes exactly one work item named by its run and
    output reference. Confirmed miner-caused failures retain their ordinary
    zero. Infrastructure failures stay explicitly incomplete, including when
    their retry budget or frozen window ends. They never erase completed work
    or receive an invented score.
    """

    if stage not in (1, 2):
        raise ArenaContractError("stage must be 1 or 2")
    positions = contracts.execution_positions(stage, execution_sequence_policy, configuration)
    latest: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    accepted: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        if int(run["stage"]) != stage:
            continue
        key = (str(run["submission_id"]), int(run["icp_position"]))
        if run.get("status") == "accepted":
            accepted[key] = run
        current = latest.get(key)
        if current is None or int(run.get("attempt") or 0) >= int(current.get("attempt") or 0):
            latest[key] = run
    items: Dict[str, Dict[str, Any]] = {}
    zero_rows: List[Dict[str, Any]] = []
    incomplete_rows: List[Dict[str, Any]] = []
    for key in sorted(latest):
        submission_id, position = key
        if position not in positions:
            raise ArenaContractError("run position %d is outside stage %d" % (position, stage))
        run = accepted.get(key)
        if run is not None:
            scored_run_id = str(run.get("run_id") or "")
            output_ref = str(run.get("output_ref") or "")
            if not scored_run_id or not output_ref:
                raise ArenaContractError("accepted run has no output reference")
            # One work item per miner and ICP: identical outputs from two miners
            # are still judged as distinct competition results.
            items[scored_run_id] = {"scored_run_id": scored_run_id, "icp_position": position, "output_ref": output_ref, "submission_id": submission_id}
            continue
        latest_run = latest[key]
        cause = str(latest_run.get("terminal_cause") or "")
        if cause not in contracts.MODEL_CAUSED_TERMINAL_CAUSES:
            # A confirmed miner-caused failure keeps its ordinary zero when
            # a later infrastructure retry could not produce accepted work.
            confirmed = [candidate for candidate in runs
                if (str(candidate.get("submission_id")), int(candidate.get("icp_position") or 0)) == key
                and candidate.get("terminal_cause") in contracts.MODEL_CAUSED_TERMINAL_CAUSES]
            if confirmed:
                cause = str(max(confirmed, key=lambda candidate: int(candidate.get("attempt") or 0))["terminal_cause"])
            elif latest_run.get("status") == "failed" and cause in (
                contracts.INFRASTRUCTURE_TERMINAL_CAUSES | {"stage_closed"}
            ):
                incomplete_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
                continue
            else:
                raise ArenaContractError("assignment %s/%d has invalid terminal evidence (%s)" % (submission_id, position, cause or "none"))
        zero_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "stage": stage,
        "work_items": [dict(item) for _, item in sorted(items.items())],
        "zero_rows": sorted(zero_rows, key=lambda row: (row["submission_id"], row["icp_position"])),
        **({"incomplete_rows": incomplete_rows} if incomplete_rows else {}),
    }
    if execution_sequence_policy is not None:
        plan["execution_sequence_policy"] = execution_sequence_policy
    return contracts.validate_scoring_plan(plan, configuration)


# ---------------------------------------------------------------------------
# Scoring workers
# ---------------------------------------------------------------------------

Scorer = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], bool], Any]
_SCOPED_RETRY_EVIDENCE_RUNNER = "_score_with_retry_evidence"


def lab_scorer(
    policy: Mapping[str, Any],
    *,
    contact_source_evidence: Optional[Mapping[str, Any]] = None,
    provider_observations: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Scorer:
    """The Lab scorer on its host path, constructed after the policy is applied.

    ``is_reference_model`` is always False: the king is a competitor, not the
    Lab's reference model.
    """

    from qualification.scoring.competition import CompetitionCompanyScorer

    validated = contracts.validate_scorer_policy(policy)
    adapter = validated["scoring_adapter_version"]
    _lab_adapter_version(adapter)
    from lab_arena import contact_policy, quality_policy
    scorer = CompetitionCompanyScorer(
        integrity_policy=contact_policy.integrity_adapter(adapter),
        contacts_required=contact_policy.scorer_enabled(validated),
        contact_source_evidence=contact_source_evidence,
        provider_observations=provider_observations,
        company_quality=quality_policy.scorer_enabled(validated),
        evidence_investigator=True,
        scorer_policy=validated,
    )

    def score(companies: Sequence[Mapping[str, Any]], icp: Mapping[str, Any], is_reference_model: bool) -> Any:
        return scorer.score_with_breakdowns(list(companies), dict(icp), bool(is_reference_model))

    def score_with_retry_evidence(
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
        retry_evidence_scope: MutableMapping[str, Any],
        *,
        original_company_indexes: Sequence[int],
        prior_terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,
        on_company_breakdown: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Any:
        return scorer.score_with_breakdowns(
            list(companies),
            dict(icp),
            bool(is_reference_model),
            retry_evidence_scope=retry_evidence_scope,
            original_company_indexes=original_company_indexes,
            **({"prior_terminal_breakdowns": prior_terminal_breakdowns,
                "on_company_breakdown": on_company_breakdown}
               if on_company_breakdown is not None else {}),
        )

    score.integrity_policy = contact_policy.integrity_adapter(adapter)
    score.contacts_required = contact_policy.scorer_enabled(validated)
    score.company_quality = quality_policy.scorer_enabled(validated)
    score.evidence_investigator = True
    setattr(score, _SCOPED_RETRY_EVIDENCE_RUNNER, score_with_retry_evidence)
    return score


def _lab_adapter_version(arena_version: str) -> str:
    from qualification.scoring.competition import SCORING_ADAPTER_VERSION

    if arena_version in (SCORING_ADAPTER_VERSION_V1, "qualification_integrity_v2", "qualification_contacts_v3"):
        return SCORING_ADAPTER_VERSION
    raise ScoringError("unsupported scoring adapter version")


def _run_scorer(
    scorer: Scorer,
    companies: Sequence[Mapping[str, Any]],
    icp: Mapping[str, Any],
    *,
    retry_evidence_scope: MutableMapping[str, Any],
    original_company_indexes: Sequence[int],
    prior_terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,
    on_company_breakdown: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    scoped_runner = getattr(scorer, _SCOPED_RETRY_EVIDENCE_RUNNER, None)
    result = (
        scoped_runner(companies, icp, False, retry_evidence_scope,
                      original_company_indexes=original_company_indexes,
                      **({"prior_terminal_breakdowns": prior_terminal_breakdowns,
                          "on_company_breakdown": on_company_breakdown}
                         if on_company_breakdown is not None else {}))
        if callable(scoped_runner)
        else scorer(companies, icp, False)
    )
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    if not isinstance(result, (list, tuple)):
        raise ScoringError("scorer returned a non-list result")
    return [dict(item) for item in result]


def _retryable_breakdown_failure_reason(breakdown: Mapping[str, Any]) -> str:
    """Return only a fixed verifier reason from one retryable breakdown."""

    receipts = breakdown.get("verifier_gate_receipts")
    if isinstance(receipts, Sequence) and not isinstance(
        receipts, (str, bytes, bytearray)
    ):
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            reason = _validated_failure_reason(
                receipt.get(FAILURE_REASON_DETAIL_KEY)
            )
            if reason and reason != "unknown":
                return reason
        # A completed investigation can still lack usable evidence because a
        # fetched page was only a provider diagnostic. Keep explicit verifier
        # reasons above; this fallback names the transport fault, not credit.
        for receipt in receipts:
            if not isinstance(receipt, Mapping) or (
                receipt.get("gate") != "company_fit"
                or receipt.get("decision") != "unavailable"
            ):
                continue
            supporting = receipt.get("supporting_receipts")
            if not isinstance(supporting, list):
                continue
            for investigation in supporting:
                if not isinstance(investigation, Mapping) or (
                    investigation.get("gate") != "company_evidence_investigation"
                    or investigation.get("completed_submitted_findings") is not True
                    or investigation.get("failure_reason") not in (None, "")
                ):
                    continue
                usage = investigation.get("usage")
                outcomes = usage.get("fetch_outcomes") if isinstance(usage, Mapping) else None
                if isinstance(outcomes, list) and any(
                    isinstance(outcome, Mapping)
                    and outcome.get("ok") is False
                    and outcome.get("error_class") == "provider_diagnostic_body"
                    for outcome in outcomes
                ):
                    return "provider_error"
    return "unknown"


_PRIVATE_FIT_DIMENSIONS = (
    "identity", "employee_size", "industry", "geography", "stage",
    "required_attribute",
)
_PRIVATE_FIT_CLASSES = frozenset({
    "insufficient_fit_evidence", "employee_size_verification_failed",
    "required_attribute_quote_absent", "company_verification_exhausted",
    "model_contract_incompatible",
})
_PRIVATE_FETCH_CLASSES = (
    "http_error", "source_not_found", "source_gone", "invalid_url", "unsupported_binary_content",
    "empty_page", "provider_request_refused", "provider_diagnostic_body",
    "fetch_failed",
)


def _private_retryable_fit_diagnostic(breakdown: Mapping[str, Any]) -> str:
    """Project one failed fit receipt to fixed labels, never source content."""

    receipts = breakdown.get("verifier_gate_receipts")
    if not isinstance(receipts, list):
        return "fit=absent"
    fit = next((row for row in receipts if isinstance(row, Mapping)
                and row.get("gate") == "company_fit"
                and row.get("decision") == "unavailable"), None)
    if fit is None:
        return "fit=absent"
    decisions = fit.get("company_fit_dimensions")
    decisions = decisions if isinstance(decisions, Mapping) else {}
    dimensions = [name for name in _PRIVATE_FIT_DIMENSIONS
                  if (decisions.get(name) == "unavailable" or
                      (name == "required_attribute" and
                       fit.get("required_attribute_decision") == "unavailable"))]
    failure_class = fit.get("failure_class")
    failure_class = (
        failure_class if isinstance(failure_class, str)
        and failure_class in _PRIVATE_FIT_CLASSES else "missing"
    )
    reason = _validated_failure_reason(fit.get(FAILURE_REASON_DETAIL_KEY)) or "missing"
    supporting = fit.get("supporting_receipts")
    investigation = next((row for row in supporting if isinstance(row, Mapping)
                          and row.get("gate") == "company_evidence_investigation"), None) if isinstance(supporting, list) else None
    completed = (
        "yes" if investigation.get("completed_submitted_findings") is True else "no"
    ) if investigation is not None else "missing"
    investigation_reason = (
        _validated_failure_reason(investigation.get("failure_reason")) or "none"
    ) if investigation is not None else "missing"
    usage = investigation.get("usage") if investigation is not None else None
    outcomes = usage.get("fetch_outcomes") if isinstance(usage, Mapping) else None
    counts = {name: 0 for name in ("ok", *_PRIVATE_FETCH_CLASSES, "other")}
    if isinstance(outcomes, list):
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                counts["other"] += 1
            elif outcome.get("ok") is True:
                counts["ok"] += 1
            else:
                error_class = outcome.get("error_class")
                counts[error_class if error_class in _PRIVATE_FETCH_CLASSES else "other"] += 1
    fetch = ",".join("%s:%d" % (name, min(count, 99))
                     for name, count in counts.items() if count) or "none"
    return (
        "fit_dims=%s fit_class=%s fit_reason=%s inv_complete=%s "
        "inv_reason=%s fetch=%s"
        % (",".join(dimensions) or "none", failure_class, reason,
           completed, investigation_reason, fetch)
    )


def _private_retryable_failure_diagnostic(
    failures: Sequence[Mapping[str, Any]],
) -> str:
    for failure in failures:
        diagnostic = _private_retryable_fit_diagnostic(failure)
        if diagnostic != "fit=absent":
            return diagnostic
    return "fit=absent"


def _has_unique_scored_company_names(
    companies: Sequence[Mapping[str, Any]], scored_indexes: Sequence[int],
) -> bool:
    """Whether company-local retries cannot change the duplicate-name gate."""

    names: List[str] = []
    for index in scored_indexes:
        company = companies[index]
        if not isinstance(company, Mapping):
            return False
        name = company.get("company_name")
        if not isinstance(name, str) or not name.strip():
            return False
        names.append(name.lower().strip())
    return bool(names) and len(names) == len(set(names))


def score_work_item(
    item: Mapping[str, Any],
    *,
    icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]],
    scorer: Scorer,
    max_retries: int = MAX_JUDGE_RETRIES,
    max_scored_companies: int = 0,
    _original_company_indexes: Optional[Sequence[int]] = None,
    prior_terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,
    on_terminal_breakdowns: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    """Score one distinct output once: the first-N slice in model order.

    For scored companies with unique, nonempty names, keep each company's
    first terminal breakdown and retry only unresolved companies. Duplicate or
    missing names keep whole-item retries because the scorer's duplicate gate
    depends on batch order. Exhausted source-local verification becomes an
    explicit company zero. Systemic exhaustion raises ``ScoringError``. The
    scorer must return exactly one breakdown per company it scores under the
    bucket-skip rule the verifier recomputes; any other count is a scorer
    contract failure. The legacy integrity checkpoint callback retains
    terminal rows during the first grouped invocation. Its retries walk the
    same ordered group and restore prior alias context without paid rejudging.
    """

    from qualification.scoring.competition import (
        scorer_breakdown_has_company_local_verification_failure,
        scorer_breakdown_has_retryable_infrastructure_failure,
        terminal_company_verification_breakdown,
    )

    sliced = verify.slice_first_n(companies, verify.icp_company_goal(icp))
    original_indexes = (
        list(range(len(companies))) if _original_company_indexes is None
        else list(_original_company_indexes)
    )
    # This private map selects execution-owned observation metadata only.
    # Breakdown indexes still use the invocation's relative numbering below.
    if len(original_indexes) != len(companies) or any(
        type(index) is not int or index < 0 for index in original_indexes
    ):
        raise ScoringError("original company index mapping is invalid")
    scored_indexes, _skipped = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)
    integrity_policy = bool(getattr(scorer, "integrity_policy", False))
    retain_terminal = integrity_policy or _has_unique_scored_company_names(sliced, scored_indexes)
    retained: List[Optional[Dict[str, Any]]] = [None] * len(scored_indexes)
    checkpoint_mode = bool(
        integrity_policy and not getattr(scorer, "company_quality", False)
        and callable(getattr(scorer, _SCOPED_RETRY_EVIDENCE_RUNNER, None))
        and (on_terminal_breakdowns is not None or prior_terminal_breakdowns is not None)
    )
    if prior_terminal_breakdowns is not None and not checkpoint_mode:
        raise ScoringError("terminal breakdown resume requires legacy integrity scoring")
    position_by_index = {index: position for position, index in enumerate(scored_indexes)}
    for row in prior_terminal_breakdowns or []:
        index = row.get("company_index")
        if index not in position_by_index or retained[position_by_index[index]] is not None:
            raise ScoringError("terminal breakdown resume index is invalid")
        if scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True):
            raise ScoringError("terminal breakdown resume is retryable")
        retained[position_by_index[index]] = dict(row)

    def checkpoint():
        if on_terminal_breakdowns is not None:
            on_terminal_breakdowns([dict(row) for row in retained if row is not None])

    def retain_completed(row):
        index = row.get("company_index")
        if type(index) is not int or index not in position_by_index:
            raise ScoringError("integrity breakdown company index mismatch")
        position = position_by_index[index]
        if ((retained[position] is None or (
                row.get("duplicate_company") is True
                and retained[position].get("duplicate_company") is not True))
            and not scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True)):
            retained[position] = dict(row)
            checkpoint()

    unresolved = [position for position, row in enumerate(retained) if row is None]
    if prior_terminal_breakdowns:
        checkpoint()
    last_failure_reason = ""
    last_private_diagnostic = "origin=exception"
    # This private dictionary is the complete lifetime boundary for source text
    # retained between retries. A later score_work_item call always gets a new
    # scope, even when its caller reuses the same scorer object.
    retry_evidence_scope: Dict[str, Any] = {}
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        if retain_terminal and attempt > 0 and not checkpoint_mode:
            invoked_positions = list(unresolved)
            invoked_companies = [
                sliced[scored_indexes[position]] for position in invoked_positions
            ]
            invoked_indexes = [
                original_indexes[scored_indexes[position]]
                for position in invoked_positions
            ]
        else:
            invoked_positions = list(range(len(scored_indexes)))
            invoked_companies = sliced
            invoked_indexes = original_indexes[:len(sliced)]
        try:
            breakdowns = _run_scorer(
                scorer,
                invoked_companies,
                icp,
                retry_evidence_scope=retry_evidence_scope,
                original_company_indexes=invoked_indexes,
                **({"prior_terminal_breakdowns": [dict(row) for row in retained if row is not None],
                    "on_company_breakdown": retain_completed} if checkpoint_mode else {}),
            )
        except Exception as exc:  # judge/provider failure: retry unresolved input
            last_failure_reason = _validated_failure_reason(
                getattr(exc, "failure_reason", "")
            ) or "unknown"
            last_private_diagnostic = "origin=exception"
            continue
        if len(breakdowns) != len(invoked_positions):
            raise ScoringError(
                "scorer returned %d breakdowns for %d scored companies"
                % (len(breakdowns), len(invoked_positions))
            )
        failed = [
            item_row
            for item_row in breakdowns
            if scorer_breakdown_has_retryable_infrastructure_failure(
                item_row, integrity_policy=integrity_policy
            )
        ]
        if not retain_terminal:
            if failed:
                if (
                    attempt == attempts - 1
                    and all(
                        scorer_breakdown_has_company_local_verification_failure(row)
                        for row in failed
                    )
                ):
                    return [
                        terminal_company_verification_breakdown(row)
                        if scorer_breakdown_has_retryable_infrastructure_failure(
                            row, integrity_policy=integrity_policy
                        )
                        else row
                        for row in breakdowns
                    ]
                last_failure_reason = _retryable_breakdown_failure_reason(
                    failed[0]
                )
                last_private_diagnostic = _private_retryable_failure_diagnostic(failed)
                continue
            return breakdowns
        for relative_index, (position, breakdown) in enumerate(zip(invoked_positions, breakdowns)):
            if integrity_policy:
                # Initial sparse indexes refer to the full first-N slice;
                # subsequent invocations contain only unresolved companies.
                expected_index = scored_indexes[position] if attempt == 0 or checkpoint_mode else relative_index
                if breakdown.get("company_index") != expected_index:
                    raise ScoringError("integrity breakdown company index mismatch")
                breakdown = {**breakdown, "company_index": scored_indexes[position]}
            retryable = scorer_breakdown_has_retryable_infrastructure_failure(
                breakdown, integrity_policy=integrity_policy
            )
            if not retryable:
                if retained[position] is None:
                    retained[position] = breakdown
                    checkpoint()
            elif (
                attempt == attempts - 1
                and scorer_breakdown_has_company_local_verification_failure(
                    breakdown
                )
            ):
                retained[position] = terminal_company_verification_breakdown(
                    breakdown
                )
                checkpoint()
        unresolved = [
            position for position, breakdown in enumerate(retained)
            if breakdown is None
        ]
        if not unresolved:
            result = [dict(breakdown) for breakdown in retained if breakdown is not None]
            if integrity_policy:
                # A recovered company can resolve to an earlier, already
                # accepted identity. Regroup in original order without
                # invoking the judge again or replacing its first verdict.
                from qualification.scoring.competition import regroup_legacy_company_breakdowns
                result = regroup_legacy_company_breakdowns(sliced, result)
            return result
        unresolved_set = set(unresolved)
        unresolved_failures = [
            breakdown
            for position, breakdown in zip(invoked_positions, breakdowns)
            if position in unresolved_set
        ]
        if unresolved_failures:
            failure = unresolved_failures[0]
            last_failure_reason = _retryable_breakdown_failure_reason(failure)
            last_private_diagnostic = _private_retryable_failure_diagnostic(
                unresolved_failures
            )
            continue
    raise ScoringError(
        ("judge_exhausted attempts=%d reason=%s %s"
         % (attempts, last_failure_reason or "unknown", last_private_diagnostic))[
             :MAX_FAILURE_DETAIL_CHARS
         ],
        failure_reason=last_failure_reason,
    )


def project_company_judgments(*args, **kwargs):
    """Convert invalid context replay into the gateway's scoring failure type."""
    from qualification.scoring.competition import apply_company_judgment_context
    try:
        return apply_company_judgment_context(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ScoringError("company judgment destination context is invalid") from exc


def _score_legacy_company_work_item(
    item, *, icp, companies, scorer, cache_context, max_scored_companies=0,
    on_company_judgments=None,
):
    from lab_arena import company_judgments
    from qualification.scoring.competition import apply_company_judgment_context, raw_company_judgment
    lease = company_judgments.validate_lease_context(cache_context)
    sliced = verify.slice_first_n(companies, verify.icp_company_goal(icp))
    refs = sorted(lease["hits"] + lease["misses"], key=lambda ref: ref["company_index"])
    if len(refs) != len(sliced):
        raise ScoringError("company cache lease does not cover this output")
    raw_by_key = {hit["cache_key"]: hit["evidence_doc"]["raw_judgment"] for hit in lease["hits"]}
    first_miss = {}
    for miss in lease["misses"]:
        first_miss.setdefault(miss["cache_key"], miss)
    new_judgments = []
    def resolve(index):
        ref = refs[index]
        key = ref["cache_key"]
        if key not in raw_by_key:
            from contextlib import nullcontext
            from qualification.scoring.signal_judgment_cache import install_exchange
            signal_exchange = getattr(scorer, "_signal_judgment_exchange", None)
            async def exchange_signal(request):
                return await signal_exchange({**request, "company_index": index})
            with install_exchange(exchange_signal) if signal_exchange is not None else nullcontext():
                rows = score_work_item(item, icp=icp, companies=[sliced[index]], scorer=scorer,
                                       _original_company_indexes=[index])
            if len(rows) != 1:
                raise ScoringError("intrinsic company judgment is missing")
            raw = {**raw_company_judgment(rows[0]), "company_judgment_policy": company_judgments.POLICY}
            if not company_judgments.raw_judgment_is_cacheable(raw):
                raise ScoringError("company judgment is not terminal")
            raw_by_key[key] = raw
            new_judgments.append({**first_miss[key], "raw_judgment": raw})
            validated = company_judgments.validate_new_company_judgments(
                new_judgments, lease_context=lease, require_complete=False,
            )
            if on_company_judgments is not None:
                on_company_judgments(validated)
        return raw_by_key[key]
    scored, _ = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)
    full = apply_company_judgment_context(sliced, [None] * len(sliced), company_quality=False,
                                        scored_indexes=scored, resolve_raw=resolve)
    validate_breakdowns_for_item(full, icp=icp, companies=sliced,
        max_scored_companies=max_scored_companies, integrity_policy=True)
    return full, company_judgments.validate_new_company_judgments(
        new_judgments, lease_context=lease, require_complete=False,
    )


def score_quality_work_item(
    item: Mapping[str, Any], *, icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]], scorer: Scorer,
    cache_context: Mapping[str, Any], max_scored_companies: int = 0,
    on_company_judgments: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Judge only missing company inputs, then apply this list's duplicate rules."""
    from lab_arena import company_judgments
    from qualification.scoring.competition import (
        apply_company_judgment_context, bucket_skipped_judgment, raw_company_judgment,
    )

    if not getattr(scorer, "integrity_policy", False):
        raise ScoringError("company judgments require integrity scoring")
    if not getattr(scorer, "company_quality", False):
        return _score_legacy_company_work_item(
            item, icp=icp, companies=companies, scorer=scorer,
            cache_context=cache_context, max_scored_companies=max_scored_companies,
            on_company_judgments=on_company_judgments,
        )
    lease = company_judgments.validate_lease_context(cache_context)
    sliced = verify.slice_first_n(companies, verify.icp_company_goal(icp))
    refs = sorted(lease["hits"] + lease["misses"], key=lambda ref: ref["company_index"])
    if len(refs) != len(sliced):
        raise ScoringError("company cache lease does not cover this output")
    raw_by_key = {
        hit["cache_key"]: hit["evidence_doc"]["raw_judgment"]
        for hit in lease["hits"]
    }
    new_judgments = []
    first_failure = None
    for miss in sorted(lease["misses"], key=lambda ref: ref["company_index"]):
        key = miss["cache_key"]
        if key in raw_by_key:
            continue
        company = sliced[miss["company_index"]]
        try:
            rows = score_work_item(
                item, icp=icp, companies=[company], scorer=scorer,
                _original_company_indexes=[miss["company_index"]],
            )
            raw = raw_company_judgment(rows[0]) if rows else bucket_skipped_judgment()
            if not company_judgments.raw_judgment_is_cacheable(raw):
                raise ScoringError("company judgment is not terminal")
        except ScoringError as exc:
            # Independent companies can still finish within the same sandbox
            # deadline. The ICP remains failed while any company is unresolved.
            if first_failure is None:
                first_failure = exc
            continue
        raw_by_key[key] = raw
        new_judgments.append({**miss, "raw_judgment": raw})
        if on_company_judgments is not None:
            on_company_judgments(company_judgments.validate_new_company_judgments(
                new_judgments, lease_context=lease, require_complete=False,
            ))
    if first_failure is not None:
        raise first_failure
    new_judgments = company_judgments.validate_new_company_judgments(new_judgments, lease_context=lease)
    full = apply_company_judgment_context(
        sliced, [raw_by_key[ref["cache_key"]] for ref in refs],
        contacts_required=bool(getattr(scorer, "contacts_required", False)),
    )
    scored, _skipped = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)
    full = [row for row in full if row["company_index"] in scored]
    validate_breakdowns_for_item(
        full, icp=icp, companies=sliced, max_scored_companies=max_scored_companies,
        integrity_policy=True, company_quality=True,
        contacts_required=bool(getattr(scorer, "contacts_required", False)),
    )
    return full, new_judgments


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def build_stage_scores(
    *,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    icps_by_position: Mapping[int, Mapping[str, Any]],
    outputs_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    breakdowns_by_item: Mapping[str, Sequence[Mapping[str, Any]]],
    configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the score rows and mean for one execution stage."""

    validated_plan = contracts.validate_scoring_plan(plan, configuration)
    validated_policy = contracts.validate_scorer_policy(policy)
    stage = int(validated_plan["stage"])
    rows: List[Dict[str, Any]] = []
    for item in validated_plan["work_items"]:
        scored_run_id = item["scored_run_id"]
        breakdowns = breakdowns_by_item.get(scored_run_id)
        if breakdowns is None:
            raise ScoringError("run %s has no breakdowns" % scored_run_id)
        icp = icps_by_position[int(item["icp_position"])]
        companies = outputs_by_run[scored_run_id]
        rows.append(verify.scored_row(item["submission_id"], item["icp_position"], scored_run_id, icp, companies, breakdowns, validated_policy))
    for zero in validated_plan["zero_rows"]:
        rows.append(verify.zero_row(zero["submission_id"], zero["icp_position"], zero["cause"]))
    rows.sort(key=lambda row: (row["submission_id"], row["icp_position"]))
    by_submission: Dict[str, List[float]] = {}
    positions_by_submission: Dict[str, set] = {}
    for row in rows:
        by_submission.setdefault(row["submission_id"], []).append(float(row["per_icp_score"]))
        positions_by_submission.setdefault(row["submission_id"], set()).add(int(row["icp_position"]))
    expected_positions = set(
        contracts.execution_positions(
            stage, validated_plan.get("execution_sequence_policy"), configuration
        )
    )
    incomplete_positions: Dict[str, set] = {}
    for item in validated_plan.get("incomplete_rows") or []:
        incomplete_positions.setdefault(item["submission_id"], set()).add(int(item["icp_position"]))
    for submission_id in positions_by_submission.keys() | incomplete_positions.keys():
        positions = positions_by_submission.get(submission_id, set())
        if positions | incomplete_positions.get(submission_id, set()) != expected_positions:
            raise ScoringError("submission %s does not cover every stage %d ICP" % (submission_id, stage))

    scores: Dict[str, float] = {}
    denominator = len(expected_positions)
    for submission_id, values in by_submission.items():
        if submission_id not in incomplete_positions:
            scores[submission_id] = verify.stage_score(values, denominator)
    return {
        **({"integrity_policy": "arena_integrity_v1"} if validated_policy["scoring_adapter_version"] in ("qualification_integrity_v2", "qualification_contacts_v3") else {}),
        "stage": stage,
        "rows": rows,
        "submission_scores": scores,
    }


def run_scores_for_store(stage_scores: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per-attempt score records for ``lab_arena_record_run_scores``.

    Accepted attempts receive their stage row's score; the latest failed
    attempt of a zero-row assignment records zero.
    """

    rows = {
        (row["submission_id"], int(row["icp_position"])): row
        for row in stage_scores["rows"]
    }
    latest: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        key = (str(run["submission_id"]), int(run["icp_position"]))
        if run.get("status") == "accepted":
            latest[key] = run
        elif key not in latest or (latest[key].get("status") != "accepted" and int(run.get("attempt") or 0) >= int(latest[key].get("attempt") or 0)):
            latest[key] = run
    records = []
    for key, run in sorted(latest.items()):
        row = rows.get(key)
        if row is None:
            continue
        record = {"run_id": run["run_id"], "per_icp_score": float(row["per_icp_score"])}
        breakdowns = row.get("breakdowns") or []
        if any("company_qualified" in item for item in breakdowns):
            record["qualification_doc"] = {"companies": [
                {
                    **{key: item[key] for key in ("company_index", "company_identity_key", "company_qualified", "duplicate_company")},
                    **({"contact_qualified": item["contact_qualified"]} if "contact_qualified" in item else {}),
                }
                for item in breakdowns
            ]}
        elif stage_scores.get("integrity_policy") == "arena_integrity_v1":
            record["qualification_doc"] = {"companies": []}
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Scoring assignments: what a validator's judge sandbox reads and writes
# ---------------------------------------------------------------------------

SCORING_INPUT_SCHEMA_VERSION = "leadpoet.lab_arena.scoring_input.v1"
SCORING_OUTPUT_SCHEMA_VERSION = "leadpoet.lab_arena.scoring_output.v1"
SCORING_FAILURES = ("judge_error", "judge_timeout")
MAX_SCORING_OUTPUT_BYTES = 2 * 1024 * 1024
BREAKDOWN_SCORE_FIELD = "final_score"


def _require_run_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ScoringError("scored_run_id is invalid")
    return value


def build_scoring_input(
    *, scored_run_id: str, icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]], policy: Mapping[str, Any],
    evaluation_date: str,
    contact_source_evidence: Optional[Mapping[str, Any]] = None,
    provider_observations: Optional[Sequence[Mapping[str, Any]]] = None,
    company_judgment_cache: Optional[Mapping[str, Any]] = None,
    prior_terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """The judge sandbox's input: one ICP, one output, and the scorer policy."""

    from lab_arena import provider_observations as observation_policy

    document = {
        "schema_version": SCORING_INPUT_SCHEMA_VERSION,
        "scored_run_id": _require_run_id(scored_run_id),
        "icp": observation_policy.scoring_icp(
            icp, policy, provider_observations
        ),
        "companies": [dict(company) for company in companies],
        "scorer_policy": contracts.validate_scorer_policy(policy),
        "evaluation_date": str(evaluation_date),
        **({"contact_source_evidence": dict(contact_source_evidence or {})}
           if policy.get("scoring_adapter_version") == "qualification_contacts_v3" else {}),
    }
    if company_judgment_cache is not None:
        from lab_arena import company_judgments, quality_policy
        if not company_judgments.enabled(policy):
            raise ScoringError("company judgments require frozen cache policy")
        document["company_judgment_cache"] = company_judgments.validate_lease_context(company_judgment_cache)
    if prior_terminal_breakdowns is not None:
        document["prior_terminal_breakdowns"] = validate_terminal_breakdowns_for_work_item(
            prior_terminal_breakdowns, icp=icp, companies=companies, policy=policy,
        )
    return document


def build_scoring_output(scored_run_id: str, breakdowns: Sequence[Mapping[str, Any]], *, company_judgments: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    result = {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": _require_run_id(scored_run_id), "breakdowns": [dict(item) for item in breakdowns]}
    if company_judgments is not None:
        result["company_judgments"] = [dict(item) for item in company_judgments]
    return result


MAX_FAILURE_DETAIL_CHARS = 300


def build_scoring_failure(
    scored_run_id: str,
    failure: str,
    detail: str = "",
    *,
    reason: str = "",
    company_judgments: Optional[Sequence[Mapping[str, Any]]] = None,
    terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """A failure document; ``detail`` is a bounded operator-facing reason, never a payload."""

    if failure not in SCORING_FAILURES:
        raise ScoringError("unknown scoring failure")
    document = {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": _require_run_id(scored_run_id), "failure": failure}
    text = str(detail or "").strip()
    if text:
        document["detail"] = text[:MAX_FAILURE_DETAIL_CHARS]
    safe_reason = _validated_failure_reason(reason)
    if safe_reason:
        document["reason"] = safe_reason
    if company_judgments is not None:
        document["company_judgments"] = [dict(item) for item in company_judgments]
    if terminal_breakdowns is not None:
        document["terminal_breakdowns"] = [dict(item) for item in terminal_breakdowns]
    return document


def scoring_output_from_bytes(raw: bytes) -> Dict[str, Any]:
    """Parse a judge sandbox output: bounded, plain JSON, the declared shape only."""

    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_SCORING_OUTPUT_BYTES:
        raise ScoringError("scoring output is missing or too large")
    try:
        document = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ScoringError("scoring output is not JSON") from None
    return validate_scoring_output_document(document)


def validate_scoring_output_document(document: Any) -> Dict[str, Any]:
    if not isinstance(document, Mapping) or document.get("schema_version") != SCORING_OUTPUT_SCHEMA_VERSION:
        raise ScoringError("scoring output schema is invalid")
    try:
        contracts.check_strict_document(document, contracts.PUBLICATION_LIMITS)
    except contracts.ArenaContractError:
        raise ScoringError("scoring output exceeds structural limits") from None
    scored_run_id = _require_run_id(document.get("scored_run_id"))
    keys = set(document)
    if "failure" in document:
        if keys - {"detail", "reason", "company_judgments", "terminal_breakdowns"} != {"schema_version", "scored_run_id", "failure"} or document["failure"] not in SCORING_FAILURES:
            raise ScoringError("scoring failure document is invalid")
        detail = document.get("detail", "")
        if not isinstance(detail, str) or len(detail) > MAX_FAILURE_DETAIL_CHARS:
            raise ScoringError("scoring failure detail is invalid")
        reason = _validated_failure_reason(document.get("reason"))
        failure_document = {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": scored_run_id, "failure": document["failure"]}
        if detail:
            failure_document["detail"] = detail
        if reason:
            failure_document["reason"] = reason
        if "company_judgments" in document:
            failure_document["company_judgments"] = _validate_output_company_judgments(document["company_judgments"])
        if "terminal_breakdowns" in document:
            if "company_judgments" in document:
                raise ScoringError("scoring failure mixes checkpoint policies")
            failure_document["terminal_breakdowns"] = _validate_terminal_breakdown_shape(document["terminal_breakdowns"])
        return failure_document
    if keys - {"company_judgments"} != {"schema_version", "scored_run_id", "breakdowns"} or not isinstance(document["breakdowns"], list):
        raise ScoringError("scoring output document is invalid")
    breakdowns = []
    for item in document["breakdowns"]:
        if not isinstance(item, Mapping):
            raise ScoringError("scoring breakdown is not an object")
        score = item.get(BREAKDOWN_SCORE_FIELD)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 100.0:
            raise ScoringError("scoring breakdown carries no valid final score")
        breakdowns.append(dict(item))
    result = {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": scored_run_id, "breakdowns": breakdowns}
    if "company_judgments" in document:
        result["company_judgments"] = _validate_output_company_judgments(document["company_judgments"])
    return result


def _validate_output_company_judgments(judgments: Any) -> List[Dict[str, Any]]:
    from lab_arena.company_judgments import raw_judgment_is_cacheable
    if not isinstance(judgments, list) or len(judgments) > 5:
        raise ScoringError("company judgments must be a bounded list")
    for row in judgments:
        if not isinstance(row, Mapping) or set(row) != {"company_index", "cache_key", "company_input_hash", "authority_slot", "raw_judgment"} or not raw_judgment_is_cacheable(row["raw_judgment"]):
            raise ScoringError("company judgment is invalid")
    return [dict(row) for row in judgments]


def _validate_terminal_breakdown_shape(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) > verify.MAX_COMPANIES_PER_ICP:
        raise ScoringError("terminal breakdowns must be a bounded list")
    indexes = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ScoringError("terminal breakdown is not an object")
        score = row.get(BREAKDOWN_SCORE_FIELD)
        index = row.get("company_index")
        if (type(index) is not int or not 0 <= index < verify.MAX_COMPANIES_PER_ICP
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not 0 <= float(score) <= 100):
            raise ScoringError("terminal breakdown score or index is invalid")
        indexes.append(index)
    if indexes != sorted(set(indexes)):
        raise ScoringError("terminal breakdown indexes must be unique and ordered")
    return [dict(row) for row in rows]


def validate_terminal_breakdowns_for_work_item(
    rows: Sequence[Mapping[str, Any]], *, icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]], policy: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Validate terminal sparse rows against their full immutable group."""
    from lab_arena import contact_policy, quality_policy
    from qualification.scoring.arena_integrity import company_fit_verified
    from qualification.scoring.competition import (
        has_verified_primary_intent, _intent_details_gate_passed,
        scorer_breakdown_has_retryable_infrastructure_failure,
    )
    validated_policy = contracts.validate_scorer_policy(policy)
    if (not contact_policy.integrity_adapter(validated_policy["scoring_adapter_version"])
        or quality_policy.scorer_enabled(validated_policy)
        or validated_policy.get("company_judgment_policy")):
        raise ScoringError("terminal breakdowns require legacy integrity scoring")
    try:
        contracts.check_strict_document(rows, contracts.PUBLICATION_LIMITS)
    except contracts.ArenaContractError:
        raise ScoringError("terminal breakdowns exceed structural limits") from None
    copied = _validate_terminal_breakdown_shape(rows)
    sliced = verify.slice_first_n(companies, verify.icp_company_goal(icp))
    scored, _ = verify.bucket_skip(icp, sliced, max_scored_companies=int(validated_policy["max_scored_companies"]))
    indexes = [row["company_index"] for row in copied]
    if not set(indexes).issubset(scored):
        raise ScoringError("terminal breakdown index is outside the scored group")
    _validate_integrity_breakdown_rows(copied, indexes, contacts_required=contact_policy.scorer_enabled(validated_policy))
    aliases_seen = set()
    for row in copied:
        if (scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True)
            or not isinstance(row.get("verifier_gate_receipts"), list)
            or not any(isinstance(receipt, Mapping)
                       and receipt.get("gate") == "company_fit"
                       and receipt.get("decision") in {"match", "mismatch", "unavailable"}
                       for receipt in row["verifier_gate_receipts"])):
            raise ScoringError("terminal breakdown lacks a terminal verifier decision")
        aliases = row.get("company_identity_alias_keys") or [row["company_identity_key"]]
        fit = company_fit_verified(row)
        if float(row[BREAKDOWN_SCORE_FIELD]) > 0 and not (
            row["company_qualified"] and fit
            and has_verified_primary_intent(row.get("intent_signals_detail") or [])
            and _intent_details_gate_passed(row)
        ):
            raise ScoringError("terminal positive breakdown lacks verified qualification")
        if row["duplicate_company"] and not aliases_seen.intersection(aliases):
            raise ScoringError("terminal duplicate lacks its completed original-group ancestor")
        if fit and _intent_details_gate_passed(row) and not row["duplicate_company"]:
            if aliases_seen.intersection(aliases):
                raise ScoringError("terminal breakdown duplicate context is inconsistent")
            aliases_seen.update(aliases)
    return copied


def _validate_integrity_breakdown_rows(breakdowns, indexes, *, contacts_required=False):
    seen = set()
    for index, row in zip(indexes, breakdowns):
        key = row.get("company_identity_key")
        qualified = row.get("company_qualified")
        duplicate = row.get("duplicate_company")
        aliases = row.get("company_identity_alias_keys") or [key]
        if (type(row.get("company_index")) is not int or row["company_index"] != index
            or not isinstance(key, str) or not key or len(key) > 1024
            or type(qualified) is not bool or type(duplicate) is not bool):
            raise ScoringError("invalid integrity qualification receipt")
        if not isinstance(aliases, list) or not 1 <= len(aliases) <= 4 or any(not isinstance(alias, str) or not alias or len(alias) > 1024 for alias in aliases):
            raise ScoringError("invalid integrity identity aliases")
        if duplicate and (qualified or float(row[BREAKDOWN_SCORE_FIELD]) != 0):
            raise ScoringError("duplicate company cannot receive credit")
        if qualified and (seen.intersection(aliases) or float(row[BREAKDOWN_SCORE_FIELD]) <= 0):
            raise ScoringError("qualified companies must be positive and unique")
        if contacts_required:
            from lab_arena.contact_policy import validate_contact_breakdown
            try:
                validate_contact_breakdown(row)
            except contracts.ArenaContractError as exc:
                raise ScoringError(str(exc)) from exc
        if qualified:
            seen.update(aliases)


def validate_breakdowns_for_item(breakdowns: Sequence[Mapping[str, Any]], *, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], max_scored_companies: int = 0, integrity_policy: bool = False, contacts_required: bool = False, company_quality: bool = False) -> List[Dict[str, Any]]:
    """A breakdown list is acceptable for a work item only when it covers exactly the scored companies."""

    scored, _skipped = verify.bucket_skip(icp, companies)
    if max_scored_companies > 0:
        scored = scored[:max_scored_companies]
    if len(breakdowns) != len(scored):
        raise ScoringError("breakdown count %d differs from the %d scored companies" % (len(breakdowns), len(scored)))
    if integrity_policy:
        _validate_integrity_breakdown_rows(breakdowns, scored, contacts_required=contacts_required)
        if company_quality:
            from qualification.scoring.competition import apply_company_judgment_context, raw_company_judgment, bucket_skipped_judgment
            # Recompute identity and qualification from evidence, not the
            # validator's list-level flags. Positive malformed claims fail.
            by_index = {row["company_index"]: raw_company_judgment(row) for row in breakdowns}
            intrinsic = [by_index.get(index, bucket_skipped_judgment()) for index in range(len(companies))]
            expected = apply_company_judgment_context(companies, intrinsic, contacts_required=contacts_required)
            fields = ("company_index", "company_identity_key", "company_identity_alias_keys", "company_qualified", "duplicate_company", BREAKDOWN_SCORE_FIELD)
            if any(any(row.get(field) != derived.get(field) for field in fields) for row, derived in zip(breakdowns, expected)):
                raise ScoringError("company quality qualification differs from evidence")
    return [dict(item) for item in breakdowns]
