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
# Every OpenRouter model the Research Lab judge calls, by the judge's own role
# for it. The broker refuses any other model on a scoring run, so this mapping
# must cover the judge image exactly; ``test_lab_arena_judge_routes`` scans
# the judge's source for model literals against it.
DEFAULT_JUDGE_MODELS = {
    "company_fit_reverification": "perplexity/sonar",
    "intent_signal_judge": "anthropic/claude-sonnet-4.5",
    "intent_verification": "openai/gpt-4o-mini",
    "intent_precheck": "google/gemini-2.5-flash-lite",
    "intent_three_stage_stage3": "perplexity/sonar-pro",
    "role_batch_check": "google/gemini-2.5-flash",
}
# Every scorer behavior the Lab reads from the environment, pinned by policy.
POLICY_ENV_BINDINGS = {
    "RESEARCH_LAB_EVAL_FP_PENALTY_POINTS": "10",
    "RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY": "10",
    "RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES": "0",
    "RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE": "0",
    "RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY": "1",
    "RESEARCH_LAB_EVAL_WORK_CONSERVING": "0",
    "RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY": "1",
    "RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY": "0",
    "RESEARCH_LAB_GLOBAL_SCORING_QUEUE": "0",
    "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX": "",
    "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID": "",
    "RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE": "0",
}
CREDENTIAL_ENV_NAMES = ("OPENROUTER_API_KEY", "QUALIFICATION_OPENROUTER_API_KEY", "SCRAPINGDOG_API_KEY", "EXA_API_KEY")
MAX_JUDGE_RETRIES = 3


class ScoringError(RuntimeError):
    """Scoring infrastructure failed; the round cancels if the window closes."""


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
) -> Dict[str, Any]:
    """Return the plain scorer settings used for every participant."""

    bindings = dict(POLICY_ENV_BINDINGS)
    return contracts.validate_scorer_policy({
        "schema_version": contracts.SCORER_POLICY_SCHEMA_VERSION,
        "scoring_adapter_version": scoring_adapter_version,
        "fp_penalty_points": float(bindings["RESEARCH_LAB_EVAL_FP_PENALTY_POINTS"]),
        "fp_unverified_primary_penalty_points": float(bindings["RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY"]),
        "fp_penalty_icp_floor": 0.0,
        "company_cap_rule": "icp_max_companies",
        "max_scored_companies": int(bindings["RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES"]),
        "judge_models": dict(judge_models),
        "provider_profile": provider_profile,
        "pre_slice_rule": "first_n_model_order",
        "employee_bucket_rule": "lab_relaxed_buckets",
        "env_bindings": bindings,
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
    for name, value in validated["env_bindings"].items():
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
    for name, value in validated["env_bindings"].items():
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
) -> Dict[str, Any]:
    """Plan from the frozen stage result set.

    ``runs`` are every attempt row of the stage. An assignment with an
    accepted attempt contributes exactly one work item named by its run and
    output reference; every other assignment contributes a zero row named
    by the cause of its latest model-caused attempt. Any infrastructure cause
    means the stage should have cancelled and is refused here.
    """

    if stage not in (1, 2):
        raise ArenaContractError("stage must be 1 or 2")
    positions = contracts.stage_positions(stage)
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
        cause = str(latest[key].get("terminal_cause") or "")
        if cause not in contracts.MODEL_CAUSED_TERMINAL_CAUSES:
            # A confirmation attempt the window closed before it ran leaves the
            # earlier model-caused failure standing.
            confirmed = [run for run in runs if (str(run.get("submission_id")), int(run.get("icp_position") or 0)) == key and str(run.get("terminal_cause") or "") in contracts.MODEL_CAUSED_TERMINAL_CAUSES]
            if confirmed:
                cause = str(max(confirmed, key=lambda run: int(run.get("attempt") or 0))["terminal_cause"])
            else:
                raise ArenaContractError("assignment %s/%d ended for an infrastructure reason (%s); the stage must cancel" % (submission_id, position, cause or "none"))
        zero_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "stage": stage,
        "work_items": [dict(item) for _, item in sorted(items.items())],
        "zero_rows": sorted(zero_rows, key=lambda row: (row["submission_id"], row["icp_position"])),
    }
    return contracts.validate_scoring_plan(plan)


# ---------------------------------------------------------------------------
# Scoring workers
# ---------------------------------------------------------------------------

Scorer = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], bool], Any]


def lab_scorer(policy: Mapping[str, Any]) -> Scorer:
    """The Lab scorer on its host path, constructed after the policy is applied.

    ``is_reference_model`` is always False: the king is a competitor, not the
    Lab's reference model.
    """

    from qualification.scoring.competition import CompetitionCompanyScorer

    validated = contracts.validate_scorer_policy(policy)
    adapter = validated["scoring_adapter_version"]
    _lab_adapter_version(adapter)
    scorer = CompetitionCompanyScorer()

    def score(companies: Sequence[Mapping[str, Any]], icp: Mapping[str, Any], is_reference_model: bool) -> Any:
        return scorer.score_with_breakdowns(list(companies), dict(icp), bool(is_reference_model))

    return score


def _lab_adapter_version(arena_version: str) -> str:
    from qualification.scoring.competition import SCORING_ADAPTER_VERSION

    if arena_version == SCORING_ADAPTER_VERSION_V1:
        return SCORING_ADAPTER_VERSION
    raise ScoringError("unsupported scoring adapter version")


def _run_scorer(scorer: Scorer, companies: Sequence[Mapping[str, Any]], icp: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result = scorer(companies, icp, False)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    if not isinstance(result, (list, tuple)):
        raise ScoringError("scorer returned a non-list result")
    return [dict(item) for item in result]


def score_work_item(
    item: Mapping[str, Any],
    *,
    icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]],
    scorer: Scorer,
    max_retries: int = MAX_JUDGE_RETRIES,
    max_scored_companies: int = 0,
) -> List[Dict[str, Any]]:
    """Score one distinct output once: the first-N slice in model order.

    A judge infrastructure failure inside a breakdown retries the whole item;
    exhaustion raises ``ScoringError`` (the service cancels if the window
    closes) and never creates a miner zero. The scorer must return exactly one
    breakdown per company it scores under the bucket-skip rule the verifier
    recomputes; any other count is a scorer contract failure.
    """

    from qualification.scoring.competition import (
        scorer_breakdown_has_retryable_infrastructure_failure,
    )

    sliced = verify.slice_first_n(companies, verify.icp_company_goal(icp))
    scored_indexes, _skipped = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)
    last_error: Optional[BaseException] = None
    for _attempt in range(max(1, int(max_retries))):
        try:
            breakdowns = _run_scorer(scorer, sliced, icp)
        except Exception as exc:  # judge/provider failure: retry the work item
            last_error = exc
            continue
        failed = [item_row for item_row in breakdowns if scorer_breakdown_has_retryable_infrastructure_failure(item_row)]
        if failed:
            last_error = ScoringError("judge reported an infrastructure failure: %s" % str(failed[0].get("failure_reason") or "")[:200])
            continue
        if len(breakdowns) != len(scored_indexes):
            raise ScoringError("scorer returned %d breakdowns for %d scored companies" % (len(breakdowns), len(scored_indexes)))
        return breakdowns
    raise ScoringError("run %s could not be scored: %s: %s" % (item.get("scored_run_id"), type(last_error).__name__ if last_error else "unknown", str(last_error or "")[:240]))


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
) -> Dict[str, Any]:
    """Compute the score rows and mean for one execution stage."""

    validated_plan = contracts.validate_scoring_plan(plan)
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
    expected_positions = set(contracts.stage_positions(stage))
    for submission_id, positions in positions_by_submission.items():
        if positions != expected_positions:
            raise ScoringError("submission %s does not cover every stage %d ICP" % (submission_id, stage))

    scores: Dict[str, float] = {}
    denominator = len(expected_positions)
    for submission_id, values in by_submission.items():
        scores[submission_id] = verify.stage_score(values, denominator)
    return {
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
        records.append({"run_id": run["run_id"], "per_icp_score": float(row["per_icp_score"])})
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


def build_scoring_input(*, scored_run_id: str, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], evaluation_date: str) -> Dict[str, Any]:
    """The judge sandbox's input: one ICP, one output, and the scorer policy."""

    return {
        "schema_version": SCORING_INPUT_SCHEMA_VERSION,
        "scored_run_id": _require_run_id(scored_run_id),
        "icp": dict(icp),
        "companies": [dict(company) for company in companies],
        "scorer_policy": contracts.validate_scorer_policy(policy),
        "evaluation_date": str(evaluation_date),
    }


def build_scoring_output(scored_run_id: str, breakdowns: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": _require_run_id(scored_run_id), "breakdowns": [dict(item) for item in breakdowns]}


MAX_FAILURE_DETAIL_CHARS = 300


def build_scoring_failure(scored_run_id: str, failure: str, detail: str = "") -> Dict[str, Any]:
    """A failure document; ``detail`` is a bounded operator-facing reason, never a payload."""

    if failure not in SCORING_FAILURES:
        raise ScoringError("unknown scoring failure")
    document = {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": _require_run_id(scored_run_id), "failure": failure}
    text = str(detail or "").strip()
    if text:
        document["detail"] = text[:MAX_FAILURE_DETAIL_CHARS]
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
        if keys - {"detail"} != {"schema_version", "scored_run_id", "failure"} or document["failure"] not in SCORING_FAILURES:
            raise ScoringError("scoring failure document is invalid")
        detail = document.get("detail", "")
        if not isinstance(detail, str) or len(detail) > MAX_FAILURE_DETAIL_CHARS:
            raise ScoringError("scoring failure detail is invalid")
        failure_document = {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": scored_run_id, "failure": document["failure"]}
        if detail:
            failure_document["detail"] = detail
        return failure_document
    if keys != {"schema_version", "scored_run_id", "breakdowns"} or not isinstance(document["breakdowns"], list):
        raise ScoringError("scoring output document is invalid")
    breakdowns = []
    for item in document["breakdowns"]:
        if not isinstance(item, Mapping):
            raise ScoringError("scoring breakdown is not an object")
        score = item.get(BREAKDOWN_SCORE_FIELD)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 100.0:
            raise ScoringError("scoring breakdown carries no valid final score")
        breakdowns.append(dict(item))
    return {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": scored_run_id, "breakdowns": breakdowns}


def validate_breakdowns_for_item(breakdowns: Sequence[Mapping[str, Any]], *, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], max_scored_companies: int = 0) -> List[Dict[str, Any]]:
    """A breakdown list is acceptable for a work item only when it covers exactly the scored companies."""

    scored, _skipped = verify.bucket_skip(icp, companies)
    if max_scored_companies > 0:
        scored = scored[:max_scored_companies]
    if len(breakdowns) != len(scored):
        raise ScoringError("breakdown count %d differs from the %d scored companies" % (len(breakdowns), len(scored)))
    return [dict(item) for item in breakdowns]
