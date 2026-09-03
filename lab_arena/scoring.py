"""Score authority (labarena.md section 12.1): the signed scorer policy, the
scoring plan by distinct output, scoring-worker execution of the Lab scorer
on its host path, breakdown storage and copy, and score-bundle assembly.

The Lab scorer (``QualificationStyleCompanyScorer``) is imported lazily inside
the worker only after the policy has been applied to the process environment,
because it reads credentials and behavior knobs at import time.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from lab_arena import contracts, verify
from lab_arena.contracts import ArenaContractError

SCORING_ADAPTER_VERSION_V1 = "qualification_style_v1"
DEFAULT_JUDGE_MODELS = {
    "company_fit": "perplexity/sonar-pro",
    "intent_verification": "perplexity/sonar-pro",
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
    "RESEARCH_LAB_AUDIT_SECRET_SCAN_MODE": "raise",
}
CREDENTIAL_ENV_NAMES = ("OPENROUTER_API_KEY", "QUALIFICATION_OPENROUTER_API_KEY", "SCRAPINGDOG_API_KEY", "EXA_API_KEY")
CACHE_DIR_ENV = "RESEARCH_LAB_SCORING_CACHE_DIR"
MAX_JUDGE_RETRIES = 3


class ScoringError(RuntimeError):
    """Scoring infrastructure failed; the round cancels if the window closes."""


class ScorerPolicyConflict(ScoringError):
    """The process environment conflicts with the signed scorer policy."""


class JudgeKeyRefused(ScoringError):
    """The scored miner's own key or quota refused the judge's provider calls: the miner's outcome, never retried."""


# Shim error codes that mean the broker refused the call on the miner's key or quota.
KEY_REFUSAL_CODES = frozenset({"budget_refused", "budget_exhausted", "call_refused"})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def build_scorer_policy(
    *,
    judge_models: Mapping[str, str] = DEFAULT_JUDGE_MODELS,
    cache_version: str = "day_scoped_v1",
    provider_profile: str = "lab_arena",
    scoring_adapter_version: str = SCORING_ADAPTER_VERSION_V1,
) -> Dict[str, Any]:
    """The signed ``ArenaScorerPolicyV1`` (section 12.1)."""

    bindings = dict(POLICY_ENV_BINDINGS)
    return contracts.finalize_scorer_policy({
        "schema_version": contracts.SCORER_POLICY_SCHEMA_VERSION,
        "scoring_adapter_version": scoring_adapter_version,
        "fp_penalty_points": float(bindings["RESEARCH_LAB_EVAL_FP_PENALTY_POINTS"]),
        "fp_unverified_primary_penalty_points": float(bindings["RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY"]),
        "fp_penalty_icp_floor": 0.0,
        "company_cap_rule": "icp_max_companies",
        "max_scored_companies": int(bindings["RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES"]),
        "judge_models": dict(judge_models),
        "cache_version": cache_version,
        "provider_profile": provider_profile,
        "pre_slice_rule": "first_n_model_order",
        "employee_bucket_rule": "lab_relaxed_buckets",
        "env_bindings": bindings,
    })


def apply_policy_to_environment(
    policy: Mapping[str, Any],
    *,
    environ: MutableMapping[str, str],
    cache_dir: str,
    credentials: Mapping[str, str],
) -> str:
    """Bind the policy into ``environ`` before the evaluator is imported.

    Refuses to start when any bound variable already holds a conflicting
    value, when the cache directory is empty, or when a credential is
    missing. Returns the applied policy hash.
    """

    validated = contracts.validate_scorer_policy(policy)
    if "policy_hash" not in validated:
        raise ScorerPolicyConflict("scorer policy is not finalized")
    for name, value in validated["env_bindings"].items():
        existing = environ.get(name)
        if existing is not None and existing != value:
            raise ScorerPolicyConflict("environment %s conflicts with the signed scorer policy" % name)
    if not str(cache_dir or "").strip():
        raise ScorerPolicyConflict("scoring cache directory is required")
    existing_cache = environ.get(CACHE_DIR_ENV)
    if existing_cache is not None and existing_cache != cache_dir:
        raise ScorerPolicyConflict("environment %s conflicts with the scoring worker cache" % CACHE_DIR_ENV)
    for name in CREDENTIAL_ENV_NAMES:
        secret = credentials.get(name)
        if not secret:
            raise ScorerPolicyConflict("scoring credential %s is missing" % name)
        existing = environ.get(name)
        if existing is not None and existing != secret:
            raise ScorerPolicyConflict("environment %s conflicts with the Arena scoring credential" % name)
    for name, value in validated["env_bindings"].items():
        environ[name] = value
    environ[CACHE_DIR_ENV] = cache_dir
    for name in CREDENTIAL_ENV_NAMES:
        environ[name] = credentials[name]
    return str(validated["policy_hash"])


# ---------------------------------------------------------------------------
# Scoring plan (section 12.1: one work item per distinct ICP and first-N output)
# ---------------------------------------------------------------------------


def build_scoring_plan(
    *,
    round_id: str,
    stage: int,
    configuration_hash: str,
    commitment_hash: str,
    scorer_policy_hash: str,
    runs: Sequence[Mapping[str, Any]],
    icp_hashes_by_position: Mapping[int, str],
) -> Dict[str, Any]:
    """Plan from the frozen stage result set.

    ``runs`` are every attempt row of the stage. An assignment with an
    accepted attempt contributes to one work item keyed by (ICP, output);
    every other assignment contributes a zero row named by the cause of its
    latest attempt (model-caused causes and ``preflight_failed`` only; any
    other cause means the stage should have cancelled and is refused).
    """

    positions = tuple(range(0, contracts.STAGE_1_ICP_COUNT)) if stage == 1 else tuple(range(contracts.STAGE_1_ICP_COUNT, contracts.BENCHMARK_ICP_COUNT))
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
        icp_hash = icp_hashes_by_position[position]
        run = accepted.get(key)
        if run is not None:
            output_hash = contracts.require_sha256(run.get("output_hash"), "output_hash")
            if run.get("icp_hash") != icp_hash:
                raise ArenaContractError("run %s binds a different ICP hash" % run.get("run_id"))
            item_id = contracts.work_item_id(icp_hash, output_hash)
            item = items.setdefault(item_id, {"work_item_id": item_id, "icp_position": position, "icp_hash": icp_hash, "output_hash": output_hash, "submission_ids": []})
            item["submission_ids"].append(submission_id)
            continue
        cause = str(latest[key].get("terminal_cause") or "")
        if cause not in contracts.MODEL_CAUSED_TERMINAL_CAUSES and cause != "preflight_failed":
            raise ArenaContractError("assignment %s/%d ended for an infrastructure reason (%s); the stage must cancel" % (submission_id, position, cause or "none"))
        zero_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "stage": stage,
        "configuration_hash": configuration_hash,
        "commitment_hash": commitment_hash,
        "scorer_policy_hash": scorer_policy_hash,
        "work_items": [dict(item, submission_ids=sorted(item["submission_ids"])) for _, item in sorted(items.items())],
        "zero_rows": sorted(zero_rows, key=lambda row: (row["submission_id"], row["icp_position"])),
    }
    return contracts.finalize_scoring_plan(plan)


# ---------------------------------------------------------------------------
# Scoring workers
# ---------------------------------------------------------------------------

Scorer = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], bool], Any]


def lab_scorer(policy: Mapping[str, Any]) -> Scorer:
    """The Lab scorer on its host path, constructed after the policy is applied.

    ``is_reference_model`` is always False: the king is a competitor, not the
    Lab's reference model.
    """

    from research_lab.eval.evaluator import QualificationStyleCompanyScorer

    validated = contracts.validate_scorer_policy(policy)
    adapter = validated["scoring_adapter_version"]
    scorer = QualificationStyleCompanyScorer(
        attested_provider_profile=validated["provider_profile"],
        reference_scoring_adapter_version=_lab_adapter_version(adapter),
        candidate_scoring_adapter_version=_lab_adapter_version(adapter),
    )

    def score(companies: Sequence[Mapping[str, Any]], icp: Mapping[str, Any], is_reference_model: bool) -> Any:
        return scorer.score_with_breakdowns(list(companies), dict(icp), bool(is_reference_model))

    return score


def _lab_adapter_version(arena_version: str) -> str:
    from research_lab.eval import evaluator

    if arena_version == SCORING_ADAPTER_VERSION_V1:
        return evaluator.QUALIFICATION_SCORING_ADAPTER_VERSION_V1
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

    from research_lab.eval.evaluator import scorer_breakdown_has_retryable_infrastructure_failure

    sliced = verify.slice_first_n(companies, verify.icp_company_goal(icp))
    scored_indexes, _skipped = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)
    last_error: Optional[BaseException] = None
    for _attempt in range(max(1, int(max_retries))):
        try:
            breakdowns = _run_scorer(scorer, sliced, icp)
        except Exception as exc:  # judge/provider failure: retry the work item
            if getattr(exc, "code", None) in KEY_REFUSAL_CODES:
                raise JudgeKeyRefused("the scored miner's key refused the judge: %s" % exc.code) from exc
            last_error = exc
            continue
        if any(scorer_breakdown_has_retryable_infrastructure_failure(item_row) for item_row in breakdowns):
            last_error = ScoringError("judge reported an infrastructure failure")
            continue
        if len(breakdowns) != len(scored_indexes):
            raise ScoringError("scorer returned %d breakdowns for %d scored companies" % (len(breakdowns), len(scored_indexes)))
        return breakdowns
    raise ScoringError("work item %s could not be scored: %s" % (item.get("work_item_id"), type(last_error).__name__ if last_error else "unknown"))


@dataclass
class ScoringResults:
    breakdowns_by_item: Dict[str, List[Dict[str, Any]]]
    judge_executions: int


def run_scoring_plan(
    plan: Mapping[str, Any],
    *,
    icps_by_position: Mapping[int, Mapping[str, Any]],
    outputs_by_hash: Mapping[str, Sequence[Mapping[str, Any]]],
    scorer: Scorer,
    workers: int = 1,
    existing: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> ScoringResults:
    """Score every work item exactly once (skipping items already scored).

    Workers pull items concurrently; the store of results is keyed by work
    item so a restart resumes from what was durably recorded.
    """

    validated = contracts.validate_scoring_plan(plan)
    results: Dict[str, List[Dict[str, Any]]] = {key: [dict(row) for row in value] for key, value in (existing or {}).items()}
    pending = [item for item in validated["work_items"] if item["work_item_id"] not in results]
    executions = 0

    def _score(item: Mapping[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        icp = icps_by_position[int(item["icp_position"])]
        companies = outputs_by_hash[item["output_hash"]]
        return item["work_item_id"], score_work_item(item, icp=icp, companies=companies, scorer=scorer)

    if workers <= 1:
        for item in pending:
            key, breakdowns = _score(item)
            results[key] = breakdowns
            executions += 1
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            for key, breakdowns in pool.map(_score, pending):
                results[key] = breakdowns
                executions += 1
    return ScoringResults(breakdowns_by_item=results, judge_executions=executions)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def build_score_bundle(
    *,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    icps_by_position: Mapping[int, Mapping[str, Any]],
    outputs_by_hash: Mapping[str, Sequence[Mapping[str, Any]]],
    breakdowns_by_item: Mapping[str, Sequence[Mapping[str, Any]]],
    stage_1_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    stage_1_bundle_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy each work item's breakdown to every submission with that output,
    synthesize zero rows, and compute stage scores (Stage 1 over 20; the
    Stage 2 bundle carries the final 50-ICP score using the Stage 1 rows)."""

    validated_plan = contracts.validate_scoring_plan(plan)
    validated_policy = contracts.validate_scorer_policy(policy)
    stage = int(validated_plan["stage"])
    rows: List[Dict[str, Any]] = []
    for item in validated_plan["work_items"]:
        breakdowns = breakdowns_by_item.get(item["work_item_id"])
        if breakdowns is None:
            raise ScoringError("work item %s has no breakdowns" % item["work_item_id"])
        icp = icps_by_position[int(item["icp_position"])]
        companies = outputs_by_hash[item["output_hash"]]
        for submission_id in item["submission_ids"]:
            rows.append(verify.scored_row(submission_id, item["icp_position"], item["icp_hash"], item["output_hash"], icp, companies, breakdowns, validated_policy))
    for zero in validated_plan["zero_rows"]:
        icp_hash = contracts.document_hash(icps_by_position[int(zero["icp_position"])])
        rows.append(verify.zero_row(zero["submission_id"], zero["icp_position"], icp_hash, zero["cause"]))
    rows.sort(key=lambda row: (row["submission_id"], row["icp_position"]))
    by_submission: Dict[str, List[float]] = {}
    for row in rows:
        by_submission.setdefault(row["submission_id"], []).append(float(row["per_icp_score"]))
    scores: Dict[str, float] = {}
    if stage == 1:
        for submission_id, values in by_submission.items():
            scores[submission_id] = verify.stage_score(values, contracts.STAGE_1_ICP_COUNT)
    else:
        if stage_1_rows is None or not stage_1_bundle_hash:
            raise ScoringError("stage 2 bundle requires the stage 1 rows and bundle hash")
        stage_1_scores: Dict[str, List[float]] = {}
        for row in stage_1_rows:
            stage_1_scores.setdefault(str(row["submission_id"]), []).append(float(row["per_icp_score"]))
        for submission_id, values in by_submission.items():
            first = stage_1_scores.get(submission_id)
            if first is None or len(first) != contracts.STAGE_1_ICP_COUNT:
                raise ScoringError("submission %s lacks its stage 1 rows" % submission_id)
            scores[submission_id] = verify.stage_score(list(first) + values, contracts.BENCHMARK_ICP_COUNT)
    document = {
        "schema_version": contracts.SCORE_BUNDLE_SCHEMA_VERSION,
        "round_id": validated_plan["round_id"],
        "stage": stage,
        "scorer_policy": validated_policy,
        "scoring_plan_hash": validated_plan["plan_hash"],
        "rows": rows,
        "submission_scores": scores,
    }
    if stage == 2:
        document["stage_1_bundle_hash"] = stage_1_bundle_hash
    return verify.finalize_score_bundle(document)


def run_scores_for_store(bundle: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], *, score_ref: str) -> List[Dict[str, Any]]:
    """Per-attempt score records for ``lab_arena_record_run_scores``.

    Accepted attempts receive their bundle row's score; the latest failed
    attempt of a zero-row assignment records zero.
    """

    rows = {(row["submission_id"], int(row["icp_position"])): row for row in bundle["rows"]}
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
        records.append({"run_id": run["run_id"], "per_icp_score": float(row["per_icp_score"]), "score_ref": score_ref})
    return records


# ---------------------------------------------------------------------------
# Scoring assignments: what a validator's judge sandbox reads and writes
# ---------------------------------------------------------------------------

SCORING_INPUT_SCHEMA_VERSION = "leadpoet.lab_arena.scoring_input.v1"
SCORING_OUTPUT_SCHEMA_VERSION = "leadpoet.lab_arena.scoring_output.v1"
SCORING_FAILURES = ("judge_error", "judge_timeout", "judge_key_refused")
MAX_SCORING_OUTPUT_BYTES = 2 * 1024 * 1024
BREAKDOWN_SCORE_FIELD = "final_score"


def build_scoring_input(*, work_item_id: str, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], evaluation_date: str) -> Dict[str, Any]:
    """The judge sandbox's input: one ICP, one output, the signed scorer policy."""

    return {
        "schema_version": SCORING_INPUT_SCHEMA_VERSION,
        "work_item_id": contracts.require_sha256(work_item_id, "work_item_id"),
        "icp": dict(icp),
        "companies": [dict(company) for company in companies],
        "scorer_policy": contracts.validate_scorer_policy(policy),
        "evaluation_date": str(evaluation_date),
    }


def build_scoring_output(work_item_id: str, breakdowns: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "work_item_id": work_item_id, "breakdowns": [dict(item) for item in breakdowns]}


def build_scoring_failure(work_item_id: str, failure: str) -> Dict[str, Any]:
    if failure not in SCORING_FAILURES:
        raise ScoringError("unknown scoring failure")
    return {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "work_item_id": work_item_id, "failure": failure}


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
    work_item_id = document.get("work_item_id")
    contracts.require_sha256(work_item_id, "work_item_id")
    keys = set(document)
    if "failure" in document:
        if keys != {"schema_version", "work_item_id", "failure"} or document["failure"] not in SCORING_FAILURES:
            raise ScoringError("scoring failure document is invalid")
        return {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "work_item_id": work_item_id, "failure": document["failure"]}
    if keys != {"schema_version", "work_item_id", "breakdowns"} or not isinstance(document["breakdowns"], list):
        raise ScoringError("scoring output document is invalid")
    breakdowns = []
    for item in document["breakdowns"]:
        if not isinstance(item, Mapping):
            raise ScoringError("scoring breakdown is not an object")
        score = item.get(BREAKDOWN_SCORE_FIELD)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 100.0:
            raise ScoringError("scoring breakdown carries no valid final score")
        breakdowns.append(dict(item))
    return {"schema_version": SCORING_OUTPUT_SCHEMA_VERSION, "work_item_id": work_item_id, "breakdowns": breakdowns}


def validate_breakdowns_for_item(breakdowns: Sequence[Mapping[str, Any]], *, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], max_scored_companies: int = 0) -> List[Dict[str, Any]]:
    """A breakdown list is acceptable for a work item only when it covers exactly the scored companies."""

    scored, _skipped = verify.bucket_skip(icp, companies)
    if max_scored_companies > 0:
        scored = scored[:max_scored_companies]
    if len(breakdowns) != len(scored):
        raise ScoringError("breakdown count %d differs from the %d scored companies" % (len(breakdowns), len(scored)))
    return [dict(item) for item in breakdowns]
