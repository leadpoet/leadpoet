"""Deterministic Lab Arena scoring and ranking rules."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

def normalize_employee_count_bucket(*args, **kwargs):
    from qualification.employee_buckets import normalize_employee_count_bucket as normalize

    return normalize(*args, **kwargs)


def normalize_observed_employee_count_bucket(*args, **kwargs):
    from qualification.employee_buckets import normalize_observed_employee_count_bucket as normalize

    return normalize(*args, **kwargs)


def _evaluator():
    """The shared competition scorer, imported on first use.

    Its import pulls the whole qualification scoring package (seconds per
    process), so the service must not pay for it until a score is actually
    derived.
    """

    from qualification.scoring import competition

    return competition


def count_penalizable_false_positives(*args, **kwargs):
    return _evaluator().count_penalizable_false_positives(*args, **kwargs)


def employee_count_buckets_for_icp(*args, **kwargs):
    return _evaluator().employee_count_buckets_for_icp(*args, **kwargs)

from lab_arena.contracts import (
    ArenaContractError,
    BENCHMARK_ICP_COUNT,
    FINALIST_COUNT,
    KING_OUTCOMES,
    STAGE_1_ICP_COUNT,
    TERMINAL_CAUSES,
    document_hash,
    validate_scorer_policy,
)

FINAL_DENOMINATOR = BENCHMARK_ICP_COUNT
STAGE_DENOMINATORS = (STAGE_1_ICP_COUNT, BENCHMARK_ICP_COUNT - STAGE_1_ICP_COUNT, FINAL_DENOMINATOR)
MAX_COMPANIES_PER_ICP = 5
ACCEPTED_CAUSE = "accepted"
ZERO_ROW_CAUSES = tuple(cause for cause in TERMINAL_CAUSES if cause != ACCEPTED_CAUSE)

# ---------------------------------------------------------------------------
# Small guards
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaContractError("%s must be an object" % name)
    return value


def _require_list(value: Any, name: str) -> List[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArenaContractError("%s must be a list" % name)
    return list(value)


def _require_score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArenaContractError("%s must be a number" % name)
    number = float(value)
    if not math.isfinite(number):
        raise ArenaContractError("%s must be finite" % name)
    return number


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArenaContractError("%s must be a non-empty string" % name)
    return value


# ---------------------------------------------------------------------------
# First-N slice and employee-bucket skip (section 12.2)
# ---------------------------------------------------------------------------


def icp_company_goal(icp: Mapping[str, Any]) -> int:
    """The ICP's ``max_companies`` clamped to 1..50, mirroring the evaluator.

    The shared daily ICP source does not need to carry this execution setting,
    so use the evaluator's fixed five-company default when it is absent.
    """

    _require_mapping(icp, "icp")
    raw = icp.get("max_companies", 5)
    if isinstance(raw, bool):
        raise ArenaContractError("ICP max_companies must be an integer")
    try:
        goal = int(raw)
    except (TypeError, ValueError) as exc:
        raise ArenaContractError("ICP max_companies must be an integer") from exc
    return max(1, min(goal, MAX_COMPANIES_PER_ICP))


def icp_has_intent_signals(icp: Mapping[str, Any]) -> bool:
    """The evaluator's ``icp_has_intents`` derivation for a mapping ICP."""

    _require_mapping(icp, "icp")
    return bool(icp.get("intent_signals") or icp.get("intent_signal"))


def slice_first_n(companies: Sequence[Any], n: int) -> List[Any]:
    """The first ``n`` companies in the model's own output order."""

    items = _require_list(companies, "companies")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ArenaContractError("n must be a positive integer")
    return items[:n]


def company_employee_bucket(company: Any) -> str:
    """Exact bucket the scorer derives for one company (empty when unknown)."""

    if not isinstance(company, Mapping):
        raise ArenaContractError("company output must be an object")
    raw = company.get("employee_count")
    return normalize_employee_count_bucket(raw, default="") or normalize_observed_employee_count_bucket(
        raw, default=""
    )


def bucket_skip(
    icp: Mapping[str, Any],
    companies: Sequence[Any],
    *,
    max_scored_companies: int = 0,
) -> Tuple[List[int], List[int]]:
    """Reproduce the scorer's skip rule: ``(scored_indexes, skipped_indexes)``.

    Mirrors ``QualificationStyleCompanyScorer._score_with_breakdowns_impl``: a
    company whose normalized bucket is empty or outside
    ``employee_count_buckets_for_icp(icp)`` is skipped without consuming a
    slot; scoring stops once the scored count reaches the ICP's company goal
    (tightened only by a non-zero ``max_scored_companies`` from the signed
    scorer policy, never from the environment). Companies after the stop are
    neither scored nor skipped.
    """

    if isinstance(max_scored_companies, bool) or not isinstance(max_scored_companies, int) or max_scored_companies < 0:
        raise ArenaContractError("max_scored_companies must be a non-negative integer")
    allowed = list(employee_count_buckets_for_icp(icp))
    goal = icp_company_goal(icp)
    cap = goal if not max_scored_companies else min(goal, max_scored_companies)
    scored = []  # type: List[int]
    skipped = []  # type: List[int]
    for index, company in enumerate(_require_list(companies, "companies")):
        if len(scored) >= cap:
            break
        bucket = company_employee_bucket(company)
        if not bucket or bucket not in allowed:
            skipped.append(index)
            continue
        scored.append(index)
    return scored, skipped


# ---------------------------------------------------------------------------
# Per-ICP score calculation (section 12.2)
# ---------------------------------------------------------------------------


def per_icp_score(
    icp: Mapping[str, Any],
    breakdowns: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """One ICP through ``compute_evaluation_aggregates`` with the scorer policy.

    ``sum(final scores clamped to 0..100) / N - penalty / N``, floored at the
    policy floor, where the penalty is the Lab's false-positive counters over
    the breakdowns multiplied by the policy points. The evaluator's default
    per-ICP mean is never used.
    """

    validated_policy = validate_scorer_policy(policy)
    goal = icp_company_goal(icp)
    rows = _require_list(breakdowns, "breakdowns")
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise ArenaContractError("breakdowns[%d] must be an object" % index)
    result = _evaluator().competition_score_from_breakdowns(
        icp,
        rows,
        fp_penalty_points=validated_policy["fp_penalty_points"],
        fp_unverified_primary_penalty_points=validated_policy[
            "fp_unverified_primary_penalty_points"
        ],
        score_floor=validated_policy["fp_penalty_icp_floor"],
    )
    return dict(result)


def zero_row(submission_id: str, icp_position: int, cause: str) -> Dict[str, Any]:
    """A synthesized zero row for a model timeout, error, or invalid output."""

    if cause not in ZERO_ROW_CAUSES:
        raise ArenaContractError("zero row cause must be one of %s" % ", ".join(ZERO_ROW_CAUSES))
    return {
        "submission_id": _require_text(submission_id, "submission_id"),
        "icp_position": _require_position(icp_position),
        "cause": cause,
        "scored_company_indexes": [],
        "skipped_company_indexes": [],
        "breakdowns": [],
        "fp_gate_count": 0,
        "fp_unverified_primary_count": 0,
        "per_icp_score": 0.0,
    }


def scored_row(
    submission_id: str,
    icp_position: int,
    scored_run_id: str,
    icp: Mapping[str, Any],
    companies: Sequence[Any],
    breakdowns: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """A score row for one accepted output: slice, skip, redact, and score."""

    validated_policy = validate_scorer_policy(policy)
    sliced = slice_first_n(companies, icp_company_goal(icp))
    scored, skipped = bucket_skip(icp, sliced, max_scored_companies=validated_policy["max_scored_companies"])
    rows = _require_list(breakdowns, "breakdowns")
    if len(rows) != len(scored):
        raise ArenaContractError("expected %d breakdowns for %d scored companies" % (len(scored), len(rows)))
    redacted = [redact_breakdown(item) for item in rows]
    result = per_icp_score(icp, redacted, validated_policy)
    return {
        "submission_id": _require_text(submission_id, "submission_id"),
        "icp_position": _require_position(icp_position),
        "scored_run_id": _require_text(scored_run_id, "scored_run_id"),
        "cause": ACCEPTED_CAUSE,
        "scored_company_indexes": scored,
        "skipped_company_indexes": skipped,
        "breakdowns": redacted,
        "fp_gate_count": result["fp_gate_count"],
        "fp_unverified_primary_count": result["fp_unverified_primary_count"],
        "per_icp_score": result["per_icp_score"],
    }


def _require_position(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < BENCHMARK_ICP_COUNT:
        raise ArenaContractError("icp_position must be within 0..%d" % (BENCHMARK_ICP_COUNT - 1))
    return value


def stage_score(per_icp_scores: Sequence[float], denominator: int) -> float:
    """Sum of exactly ``denominator`` per-ICP scores divided by it.

    Summation is exact (rational arithmetic over each score's shortest
    decimal form) so the result is independent of row order and an exact
    challenger-versus-king tie is a true equality.
    """

    if denominator not in STAGE_DENOMINATORS:
        raise ArenaContractError("stage denominator must be one of %s" % (STAGE_DENOMINATORS,))
    scores = _require_list(per_icp_scores, "per_icp_scores")
    if len(scores) != denominator:
        raise ArenaContractError("expected exactly %d per-ICP scores, got %d" % (denominator, len(scores)))
    total = Fraction(0)
    for index, value in enumerate(scores):
        total += Fraction(repr(_require_score(value, "per_icp_scores[%d]" % index)))
    return float(total / denominator)


# ---------------------------------------------------------------------------
# Ranking and king decision (section 12.3)
# ---------------------------------------------------------------------------


def stage1_ranking(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Rank challengers by Stage 1 mean, then by stable submission ID."""

    seen = set()
    challengers: List[Dict[str, Any]] = []
    for entry in entries:
        _require_mapping(entry, "ranking entry")
        submission_id = _require_text(entry.get("submission_id"), "submission_id")
        if submission_id in seen:
            raise ArenaContractError("duplicate submission %s in ranking entries" % submission_id)
        seen.add(submission_id)
        if bool(entry.get("is_king", False)):
            continue
        challengers.append({
            "submission_id": submission_id,
            "stage1_score": _require_score(entry.get("stage1_score"), "stage1_score"),
        })
    challengers.sort(key=lambda row: (-row["stage1_score"], row["submission_id"]))
    return [dict(row, rank=index + 1) for index, row in enumerate(challengers)]


def select_finalists(ranking: Sequence[Mapping[str, Any]]) -> List[str]:
    """Select the first ten challengers, or all when fewer exist."""

    return [
        _require_text(_require_mapping(row, "ranking row").get("submission_id"), "submission_id")
        for row in list(ranking)[:FINALIST_COUNT]
    ]


def _final_entry(entry: Any) -> Dict[str, Any]:
    _require_mapping(entry, "final entry")
    score = entry.get("final_score")
    return {
        "submission_id": _require_text(entry.get("submission_id"), "submission_id"),
        "hotkey": _require_text(entry.get("hotkey"), "hotkey"),
        "final_score": None if score is None else _require_score(score, "final_score"),
        "is_king": bool(entry.get("is_king", False)),
    }


def final_ranking(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Rank by score, then keep the baseline, then use the stable submission ID.

    ``final_score`` is ``None`` for a participant without a valid result; such
    rows sort last. Bundle bytes and digests never affect rank.
    """

    rows = [_final_entry(entry) for entry in entries]
    if len({row["submission_id"] for row in rows}) != len(rows):
        raise ArenaContractError("duplicate submission in final entries")
    rows.sort(
        key=lambda row: (
            0 if row["final_score"] is not None else 1,
            -(row["final_score"] or 0.0),
            0 if row["is_king"] else 1,
            row["submission_id"],
        )
    )
    ranked = []  # type: List[Dict[str, Any]]
    for index, row in enumerate(rows):
        ranked.append(
            {
                "rank": index + 1,
                "submission_id": row["submission_id"],
                "final_score": row["final_score"],
                "is_baseline": row["is_king"],
            }
        )
    return ranked


def _decision(outcome: str, king: Optional[Mapping[str, Any]], winner: Optional[str]) -> Dict[str, Any]:
    if outcome not in KING_OUTCOMES:
        raise ArenaContractError("king outcome must be one of %s" % ", ".join(KING_OUTCOMES))
    return {
        "outcome": outcome,
        "king_submission_id": None if king is None else king["submission_id"],
        "king_hotkey": "" if king is None else king["hotkey"],
        "winner_submission_id": winner,
    }


def king_decision(
    finalists_final_scores: Sequence[Mapping[str, Any]],
    king_entry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Choose a miner only when it strictly beats the daily baseline.

    Entries are ``{"submission_id", "hotkey", "final_score"}``
    where ``final_score`` is ``None`` for a participant with no valid
    full daily ICP result. Contenders are challengers with any valid score.
    The highest contender (ties by stable submission ID) is crowned only when
    both it and the organizer baseline have valid scores and the contender's
    score is strictly higher. A tie, no contender, or no valid baseline score
    records ``no_king``. The baseline is a threshold, never a champion.
    """

    contenders = []  # type: List[Dict[str, Any]]
    seen = set()  # type: set
    for entry in finalists_final_scores:
        row = _final_entry(entry)
        if row["is_king"]:
            raise ArenaContractError("the king must be passed as king_entry, not as a finalist")
        if row["submission_id"] in seen:
            raise ArenaContractError("duplicate finalist %s" % row["submission_id"])
        seen.add(row["submission_id"])
        if row["final_score"] is not None:
            contenders.append(row)
    contenders.sort(key=lambda row: (-row["final_score"], row["submission_id"]))
    best = contenders[0] if contenders else None

    if king_entry is None:
        return _decision("no_king", None, None)

    king = _final_entry(king_entry)
    if king["submission_id"] in seen:
        raise ArenaContractError("the king cannot also be a finalist")
    if (
        king["final_score"] is not None
        and best is not None
        and best["final_score"] > king["final_score"]
    ):
        return _decision("crowned", best, best["submission_id"])
    return _decision("no_king", None, None)


def result_is_valid(rows_by_position: Mapping[int, Mapping[str, Any]], positions: Sequence[int]) -> bool:
    """A valid result covers every position and has at least one accepted row.

    A missed stage or a model that never produced an accepted output therefore
    has no valid result (12.3, 13.3);
    ordinary model-caused zero rows on some ICPs do not invalidate a result.
    """

    if any(position not in rows_by_position for position in positions):
        return False
    return any(rows_by_position[position]["cause"] == ACCEPTED_CAUSE for position in positions)


# ---------------------------------------------------------------------------
# Breakdown redaction (section 12.4)
# ---------------------------------------------------------------------------

# Allow-lists. Anything not named here is dropped, so provider payloads,
# verification traces, quotes, page text, prompts, and stage verdict payloads
# can never reach a public bundle through a new key. Every field read by
# ``count_penalizable_false_positives`` and the ``scorer_breakdown_has_*``
# helpers in ``qualification/scoring/competition.py`` is kept.
BREAKDOWN_FIELDS = (
    "icp_fit",
    "decision_maker",
    "intent_signal_raw",
    "time_decay_multiplier",
    "intent_signal_final",
    "cost_penalty",
    "time_penalty",
    "final_score",
    "failure_reason",
)
SIGNAL_DETAIL_FIELDS = (
    "raw",
    "after_decay",
    "decay",
    "confidence",
    "date_status",
    "matched_icp_signal",
    "evidence_type",
)
JUDGE_VERDICT_FIELDS = ("decision", "error_class", "pipeline_decision")
GATE_RECEIPT_FIELDS = (
    "gate",
    "contract_id",
    "contract_version",
    "decision",
    "failure_class",
    "company_fit_decision",
    "company_fit_stage_required",
    "required_attribute_decision",
)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _keep_scalars(source: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    out = {}  # type: Dict[str, Any]
    for name in names:
        if name in source and _is_scalar(source[name]):
            out[name] = source[name]
    return out


def redact_signal_detail(detail: Any) -> Dict[str, Any]:
    if not isinstance(detail, Mapping):
        return {}
    out = _keep_scalars(detail, SIGNAL_DETAIL_FIELDS)
    verdict = detail.get("judge_verdict")
    if isinstance(verdict, Mapping):
        redacted_verdict = {}  # type: Dict[str, Any]
        for name in JUDGE_VERDICT_FIELDS:
            if name in verdict:
                value = verdict[name]
                redacted_verdict[name] = value if _is_scalar(value) else bool(value)
        out["judge_verdict"] = redacted_verdict
    return out


def redact_gate_receipt(receipt: Any) -> Dict[str, Any]:
    if not isinstance(receipt, Mapping):
        return {}
    out = _keep_scalars(receipt, GATE_RECEIPT_FIELDS)
    dimensions = receipt.get("company_fit_dimensions")
    if isinstance(dimensions, Mapping):
        out["company_fit_dimensions"] = {
            str(name): (value if _is_scalar(value) else None) for name, value in dimensions.items()
        }
    evidence = receipt.get("dimension_evidence")
    if isinstance(evidence, Mapping):
        redacted_evidence = {}  # type: Dict[str, Dict[str, Any]]
        for name, value in evidence.items():
            if not isinstance(value, Mapping):
                continue
            decision = value.get("decision")
            redacted_evidence[str(name)] = {"decision": decision} if "decision" in value and _is_scalar(decision) else {}
        out["dimension_evidence"] = redacted_evidence
    return out


def redact_breakdown(breakdown: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop every payload field; keep exactly what the FP derivation reads.

    Removed: verification traces, quotes, snippets, page content, prompts,
    raw provider responses, judge verdict payloads (only ``decision``,
    ``error_class`` and ``pipeline_decision`` survive), and every
    ``dimension_evidence`` value except its ``decision``. Non-object items in
    ``intent_signals_detail`` and ``verifier_gate_receipts`` become empty
    objects so list emptiness, which the unverified-primary rule reads, is
    preserved. Idempotent: redacting a redacted breakdown changes nothing.
    """

    if not isinstance(breakdown, Mapping):
        raise ArenaContractError("breakdown must be an object")
    out = _keep_scalars(breakdown, BREAKDOWN_FIELDS)
    details = breakdown.get("intent_signals_detail")
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
        out["intent_signals_detail"] = [redact_signal_detail(item) for item in details]
    receipts = breakdown.get("verifier_gate_receipts")
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)):
        out["verifier_gate_receipts"] = [redact_gate_receipt(item) for item in receipts]
    return out


def breakdown_is_redacted(breakdown: Mapping[str, Any]) -> bool:
    return document_hash(breakdown) == document_hash(redact_breakdown(breakdown))
