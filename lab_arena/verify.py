"""Lab Arena deterministic public verifier (labarena.md sections 12.2-12.5).

Everything here is pure, free, and pinned in code: no environment flag, no
network, no provider call. From a published round bundle the verifier
recomputes the first-N slice and employee-bucket skip (12.2), every per-ICP
score through the Lab's published-bundle arithmetic
(``leadpoet_verifier.research_evaluation.compute_evaluation_aggregates`` over
the Lab's false-positive counters), the Stage 1 and final stage scores, the
Stage 1 ranking, the finalist set, the final ranking, and the king decision
(12.3), and it checks every hash and Arena signature (12.4).

What this module can and cannot say: it verifies published documents,
hashes, signatures, and recomputed aggregates. It claims nothing about the
original execution of any model. Its report vocabulary is ``verified`` /
``failed`` / ``not_checked``; the "generally aligned" / "materially
divergent" wording belongs exclusively to the separate rerun command (12.5),
which re-executes models with the user's own credentials.

Repository imports are limited to the modules listed in labarena.md section
3.1 and their runtime import closure contains no ``gateway.tee`` or
``gateway.db`` module.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from leadpoet_verifier.research_evaluation import compute_evaluation_aggregates
from research_lab.employee_buckets import (
    normalize_employee_count_bucket,
    normalize_observed_employee_count_bucket,
)
from research_lab.eval.evaluator import (
    count_penalizable_false_positives,
    employee_count_buckets_for_icp,
)

from lab_arena.contracts import (
    ArenaContractError,
    ArenaSignatureError,
    BENCHMARK_ICP_COUNT,
    F,
    FINALIST_COUNT,
    KING_OUTCOMES,
    MAX_CHALLENGERS,
    PUBLICATION_LIMITS,
    SCORE_BUNDLE_SCHEMA_VERSION,
    STAGE_1_ICP_COUNT,
    TERMINAL_CAUSES,
    benchmark_roots,
    check_strict_document,
    document_hash,
    hashed_document,
    participant_set_hash,
    require_sha256,
    validate_benchmark_commitment,
    validate_document,
    validate_reward_basis,
    validate_round_configuration,
    validate_scorer_policy,
    validate_scoring_plan,
    verify_hashed_document,
)
from lab_arena.signing import (
    load_public_key_from_document,
    public_key_hash,
    verify_document_signature,
)

STAGE_1_DENOMINATOR = STAGE_1_ICP_COUNT
FINAL_DENOMINATOR = BENCHMARK_ICP_COUNT
STAGE_DENOMINATORS = (STAGE_1_DENOMINATOR, FINAL_DENOMINATOR)
MAX_COMPANIES_PER_ICP = 50
ACCEPTED_CAUSE = "accepted"
ZERO_ROW_CAUSES = tuple(cause for cause in TERMINAL_CAUSES if cause != ACCEPTED_CAUSE)

VERIFIER_STATEMENT = (
    "This report verifies published documents, hashes, signatures, and "
    "recomputed aggregates only. It claims nothing about the original "
    "execution of any model."
)

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

    The evaluator falls back to a fixed five-lead budget when the key is
    absent; every accepted Arena ICP carries ``max_companies`` (section 18.6),
    so an ICP without it is a contract violation here rather than a silent
    five.
    """

    _require_mapping(icp, "icp")
    raw = icp.get("max_companies")
    if raw is None or isinstance(raw, bool):
        raise ArenaContractError("ICP must carry max_companies")
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
# Per-ICP score through the published-bundle arithmetic (section 12.2)
# ---------------------------------------------------------------------------


def per_icp_score(
    icp: Mapping[str, Any],
    breakdowns: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    icp_hash: str = "",
) -> Dict[str, Any]:
    """One ICP through ``compute_evaluation_aggregates`` with the signed policy.

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
    gate, primary = count_penalizable_false_positives(
        rows, icp_has_intent_signals=icp_has_intent_signals(icp)
    )
    company_scores = [float(item.get("final_score", 0.0) or 0.0) for item in rows]
    row = {
        "icp_ref": str(icp_hash),
        "icp_hash": str(icp_hash),
        "icp_company_goal": goal,
        "base_company_scores": [],
        "candidate_company_scores": company_scores,
        "candidate_fp_gate_count": gate,
        "candidate_fp_unverified_primary_count": primary,
    }
    aggregates = compute_evaluation_aggregates(
        [row],
        leads_per_icp_normalizer=goal,
        fp_penalty_points=validated_policy["fp_penalty_points"],
        fp_unverified_primary_penalty_points=validated_policy["fp_unverified_primary_penalty_points"],
        fp_penalty_icp_floor=validated_policy["fp_penalty_icp_floor"],
    )
    score = float(aggregates["per_icp_results"][0]["candidate_per_icp_score"])
    return {
        "per_icp_score": score,
        "fp_gate_count": int(gate),
        "fp_unverified_primary_count": int(primary),
        "company_goal": goal,
        "company_scores": company_scores,
    }


def zero_row(submission_id: str, icp_position: int, icp_hash: str, cause: str) -> Dict[str, Any]:
    """A synthesized zero row for a timeout, missing, invalid, refused, or preflight-failed ICP."""

    if cause not in ZERO_ROW_CAUSES:
        raise ArenaContractError("zero row cause must be one of %s" % ", ".join(ZERO_ROW_CAUSES))
    return {
        "submission_id": _require_text(submission_id, "submission_id"),
        "icp_position": _require_position(icp_position),
        "icp_hash": require_sha256(icp_hash, "icp_hash"),
        "output_hash": None,
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
    icp_hash: str,
    output_hash: str,
    icp: Mapping[str, Any],
    companies: Sequence[Any],
    breakdowns: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """A score-bundle row for one accepted output: slice, skip, redact, score."""

    validated_policy = validate_scorer_policy(policy)
    sliced = slice_first_n(companies, icp_company_goal(icp))
    scored, skipped = bucket_skip(icp, sliced, max_scored_companies=validated_policy["max_scored_companies"])
    rows = _require_list(breakdowns, "breakdowns")
    if len(rows) != len(scored):
        raise ArenaContractError("expected %d breakdowns for %d scored companies" % (len(scored), len(rows)))
    redacted = [redact_breakdown(item) for item in rows]
    result = per_icp_score(icp, redacted, validated_policy, icp_hash=icp_hash)
    return {
        "submission_id": _require_text(submission_id, "submission_id"),
        "icp_position": _require_position(icp_position),
        "icp_hash": require_sha256(icp_hash, "icp_hash"),
        "output_hash": require_sha256(output_hash, "output_hash"),
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
    """Sum of exactly ``denominator`` per-ICP scores divided by it (20 or 50).

    Summation is exact (rational arithmetic over each score's shortest
    decimal form) so the result is independent of row order and an exact
    challenger-versus-king tie is a true equality.
    """

    if denominator not in STAGE_DENOMINATORS:
        raise ArenaContractError("stage denominator must be %d or %d" % STAGE_DENOMINATORS)
    scores = _require_list(per_icp_scores, "per_icp_scores")
    if len(scores) != denominator:
        raise ArenaContractError("expected exactly %d per-ICP scores, got %d" % (denominator, len(scores)))
    total = Fraction(0)
    for index, value in enumerate(scores):
        total += Fraction(repr(_require_score(value, "per_icp_scores[%d]" % index)))
    return float(total / denominator)


# ---------------------------------------------------------------------------
# Ranking, finalists, and king decision (section 12.3)
# ---------------------------------------------------------------------------


def tie_break_hash(salt: str, artifact_hash: str) -> str:
    """Hash of the post-cutoff salt plus artifact hash; lower ranks first."""

    return document_hash(
        {"salt": _require_text(salt, "salt"), "artifact_hash": require_sha256(artifact_hash, "artifact_hash")}
    )


def stage1_ranking(entries: Sequence[Mapping[str, Any]], salt: str) -> List[Dict[str, Any]]:
    """Challengers by higher Stage 1 score, then lower salted artifact hash.

    Each entry is ``{"submission_id", "artifact_hash", "stage1_score",
    "is_king"}``; the king is excluded and advances separately.
    """

    seen = set()  # type: set
    challengers = []  # type: List[Dict[str, Any]]
    for entry in entries:
        _require_mapping(entry, "ranking entry")
        submission_id = _require_text(entry.get("submission_id"), "submission_id")
        if submission_id in seen:
            raise ArenaContractError("duplicate submission %s in ranking entries" % submission_id)
        seen.add(submission_id)
        if bool(entry.get("is_king", False)):
            continue
        challengers.append(
            {
                "submission_id": submission_id,
                "stage1_score": _require_score(entry.get("stage1_score"), "stage1_score"),
                "tie_break_hash": tie_break_hash(salt, entry.get("artifact_hash")),
            }
        )
    challengers.sort(key=lambda row: (-row["stage1_score"], row["tie_break_hash"]))
    ranked = []  # type: List[Dict[str, Any]]
    for index, row in enumerate(challengers):
        ranked.append(
            {
                "rank": index + 1,
                "submission_id": row["submission_id"],
                "stage1_score": row["stage1_score"],
                "tie_break_hash": row["tie_break_hash"],
            }
        )
    return ranked


def select_finalists(ranking: Sequence[Mapping[str, Any]]) -> List[str]:
    """The first ``FINALIST_COUNT`` challengers, or all when fewer exist."""

    return [str(_require_mapping(row, "ranking row")["submission_id"]) for row in list(ranking)[:FINALIST_COUNT]]


def _final_entry(entry: Any, salt: str) -> Dict[str, Any]:
    _require_mapping(entry, "final entry")
    score = entry.get("final_score")
    return {
        "submission_id": _require_text(entry.get("submission_id"), "submission_id"),
        "hotkey": _require_text(entry.get("hotkey"), "hotkey"),
        "final_score": None if score is None else _require_score(score, "final_score"),
        "is_king": bool(entry.get("is_king", False)),
        "tie_break_hash": tie_break_hash(salt, entry.get("artifact_hash")),
    }


def final_ranking(entries: Sequence[Mapping[str, Any]], salt: str) -> List[Dict[str, Any]]:
    """Finalists and king by valid final score, king first on an exact tie.

    ``final_score`` is ``None`` for a participant without a valid result; such
    rows sort last, by salted artifact hash.
    """

    rows = [_final_entry(entry, salt) for entry in entries]
    if len({row["submission_id"] for row in rows}) != len(rows):
        raise ArenaContractError("duplicate submission in final entries")
    rows.sort(
        key=lambda row: (
            0 if row["final_score"] is not None else 1,
            -(row["final_score"] or 0.0),
            0 if row["is_king"] else 1,
            row["tie_break_hash"],
        )
    )
    ranked = []  # type: List[Dict[str, Any]]
    for index, row in enumerate(rows):
        ranked.append(
            {
                "rank": index + 1,
                "submission_id": row["submission_id"],
                "final_score": row["final_score"],
                "is_king": row["is_king"],
                "tie_break_hash": row["tie_break_hash"],
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
    salt: str,
) -> Dict[str, Any]:
    """Section 12.3 decision from finalist and king final results.

    Entries are ``{"submission_id", "hotkey", "artifact_hash", "final_score"}``
    where ``final_score`` is ``None`` for a participant with no valid
    50-ICP result. Contenders are challengers with a valid score above zero.
    The highest contender (ties by lower salted artifact hash) is crowned
    when it strictly exceeds the king's valid score, or whenever the king has
    no valid result; an exact tie keeps the king (``defended``). With no
    contender the king is ``defended`` when it has a valid result and
    ``retained_ineligible`` otherwise; with no king at all the round records
    ``no_king``.
    """

    contenders = []  # type: List[Dict[str, Any]]
    seen = set()  # type: set
    for entry in finalists_final_scores:
        row = _final_entry(entry, salt)
        if row["is_king"]:
            raise ArenaContractError("the king must be passed as king_entry, not as a finalist")
        if row["submission_id"] in seen:
            raise ArenaContractError("duplicate finalist %s" % row["submission_id"])
        seen.add(row["submission_id"])
        if row["final_score"] is not None and row["final_score"] > 0.0:
            contenders.append(row)
    contenders.sort(key=lambda row: (-row["final_score"], row["tie_break_hash"]))
    best = contenders[0] if contenders else None

    if king_entry is None:
        if best is None:
            return _decision("no_king", None, None)
        return _decision("crowned", best, best["submission_id"])

    king = _final_entry(king_entry, salt)
    if king["submission_id"] in seen:
        raise ArenaContractError("the king cannot also be a finalist")
    if king["final_score"] is not None:
        if best is not None and best["final_score"] > king["final_score"]:
            return _decision("crowned", best, best["submission_id"])
        return _decision("defended", king, None)
    if best is not None:
        return _decision("crowned", best, best["submission_id"])
    return _decision("retained_ineligible", king, None)


def result_is_valid(rows_by_position: Mapping[int, Mapping[str, Any]], positions: Sequence[int]) -> bool:
    """A valid result covers every position and has at least one accepted row.

    A ``preflight_failed`` king, a missed stage, or a model that never
    produced an accepted output therefore has no valid result (12.3, 13.3);
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
# helpers in ``research_lab/eval/evaluator.py`` is kept.
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


# ---------------------------------------------------------------------------
# Score bundle schema (section 12.4)
# ---------------------------------------------------------------------------

SCORE_BUNDLE_ROW_FIELDS = (
    F("submission_id", "str", minimum=1, maximum=64),
    F("icp_position", "int", minimum=0, maximum=BENCHMARK_ICP_COUNT - 1),
    F("icp_hash", "sha256"),
    F("output_hash", "sha256", required=False),
    F("cause", "str", choices=TERMINAL_CAUSES),
    F("scored_company_indexes", "list[int]", minimum=0, maximum=MAX_COMPANIES_PER_ICP),
    F("skipped_company_indexes", "list[int]", minimum=0, maximum=MAX_COMPANIES_PER_ICP),
    F("breakdowns", "list[object]", minimum=0, maximum=MAX_COMPANIES_PER_ICP),
    F("fp_gate_count", "int", minimum=0, maximum=MAX_COMPANIES_PER_ICP),
    F("fp_unverified_primary_count", "int", minimum=0, maximum=MAX_COMPANIES_PER_ICP),
    F("per_icp_score", "float", minimum=-100, maximum=100),
)

SCORE_BUNDLE_FIELDS = (
    F("schema_version", "str", choices=(SCORE_BUNDLE_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("stage", "int", minimum=1, maximum=2),
    F("scorer_policy", "object"),
    F("scoring_plan_hash", "sha256"),
    F("stage_1_bundle_hash", "sha256", required=False),
    F(
        "rows",
        "list[object]",
        fields=SCORE_BUNDLE_ROW_FIELDS,
        minimum=0,
        maximum=BENCHMARK_ICP_COUNT * (MAX_CHALLENGERS + 1),
    ),
    F("submission_scores", "object"),
    F("bundle_hash", "sha256", required=False),
    F("signature", "object", required=False),
)


def _stage_positions(stage: int) -> Tuple[int, ...]:
    if stage == 1:
        return tuple(range(0, STAGE_1_ICP_COUNT))
    if stage == 2:
        return tuple(range(STAGE_1_ICP_COUNT, BENCHMARK_ICP_COUNT))
    raise ArenaContractError("stage must be 1 or 2")


def _validate_indexes(values: Sequence[int], path: str) -> None:
    previous = -1
    for value in values:
        if not 0 <= value < MAX_COMPANIES_PER_ICP:
            raise ArenaContractError("%s index out of range" % path)
        if value <= previous:
            raise ArenaContractError("%s must be strictly increasing" % path)
        previous = value


def _validate_row(row: Mapping[str, Any], stage: int, path: str) -> None:
    if row["icp_position"] not in _stage_positions(stage):
        raise ArenaContractError("%s.icp_position is outside stage %d" % (path, stage))
    scored = row["scored_company_indexes"]
    skipped = row["skipped_company_indexes"]
    _validate_indexes(scored, path + ".scored_company_indexes")
    _validate_indexes(skipped, path + ".skipped_company_indexes")
    if set(scored) & set(skipped):
        raise ArenaContractError("%s scored and skipped indexes overlap" % path)
    if row["cause"] == ACCEPTED_CAUSE:
        if row.get("output_hash") is None:
            raise ArenaContractError("%s accepted row requires output_hash" % path)
        if len(row["breakdowns"]) != len(scored):
            raise ArenaContractError("%s breakdown count must equal scored company count" % path)
        for index, breakdown in enumerate(row["breakdowns"]):
            if not breakdown_is_redacted(breakdown):
                raise ArenaContractError("%s.breakdowns[%d] is not redacted" % (path, index))
        return
    if row.get("output_hash") is not None:
        raise ArenaContractError("%s zero row cannot carry an output_hash" % path)
    if scored or skipped or row["breakdowns"]:
        raise ArenaContractError("%s zero row cannot carry companies or breakdowns" % path)
    if row["fp_gate_count"] or row["fp_unverified_primary_count"] or row["per_icp_score"] != 0.0:
        raise ArenaContractError("%s zero row must score exactly zero" % path)


def validate_score_bundle(document: Any) -> Dict[str, Any]:
    """Validate a score bundle; ``bundle_hash`` is verified over the published bytes."""

    _require_mapping(document, "score bundle")
    check_strict_document(document, PUBLICATION_LIMITS)
    bundle = validate_document(document, SCORE_BUNDLE_FIELDS)
    policy = validate_scorer_policy(bundle["scorer_policy"])
    if "policy_hash" not in policy:
        raise ArenaContractError("score bundle scorer_policy must carry policy_hash")
    bundle["scorer_policy"] = policy
    stage = bundle["stage"]
    if stage == 2 and not bundle.get("stage_1_bundle_hash"):
        raise ArenaContractError("stage 2 bundle must bind stage_1_bundle_hash")
    if stage == 1 and bundle.get("stage_1_bundle_hash") is not None:
        raise ArenaContractError("stage 1 bundle cannot bind a stage 1 bundle")
    seen = set()  # type: set
    submission_ids = set()  # type: set
    for index, row in enumerate(bundle["rows"]):
        path = "$.rows[%d]" % index
        key = (row["submission_id"], row["icp_position"])
        if key in seen:
            raise ArenaContractError("%s duplicates submission/ICP pair" % path)
        seen.add(key)
        submission_ids.add(row["submission_id"])
        _validate_row(row, stage, path)
    scores = {}  # type: Dict[str, float]
    for key, value in bundle["submission_scores"].items():
        if not isinstance(key, str) or not key:
            raise ArenaContractError("submission_scores keys must be submission ids")
        scores[key] = _require_score(value, "submission_scores.%s" % key)
    if set(scores) != submission_ids:
        raise ArenaContractError("submission_scores must cover exactly the submissions with rows")
    bundle["submission_scores"] = scores
    if "bundle_hash" in document:
        verify_hashed_document(document, "bundle_hash")
    return bundle


def finalize_score_bundle(document: Mapping[str, Any]) -> Dict[str, Any]:
    unsigned = {k: v for k, v in document.items() if k not in ("bundle_hash", "signature")}
    validate_score_bundle(unsigned)
    return hashed_document(unsigned, "bundle_hash")


def output_companies(output_document: Any) -> List[Any]:
    """Companies of a published output document (an object with ``companies`` or a bare list)."""

    if isinstance(output_document, Mapping):
        return _require_list(output_document.get("companies"), "output.companies")
    return _require_list(output_document, "output")


def participant_artifact_hash(participant: Mapping[str, Any]) -> str:
    """The frozen artifact hash used for salted tie-breaks (``artifact_hash``, else ``source_tree_hash``)."""

    value = participant.get("artifact_hash")
    if value is None:
        value = participant.get("source_tree_hash")
    return require_sha256(value, "artifact_hash")


# ---------------------------------------------------------------------------
# Whole-round rebuild (sections 12.2-12.4)
# ---------------------------------------------------------------------------


class VerificationReport:
    """Ordered check ledger with the closed vocabulary verified/failed/not_checked."""

    def __init__(self) -> None:
        self.checks = []  # type: List[Dict[str, str]]

    def verified(self, check: str, detail: str = "") -> None:
        self.checks.append({"check": check, "status": "verified", "detail": detail})

    def failed(self, check: str, detail: str) -> None:
        self.checks.append({"check": check, "status": "failed", "detail": detail})

    def not_checked(self, check: str, detail: str) -> None:
        self.checks.append({"check": check, "status": "not_checked", "detail": detail})

    @property
    def ok(self) -> bool:
        return bool(self.checks) and not any(item["status"] == "failed" for item in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": list(self.checks), "statement": VERIFIER_STATEMENT}


def _verify_signed(
    document: Mapping[str, Any],
    *,
    hash_field: str,
    public_key_der: bytes,
    key_hash: str,
) -> str:
    return verify_document_signature(
        document,
        hash_field=hash_field,
        public_key_der=public_key_der,
        expected_public_key_hash=key_hash,
    )


def _verify_authority_documents(
    bundle: Mapping[str, Any],
    signing_key_document: Mapping[str, Any],
    report: VerificationReport,
    icp_hash_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    public_key_der = load_public_key_from_document(signing_key_document)
    key_hash = public_key_hash(public_key_der)

    config_doc = _require_mapping(bundle.get("round_configuration"), "round_configuration")
    config = validate_round_configuration(config_doc)
    if config.get("signing_public_key_hash") != key_hash:
        raise ArenaSignatureError("round configuration pins a different signing key")
    _verify_signed(config_doc, hash_field="configuration_hash", public_key_der=public_key_der, key_hash=key_hash)
    report.verified("round_configuration.signature", config["configuration_hash"])

    commitment_doc = _require_mapping(bundle.get("benchmark_commitment"), "benchmark_commitment")
    commitment = validate_benchmark_commitment(commitment_doc)
    _verify_signed(commitment_doc, hash_field="commitment_hash", public_key_der=public_key_der, key_hash=key_hash)
    if commitment["configuration_hash"] != config["configuration_hash"] or commitment["round_id"] != config["round_id"]:
        raise ArenaContractError("benchmark commitment does not bind this round configuration")
    report.verified("benchmark_commitment.signature", commitment["commitment_hash"])

    participants = [_require_mapping(item, "participant") for item in _require_list(bundle.get("participants"), "participants")]
    if participant_set_hash(participants) != commitment["participant_set_hash"]:
        raise ArenaContractError("participant set does not match the commitment")
    ids = [str(item["submission_id"]) for item in participants]
    if len(set(ids)) != len(ids):
        raise ArenaContractError("duplicate participant submission ids")
    kings = [item for item in participants if bool(item.get("is_king", False))]
    if len(kings) > 1:
        raise ArenaContractError("more than one king in the participant set")
    challengers = [item for item in participants if not bool(item.get("is_king", False))]
    if len(challengers) > config["max_challengers"]:
        raise ArenaContractError("participant set exceeds max_challengers")
    for item in participants:
        participant_artifact_hash(item)
    report.verified("participants.set_hash", commitment["participant_set_hash"])

    policy_doc = _require_mapping(bundle.get("scorer_policy"), "scorer_policy")
    policy = validate_scorer_policy(policy_doc)
    if policy.get("policy_hash") != config["scorer_policy_hash"]:
        raise ArenaContractError("scorer policy hash does not match the round configuration")
    report.verified("scorer_policy.hash", policy["policy_hash"])

    benchmark = [_require_mapping(item, "benchmark ICP") for item in _require_list(bundle.get("benchmark"), "benchmark")]
    if len(benchmark) != BENCHMARK_ICP_COUNT:
        raise ArenaContractError("benchmark must publish exactly %d ICPs" % BENCHMARK_ICP_COUNT)
    icp_hashes = [require_sha256(icp_hash_fn(icp), "icp_hash") for icp in benchmark]
    roots = benchmark_roots(icp_hashes)
    if roots["icp_leaf_hashes"] != commitment["icp_leaf_hashes"] or roots["benchmark_root"] != commitment["benchmark_root"]:
        raise ArenaContractError("published benchmark does not match the committed leaves")
    report.verified("benchmark.commitment_leaves", commitment["benchmark_root"])

    plans_doc = _require_mapping(bundle.get("stage_plans"), "stage_plans")
    bundles_doc = _require_mapping(bundle.get("score_bundles"), "score_bundles")
    plans = {}  # type: Dict[int, Dict[str, Any]]
    score_bundles = {}  # type: Dict[int, Dict[str, Any]]
    for stage in (1, 2):
        plan_doc = _require_mapping(plans_doc.get(str(stage)), "stage_plans.%d" % stage)
        plan = validate_scoring_plan(plan_doc)
        _verify_signed(plan_doc, hash_field="plan_hash", public_key_der=public_key_der, key_hash=key_hash)
        if (
            plan["stage"] != stage
            or plan["round_id"] != config["round_id"]
            or plan["configuration_hash"] != config["configuration_hash"]
            or plan["commitment_hash"] != commitment["commitment_hash"]
            or plan["scorer_policy_hash"] != policy["policy_hash"]
        ):
            raise ArenaContractError("stage %d scoring plan does not bind this round" % stage)
        plans[stage] = plan
        report.verified("scoring_plan.%d.signature" % stage, plan["plan_hash"])

        bundle_doc = _require_mapping(bundles_doc.get(str(stage)), "score_bundles.%d" % stage)
        score_bundle = validate_score_bundle(bundle_doc)
        _verify_signed(bundle_doc, hash_field="bundle_hash", public_key_der=public_key_der, key_hash=key_hash)
        if (
            score_bundle["stage"] != stage
            or score_bundle["round_id"] != config["round_id"]
            or score_bundle["scoring_plan_hash"] != plan["plan_hash"]
            or score_bundle["scorer_policy"]["policy_hash"] != policy["policy_hash"]
        ):
            raise ArenaContractError("stage %d score bundle does not bind its plan and policy" % stage)
        if stage == 2 and score_bundle.get("stage_1_bundle_hash") != score_bundles[1]["bundle_hash"]:
            raise ArenaContractError("stage 2 score bundle does not bind the stage 1 bundle")
        score_bundles[stage] = score_bundle
        report.verified("score_bundle.%d.signature" % stage, score_bundle["bundle_hash"])

    return {
        "public_key_der": public_key_der,
        "key_hash": key_hash,
        "config": config,
        "commitment": commitment,
        "participants": participants,
        "king": kings[0] if kings else None,
        "challengers": challengers,
        "policy": policy,
        "benchmark": benchmark,
        "icp_hashes": icp_hashes,
        "plans": plans,
        "score_bundles": score_bundles,
        "salt": commitment["tie_break_block_hash"],
    }


def _plan_indexes(plan: Mapping[str, Any]) -> Tuple[Dict[Tuple[str, str], Mapping[str, Any]], Dict[Tuple[str, int], str]]:
    work_items = {}  # type: Dict[Tuple[str, str], Mapping[str, Any]]
    for item in plan["work_items"]:
        work_items[(item["icp_hash"], item["output_hash"])] = item
    zero_rows = {}  # type: Dict[Tuple[str, int], str]
    for item in plan["zero_rows"]:
        key = (item["submission_id"], item["icp_position"])
        if key in zero_rows:
            raise ArenaContractError("scoring plan repeats zero row %s/%d" % key)
        zero_rows[key] = item["cause"]
    return work_items, zero_rows


def _recompute_stage_rows(
    stage: int,
    context: Mapping[str, Any],
    outputs: Mapping[str, Any],
    output_hash_fn: Callable[[Any], str],
    expected_submissions: Sequence[str],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Recompute every row of one stage; returns ``{submission_id: {position: row}}``."""

    plan = context["plans"][stage]
    score_bundle = context["score_bundles"][stage]
    policy = context["policy"]
    work_items, zero_rows = _plan_indexes(plan)
    positions = _stage_positions(stage)
    expected = {(submission_id, position) for submission_id in expected_submissions for position in positions}
    seen = set()  # type: set
    breakdown_hash_by_work_item = {}  # type: Dict[str, str]
    rows_by_submission = {}  # type: Dict[str, Dict[int, Dict[str, Any]]]
    for index, row in enumerate(score_bundle["rows"]):
        path = "stage %d row %d" % (stage, index)
        submission_id = row["submission_id"]
        position = row["icp_position"]
        key = (submission_id, position)
        if key not in expected:
            raise ArenaContractError("%s covers %s at ICP %d which this stage must not score" % (path, submission_id, position))
        seen.add(key)
        icp = context["benchmark"][position]
        icp_hash = context["icp_hashes"][position]
        if row["icp_hash"] != icp_hash:
            raise ArenaContractError("%s icp_hash does not match the committed ICP" % path)
        if row["cause"] == ACCEPTED_CAUSE:
            output_hash = row["output_hash"]
            work_item = work_items.get((icp_hash, output_hash))
            if work_item is None or submission_id not in work_item["submission_ids"]:
                raise ArenaContractError("%s has no scoring-plan work item for its ICP and output" % path)
            if output_hash not in outputs:
                raise ArenaContractError("%s output %s is not published" % (path, output_hash))
            output_document = outputs[output_hash]
            if output_hash_fn(output_document) != output_hash:
                raise ArenaContractError("%s published output does not hash to %s" % (path, output_hash))
            sliced = slice_first_n(output_companies(output_document), icp_company_goal(icp))
            scored, skipped = bucket_skip(icp, sliced, max_scored_companies=policy["max_scored_companies"])
            if scored != list(row["scored_company_indexes"]) or skipped != list(row["skipped_company_indexes"]):
                raise ArenaContractError("%s first-N slice or bucket skip does not match the published output" % path)
            result = per_icp_score(icp, row["breakdowns"], policy, icp_hash=icp_hash)
            if (
                result["fp_gate_count"] != row["fp_gate_count"]
                or result["fp_unverified_primary_count"] != row["fp_unverified_primary_count"]
            ):
                raise ArenaContractError("%s false-positive counts do not match the breakdowns" % path)
            if result["per_icp_score"] != row["per_icp_score"]:
                raise ArenaContractError(
                    "%s per-ICP score %r does not recompute (%r)" % (path, row["per_icp_score"], result["per_icp_score"])
                )
            breakdown_hash = document_hash(row["breakdowns"])
            previous = breakdown_hash_by_work_item.setdefault(work_item["work_item_id"], breakdown_hash)
            if previous != breakdown_hash:
                raise ArenaContractError("%s breakdowns differ from another row of the same work item" % path)
        else:
            planned_cause = zero_rows.get(key)
            if planned_cause is None or planned_cause != row["cause"]:
                raise ArenaContractError("%s zero row is not in the scoring plan with cause %s" % (path, row["cause"]))
        rows_by_submission.setdefault(submission_id, {})[position] = row
    if seen != expected:
        missing = sorted(expected - seen)[:5]
        raise ArenaContractError("stage %d rows are incomplete; first missing: %s" % (stage, missing))
    return rows_by_submission


def _compare_rows(published: Any, rebuilt: Sequence[Mapping[str, Any]], name: str) -> None:
    rows = _require_list(published, name)
    if len(rows) != len(rebuilt):
        raise ArenaContractError("%s length differs from the rebuilt ranking" % name)
    for index, (left, right) in enumerate(zip(rows, rebuilt)):
        _require_mapping(left, "%s[%d]" % (name, index))
        for key, value in right.items():
            if left.get(key) != value:
                raise ArenaContractError("%s[%d].%s differs from the rebuilt value" % (name, index, key))


def _rebuild_scores_and_decision(
    bundle: Mapping[str, Any],
    context: Mapping[str, Any],
    report: VerificationReport,
    output_hash_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    outputs = _require_mapping(bundle.get("outputs"), "outputs")
    config = context["config"]
    participants = context["participants"]
    king = context["king"]
    challengers = context["challengers"]
    salt = context["salt"]
    by_id = {str(item["submission_id"]): item for item in participants}
    all_ids = [str(item["submission_id"]) for item in participants]

    stage1_rows = _recompute_stage_rows(1, context, outputs, output_hash_fn, all_ids)
    stage1_scores = {}  # type: Dict[str, float]
    for submission_id in all_ids:
        rows = stage1_rows[submission_id]
        stage1_scores[submission_id] = stage_score(
            [rows[position]["per_icp_score"] for position in _stage_positions(1)], STAGE_1_DENOMINATOR
        )
    published_stage1 = context["score_bundles"][1]["submission_scores"]
    if published_stage1 != stage1_scores:
        raise ArenaContractError("stage 1 submission scores do not recompute")
    report.verified("stage_1.per_icp_and_stage_scores", "%d submissions" % len(stage1_scores))

    ranking = stage1_ranking(
        [
            {
                "submission_id": submission_id,
                "artifact_hash": participant_artifact_hash(by_id[submission_id]),
                "stage1_score": stage1_scores[submission_id],
                "is_king": bool(by_id[submission_id].get("is_king", False)),
            }
            for submission_id in all_ids
        ],
        salt,
    )
    _compare_rows(bundle.get("stage1_ranking"), ranking, "stage1_ranking")
    report.verified("stage_1.ranking", "%d challengers" % len(ranking))
    finalists = select_finalists(ranking)
    if list(_require_list(bundle.get("finalists"), "finalists")) != finalists:
        raise ArenaContractError("published finalists differ from the rebuilt finalist set")
    report.verified("stage_1.finalists", ", ".join(finalists))

    if config["all_participants_run_stage_2"]:
        stage2_ids = list(all_ids)
    else:
        stage2_ids = list(finalists) + ([str(king["submission_id"])] if king is not None else [])
    stage2_rows = _recompute_stage_rows(2, context, outputs, output_hash_fn, stage2_ids)

    final_scores = {}  # type: Dict[str, float]
    valid = {}  # type: Dict[str, bool]
    for submission_id in stage2_ids:
        rows = dict(stage1_rows[submission_id])
        rows.update(stage2_rows[submission_id])
        all_positions = _stage_positions(1) + _stage_positions(2)
        final_scores[submission_id] = stage_score(
            [rows[position]["per_icp_score"] for position in all_positions], FINAL_DENOMINATOR
        )
        valid[submission_id] = result_is_valid(rows, all_positions)
    published_final = context["score_bundles"][2]["submission_scores"]
    if published_final != final_scores:
        raise ArenaContractError("final submission scores do not recompute")
    report.verified("final.per_icp_and_final_scores", "%d submissions" % len(final_scores))

    final_entries = [
        {
            "submission_id": submission_id,
            "hotkey": str(by_id[submission_id]["miner_hotkey"]),
            "artifact_hash": participant_artifact_hash(by_id[submission_id]),
            "final_score": final_scores[submission_id] if valid[submission_id] else None,
            "is_king": bool(by_id[submission_id].get("is_king", False)),
        }
        for submission_id in stage2_ids
    ]
    ranked = final_ranking(final_entries, salt)
    _compare_rows(bundle.get("final_ranking"), ranked, "final_ranking")
    report.verified("final.ranking", "%d participants" % len(ranked))

    king_entry = None
    finalist_entries = []  # type: List[Dict[str, Any]]
    for entry in final_entries:
        if entry["is_king"]:
            king_entry = entry
        else:
            finalist_entries.append(entry)
    decision = king_decision(finalist_entries, king_entry, salt)
    published_decision = _require_mapping(bundle.get("king_decision"), "king_decision")
    if {key: published_decision.get(key) for key in decision} != decision:
        raise ArenaContractError("published king decision differs from the rebuilt decision")
    report.verified("final.king_decision", decision["outcome"])

    if "reward_basis" in bundle:
        basis_doc = _require_mapping(bundle.get("reward_basis"), "reward_basis")
        basis = validate_reward_basis(basis_doc)
        _verify_signed(
            basis_doc,
            hash_field="reward_basis_hash",
            public_key_der=context["public_key_der"],
            key_hash=context["key_hash"],
        )
        if (
            basis["round_id"] != config["round_id"]
            or basis["configuration_hash"] != config["configuration_hash"]
            or basis["commitment_hash"] != context["commitment"]["commitment_hash"]
            or basis["king_outcome"] != decision["outcome"]
            or basis["king_hotkey"] != decision["king_hotkey"]
        ):
            raise ArenaContractError("reward basis does not match this round's rebuilt decision")
        report.verified("reward_basis.signature_and_decision", basis["reward_basis_hash"])
        report.not_checked(
            "reward_basis.result_bundle_hash",
            "the result-bundle hash binding is defined by publication, not by this verifier",
        )
    else:
        report.not_checked("reward_basis", "no reward basis published in this bundle")
    return {
        "stage1_scores": stage1_scores,
        "finalists": finalists,
        "final_scores": final_scores,
        "decision": decision,
    }


def rebuild_round(
    public_bundle: Mapping[str, Any],
    signing_key_document: Mapping[str, Any],
    *,
    output_hash_fn: Optional[Callable[[Any], str]] = None,
    icp_hash_fn: Optional[Callable[[Any], str]] = None,
) -> Dict[str, Any]:
    """Rebuild every aggregate and decision of a published round.

    ``public_bundle`` holds ``round_configuration``, ``benchmark_commitment``,
    ``benchmark`` (50 ICPs in slot order), ``participants``, ``scorer_policy``,
    ``stage_plans`` and ``score_bundles`` keyed ``"1"``/``"2"``, ``outputs``
    keyed by output hash, ``stage1_ranking``, ``finalists``, ``final_ranking``,
    ``king_decision`` and optionally ``reward_basis``. ``signing_key_document``
    is the ``GET /signing-key`` document; its key hash must equal
    ``round_configuration.signing_public_key_hash``. ``output_hash_fn`` and
    ``icp_hash_fn`` default to the canonical document hash.

    Returns ``{"ok", "checks", "statement"}``. A failed check never raises;
    an unexpected exception inside a phase is recorded as a failed check and
    the dependent phase is ``not_checked``. This function verifies published
    material only and claims nothing about original execution.
    """

    report = VerificationReport()
    output_hash = output_hash_fn or document_hash
    icp_hash = icp_hash_fn or document_hash
    try:
        context = _verify_authority_documents(public_bundle, signing_key_document, report, icp_hash)
    except (ArenaContractError, ValueError, TypeError, KeyError) as exc:
        report.failed("authority_documents", "%s: %s" % (type(exc).__name__, exc))
        report.not_checked("rebuild", "authority documents failed verification")
        return report.to_dict()
    try:
        _rebuild_scores_and_decision(public_bundle, context, report, output_hash)
    except (ArenaContractError, ValueError, TypeError, KeyError) as exc:
        report.failed("rebuild", "%s: %s" % (type(exc).__name__, exc))
    return report.to_dict()
