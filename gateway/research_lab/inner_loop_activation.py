"""Automatic, fail-closed activation policy for Research Lab inner-loop ranking."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import math
import os
from typing import Any, Iterable, Mapping, Sequence

from research_lab.canonical import sha256_json

INNER_LOOP_MODE_ENV = "RESEARCH_LAB_INNER_LOOP_MODE"
INNER_LOOP_MODES = frozenset({"off", "auto", "observe", "shadow", "rank"})
INNER_LOOP_PHASES = frozenset({"off", "observe", "shadow", "rank"})
INNER_LOOP_EVENT_TABLE = "research_lab_inner_loop_activation_events"

DEFAULT_DEV_SET_SIZE = 8
DEFAULT_CANDIDATE_WIDTH = 3
DEFAULT_PAID_FINALIST_COUNT = 1
DEFAULT_SNAPSHOT_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
DEFAULT_FINALIZATION_RESERVE_SECONDS = 120

OBSERVE_RUN_GATE = 10
OBSERVE_SPAN_SECONDS_GATE = 24 * 60 * 60
SHADOW_RUN_GATE = 20
SHADOW_SPAN_SECONDS_GATE = 7 * 24 * 60 * 60
HISTORICAL_PAIR_GATE = 30
SPEARMAN_GATE = 0.20
CANDIDATE_ELIGIBILITY_GATE = 0.95


def configured_inner_loop_mode(value: str | None = None) -> str:
    raw = str(value if value is not None else os.getenv(INNER_LOOP_MODE_ENV) or "off")
    mode = raw.strip().lower()
    return mode if mode in INNER_LOOP_MODES else "off"


@dataclass(frozen=True)
class InnerLoopEvidence:
    current_phase: str = "observe"
    observe_eligible_runs: int = 0
    observe_span_seconds: float = 0.0
    shadow_healthy_runs: int = 0
    shadow_span_seconds: float = 0.0
    rank_healthy_runs: int = 0
    rank_span_seconds: float = 0.0
    historical_pair_count: int = 0
    spearman_rho: float | None = None
    top_quartile_lift: float | None = None
    candidate_eligibility_rate: float = 0.0
    unclassified_error_count: int = 0
    silent_miss_count: int = 0
    candidate_width_invariant_violations: int = 0
    paid_finalist_invariant_violations: int = 0
    protected_workflow_invariant_violations: int = 0
    evidence_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "observe_eligible_runs": self.observe_eligible_runs,
            "observe_span_seconds": round(self.observe_span_seconds, 3),
            "shadow_healthy_runs": self.shadow_healthy_runs,
            "shadow_span_seconds": round(self.shadow_span_seconds, 3),
            "rank_healthy_runs": self.rank_healthy_runs,
            "rank_span_seconds": round(self.rank_span_seconds, 3),
            "historical_pair_count": self.historical_pair_count,
            "spearman_rho": self.spearman_rho,
            "top_quartile_lift": self.top_quartile_lift,
            "candidate_eligibility_rate": round(self.candidate_eligibility_rate, 6),
            "unclassified_error_count": self.unclassified_error_count,
            "silent_miss_count": self.silent_miss_count,
            "candidate_width_invariant_violations": self.candidate_width_invariant_violations,
            "paid_finalist_invariant_violations": self.paid_finalist_invariant_violations,
            "protected_workflow_invariant_violations": self.protected_workflow_invariant_violations,
            "evidence_error": self.evidence_error,
        }


@dataclass(frozen=True)
class InnerLoopPolicy:
    requested_mode: str
    effective_phase: str
    evaluator_enabled: bool
    ranking_enabled: bool
    shadow_enabled: bool
    candidate_width: int
    paid_finalist_count: int
    strict_fallback: bool
    fallback_reason: str
    snapshot_manifest_hash: str = ""
    dev_set_hash: str = ""
    snapshot_recorded_at: str = ""
    snapshot_age_seconds: float | None = None
    dev_set_size: int = 0
    evaluation_timeout_seconds: int = 300
    finalization_reserve_seconds: int = DEFAULT_FINALIZATION_RESERVE_SECONDS
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def evaluator_commitment(self) -> dict[str, Any]:
        payload = {
            "schema_version": "research_lab.inner_loop_evaluator_commitment.v1",
            "effective_phase": self.effective_phase,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "dev_set_hash": self.dev_set_hash,
            "dev_set_size": self.dev_set_size,
            "miss_policy": "strict",
            "strict_fallback": self.strict_fallback,
        }
        return {**payload, "commitment_hash": sha256_json(payload)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "research_lab.inner_loop_policy.v1",
            "requested_mode": self.requested_mode,
            "effective_phase": self.effective_phase,
            "evaluator_enabled": self.evaluator_enabled,
            "ranking_enabled": self.ranking_enabled,
            "shadow_enabled": self.shadow_enabled,
            "candidate_width": self.candidate_width,
            "paid_finalist_count": self.paid_finalist_count,
            "strict_fallback": self.strict_fallback,
            "fallback_reason": self.fallback_reason,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "dev_set_hash": self.dev_set_hash,
            "snapshot_recorded_at": self.snapshot_recorded_at,
            "snapshot_age_seconds": self.snapshot_age_seconds,
            "dev_set_size": self.dev_set_size,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "evaluator_commitment": self.evaluator_commitment,
            "evidence": dict(self.evidence),
        }


def decide_inner_loop_policy(
    *,
    requested_mode: str,
    snapshot_readiness: Mapping[str, Any] | None,
    evidence: InnerLoopEvidence | None = None,
    dev_eval_kill_switch_enabled: bool,
    configured_candidate_width: int = DEFAULT_CANDIDATE_WIDTH,
    configured_paid_finalist_count: int = DEFAULT_PAID_FINALIST_COUNT,
    stop_at_candidate_cap_enabled: bool = True,
    evaluation_timeout_seconds: int = 300,
) -> InnerLoopPolicy:
    """Return one deterministic policy; unsafe inputs always reduce scope."""

    mode = configured_inner_loop_mode(requested_mode)
    readiness = dict(snapshot_readiness or {})
    metrics = evidence or InnerLoopEvidence(evidence_error="activation_evidence_unavailable")
    snapshot_ready = bool(readiness.get("ready"))
    snapshot_age = _finite_or_none(readiness.get("snapshot_age_seconds"))
    snapshot_fresh = (
        snapshot_ready
        and snapshot_age is not None
        and snapshot_age <= DEFAULT_SNAPSHOT_MAX_AGE_SECONDS
    )
    dev_set_size = _int_nonnegative(readiness.get("dev_set_size"))
    snapshot_valid = snapshot_fresh and dev_set_size == DEFAULT_DEV_SET_SIZE
    evaluator_ready = bool(dev_eval_kill_switch_enabled and snapshot_valid)

    fallback_reason = ""
    if mode == "off":
        phase = "off"
    elif mode == "auto":
        phase = _automatic_phase(metrics)
    else:
        phase = mode

    if configured_paid_finalist_count != DEFAULT_PAID_FINALIST_COUNT:
        phase = "observe"
        fallback_reason = "paid_finalist_count_must_equal_one"
    elif int(configured_candidate_width or 0) != DEFAULT_CANDIDATE_WIDTH and phase in {"shadow", "rank"}:
        phase = "observe"
        fallback_reason = "candidate_width_must_equal_three"
    elif not stop_at_candidate_cap_enabled and phase in {"shadow", "rank"}:
        phase = "observe"
        fallback_reason = "stop_at_candidate_cap_must_be_enabled"
    elif not dev_eval_kill_switch_enabled and phase != "off":
        phase = "observe"
        fallback_reason = "dev_eval_kill_switch_disabled"
    elif not snapshot_valid and phase != "off":
        phase = "observe"
        fallback_reason = str(readiness.get("reason") or "snapshot_not_ready")

    if metrics.protected_workflow_invariant_violations > 0:
        phase = "off"
        fallback_reason = "protected_workflow_invariant_violation"

    evaluator_enabled = evaluator_ready and phase in {"observe", "shadow", "rank"}
    candidate_width = (
        max(2, min(DEFAULT_CANDIDATE_WIDTH, int(configured_candidate_width or 1)))
        if evaluator_enabled and phase in {"shadow", "rank"}
        else 1
    )
    ranking_enabled = evaluator_enabled and phase == "rank"
    shadow_enabled = evaluator_enabled and phase == "shadow"

    return InnerLoopPolicy(
        requested_mode=mode,
        effective_phase=phase,
        evaluator_enabled=evaluator_enabled,
        ranking_enabled=ranking_enabled,
        shadow_enabled=shadow_enabled,
        candidate_width=candidate_width,
        paid_finalist_count=DEFAULT_PAID_FINALIST_COUNT,
        strict_fallback=True,
        fallback_reason=fallback_reason,
        snapshot_manifest_hash=str(readiness.get("manifest_hash") or ""),
        dev_set_hash=str(readiness.get("dev_set_hash") or ""),
        snapshot_recorded_at=str(readiness.get("recorded_at") or ""),
        snapshot_age_seconds=snapshot_age,
        dev_set_size=dev_set_size,
        evaluation_timeout_seconds=max(30, int(evaluation_timeout_seconds or 300)),
        evidence=metrics.to_dict(),
    )


async def resolve_inner_loop_policy(
    *,
    requested_mode: str,
    snapshot_readiness: Mapping[str, Any] | None,
    dev_eval_kill_switch_enabled: bool,
    configured_candidate_width: int,
    configured_paid_finalist_count: int,
    stop_at_candidate_cap_enabled: bool,
    evaluation_timeout_seconds: int = 300,
    store: Any | None = None,
) -> InnerLoopPolicy:
    """Resolve and persist one automatic phase transition, retrying a race once."""
    mode = configured_inner_loop_mode(requested_mode)
    if mode != "auto":
        return decide_inner_loop_policy(
            requested_mode=mode,
            snapshot_readiness=snapshot_readiness,
            evidence=InnerLoopEvidence(current_phase=mode),
            dev_eval_kill_switch_enabled=dev_eval_kill_switch_enabled,
            configured_candidate_width=configured_candidate_width,
            configured_paid_finalist_count=configured_paid_finalist_count,
            stop_at_candidate_cap_enabled=stop_at_candidate_cap_enabled,
            evaluation_timeout_seconds=evaluation_timeout_seconds,
        )
    try:
        evidence = await load_inner_loop_evidence(store=store)
    except Exception as exc:
        evidence = InnerLoopEvidence(
            evidence_error=f"activation_evidence_error:{type(exc).__name__}"
        )
    policy = decide_inner_loop_policy(
        requested_mode=requested_mode,
        snapshot_readiness=snapshot_readiness,
        evidence=evidence,
        dev_eval_kill_switch_enabled=dev_eval_kill_switch_enabled,
        configured_candidate_width=configured_candidate_width,
        configured_paid_finalist_count=configured_paid_finalist_count,
        stop_at_candidate_cap_enabled=stop_at_candidate_cap_enabled,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
    )
    if evidence.evidence_error:
        return replace(
            policy,
            effective_phase="observe",
            ranking_enabled=False,
            shadow_enabled=False,
            candidate_width=1,
            fallback_reason=evidence.evidence_error,
        )
    if policy.effective_phase == evidence.current_phase:
        return policy
    try:
        await record_inner_loop_event(
            event_type="phase_transition",
            phase=policy.effective_phase,
            evidence_doc={
                "from_phase": evidence.current_phase,
                "to_phase": policy.effective_phase,
                "reason": policy.fallback_reason or "automatic_evidence_gate",
                "evidence": evidence.to_dict(),
            },
            expected_current_phase=evidence.current_phase,
            store=store,
        )
        return policy
    except Exception as transition_exc:
        try:
            refreshed = await load_inner_loop_evidence(store=store)
        except Exception as exc:
            return replace(
                policy,
                effective_phase="observe",
                ranking_enabled=False,
                shadow_enabled=False,
                candidate_width=1,
                fallback_reason=f"activation_transition_error:{type(exc).__name__}",
            )
        if refreshed.current_phase == policy.effective_phase:
            return decide_inner_loop_policy(
                requested_mode=requested_mode,
                snapshot_readiness=snapshot_readiness,
                evidence=refreshed,
                dev_eval_kill_switch_enabled=dev_eval_kill_switch_enabled,
                configured_candidate_width=configured_candidate_width,
                configured_paid_finalist_count=configured_paid_finalist_count,
                stop_at_candidate_cap_enabled=stop_at_candidate_cap_enabled,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
            )
        safe_phase = "off" if policy.effective_phase == "off" else "observe"
        return replace(
            policy,
            effective_phase=safe_phase,
            evaluator_enabled=(policy.evaluator_enabled and safe_phase == "observe"),
            ranking_enabled=False,
            shadow_enabled=False,
            candidate_width=1,
            fallback_reason=(
                "activation_transition_uncommitted:"
                + type(transition_exc).__name__
            ),
        )


def _automatic_phase(evidence: InnerLoopEvidence) -> str:
    current = evidence.current_phase if evidence.current_phase in INNER_LOOP_PHASES else "observe"
    if evidence.protected_workflow_invariant_violations > 0:
        return "off"
    if evidence.paid_finalist_invariant_violations > 0:
        return "observe"
    if evidence.candidate_width_invariant_violations > 0:
        return "observe"
    if current == "rank" and (
        evidence.unclassified_error_count > 0
        or evidence.silent_miss_count > 0
        or evidence.candidate_eligibility_rate < CANDIDATE_ELIGIBILITY_GATE
    ):
        return "shadow"
    if current in {"shadow", "rank"} and _rank_gate_passes(evidence):
        return "rank"
    if current in {"shadow", "rank"}:
        return "shadow"
    if (
        evidence.observe_eligible_runs >= OBSERVE_RUN_GATE
        and evidence.observe_span_seconds >= OBSERVE_SPAN_SECONDS_GATE
        and evidence.unclassified_error_count == 0
    ):
        return "shadow"
    return "observe"


def _rank_gate_passes(evidence: InnerLoopEvidence) -> bool:
    return bool(
        evidence.shadow_healthy_runs >= SHADOW_RUN_GATE
        and evidence.shadow_span_seconds >= SHADOW_SPAN_SECONDS_GATE
        and evidence.historical_pair_count >= HISTORICAL_PAIR_GATE
        and evidence.spearman_rho is not None
        and evidence.spearman_rho >= SPEARMAN_GATE
        and evidence.top_quartile_lift is not None
        and evidence.top_quartile_lift > 0.0
        and evidence.candidate_eligibility_rate >= CANDIDATE_ELIGIBILITY_GATE
        and evidence.unclassified_error_count == 0
        and evidence.silent_miss_count == 0
        and evidence.candidate_width_invariant_violations == 0
        and evidence.paid_finalist_invariant_violations == 0
        and evidence.protected_workflow_invariant_violations == 0
    )


async def load_inner_loop_evidence(*, store: Any | None = None) -> InnerLoopEvidence:
    """Load restart-stable activation evidence with explicit pagination."""

    if store is None:
        from gateway.research_lab import store as store_module

        store = store_module
    events = await store.select_all(
        INNER_LOOP_EVENT_TABLE,
        columns="seq,event_type,phase,run_id,evidence_doc,created_at",
        filters=(),
        order_by=(("seq", True),),
        batch_size=500,
        max_rows=10000,
    )
    calibrations = await store.select_all(
        "research_lab_score_calibration",
        columns="candidate_id,dev_score,realized_mean_delta,created_at",
        filters=(),
        order_by=(("created_at", True),),
        batch_size=500,
        max_rows=10000,
        allow_partial=True,
    )
    return build_inner_loop_evidence(events=events, calibrations=calibrations)


def build_inner_loop_evidence(
    *,
    events: Sequence[Mapping[str, Any]],
    calibrations: Sequence[Mapping[str, Any]],
) -> InnerLoopEvidence:
    transitions = [row for row in events if row.get("event_type") == "phase_transition"]
    current_phase = str((transitions[0] if transitions else {}).get("phase") or "observe")
    observations = _dedupe_run_observations(
        row for row in events if row.get("event_type") == "run_observed"
    )

    observe_rows = _consecutive_healthy_rows(observations, "observe")
    shadow_rows = [
        row for row in observations if row.get("phase") == "shadow" and _healthy(row)
    ]
    rank_rows = [
        row for row in observations if row.get("phase") == "rank" and _healthy(row)
    ]
    recent_phase_rows = [
        row for row in observations if row.get("phase") == current_phase
    ][:20]
    if not recent_phase_rows and current_phase == "rank":
        # A just-entered rank phase has no rank observations yet. Carry the
        # shadow gate's recent evidence until the first ranked run arrives,
        # otherwise the controller would immediately undo its own transition.
        recent_phase_rows = [
            row for row in observations if row.get("phase") == "shadow"
        ][:20]
    candidate_total = sum(
        _doc_int(row, "candidate_count") for row in recent_phase_rows
    )
    eligible_total = sum(
        _doc_int(row, "eligible_candidate_count") for row in recent_phase_rows
    )
    eligibility_rate = eligible_total / candidate_total if candidate_total else 0.0
    unclassified_errors = sum(
        _doc_int(row, "unclassified_error_count") for row in recent_phase_rows
    )
    silent_misses = sum(
        _doc_int(row, "silent_miss_count") for row in recent_phase_rows
    )
    width_violations = sum(
        int(bool(_doc_value(row, "candidate_width_mismatch")))
        for row in recent_phase_rows
    )
    paid_violations = sum(
        _doc_int(row, "paid_finalist_invariant_violations")
        for row in recent_phase_rows
    )
    protected_violations = sum(
        _doc_int(row, "protected_workflow_invariant_violations")
        for row in recent_phase_rows
    )

    pairs = _unique_calibration_pairs(calibrations)
    return InnerLoopEvidence(
        current_phase=current_phase,
        observe_eligible_runs=len(observe_rows),
        observe_span_seconds=_row_span_seconds(observe_rows),
        shadow_healthy_runs=len(shadow_rows),
        shadow_span_seconds=_row_span_seconds(shadow_rows),
        rank_healthy_runs=len(rank_rows),
        rank_span_seconds=_row_span_seconds(rank_rows),
        historical_pair_count=len(pairs),
        spearman_rho=spearman_correlation(pairs),
        top_quartile_lift=top_quartile_lift(pairs),
        candidate_eligibility_rate=eligibility_rate,
        unclassified_error_count=unclassified_errors,
        silent_miss_count=silent_misses,
        candidate_width_invariant_violations=width_violations,
        paid_finalist_invariant_violations=paid_violations,
        protected_workflow_invariant_violations=protected_violations,
    )


async def record_inner_loop_event(
    *,
    event_type: str,
    phase: str,
    evidence_doc: Mapping[str, Any],
    run_id: str = "",
    expected_current_phase: str | None = None,
    store: Any | None = None,
) -> None:
    if store is None:
        from gateway.research_lab import store as store_module

        store = store_module
    payload = {
        "schema_version": "research_lab.inner_loop_activation_event.v1",
        "event_type": str(event_type),
        "phase": str(phase),
        "run_id": str(run_id) if run_id else None,
        "evidence_doc": dict(evidence_doc),
    }
    row = {**payload, "event_hash": sha256_json(payload)}
    try:
        if event_type == "phase_transition" and hasattr(store, "call_rpc"):
            await store.call_rpc(
                "append_research_lab_inner_loop_activation_event",
                {
                    "requested_event_type": str(event_type),
                    "requested_phase": str(phase),
                    "requested_run_id": str(run_id) if run_id else None,
                    "requested_evidence_doc": dict(evidence_doc),
                    "requested_event_hash": row["event_hash"],
                    "expected_current_phase": expected_current_phase,
                },
            )
        else:
            await store.insert_row(INNER_LOOP_EVENT_TABLE, row)
    except Exception as exc:
        text = str(exc).lower()
        if "23505" not in text and "duplicate" not in text and "unique" not in text:
            raise


def spearman_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x_ranks = _ranks([pair[0] for pair in pairs])
    y_ranks = _ranks([pair[1] for pair in pairs])
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks))
    x_var = sum((x - x_mean) ** 2 for x in x_ranks)
    y_var = sum((y - y_mean) ** 2 for y in y_ranks)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return round(numerator / math.sqrt(x_var * y_var), 6)


def top_quartile_lift(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 4:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    top_count = max(1, math.ceil(len(ordered) / 4))
    top_mean = sum(pair[1] for pair in ordered[:top_count]) / top_count
    overall_mean = sum(pair[1] for pair in ordered) / len(ordered)
    return round(top_mean - overall_mean, 6)


def _unique_calibration_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    """Use the newest finite realized pair for each immutable candidate."""
    pairs: list[tuple[float, float]] = []
    seen: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    ):
        candidate_id = str(row.get("candidate_id") or "").strip()
        dev_score = _finite_or_none(row.get("dev_score"))
        realized = _finite_or_none(row.get("realized_mean_delta"))
        if not candidate_id or candidate_id in seen or dev_score is None or realized is None:
            continue
        seen.add(candidate_id)
        pairs.append((dev_score, realized))
    return pairs


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original_index, _value in ordered[index:end]:
            ranks[original_index] = average
        index = end
    return ranks


def _consecutive_healthy_rows(
    rows: Sequence[Mapping[str, Any]], phase: str
) -> list[Mapping[str, Any]]:
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("phase") != phase or not _healthy(row):
            break
        kept.append(row)
    return kept


def _dedupe_run_observations(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep the newest observation for each logical run.

    The database uniqueness guard is authoritative. This defensive reduction
    keeps pre-migration or partially retried evidence from advancing gates.
    Input rows are already newest-first from ``load_inner_loop_evidence``.
    """
    kept: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        kept.append(row)
    return kept


def _healthy(row: Mapping[str, Any]) -> bool:
    doc = row.get("evidence_doc") if isinstance(row.get("evidence_doc"), Mapping) else {}
    return bool(doc.get("run_eligible")) and _doc_int(row, "unclassified_error_count") == 0


def _doc_int(row: Mapping[str, Any], key: str) -> int:
    return _int_nonnegative(_doc_value(row, key))


def _doc_value(row: Mapping[str, Any], key: str) -> Any:
    doc = row.get("evidence_doc") if isinstance(row.get("evidence_doc"), Mapping) else {}
    return doc.get(key)


def _row_span_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    timestamps = [_parse_timestamp(row.get("created_at")) for row in rows]
    valid = [stamp for stamp in timestamps if stamp is not None]
    return max(0.0, (max(valid) - min(valid)).total_seconds()) if len(valid) >= 2 else 0.0


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_nonnegative(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
