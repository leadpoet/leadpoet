"""Lab-only Research Loop v0 calibration harness.

Engine v0 is a bootstrap instrument: it runs a baseline-first loop against
frozen L1 fixtures, emits schema-valid trajectory/results records, and measures
rough loop depth. It does not write production data or alter champion gating.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from math import sqrt
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .canonical import sha256_json, sha256_text
from .schema_validation import assert_schema_record


ENGINE_VERSION = "research-loop-v0.1.0"
DEFAULT_ISLAND = "generalist"
DEFAULT_LOOP_BALANCE_USD = 10.0
DEFAULT_CREATED_AT = "2026-06-16T00:00:00Z"
DEFAULT_ENGINE_PROGRAM_PATH = Path(__file__).with_name("engine_program.md")
DEFAULT_FIXTURE_PATH = Path(__file__).with_name("fixtures").joinpath(
    "research_loop_v0_fixtures.json"
)
DEV_EVAL_T_ONE_SIDED_95_BY_DF = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    13: 1.771,
    14: 1.761,
    15: 1.753,
    16: 1.746,
    17: 1.740,
    18: 1.734,
    19: 1.729,
    20: 1.725,
    21: 1.721,
    22: 1.717,
    23: 1.714,
    24: 1.711,
    25: 1.708,
    26: 1.706,
    27: 1.703,
    28: 1.701,
    29: 1.699,
    30: 1.697,
}


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    parent_id: Optional[str]
    module: str
    function: str
    operator: str
    component: str
    patch_type: str
    patch_ref: str
    targeted_metric: str
    hypothesis: Dict[str, Any]
    model_used: str
    tokens_in: int
    tokens_out: int
    draft_cost_usd: float
    eval_cost_usd: float
    reflect_cost_usd: float
    diff_loc: int
    prompt_pack_tokens: int
    component_count: int


@dataclass(frozen=True)
class EvaluationResult:
    node_id: str
    status: str
    fixture_scores: List[float]
    metrics: Dict[str, Any]
    paired_lcb_vs_parent: Optional[float]
    delta_vs_parent: Optional[float]
    latency_s: float
    cache_hits: Dict[str, int]


@dataclass(frozen=True)
class LoopRun:
    trajectory: Dict[str, Any]
    ledger_rows: List[Dict[str, Any]]
    baseline: EvaluationResult
    evaluations: List[EvaluationResult]
    best_node_id: str


def default_node_specs() -> List[NodeSpec]:
    """Return the default two-node Phase 0 calibration batch."""
    return [
        NodeSpec(
            node_id="node-source-routing-v0",
            parent_id="baseline-reference",
            module="miner_models.qualification_research_loop_v0.qualify",
            function="qualify_source_routing_v0",
            operator="draft",
            component="qualification.discovery",
            patch_type="STRATEGY_SWAP",
            patch_ref="research-loop-v0-source-routing",
            targeted_metric="coverage",
            hypothesis={
                "failure_mode": "reference discovery misses company-domain evidence when a news result is sparse",
                "mechanism": "route candidate discovery through source-specific fixture evidence before ranking",
                "predicted_delta": 12.0,
                "falsifier": "coverage fails to improve on the frozen L1 fixtures",
            },
            model_used="engine-program-v0-local",
            tokens_in=1800,
            tokens_out=420,
            draft_cost_usd=0.08,
            eval_cost_usd=0.24,
            reflect_cost_usd=0.04,
            diff_loc=42,
            prompt_pack_tokens=0,
            component_count=1,
        ),
        NodeSpec(
            node_id="node-overbroad-v0",
            parent_id="baseline-reference",
            module="miner_models.qualification_research_loop_v0.qualify",
            function="qualify_overbroad_v0",
            operator="draft",
            component="qualification.discovery",
            patch_type="STRATEGY_SWAP",
            patch_ref="research-loop-v0-overbroad-routing",
            targeted_metric="coverage",
            hypothesis={
                "failure_mode": "raising recall by broadening queries may admit generic false positives",
                "mechanism": "over-broad routing keeps all high-scoring fixture candidates",
                "predicted_delta": 3.0,
                "falsifier": "evidence defect rate rises above the guardrail",
            },
            model_used="engine-program-v0-local",
            tokens_in=1600,
            tokens_out=360,
            draft_cost_usd=0.07,
            eval_cost_usd=0.24,
            reflect_cost_usd=0.04,
            diff_loc=19,
            prompt_pack_tokens=0,
            component_count=1,
        ),
    ]


def load_fixtures(path: Optional[Path | str] = None) -> Dict[str, Any]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURE_PATH
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_research_loop_v0(
    *,
    fixtures: Optional[Dict[str, Any]] = None,
    node_specs: Optional[Sequence[NodeSpec]] = None,
    created_at: str = DEFAULT_CREATED_AT,
    loop_balance_usd: float = DEFAULT_LOOP_BALANCE_USD,
    measure_latency: bool = False,
    validate: bool = True,
) -> LoopRun:
    """Run the lab-only v0 loop and return generated records.

    By default the emitted trajectory is byte-reproducible: ``latency_s`` is a
    deterministic fixture estimate. Real timing can be measured later during
    lab-only external-cost calibration by passing ``measure_latency=True``.
    """
    fixture_set = fixtures or load_fixtures()
    specs = list(node_specs or default_node_specs())
    fixture_rows = list(fixture_set["fixtures"])
    fixture_ids = [row["fixture_id"] for row in fixture_rows]
    engine_program_hash = _file_hash(DEFAULT_ENGINE_PROGRAM_PATH)
    champion_base = _file_hash(
        Path(__file__).resolve().parents[1]
        / "miner_models"
        / "qualification_research_loop_v0"
        / "qualify.py"
    )
    brief_hash = sha256_json(
        {
            "fixture_set_id": fixture_set["fixture_set_id"],
            "target": "improve frozen L1 fixture coverage without defect regression",
        }
    )
    trajectory_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ENGINE_VERSION}:{brief_hash}:trajectory"))
    brief_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ENGINE_VERSION}:{brief_hash}:brief"))

    baseline_fn = _resolve_callable(
        "miner_models.qualification_research_loop_v0.qualify",
        "qualify_reference",
    )
    baseline = _evaluate_callable(
        node_id="baseline-reference",
        qualify_fn=baseline_fn,
        fixture_rows=fixture_rows,
        parent_scores=None,
        complexity={"diff_loc": 0, "prompt_pack_tokens": 0, "component_count": 1},
        measure_latency=measure_latency,
    )

    events: List[Dict[str, Any]] = []
    ledger_rows: List[Dict[str, Any]] = []
    seq = 0

    seq = _append_event(
        events,
        seq,
        {
            "ts": _offset_ts(created_at, seq),
            "type": "PROBE",
            "cost_usd": 0.0,
            "probe_id": "baseline-reference-probe",
            "result_quantized": round(_mean(baseline.fixture_scores), 6),
            "fixtures_used": fixture_ids,
        },
    )
    seq = _append_event(
        events,
        seq,
        {
            "ts": _offset_ts(created_at, seq),
            "type": "LOOP_FUNDED",
            "cost_usd": 0.0,
            "loop_n": 1,
            "balance_before": round(float(loop_balance_usd), 6),
        },
    )

    evaluations: List[EvaluationResult] = []
    best_node_id = baseline.node_id
    best_score = _mean(baseline.fixture_scores)
    spent = 0.0

    for spec in specs:
        seq = _append_event(
            events,
            seq,
            {
                "ts": _offset_ts(created_at, seq),
                "type": "NODE_DRAFTED",
                "cost_usd": _round_cost(spec.draft_cost_usd),
                "node_id": spec.node_id,
                "parent_id": spec.parent_id,
                "operator": spec.operator,
                "component": spec.component,
                "patch_type": spec.patch_type,
                "hypothesis": spec.hypothesis,
                "patch_ref": spec.patch_ref,
                "model_used": spec.model_used,
                "tokens": {"in": spec.tokens_in, "out": spec.tokens_out},
            },
        )
        spent += spec.draft_cost_usd

        qualify_fn = _resolve_callable(spec.module, spec.function)
        result = _evaluate_callable(
            node_id=spec.node_id,
            qualify_fn=qualify_fn,
            fixture_rows=fixture_rows,
            parent_scores=baseline.fixture_scores,
            complexity={
                "diff_loc": spec.diff_loc,
                "prompt_pack_tokens": spec.prompt_pack_tokens,
                "component_count": spec.component_count,
            },
            measure_latency=measure_latency,
        )
        evaluations.append(result)
        spent += spec.eval_cost_usd

        seq = _append_event(
            events,
            seq,
            {
                "ts": _offset_ts(created_at, seq),
                "type": "NODE_EVALUATED",
                "cost_usd": _round_cost(spec.eval_cost_usd),
                "node_id": spec.node_id,
                "status": result.status,
                "rung": "L1",
                "metrics": result.metrics,
                "paired_lcb_vs_parent": result.paired_lcb_vs_parent,
                "fixtures": fixture_ids,
                "cache_hits": result.cache_hits,
                "execution_trace_ref": None,
            },
        )

        keep = result.status == "scored" and (result.paired_lcb_vs_parent or 0.0) > 0.0
        if keep and _mean(result.fixture_scores) > best_score:
            best_node_id = result.node_id
            best_score = _mean(result.fixture_scores)

        lesson = _reflect(spec, result, keep=keep)
        seq = _append_event(
            events,
            seq,
            {
                "ts": _offset_ts(created_at, seq),
                "type": "NODE_REFLECTED",
                "cost_usd": _round_cost(spec.reflect_cost_usd),
                "node_id": spec.node_id,
                "lesson": lesson,
                "lesson_embedding_ref": f"embedding:{spec.node_id}",
                "lesson_provenance": {
                    "champion_base": champion_base,
                    "component": spec.component,
                    "eval_version": fixture_set["fixture_set_id"],
                },
            },
        )
        spent += spec.reflect_cost_usd

        ledger_rows.append(
            _ledger_row(
                trajectory_id=trajectory_id,
                node_id=spec.node_id,
                commit=spec.patch_ref,
                island=DEFAULT_ISLAND,
                brief_id=brief_id,
                targeted_metric=spec.targeted_metric,
                delta_vs_parent=result.delta_vs_parent,
                cost_usd=spec.draft_cost_usd + spec.eval_cost_usd + spec.reflect_cost_usd,
                status="keep" if keep else "discard",
                description=lesson["why"],
                created_at=_offset_ts(created_at, seq),
            )
        )

    seq = _append_event(
        events,
        seq,
        {
            "ts": _offset_ts(created_at, seq),
            "type": "PLATEAU_STOP",
            "cost_usd": 0.0,
            "reason": "balance_exhausted",
            "best_node_id": best_node_id,
        },
    )

    trajectory = {
        "trajectory_id": trajectory_id,
        "schema_version": "1.0",
        "brief_id": brief_id,
        "island": DEFAULT_ISLAND,
        "funder_hotkey": None,
        "brief_sanitized_ref": brief_hash,
        "novelty_gate": {
            "result": "pass",
            "similarity": 0.0,
            "nearest_prior_receipt": None,
        },
        "engine_version": f"{ENGINE_VERSION}+{engine_program_hash.split(':', 1)[1][:12]}",
        "champion_base": champion_base,
        "created_at": created_at,
        "events": events,
        "final": {
            "settlement": {
                "loops_consumed": 1,
                "probation_charged": False,
                "balance_returned": max(0.0, round(float(loop_balance_usd) - spent, 6)),
                "crown": None,
                "grant_state": "none",
                "receipt_ref": sha256_json({"trajectory_id": trajectory_id, "events": events}),
            }
        },
    }

    if validate:
        assert_schema_record("research_trajectory.schema.json", trajectory)
        for row in ledger_rows:
            assert_schema_record("results_ledger_row.schema.json", row)

    return LoopRun(
        trajectory=trajectory,
        ledger_rows=ledger_rows,
        baseline=baseline,
        evaluations=evaluations,
        best_node_id=best_node_id,
    )


def _evaluate_callable(
    *,
    node_id: str,
    qualify_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    fixture_rows: Sequence[Dict[str, Any]],
    parent_scores: Optional[Sequence[float]],
    complexity: Dict[str, int],
    measure_latency: bool = False,
) -> EvaluationResult:
    fixture_scores: List[float] = []
    defect_counts: Dict[str, int] = {}
    total_leads = 0
    start = time.perf_counter() if measure_latency else None
    status = "scored"

    try:
        for fixture in fixture_rows:
            leads = qualify_fn(fixture["icp"])
            score, defects = _score_leads(fixture, leads)
            fixture_scores.append(score)
            total_leads += len(leads)
            for category, count in defects.items():
                defect_counts[category] = defect_counts.get(category, 0) + count
    except Exception:
        status = "crash"
        fixture_scores = [0.0 for _ in fixture_rows]

    if start is not None:
        elapsed = max(0.0, time.perf_counter() - start)
    else:
        elapsed = _deterministic_latency_s(fixture_count=len(fixture_rows), lead_count=total_leads)
    defect_rates = {
        category: round(count / max(total_leads, 1), 6)
        for category, count in sorted(defect_counts.items())
    }
    total_defect_rate = sum(defect_counts.values()) / max(total_leads, 1)
    coverage = round(_mean([score / 100.0 for score in fixture_scores]), 6)
    schema_validity = status == "scored"
    if total_defect_rate > 0.25:
        status = "guardrail_breach"
        schema_validity = True

    paired_lcb: Optional[float] = None
    delta_vs_parent: Optional[float] = None
    if parent_scores is not None:
        deltas = [
            score - parent
            for score, parent in zip(fixture_scores, parent_scores)
        ]
        delta_vs_parent = round(_mean(deltas), 6)
        paired_lcb = round(dev_eval_lower_confidence_bound(deltas)["lower_confidence_bound"], 6)

    metrics = {
        "proxy_score": round(_mean(fixture_scores), 6),
        "evidence_defect_rate_by_category": defect_rates,
        "coverage": coverage,
        "cost_per_icp": 0.08,
        "latency_s": round(elapsed, 6),
        "schema_validity": schema_validity,
        "complexity": complexity,
    }

    return EvaluationResult(
        node_id=node_id,
        status=status,
        fixture_scores=[round(score, 6) for score in fixture_scores],
        metrics=metrics,
        paired_lcb_vs_parent=paired_lcb,
        delta_vs_parent=delta_vs_parent,
        latency_s=round(elapsed, 6),
        cache_hits={"snapshot": len(fixture_rows), "verdict": len(fixture_rows)},
    )


def dev_eval_lower_confidence_bound(deltas: Iterable[float]) -> Dict[str, float]:
    """Plain paired-t LCB for L1 dev-eval, distinct from probation crowning.

    Dev-eval uses ``mean(d) - t(0.95, n-1) * sd(d) / sqrt(n)`` over any
    fixture count ``n >= 2``. It intentionally does not use the P0.5
    group-sequential probation boundaries, which only apply to same-day
    crowning looks.
    """
    values = [float(delta) for delta in deltas]
    if len(values) < 2:
        raise ValueError("dev-eval paired LCB requires at least two deltas")
    mean_delta = _mean(values)
    variance = sum((value - mean_delta) ** 2 for value in values) / (len(values) - 1)
    sd_delta = sqrt(variance)
    se_delta = sd_delta / sqrt(len(values))
    boundary = _dev_eval_t_critical(len(values) - 1)
    return {
        "n": len(values),
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "se_delta": se_delta,
        "boundary": boundary,
        "lower_confidence_bound": mean_delta - boundary * se_delta,
    }


def _score_leads(fixture: Dict[str, Any], leads: Sequence[Dict[str, Any]]) -> tuple[float, Dict[str, int]]:
    answer_key = {_norm(name) for name in fixture["answer_key"]}
    matched = set()
    defects: Dict[str, int] = {}
    score = 0.0

    for rank, lead in enumerate(leads[:5], start=1):
        company = _norm(str(lead.get("company_name") or ""))
        fixture_candidate = _candidate_for_company(fixture, company)
        if company in answer_key:
            matched.add(company)
            score += max(0.0, 38.0 - (rank - 1) * 3.0)
        else:
            defects["wrong_company"] = defects.get("wrong_company", 0) + 1
            score -= 12.0
        if fixture_candidate:
            for defect in fixture_candidate.get("defects", []):
                defects[defect] = defects.get(defect, 0) + 1

    coverage_bonus = 24.0 * (len(matched) / max(len(answer_key), 1))
    score += coverage_bonus
    score -= 8.0 * sum(defects.values())
    return max(0.0, min(100.0, score)), defects


def _reflect(spec: NodeSpec, result: EvaluationResult, *, keep: bool) -> Dict[str, str]:
    if keep:
        return {
            "worked": f"{spec.node_id} improved {spec.targeted_metric} on frozen L1 fixtures.",
            "failed": "No production L2 or probation evidence was collected in v0.",
            "why": (
                f"{spec.patch_ref} kept: paired LCB {result.paired_lcb_vs_parent:.3f} "
                "cleared the parent on cached fixtures without guardrail breach."
            ),
            "next_question": "Re-run against broader snapshot fixtures and cluster winning diffs.",
        }
    return {
        "worked": "The node exercised the v0 guardrail path.",
        "failed": f"{spec.node_id} did not clear the paired improvement bar.",
        "why": (
            f"{spec.patch_ref} discarded: status={result.status}, "
            f"paired LCB={result.paired_lcb_vs_parent}."
        ),
        "next_question": "Constrain source routing before increasing recall.",
    }


def _ledger_row(
    *,
    trajectory_id: str,
    node_id: str,
    commit: str,
    island: str,
    brief_id: str,
    targeted_metric: str,
    delta_vs_parent: Optional[float],
    cost_usd: float,
    status: str,
    description: str,
    created_at: str,
) -> Dict[str, Any]:
    row_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{trajectory_id}:{node_id}:ledger"))
    return {
        "ledger_row_id": row_id,
        "schema_version": "1.0",
        "trajectory_id": trajectory_id,
        "node_id": node_id,
        "commit": commit,
        "island": island,
        "brief_id": brief_id,
        "targeted_metric": targeted_metric,
        "delta_vs_parent": delta_vs_parent,
        "cost_usd": _round_cost(cost_usd),
        "status": status,
        "description": description[:2048],
        "created_at": created_at,
    }


def _append_event(events: List[Dict[str, Any]], seq: int, payload: Dict[str, Any]) -> int:
    event = {"seq": seq, **payload}
    event["cost_usd"] = _round_cost(event.get("cost_usd", 0.0))
    event["anchored_hash"] = sha256_json(event)
    events.append(event)
    return seq + 1


def _resolve_callable(module: str, function: str) -> Callable[[Dict[str, Any]], List[Dict[str, Any]]]:
    imported = importlib.import_module(module)
    fn = getattr(imported, function)
    if not callable(fn):
        raise TypeError(f"{module}.{function} is not callable")
    return fn


def _candidate_for_company(fixture: Dict[str, Any], normalized_company: str) -> Optional[Dict[str, Any]]:
    for candidate in fixture["candidate_pool"]:
        if _norm(candidate["company_name"]) == normalized_company:
            return candidate
    return None


def _offset_ts(created_at: str, seq: int) -> str:
    from datetime import datetime, timedelta, timezone

    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt + timedelta(seconds=seq * 30)).astimezone(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def _file_hash(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def _round_cost(value: float) -> float:
    return round(float(value), 6)


def _deterministic_latency_s(*, fixture_count: int, lead_count: int) -> float:
    return round(0.05 * int(fixture_count) + 0.01 * int(lead_count), 6)


def _dev_eval_t_critical(df: int) -> float:
    if df < 1:
        raise ValueError("df must be positive")
    if df in DEV_EVAL_T_ONE_SIDED_95_BY_DF:
        return DEV_EVAL_T_ONE_SIDED_95_BY_DF[df]
    return 1.645


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _norm(value: str) -> str:
    return " ".join(value.lower().strip().split())
