from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import pytest

from gateway.research_lab import global_icp_queue
from gateway.research_lab import scoring_worker
from gateway.research_lab import store
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee import scoring_executor_v2
from gateway.tee.scoring_executor_v2 import ScoringExecutorV2
from leadpoet_canonical.attested_v2 import sha256_json
from gateway.research_lab.icp_window import (
    RollingIcpWindowUnavailable,
    select_rolling_icp_window_from_sets,
)
from gateway.research_lab.public_benchmarks import build_public_benchmark_report
from research_lab.eval import evaluator
from research_lab.eval.artifacts import PrivateModelArtifactManifest
from research_lab.eval.baseline_summary import build_baseline_score_summary
from research_lab.eval.conditional_validation import (
    ConditionalValidationPolicy,
    build_conditional_category_assignment,
    normalized_icp_intent_signature,
)
from research_lab.eval.private_runtime import PrivateModelRuntimeError
from research_lab.eval.promotion_metric import (
    preliminary_promotion_gate_projection,
    promotion_gate_decision,
    promotion_improvement_metric,
)


WINDOW_HASH = "sha256:" + "a" * 64
MODEL_HASH = "sha256:" + "b" * 64
PROVIDER_RETRYABLE = (
    "docker private model provider-backed sourcing failed before returning companies: "
    "HTTPError: too many requests; status=429; url=https://api.example.test/search"
)


def _bank(count: int = 40) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index in range(count):
        ref = f"icp-{index:02d}"
        digest = "sha256:" + f"{index + 1:064x}"
        items.append(
            {
                "icp_ref": ref,
                "icp_hash": digest,
                "intent_signal_signature": f"intent-{index:02d}",
                "set_id": 20260714 if index < 20 else 20260713,
                "day_index": 0 if index < 20 else 1,
                "day_rank": (index % 20) + 1,
                "icp": {
                    "name": ref,
                    "industry": f"industry-{index}",
                    "intent_signals": [f"intent-{index:02d}"],
                },
            }
        )
        summaries.append(
            {
                "icp_ref": ref,
                "icp_hash": digest,
                "score": float(index),
                "company_count": 5,
                "industry": f"industry-{index}",
                "diagnostics": {},
            }
        )
    return items, summaries


def _policy(**overrides: Any) -> ConditionalValidationPolicy:
    values = {
        "mode": "enforce",
        "public_total_icps": 10,
        "public_weak_total": 7,
        "private_total_icps": 10,
        "private_weak_total": 3,
        "conditional_total_icps": 20,
        "fresh_icp_count": 20,
        "threshold_points": 1.0,
    }
    values.update(overrides)
    return ConditionalValidationPolicy(**values)


def _assignment() -> dict[str, Any]:
    items, summaries = _bank()
    return build_conditional_category_assignment(
        rolling_window_hash=WINDOW_HASH,
        benchmark_items=items,
        per_icp_summaries=summaries,
        policy=_policy(),
        baseline_serving_model_version_hash=MODEL_HASH,
    )


def _benchmark_row(
    *,
    bundle_id: str,
    assignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    score_summary_doc: dict[str, Any] = {
        "aggregate_score": 42.0,
        "per_icp_summaries": [{"company_count": 1, "score": 42.0}],
    }
    if assignment is not None:
        score_summary_doc["category_assignment"] = dict(assignment)
    return {
        "benchmark_bundle_id": bundle_id,
        "private_model_manifest_hash": MODEL_HASH,
        "rolling_window_hash": WINDOW_HASH,
        "benchmark_quality": "passed",
        "evaluation_epoch": 24000,
        "score_summary_doc": score_summary_doc,
        "current_benchmark_status": "completed",
    }


def test_default_policy_derives_40_with_20_retained() -> None:
    policy = _policy()
    assert policy.total_icps == 40
    assert policy.fresh_icp_count == 20
    assert policy.retained_icp_count == 20
    assert policy.to_dict()["threshold_points"] == 1.0


@pytest.mark.asyncio
async def test_policy_cutover_replaces_only_incompatible_same_day_baseline(
    monkeypatch,
) -> None:
    expected_policy_hash = str(_policy().to_dict()["policy_hash"])
    legacy = _benchmark_row(bundle_id="legacy")
    matching = _benchmark_row(bundle_id="matching", assignment=_assignment())
    rows = [legacy]

    async def select_many(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return rows

    monkeypatch.setattr(scoring_worker, "select_many", select_many)
    worker = object.__new__(scoring_worker.ResearchLabGatewayScoringWorker)

    assert scoring_worker._private_benchmark_matches_policy(
        legacy,
        expected_policy_hash=expected_policy_hash,
    ) is False
    assert scoring_worker._private_benchmark_matches_policy(
        legacy,
        expected_policy_hash="",
    ) is True
    assert scoring_worker._private_benchmark_matches_policy(
        matching,
        expected_policy_hash="",
    ) is False
    assert (
        await worker._reusable_same_day_benchmark(
            today="2026-07-19",
            manifest_hash=MODEL_HASH,
            expected_policy_hash=expected_policy_hash,
        )
        is None
    )
    assert (
        await worker._same_day_reference_recorded_while_running(
            today="2026-07-19",
            window_hash=WINDOW_HASH,
            manifest_hash=MODEL_HASH,
            expected_policy_hash=expected_policy_hash,
        )
        is None
    )

    rows[:] = [legacy, matching]
    assert (
        await worker._reusable_same_day_benchmark(
            today="2026-07-19",
            manifest_hash=MODEL_HASH,
            expected_policy_hash=expected_policy_hash,
        )
    )["benchmark_bundle_id"] == "matching"
    assert (
        await worker._same_day_reference_recorded_while_running(
            today="2026-07-19",
            window_hash=WINDOW_HASH,
            manifest_hash=MODEL_HASH,
            expected_policy_hash=expected_policy_hash,
        )
    )["benchmark_bundle_id"] == "matching"


@pytest.mark.asyncio
async def test_enforced_candidate_gate_rejects_legacy_same_day_baseline(
    monkeypatch,
) -> None:
    async def select_many(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [_benchmark_row(bundle_id="legacy")]

    monkeypatch.setattr(scoring_worker, "select_many", select_many)
    worker = object.__new__(scoring_worker.ResearchLabGatewayScoringWorker)
    worker.config = types.SimpleNamespace(conditional_validation_policy=_policy)

    with pytest.raises(
        scoring_worker.CandidateBaselineNotReady,
        match="same_day_completed_private_baseline_required",
    ):
        await worker._daily_candidate_scoring_window_and_gate(
            artifact=types.SimpleNamespace(manifest_hash=MODEL_HASH),
            now=datetime(2026, 7, 19, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"public_weak_total": 6},
        {"threshold_points": float("nan")},
        {"threshold_points": float("inf")},
        {"fresh_icp_count": 40},
    ),
)
def test_policy_rejects_noncentered_nonfinite_or_no_retained(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _policy(**overrides)


def test_assignment_uses_exact_center_and_hash_rotated_tails() -> None:
    assignment = _assignment()
    by_category = {
        name: [row for row in assignment["items"] if row["category"] == name]
        for name in ("public", "private", "conditional")
    }
    assert assignment["category_counts"] == {
        "public": 10,
        "private": 10,
        "conditional": 20,
    }
    assert {row["icp_ref"] for row in by_category["conditional"]} == {
        f"icp-{index:02d}" for index in range(10, 30)
    }
    assert sum(row["strength_label"] == "weak" for row in by_category["public"]) == 7
    assert sum(row["strength_label"] == "strong" for row in by_category["public"]) == 3
    assert sum(row["strength_label"] == "weak" for row in by_category["private"]) == 3
    assert sum(row["strength_label"] == "strong" for row in by_category["private"]) == 7


def test_assignment_ties_are_deterministic_and_duplicates_fail_closed() -> None:
    items, summaries = _bank()
    tied = [{**summary, "score": 50.0} for summary in summaries]
    first = build_conditional_category_assignment(
        rolling_window_hash=WINDOW_HASH,
        benchmark_items=items,
        per_icp_summaries=tied,
        policy=_policy(),
        baseline_serving_model_version_hash=MODEL_HASH,
    )
    second = build_conditional_category_assignment(
        rolling_window_hash=WINDOW_HASH,
        benchmark_items=items,
        per_icp_summaries=tied,
        policy=_policy(),
        baseline_serving_model_version_hash=MODEL_HASH,
    )
    assert first == second

    duplicated = [dict(item) for item in items]
    duplicated[1]["intent_signal_signature"] = duplicated[0]["intent_signal_signature"]
    with pytest.raises(ValueError, match="duplicate normalized intent signature"):
        build_conditional_category_assignment(
            rolling_window_hash=WINDOW_HASH,
            benchmark_items=duplicated,
            per_icp_summaries=summaries,
            policy=_policy(),
            baseline_serving_model_version_hash=MODEL_HASH,
        )


def _set_row(set_id: int, signals: list[str], hash_char: str) -> dict[str, Any]:
    return {
        "set_id": set_id,
        "icp_set_hash": "sha256:" + hash_char * 64,
        "is_active": True,
        "icps": [
            {
                "icp_id": f"{set_id}-{index}",
                "industry": f"industry-{signal}",
                "sub_industry": f"sub-{signal}",
                "intent_signals": [signal],
            }
            for index, signal in enumerate(signals)
        ],
    }


def test_strict_window_selects_20_fresh_and_backfills_20_unique_retained() -> None:
    fresh_signals = [f"fresh-{index}" for index in range(20)]
    rows = [
        _set_row(20260714, fresh_signals, "1"),
        _set_row(
            20260713,
            [*fresh_signals[:10], *[f"retained-a-{index}" for index in range(10)]],
            "2",
        ),
        _set_row(20260712, [f"retained-b-{index}" for index in range(20)], "3"),
    ]
    window = select_rolling_icp_window_from_sets(
        rows,
        days=10,
        icps_per_day=2,
        fresh_icp_count=20,
        retained_icp_count=20,
        min_new_icp_count=20,
        required_total_icps=40,
        require_unique_icps=True,
        required_fresh_set_id=20260714,
    )
    assert len(window.benchmark_items) == 40
    assert sum(item["cohort"] == "fresh" for item in window.benchmark_items) == 20
    assert sum(item["cohort"] == "retained" for item in window.benchmark_items) == 20
    assert len({item["icp_hash"] for item in window.benchmark_items}) == 40
    assert len({item["intent_signal_signature"] for item in window.benchmark_items}) == 40
    assert 20260712 in window.set_ids
    assert window.public_doc["required_icp_count"] == 40


def test_strict_window_fails_when_unique_retained_history_is_incomplete() -> None:
    fresh_signals = [f"fresh-{index}" for index in range(20)]
    rows = [
        _set_row(20260714, fresh_signals, "1"),
        _set_row(20260713, fresh_signals, "2"),
    ]
    with pytest.raises(RollingIcpWindowUnavailable, match="retained_icps_found_0"):
        select_rolling_icp_window_from_sets(
            rows,
            days=10,
            icps_per_day=2,
            fresh_icp_count=20,
            retained_icp_count=20,
            min_new_icp_count=20,
            required_total_icps=40,
            require_unique_icps=True,
            required_fresh_set_id=20260714,
        )


def test_strict_window_allows_repeated_signals_across_distinct_markets() -> None:
    fresh = _set_row(20260714, [f"fresh-{index}" for index in range(20)], "1")
    fresh["icps"][1].pop("intent_signals")
    fresh["icps"][1]["intent_signal"] = "  FRESH-0  "
    rows = [fresh, _set_row(20260713, [f"retained-{index}" for index in range(20)], "2")]

    window = select_rolling_icp_window_from_sets(
        rows,
        days=10,
        icps_per_day=2,
        fresh_icp_count=20,
        retained_icp_count=20,
        min_new_icp_count=20,
        required_total_icps=40,
        require_unique_icps=True,
        required_fresh_set_id=20260714,
    )
    assert len(window.benchmark_items) == 40
    assert len({item["intent_signal_signature"] for item in window.benchmark_items}) == 40


def test_strict_window_rejects_same_normalized_market_intent() -> None:
    fresh = _set_row(20260714, [f"fresh-{index}" for index in range(20)], "1")
    duplicate = fresh["icps"][1]
    duplicate.update(
        {
            "industry": fresh["icps"][0]["industry"].upper(),
            "sub_industry": "  " + fresh["icps"][0]["sub_industry"].upper() + "  ",
            "intent_signals": ["  FRESH-0  "],
        }
    )
    rows = [fresh, _set_row(20260713, [f"retained-{index}" for index in range(20)], "2")]

    with pytest.raises(RollingIcpWindowUnavailable, match="fresh_icps_found_19"):
        select_rolling_icp_window_from_sets(
            rows,
            days=10,
            icps_per_day=2,
            fresh_icp_count=20,
            retained_icp_count=20,
            min_new_icp_count=20,
            required_total_icps=40,
            require_unique_icps=True,
            required_fresh_set_id=20260714,
        )
    assert normalized_icp_intent_signature(duplicate) == normalized_icp_intent_signature(
        fresh["icps"][0]
    )


def test_enforced_baseline_stamps_immutable_v11_assignment() -> None:
    items, summaries = _bank()
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "1" * 64,
        git_commit_sha="2" * 40,
        image_digest="repo@sha256:" + "3" * 64,
        config_hash="sha256:" + "4" * 64,
        component_registry_version="v1",
        scoring_adapter_version="v1",
        manifest_uri="s3://private/model.json",
        manifest_hash="sha256:" + "5" * 64,
        signature_ref="kms://signature",
        build_id="build-1",
    )
    result = build_baseline_score_summary(
        artifact_manifest=artifact.to_dict(),
        benchmark_date="2026-07-14",
        benchmark_attempt=1,
        rolling_window_hash=WINDOW_HASH,
        evaluation_epoch=42,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
        public_total_icps=10,
        public_weak_total=7,
        retried=0,
        recovered=0,
        max_unresolved_icps=0,
        day_jump_points=None,
        elapsed_seconds=12.0,
        conditional_validation_policy=_policy().to_dict(),
    )
    doc = result["score_summary_doc"]
    assert doc["schema_version"] == "1.1"
    assert doc["aggregate_score"] == pytest.approx(19.5)
    assert doc["category_assignment"]["assignment_hash"].startswith("sha256:")
    assert doc["category_assignment"]["policy_hash"] == _policy().to_dict()["policy_hash"]
    assert doc["category_scores"] == doc["category_assignment"]["category_scores"]


def test_conditional_mode_off_is_byte_equivalent_to_legacy_baseline() -> None:
    items, summaries = _bank(20)
    kwargs = {
        "artifact_manifest": {
            "model_artifact_hash": "sha256:" + "1" * 64,
            "git_commit_sha": "2" * 40,
            "image_digest": "repo@sha256:" + "3" * 64,
            "config_hash": "sha256:" + "4" * 64,
            "component_registry_version": "v1",
            "scoring_adapter_version": "v1",
            "manifest_uri": "s3://private/model.json",
            "manifest_hash": "sha256:" + "5" * 64,
            "signature_ref": "kms://signature",
            "build_id": "build-1",
        },
        "benchmark_date": "2026-07-14",
        "benchmark_attempt": 1,
        "rolling_window_hash": WINDOW_HASH,
        "evaluation_epoch": 42,
        "benchmark_items": items,
        "per_icp_summaries": summaries,
        "public_icps_per_day": 3,
        "public_weak_per_day": 2,
        "public_total_icps": 10,
        "public_weak_total": 7,
        "retried": 0,
        "recovered": 0,
        "max_unresolved_icps": 0,
        "day_jump_points": None,
        "elapsed_seconds": 12.0,
    }
    legacy = build_baseline_score_summary(**kwargs)
    disabled = build_baseline_score_summary(
        **kwargs,
        conditional_validation_policy=_policy(mode="off").to_dict(),
    )
    assert disabled == legacy
    assert disabled["score_summary_doc"]["schema_version"] == "1.0"


def test_public_report_exposes_overall_and_public_only() -> None:
    items, summaries = _bank()
    assignment = _assignment()
    report = build_public_benchmark_report(
        benchmark_date="2026-07-14",
        rolling_window_hash=WINDOW_HASH,
        aggregate_score=assignment["aggregate_score"],
        benchmark_items=items,
        per_icp_summaries=summaries,
        category_assignment=assignment,
    )
    assert report["schema_version"] == "1.3"
    assert report["aggregate_score"] == assignment["aggregate_score"]
    assert report["public_score"] == assignment["category_scores"]["public"]
    assert report["conditional_holdout_icp_count"] == 20
    encoded = json.dumps(report, sort_keys=True)
    hidden_refs = {
        row["icp_ref"]
        for row in assignment["items"]
        if row["category"] != "public"
    }
    assert all(ref not in encoded for ref in hidden_refs)
    assert "private_score" not in report
    assert "conditional_score" not in report


class _Runner:
    def __init__(
        self,
        *,
        default_score: float,
        scores: Mapping[str, float] | None = None,
        failures: Mapping[str, list[Exception]] | None = None,
    ) -> None:
        self.default_score = default_score
        self.scores = dict(scores or {})
        self.failures = {key: list(value) for key, value in (failures or {}).items()}
        self.calls: list[str] = []

    async def __call__(self, icp: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, float]]:
        ref = str(icp["name"])
        self.calls.append(ref)
        scripted = self.failures.get(ref)
        if scripted:
            raise scripted.pop(0)
        score = self.scores.get(ref, self.default_score)
        return [] if score < 0 else [{"score": score}]


async def _scorer(
    companies: list[Mapping[str, Any]],
    icp: Mapping[str, Any],
    is_reference_model: bool,
) -> list[float]:
    return [float(company["score"]) for company in companies]


def _gate(
    *,
    baseline_public: float = 40.0,
    baseline_preliminary: float = 40.0,
    baseline_total: float = 40.0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "public_icp_refs": [f"icp-{index:02d}" for index in range(10)],
        "private_icp_refs": [f"icp-{index:02d}" for index in range(10, 20)],
        "conditional_icp_refs": [f"icp-{index:02d}" for index in range(20, 40)],
        "baseline_public_score": baseline_public,
        "baseline_private_score": 40.0,
        "baseline_conditional_score": 40.0,
        "baseline_preliminary_score": baseline_preliminary,
        "baseline_aggregate_score": baseline_total,
        "threshold_points": 1.0,
        "baseline_benchmark_bundle_id": "private_benchmark:" + "1" * 64,
        "baseline_benchmark_hash": "sha256:" + "2" * 64,
        "category_assignment_hash": "sha256:" + "3" * 64,
        "conditional_validation_policy_hash": "sha256:" + "4" * 64,
        "conditional_validation_required": True,
    }


def test_holdout_result_propagates_paired_promotion_metric_version() -> None:
    gate = {
        **_gate(),
        "promotion_metric_version": "paired_lcb_v1",
    }
    public_results = [
        {
            "icp_ref": f"icp-{index:02d}",
            "candidate_company_scores": [50.0],
        }
        for index in range(10)
    ]

    _results, gate_result = evaluator.build_holdout_gate_result(
        public_results=public_results,
        private_results=(),
        conditional_results=(),
        public_icp_count=10,
        private_icp_count=10,
        conditional_icp_count=20,
        gate=gate,
    )

    assert gate_result["promotion_metric_version"] == "paired_lcb_v1"


async def _run_gate(
    runner: _Runner,
    gate: Mapping[str, Any],
    *,
    transition=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if transition is None:
        async def transition(
            _action: str,
            _payload: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            return {
                "preliminary_promotion_gate": {
                    "proof_hash": "sha256:" + "9" * 64,
                }
            }
    items, _summaries = _bank()
    return await evaluator._score_with_private_holdout_gate(
        benchmark_items=items,
        base_runner=None,
        candidate_runner=runner,
        scorer=_scorer,
        run_context={"run_id": "run-1"},
        image_candidate=True,
        runtime_patch=None,
        gate=gate,
        holdout_transition_hook=transition,
    )


@pytest.mark.asyncio
async def test_public_rejection_executes_exactly_10_icps() -> None:
    runner = _Runner(default_score=50.0)
    results, gate = await _run_gate(runner, _gate(baseline_public=60.0))
    assert len(runner.calls) == 10
    assert len(results) == 10
    assert gate["decision"] == "rejected_before_private_holdout"
    assert gate["private_holdout_evaluated"] is False
    assert gate["conditional_holdout_evaluated"] is False


@pytest.mark.asyncio
async def test_preliminary_rejection_executes_exactly_20_icps() -> None:
    runner = _Runner(default_score=50.0)
    results, gate = await _run_gate(runner, _gate(baseline_preliminary=50.0))
    assert len(runner.calls) == 20
    assert len(results) == 20
    assert gate["decision"] == "rejected_before_conditional_validation"
    assert gate["conditional_holdout_evaluated"] is False


@pytest.mark.asyncio
async def test_preliminary_pass_executes_40_and_persists_boundary_first() -> None:
    runner = _Runner(default_score=50.0)
    transitions: list[tuple[str, int]] = []

    async def transition(action: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        transitions.append((action, len(runner.calls)))
        assert payload["decision"] == "conditional_validation_required"
        return {
            "preliminary_promotion_gate": {
                "proof_hash": "sha256:" + "9" * 64,
            }
        }

    results, gate = await _run_gate(runner, _gate(), transition=transition)
    assert transitions == [("conditional_validation_started", 20)]
    assert len(runner.calls) == 40
    assert len(results) == 40
    assert gate["decision"] == "conditional_validation_approved"
    assert gate["candidate_delta_vs_daily_baseline"] == 10.0
    assert gate["conditional_holdout_evaluated"] is True


@pytest.mark.asyncio
async def test_missing_preliminary_authority_stops_before_conditional_icps() -> None:
    runner = _Runner(default_score=50.0)
    items, _summaries = _bank()
    with pytest.raises(
        evaluator.ConditionalValidationRetryableError,
        match="conditional_preliminary_authority_unavailable",
    ):
        await evaluator._score_with_private_holdout_gate(
            benchmark_items=items,
            base_runner=None,
            candidate_runner=runner,
            scorer=_scorer,
            run_context={"run_id": "run-1"},
            image_candidate=True,
            runtime_patch=None,
            gate=_gate(),
            holdout_transition_hook=None,
        )
    assert len(runner.calls) == 20


@pytest.mark.asyncio
async def test_preliminary_authority_proof_is_returned_at_icp_20_and_frozen() -> None:
    runner = _Runner(default_score=50.0)
    proof = {
        "schema_version": "research_lab_preliminary_promotion_gate.v1",
        "proof_hash": "sha256:" + "f" * 64,
    }
    calls: list[int] = []

    async def transition(
        action: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert action == "conditional_validation_started"
        assert len(payload["_preliminary_results"]) == 20
        calls.append(len(runner.calls))
        return {"preliminary_promotion_gate": proof}

    results, gate = await _run_gate(runner, _gate(), transition=transition)
    assert calls == [20]
    assert len(results) == 40
    assert gate["preliminary_promotion_gate"] == proof


def test_preliminary_projection_uses_the_unchanged_promotion_gate() -> None:
    preliminary = {
        **_gate(),
        "decision": "conditional_validation_required",
        "private_holdout_evaluated": True,
        "candidate_preliminary_score": 42.0,
        "candidate_preliminary_delta": 2.0,
    }
    projected = preliminary_promotion_gate_projection(preliminary)
    bundle = {"private_holdout_gate": projected, "aggregates": {}}
    metric = promotion_improvement_metric(bundle)
    decision = promotion_gate_decision(
        bundle,
        candidate_kind="image_build",
        candidate_parent=MODEL_HASH,
        active_parent=MODEL_HASH,
        threshold_points=1.0,
        auto_promotion_enabled=True,
    )
    assert projected["decision"] == "private_holdout_approved"
    assert projected["conditional_validation_required"] is False
    assert metric.improvement_points == 2.0
    assert decision.status == "promotion_passed"


def test_preliminary_policy_keeps_the_original_20_icp_success_floor(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LAB_MIN_SUCCESSFUL_ICPS", raising=False)
    worker = object.__new__(scoring_worker.ResearchLabGatewayScoringWorker)
    worker.config = types.SimpleNamespace(
        improvement_threshold_points=1.0,
        lab_champion_eval_days=10,
        lab_champion_icps_per_day=2,
        conditional_validation_policy=lambda: _policy(),
    )
    assert worker._evaluation_policy()["min_successful_icps"] == 40
    assert worker._preliminary_evaluation_policy()["min_successful_icps"] == 20


@pytest.mark.asyncio
async def test_preliminary_authority_runs_bundle_metric_then_unchanged_gate(
    monkeypatch,
) -> None:
    order: list[str] = []
    parent_hash = "sha256:" + "7" * 64
    candidate_hash = "sha256:" + "6" * 64
    preliminary_gate = {
        **_gate(),
        "decision": "conditional_validation_required",
        "private_holdout_evaluated": True,
        "candidate_preliminary_score": 42.0,
        "candidate_preliminary_delta": 2.0,
        "public_icp_count": 10,
        "private_holdout_icp_count": 10,
    }
    projected = preliminary_promotion_gate_projection(preliminary_gate)
    provisional_bundle = {
        "score_bundle_hash": "sha256:" + "1" * 64,
        "private_holdout_gate": projected,
        "aggregates": {},
    }

    async def compare_bundle(**_kwargs: Any) -> Mapping[str, Any]:
        order.append("score_bundle")
        return {
            "execution_receipt": {
                "receipt_hash": "sha256:" + "2" * 64,
                "output_root": "sha256:" + "a" * 64,
            }
        }

    async def compare_metric(**_kwargs: Any) -> Mapping[str, Any]:
        order.append("promotion_metric")
        return {
            "execution_receipt": {
                "receipt_hash": "sha256:" + "3" * 64,
                "output_root": "sha256:" + "b" * 64,
            }
        }

    async def compare_decision(**kwargs: Any) -> Mapping[str, Any]:
        order.append("promotion_decision")
        decision = dict(kwargs["expected_decision"])
        return {
            "execution_receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json({"decision": decision}),
            }
        }

    async def active(*_args: Any, **_kwargs: Any) -> Any:
        return types.SimpleNamespace(
            artifact=types.SimpleNamespace(model_artifact_hash=parent_hash)
        )

    monkeypatch.setattr(
        scoring_worker,
        "build_score_bundle_from_scored_icps",
        lambda **_kwargs: dict(provisional_bundle),
    )
    monkeypatch.setattr(
        scoring_worker,
        "_compare_candidate_score_bundle_in_enclave",
        compare_bundle,
    )
    monkeypatch.setattr(
        scoring_worker,
        "compare_attested_promotion_metric",
        compare_metric,
    )
    monkeypatch.setattr(
        scoring_worker,
        "compare_attested_promotion_gate_decision",
        compare_decision,
    )
    monkeypatch.setattr(scoring_worker, "load_active_private_model", active)
    monkeypatch.setattr(
        scoring_worker,
        "scoring_configuration_hash",
        lambda: "sha256:" + "8" * 64,
    )

    proof = await scoring_worker._authorize_conditional_preliminary_gate(
        config=types.SimpleNamespace(
            improvement_threshold_points=1.0,
            auto_promotion_enabled=True,
        ),
        evaluation_epoch=42,
        candidate={
            "candidate_kind": "image_build",
            "parent_artifact_hash": parent_hash,
        },
        artifact=types.SimpleNamespace(model_artifact_hash=parent_hash),
        benchmark=object(),
        patch={},
        candidate_artifact=types.SimpleNamespace(
            model_artifact_hash=candidate_hash,
            to_dict=lambda: {},
        ),
        preliminary_results=[{"icp_ref": f"icp-{index:02d}"} for index in range(20)],
        run_context={"rolling_window_hash": WINDOW_HASH},
        policy={"min_successful_icps": 20},
        preliminary_gate=preliminary_gate,
        parent_receipts=[],
    )
    assert order == ["score_bundle", "promotion_metric", "promotion_decision"]
    assert proof["status"] == "promotion_passed"
    assert proof["candidate_parent_artifact_hash"] == parent_hash
    assert proof["proof_hash"] == sha256_json(
        {key: value for key, value in proof.items() if key != "proof_hash"}
    )


@pytest.mark.asyncio
async def test_transition_write_failure_prevents_conditional_provider_calls() -> None:
    runner = _Runner(default_score=50.0)

    async def fail_transition(action: str, payload: Mapping[str, Any]) -> None:
        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        await _run_gate(runner, _gate(), transition=fail_transition)
    assert len(runner.calls) == 20


@pytest.mark.asyncio
async def test_conditional_provider_failure_is_retryable_not_a_zero_score(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "_PROVIDER_429_RETRY_BACKOFF_SECONDS", 0.0)
    failures = {
        "icp-20": [
            PrivateModelRuntimeError(PROVIDER_RETRYABLE),
            PrivateModelRuntimeError(PROVIDER_RETRYABLE),
            PrivateModelRuntimeError(PROVIDER_RETRYABLE),
        ]
    }
    runner = _Runner(default_score=50.0, failures=failures)
    with pytest.raises(
        evaluator.ConditionalValidationRetryableError,
        match="conditional_validation_incomplete",
    ):
        await _run_gate(runner, _gate())
    assert runner.calls[:20] == [f"icp-{index:02d}" for index in range(20)]


@pytest.mark.asyncio
async def test_legitimate_conditional_zero_is_a_final_rejection_not_retryable() -> None:
    conditional_zero = {f"icp-{index:02d}": -1.0 for index in range(20, 40)}
    runner = _Runner(default_score=50.0, scores=conditional_zero)
    results, gate = await _run_gate(runner, _gate())
    assert len(results) == 40
    assert gate["decision"] == "rejected_after_conditional_validation"
    assert gate["candidate_conditional_score"] == 0.0


def test_promotion_metric_requires_completed_final_gate_and_keeps_legacy_path() -> None:
    final_gate = {
        **_gate(),
        "decision": "conditional_validation_approved",
        "conditional_holdout_evaluated": True,
        "candidate_total_score": 44.0,
        "candidate_delta_vs_daily_baseline": 4.0,
    }
    final_metric = promotion_improvement_metric(
        {"private_holdout_gate": final_gate, "aggregates": {}}
    )
    assert final_metric.rejection_status is None
    assert final_metric.improvement_points == 4.0

    preliminary_metric = promotion_improvement_metric(
        {
            "private_holdout_gate": {
                **_gate(),
                "decision": "conditional_validation_required",
                "conditional_holdout_evaluated": False,
            },
            "aggregates": {},
        }
    )
    assert preliminary_metric.rejection_status == "conditional_validation_incomplete"

    legacy_metric = promotion_improvement_metric({"aggregates": {"mean_delta": 2.5}})
    assert legacy_metric.rejection_status is None
    assert legacy_metric.improvement_points == 2.5


@pytest.mark.asyncio
async def test_queue_rejects_overlapping_categories_before_writes(monkeypatch) -> None:
    async def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("store must not be called for an invalid partition")

    monkeypatch.setattr(global_icp_queue, "select_one", unexpected)
    with pytest.raises(ValueError, match="categories overlap"):
        await global_icp_queue.enqueue_candidate(
            candidate_id="candidate:" + "1" * 64,
            window_hash=WINDOW_HASH,
            public_items=[{"icp_ref": "same"}],
            private_items=[{"icp_ref": "private"}],
            conditional_items=[{"icp_ref": "same"}],
            baseline_public_score=40.0,
            baseline_preliminary_score=40.0,
            threshold_points=1.0,
            baseline_benchmark_bundle_id="private_benchmark:" + "2" * 64,
            baseline_benchmark_hash="sha256:" + "3" * 64,
            category_assignment_hash="sha256:" + "4" * 64,
            conditional_policy_hash="sha256:" + "5" * 64,
            worker_ref="worker:test",
            seq_base=0,
        )


@pytest.mark.asyncio
async def test_queue_refill_rejects_changed_frozen_counts(monkeypatch) -> None:
    async def existing(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "queue_generation_id": "11111111-1111-4111-8111-111111111111",
            "public_total": 9,
            "private_total": 10,
            "conditional_total": 20,
            "baseline_public_score": 40.0,
            "baseline_preliminary_score": 40.0,
            "threshold_points": 1.0,
            "baseline_benchmark_bundle_id": "private_benchmark:" + "2" * 64,
            "baseline_benchmark_hash": "sha256:" + "3" * 64,
            "category_assignment_hash": "sha256:" + "4" * 64,
            "conditional_policy_hash": "sha256:" + "5" * 64,
            "candidate_artifact_hash": "sha256:" + "6" * 64,
            "candidate_parent_artifact_hash": "sha256:" + "7" * 64,
            "scoring_configuration_hash": "sha256:" + "8" * 64,
        }

    monkeypatch.setattr(global_icp_queue, "select_one", existing)
    with pytest.raises(RuntimeError, match="public_total"):
        await global_icp_queue.enqueue_candidate(
            candidate_id="candidate:" + "1" * 64,
            window_hash=WINDOW_HASH,
            public_items=[{"icp_ref": f"public-{index}"} for index in range(10)],
            private_items=[{"icp_ref": f"private-{index}"} for index in range(10)],
            conditional_items=[{"icp_ref": f"conditional-{index}"} for index in range(20)],
            baseline_public_score=40.0,
            baseline_preliminary_score=40.0,
            threshold_points=1.0,
            baseline_benchmark_bundle_id="private_benchmark:" + "2" * 64,
            baseline_benchmark_hash="sha256:" + "3" * 64,
            category_assignment_hash="sha256:" + "4" * 64,
            conditional_policy_hash="sha256:" + "5" * 64,
            candidate_artifact_hash="sha256:" + "6" * 64,
            candidate_parent_artifact_hash="sha256:" + "7" * 64,
            scoring_configuration_hash="sha256:" + "8" * 64,
            worker_ref="worker:test",
            seq_base=0,
        )


@pytest.mark.asyncio
async def test_conditional_queue_retry_uses_atomic_lifecycle_rpc(monkeypatch) -> None:
    calls: list[tuple[str, Mapping[str, Any]]] = []

    async def rpc(name: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append((name, dict(params)))
        return {"committed": True}

    monkeypatch.setattr(global_icp_queue, "call_rpc", rpc)
    committed = await global_icp_queue.requeue_job(
        job={
            "job_id": "11111111-1111-4111-8111-111111111111",
            "phase": "conditional",
            "claimed_by": "worker:test",
            "attempt_count": 7,
        },
        result_doc={
            "retryable": True,
            "failure_class": "conditional_validation_retryable_failure",
        },
    )
    assert committed is True
    assert calls == [
        (
            "research_lab_requeue_conditional_scoring_job",
            {
                "target_job_id": "11111111-1111-4111-8111-111111111111",
                "expected_claimed_by": "worker:test",
                "expected_attempt_count": 7,
                "target_failure_class": "conditional_validation_retryable_failure",
            },
        )
    ]


@pytest.mark.asyncio
async def test_queue_result_receipts_survive_restart_without_leaking_into_bundle_rows(
    monkeypatch,
) -> None:
    first_hash = "sha256:" + "1" * 64
    second_hash = "sha256:" + "2" * 64

    async def jobs(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "phase": "public",
                "status": "done",
                "item_index": 0,
                "result_doc": {
                    "icp_ref": "public-0",
                    global_icp_queue.ATTESTED_RECEIPT_HASHES_FIELD: [first_hash],
                },
            },
            {
                "phase": "private",
                "status": "done",
                "item_index": 0,
                "result_doc": {
                    "icp_ref": "private-0",
                    global_icp_queue.ATTESTED_RECEIPT_HASHES_FIELD: [
                        second_hash,
                        first_hash,
                    ],
                },
            },
        ]

    monkeypatch.setattr(global_icp_queue, "select_many", jobs)
    docs = await global_icp_queue.candidate_result_docs("generation-1")
    assert docs == {
        "public": [{"icp_ref": "public-0"}],
        "private": [{"icp_ref": "private-0"}],
        "conditional": [],
        "attested_receipt_hashes": [first_hash, second_hash],
    }
    assert global_icp_queue.ATTESTED_RECEIPT_HASHES_FIELD not in json.dumps(docs)

    parents = scoring_worker._queue_attested_parent_receipts(docs)
    assert parents == [
        {"receipt_hash": first_hash},
        {"receipt_hash": second_hash},
    ]


@pytest.mark.asyncio
async def test_queue_result_rejects_malformed_persisted_receipt_sidecar(monkeypatch) -> None:
    async def jobs(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "phase": "public",
                "status": "done",
                "item_index": 0,
                "result_doc": {
                    "icp_ref": "public-0",
                    global_icp_queue.ATTESTED_RECEIPT_HASHES_FIELD: ["not-a-hash"],
                },
            }
        ]

    monkeypatch.setattr(global_icp_queue, "select_many", jobs)
    with pytest.raises(RuntimeError, match="invalid attested receipt hashes"):
        await global_icp_queue.candidate_result_docs("generation-1")


def test_queue_receipt_retry_is_fail_closed_in_every_v2_phase() -> None:
    missing = evaluator.ConditionalValidationRetryableError(
        "conditional_queue_attested_receipt_missing"
    )
    for phase in ("public", "private", "conditional"):
        assert scoring_worker._queue_job_error_is_retryable(
            {"phase": phase},
            missing,
        )
    assert not scoring_worker._queue_job_error_is_retryable(
        {"phase": "public"},
        RuntimeError("ordinary candidate failure"),
    )


def test_conditional_queue_missing_context_or_item_is_retryable() -> None:
    job = {"phase": "conditional", "icp_ref": "conditional-1"}
    with pytest.raises(
        evaluator.ConditionalValidationRetryableError,
        match="scoring_context_missing",
    ):
        scoring_worker._queue_scoring_item(None, job)
    with pytest.raises(
        evaluator.ConditionalValidationRetryableError,
        match="scoring_item_missing",
    ):
        scoring_worker._queue_scoring_item({"items_by_ref": {}}, job)
    assert scoring_worker._queue_scoring_item(None, {**job, "phase": "public"}) is None


def test_conditional_queue_rejects_nonhex_commitment_hashes() -> None:
    values = {
        "baseline_public_score": 40.0,
        "baseline_preliminary_score": 40.0,
        "threshold_points": 1.0,
        "baseline_benchmark_bundle_id": "private_benchmark:test",
        "baseline_benchmark_hash": "sha256:" + "1" * 64,
        "category_assignment_hash": "sha256:" + "2" * 64,
        "conditional_policy_hash": "sha256:" + "3" * 64,
        "candidate_artifact_hash": "sha256:" + "4" * 64,
        "candidate_parent_artifact_hash": "sha256:" + "5" * 64,
        "scoring_configuration_hash": "sha256:" + "6" * 64,
    }
    global_icp_queue._validate_conditional_generation_values(**values)
    with pytest.raises(ValueError, match="valid category_assignment_hash"):
        global_icp_queue._validate_conditional_generation_values(
            **{**values, "category_assignment_hash": "sha256:" + "z" * 64}
        )


def test_deterministic_attested_receipt_retry_is_bound_to_current_queue_result() -> None:
    receipt_hash = "sha256:" + "a" * 64
    scorer = evaluator.QualificationStyleCompanyScorer()
    scorer._record_attested_outcome(
        {"receipt": {"receipt_hash": receipt_hash}}
    )
    count_before = scorer.attested_outcome_count()
    hashes_before = {
        str(item["receipt_hash"]) for item in scorer.attested_receipts()
    }

    scorer._record_attested_outcome(
        {"receipt": {"receipt_hash": receipt_hash}}
    )

    assert scorer.attested_receipts() == [{"receipt_hash": receipt_hash}]
    assert scorer.attested_outcome_count() == count_before + 1
    assert scoring_worker._queue_current_attested_receipt_hashes(
        scorer=scorer,
        scorer_outcome_count_before=count_before,
        receipt_hashes_before=hashes_before,
        receipt_hashes_after={
            str(item["receipt_hash"]) for item in scorer.attested_receipts()
        },
    ) == [receipt_hash]


def test_queue_receipt_binding_rejects_stale_receipts_without_current_call() -> None:
    receipt_hash = "sha256:" + "b" * 64
    scorer = evaluator.QualificationStyleCompanyScorer()
    scorer._record_attested_outcome(
        {"receipt": {"receipt_hash": receipt_hash}}
    )
    count_before = scorer.attested_outcome_count()

    with pytest.raises(
        evaluator.ConditionalValidationRetryableError,
        match="conditional_queue_attested_receipt_missing",
    ):
        scoring_worker._queue_current_attested_receipt_hashes(
            scorer=scorer,
            scorer_outcome_count_before=count_before,
            receipt_hashes_before={receipt_hash},
            receipt_hashes_after={receipt_hash},
        )


@pytest.mark.asyncio
async def test_preliminary_gate_restart_reconciliation_has_one_winner_across_ten_workers(
    monkeypatch,
) -> None:
    generation_id = "11111111-1111-4111-8111-111111111111"
    candidate_row = {
        "queue_generation_id": generation_id,
        "candidate_id": "candidate:" + "1" * 64,
        "conditional_total": 20,
        "gate_status": "passed",
        "preliminary_gate_status": "pending",
        "assembly_status": "pending",
        "baseline_preliminary_score": 40.0,
        "threshold_points": 1.0,
        "preliminary_gate_attempt_count": 0,
    }
    state = {"claimed": False, "decided": False, "authorizations": 0}
    lock = asyncio.Lock()

    async def no_recovery(**_kwargs: Any) -> int:
        return 0

    async def pending(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [] if state["decided"] else [dict(candidate_row)]

    async def current_candidate(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return None if state["decided"] else dict(candidate_row)

    async def complete(*_args: Any, **_kwargs: Any) -> bool:
        return True

    async def claim(**kwargs: Any) -> dict[str, Any] | None:
        async with lock:
            if state["claimed"] or state["decided"]:
                return None
            state["claimed"] = True
            return {
                **candidate_row,
                "preliminary_gate_status": "deciding",
                "preliminary_gate_attempt_count": 1,
                "preliminary_gate_claimed_by": kwargs["worker_ref"],
            }

    async def docs(_generation_id: str) -> dict[str, list[dict[str, Any]]]:
        return {
            "public": [{"icp_ref": f"public-{index}"} for index in range(10)],
            "private": [{"icp_ref": f"private-{index}"} for index in range(10)],
            "conditional": [],
        }

    async def authorize(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        state["authorizations"] += 1
        await asyncio.sleep(0)
        return {"proof_hash": "sha256:" + "9" * 64}

    async def decide(**kwargs: Any) -> str:
        assert kwargs["preliminary_proof"]["proof_hash"].startswith("sha256:")
        async with lock:
            state["decided"] = True
        return "passed"

    async def no_job(**_kwargs: Any) -> None:
        return None

    async def never_assemble(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("restart reconciliation must not assemble before conditional jobs")

    monkeypatch.setattr(global_icp_queue, "recover_stale_leases", no_recovery)
    monkeypatch.setattr(global_icp_queue, "select_many", pending)
    monkeypatch.setattr(global_icp_queue, "select_one", current_candidate)
    monkeypatch.setattr(global_icp_queue, "phase_set_complete", complete)
    monkeypatch.setattr(global_icp_queue, "claim_preliminary_gate", claim)
    monkeypatch.setattr(global_icp_queue, "candidate_result_docs", docs)
    monkeypatch.setattr(global_icp_queue, "try_decide_preliminary_gate", decide)
    monkeypatch.setattr(global_icp_queue, "claim_next_job", no_job)

    results = await asyncio.gather(
        *(
            global_icp_queue.run_queue_scoring_pass(
                worker_ref=f"worker:{index}",
                lease_seconds=120,
                score_icp=lambda _job: None,  # type: ignore[arg-type,return-value]
                compute_public_score=lambda _rows: 42.0,
                compute_preliminary_score=lambda _rows: 42.0,
                preliminary_gate_authorizer=authorize,
                assemble_candidate=never_assemble,
            )
            for index in range(10)
        )
    )
    assert state["authorizations"] == 1
    assert state["decided"] is True
    assert sum(result["preliminary_gates_decided"] for result in results) == 1


def test_conditional_retryable_failure_never_becomes_terminal_from_attempt_cap() -> None:
    assert scoring_worker._candidate_scoring_should_requeue(
        failure_class="conditional_validation_retryable_failure",
        retryable=True,
        claim_attempts=100,
        max_attempts=3,
    )
    assert not scoring_worker._candidate_scoring_should_requeue(
        failure_class="candidate_runtime_error",
        retryable=False,
        claim_attempts=1,
        max_attempts=3,
    )
    assert not scoring_worker._candidate_scoring_should_requeue(
        failure_class="adapter_timeout",
        retryable=True,
        claim_attempts=3,
        max_attempts=3,
    )


@pytest.mark.asyncio
async def test_final_conditional_events_precede_promotion_side_effects(monkeypatch) -> None:
    events: list[dict[str, Any]] = []

    async def record(**kwargs: Any) -> dict[str, Any]:
        events.append(dict(kwargs))
        return dict(kwargs)

    monkeypatch.setattr(scoring_worker, "create_conditional_validation_event", record)
    await scoring_worker._persist_conditional_finalization_events(
        {
            **_gate(),
            "decision": "conditional_validation_approved",
            "candidate_conditional_score": 45.0,
            "candidate_total_score": 44.0,
            "candidate_delta_vs_daily_baseline": 4.0,
            "conditional_holdout_evaluated": True,
            "public_icp_count": 10,
            "private_holdout_icp_count": 10,
            "conditional_holdout_icp_count": 20,
        },
        candidate_id="candidate:" + "6" * 64,
        source_score_bundle_id="score_bundle:" + "7" * 64,
        rolling_window_hash=WINDOW_HASH,
    )
    assert [event["event_type"] for event in events] == [
        "conditional_completed",
        "final_pass",
    ]
    assert events[-1]["decision_score"] == 44.0
    assert events[-1]["event_doc"]["total_icp_count"] == 40


@pytest.mark.asyncio
async def test_lifecycle_store_write_is_idempotent_and_private(monkeypatch) -> None:
    inserted: list[tuple[str, dict[str, Any]]] = []

    async def missing(*args: Any, **kwargs: Any) -> None:
        return None

    async def insert(table: str, row: dict[str, Any]) -> dict[str, Any]:
        inserted.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(store, "select_one", missing)
    monkeypatch.setattr(store, "insert_row", insert)
    row = await store.create_conditional_validation_event(
        candidate_id="candidate:" + "8" * 64,
        event_type="conditional_started",
        assignment_hash="sha256:" + "9" * 64,
        policy_hash="sha256:" + "a" * 64,
        rolling_window_hash=WINDOW_HASH,
        baseline_benchmark_bundle_id="private_benchmark:" + "b" * 64,
        source_ref="direct:test",
        decision_score=42.0,
        threshold_points=1.0,
        event_doc={"path": "test"},
    )
    assert row["event_hash"].startswith("sha256:")
    assert inserted[0][0] == "research_lab_conditional_validation_events"
    encoded = json.dumps(inserted[0][1], sort_keys=True).lower()
    assert "intent_signals" not in encoded
    assert "provider_output" not in encoded


def test_migration_serializes_gate_release_and_retry_recording() -> None:
    sql = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "97-research-lab-conditional-validation.sql"
    ).read_text(encoding="utf-8")
    preliminary = sql.split(
        "CREATE OR REPLACE FUNCTION public.research_lab_decide_conditional_preliminary_gate",
        1,
    )[1].split(
        "CREATE OR REPLACE FUNCTION public.research_lab_requeue_conditional_scoring_job",
        1,
    )[0]
    assert preliminary.index(
        "INSERT INTO public.research_lab_conditional_validation_events"
    ) < preliminary.index("AND phase = 'conditional'")
    assert "preliminary_gate_status <> 'deciding'" in preliminary
    assert "preliminary_gate_claimed_by <> expected_claimed_by" in preliminary
    assert "preliminary_gate_proof = target_preliminary_proof" in preliminary
    assert "IS DISTINCT FROM candidate_row.scoring_configuration_hash" in preliminary
    assert "research_lab_claim_conditional_preliminary_gate" in sql
    assert "research_lab_cancel_conditional_generation" in sql
    assert (
        "research_lab_decide_conditional_preliminary_gate(UUID, DOUBLE PRECISION, JSONB, TEXT, INTEGER)"
        in sql
    )
    assert "prevent_research_lab_conditional_validation_events_mutation" in sql
    assert "ALTER TABLE public.research_lab_conditional_validation_events ENABLE ROW LEVEL SECURITY" in sql
    assert "research_lab_requeue_conditional_scoring_job" in sql
    assert sql.index("'retryable_failure'") < sql.index("SET status = 'queued'")


def test_reimbursement_policy_hash_inputs_do_not_include_conditional_controls() -> None:
    policy = ResearchLabGatewayConfig().reimbursement_policy_doc()
    assert not any("conditional" in key for key in policy)


def test_checkpoint_reuse_requires_exact_commitment_and_verified_readback(monkeypatch) -> None:
    objects: dict[tuple[str, str], bytes] = {}

    class S3:
        def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: Any) -> None:
            objects[(Bucket, Key)] = bytes(Body)

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            return {"Body": io.BytesIO(objects[(Bucket, Key)])}

    client = S3()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(client=lambda _service: client),
    )
    rows = [{"icp_ref": "public-1", "candidate_company_scores": [50.0]}]
    commitment = "sha256:" + "c" * 64
    progress_hash = scoring_worker._store_scoring_progress(
        "bucket",
        "progress.json",
        candidate_id="candidate:" + "d" * 64,
        window_hash=WINDOW_HASH,
        candidate_artifact_hash="sha256:" + "e" * 64,
        rows=rows,
        commitment_hash=commitment,
    )
    assert progress_hash.startswith("sha256:")
    assert scoring_worker._load_scoring_progress(
        "bucket",
        "progress.json",
        window_hash=WINDOW_HASH,
        candidate_artifact_hash="sha256:" + "e" * 64,
        commitment_hash=commitment,
    ) == rows
    assert scoring_worker._load_scoring_progress(
        "bucket",
        "progress.json",
        window_hash=WINDOW_HASH,
        candidate_artifact_hash="sha256:" + "e" * 64,
        commitment_hash="sha256:" + "f" * 64,
    ) == []


@pytest.mark.asyncio
async def test_reused_signed_bundle_repairs_derivatives_before_scored_event(monkeypatch) -> None:
    worker = scoring_worker.ResearchLabGatewayScoringWorker(
        ResearchLabGatewayConfig(),
        worker_ref="worker:test",
    )
    order: list[str] = []

    async def finalization(*args: Any, **kwargs: Any) -> None:
        order.append("lifecycle")

    async def categories(*args: Any, **kwargs: Any) -> None:
        order.append("categories")

    async def scored(*args: Any, **kwargs: Any) -> None:
        order.append("scored")
        raise RuntimeError("stop-after-order-proof")

    monkeypatch.setattr(scoring_worker, "_persist_conditional_finalization_events", finalization)
    monkeypatch.setattr(scoring_worker, "_persist_candidate_category_results", categories)
    monkeypatch.setattr(worker, "_create_scored_evaluation_event", scored)
    monkeypatch.setattr(worker, "_scoring_health_gate_result", lambda _bundle: {"decision": "healthy"})
    with pytest.raises(RuntimeError, match="stop-after-order-proof"):
        await worker._complete_candidate_from_reused_bundle(
            {
                "candidate_id": "candidate:" + "1" * 64,
                "run_id": "11111111-1111-4111-8111-111111111111",
                "ticket_id": "22222222-2222-4222-8222-222222222222",
            },
            candidate_id="candidate:" + "1" * 64,
            bundle_row={
                "score_bundle_id": "score_bundle:" + "2" * 64,
                "score_bundle_doc": {
                    "icp_set_hash": WINDOW_HASH,
                    "private_holdout_gate": {
                        **_gate(),
                        "decision": "conditional_validation_approved",
                        "conditional_holdout_evaluated": True,
                    },
                },
            },
            evaluation_epoch=42,
            start=0.0,
        )
    assert order == ["lifecycle", "categories", "scored"]


@pytest.mark.asyncio
async def test_global_queue_final_bundle_reaches_one_promotion_handoff(monkeypatch) -> None:
    worker = object.__new__(scoring_worker.ResearchLabGatewayScoringWorker)
    worker.config = types.SimpleNamespace(
        score_bundle_kms_key_id="test-key",
        score_bundle_signature_uri_prefix="s3://private/signatures",
    )
    worker.worker_ref = "worker:test"
    worker.proxy_ref_hash = None
    order: list[str] = []
    gate = {
        **_gate(),
        "decision": "conditional_validation_approved",
        "candidate_conditional_score": 45.0,
        "candidate_total_score": 44.0,
        "candidate_delta_vs_daily_baseline": 4.0,
        "conditional_holdout_evaluated": True,
        "public_icp_count": 10,
        "private_holdout_icp_count": 10,
        "conditional_holdout_icp_count": 20,
    }
    score_bundle = {
        "bundle_type": "research_lab_evaluation_score_bundle",
        "schema_version": "1.1",
        "score_bundle_hash": "sha256:" + "3" * 64,
        "anchored_hash": "sha256:" + "3" * 64,
        "private_holdout_gate": gate,
    }

    monkeypatch.setattr(
        scoring_worker,
        "build_holdout_gate_result",
        lambda **_kwargs: ([{"icp_ref": "conditional-1"}], gate),
    )
    monkeypatch.setattr(
        scoring_worker,
        "build_score_bundle_from_scored_icps",
        lambda **_kwargs: dict(score_bundle),
    )

    async def compare(**_kwargs: Any) -> None:
        order.append("attested_compare")

    async def create_bundle(_request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        order.append("bundle")
        return ({"score_bundle_id": "score_bundle:" + "4" * 64}, {})

    async def lifecycle(*_args: Any, **_kwargs: Any) -> None:
        order.append("lifecycle")

    async def categories(*_args: Any, **_kwargs: Any) -> None:
        order.append("categories")

    async def scored(*_args: Any, **_kwargs: Any) -> None:
        order.append("scored")

    async def dispatch(*_args: Any, **_kwargs: Any) -> None:
        order.append("dispatch")

    async def promote(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        order.append("promotion")
        return {"status": "not_promoted"}

    async def backfill(*_args: Any, **_kwargs: Any) -> None:
        order.append("backfill")

    async def finalize(*_args: Any, **_kwargs: Any) -> None:
        order.append("receipt")

    async def project(*_args: Any, **_kwargs: Any) -> None:
        order.append("projection")

    async def audit(*_args: Any, **_kwargs: Any) -> None:
        order.append("audit")

    monkeypatch.setattr(scoring_worker, "_compare_candidate_score_bundle_in_enclave", compare)
    monkeypatch.setattr(scoring_worker, "_attested_receipts_from", lambda *_args: [])
    monkeypatch.setattr(scoring_worker, "sign_digest_with_kms", lambda **_kwargs: "kms:test")
    monkeypatch.setattr(
        scoring_worker,
        "ResearchLabScoreBundleCreateRequest",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(scoring_worker, "create_score_bundle", create_bundle)
    monkeypatch.setattr(scoring_worker, "_persist_conditional_finalization_events", lifecycle)
    monkeypatch.setattr(scoring_worker, "_persist_candidate_category_results", categories)
    monkeypatch.setattr(scoring_worker, "create_scoring_dispatch_event", dispatch)
    monkeypatch.setattr(scoring_worker, "safe_project_public_loop_activity", project)
    monkeypatch.setattr(worker, "_scoring_health_gate_result", lambda _bundle: {"decision": "observe_only"})
    monkeypatch.setattr(worker, "_create_scored_evaluation_event", scored)
    monkeypatch.setattr(worker, "_maybe_promote_scored_candidate", promote)
    monkeypatch.setattr(worker, "_maybe_record_score_backfill", backfill)
    monkeypatch.setattr(worker, "_maybe_finalize_candidate_receipt", finalize)
    monkeypatch.setattr(worker, "_write_audit_bundle", audit)

    await worker._queue_assemble_candidate(
        "candidate:" + "1" * 64,
        {
            "public": [{"icp_ref": "public-1"}],
            "private": [{"icp_ref": "private-1"}],
            "conditional": [{"icp_ref": "conditional-1"}],
        },
        {
            "artifact": object(),
            "benchmark": object(),
            "patch": object(),
            "candidate_artifact": types.SimpleNamespace(to_dict=lambda: {}),
            "public_items": [object()] * 10,
            "private_items": [object()] * 10,
            "conditional_items": [object()] * 20,
            "run_context": {"evaluation_epoch": 42},
            "evaluation_policy": {},
            "gate": {"conditional_validation_required": False},
            "runner": object(),
            "scorer": object(),
            "candidate": {
                "candidate_id": "candidate:" + "1" * 64,
                "run_id": "11111111-1111-4111-8111-111111111111",
                "ticket_id": "22222222-2222-4222-8222-222222222222",
            },
            "window": types.SimpleNamespace(window_hash=WINDOW_HASH),
        },
    )
    assert order == [
        "attested_compare",
        "bundle",
        "lifecycle",
        "categories",
        "scored",
        "dispatch",
        "promotion",
        "backfill",
        "receipt",
        "projection",
        "audit",
    ]


def test_v2_measured_baseline_requires_exact_conditional_policy() -> None:
    executor = object.__new__(ScoringExecutorV2)

    class Config:
        public_benchmark_public_icps_per_day = 3
        public_benchmark_public_weak_per_day = 2
        public_benchmark_public_total_icps = 10
        public_benchmark_public_weak_total = 7

        @staticmethod
        def conditional_validation_policy() -> ConditionalValidationPolicy:
            return _policy()

    executor._config = Config()
    payload = {
        "public_icps_per_day": 3,
        "public_weak_per_day": 2,
        "public_total_icps": 10,
        "public_weak_total": 7,
        "max_unresolved_icps": 2,
        "conditional_validation_policy": _policy().to_dict(),
    }
    executor._validate_baseline_configuration(payload)
    with pytest.raises(ValueError, match="conditional validation policy differs"):
        executor._validate_baseline_configuration(
            {
                **payload,
                "conditional_validation_policy": {
                    **_policy().to_dict(),
                    "conditional_total_icps": 10,
                },
            }
        )


def test_v2_final_bundle_requires_exact_preliminary_decision_ancestry(
    monkeypatch,
) -> None:
    executor = object.__new__(ScoringExecutorV2)
    executor._config = types.SimpleNamespace(improvement_threshold_points=1.0)
    config_hash = "sha256:" + "8" * 64
    monkeypatch.setattr(scoring_executor_v2, "configuration_hash", lambda: config_hash)
    candidate_hash = "sha256:" + "6" * 64
    parent_hash = "sha256:" + "7" * 64
    decision_receipt_hash = "sha256:" + "9" * 64
    metric_receipt_hash = "sha256:" + "3" * 64
    score_bundle_receipt_hash = "sha256:" + "2" * 64
    decision = {
        "status": "promotion_passed",
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "candidate_kind": "image_build",
        "auto_promotion_enabled": True,
        "active_parent_matches": True,
        "metric_rejection_status": None,
    }
    decision_output_root = sha256_json({"decision": decision})
    proof_body = {
        "schema_version": "research_lab_preliminary_promotion_gate.v1",
        "status": "promotion_passed",
        "preliminary_score_bundle_hash": "sha256:" + "1" * 64,
        "score_bundle_receipt_hash": score_bundle_receipt_hash,
        "promotion_metric_receipt_hash": metric_receipt_hash,
        "promotion_decision_receipt_hash": decision_receipt_hash,
        "promotion_decision_output_root": decision_output_root,
        "candidate_artifact_hash": candidate_hash,
        "candidate_parent_artifact_hash": parent_hash,
        "active_parent_artifact_hash": parent_hash,
        "rolling_window_hash": WINDOW_HASH,
        "category_assignment_hash": "sha256:" + "4" * 64,
        "conditional_validation_policy_hash": "sha256:" + "5" * 64,
        "scoring_configuration_hash": config_hash,
        "threshold_points": 1.0,
        "decision": decision,
    }
    proof = {**proof_body, "proof_hash": sha256_json(proof_body)}
    gate = {
        "conditional_validation_required": True,
        "conditional_holdout_evaluated": True,
        "threshold_points": 1.0,
        "category_assignment_hash": proof["category_assignment_hash"],
        "conditional_validation_policy_hash": proof[
            "conditional_validation_policy_hash"
        ],
        "preliminary_promotion_gate": proof,
    }
    payload = {
        "artifact_manifest": {"model_artifact_hash": parent_hash},
        "candidate_artifact_manifest": {"model_artifact_hash": candidate_hash},
        "patch_manifest": {"parent_artifact_hash": parent_hash},
        "run_context": {"rolling_window_hash": WINDOW_HASH},
        "extra_bundle_fields": {"private_holdout_gate": gate},
    }
    receipt = {
        "receipt_hash": decision_receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.promotion_decision.v2",
        "status": "succeeded",
        "output_root": decision_output_root,
        "parent_receipt_hashes": [metric_receipt_hash],
    }
    metric_receipt = {
        "receipt_hash": metric_receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.ranking.v2",
        "status": "succeeded",
        "parent_receipt_hashes": [score_bundle_receipt_hash],
    }
    score_bundle_receipt = {
        "receipt_hash": score_bundle_receipt_hash,
        "role": "gateway_scoring",
        "purpose": "research_lab.candidate_score.v2",
        "status": "succeeded",
        "parent_receipt_hashes": [],
    }
    context = types.SimpleNamespace(
        parent_receipt_hashes=(decision_receipt_hash,),
        external_receipt_graphs=(
            {
                "root_receipt_hash": decision_receipt_hash,
                "receipts": [score_bundle_receipt, metric_receipt, receipt],
            },
        ),
    )
    executor._validate_conditional_preliminary_ancestry(payload, context)

    missing = types.SimpleNamespace(parent_receipt_hashes=(), external_receipt_graphs=())
    with pytest.raises(ValueError, match="ancestry is missing"):
        executor._validate_conditional_preliminary_ancestry(payload, missing)

    broken_chain = types.SimpleNamespace(
        parent_receipt_hashes=(decision_receipt_hash,),
        external_receipt_graphs=(
            {
                "root_receipt_hash": decision_receipt_hash,
                "receipts": [
                    score_bundle_receipt,
                    metric_receipt,
                    {**receipt, "parent_receipt_hashes": [score_bundle_receipt_hash]},
                ],
            },
        ),
    )
    with pytest.raises(ValueError, match="ancestry differs"):
        executor._validate_conditional_preliminary_ancestry(payload, broken_chain)

    tampered_gate = {
        **gate,
        "preliminary_promotion_gate": {**proof, "proof_hash": "sha256:" + "0" * 64},
    }
    with pytest.raises(ValueError, match="proof differs"):
        executor._validate_conditional_preliminary_ancestry(
            {
                **payload,
                "extra_bundle_fields": {"private_holdout_gate": tampered_gate},
            },
            context,
        )
