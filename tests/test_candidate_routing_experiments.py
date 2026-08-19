"""Focused contracts for the replay-first company-routing experiment lane."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from research_lab.canonical import sha256_json
from research_lab.candidate_routing_experiments import (
    CandidateRoutingArm,
    CandidateRoutingAttempt,
    CandidateRoutingExperiment,
    CandidateRoutingRun,
    RoutingEvaluationPolicy,
    RoutingExperimentError,
    candidate_routing_attempt_from_model_receipt,
    evaluate_routing_replay,
    validate_candidate_routing_model_runtime,
)


def _hash(value: str) -> str:
    return sha256_json({"fixture": value})


def _model_hash(value: str) -> str:
    return _hash(value).removeprefix("sha256:")


def _experiment() -> CandidateRoutingExperiment:
    return CandidateRoutingExperiment(
        experiment_id="routing_experiment:0000000000000001",
        name="company source profile replay",
        model_commit="a" * 40,
        model_artifact_hash=_hash("model"),
        routing_contract_hash=_model_hash("routing"),
        profile_registry_hash=_model_hash("profiles"),
        provider_catalog_hash=_model_hash("catalog"),
        dev_set_hash=_hash("dev"),
        snapshot_manifest_hash=_hash("snapshot"),
        target_qualified_count=2,
        max_provider_calls=100,
        max_cost_microusd=100_000,
        max_duration_ms=100_000,
    )


def _arms(experiment: CandidateRoutingExperiment) -> tuple[CandidateRoutingArm, CandidateRoutingArm]:
    return (
        CandidateRoutingArm(
            arm_id="arm:control",
            experiment_id=experiment.experiment_id,
            experiment_hash=experiment.experiment_hash,
            label="control",
            profile_id="profile:default",
            profile_hash=_model_hash("control-profile"),
            is_control=True,
        ),
        CandidateRoutingArm(
            arm_id="arm:candidate",
            experiment_id=experiment.experiment_id,
            experiment_hash=experiment.experiment_hash,
            label="candidate",
            profile_id="profile:industry-intent",
            profile_hash=_model_hash("candidate-profile"),
        ),
    )


def _run(experiment: CandidateRoutingExperiment, arm: CandidateRoutingArm, index: int) -> CandidateRoutingRun:
    return CandidateRoutingRun(
        run_id=f"run:{arm.arm_id.split(':')[-1]}:{index:04d}",
        experiment_id=experiment.experiment_id,
        experiment_hash=experiment.experiment_hash,
        arm_id=arm.arm_id,
        icp_ref=f"icp:{index:04d}",
        icp_hash=_hash(f"icp-{index}"),
        snapshot_manifest_hash=experiment.snapshot_manifest_hash,
        route_plan_hash=_model_hash(f"plan-{arm.arm_id}-{index}"),
    )


def _attempt(
    experiment: CandidateRoutingExperiment,
    arm: CandidateRoutingArm,
    run: CandidateRoutingRun,
    index: int,
    *,
    published: int,
    cost: int,
    outcome: str = "success",
    snapshot_hit: bool = True,
) -> CandidateRoutingAttempt:
    return CandidateRoutingAttempt(
        attempt_id=f"attempt:{arm.arm_id.split(':')[-1]}:{index:04d}",
        run_id=run.run_id,
        experiment_id=experiment.experiment_id,
        experiment_hash=experiment.experiment_hash,
        arm_id=arm.arm_id,
        icp_ref=run.icp_ref,
        step_order=0,
        attempt_sequence=0,
        tool_id="candidate.registry_feed",
        disposition="succeeded" if outcome == "success" else "failed",
        outcome=outcome,
        route_plan_hash=run.route_plan_hash,
        stop_policy_hash=_model_hash("stop-policy"),
        attempt_receipt_hash=_model_hash(
            f"attempt-receipt-{arm.arm_id}-{index}"
        ),
        verification_receipt_hash=(
            _model_hash(f"verification-{arm.arm_id}-{index}")
            if outcome == "success"
            else ""
        ),
        provider_id="registry",
        provider_call_count=1,
        cost_microusd=cost,
        latency_ms=100,
        raw_count=10 if outcome == "success" else 0,
        unique_count=5 if outcome == "success" else 0,
        verified_count=4 if outcome == "success" else 0,
        qualified_count=published if outcome == "success" else 0,
        published_count=published if outcome == "success" else 0,
        snapshot_hit=snapshot_hit,
        result_hash=_hash(f"result-{arm.arm_id}-{index}"),
        failure_code="snapshot_missing" if outcome == "replay_miss" else "",
    )


def test_contract_hashes_are_stable_and_secret_material_is_rejected():
    experiment = _experiment()
    equivalent = CandidateRoutingExperiment(**{**experiment.payload(), "metadata": {"b": 2, "a": 1}})
    equivalent_reordered = CandidateRoutingExperiment(**{**experiment.payload(), "metadata": {"a": 1, "b": 2}})
    assert equivalent.experiment_hash == equivalent_reordered.experiment_hash
    assert experiment.to_dict()["experiment_hash"] == experiment.experiment_hash
    assert len(experiment.routing_contract_hash) == 64
    assert not experiment.routing_contract_hash.startswith("sha256:")

    with pytest.raises(RoutingExperimentError, match="secret-like"):
        CandidateRoutingExperiment(**{**experiment.payload(), "metadata": {"api_key": "not-a-secret"}})

    with pytest.raises(RoutingExperimentError, match="at least one"):
        CandidateRoutingExperiment(**{**experiment.payload(), "target_qualified_count": 0})


def test_attempt_counts_are_incremental_and_monotonic():
    experiment = _experiment()
    arms = _arms(experiment)
    run = _run(experiment, arms[0], 0)
    with pytest.raises(RoutingExperimentError, match="raw >= unique"):
        _attempt(experiment, arms[0], run, 0, published=6, cost=1)


def test_branch_model_attempt_receipt_is_validated_and_projected():
    experiment = _experiment()
    arm = _arms(experiment)[0]
    run = _run(experiment, arm, 0)
    receipt_hash = _model_hash("model-attempt")
    receipt = SimpleNamespace(
        plan_sha256=run.route_plan_hash,
        stop_policy_sha256=_model_hash("stop-policy"),
        step_order=0,
        attempt=0,
        tool_id="candidate.registry_feed",
        disposition="succeeded",
        outcome_code="verified_candidates",
        provider_call_count=1,
        estimated_cost_usd=0.0125,
        latency_seconds=0.25,
        raw_candidate_count=4,
        unique_candidate_count=2,
        verified_qualified_count=1,
        verification_receipt_sha256=_model_hash("verification"),
        sha256=lambda: receipt_hash,
    )
    runtime = SimpleNamespace(
        candidate_waterfall_execution_contract_identity=lambda: {
            "contract_sha256": experiment.routing_contract_hash,
            "provider_results_can_satisfy_target": False,
        },
        compile_candidate_stop_policy=lambda *args, **kwargs: None,
        compile_profiled_candidate_acquisition_route=lambda *args, **kwargs: None,
        evaluate_candidate_waterfall=lambda *args, **kwargs: None,
        runtime_catalog=lambda *args, **kwargs: None,
        runtime_policy=lambda *args, **kwargs: None,
        runtime_tool_definitions=lambda *args, **kwargs: None,
        CandidateStepAttemptReceipt=SimpleNamespace(
            from_payload=lambda _payload: receipt,
        ),
    )
    attempt = candidate_routing_attempt_from_model_receipt(
        experiment=experiment,
        arm=arm,
        run=run,
        attempt_id="attempt:model:0001",
        receipt_payload={"attempt_receipt_sha256": receipt_hash},
        model_runtime=runtime,
    )

    assert attempt.route_plan_hash == run.route_plan_hash
    assert attempt.stop_policy_hash == receipt.stop_policy_sha256
    assert attempt.attempt_receipt_hash == receipt_hash
    assert attempt.qualified_count == 1
    assert attempt.cost_microusd == 12_500


def test_branch_runtime_preflight_requires_complete_safe_router_surface():
    experiment = _experiment()
    with pytest.raises(RoutingExperimentError, match="contract is incomplete"):
        validate_candidate_routing_model_runtime(
            experiment=experiment,
            model_runtime=SimpleNamespace(
                candidate_waterfall_execution_contract_identity=lambda: {
                    "contract_sha256": experiment.routing_contract_hash,
                    "provider_results_can_satisfy_target": False,
                },
            ),
        )


def test_replay_evaluator_allows_candidate_canary_when_yield_improves():
    experiment = _experiment()
    control, candidate = _arms(experiment)
    runs = []
    attempts = []
    for index in range(3):
        control_run = _run(experiment, control, index)
        candidate_run = _run(experiment, candidate, index)
        runs.extend((control_run, candidate_run))
        attempts.extend(
            (
                _attempt(experiment, control, control_run, index, published=2, cost=5_000),
                _attempt(experiment, candidate, candidate_run, index, published=3, cost=5_000),
            )
        )
    result = evaluate_routing_replay(
        experiment=experiment,
        arms=(control, candidate),
        runs=runs,
        attempts=attempts,
        policy=RoutingEvaluationPolicy(min_runs=3),
    )
    states = {item.arm_id: item.state for item in result.decisions}
    assert states == {"arm:control": "eligible_for_shadow", "arm:candidate": "eligible_for_canary"}
    assert result.evaluation_hash.startswith("sha256:")
    assert result.metrics[0].metric_hash.startswith("sha256:")
    assert all(metric.fulfilled_run_count == 3 for metric in result.metrics)


def test_candidate_cannot_reach_canary_when_per_run_target_is_not_met():
    experiment = CandidateRoutingExperiment(
        **{**_experiment().payload(), "target_qualified_count": 4}
    )
    control, candidate = _arms(experiment)
    runs = []
    attempts = []
    for index in range(3):
        control_run = _run(experiment, control, index)
        candidate_run = _run(experiment, candidate, index)
        runs.extend((control_run, candidate_run))
        attempts.extend((
            _attempt(experiment, control, control_run, index, published=2, cost=5_000),
            _attempt(experiment, candidate, candidate_run, index, published=3, cost=5_000),
        ))
    result = evaluate_routing_replay(
        experiment=experiment,
        arms=(control, candidate),
        runs=runs,
        attempts=attempts,
        policy=RoutingEvaluationPolicy(min_runs=3),
    )
    decision = next(item for item in result.decisions if item.arm_id == candidate.arm_id)
    assert decision.state == "eligible_for_shadow"
    assert "qualified_target_not_met_for_all_runs" in decision.reason_codes


def test_replay_sample_shortfall_is_replay_only_and_snapshot_miss_is_rejected():
    experiment = _experiment()
    control, candidate = _arms(experiment)
    control_run = _run(experiment, control, 0)
    candidate_run = _run(experiment, candidate, 0)
    miss_run = CandidateRoutingRun(
        **{**candidate_run.payload(), "run_id": "run:candidate:miss", "status": "replay_miss"}
    )
    runs = (control_run, candidate_run, miss_run)
    attempts = (
        _attempt(experiment, control, control_run, 0, published=1, cost=1_000),
        _attempt(experiment, candidate, candidate_run, 0, published=1, cost=1_000),
        _attempt(
            experiment,
            candidate,
            miss_run,
            9,
            published=0,
            cost=0,
            outcome="replay_miss",
            snapshot_hit=False,
        ),
    )
    result = evaluate_routing_replay(
        experiment=experiment,
        arms=(control, candidate),
        runs=runs,
        attempts=attempts,
        policy=RoutingEvaluationPolicy(min_runs=2),
    )
    states = {item.arm_id: item.state for item in result.decisions}
    assert states["arm:candidate"] == "rejected"
    assert "snapshot_replay_miss" in next(item for item in result.decisions if item.arm_id == "arm:candidate").reason_codes


def test_replay_evaluator_rejects_unbound_attempt():
    experiment = _experiment()
    control, candidate = _arms(experiment)
    run = _run(experiment, control, 0)
    bad = _attempt(experiment, candidate, run, 0, published=1, cost=1_000)
    with pytest.raises(RoutingExperimentError, match="does not match"):
        evaluate_routing_replay(
            experiment=experiment,
            arms=(control, candidate),
            runs=(run,),
            attempts=(bad,),
        )


def test_sql_migration_is_append_only_and_routing_scoped():
    sql = (Path(__file__).parents[1] / "scripts" / "156-research-lab-candidate-routing-experiments.sql").read_text()
    for table in (
        "research_lab_candidate_routing_experiments",
        "research_lab_candidate_routing_arms",
        "research_lab_candidate_routing_runs",
        "research_lab_candidate_routing_attempts",
        "research_lab_candidate_routing_metrics",
        "research_lab_candidate_routing_decisions",
    ):
        assert f"CREATE TABLE IF NOT EXISTS public.{table}" in sql
        assert table in sql
    assert "ON DELETE CASCADE" not in sql
    assert "decision_state         TEXT NOT NULL CHECK" in sql
    assert "eligible_for_shadow" in sql
    assert "eligible_for_canary" in sql
    assert "promotion_scope' = 'candidate_routing_experiment'" in sql
    assert "FOREIGN KEY (experiment_id, arm_id) REFERENCES" in sql
    assert "FOREIGN KEY (experiment_id, arm_id, run_id) REFERENCES" in sql
    assert "FOREIGN KEY (experiment_id, arm_id, metric_hash) REFERENCES" in sql
    assert sql.count("FORCE ROW LEVEL SECURITY") == 6
    assert "FOR SELECT TO service_role USING (true)" in sql
    assert "FOR INSERT TO service_role WITH CHECK (true)" in sql
    assert "auth.role()" not in sql
    assert "target_qualified_count')::INTEGER >= 1" in sql
    assert "route_plan_hash ~ '^[0-9a-f]{64}$'" in sql
    assert "attempt_receipt_hash" in sql
    assert "stop_policy_hash" in sql
    assert "verification_receipt_hash" in sql
    assert "DROP TRIGGER" not in sql
    assert "DROP POLICY" not in sql
    assert "research_lab_candidate_evaluation_events" not in sql
