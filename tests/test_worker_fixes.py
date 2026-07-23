"""Tests for hosted Research Lab worker fixes (fableanalysis bugs 1, 26, 27, 28).

Covers:
  * Bug #1  — scripts/59 same-worker heartbeat allowance in the run claim guard
              (pure-Python re-implementation of the predicate + SQL markers) and
              the worker-side heartbeat claim-lost abort.
  * Bug #26 — loop-start credit ref resolved past the 20-event context window.
  * Bug #27 — stale-paused reaper no longer revives blocked_for_credit runs.
  * Bug #28 — requeue capacity/hotkey conflicts park the run as recoverable
              `paused` instead of leaving it wedged `started`.
  * P2      — oldest-first stale-recovery scans, generation-stats mode flag,
              worker proxy opener for worker LLM traffic.

The guard predicate tests mirror only the allow/reject decision of
scripts/59-research-lab-heartbeat-claim-guard.sql; real trigger behavior
(advisory-lock serialization, seq/created_at ordering, ERRCODE) must be
verified against a staging database before production rollout.
"""

from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

import gateway.research_lab.maintenance as maintenance_mod
import gateway.research_lab.worker as worker_mod
from gateway.research_lab.config import ResearchLabGatewayConfig


ROOT = Path(__file__).resolve().parents[1]
GUARD_SQL_PATH = ROOT / "scripts" / "59-research-lab-heartbeat-claim-guard.sql"

RUN_ID = "33333333-3333-4333-8333-333333333333"
TICKET_ID = "44444444-4444-4444-8444-444444444444"
MINER_HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX"


@pytest.fixture
def hosted_worker():
    return worker_mod.ResearchLabHostedWorker(ResearchLabGatewayConfig(), worker_ref="worker-a")


def test_v2_autoresearch_never_resolves_plaintext_openrouter_keys_on_parent():
    source = inspect.getsource(worker_mod.ResearchLabHostedWorker._process_run)
    for marker in (
        "key_resolver.resolve(",
        "key_resolver.resolve_management_key(",
        "_preflight_openrouter_credit(",
        "build_code_edit_dev_evaluator()",
        "legacy_v1_enabled()",
    ):
        assert marker not in source
    assert "build_attested_code_edit_dev_evaluator_v2(" in source
    assert "verify_openrouter_guard_v2(" in source


def test_tree_policy_owns_topology_and_paid_handoff(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_TREE_MODE", "active")
    monkeypatch.setenv("RESEARCH_LAB_TREE_MAX_NODES", "8")
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(), worker_ref="worker-a"
    )

    assert worker.tree_policy.mode == "active"
    assert worker.tree_policy.max_nodes == 8
    assert worker.tree_policy.live_max_icps_per_node == 5


@pytest.mark.parametrize(
    "message",
    (
        "research_lab_git_tree_create_stale_active_root",
        "research_lab_git_tree_handoff_stale_active_root",
    ),
)
def test_stale_tree_root_database_fences_are_retryable(message):
    assert worker_mod._is_retryable_worker_exception(RuntimeError(message))


def test_active_tree_rejects_legacy_v1_before_worker_readiness(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_TREE_MODE", "active")
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1")

    with pytest.raises(
        worker_mod.HostedResearchLabWorkerError,
        match="V1 authority is retired",
    ):
        worker_mod.ResearchLabHostedWorker(
            ResearchLabGatewayConfig(), worker_ref="worker-a"
        )


def test_tree_mode_off_still_rejects_legacy_v1(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_TREE_MODE", "off")
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1")

    with pytest.raises(
        worker_mod.HostedResearchLabWorkerError,
        match="V1 authority is retired",
    ):
        worker_mod.ResearchLabHostedWorker(
            ResearchLabGatewayConfig(), worker_ref="worker-a"
        )


@pytest.mark.asyncio
async def test_tree_mode_off_returns_before_queue_lookup(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_TREE_MODE", "off")
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "v2")
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(hosted_worker_dry_run=True),
        worker_ref="worker-a",
    )
    worker._require_enabled = lambda: None

    async def maintenance_state():
        return {"paused": False, "reason": ""}

    async def forbidden_queue_lookup():
        pytest.fail("tree mode off must not inspect or claim queued work")

    monkeypatch.setattr(
        worker_mod, "get_autoresearch_maintenance_state", maintenance_state
    )
    worker._next_queued_run = forbidden_queue_lookup

    outcome = await worker.run_once()

    assert outcome.status == "git_tree_mode_off"
    assert outcome.processed is False


def test_inner_loop_rejects_multiple_paid_finalists():
    candidates = (object(), object(), object())

    with pytest.raises(
        worker_mod.HostedResearchLabWorkerError,
        match="exactly one candidate",
    ):
        worker_mod._single_paid_finalist_candidates(candidates, 1)
    candidate = object()
    assert worker_mod._single_paid_finalist_candidates((candidate,), 1) == (candidate,)


def test_tree_observation_uses_terminal_evaluation_summary():
    summary = {
        "schema_version": "research_lab.git_tree_evaluation_summary.v1",
        "node_count": 6,
        "built_node_count": 4,
        "evaluated_node_count": 4,
        "eligible_node_count": 3,
        "missing_evaluation_count": 0,
        "unclassified_error_count": 0,
        "snapshot_miss_count": 1,
        "true_miss_count": 1,
        "failure_count": 0,
        "zero_output_count": 2,
        "evaluation_mode_counts": {"hybrid": 2, "replay": 2},
        "ineligible_reason_counts": {"snapshot_miss": 1},
        "node_status_counts": {"eligible": 3, "failed": 2, "ineligible": 1},
    }
    observed = worker_mod._tree_observation_from_selection(
        {
            "node_count": 6,
            "built_node_count": 4,
            "eligible_node_count": 3,
            "evaluation_summary": summary,
        }
    )

    assert observed == summary
    assert observed["missing_evaluation_count"] == 0


def test_tree_observation_rejects_inconsistent_summary():
    with pytest.raises(
        worker_mod.HostedResearchLabWorkerError,
        match="counts are inconsistent",
    ):
        worker_mod._tree_observation_from_selection(
            {
                "node_count": 2,
                "built_node_count": 2,
                "eligible_node_count": 1,
                "evaluation_summary": {
                    "schema_version": "research_lab.git_tree_evaluation_summary.v1",
                    "node_count": 2,
                    "built_node_count": 2,
                    "evaluated_node_count": 1,
                    "eligible_node_count": 1,
                    "missing_evaluation_count": 0,
                    "unclassified_error_count": 0,
                    "snapshot_miss_count": 0,
                    "true_miss_count": 0,
                    "failure_count": 0,
                    "zero_output_count": 0,
                    "evaluation_mode_counts": {"replay": 1},
                    "ineligible_reason_counts": {},
                    "node_status_counts": {"eligible": 1, "evaluating": 1},
                },
            }
        )


async def test_tree_evaluation_usage_restores_terminal_cost_and_recovery_evidence(
    monkeypatch,
):
    async def fake_call_rpc(name, params):
        assert name == "research_lab_autoresearch_run_evaluation_usage"
        assert params == {"requested_run_id": RUN_ID}
        return {
            "settled_cost_microusd": 130,
            "provider_call_count": 3,
            "terminal_operation_count": 3,
            "unsettled_operation_ids": ["sha256:" + "3" * 64],
            "indeterminate_operation_ids": ["sha256:" + "4" * 64],
        }

    monkeypatch.setattr(worker_mod, "call_rpc", fake_call_rpc)
    usage = await worker_mod._load_tree_evaluation_usage(run_id=RUN_ID)

    assert usage["settled_cost_microusd"] == 130
    assert usage["provider_call_count"] == 3
    assert usage["terminal_operation_count"] == 3
    assert usage["unsettled_operation_ids"] == ("sha256:" + "3" * 64,)
    assert usage["indeterminate_operation_ids"] == ("sha256:" + "4" * 64,)


def test_tree_evaluator_commitment_requires_immutable_resolved_snapshot(
    hosted_worker,
):
    readiness = {
        "resolved_snapshot_uri": "s3://private-dev/current.json",
        "pointer_hash": "sha256:" + "1" * 64,
    }
    with pytest.raises(
        worker_mod.HostedResearchLabWorkerError,
        match="immutable snapshot URI",
    ):
        worker_mod._tree_evaluator_commitment(
            readiness,
            policy=hosted_worker.tree_policy,
        )

    readiness["resolved_snapshot_uri"] = "s3://private-dev/snapshots/" + "2" * 64
    commitment = worker_mod._tree_evaluator_commitment(
        readiness,
        policy=hosted_worker.tree_policy,
    )
    assert commitment["schema_version"] == (
        "research_lab.git_tree_evaluator_commitment.v3"
    )
    assert commitment["resolved_snapshot_uri"] == readiness["resolved_snapshot_uri"]
    assert commitment["snapshot_pointer_hash"] == readiness["pointer_hash"]


async def test_existing_tree_restores_exact_pinned_evaluator_commitment(
    monkeypatch,
    hosted_worker,
):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )
    commitment = {
        "schema_version": "research_lab.git_tree_evaluator_commitment.v3",
        "resolved_snapshot_uri": "s3://private-dev/snapshots/" + "3" * 64,
        "snapshot_pointer_hash": "sha256:" + "4" * 64,
    }

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        return {
            "tree_id": "sha256:" + "5" * 64,
            "run_id": RUN_ID,
            "tree_generation": 0,
            "replaces_tree_id": None,
            "root_artifact_hash": artifact.model_artifact_hash,
            "root_manifest_hash": artifact.manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "evaluator_commitment_hash": worker_mod.sha256_json(commitment),
            "tree_doc": {"evaluator_commitment": commitment},
            "current_event_type": "checkpoint_committed",
            "current_event_hash": "sha256:" + "6" * 64,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    resolution = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc=None,
    )

    assert resolution.evaluator_commitment == commitment
    assert resolution.requires_evaluator_replacement is False


async def test_legacy_tree_requires_one_replacement_for_snapshot_pinning(
    monkeypatch,
    hosted_worker,
):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        return {
            "tree_id": "sha256:" + "3" * 64,
            "run_id": RUN_ID,
            "tree_generation": 0,
            "replaces_tree_id": None,
            "root_artifact_hash": artifact.model_artifact_hash,
            "root_manifest_hash": artifact.manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "evaluator_commitment_hash": "sha256:" + "4" * 64,
            "tree_doc": {
                "evaluator_commitment": {
                    "schema_version": "research_lab.git_tree_evaluator_commitment.v2"
                }
            },
            "current_event_type": "checkpoint_committed",
            "current_event_hash": "sha256:" + "5" * 64,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    resolution = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc=None,
    )

    assert resolution.evaluator_commitment is None
    assert resolution.requires_evaluator_replacement is True


def test_tree_resume_rejects_pre_tree_and_changed_root_checkpoints(hosted_worker):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "a" * 64,
        manifest_hash="sha256:" + "b" * 64,
    )
    expected_tree_id = "sha256:" + "c" * 64

    assert hosted_worker._tree_resume_block_reason(
        {"artifact_hash": artifact.model_artifact_hash},
        artifact,
        expected_tree_id=expected_tree_id,
    ) == "tree_checkpoint_not_authoritative"
    assert hosted_worker._tree_resume_block_reason(
        {
            "artifact_hash": "sha256:" + "d" * 64,
            "manifest_hash": artifact.manifest_hash,
            "git_tree_checkpoint": {"tree_id": expected_tree_id},
        },
        artifact,
        expected_tree_id=expected_tree_id,
    ) == "tree_checkpoint_root_changed"


def test_tree_resume_classifies_malformed_replacement_authority(hosted_worker):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "a" * 64,
        manifest_hash="sha256:" + "b" * 64,
    )

    assert hosted_worker._tree_resume_block_reason(
        {
            "run_id": RUN_ID,
            "tree_replacement": {
                "schema_version": "research_lab.git_tree_replacement.v1",
                "replacement_hash": "sha256:" + "0" * 64,
            },
        },
        artifact,
        expected_tree_id="sha256:" + "c" * 64,
        expected_policy=hosted_worker.tree_policy,
    ) == "tree_replacement_authority_invalid"


async def test_existing_tree_root_change_is_cancelled_and_replaced(
    monkeypatch, hosted_worker
):
    old_tree_id = "sha256:" + "1" * 64
    old_artifact_hash = "sha256:" + "3" * 64
    old_manifest_hash = "sha256:" + "6" * 64
    new_artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "4" * 64,
        manifest_hash="sha256:" + "5" * 64,
    )
    recorded = []

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        assert filters == (("run_id", RUN_ID),)
        return {
            "tree_id": old_tree_id,
            "run_id": RUN_ID,
            "tree_generation": 0,
            "replaces_tree_id": None,
            "root_artifact_hash": old_artifact_hash,
            "root_manifest_hash": old_manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "tree_doc": {},
            "current_event_type": "checkpoint_committed",
            "current_event_hash": "sha256:" + "7" * 64,
        }

    class FakeTreeStore:
        async def append_event_next(self, **kwargs):
            recorded.append(kwargs)
            return {
                "event_hash": "sha256:" + "8" * 64,
                "event_doc": kwargs["event_doc"],
            }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "GitTreeStore", FakeTreeStore)

    resolution = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=new_artifact,
        checkpoint_doc={
            "elapsed_seconds": 12.5,
            "actual_openrouter_cost_microusd": 123,
            "built_candidate_count": 4,
        },
    )

    assert resolution.outcome is None
    assert resolution.tree_id != old_tree_id
    assert resolution.replacement is not None
    assert resolution.replacement.generation == 1
    assert resolution.replacement.replaces_tree_id == old_tree_id
    assert resolution.replacement.root_artifact_hash == (
        new_artifact.model_artifact_hash
    )
    assert resolution.replacement.root_manifest_hash == new_artifact.manifest_hash
    assert resolution.resume_state["elapsed_seconds"] == 12.5
    assert resolution.resume_state["actual_openrouter_cost_microusd"] == 123
    assert resolution.resume_state["built_candidate_count"] == 0
    assert "git_tree_checkpoint" not in resolution.resume_state
    assert recorded[0]["tree_id"] == old_tree_id
    assert recorded[0]["event_type"] == "tree_cancelled_root_changed"
    assert recorded[0]["event_doc"]["new_root_artifact_hash"] == (
        new_artifact.model_artifact_hash
    )
    assert hosted_worker._tree_resume_block_reason(
        resolution.resume_state,
        new_artifact,
        expected_tree_id=resolution.tree_id,
        expected_policy=hosted_worker.tree_policy,
    ) == ""


async def test_cancelled_tree_replacement_is_deterministic_across_retries(
    monkeypatch, hosted_worker
):
    old_tree_id = "sha256:" + "1" * 64
    cancellation_hash = "sha256:" + "2" * 64
    old_artifact_hash = "sha256:" + "3" * 64
    old_manifest_hash = "sha256:" + "4" * 64
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "5" * 64,
        manifest_hash="sha256:" + "6" * 64,
    )
    cancellation_doc = {
        "new_root_artifact_hash": artifact.model_artifact_hash,
        "new_root_manifest_hash": artifact.manifest_hash,
        "new_policy_hash": hosted_worker.tree_policy.policy_hash,
    }

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        return {
            "tree_id": old_tree_id,
            "run_id": RUN_ID,
            "tree_generation": 0,
            "replaces_tree_id": None,
            "root_artifact_hash": old_artifact_hash,
            "root_manifest_hash": old_manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "tree_doc": {},
            "current_event_type": "tree_cancelled_root_changed",
            "current_event_doc": cancellation_doc,
            "current_event_hash": cancellation_hash,
        }

    class ForbiddenTreeStore:
        async def append_event_next(self, **kwargs):
            pytest.fail("an existing cancellation must not be appended twice")

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "GitTreeStore", ForbiddenTreeStore)

    first = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc={"openrouter_call_count": 2},
    )
    second = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc={"openrouter_call_count": 2},
    )

    assert first.tree_id == second.tree_id
    assert first.replacement == second.replacement
    assert first.resume_state["openrouter_call_count"] == 2
    assert first.resume_state["iterations_completed"] == 0


async def test_replacement_tree_discards_predecessor_checkpoint_nodes(
    monkeypatch, hosted_worker
):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "4" * 64,
        manifest_hash="sha256:" + "5" * 64,
    )
    replacement = worker_mod.TreeReplacement(
        generation=1,
        replaces_tree_id="sha256:" + "1" * 64,
        cancellation_event_hash="sha256:" + "2" * 64,
        prior_root_artifact_hash="sha256:" + "3" * 64,
        prior_root_manifest_hash="sha256:" + "6" * 64,
        prior_policy_hash=hosted_worker.tree_policy.policy_hash,
        root_artifact_hash=artifact.model_artifact_hash,
        root_manifest_hash=artifact.manifest_hash,
        policy_hash=hosted_worker.tree_policy.policy_hash,
    )
    tree_id = worker_mod.derive_tree_id(
        run_id=RUN_ID,
        root_artifact_hash=artifact.model_artifact_hash,
        policy=hosted_worker.tree_policy,
        replacement=replacement,
    )

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        return {
            "tree_id": tree_id,
            "run_id": RUN_ID,
            "tree_generation": 1,
            "replaces_tree_id": replacement.replaces_tree_id,
            "root_artifact_hash": artifact.model_artifact_hash,
            "root_manifest_hash": artifact.manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "tree_doc": {"replacement": replacement.to_dict()},
            "current_event_type": "tree_created",
            "current_event_hash": "sha256:" + "7" * 64,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    resolution = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc={
            "elapsed_seconds": 20,
            "git_tree_checkpoint": {
                "tree_id": replacement.replaces_tree_id,
                "nodes": [{"node_id": "stale"}],
            },
        },
    )

    assert resolution.tree_id == tree_id
    assert resolution.replacement == replacement
    assert resolution.resume_state["elapsed_seconds"] == 20
    assert "git_tree_checkpoint" not in resolution.resume_state
    assert hosted_worker._tree_resume_block_reason(
        resolution.resume_state,
        artifact,
        expected_tree_id=tree_id,
        expected_policy=hosted_worker.tree_policy,
    ) == ""


async def test_active_private_model_change_is_detected_from_authoritative_lineage(
    monkeypatch, hosted_worker
):
    old_artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )
    new_artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "3" * 64,
        manifest_hash="sha256:" + "4" * 64,
    )
    loaded = SimpleNamespace(artifact=new_artifact)

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_private_model_version_current"
        assert filters == (("current_version_status", "active"),)
        return {
            "private_model_version_id": "private_model_version:new",
            "model_artifact_hash": new_artifact.model_artifact_hash,
            "private_model_manifest_hash": new_artifact.manifest_hash,
            "current_version_status": "active",
        }

    async def fake_load_active_private_model(config, *, register_bootstrap=False):
        assert config is hosted_worker.config
        assert register_bootstrap is False
        return loaded

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(
        worker_mod,
        "load_active_private_model",
        fake_load_active_private_model,
    )

    assert (
        await hosted_worker._active_private_model_if_changed(old_artifact)
        is loaded
    )


async def test_current_private_model_root_avoids_manifest_reload(
    monkeypatch, hosted_worker
):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_private_model_version_current"
        return {
            "private_model_version_id": "private_model_version:current",
            "model_artifact_hash": artifact.model_artifact_hash,
            "private_model_manifest_hash": artifact.manifest_hash,
            "current_version_status": "active",
        }

    async def forbidden_load(*_args, **_kwargs):
        pytest.fail("an unchanged active root must not reload its manifest")

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "load_active_private_model", forbidden_load)

    assert await hosted_worker._active_private_model_if_changed(artifact) is None


async def test_inflight_root_change_requeues_replacement_instead_of_pausing(
    monkeypatch, hosted_worker
):
    context = _make_context(receipt_id="55555555-5555-4555-8555-555555555555")
    context.ticket = {
        **dict(context.ticket),
        "miner_openrouter_key_ref": "test-key-ref",
    }
    prior_artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )
    new_artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "3" * 64,
        manifest_hash="sha256:" + "4" * 64,
    )
    replacement = worker_mod.TreeReplacement(
        generation=1,
        replaces_tree_id="sha256:" + "5" * 64,
        cancellation_event_hash="sha256:" + "6" * 64,
        prior_root_artifact_hash=prior_artifact.model_artifact_hash,
        prior_root_manifest_hash=prior_artifact.manifest_hash,
        prior_policy_hash=hosted_worker.tree_policy.policy_hash,
        root_artifact_hash=new_artifact.model_artifact_hash,
        root_manifest_hash=new_artifact.manifest_hash,
        policy_hash=hosted_worker.tree_policy.policy_hash,
    )
    replacement_tree_id = worker_mod.derive_tree_id(
        run_id=RUN_ID,
        root_artifact_hash=new_artifact.model_artifact_hash,
        policy=hosted_worker.tree_policy,
        replacement=replacement,
    )
    recorded = []

    async def fake_resolve_tree_authority(**kwargs):
        assert kwargs["artifact"] is new_artifact
        assert kwargs["change_reason"] == "active_private_model_changed_during_tree"
        return worker_mod.TreeAuthorityResolution(
            tree_id=replacement_tree_id,
            resume_state={},
            replacement=replacement,
        )

    async def capture(name, **kwargs):
        recorded.append((name, kwargs))
        return {}

    monkeypatch.setattr(
        hosted_worker,
        "_resolve_tree_authority",
        fake_resolve_tree_authority,
    )
    monkeypatch.setattr(
        worker_mod,
        "create_receipt_event",
        lambda **kwargs: capture("receipt", **kwargs),
    )
    monkeypatch.setattr(
        worker_mod,
        "create_queue_event",
        lambda **kwargs: capture("queue", **kwargs),
    )
    monkeypatch.setattr(
        worker_mod,
        "create_ticket_event",
        lambda **kwargs: capture("ticket", **kwargs),
    )
    monkeypatch.setattr(
        worker_mod,
        "safe_project_public_loop_activity",
        lambda *args, **kwargs: capture("public", args=args, **kwargs),
    )
    loop_result = SimpleNamespace(
        iterations_completed=2,
        elapsed_seconds=31.25,
        stop_reason="maintenance_pause_requested",
        provider_usage=(),
        cost_ledger=lambda: {
            "schema_version": "1.0",
            "actual_openrouter_cost_microusd": 123,
        },
    )

    outcome = await hosted_worker._requeue_tree_root_replacement(
        context=context,
        prior_artifact=prior_artifact,
        active_model=SimpleNamespace(artifact=new_artifact),
        loop_result=loop_result,
        checkpoint_doc={"checkpoint_hash": "sha256:" + "7" * 64},
        change_reason="active_private_model_changed_during_tree",
    )

    assert outcome.status == "git_tree_root_replaced_requeued"
    assert [name for name, _kwargs in recorded] == [
        "receipt",
        "queue",
        "ticket",
        "public",
    ]
    queue_event = recorded[1][1]
    assert queue_event["event_type"] == "queued"
    assert queue_event["reason"] == "git_tree_root_replaced_requeued"
    assert queue_event["event_doc"]["replacement_tree_id"] == replacement_tree_id
    assert queue_event["event_doc"]["replacement"] == replacement.to_dict()
    assert all(
        kwargs.get("event_type") != "paused" for _name, kwargs in recorded
    )


async def test_atomic_handoff_root_race_requeues_without_generic_retry_budget(
    monkeypatch, hosted_worker
):
    context = _make_context(receipt_id="55555555-5555-4555-8555-555555555555")
    prior_artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )
    active_model = SimpleNamespace(
        artifact=SimpleNamespace(
            model_artifact_hash="sha256:" + "3" * 64,
            manifest_hash="sha256:" + "4" * 64,
        )
    )
    expected = worker_mod.HostedWorkerOutcome(
        processed=True,
        dry_run=False,
        run_id=RUN_ID,
        ticket_id=TICKET_ID,
        status="git_tree_root_replaced_requeued",
    )
    calls = []

    async def fail_atomic_handoff(label, operation, *, attempts=3):
        calls.append(("store", label, attempts))
        raise RuntimeError("research_lab_git_tree_handoff_stale_active_root")

    async def changed_root(artifact):
        calls.append(("active_root", artifact))
        return active_model

    async def requeue(**kwargs):
        calls.append(("requeue", kwargs))
        return expected

    monkeypatch.setattr(
        hosted_worker,
        "_store_write_with_retry",
        fail_atomic_handoff,
    )
    monkeypatch.setattr(
        hosted_worker,
        "_active_private_model_if_changed",
        changed_root,
    )
    monkeypatch.setattr(
        hosted_worker,
        "_requeue_tree_root_replacement",
        requeue,
    )
    loop_result = SimpleNamespace(checkpoint_doc={"checkpoint_hash": "sha256:" + "5" * 64})

    outcome = await hosted_worker._create_candidate_with_root_fence(
        context=context,
        request=SimpleNamespace(),
        final_artifact=prior_artifact,
        loop_result=loop_result,
        checkpoint_doc=loop_result.checkpoint_doc,
    )

    assert outcome is expected
    assert calls[0] == ("store", "candidate_artifact_create", 3)
    assert calls[1] == ("active_root", prior_artifact)
    assert calls[2][0] == "requeue"
    assert calls[2][1]["prior_artifact"] is prior_artifact
    assert calls[2][1]["active_model"] is active_model
    assert calls[2][1]["change_reason"] == (
        "active_private_model_changed_during_handoff"
    )


async def test_atomic_handoff_non_root_error_is_not_reclassified(
    monkeypatch, hosted_worker
):
    async def fail_store(*_args, **_kwargs):
        raise RuntimeError("candidate artifact persistence failed")

    async def forbidden_root_check(*_args, **_kwargs):
        pytest.fail("non-root errors must not enter root replacement")

    monkeypatch.setattr(hosted_worker, "_store_write_with_retry", fail_store)
    monkeypatch.setattr(
        hosted_worker,
        "_active_private_model_if_changed",
        forbidden_root_check,
    )

    with pytest.raises(RuntimeError, match="candidate artifact persistence failed"):
        await hosted_worker._create_candidate_with_root_fence(
            context=_make_context(),
            request=SimpleNamespace(),
            final_artifact=SimpleNamespace(),
            loop_result=SimpleNamespace(),
            checkpoint_doc=None,
        )


async def test_existing_no_finalist_tree_becomes_terminal_run_failure(
    monkeypatch, hosted_worker
):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "3" * 64,
        manifest_hash="sha256:" + "4" * 64,
    )
    tree_id = "sha256:" + "5" * 64

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        return {
            "tree_id": tree_id,
            "run_id": RUN_ID,
            "root_artifact_hash": artifact.model_artifact_hash,
            "root_manifest_hash": artifact.manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "current_event_type": "tree_failed",
            "current_event_hash": "sha256:" + "6" * 64,
        }

    async def fake_mark_failed(context, error, **kwargs):
        assert context.run_id == RUN_ID
        assert "without an eligible finalist" in error
        assert kwargs["reason"] == "git_tree_no_eligible_finalist"
        return worker_mod.HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            status=kwargs["reason"],
        )

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(hosted_worker, "_mark_failed", fake_mark_failed)

    resolution = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc=None,
    )

    assert resolution.outcome is not None
    assert resolution.outcome.status == "git_tree_no_eligible_finalist"


async def test_final_selected_tree_without_candidate_resumes(monkeypatch, hosted_worker):
    artifact = SimpleNamespace(
        model_artifact_hash="sha256:" + "3" * 64,
        manifest_hash="sha256:" + "4" * 64,
    )
    tree_id = "sha256:" + "5" * 64

    async def fake_select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_run_tree_current"
        return {
            "tree_id": tree_id,
            "run_id": RUN_ID,
            "root_artifact_hash": artifact.model_artifact_hash,
            "root_manifest_hash": artifact.manifest_hash,
            "policy_hash": hosted_worker.tree_policy.policy_hash,
            "current_event_type": "final_selected",
            "current_event_hash": "sha256:" + "6" * 64,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)

    resolution = await hosted_worker._resolve_tree_authority(
        context=_make_context(),
        artifact=artifact,
        checkpoint_doc=None,
    )
    assert resolution.outcome is None
    assert resolution.tree_id == tree_id


async def test_candidate_insert_without_handoff_is_completed_on_resume(
    monkeypatch, hosted_worker
):
    tree_id = "sha256:" + "1" * 64
    node_id = "tree-node:" + "2" * 64
    candidate_id = "candidate:" + "3" * 64
    artifact_hash = "sha256:" + "7" * 64
    manifest_hash = "sha256:" + "8" * 64
    context = _make_context(receipt_id="receipt-1")
    calls = []

    async def fake_select_many(table, **kwargs):
        assert table == "research_lab_autoresearch_tree_handoffs"
        return []

    async def fake_candidate_rows(run_id):
        assert run_id == RUN_ID
        return [
            {
                "candidate_id": candidate_id,
                "git_tree_id": tree_id,
                "git_tree_node_id": node_id,
                "git_tree_root_commit": "4" * 64,
                "git_tree_node_commit": "5" * 64,
                "git_tree_lineage_hash": "sha256:" + "6" * 64,
            }
        ]

    async def fake_select_one(table, **kwargs):
        if table == "research_lab_autoresearch_run_tree_current":
            return {
                "tree_id": tree_id,
                "root_artifact_hash": artifact_hash,
                "root_manifest_hash": manifest_hash,
                "current_event_type": "final_selected",
            }
        if table == "research_lab_private_model_version_current":
            return {
                "model_artifact_hash": artifact_hash,
                "private_model_manifest_hash": manifest_hash,
                "current_version_status": "active",
            }
        raise AssertionError(table)

    async def fake_handoff(**kwargs):
        calls.append(("handoff", kwargs))

    async def fake_queue_event(**kwargs):
        calls.append(("queue", kwargs))

    async def fake_receipt_event(**kwargs):
        calls.append(("receipt", kwargs))

    async def fake_projection(*args, **kwargs):
        calls.append(("projection", kwargs))

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)
    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(hosted_worker, "_candidate_rows_for_run", fake_candidate_rows)
    monkeypatch.setattr(worker_mod, "record_candidate_tree_handoff", fake_handoff)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_queue_event)
    monkeypatch.setattr(worker_mod, "create_receipt_event", fake_receipt_event)
    monkeypatch.setattr(hosted_worker, "_ensure_terminal_loop_projection", fake_projection)

    outcome = await hosted_worker._complete_from_existing_candidate_artifacts(context)

    assert outcome is not None
    assert outcome.candidate_ids == (candidate_id,)
    assert [name for name, _kwargs in calls] == [
        "handoff",
        "queue",
        "projection",
        "receipt",
    ]
    assert calls[0][1]["tree_id"] == tree_id
    assert calls[0][1]["node_id"] == node_id


async def test_stale_generation_candidate_is_ignored_during_replacement_recovery(
    monkeypatch, hosted_worker
):
    stale_tree_id = "sha256:" + "1" * 64
    replacement_tree_id = "sha256:" + "2" * 64
    artifact_hash = "sha256:" + "3" * 64
    manifest_hash = "sha256:" + "4" * 64
    context = _make_context(receipt_id="receipt-1")

    async def fake_select_many(table, **kwargs):
        assert table == "research_lab_autoresearch_tree_handoffs"
        return []

    async def fake_candidate_rows(run_id):
        assert run_id == RUN_ID
        return [
            {
                "candidate_id": "candidate:" + "5" * 64,
                "git_tree_id": stale_tree_id,
                "git_tree_node_id": "tree-node:" + "6" * 64,
                "git_tree_root_commit": "7" * 64,
                "git_tree_node_commit": "8" * 64,
                "git_tree_lineage_hash": "sha256:" + "9" * 64,
            }
        ]

    async def fake_select_one(table, **kwargs):
        if table == "research_lab_autoresearch_run_tree_current":
            return {
                "tree_id": replacement_tree_id,
                "root_artifact_hash": artifact_hash,
                "root_manifest_hash": manifest_hash,
                "current_event_type": "tree_created",
            }
        if table == "research_lab_private_model_version_current":
            return {
                "model_artifact_hash": artifact_hash,
                "private_model_manifest_hash": manifest_hash,
                "current_version_status": "active",
            }
        raise AssertionError(table)

    async def fail_handoff(**kwargs):
        raise AssertionError("stale candidate must not be handed off")

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)
    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(hosted_worker, "_candidate_rows_for_run", fake_candidate_rows)
    monkeypatch.setattr(worker_mod, "record_candidate_tree_handoff", fail_handoff)

    outcome = await hosted_worker._complete_from_existing_candidate_artifacts(context)

    assert outcome is None


async def test_candidate_recovery_waits_for_root_replacement_when_model_changed(
    monkeypatch, hosted_worker
):
    tree_id = "sha256:" + "1" * 64
    context = _make_context(receipt_id="receipt-1")

    async def fake_select_many(table, **kwargs):
        assert table == "research_lab_autoresearch_tree_handoffs"
        return []

    async def fake_candidate_rows(run_id):
        assert run_id == RUN_ID
        return [
            {
                "candidate_id": "candidate:" + "2" * 64,
                "git_tree_id": tree_id,
                "git_tree_node_id": "tree-node:" + "3" * 64,
                "git_tree_root_commit": "4" * 64,
                "git_tree_node_commit": "5" * 64,
                "git_tree_lineage_hash": "sha256:" + "6" * 64,
            }
        ]

    async def fake_select_one(table, **kwargs):
        if table == "research_lab_autoresearch_run_tree_current":
            return {
                "tree_id": tree_id,
                "root_artifact_hash": "sha256:" + "7" * 64,
                "root_manifest_hash": "sha256:" + "8" * 64,
                "current_event_type": "final_selected",
            }
        if table == "research_lab_private_model_version_current":
            return {
                "model_artifact_hash": "sha256:" + "9" * 64,
                "private_model_manifest_hash": "sha256:" + "a" * 64,
                "current_version_status": "active",
            }
        raise AssertionError(table)

    async def fail_handoff(**kwargs):
        raise AssertionError("candidate from a stale active root must not be handed off")

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)
    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(hosted_worker, "_candidate_rows_for_run", fake_candidate_rows)
    monkeypatch.setattr(worker_mod, "record_candidate_tree_handoff", fail_handoff)

    outcome = await hosted_worker._complete_from_existing_candidate_artifacts(context)

    assert outcome is None


def _make_context(queue_events=(), receipt_id=None):
    queue_row = {
        "run_id": RUN_ID,
        "ticket_id": TICKET_ID,
        "queue_priority": 0,
        "current_event_hash": "sha256:" + "a" * 64,
        "current_status_at": "2026-07-01T00:00:00+00:00",
    }
    ticket = {
        "miner_hotkey": MINER_HOTKEY,
        "island": "generalist",
        "requested_loop_count": 1,
        "ticket_doc": {},
    }
    return worker_mod.HostedRunContext(
        queue_row=queue_row,
        ticket=ticket,
        payment=None,
        queue_events=tuple(queue_events),
        receipt_id=receipt_id,
    )


def _queue_event(seq, event_type, event_doc):
    return {"run_id": RUN_ID, "seq": seq, "event_type": event_type, "event_doc": event_doc}


async def test_ticket_reconciliation_runs_periodically_on_configured_worker(monkeypatch):
    calls = []

    async def fake_reconcile_terminal_ticket_statuses(**kwargs):
        calls.append(kwargs)
        return {"planned_count": 1, "repaired_count": 1, "skipped_count": 0}

    monkeypatch.setattr(
        worker_mod,
        "reconcile_terminal_ticket_statuses",
        fake_reconcile_terminal_ticket_statuses,
    )
    config = ResearchLabGatewayConfig(
        hosted_worker_index=0,
        hosted_worker_total_workers=3,
        ticket_reconciliation_worker_index=0,
        ticket_reconciliation_interval_seconds=300,
        ticket_reconciliation_limit=7,
    )
    worker = worker_mod.ResearchLabHostedWorker(config, worker_ref="ticket-reconciler")

    await worker._maybe_reconcile_terminal_tickets()
    await worker._maybe_reconcile_terminal_tickets()

    assert len(calls) == 1
    assert calls[0]["limit"] == 7
    assert calls[0]["dry_run"] is False
    assert calls[0]["actor_ref"] == "ticket-reconciler"
    assert calls[0]["reason"] == "hosted_worker_periodic_ticket_lifecycle_reconciler"


async def test_ticket_reconciliation_skips_non_configured_worker(monkeypatch):
    calls = []

    async def fake_reconcile_terminal_ticket_statuses(**kwargs):
        calls.append(kwargs)
        return {"planned_count": 1, "repaired_count": 1, "skipped_count": 0}

    monkeypatch.setattr(
        worker_mod,
        "reconcile_terminal_ticket_statuses",
        fake_reconcile_terminal_ticket_statuses,
    )
    config = ResearchLabGatewayConfig(
        hosted_worker_index=1,
        hosted_worker_total_workers=3,
        ticket_reconciliation_worker_index=0,
    )
    worker = worker_mod.ResearchLabHostedWorker(config, worker_ref="ordinary-worker")

    await worker._maybe_reconcile_terminal_tickets()

    assert calls == []


def _own_started_row():
    return {
        "run_id": RUN_ID,
        "current_queue_status": "started",
        "worker_ref": "worker-a",
        "current_event_hash": "sha256:" + "b" * 64,
        "current_event_seq": 5,
        "queue_priority": 0,
    }


def test_postgrest_short_fraction_timestamps_count_as_stale():
    base_status_at = (
        datetime.now(timezone.utc) - timedelta(minutes=20)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-1]
    values = (
        base_status_at + "+00:00",
        base_status_at + "+0000",
        base_status_at + "+00",
        (base_status_at + "+00:00").replace("T", " ", 1),
    )

    for value in values:
        assert worker_mod._status_age_seconds(value) > 19 * 60
        assert worker_mod._status_is_stale(value, 300)
        assert maintenance_mod._parse_iso(value) is not None
        assert maintenance_mod._status_is_stale(value, 300)


# ---------------------------------------------------------------------------
# Bug #1 — scripts/59 claim guard predicate (heartbeat allowance)
# ---------------------------------------------------------------------------


def _guard_allows(*, new_event_type, new_worker_ref, latest_status, latest_worker_ref):
    """Pure-Python re-implementation of the scripts/59 guard decision.

    Mirrors public.guard_research_lab_run_claim after the heartbeat fix:
    non-`started` inserts pass; `started` passes when the latest event is
    `queued` (`IS DISTINCT FROM 'queued'` false), or when the latest event is
    `started` with the same non-null worker_ref (same-worker heartbeat).
    Everything else conflicts, exactly as in scripts/42.
    """
    if new_event_type != "started":
        return True
    if latest_status == "queued":
        return True
    if (
        latest_status == "started"
        and new_worker_ref is not None
        and latest_worker_ref is not None
        and new_worker_ref == latest_worker_ref
    ):
        return True
    return False


@pytest.mark.parametrize(
    ("new_event_type", "new_worker_ref", "latest_status", "latest_worker_ref", "allowed"),
    (
        # Non-claim inserts are never guarded (unchanged).
        ("queued", None, "started", "worker-a", True),
        ("paused", "worker-a", "started", "worker-a", True),
        ("failed", "worker-a", "started", "worker-a", True),
        # Fresh claim: latest is queued (unchanged).
        ("started", "worker-a", "queued", None, True),
        ("started", "worker-a", "queued", "worker-z", True),
        # First event of a run cannot be started (unchanged: NULL latest conflicts).
        ("started", "worker-a", None, None, False),
        # NEW: same-worker heartbeat while started is latest.
        ("started", "worker-a", "started", "worker-a", True),
        # Cross-worker fencing preserved: after another worker re-claims, the
        # superseded worker's heartbeat still conflicts.
        ("started", "worker-a", "started", "worker-b", False),
        # A worker_ref-less event never grants heartbeat rights (NULL <> NULL).
        ("started", None, "started", None, False),
        ("started", "worker-a", "started", None, False),
        ("started", None, "started", "worker-a", False),
        # Every other latest status still conflicts (unchanged).
        ("started", "worker-a", "paused", "worker-a", False),
        ("started", "worker-a", "completed", "worker-a", False),
        ("started", "worker-a", "failed", "worker-a", False),
        ("started", "worker-a", "cancelled", "worker-a", False),
    ),
)
def test_guard_predicate(new_event_type, new_worker_ref, latest_status, latest_worker_ref, allowed):
    assert (
        _guard_allows(
            new_event_type=new_event_type,
            new_worker_ref=new_worker_ref,
            latest_status=latest_status,
            latest_worker_ref=latest_worker_ref,
        )
        is allowed
    )


def test_guard_sql_contains_same_worker_allowance_and_keeps_rejections():
    sql = GUARD_SQL_PATH.read_text(encoding="utf-8")
    # The migration must redefine the run claim guard only.
    assert "CREATE OR REPLACE FUNCTION public.guard_research_lab_run_claim()" in sql
    assert "guard_research_lab_candidate_claim" not in sql
    assert "guard_research_lab_credit_consume" not in sql
    # Same-worker heartbeat allowance.
    assert "latest_status = 'started'" in sql
    assert "NEW.worker_ref = latest_worker_ref" in sql
    assert "NEW.worker_ref IS NOT NULL" in sql
    assert "latest_worker_ref IS NOT NULL" in sql
    # Every other rejection is unchanged.
    assert "IF latest_status IS DISTINCT FROM 'queued' THEN" in sql
    assert "research_lab_run_claim_conflict" in sql
    assert "USING ERRCODE = '23505'" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "IF NEW.event_type <> 'started' THEN" in sql


# ---------------------------------------------------------------------------
# Bug #1 — worker-side heartbeat claim-lost abort
# ---------------------------------------------------------------------------


async def test_heartbeat_success_appends_started_event(hosted_worker, monkeypatch):
    context = _make_context()

    async def fake_select_one(table, **kwargs):
        return _own_started_row()

    inserts = []

    async def fake_create_queue_event(**kwargs):
        inserts.append(kwargs)
        return {"seq": 6, "anchored_hash": "sha256:" + "c" * 64}

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)

    assert await hosted_worker._append_queue_heartbeat(
        context, source_event_type="build", source_event_seq=None, source_event_hash=None
    )
    assert len(inserts) == 1
    # The heartbeat is a `started` event with our worker_ref; the queue view's
    # current_status_at derives from the latest event's created_at
    # (scripts/29 research_loop_run_queue_current), so a landed heartbeat
    # refreshes it — and scripts/54 capacity counting — automatically.
    assert inserts[0]["event_type"] == "started"
    assert inserts[0]["reason"] == "hosted_worker_heartbeat"
    assert inserts[0]["worker_ref"] == "worker-a"
    assert context.claim_lost is False


async def test_heartbeat_foreign_started_owner_raises_claim_lost(hosted_worker, monkeypatch):
    context = _make_context()

    async def fake_select_one(table, **kwargs):
        return {**_own_started_row(), "worker_ref": "worker-b"}

    inserts = []

    async def fake_create_queue_event(**kwargs):
        inserts.append(kwargs)
        return {}

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)

    with pytest.raises(worker_mod.HostedResearchLabClaimLost):
        await hosted_worker._append_queue_heartbeat(
            context, source_event_type="build", source_event_seq=None, source_event_hash=None
        )
    assert context.claim_lost is True
    # Clean local abort: no heartbeat insert, no terminal event write — the new
    # claimant owns the run now.
    assert inserts == []


async def test_heartbeat_non_started_status_still_skips_quietly(hosted_worker, monkeypatch):
    context = _make_context()

    async def fake_select_one(table, **kwargs):
        return {**_own_started_row(), "current_queue_status": "paused"}

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)

    assert not await hosted_worker._append_queue_heartbeat(
        context, source_event_type="build", source_event_seq=None, source_event_hash=None
    )
    assert context.claim_lost is False


async def test_heartbeat_guard_conflict_claim_lost_when_enabled(hosted_worker, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_HEARTBEAT_CONFLICT_CLAIM_LOST_ENABLED", "true")
    context = _make_context()

    async def fake_select_one(table, **kwargs):
        return _own_started_row()

    async def fake_create_queue_event(**kwargs):
        raise RuntimeError(
            "research_lab_run_claim_conflict: run "
            f"{RUN_ID} latest status started"
        )

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)

    with pytest.raises(worker_mod.HostedResearchLabClaimLost):
        await hosted_worker._append_queue_heartbeat(
            context, source_event_type="build", source_event_seq=None, source_event_hash=None
        )
    assert context.claim_lost is True


async def test_heartbeat_guard_conflict_default_keeps_current_behavior(hosted_worker, monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_HEARTBEAT_CONFLICT_CLAIM_LOST_ENABLED", raising=False)
    context = _make_context()

    async def fake_select_one(table, **kwargs):
        return _own_started_row()

    async def fake_create_queue_event(**kwargs):
        raise RuntimeError("research_lab_run_claim_conflict: run x latest status started")

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)

    assert not await hosted_worker._append_queue_heartbeat(
        context, source_event_type="build", source_event_seq=None, source_event_hash=None
    )
    assert context.claim_lost is False


async def test_heartbeat_non_conflict_insert_error_still_warns_and_skips(hosted_worker, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_HEARTBEAT_CONFLICT_CLAIM_LOST_ENABLED", "true")
    context = _make_context()

    async def fake_select_one(table, **kwargs):
        return _own_started_row()

    async def fake_create_queue_event(**kwargs):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)

    assert not await hosted_worker._append_queue_heartbeat(
        context, source_event_type="build", source_event_seq=None, source_event_hash=None
    )
    assert context.claim_lost is False


async def test_terminal_loop_projection_carries_failed_run_cost_ledger(hosted_worker, monkeypatch):
    context = _make_context(receipt_id="55555555-5555-4555-8555-555555555555")
    captured = []

    async def fake_select_one(table, **kwargs):
        assert table == "research_lab_auto_research_loop_current"
        return None

    async def fake_create_auto_research_loop_event(**kwargs):
        captured.append(kwargs)
        return {"seq": 7, "anchored_hash": "sha256:" + "e" * 64}

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(worker_mod, "create_auto_research_loop_event", fake_create_auto_research_loop_event)

    await hosted_worker._ensure_terminal_loop_projection(
        context,
        event_type="loop_failed",
        loop_status="failed",
        reason="system_error_after_spend",
        event_doc={
            "final_cost_ledger": {
                "schema_version": "1.0",
                "status": "failed",
                "actual_openrouter_cost_microusd": 123456,
                "actual_openrouter_cost_usd": 0.123456,
                "total_usd": 0.123456,
            },
            "provider_usage": [{"provider": "openrouter", "response_id": "spent-before-error"}],
        },
    )

    assert len(captured) == 1
    assert captured[0]["event_type"] == "loop_failed"
    assert captured[0]["cost_ledger"]["actual_openrouter_cost_microusd"] == 123456
    assert captured[0]["provider_usage"][0]["response_id"] == "spent-before-error"


def test_failure_exception_cost_evidence_merges_unpersisted_spend():
    class CostedFailure(RuntimeError):
        cost_microusd = 2500
        provider_usage = {"provider": "openrouter", "response_id": "failed-generation"}

    evidence = worker_mod._merge_failure_exception_cost_evidence(
        {
            "source": "loop_event",
            "trusted_cost_ledger": True,
            "cost_ledger": {
                "schema_version": "1.0",
                "actual_openrouter_cost_microusd": 1000,
                "actual_openrouter_cost_usd": 0.001,
                "total_usd": 0.001,
                "openrouter_call_count": 1,
            },
            "provider_usage": [{"provider": "openrouter", "response_id": "persisted-generation"}],
        },
        CostedFailure("provider charged before failure"),
    )

    assert evidence["actual_openrouter_cost_microusd"] == 3500
    assert evidence["cost_ledger"]["failure_exception_cost_microusd"] == 2500
    assert evidence["provider_usage"][0]["response_id"] == "persisted-generation"
    assert evidence["provider_usage"][1]["response_id"] == "failed-generation"


async def test_to_thread_heartbeat_claim_lost_aborts_before_running_phase(hosted_worker, monkeypatch):
    context = _make_context()

    async def raising_heartbeat(ctx, **kwargs):
        ctx.claim_lost = True
        raise worker_mod.HostedResearchLabClaimLost("run stolen")

    monkeypatch.setattr(hosted_worker, "_append_queue_heartbeat", raising_heartbeat)
    ran = []

    def phase():
        ran.append(1)
        return "done"

    with pytest.raises(worker_mod.HostedResearchLabClaimLost):
        await hosted_worker._to_thread_with_queue_heartbeat(
            context, heartbeat_label="private_runtime_metadata", func=phase
        )
    assert ran == []
    assert context.claim_lost is True


# ---------------------------------------------------------------------------
# Bug #27 — stale-paused reaper must not revive blocked_for_credit
# ---------------------------------------------------------------------------


async def test_stale_paused_reaper_skips_blocked_for_credit(hosted_worker, monkeypatch):
    stale_at = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()

    def _paused_row(run_id, reason):
        return {
            "run_id": run_id,
            "ticket_id": TICKET_ID,
            "current_queue_status": "paused",
            "current_reason": reason,
            "current_status_at": stale_at,
            "current_event_hash": "sha256:" + "d" * 64,
            "queue_priority": 0,
            "worker_ref": "worker-z",
        }

    rows = [
        _paused_row("11111111-aaaa-4aaa-8aaa-111111111111", "blocked_for_credit"),
        _paused_row("22222222-bbbb-4bbb-8bbb-222222222222", "maintenance_pause_stale_started"),
    ]
    captured = {}

    async def fake_select_many(table, **kwargs):
        captured["table"] = table
        captured.update(kwargs)
        return rows

    async def fake_blocks(run_id, stale_after_seconds):
        return False

    requeued = []

    async def fake_create_queue_event(**kwargs):
        requeued.append(kwargs)
        return {}

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)
    monkeypatch.setattr(hosted_worker, "_loop_activity_blocks_stale_requeue", fake_blocks)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)

    recovered = await hosted_worker._recover_stale_paused_runs()
    assert recovered == 1
    assert [event["run_id"] for event in requeued] == ["22222222-bbbb-4bbb-8bbb-222222222222"]
    # The reaper must also fetch current_reason to make the exclusion decision.
    assert "current_reason" in captured["columns"]
    # P2: oldest-first so the stalest rows are seen under backlog.
    assert captured["order_by"] == (("current_status_at", False),)


async def test_stale_started_reaper_scans_oldest_first(hosted_worker, monkeypatch):
    captured = {}

    async def fake_select_many(table, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)
    assert await hosted_worker._recover_stale_started_runs() == 0
    assert captured["order_by"] == (("current_status_at", False),)


def test_blocked_for_credit_is_the_only_reaper_exclusion():
    # requeue_capacity_conflict_parked (bug #28) must stay revivable by the
    # reaper — it is that run's recovery path.
    assert worker_mod._STALE_PAUSED_REAPER_EXCLUDED_REASONS == frozenset({"blocked_for_credit"})


async def test_requeue_stale_started_runs_discovers_and_dry_runs(monkeypatch):
    stale_at = (
        datetime.now(timezone.utc) - timedelta(minutes=20)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-1] + "+00:00"
    fresh_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "run_id": "run-stale-1",
            "ticket_id": "ticket-1",
            "current_queue_status": "started",
            "current_status_at": stale_at,
        },
        {
            "run_id": "run-fresh",
            "ticket_id": "ticket-2",
            "current_queue_status": "started",
            "current_status_at": fresh_at,
        },
        {
            "run_id": "run-completed",
            "ticket_id": "ticket-3",
            "current_queue_status": "started",
            "current_status_at": stale_at,
        },
    ]
    requeue_calls = []

    async def fake_select_all(table, **kwargs):
        assert table == "research_loop_run_queue_current"
        return rows

    async def fake_select_one(table, **kwargs):
        run_id = kwargs["filters"][0][1]
        if run_id == "run-completed":
            return {"run_id": run_id, "current_loop_status": "completed", "current_event_type": "loop_completed"}
        return {"run_id": run_id, "current_loop_status": "running"}

    async def fake_requeue_failed_loop(**kwargs):
        requeue_calls.append(kwargs)
        return {"ok": True, "run_id": kwargs["run_id"], "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(maintenance_mod, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance_mod, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance_mod, "requeue_failed_loop", fake_requeue_failed_loop)

    result = await maintenance_mod.requeue_stale_started_autoresearch_runs(
        dry_run=True,
        actor_ref="operator:test",
        max_batch_size=10,
    )

    assert result["dry_run"] is True
    assert result["discovered_started"] == 3
    assert result["stale_started"] == 2
    assert result["planned_or_requeued"] == 1
    assert result["skipped"][0]["run_id"] == "run-completed"
    assert requeue_calls == [
        {
            "run_id": "run-stale-1",
            "reason": "operator_requeue_stale_started",
            "actor_ref": "operator:test",
            "dry_run": True,
            "force": True,
        }
    ]


# ---------------------------------------------------------------------------
# Bug #28 — requeue capacity/hotkey conflict parks the run as paused
# ---------------------------------------------------------------------------


def _wire_parking_fakes(monkeypatch, events, conflict_message):
    async def fake_create_queue_event(**kwargs):
        events.append(kwargs)
        if kwargs["event_type"] == "queued":
            raise RuntimeError(conflict_message)
        return {}

    ticket_events = []

    async def fake_create_ticket_event(**kwargs):
        ticket_events.append(kwargs)
        return {}

    projections = []

    async def fake_project(ticket_id, **kwargs):
        projections.append(ticket_id)
        return None

    monkeypatch.setattr(worker_mod, "create_queue_event", fake_create_queue_event)
    monkeypatch.setattr(worker_mod, "create_ticket_event", fake_create_ticket_event)
    monkeypatch.setattr(worker_mod, "safe_project_public_loop_activity", fake_project)
    return ticket_events, projections


async def test_mark_retryable_hotkey_conflict_parks_run(hosted_worker, monkeypatch):
    events = []
    ticket_events, projections = _wire_parking_fakes(
        monkeypatch,
        events,
        f"research_lab_queue_hotkey_conflict: miner {MINER_HOTKEY} already has active run",
    )
    context = _make_context()

    outcome = await hosted_worker._mark_retryable(context, "transient boom", retry_count=1)
    assert outcome.status == "requeue_capacity_conflict_parked"
    assert outcome.processed is True

    paused = [event for event in events if event["event_type"] == "paused"]
    assert len(paused) == 1
    assert paused[0]["reason"] == "requeue_capacity_conflict_parked"
    assert paused[0]["event_doc"]["requeue_conflict_reason"] == "transient_worker_error_requeued"
    assert "research_lab_queue_hotkey_conflict" in paused[0]["event_doc"]["requeue_conflict_error"]
    # Explanatory ticket event + public projection still happen.
    assert [event["reason"] for event in ticket_events] == ["requeue_capacity_conflict_parked"]
    assert projections == [TICKET_ID]


async def test_mark_builder_not_ready_capacity_conflict_parks_run(hosted_worker, monkeypatch):
    events = []
    _wire_parking_fakes(
        monkeypatch,
        events,
        "research_lab_queue_capacity_conflict: active 3 capacity 3",
    )
    context = _make_context()

    outcome = await hosted_worker._mark_builder_not_ready(context, "builder not ready")
    assert outcome.status == "requeue_capacity_conflict_parked"
    paused = [event for event in events if event["event_type"] == "paused"]
    assert len(paused) == 1
    assert paused[0]["event_doc"]["requeue_conflict_reason"] == "code_edit_builder_not_ready_requeued"


async def test_mark_retryable_non_conflict_error_still_raises(hosted_worker, monkeypatch):
    events = []
    _wire_parking_fakes(monkeypatch, events, "connection reset by peer")
    context = _make_context()

    with pytest.raises(RuntimeError, match="connection reset"):
        await hosted_worker._mark_retryable(context, "transient boom", retry_count=1)
    # No parking event was written for a non-conflict failure.
    assert [event["event_type"] for event in events] == ["queued"]


# ---------------------------------------------------------------------------
# Bug #26 — credit ref resolved past the 20-event context window
# ---------------------------------------------------------------------------


def _full_window_without_credit():
    # A much-requeued run: the newest 20 queue events carry no credit ref.
    return tuple(
        _queue_event(seq, "started", {"worker_ref": "worker-a"})
        for seq in range(40, 40 - worker_mod._RUN_CONTEXT_QUEUE_EVENT_LIMIT, -1)
    )


async def test_credit_ref_full_history_fallback(hosted_worker, monkeypatch):
    context = _make_context(queue_events=_full_window_without_credit())
    captured = {}
    calls = []

    async def fake_select_many(table, **kwargs):
        calls.append(table)
        captured["table"] = table
        captured.update(kwargs)
        return [
            _queue_event(1, "queued", {"loop_start_credit_id": "credit-123", "payment_id": "p-1"}),
            _queue_event(7, "queued", {"resume_source": "hosted_worker_stale_paused_reaper"}),
        ]

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)

    assert await hosted_worker._resolve_loop_start_credit_id(context) == "credit-123"
    assert captured["table"] == "research_loop_run_queue_events"
    # Targeted fetch: this run's queued events only, earliest first.
    assert captured["filters"] == (("run_id", context.run_id), ("event_type", "queued"))
    assert captured["order_by"] == (("seq", False),)
    # Sync receipt builders see the resolved value.
    assert worker_mod._context_loop_start_credit_id(context) == "credit-123"
    # Resolution is cached: no second fetch.
    assert await hosted_worker._resolve_loop_start_credit_id(context) == "credit-123"
    assert len(calls) == 1


async def test_credit_ref_in_window_short_circuits(hosted_worker, monkeypatch):
    events = (_queue_event(1, "queued", {"loop_start_credit_id": "credit-9"}),)
    context = _make_context(queue_events=events)

    async def fail_select_many(table, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("full-history fetch should not run when the ref is in-window")

    monkeypatch.setattr(worker_mod, "select_many", fail_select_many)
    assert await hosted_worker._resolve_loop_start_credit_id(context) == "credit-9"
    assert worker_mod._context_loop_start_credit_id(context) == "credit-9"


async def test_credit_ref_absent_small_history_skips_fetch(hosted_worker, monkeypatch):
    # Fewer events than the window: full history already visible, ref truly absent.
    events = tuple(_queue_event(seq, "queued", {"payment_id": "p"}) for seq in (3, 2, 1))
    context = _make_context(queue_events=events)

    async def fail_select_many(table, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("no fetch needed when history fits the window")

    monkeypatch.setattr(worker_mod, "select_many", fail_select_many)
    assert await hosted_worker._resolve_loop_start_credit_id(context) is None
    assert context.loop_start_credit_id_resolved is True


async def test_credit_ref_fetch_failure_is_not_cached(hosted_worker, monkeypatch):
    context = _make_context(queue_events=_full_window_without_credit())
    attempts = []

    async def flaky_select_many(table, **kwargs):
        attempts.append(table)
        if len(attempts) == 1:
            raise RuntimeError("postgrest connection reset")
        return [_queue_event(1, "queued", {"loop_start_credit_id": "credit-77"})]

    monkeypatch.setattr(worker_mod, "select_many", flaky_select_many)
    # Transient failure: deny nothing permanently — stay unresolved.
    assert await hosted_worker._resolve_loop_start_credit_id(context) is None
    assert context.loop_start_credit_id_resolved is False
    # A later call retries and finds the ref.
    assert await hosted_worker._resolve_loop_start_credit_id(context) == "credit-77"


# ---------------------------------------------------------------------------
# P2 — generation-stats mode flag (default preserves current behavior)
# ---------------------------------------------------------------------------


def test_generation_stats_mode_defaults_to_full(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_GENERATION_STATS_MODE", raising=False)
    assert worker_mod._generation_stats_mode() == "full"
    monkeypatch.setenv("RESEARCH_LAB_GENERATION_STATS_MODE", "bogus")
    assert worker_mod._generation_stats_mode() == "full"
    monkeypatch.setenv("RESEARCH_LAB_GENERATION_STATS_MODE", "OFF")
    assert worker_mod._generation_stats_mode() == "off"
    monkeypatch.setenv("RESEARCH_LAB_GENERATION_STATS_MODE", "best_effort_once")
    assert worker_mod._generation_stats_mode() == "best_effort_once"


def test_generation_stats_off_skips_fetch(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_GENERATION_STATS_MODE", "off")
    calls = []

    def opener(req, timeout):  # pragma: no cover - must not run
        calls.append(req)
        raise AssertionError("generation stats fetch must be skipped when off")

    stats, status = worker_mod._fetch_openrouter_generation_stats(
        api_key="key", response_id="gen-1", opener=opener
    )
    assert stats is None
    assert status == "disabled"
    assert calls == []


def test_generation_stats_best_effort_once_single_attempt_no_sleep(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_GENERATION_STATS_MODE", "best_effort_once")
    calls = []

    def opener(req, timeout):
        calls.append(1)
        raise URLError("connection refused")

    def no_sleep(seconds):  # pragma: no cover - must not run
        raise AssertionError("best_effort_once must not sleep between retries")

    monkeypatch.setattr(worker_mod.time, "sleep", no_sleep)
    stats, status = worker_mod._fetch_openrouter_generation_stats(
        api_key="key", response_id="gen-1", opener=opener
    )
    assert stats is None
    assert status.startswith("url_error")
    assert len(calls) == 1


def test_generation_stats_full_mode_retries_preserved(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_GENERATION_STATS_MODE", raising=False)
    calls = []
    sleeps = []

    def opener(req, timeout):
        calls.append(1)
        raise URLError("connection refused")

    monkeypatch.setattr(worker_mod.time, "sleep", sleeps.append)
    stats, _status = worker_mod._fetch_openrouter_generation_stats(
        api_key="key", response_id="gen-1", opener=opener
    )
    assert stats is None
    assert len(calls) == worker_mod._OPENROUTER_GENERATION_STATS_ATTEMPTS
    assert len(sleeps) == worker_mod._OPENROUTER_GENERATION_STATS_ATTEMPTS - 1


# ---------------------------------------------------------------------------
# P2 — worker proxy opener for worker LLM traffic (default off)
# ---------------------------------------------------------------------------


def test_worker_llm_proxy_opener_default_off(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_WORKER_PROXY_APPLY_TO_LLM", raising=False)
    config = ResearchLabGatewayConfig(hosted_worker_proxy_url="http://proxy.internal:3128")
    assert worker_mod._worker_llm_proxy_opener(config) is None


def test_worker_llm_proxy_opener_enabled_requires_proxy_url(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_WORKER_PROXY_APPLY_TO_LLM", "true")
    config = ResearchLabGatewayConfig(hosted_worker_proxy_url="http://proxy.internal:3128")
    opener = worker_mod._worker_llm_proxy_opener(config)
    assert opener is not None
    assert hasattr(opener, "open")
    assert worker_mod._worker_llm_proxy_opener(ResearchLabGatewayConfig()) is None
