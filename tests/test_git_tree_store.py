from __future__ import annotations

import asyncio

import pytest

from gateway.research_lab.git_tree_models import (
    TreePolicy,
    derive_child_slot,
    derive_tree_id,
    generation_operation_id,
)
from gateway.research_lab.git_tree_store import GitTreeStore
from gateway.research_lab.store import (
    _create_candidate_tree_handoff,
    create_candidate_artifact,
    record_candidate_tree_handoff,
)
from research_lab.canonical import sha256_json


def _slot():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="33333333-3333-4333-8333-333333333333",
        root_artifact_hash="sha256:" + "a" * 64,
        policy=policy,
    )
    return derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )


def test_tree_store_uses_stable_logical_operation_identity(monkeypatch):
    calls = []

    async def rpc(name, params):
        calls.append((name, dict(params)))
        return {
            "created": True,
            "node": {"node_id": params["requested_node_id"]},
            "operation": {
                "logical_operation_id": params[
                    "requested_generation_operation_id"
                ],
                "operation_status": "reserved",
            },
        }

    monkeypatch.setattr("gateway.research_lab.git_tree_store.store.call_rpc", rpc)
    slot = _slot()
    client = GitTreeStore()
    request_hash = "sha256:" + "b" * 64
    first = asyncio.run(
        client.plan_node(
            slot=slot,
            request_hash=request_hash,
            node_doc={"direction_hash": "x"},
        )
    )
    second = asyncio.run(
        client.plan_node(
            slot=slot,
            request_hash=request_hash,
            node_doc={"direction_hash": "x"},
        )
    )

    assert first["operation"]["logical_operation_id"] == generation_operation_id(slot)
    assert second["operation"]["logical_operation_id"] == generation_operation_id(slot)
    assert calls[0][1] == calls[1][1]


def test_operation_transition_carries_one_terminal_cost(monkeypatch):
    observed = {}

    async def rpc(name, params):
        observed.update(params)
        return params

    monkeypatch.setattr("gateway.research_lab.git_tree_store.store.call_rpc", rpc)
    slot = _slot()
    asyncio.run(
        GitTreeStore().transition_operation(
            logical_operation_id=generation_operation_id(slot),
            tree_id=slot.tree_id,
            node_id=slot.node_id,
            operation_kind="generation",
            operation_status="succeeded",
            request_hash="sha256:" + "b" * 64,
            result_hash="sha256:" + "c" * 64,
            settled_cost_microusd=123,
            provider_call_count=1,
            expected_current_status="reserved",
        )
    )
    assert observed["requested_settled_cost_microusd"] == 123
    assert observed["requested_provider_call_count"] == 1
    assert observed["expected_current_status"] == "reserved"


def test_operation_inspection_reads_current_projection(monkeypatch):
    slot = _slot()

    async def select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_operation_current"
        assert filters == (("logical_operation_id", generation_operation_id(slot)),)
        return {
            "logical_operation_id": generation_operation_id(slot),
            "tree_id": slot.tree_id,
            "node_id": slot.node_id,
            "operation_kind": "generation",
            "operation_status": "reserved",
            "request_hash": "sha256:" + "b" * 64,
            "result_hash": None,
            "settled_cost_microusd": 0,
            "provider_call_count": 0,
            "settlement_doc": {},
        }

    monkeypatch.setattr(
        "gateway.research_lab.git_tree_store.store.select_one", select_one
    )
    result = asyncio.run(
        GitTreeStore().get_operation(
            logical_operation_id=generation_operation_id(slot)
        )
    )
    assert result["exists"] is True
    assert result["operation"]["operation_status"] == "reserved"


def test_tree_store_reads_exact_tree_authority(monkeypatch):
    slot = _slot()

    async def select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_tree_current"
        assert filters == (("tree_id", slot.tree_id),)
        assert "root_source_tree_hash" in columns
        assert "evaluator_commitment_hash" in columns
        return {
            "tree_id": slot.tree_id,
            "current_event_type": "checkpoint_committed",
        }

    monkeypatch.setattr(
        "gateway.research_lab.git_tree_store.store.select_one", select_one
    )
    row = asyncio.run(GitTreeStore().get_tree_current(tree_id=slot.tree_id))
    assert row == {
        "tree_id": slot.tree_id,
        "current_event_type": "checkpoint_committed",
    }


def test_tree_store_paginates_all_operation_settlements(monkeypatch):
    slot = _slot()
    calls = []

    async def select_all(table, **kwargs):
        calls.append((table, kwargs))
        return [
            {
                "logical_operation_id": generation_operation_id(slot),
                "tree_id": slot.tree_id,
                "operation_status": "succeeded",
            }
        ]

    monkeypatch.setattr(
        "gateway.research_lab.git_tree_store.store.select_all", select_all
    )
    rows = asyncio.run(GitTreeStore().list_operations(tree_id=slot.tree_id))
    assert rows[0]["operation_status"] == "succeeded"
    table, kwargs = calls[0]
    assert table == "research_lab_autoresearch_operation_current"
    assert kwargs["filters"] == (("tree_id", slot.tree_id),)
    assert kwargs["order_by"] == (("logical_operation_id", False),)
    assert kwargs["batch_size"] == 100
    assert kwargs["max_rows"] == 1000


def test_tree_store_reads_latest_recovery_and_unique_node_event(monkeypatch):
    slot = _slot()
    many_calls = []
    one_calls = []

    async def select_many(table, **kwargs):
        many_calls.append((table, kwargs))
        return [
            {
                "tree_id": slot.tree_id,
                "seq": 7,
                "event_type": "checkpoint_committed",
                "event_doc": {"recovery": {"recovery_hash": "sha256:" + "d" * 64}},
            }
        ]

    async def select_one(table, **kwargs):
        one_calls.append((table, kwargs))
        return {
            "tree_id": slot.tree_id,
            "node_id": slot.node_id,
            "event_type": "node_generated",
        }

    monkeypatch.setattr(
        "gateway.research_lab.git_tree_store.store.select_many", select_many
    )
    monkeypatch.setattr(
        "gateway.research_lab.git_tree_store.store.select_one", select_one
    )
    latest = asyncio.run(
        GitTreeStore().get_latest_recovery_event(tree_id=slot.tree_id)
    )
    node_event = asyncio.run(
        GitTreeStore().get_node_event(
            tree_id=slot.tree_id,
            node_id=slot.node_id,
            event_type="node_generated",
        )
    )
    assert latest["seq"] == 7
    assert node_event["node_id"] == slot.node_id
    assert many_calls[0][1]["order_by"] == (("seq", True),)
    assert many_calls[0][1]["limit"] == 1
    assert (
        "event_type",
        "in",
        ["node_generated", "checkpoint_committed"],
    ) in many_calls[0][1]["filters"]
    assert one_calls[0][1]["filters"] == (
        ("tree_id", slot.tree_id),
        ("node_id", slot.node_id),
        ("event_type", "node_generated"),
    )


def test_frontier_retry_returns_existing_commitment_without_new_rpc(monkeypatch):
    slot = _slot()
    frontier_hash = "sha256:" + "f" * 64
    frontier_doc = {"node_ids": [slot.node_id]}
    rpc_calls = []

    async def select_one(table, *, columns="*", filters=()):
        if table == "research_lab_autoresearch_tree_current":
            return {
                "current_round_index": 2,
                "current_frontier_hash": frontier_hash,
                "current_frontier_doc": frontier_doc,
            }
        assert table == "research_lab_autoresearch_frontier_commitments"
        return {
            "tree_id": slot.tree_id,
            "round_index": 2,
            "schema_version": "research_lab.git_tree_frontier.v1",
            "expected_previous_hash": "sha256:" + "e" * 64,
            "frontier_hash": frontier_hash,
            "frontier_doc": frontier_doc,
            "commitment_hash": "sha256:" + "d" * 64,
            "created_at": "2026-07-14T00:00:00Z",
        }

    async def rpc(name, params):
        rpc_calls.append((name, params))

    monkeypatch.setattr(
        "gateway.research_lab.git_tree_store.store.select_one", select_one
    )
    monkeypatch.setattr("gateway.research_lab.git_tree_store.store.call_rpc", rpc)

    result = asyncio.run(
        GitTreeStore().commit_frontier_next(
            tree_id=slot.tree_id,
            frontier_hash=frontier_hash,
            frontier_doc=frontier_doc,
        )
    )

    assert result["commitment_hash"] == "sha256:" + "d" * 64
    assert rpc_calls == []


def test_candidate_handoff_atomically_commits_tree_completion(monkeypatch):
    slot = _slot()
    observed = {}

    async def select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_tree_current"
        return {
            "tree_id": slot.tree_id,
            "current_event_type": "checkpoint_committed",
            "current_event_doc": {},
            "current_event_hash": "sha256:" + "d" * 64,
        }

    async def rpc(name, params):
        observed.update(params)
        return {
            "created": True,
            "handoff": {"tree_id": slot.tree_id},
            "completion_event": {
                "tree_id": slot.tree_id,
                "event_type": "tree_completed",
            },
        }

    monkeypatch.setattr("gateway.research_lab.store.select_one", select_one)
    monkeypatch.setattr("gateway.research_lab.store.call_rpc", rpc)
    asyncio.run(
        record_candidate_tree_handoff(
            tree_id=slot.tree_id,
            run_id="33333333-3333-4333-8333-333333333333",
            candidate_id="candidate:" + "e" * 64,
            node_id=slot.node_id,
            root_git_commit="1" * 64,
            node_git_commit="2" * 64,
            lineage_hash="sha256:" + "3" * 64,
        )
    )
    assert observed["requested_previous_event_hash"] == "sha256:" + "d" * 64
    assert observed["requested_completed_event_hash"].startswith("sha256:")
    assert observed["requested_handoff_doc"]["candidate_id"] == (
        "candidate:" + "e" * 64
    )


def test_candidate_row_and_tree_handoff_use_one_active_root_rpc(monkeypatch):
    slot = _slot()
    run_id = "33333333-3333-4333-8333-333333333333"
    candidate_id = "candidate:" + "e" * 64
    request = type(
        "Request",
        (),
        {
            "git_tree_id": slot.tree_id,
            "git_tree_node_id": slot.node_id,
            "git_tree_root_commit": "1" * 64,
            "git_tree_node_commit": "2" * 64,
            "git_tree_lineage_hash": "sha256:" + "3" * 64,
            "run_id": run_id,
        },
    )()
    candidate_row = {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "git_tree_id": slot.tree_id,
        "git_tree_node_id": slot.node_id,
        "git_tree_root_commit": request.git_tree_root_commit,
        "git_tree_node_commit": request.git_tree_node_commit,
        "git_tree_lineage_hash": request.git_tree_lineage_hash,
    }
    observed = {}

    async def select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_autoresearch_tree_current"
        return {
            "tree_id": slot.tree_id,
            "current_event_type": "final_selected",
            "current_event_doc": {},
            "current_event_hash": "sha256:" + "d" * 64,
        }

    async def rpc(name, params):
        assert name == "create_research_lab_git_tree_candidate_handoff"
        observed.update(params)
        return {
            "created": True,
            "candidate": dict(candidate_row),
            "handoff": {
                "handoff": {"tree_id": slot.tree_id},
                "completion_event": {
                    "tree_id": slot.tree_id,
                    "event_type": "tree_completed",
                },
            },
        }

    monkeypatch.setattr("gateway.research_lab.store.select_one", select_one)
    monkeypatch.setattr("gateway.research_lab.store.call_rpc", rpc)

    result = asyncio.run(
        _create_candidate_tree_handoff(
            request=request,
            candidate_id=candidate_id,
            candidate_row=candidate_row,
        )
    )

    assert result == candidate_row
    assert observed["requested_candidate_doc"] == candidate_row
    assert observed["requested_tree_id"] == slot.tree_id
    assert observed["requested_previous_event_hash"] == "sha256:" + "d" * 64
    assert observed["requested_completed_event_hash"].startswith("sha256:")


def test_stale_active_root_candidate_handoff_does_not_retry_or_leave_candidate(
    monkeypatch,
):
    slot = _slot()
    run_id = "33333333-3333-4333-8333-333333333333"
    candidate_id = "candidate:" + "e" * 64
    request = type(
        "Request",
        (),
        {
            "git_tree_id": slot.tree_id,
            "git_tree_node_id": slot.node_id,
            "git_tree_root_commit": "1" * 64,
            "git_tree_node_commit": "2" * 64,
            "git_tree_lineage_hash": "sha256:" + "3" * 64,
            "run_id": run_id,
        },
    )()
    rpc_calls = []

    async def select_one(table, *, columns="*", filters=()):
        if table == "research_lab_autoresearch_tree_current":
            return {
                "tree_id": slot.tree_id,
                "current_event_type": "final_selected",
                "current_event_doc": {},
                "current_event_hash": "sha256:" + "d" * 64,
            }
        assert table in {
            "research_lab_candidate_artifacts",
            "research_lab_autoresearch_tree_handoffs",
        }
        return None

    async def rpc(name, params):
        rpc_calls.append((name, params))
        raise RuntimeError(
            "40001 research_lab_git_tree_handoff_stale_active_root"
        )

    monkeypatch.setattr("gateway.research_lab.store.select_one", select_one)
    monkeypatch.setattr("gateway.research_lab.store.call_rpc", rpc)

    with pytest.raises(RuntimeError, match="stale_active_root"):
        asyncio.run(
            _create_candidate_tree_handoff(
                request=request,
                candidate_id=candidate_id,
                candidate_row={
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "git_tree_id": slot.tree_id,
                    "git_tree_node_id": slot.node_id,
                    "git_tree_root_commit": request.git_tree_root_commit,
                    "git_tree_node_commit": request.git_tree_node_commit,
                    "git_tree_lineage_hash": request.git_tree_lineage_hash,
                },
            )
        )

    assert len(rpc_calls) == 1


def test_candidate_handoff_timeout_recovery_rejects_content_conflict(monkeypatch):
    slot = _slot()
    run_id = "33333333-3333-4333-8333-333333333333"
    candidate_id = "candidate:" + "e" * 64
    request = type(
        "Request",
        (),
        {
            "git_tree_id": slot.tree_id,
            "git_tree_node_id": slot.node_id,
            "git_tree_root_commit": "1" * 64,
            "git_tree_node_commit": "2" * 64,
            "git_tree_lineage_hash": "sha256:" + "3" * 64,
            "run_id": run_id,
        },
    )()
    candidate_row = {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "candidate_artifact_hash": "sha256:" + "4" * 64,
        "candidate_model_manifest_hash": "sha256:" + "5" * 64,
        "candidate_model_manifest_doc": {"manifest": "requested"},
        "candidate_source_diff_hash": "sha256:" + "6" * 64,
        "candidate_patch_hash": "sha256:" + "7" * 64,
        "private_model_manifest_hash": "sha256:" + "8" * 64,
        "git_tree_id": slot.tree_id,
        "git_tree_node_id": slot.node_id,
        "git_tree_root_commit": request.git_tree_root_commit,
        "git_tree_node_commit": request.git_tree_node_commit,
        "git_tree_lineage_hash": request.git_tree_lineage_hash,
    }

    async def select_one(table, *, columns="*", filters=()):
        if table == "research_lab_autoresearch_tree_current":
            return {
                "tree_id": slot.tree_id,
                "current_event_type": "final_selected",
                "current_event_hash": "sha256:" + "d" * 64,
            }
        if table == "research_lab_candidate_artifacts":
            return {
                **candidate_row,
                "candidate_artifact_hash": "sha256:" + "9" * 64,
            }
        assert table == "research_lab_autoresearch_tree_handoffs"
        return {
            "tree_id": slot.tree_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "handoff_hash": "sha256:" + "a" * 64,
        }

    async def rpc(_name, _params):
        raise TimeoutError("database response timed out after commit")

    monkeypatch.setattr("gateway.research_lab.store.select_one", select_one)
    monkeypatch.setattr("gateway.research_lab.store.call_rpc", rpc)

    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(
            _create_candidate_tree_handoff(
                request=request,
                candidate_id=candidate_id,
                candidate_row=candidate_row,
            )
        )


def test_existing_candidate_reuse_rejects_stale_private_model_root(monkeypatch):
    slot = _slot()
    run_id = "33333333-3333-4333-8333-333333333333"
    candidate_artifact_hash = "sha256:" + "e" * 64
    request = type(
        "Request",
        (),
        {
            "run_id": run_id,
            "ticket_id": "44444444-4444-4444-8444-444444444444",
            "receipt_id": None,
            "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
            "island": "generalist",
            "candidate_kind": "image_build",
            "private_model_manifest": {
                "manifest_hash": "sha256:" + "a" * 64,
            },
            "candidate_patch_manifest": {
                "parent_artifact_hash": "sha256:" + "b" * 64,
                "candidate_artifact_hash": candidate_artifact_hash,
            },
            "candidate_model_manifest": {
                "model_artifact_hash": candidate_artifact_hash,
                "manifest_hash": "sha256:" + "c" * 64,
            },
            "candidate_source_diff_hash": "sha256:" + "d" * 64,
            "candidate_build_doc": {},
            "hypothesis_doc": {},
            "redacted_public_summary": "",
            "git_tree_id": slot.tree_id,
            "git_tree_node_id": slot.node_id,
            "git_tree_root_commit": "1" * 64,
            "git_tree_node_commit": "2" * 64,
            "git_tree_lineage_hash": "sha256:" + "3" * 64,
        },
    )()

    async def select_one(table, *, columns="*", filters=()):
        assert table == "research_lab_candidate_artifacts"
        return {
            "candidate_id": "candidate:" + "e" * 64,
            "candidate_artifact_hash": candidate_artifact_hash,
            "candidate_model_manifest_hash": "sha256:" + "c" * 64,
            "candidate_model_manifest_doc": dict(
                request.candidate_model_manifest
            ),
            "candidate_source_diff_hash": request.candidate_source_diff_hash,
            "candidate_patch_hash": sha256_json(
                request.candidate_patch_manifest
            ),
            "private_model_manifest_hash": "sha256:" + "9" * 64,
        }

    monkeypatch.setattr("gateway.research_lab.store.select_one", select_one)

    with pytest.raises(ValueError, match="current Git-tree root or content"):
        asyncio.run(create_candidate_artifact(request))
