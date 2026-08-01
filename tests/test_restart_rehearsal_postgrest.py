from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.restart_rehearsal.gateway_boundary_service import (
    LocalPostgRESTState,
    _matches_filter,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
    frontier_artifact_hashes_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json


ROOT = Path(__file__).resolve().parents[1]


def test_postgrest_boundary_imports_candidate_source_tree() -> None:
    script = (
        ROOT / "tests" / "restart_rehearsal" / "run_inside.sh"
    ).read_text(encoding="utf-8")
    assert (
        'PYTHONPATH="/source:/harness" /usr/bin/python3.11 \\\n'
        "    /harness/gateway_boundary_service.py"
    ) in script


def _maintenance_lease_state(tmp_path: Path) -> LocalPostgRESTState:
    state_root = tmp_path / "state"
    state_root.mkdir()
    return LocalPostgRESTState(
        state_root=state_root,
        fixture={},
        source_root=ROOT,
        tables={"research_lab_maintenance_lease"},
        rpcs={"research_lab_acquire_maintenance_lease"},
        relation_columns={
            "research_lab_maintenance_lease": frozenset(
                {
                    "lease_name",
                    "holder_ref",
                    "acquired_at",
                    "expires_at",
                    "updated_at",
                }
            )
        },
        durable_state_path=tmp_path / "durable.json",
        durable_schema_sha="1" * 40,
    )


def _allocation_frontier_state(
    tmp_path: Path,
) -> tuple[LocalPostgRESTState, dict[str, object]]:
    frontier_table = "research_lab_allocation_settlement_frontiers_v2"
    activation_table = (
        "research_lab_allocation_settlement_frontier_activation_v2"
    )
    execution_table = "research_lab_attested_execution_results_v2"
    receipt_table = "research_lab_attested_execution_receipts_v2"
    state_root = tmp_path / "frontier-state"
    state_root.mkdir()
    state = LocalPostgRESTState(
        state_root=state_root,
        fixture={},
        source_root=ROOT,
        tables={
            frontier_table,
            activation_table,
            execution_table,
            receipt_table,
        },
        rpcs={"persist_research_lab_allocation_settlement_frontier_v2"},
        relation_columns={
            frontier_table: frozenset(
                {
                    "netuid",
                    "allocation_epoch",
                    "settled_through_epoch",
                    "schema_version",
                    "frontier_hash",
                    "predecessor_frontier_hash",
                    "source_receipt_hash",
                    "source_state_hash",
                    "frontier_doc",
                    "created_at",
                }
            ),
            activation_table: frozenset(
                {
                    "netuid",
                    "schema_version",
                    "first_allocation_epoch",
                    "first_frontier_hash",
                    "source_receipt_hash",
                    "activated_at",
                }
            ),
            execution_table: frozenset(),
            receipt_table: frozenset(),
        },
        durable_state_path=tmp_path / "frontier-durable.json",
        durable_schema_sha="2" * 40,
    )
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    receipt_hash = "sha256:" + "4" * 64
    source_state = {
        "epoch": 100,
        "netuid": 71,
        "settlement_frontier": frontier,
    }
    source_state_hash = sha256_json(source_state)
    shared_receipt_fields = {
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "job_id": "allocation:100",
        "epoch_id": 100,
        "sequence": 0,
        "input_root": "sha256:" + "5" * 64,
        "output_root": "sha256:" + "6" * 64,
        "artifact_root": "sha256:" + "7" * 64,
    }
    state.rows[execution_table].append(
        {
            **shared_receipt_fields,
            "receipt_hash": receipt_hash,
            "operation": "research_lab_allocation",
            "result_doc": {
                "source_state": source_state,
                "source_state_hash": source_state_hash,
            },
            "artifact_hashes": list(frontier_artifact_hashes_v2(frontier))
            + [source_state_hash],
        }
    )
    state.rows[receipt_table].append(
        {
            **shared_receipt_fields,
            "receipt_hash": receipt_hash,
            "receipt_status": "succeeded",
        }
    )
    body = {
        "requested_frontier": frontier,
        "requested_source_receipt_hash": receipt_hash,
        "requested_source_state_hash": source_state_hash,
    }
    return state, body


def test_json_filters_match_postgrest_text_and_json_semantics() -> None:
    row = {
        "allocation_doc": {
            "historical_compute_fallback": True,
            "historical_compute_fallback_source_epoch": None,
            "reimbursement_allocations": [],
        }
    }

    assert _matches_filter(
        row,
        "allocation_doc->>historical_compute_fallback",
        "eq.true",
    )
    assert _matches_filter(
        row,
        "allocation_doc->>historical_compute_fallback_source_epoch",
        "is.null",
    )
    assert not _matches_filter(
        row,
        "allocation_doc->reimbursement_allocations",
        "not.eq.[]",
    )


def test_json_array_filter_selects_nonempty_compute_allocations() -> None:
    row = {
        "allocation_doc": {
            "reimbursement_allocations": [{"uid": 2}],
        }
    }

    assert _matches_filter(
        row,
        "allocation_doc->reimbursement_allocations",
        "not.eq.[]",
    )


def test_durable_postgrest_state_survives_process_replacement(
    tmp_path: Path,
) -> None:
    durable_path = tmp_path / "durable" / "postgrest.json"
    schema_sha = "1" * 40
    common = {
        "fixture": {},
        "source_root": ROOT,
        "tables": {"durable_rows"},
        "rpcs": set(),
        "relation_columns": {
            "durable_rows": frozenset({"id", "value"})
        },
        "durable_state_path": durable_path,
        "durable_schema_sha": schema_sha,
    }
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = LocalPostgRESTState(
        state_root=first_root,
        **common,
    )
    start = first.durable_state_identity()
    with first.lock:
        first.rows["durable_rows"].append(
            {"id": "settlement-1", "value": "finalized"}
        )
        first._write_durable_state_locked(mutated=True)
    written = first.durable_state_identity()

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = LocalPostgRESTState(
        state_root=second_root,
        **common,
    )

    assert start["revision"] == 0
    assert written["revision"] == 1
    assert second.durable_state_identity() == written
    assert second.rows["durable_rows"] == [
        {"id": "settlement-1", "value": "finalized"}
    ]


def test_durable_postgrest_state_rejects_schema_or_hash_drift(
    tmp_path: Path,
) -> None:
    durable_path = tmp_path / "postgrest.json"
    state_root = tmp_path / "first"
    state_root.mkdir()
    state = LocalPostgRESTState(
        state_root=state_root,
        fixture={},
        source_root=ROOT,
        tables={"durable_rows"},
        rpcs=set(),
        relation_columns={
            "durable_rows": frozenset({"id", "value"})
        },
        durable_state_path=durable_path,
        durable_schema_sha="1" * 40,
    )
    document = json.loads(durable_path.read_text(encoding="utf-8"))
    document["rows"]["durable_rows"].append(
        {"id": "tampered", "value": "unknown"}
    )
    durable_path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    second_root = tmp_path / "second"
    second_root.mkdir()
    with pytest.raises(
        ValueError,
        match="durable PostgREST state identity differs",
    ):
        LocalPostgRESTState(
            state_root=second_root,
            fixture={},
            source_root=ROOT,
            tables={"durable_rows"},
            rpcs=set(),
            relation_columns={
                "durable_rows": frozenset({"id", "value"})
            },
            durable_state_path=durable_path,
            durable_schema_sha="1" * 40,
        )


def test_frontier_boundary_persists_and_replays_exact_request(
    tmp_path: Path,
) -> None:
    state, body = _allocation_frontier_state(tmp_path)

    first = state.persist_allocation_settlement_frontier(body)
    replay = state.persist_allocation_settlement_frontier(body)

    assert first["status"] == "persisted"
    assert replay["status"] == "already_persisted"
    assert first["frontier_hash"] == body["requested_frontier"][
        "frontier_hash"
    ]


def test_frontier_boundary_fails_closed_without_activation(
    tmp_path: Path,
) -> None:
    state, body = _allocation_frontier_state(tmp_path)
    state.persist_allocation_settlement_frontier(body)
    state.rows[
        "research_lab_allocation_settlement_frontier_activation_v2"
    ].clear()

    with pytest.raises(ValueError, match="activation is invalid"):
        state.persist_allocation_settlement_frontier(body)


def test_frontier_boundary_rejects_alternate_receipt_authority(
    tmp_path: Path,
) -> None:
    state, body = _allocation_frontier_state(tmp_path)
    altered = {
        **body,
        "requested_source_receipt_hash": "sha256:" + "8" * 64,
    }

    with pytest.raises(ValueError, match="source is invalid"):
        state.persist_allocation_settlement_frontier(altered)


def test_maintenance_lease_matches_acquire_renew_and_contention_contract(
    tmp_path: Path,
) -> None:
    state = _maintenance_lease_state(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    first = state.acquire_maintenance_lease(
        {
            "p_lease_name": "scoring_worker_recovery",
            "p_holder_ref": "worker-a",
            "p_ttl_seconds": 180,
        },
        now=now,
    )
    contender = state.acquire_maintenance_lease(
        {
            "p_lease_name": "scoring_worker_recovery",
            "p_holder_ref": "worker-b",
            "p_ttl_seconds": 180,
        },
        now=now + timedelta(seconds=30),
    )
    renewal = state.acquire_maintenance_lease(
        {
            "p_lease_name": "scoring_worker_recovery",
            "p_holder_ref": "worker-a",
            "p_ttl_seconds": 180,
        },
        now=now + timedelta(seconds=60),
    )

    row = state.rows["research_lab_maintenance_lease"][0]
    assert first["acquired"] is True
    assert contender["acquired"] is False
    assert contender["holder_ref"] == "worker-a"
    assert renewal["acquired"] is True
    assert row["holder_ref"] == "worker-a"
    assert row["acquired_at"] == now.isoformat()
    assert row["updated_at"] == (now + timedelta(seconds=60)).isoformat()


def test_maintenance_lease_allows_expired_takeover_and_persists_it(
    tmp_path: Path,
) -> None:
    state = _maintenance_lease_state(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    body = {
        "p_lease_name": "hosted_worker_maintenance",
        "p_holder_ref": "worker-a",
        "p_ttl_seconds": 30,
    }
    state.acquire_maintenance_lease(body, now=now)
    takeover = state.acquire_maintenance_lease(
        {**body, "p_holder_ref": "worker-b"},
        now=now + timedelta(seconds=31),
    )

    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    restored = LocalPostgRESTState(
        state_root=restored_root,
        fixture={},
        source_root=ROOT,
        tables={"research_lab_maintenance_lease"},
        rpcs={"research_lab_acquire_maintenance_lease"},
        relation_columns=state.relation_columns,
        durable_state_path=tmp_path / "durable.json",
        durable_schema_sha="1" * 40,
    )

    assert takeover["acquired"] is True
    assert takeover["holder_ref"] == "worker-b"
    assert restored.rows["research_lab_maintenance_lease"] == (
        state.rows["research_lab_maintenance_lease"]
    )


@pytest.mark.parametrize(
    "body",
    [
        {},
        {
            "p_lease_name": "lease",
            "p_holder_ref": "holder",
            "p_ttl_seconds": 0,
        },
        {
            "p_lease_name": "lease",
            "p_holder_ref": "holder",
            "p_ttl_seconds": 86401,
        },
        {
            "p_lease_name": "lease",
            "p_holder_ref": "holder",
            "p_ttl_seconds": True,
        },
    ],
)
def test_maintenance_lease_rejects_malformed_requests(
    tmp_path: Path,
    body: dict[str, object],
) -> None:
    state = _maintenance_lease_state(tmp_path)

    with pytest.raises(ValueError, match="maintenance lease"):
        state.acquire_maintenance_lease(body)
