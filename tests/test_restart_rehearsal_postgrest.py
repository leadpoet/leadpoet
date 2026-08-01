from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.restart_rehearsal.gateway_boundary_service import (
    LocalPostgRESTState,
    _matches_filter,
)


ROOT = Path(__file__).resolve().parents[1]


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
