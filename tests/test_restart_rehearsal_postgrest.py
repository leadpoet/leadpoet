from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.restart_rehearsal.gateway_boundary_service import (
    LocalPostgRESTState,
    _matches_filter,
)


ROOT = Path(__file__).resolve().parents[1]


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
