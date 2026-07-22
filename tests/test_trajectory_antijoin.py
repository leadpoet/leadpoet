"""Regression tests for the server-side trajectory anti-join (egress reduction).

project_completed_runs() previously issued one existence SELECT per terminal
run (the N+1 that multiplied across workers/passes). These tests pin that the
discovery now does a single anti-join RPC, preserves newest-first order, and
falls back to the per-run path only for injected fakes without call_rpc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import gateway.research_lab.trajectory_projector as tp


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "116-research-lab-trajectory-antijoin.sql"
).read_text(encoding="utf-8")


class _RpcStore:
    def __init__(self, missing_tids: list[str]) -> None:
        self._missing = missing_tids
        self.rpc_calls: list[tuple[str, dict[str, Any]]] = []
        self.select_one_calls = 0

    async def call_rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        self.rpc_calls.append((function_name, params))
        # SETOF UUID comes back as [{"<fn>": uuid}, ...].
        return [{"research_lab_missing_trajectory_ids": t} for t in self._missing]

    async def select_one(self, *args: Any, **kwargs: Any) -> Any:
        self.select_one_calls += 1
        return None


class _PerRunStore:
    def __init__(self, existing_tids: set[str]) -> None:
        self._existing = existing_tids
        self.select_one_calls = 0

    async def select_one(self, table: str, *, filters: Any = (), **kwargs: Any) -> Any:
        self.select_one_calls += 1
        tid = dict(filters).get("trajectory_id")
        return {"trajectory_id": tid} if tid in self._existing else None


@pytest.mark.asyncio
async def test_antijoin_uses_one_rpc_and_preserves_order() -> None:
    run_ids = ["run-a", "run-b", "run-c"]
    tids = {r: tp.trajectory_id_for_run(r) for r in run_ids}
    # Only run-a and run-c are unprojected (run-b already has a trajectory).
    store = _RpcStore(missing_tids=[tids["run-a"], tids["run-c"]])

    missing = await tp._unprojected_run_ids(store, run_ids)

    assert missing == ["run-a", "run-c"]  # newest-first order preserved
    assert len(store.rpc_calls) == 1  # single anti-join, not one-per-run
    assert store.rpc_calls[0][0] == "research_lab_missing_trajectory_ids"
    assert set(store.rpc_calls[0][1]["candidate_ids"]) == set(tids.values())
    assert store.select_one_calls == 0


@pytest.mark.asyncio
async def test_antijoin_dedupes_and_ignores_blank_run_ids() -> None:
    store = _RpcStore(missing_tids=[tp.trajectory_id_for_run("run-a")])
    missing = await tp._unprojected_run_ids(store, ["run-a", "run-a", "", None])
    assert missing == ["run-a"]
    assert len(store.rpc_calls[0][1]["candidate_ids"]) == 1


@pytest.mark.asyncio
async def test_antijoin_falls_back_to_per_run_without_call_rpc() -> None:
    tids = {r: tp.trajectory_id_for_run(r) for r in ("run-a", "run-b")}
    store = _PerRunStore(existing_tids={tids["run-a"]})  # run-a projected already
    missing = await tp._unprojected_run_ids(store, ["run-a", "run-b"])
    assert missing == ["run-b"]
    assert store.select_one_calls == 2  # per-run path


@pytest.mark.asyncio
async def test_antijoin_rpc_error_falls_back_to_per_run() -> None:
    class _Boom:
        async def call_rpc(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("edge blip")

        async def select_one(self, *args: Any, **kwargs: Any) -> Any:
            return None  # nothing projected → all missing

        pass

    missing = await tp._unprojected_run_ids(_Boom(), ["run-a", "run-b"])
    assert missing == ["run-a", "run-b"]


def test_rpc_uuid_list_parses_dict_and_scalar_rows() -> None:
    assert tp._rpc_uuid_list([{"f": "u1"}, {"f": "u2"}]) == ["u1", "u2"]
    assert tp._rpc_uuid_list(["u1", "u2"]) == ["u1", "u2"]
    assert tp._rpc_uuid_list([]) == []
    assert tp._rpc_uuid_list(None) == []


def test_migration_is_a_locked_down_antijoin() -> None:
    assert "research_lab_missing_trajectory_ids" in MIGRATION
    assert "NOT EXISTS" in MIGRATION
    assert "public.research_trajectories" in MIGRATION
    assert "SECURITY DEFINER" in MIGRATION
    assert "SET search_path = ''" in MIGRATION
    assert "TO service_role;" in MIGRATION
    assert "FROM PUBLIC, anon, authenticated;" in MIGRATION
