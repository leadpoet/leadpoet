"""Genuine rich-capture failure-recovery regression.

Exercises the REAL backfill dispatcher end to end (discovery RPC -> per-run
inspection -> repair -> completeness marker), not just the insert helper:

  projected run, execution trace present, but a TRANSIENT provider-ledger
  failure -> the run is NOT marked complete -> normal discovery keeps returning
  it -> the next pass restores every missing provider-usage row exactly once ->
  the run is then marked complete and no longer re-inspected.

This is the recovery hole the review flagged: the old delta only surfaced runs
missing an execution trace, and a swallowed provider-ledger failure was recorded
as "0 inserted" while the run was treated as done.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import gateway.research_lab.trajectory_projector as tp


def _usage_row(rid: str) -> dict[str, Any]:
    return {"usage_row_id": rid, "provider_id": "exa", "utc_day": "2026-07-23"}


class _RecoveryStore:
    """In-memory store modeling the corpus + completeness RPCs.

    - provider-usage insert: ON CONFLICT DO NOTHING, with a one-shot transient
      failure (returns via the real helper as None).
    - needing-corpus discovery: returns the run until a completeness marker
      exists (mirrors research_lab_terminal_runs_needing_corpus).
    - mark-complete: records the marker.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.ledger: dict[str, dict[str, Any]] = {}
        self.complete: set[str] = set()  # trajectory_ids marked complete
        self.fail_provider_next = False
        self.discovery_calls = 0

    async def call_rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        if function_name == "insert_research_lab_provider_usage_ledger_rows":
            if self.fail_provider_next:
                self.fail_provider_next = False
                raise RuntimeError("transient supabase 503")
            inserted = 0
            for row in params["rows"]:
                rid = row["usage_row_id"]
                if rid not in self.ledger:
                    self.ledger[rid] = row
                    inserted += 1
            return {"requested": len(params["rows"]), "inserted": inserted}
        if function_name == "research_lab_terminal_runs_needing_corpus":
            self.discovery_calls += 1
            tid = tp.trajectory_id_for_run(self.run_id)
            rows = [] if tid in self.complete else [self.run_id]
            return [{"run_id": r} for r in rows]
        if function_name == "research_lab_mark_corpus_complete":
            self.complete.add(str(params["p_trajectory_id"]))
            return None
        raise AssertionError(f"unexpected rpc {function_name}")

    async def select_one(self, table: str, *, columns: str = "*", filters: Any = ()) -> Any:
        # The trajectory envelope exists (the run IS projected).
        if table == tp.TRAJECTORIES_TABLE:
            return {"trajectory_id": tp.trajectory_id_for_run(self.run_id)}
        return None


def _install_projection(monkeypatch, store: _RecoveryStore, usage_rows) -> None:
    """Stub the projection build so only provider-usage is ever 'missing'."""
    monkeypatch.setattr(tp, "projector_enabled", lambda: True)

    async def fake_inputs(run_id, s):
        return {"loop_events": [{"seq": 0}], "candidate_rows": [], "evaluation_events": []}

    monkeypatch.setattr(tp, "load_projection_inputs", fake_inputs)
    monkeypatch.setattr(tp, "_projection_readiness_errors", lambda **k: [])
    monkeypatch.setattr(
        tp,
        "build_trajectory_projection",
        lambda **inputs: SimpleNamespace(
            errors=[],
            execution_trace_rows=[],      # trace already present
            evidence_bundle_rows=[],      # evidence present
            provider_usage_ledger_rows=list(usage_rows),
            event_rows=[],
            ledger_rows=[],
            trajectory_id=tp.trajectory_id_for_run(store.run_id),
        ),
    )

    async def none_missing(_s, _p):
        return []

    monkeypatch.setattr(tp, "_missing_trajectory_event_rows", none_missing)
    monkeypatch.setattr(tp, "_missing_results_ledger_rows", none_missing)


@pytest.mark.asyncio
async def test_transient_provider_failure_recovers_through_real_dispatcher(monkeypatch) -> None:
    run_id = "3f2a1c00-0000-4000-8000-000000000abc"
    tid = tp.trajectory_id_for_run(run_id)
    usage_rows = [_usage_row(f"{i:032d}") for i in range(5)]
    store = _RecoveryStore(run_id)
    _install_projection(monkeypatch, store, usage_rows)

    # Pass 1 — provider-ledger insert fails transiently.
    store.fail_provider_next = True
    results = await tp.backfill_corpus_trace_rows(store=store, dry_run=False, batch_size=5)
    assert len(results) == 1
    # The failure must NOT be recorded as complete, and NO row persisted.
    assert tid not in store.complete, "a failed provider insert must not mark the run complete"
    assert store.ledger == {}

    # Pass 2 — discovery STILL returns the run (not marked); provider healthy now.
    results = await tp.backfill_corpus_trace_rows(store=store, dry_run=False, batch_size=5)
    assert len(results) == 1
    assert results[0].status == "skipped_traces_existing"
    # Every missing provider-usage row restored exactly once...
    assert set(store.ledger) == {r["usage_row_id"] for r in usage_rows}
    assert len(store.ledger) == 5
    # ...and the run is now marked complete.
    assert tid in store.complete

    # Pass 3 — discovery returns nothing (marker present): no re-inspection.
    calls_before = store.discovery_calls
    results = await tp.backfill_corpus_trace_rows(store=store, dry_run=False, batch_size=5)
    assert results == []
    assert store.discovery_calls == calls_before + 1  # one discovery call, zero runs
    assert len(store.ledger) == 5  # no duplicates
