"""Tests for the traces backfill entry points and bundle→trace linkage.

Closes the 2026-07-02 audit gaps: (a) the ``--traces-backfill`` guard rails
(disabled / unprojected / incomplete runs) had no unit coverage, and (b) the
scoring worker's ``execution_trace_ref`` must stamp the SAME deterministic
uuid5 the trajectory projector writes for the candidate's node — that ref is
the join key connecting score bundles, ``engine_trace_mappings``, and
``execution_traces``.
"""

from __future__ import annotations

import uuid
from typing import Any

from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker
from gateway.research_lab.trajectory_projector import (
    PROJECTOR_ENABLED_ENV,
    backfill_run_corpus_trace_rows,
    evidence_bundle_id_for_node,
    execution_trace_id_for_node,
    execution_trace_id_for_score_bundle,
    trajectory_id_for_run,
)

RUN_ID = "11111111-1111-4111-8111-111111111111"
NODE_ID = "node-3"


# ---------------------------------------------------------------------------
# Deterministic id helpers
# ---------------------------------------------------------------------------


def test_trace_ids_are_stable_valid_uuids() -> None:
    first = execution_trace_id_for_node(RUN_ID, NODE_ID)
    assert first == execution_trace_id_for_node(RUN_ID, NODE_ID)
    uuid.UUID(first)  # must parse
    assert first != execution_trace_id_for_node(RUN_ID, "node-4")
    assert first != execution_trace_id_for_score_bundle(RUN_ID, "score_bundle:abc")
    assert first != trajectory_id_for_run(RUN_ID)


def test_evidence_ids_key_on_node_and_bundle() -> None:
    a = evidence_bundle_id_for_node(RUN_ID, NODE_ID, "score_bundle:abc")
    assert a == evidence_bundle_id_for_node(RUN_ID, NODE_ID, "score_bundle:abc")
    assert a != evidence_bundle_id_for_node(RUN_ID, NODE_ID, "score_bundle:def")


# ---------------------------------------------------------------------------
# Bundle→trace linkage in the scoring worker run context
# ---------------------------------------------------------------------------


def _run_context(candidate: dict[str, Any]) -> dict[str, Any]:
    worker = object.__new__(ResearchLabGatewayScoringWorker)
    worker.worker_ref = "test-worker-1"
    return worker._candidate_run_context(
        candidate, window_hash="sha256:" + "d" * 64, evaluation_epoch=7
    )


def _candidate(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": RUN_ID,
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "miner_hotkey": "miner-hotkey-1",
        "island": "generalist",
        "candidate_id": "cand-1",
        "candidate_kind": "image_build",
    }
    base.update(overrides)
    return base


def test_run_context_stamps_deterministic_trace_ref_from_loop_node_id() -> None:
    context = _run_context(_candidate(candidate_build_doc={"loop_node_id": NODE_ID}))
    expected = f"execution_trace:{execution_trace_id_for_node(RUN_ID, NODE_ID)}"
    assert context["execution_trace_ref"] == expected


def test_run_context_falls_back_to_legacy_ref_without_node_linkage() -> None:
    for candidate in (
        _candidate(),  # no build doc at all
        _candidate(candidate_build_doc={}),  # doc without the annotation
        _candidate(candidate_build_doc={"loop_node_id": ""}),  # empty annotation
    ):
        context = _run_context(candidate)
        assert context["execution_trace_ref"].startswith(
            "gateway_qualification_worker:test-worker-1:"
        )


def test_run_context_ref_matches_projector_row_for_same_node() -> None:
    """The exact equality the whole join depends on, spelled out."""
    context = _run_context(_candidate(candidate_build_doc={"loop_node_id": NODE_ID}))
    projector_row_id = execution_trace_id_for_node(RUN_ID, NODE_ID)
    assert context["execution_trace_ref"].split(":", 1)[1] == projector_row_id


# ---------------------------------------------------------------------------
# Backfill guard rails
# ---------------------------------------------------------------------------


class _MiniStore:
    """Just enough store surface for the backfill guard paths."""

    def __init__(self, envelope: dict[str, Any] | None) -> None:
        self.envelope = envelope

    async def select_one(self, table: str, *, filters: Any = (), **kwargs: Any):
        if table == "research_trajectories":
            return self.envelope
        return None

    async def select_many(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def select_all(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class _ExplodingStore:
    def __getattr__(self, name: str):
        raise AssertionError("store must not be touched when the projector is disabled")


async def test_backfill_write_mode_requires_projector_flag(monkeypatch) -> None:
    monkeypatch.delenv(PROJECTOR_ENABLED_ENV, raising=False)
    result = await backfill_run_corpus_trace_rows(
        RUN_ID, store=_ExplodingStore(), dry_run=False
    )
    assert result.status == "skipped_disabled"


async def test_backfill_skips_unprojected_runs(monkeypatch) -> None:
    monkeypatch.delenv(PROJECTOR_ENABLED_ENV, raising=False)
    result = await backfill_run_corpus_trace_rows(
        RUN_ID, store=_MiniStore(envelope=None), dry_run=True
    )
    assert result.status == "skipped_unprojected"
    assert result.trajectory_id == trajectory_id_for_run(RUN_ID)


async def test_backfill_reports_incomplete_when_no_loop_events(monkeypatch) -> None:
    monkeypatch.delenv(PROJECTOR_ENABLED_ENV, raising=False)
    envelope = {"trajectory_id": trajectory_id_for_run(RUN_ID), "run_id": RUN_ID}
    result = await backfill_run_corpus_trace_rows(
        RUN_ID, store=_MiniStore(envelope=envelope), dry_run=True
    )
    assert result.status == "skipped_incomplete"


async def test_backfill_is_best_effort_on_store_errors() -> None:
    class BrokenStore:
        async def select_one(self, *args: Any, **kwargs: Any):
            raise RuntimeError("db down")

    result = await backfill_run_corpus_trace_rows(
        RUN_ID, store=BrokenStore(), dry_run=True
    )
    assert result.status == "failed"
