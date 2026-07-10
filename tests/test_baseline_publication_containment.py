from __future__ import annotations

import json
import re
import time

import pytest

from gateway.research_lab import scoring_worker as sw


EVENT_DOC_BANNED_RE = re.compile(
    r"(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|"
    r"private_repo|judge_prompt|hidden_icp|icp_plaintext|\.dkr\.ecr\.|"
    r"image_digest|private_model_manifest_doc|candidate_patch_manifest|"
    r"proxy[_-]?url|://[^/]+:[^/@]+@)",
    re.IGNORECASE,
)


def test_next_benchmark_attempt_includes_dispatch_event_history():
    rows = [
        {"benchmark_attempt": 1},
        {"event_doc": {"benchmark_attempt": 3}},
        {"event_doc": {"benchmark_attempt": "2"}},
    ]

    assert sw._next_benchmark_attempt(rows) == 4


def test_terminal_publication_failure_blocks_same_worker_source_and_token(monkeypatch):
    source_hash = "sha256:" + "1" * 64
    token_hash = "sha256:" + "2" * 64
    monkeypatch.setattr(sw, "_scoring_worker_source_hash", lambda: source_hash)
    monkeypatch.setattr(sw, "_baseline_publication_retry_token_hash", lambda: token_hash)
    failed = {
        "dispatch_status": "failed",
        "event_doc": {
            "failure_phase": "publication",
            "terminal_no_automatic_retry": True,
            "scoring_worker_source_hash": source_hash,
            "publication_retry_token_hash": token_hash,
        },
    }

    assert sw._latest_terminal_baseline_publication_failure([failed]) is failed
    assert sw._baseline_publication_retry_authorization(failed) == ""
    assert sw._baseline_publication_retry_decision(
        [failed],
        scope_key="2026-07-10:window:model",
        in_process_failures=set(),
    ) == (True, "")
    assert sw._baseline_publication_retry_decision(
        [],
        scope_key="2026-07-10:window:model",
        in_process_failures={"2026-07-10:window:model"},
    ) == (True, "")


def test_terminal_publication_failure_allows_changed_source_or_retry_token(monkeypatch):
    failed = {
        "dispatch_status": "failed",
        "event_doc": {
            "failure_phase": "publication",
            "terminal_no_automatic_retry": True,
            "scoring_worker_source_hash": "sha256:" + "1" * 64,
            "publication_retry_token_hash": "sha256:" + "2" * 64,
        },
    }
    monkeypatch.setattr(sw, "_scoring_worker_source_hash", lambda: "sha256:" + "3" * 64)
    monkeypatch.setattr(sw, "_baseline_publication_retry_token_hash", lambda: "sha256:" + "2" * 64)
    assert sw._baseline_publication_retry_authorization(failed) == "scoring_worker_source_changed"

    monkeypatch.setattr(sw, "_scoring_worker_source_hash", lambda: "sha256:" + "1" * 64)
    monkeypatch.setattr(sw, "_baseline_publication_retry_token_hash", lambda: "sha256:" + "4" * 64)
    assert sw._baseline_publication_retry_authorization(failed) == "operator_retry_token_changed"


def test_nonpublication_failure_and_newer_assignment_do_not_trip_terminal_guard(monkeypatch):
    monkeypatch.setattr(sw, "_scoring_worker_source_hash", lambda: "sha256:" + "1" * 64)
    monkeypatch.setattr(sw, "_baseline_publication_retry_token_hash", lambda: "")
    computation_failure = {
        "dispatch_status": "failed",
        "event_doc": {
            "failure_phase": "computation",
            "terminal_no_automatic_retry": False,
        },
    }
    publication_failure = {
        "dispatch_status": "failed",
        "event_doc": {
            "failure_phase": "publication",
            "terminal_no_automatic_retry": True,
            "scoring_worker_source_hash": "sha256:" + "1" * 64,
        },
    }
    newer_assignment = {
        "dispatch_status": "assigned",
        "event_doc": {"benchmark_attempt": 2},
    }

    assert sw._baseline_publication_retry_decision(
        [computation_failure], scope_key="scope", in_process_failures=set()
    ) == (False, "")
    assert sw._baseline_publication_retry_decision(
        [newer_assignment, publication_failure], scope_key="scope", in_process_failures=set()
    ) == (False, "")


@pytest.mark.asyncio
async def test_uncaught_baseline_publication_failure_is_terminal_and_does_not_escape(monkeypatch):
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "research-lab-scorer-test"
    worker.proxy_ref_hash = "sha256:" + "a" * 64
    worker._active_baseline_context = None
    worker._baseline_publication_failures_in_process = set()
    worker._baseline_publication_failure_logged_key = None
    captured: list[dict] = []

    async def fail_at_publication():
        worker._active_baseline_context = {
            "benchmark_date": "2026-07-10",
            "benchmark_attempt": 4,
            "rolling_window_hash": "sha256:" + "b" * 64,
            "private_model_manifest_hash": "sha256:" + "c" * 64,
            "selected_icp_count": 20,
            "started_at": time.time(),
            "publication_stage": "private_bundle_insert",
        }
        raise RuntimeError(
            "research_lab_private_model_benchmark_bundles score_summary_doc_check violated"
        )

    async def capture_dispatch(**kwargs):
        captured.append(kwargs)
        return {"dispatch_event_id": "dispatch-test"}

    monkeypatch.setattr(worker, "_maybe_run_private_baseline", fail_at_publication)
    monkeypatch.setattr(sw, "create_scoring_dispatch_event", capture_dispatch)
    monkeypatch.setattr(sw, "_scoring_worker_source_hash", lambda: "sha256:" + "d" * 64)
    monkeypatch.setattr(sw, "_baseline_publication_retry_token_hash", lambda: "")

    result = await worker._run_private_baseline_contained()

    assert result["status"] == "baseline_publication_failed_terminal"
    assert result["benchmark_attempt"] == 4
    assert result["failure_stage"] == "private_bundle_insert"
    assert worker._active_baseline_context is None
    assert len(worker._baseline_publication_failures_in_process) == 1
    assert len(captured) == 1
    assert captured[0]["dispatch_status"] == "failed"
    event_doc = captured[0]["event_doc"]
    assert event_doc["failure_phase"] == "publication"
    assert event_doc["terminal_no_automatic_retry"] is True
    assert event_doc["benchmark_attempt"] == 4
    assert EVENT_DOC_BANNED_RE.search(json.dumps(event_doc, sort_keys=True)) is None


@pytest.mark.asyncio
async def test_publication_failure_still_latches_when_failed_dispatch_write_is_unavailable(monkeypatch):
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "research-lab-scorer-test"
    worker.proxy_ref_hash = None
    worker._baseline_publication_failures_in_process = set()
    worker._active_baseline_context = {
        "benchmark_date": "2026-07-10",
        "benchmark_attempt": 5,
        "rolling_window_hash": "sha256:" + "b" * 64,
        "private_model_manifest_hash": "sha256:" + "c" * 64,
        "selected_icp_count": 20,
        "started_at": time.time(),
        "publication_stage": "private_bundle_insert",
    }

    async def unavailable_dispatch(**kwargs):
        raise RuntimeError("temporary dispatch storage failure")

    monkeypatch.setattr(sw, "create_scoring_dispatch_event", unavailable_dispatch)
    monkeypatch.setattr(sw, "_scoring_worker_source_hash", lambda: "sha256:" + "d" * 64)
    monkeypatch.setattr(sw, "_baseline_publication_retry_token_hash", lambda: "")

    result = await worker._contain_private_baseline_publication_failure(
        RuntimeError("score_summary_doc_check violated")
    )

    assert result["status"] == "baseline_publication_failed_terminal"
    assert len(worker._baseline_publication_failures_in_process) == 1


@pytest.mark.asyncio
async def test_baseline_dispatch_history_is_scoped_to_date_window_and_model(monkeypatch):
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    window_hash = "sha256:" + "1" * 64
    manifest_hash = "sha256:" + "2" * 64

    async def fake_select_many(table, *, columns, filters, order_by, limit):
        assert table == "research_lab_scoring_dispatch_events"
        assert ("rolling_window_hash", window_hash) in filters
        return [
            {
                "dispatch_status": "failed",
                "event_doc": {
                    "benchmark_date": "2026-07-10",
                    "private_model_manifest_hash": manifest_hash,
                },
            },
            {
                "dispatch_status": "failed",
                "event_doc": {
                    "benchmark_date": "2026-07-09",
                    "private_model_manifest_hash": manifest_hash,
                },
            },
            {
                "dispatch_status": "failed",
                "event_doc": {
                    "benchmark_date": "2026-07-10",
                    "private_model_manifest_hash": "sha256:" + "3" * 64,
                },
            },
        ]

    monkeypatch.setattr(sw, "select_many", fake_select_many)
    rows = await worker._baseline_dispatch_history(
        today="2026-07-10",
        window_hash=window_hash,
        manifest_hash=manifest_hash,
    )

    assert len(rows) == 1
    assert rows[0]["event_doc"]["private_model_manifest_hash"] == manifest_hash
