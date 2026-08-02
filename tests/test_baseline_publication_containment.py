from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace

import pytest

from gateway.research_lab import scoring_worker as sw
from leadpoet_canonical.attested_v2 import sha256_json as attested_v2_sha256_json
from research_lab.eval.baseline_summary import build_baseline_score_summary


EVENT_DOC_BANNED_RE = re.compile(
    r"(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|"
    r"private_repo|judge_prompt|hidden_icp|icp_plaintext|\.dkr\.ecr\.|"
    r"image_digest|private_model_manifest_doc|candidate_patch_manifest|"
    r"proxy[_-]?url|://[^/]+:[^/@]+@)",
    re.IGNORECASE,
)


def _persisted_baseline_publication_fixture(
    *,
    current_status="completed",
    unicode_diagnostics: bool = False,
):
    artifact = sw.PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "1" * 64,
        git_commit_sha="2" * 40,
        image_digest="repo@sha256:" + "3" * 64,
        config_hash="sha256:" + "4" * 64,
        component_registry_version="v1",
        scoring_adapter_version="v1",
        manifest_uri="s3://private/model.json",
        manifest_hash="sha256:" + "5" * 64,
        signature_ref="kms://model-signature",
        build_id="build-1",
    )
    items = []
    summaries = []
    for index in range(10):
        digest = "sha256:" + f"{index:x}" * 64
        items.append(
            {
                "icp_ref": f"icp:{index}",
                "icp_hash": digest,
                "set_id": 1,
                "day_index": 1,
                "day_rank": index + 1,
                "icp": {"industry": f"industry-{index}"},
            }
        )
        summaries.append(
            {
                "icp_ref": f"icp:{index}",
                "icp_hash": digest,
                "score": float(index + 1),
                "company_count": 1,
                "diagnostics": (
                    {"provider_note": "München – 東京"}
                    if unicode_diagnostics and index == 0
                    else {}
                ),
            }
        )
    window_hash = "sha256:" + "6" * 64
    result = build_baseline_score_summary(
        artifact_manifest=artifact.to_dict(),
        benchmark_date="2026-07-10",
        benchmark_attempt=2,
        rolling_window_hash=window_hash,
        evaluation_epoch=42,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
        public_total_icps=3,
        public_weak_total=2,
        retried=1,
        recovered=1,
        max_unresolved_icps=0,
        day_jump_points=None,
        elapsed_seconds=12.5,
    )
    payload = {
        "benchmark_date": "2026-07-10",
        "private_model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "rolling_window_hash": window_hash,
        "evaluation_epoch": 42,
        "benchmark_attempt": 2,
        "benchmark_quality": "passed",
        "aggregate_score": result["aggregate_score"],
        "scoring_worker_ref": "research-lab-scorer-1",
        "proxy_ref_hash": None,
        "signature_ref": "kms://score-signature",
        "score_summary_doc": result["score_summary_doc"],
    }
    bundle_hash = sw.canonical_hash(payload)
    row = {
        "benchmark_bundle_id": "private_benchmark:" + bundle_hash.split(":", 1)[1],
        "schema_version": result["score_summary_doc"]["schema_version"],
        **payload,
        "benchmark_bundle_hash": bundle_hash,
        "anchored_hash": bundle_hash,
        "current_benchmark_status": current_status,
    }
    window = SimpleNamespace(
        window_hash=window_hash,
        benchmark_items=tuple(items),
        item_refs=tuple(item["icp_ref"] for item in items),
    )
    return artifact, window, result, row


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


@pytest.mark.asyncio
async def test_persisted_baseline_repairs_post_bundle_publication_without_rescoring(
    monkeypatch,
):
    artifact, window, protected_result, row = _persisted_baseline_publication_fixture(
        unicode_diagnostics=True
    )
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.config = SimpleNamespace(
        public_benchmark_public_icps_per_day=3,
        public_benchmark_public_weak_per_day=2,
        public_benchmark_public_total_icps=3,
        public_benchmark_public_weak_total=2,
    )
    worker.worker_ref = "research-lab-scorer-restarted"
    worker.proxy_ref_hash = None
    worker._active_baseline_context = None
    worker._baseline_publication_failures_in_process = set()
    worker._baseline_publication_failure_logged_key = None
    captured = {
        "bundle": [],
        "links": [],
        "dispatch": [],
        "report": [],
        "audit": [],
    }
    summary_hash = attested_v2_sha256_json(protected_result["score_summary_doc"])
    assert summary_hash != sw.canonical_hash(protected_result["score_summary_doc"])
    root_receipt = {
        "receipt_hash": "sha256:" + "7" * 64,
        "role": "gateway_scoring",
        "purpose": "research_lab.rebenchmark.v2",
        "status": "succeeded",
        "epoch_id": 42,
        "artifact_root": sw.merkle_root(
            (summary_hash,),
            domain="leadpoet-artifact-v2",
        ),
        "output_root": sw.sha256_json(protected_result),
    }

    async def recover_bundle(**kwargs):
        captured["bundle"].append(kwargs)
        return dict(row), {"benchmark_status": "completed"}

    async def resolve_lineage(**kwargs):
        assert kwargs["artifact_hash"] == summary_hash
        return root_receipt, [root_receipt]

    async def select_bundle(*args, **kwargs):
        if args[0] == "research_lab_private_model_benchmark_current":
            return dict(row)
        assert args[0] == "research_lab_public_benchmark_report_current"
        return {
            "report_id": "public-report-1",
            "report_hash": "sha256:" + "8" * 64,
            "current_report_status": "published",
        }

    async def persist_links(outcome, *, artifact_links):
        captured["links"].append((outcome, artifact_links))
        return "persisted"

    async def select_dispatch(*args, **kwargs):
        assert args[0] == "research_lab_scoring_dispatch_events"
        return []

    async def create_dispatch(**kwargs):
        captured["dispatch"].append(kwargs)
        return {"dispatch_event_id": kwargs["dispatch_event_id"]}

    async def create_report(**kwargs):
        captured["report"].append(kwargs)
        return {
            "report_id": "public-report-1",
            "report_hash": "sha256:" + "8" * 64,
        }, {"report_status": "published"}

    async def write_audit(epoch):
        captured["audit"].append(epoch)

    async def emit_event(*args, **kwargs):
        return None

    monkeypatch.setattr(sw, "legacy_v1_enabled", lambda: False)
    monkeypatch.setattr(sw, "create_private_model_benchmark_bundle", recover_bundle)
    monkeypatch.setattr(sw, "select_one", select_bundle)
    monkeypatch.setattr(sw, "resolve_attested_artifact_lineage", resolve_lineage)
    monkeypatch.setattr(sw, "persist_attested_outcome_artifact_links", persist_links)
    monkeypatch.setattr(sw, "select_many", select_dispatch)
    monkeypatch.setattr(sw, "create_scoring_dispatch_event", create_dispatch)
    monkeypatch.setattr(sw, "create_public_benchmark_report", create_report)
    monkeypatch.setattr(sw, "emit_run_event", emit_event)
    worker._write_audit_bundle_inner = write_audit

    publication = await worker._publish_private_baseline_bundle(
        bundle=row,
        window=window,
        artifact=artifact,
        expected_policy_hash="",
        attested_baseline_outcome=None,
        baseline_telemetry_session=None,
        publication_scope_key="scope",
        start=time.time(),
        recover_existing=True,
    )

    assert captured["bundle"][0]["score_summary_doc"] == row["score_summary_doc"]
    assert captured["links"][0][0]["execution_receipt"] == root_receipt
    assert captured["links"][0][1][0]["artifact_hash"] == summary_hash
    assert captured["dispatch"][0]["event_doc"]["publication_recovered_after_restart"] is True
    assert captured["report"][0]["aggregate_score"] == row["aggregate_score"]
    assert captured["audit"] == [42]
    assert publication["public_report"]["report_id"] == "public-report-1"


def test_baseline_recovery_returns_exact_v2_unicode_summary_hash():
    artifact, window, protected_result, row = _persisted_baseline_publication_fixture(
        unicode_diagnostics=True
    )

    validated = sw._validate_private_baseline_publication_bundle(
        row,
        artifact=artifact,
        window=window,
        expected_policy_hash="",
    )

    expected = attested_v2_sha256_json(protected_result["score_summary_doc"])
    assert validated["score_summary_hash"] == expected
    assert expected != sw.canonical_hash(protected_result["score_summary_doc"])


@pytest.mark.asyncio
async def test_bundle_event_recovery_rejects_newer_failed_event(monkeypatch):
    artifact, window, _protected_result, row = _persisted_baseline_publication_fixture(
        current_status=None
    )
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.config = SimpleNamespace(
        public_benchmark_public_icps_per_day=3,
        public_benchmark_public_weak_per_day=2,
        public_benchmark_public_total_icps=3,
        public_benchmark_public_weak_total=2,
    )
    worker._active_baseline_context = None

    async def recover_bundle(**kwargs):
        return dict(row), {"benchmark_status": "failed"}

    async def select_bundle(*args, **kwargs):
        return {**row, "current_benchmark_status": "failed"}

    monkeypatch.setattr(sw, "create_private_model_benchmark_bundle", recover_bundle)
    monkeypatch.setattr(sw, "select_one", select_bundle)

    with pytest.raises(RuntimeError, match="not durably completed"):
        await worker._publish_private_baseline_bundle(
            bundle=row,
            window=window,
            artifact=artifact,
            expected_policy_hash="",
            attested_baseline_outcome=None,
            baseline_telemetry_session=None,
            publication_scope_key="scope",
            start=time.time(),
            recover_existing=True,
        )


@pytest.mark.asyncio
async def test_incomplete_bundle_event_enters_publication_recovery(monkeypatch):
    artifact, window, _protected_result, row = _persisted_baseline_publication_fixture(
        current_status=None
    )
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    policy = SimpleNamespace(enabled=False, mode="off")
    worker.config = SimpleNamespace(
        baseline_start_utc_offset_seconds=0,
        lab_champion_eval_days=1,
        lab_champion_icps_per_day=10,
        scoring_worker_index=0,
        scoring_worker_total_workers=1,
        scoring_worker_allow_partial_icp_window=False,
        conditional_validation_policy=lambda: policy,
    )
    worker.worker_ref = "research-lab-scorer-restarted"
    worker.proxy_ref_hash = None
    worker._baseline_already_logged_date = None
    worker._baseline_publication_verified_keys = set()
    captured = []

    async def resolve_epoch():
        return 42

    async def sync_repo(*args, **kwargs):
        return {"ok": True, "repo_main_sha": artifact.git_commit_sha}

    async def fetch_window(**kwargs):
        return window

    async def load_model(*args, **kwargs):
        return SimpleNamespace(artifact=artifact)

    async def select_rows(*args, **kwargs):
        return [row]

    async def publish(**kwargs):
        captured.append(kwargs)
        return {"public_report": {"report_id": "public-report-1"}}

    monkeypatch.setattr(sw, "sync_active_model_to_repo_head", sync_repo)
    monkeypatch.setattr(sw, "fetch_rolling_icp_window", fetch_window)
    monkeypatch.setattr(sw, "load_active_private_model", load_model)
    monkeypatch.setattr(sw, "select_many", select_rows)
    worker._resolve_evaluation_epoch = resolve_epoch
    worker._publish_private_baseline_bundle = publish

    result = await worker._maybe_run_private_baseline()

    assert result["status"] == "already_benchmarked"
    assert len(captured) == 1
    assert captured[0]["recover_existing"] is True
    assert captured[0]["bundle"]["current_benchmark_status"] is None


def test_publication_recovery_fails_closed_on_changed_persisted_score():
    artifact, window, _protected_result, row = _persisted_baseline_publication_fixture()
    row = {**row, "score_summary_doc": dict(row["score_summary_doc"])}
    row["score_summary_doc"]["aggregate_score"] += 1.0

    with pytest.raises(RuntimeError, match="aggregate differs"):
        sw._validate_private_baseline_publication_bundle(
            row,
            artifact=artifact,
            window=window,
            expected_policy_hash="",
        )
