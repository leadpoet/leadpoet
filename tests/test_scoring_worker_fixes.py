"""Tests for the scoring_worker fixes from fableanalysis.md.

Covers: bug #37 (retry classifier), §5.2-1 (baseline health diagnostics), bug #6
(claim-attempt accounting), bug #10 (unknown errors default retryable),
bug #36 write side (sanitized event error docs), bug #9 (audit event window
scoping), and the same-day baseline replacement guard.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gateway.research_lab import scoring_worker as sw
from gateway.research_lab.promotion import (
    PrivateModelLineageUnavailableError,
    PromotionPausedError,
)


EVENT_DOC_BANNED_RE = re.compile(
    r"(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|"
    r"private_repo|judge_prompt|hidden_icp|icp_plaintext|\.dkr\.ecr\.|"
    r"image_digest|private_model_manifest_doc|candidate_patch_manifest|"
    r"proxy[_-]?url|://[^/]+:[^/@]+@)",
    re.IGNORECASE,
)


def test_scored_event_serving_model_version_matches_db_leak_guard_contract():
    score_bundle = {
        "score_bundle_hash": "sha256:" + "6" * 64,
        "serving_model_version": {
            "schema_version": "research_lab_serving_model_version.v1",
            "result_role": "candidate_scoring",
            "run_id": "run-versioned",
            "ticket_id": "ticket-private",
            "candidate_id": "candidate:" + "4" * 64,
            "private_model_version_id": "private_model_version:abc",
            "evaluation_epoch": 23852,
            "benchmark_date": "2026-07-09",
            "benchmark_attempt": 1,
            "run_scope": "candidate_scoring",
            "icp_set_hash": "sha256:" + "5" * 64,
            "scoring_code_version": "qualification-company-scorer:v1",
            "evaluator_version": "leadpoet-gateway-private-scoring:v1",
            "parent_model": {
                "model_artifact_hash": "sha256:" + "1" * 64,
                "manifest_hash": "sha256:" + "2" * 64,
                "git_commit_sha": "abcdef123456",
                "manifest_uri": "s3://private-bucket/manifest.json",
                "image_digest": (
                    "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:"
                    + "7" * 64
                ),
            },
            "candidate_model": {
                "manifest_uri": "s3://private-bucket/candidate.json",
                "image_digest": (
                    "123456789012.dkr.ecr.us-east-1.amazonaws.com/candidate@sha256:"
                    + "8" * 64
                ),
            },
            "version_stamp_hash": "sha256:" + "3" * 64,
        },
    }
    event_doc = {
        "score_bundle_hash": score_bundle["score_bundle_hash"],
        "serving_model_version": sw._event_serving_model_version(score_bundle),
    }

    dumped = json.dumps(event_doc, sort_keys=True)
    assert EVENT_DOC_BANNED_RE.search(dumped) is None
    assert set(event_doc["serving_model_version"]) == {
        "schema_version",
        "result_role",
        "run_id",
        "evaluation_epoch",
        "benchmark_date",
        "benchmark_attempt",
        "run_scope",
        "icp_set_hash",
        "scoring_code_version",
        "evaluator_version",
        "model_artifact_hash",
        "manifest_hash",
        "git_commit_sha",
        "version_stamp_hash",
        "public_stamp_hash",
    }
    assert event_doc["serving_model_version"]["model_artifact_hash"] == "sha256:" + "1" * 64
    assert event_doc["serving_model_version"]["manifest_hash"] == "sha256:" + "2" * 64
    assert event_doc["serving_model_version"]["git_commit_sha"] == "abcdef123456"
    assert event_doc["serving_model_version"]["version_stamp_hash"] == "sha256:" + "3" * 64
    assert event_doc["serving_model_version"]["public_stamp_hash"].startswith("sha256:")


def test_event_serving_model_version_handles_empty_score_bundle():
    doc = sw._event_serving_model_version(None)

    assert doc["public_stamp_hash"].startswith("sha256:")
    assert EVENT_DOC_BANNED_RE.search(json.dumps(doc, sort_keys=True)) is None


def _provider_cost_trace_entry(seq: int = 1) -> dict:
    return {
        "seq": seq,
        "provider_cost_event": {
            "provider": "sd",
            "endpoint": "/profile",
            "model": "",
            "request_fingerprint": "a" * 64,
            "status_code": 200,
            "billable": True,
            "cost_usd": 0.0005,
            "cost_source": "scrapingdog_credit_map",
            "credits": 10,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cap_usd": 0.5,
            "spent_before_usd": 0.0,
            "spent_after_usd": 0.0005,
            "scope": "scope-test",
        },
    }


@pytest.mark.asyncio
async def test_provider_cost_persists_deepline_provider_label(monkeypatch):
    inserted: list[dict] = []

    async def capture_insert(table: str, row: dict) -> dict:
        assert table == "research_lab_provider_cost_events"
        inserted.append(row)
        return row

    entry = _provider_cost_trace_entry()
    entry["provider_cost_event"].update(
        {
            "provider": "deepline",
            "endpoint": "/api/v2/runs/run_123",
            "cost_source": "deepline_response_credits",
        }
    )

    monkeypatch.setattr(sw, "insert_row", capture_insert)

    await sw._persist_provider_cost_events(
        entries=[entry],
        run_type="private_baseline_rebenchmark",
        icp_ref="qualification_icp:test",
        runner_role="baseline",
    )

    assert inserted
    assert inserted[0]["provider"] == "deepline"


@pytest.mark.asyncio
async def test_provider_cost_duplicate_insert_is_idempotent(monkeypatch, caplog):
    async def duplicate_insert(table: str, row: dict) -> dict:
        assert table == "research_lab_provider_cost_events"
        raise RuntimeError(
            "postgrest.exceptions.APIError: {'message': 'duplicate key value violates unique "
            "constraint \"research_lab_provider_cost_events_pkey\"', 'code': '23505'}"
        )

    monkeypatch.setattr(sw, "insert_row", duplicate_insert)
    caplog.set_level("WARNING", logger=sw.logger.name)

    await sw._persist_provider_cost_events(
        entries=[_provider_cost_trace_entry()],
        run_type="private_baseline_rebenchmark",
        icp_ref="qualification_icp:test",
        runner_role="baseline",
    )

    assert "research_lab_provider_cost_event_insert_failed" not in caplog.text


@pytest.mark.asyncio
async def test_provider_cost_non_duplicate_insert_failure_still_warns(monkeypatch, caplog):
    async def failing_insert(table: str, row: dict) -> dict:
        assert table == "research_lab_provider_cost_events"
        raise RuntimeError("supabase timeout")

    monkeypatch.setattr(sw, "insert_row", failing_insert)
    caplog.set_level("WARNING", logger=sw.logger.name)

    await sw._persist_provider_cost_events(
        entries=[_provider_cost_trace_entry()],
        run_type="private_baseline_rebenchmark",
        icp_ref="qualification_icp:test",
        runner_role="baseline",
    )

    assert "research_lab_provider_cost_event_insert_failed" in caplog.text


def test_scoring_worker_short_fraction_timestamps_count_as_stale():
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
        assert sw._status_age_seconds(value) > 19 * 60
        assert sw._status_is_stale(value, 300)


def test_scoring_claim_predating_worker_boot_is_restart_orphan():
    worker_started_at = datetime(2026, 7, 3, 16, 24, 0, tzinfo=timezone.utc)
    row = {
        "current_evaluator_ref": "research-lab-scorer-8",
        "current_status_at": "2026-07-03T16:23:20.12345+00:00",
    }

    assert sw._claim_predates_worker_boot(
        row,
        worker_ref="research-lab-scorer-8",
        worker_started_at=worker_started_at,
        grace_seconds=30,
    )
    assert not sw._claim_predates_worker_boot(
        row,
        worker_ref="research-lab-scorer-7",
        worker_started_at=worker_started_at,
        grace_seconds=30,
    )
    assert not sw._claim_predates_worker_boot(
        {**row, "current_status_at": "2026-07-03T16:23:45.12345+00:00"},
        worker_ref="research-lab-scorer-8",
        worker_started_at=worker_started_at,
        grace_seconds=30,
    )


def test_restart_orphan_short_grace_catches_recent_pre_boot_heartbeat():
    worker_started_at = datetime(2026, 7, 3, 16, 24, 0, tzinfo=timezone.utc)
    row = {
        "current_evaluator_ref": "research-lab-scorer-8",
        "current_status_at": "2026-07-03T16:23:40+00:00",
    }

    assert not sw._claim_predates_worker_boot(
        row,
        worker_ref="research-lab-scorer-8",
        worker_started_at=worker_started_at,
        grace_seconds=30,
    )
    assert sw._claim_predates_worker_boot(
        row,
        worker_ref="research-lab-scorer-8",
        worker_started_at=worker_started_at,
        grace_seconds=5,
    )


def test_stale_claim_recovery_owner_is_stable_and_bounded():
    candidate_id = "candidate:97a10903d96c91880b35a423aa9d44d9a9593c9bc68b2e776fb531a18cb75eb0"
    owner = sw._stale_claim_recovery_owner_index(candidate_id, 25)

    assert 0 <= owner < 25
    assert sw._stale_claim_recovery_owner_index(candidate_id, 25) == owner
    assert sw._stale_claim_recovery_owner_index(candidate_id, 0) == 0


def test_candidate_claim_slot_gate_limits_claiming_workers():
    assert sw._worker_can_claim_candidate_slot(0, 8)
    assert sw._worker_can_claim_candidate_slot(7, 8)
    assert not sw._worker_can_claim_candidate_slot(8, 8)
    assert not sw._worker_can_claim_candidate_slot(24, 8)
    assert sw._worker_can_claim_candidate_slot(24, 0)


def test_active_claim_capacity_gate_blocks_at_cap():
    assert sw._active_claim_capacity_available(0, 8)
    assert sw._active_claim_capacity_available(7, 8)
    assert not sw._active_claim_capacity_available(8, 8)
    assert not sw._active_claim_capacity_available(12, 8)
    assert sw._active_claim_capacity_available(12, 0)


def test_progress_doc_completed_icp_count_variants():
    assert sw._completed_icp_count_from_progress_doc({"completed_icp_count": 3}) == 3
    assert sw._completed_icp_count_from_progress_doc({"completed_icps": "4"}) == 4
    assert (
        sw._completed_icp_count_from_progress_doc(
            {"scoring_progress": {"completed_icp_count": 5}}
        )
        == 5
    )
    assert sw._completed_icp_count_from_progress_doc({"per_icp_results": [{}, {}]}) == 2
    assert sw._completed_icp_count_from_progress_doc({"completed_icp_count": "bad"}) == 0


def test_latest_scoring_progress_from_events_prefers_largest_safe_count():
    rows = [
        {"event_doc": {"completed_icp_count": 1, "rolling_window_hash": "sha256:" + "1" * 64}},
        {
            "event_doc": {
                "scoring_progress": {
                    "completed_icp_count": 7,
                    "rolling_window_hash": "sha256:" + "2" * 64,
                }
            }
        },
        {"event_doc": {"completed_icp_count": 3}},
    ]

    summary = sw._latest_scoring_progress_from_events(rows)

    assert summary["source"] == "candidate_events"
    assert summary["checkpoint_found"] is True
    assert summary["completed_icp_count"] == 7
    assert summary["rolling_window_hash"] == "sha256:" + "2" * 64


def test_scoring_host_pressure_gate_blocks_new_claims():
    memory_pressure = sw._scoring_host_pressure_capacity(
        min_available_memory_mb=4096,
        max_load_per_cpu=4.0,
        available_memory_mb=1024,
        load_per_cpu=1.0,
    )
    assert memory_pressure["available"] is False
    assert memory_pressure["reason"] == "host_memory_pressure"

    load_pressure = sw._scoring_host_pressure_capacity(
        min_available_memory_mb=4096,
        max_load_per_cpu=4.0,
        available_memory_mb=8192,
        load_per_cpu=5.5,
    )
    assert load_pressure["available"] is False
    assert load_pressure["reason"] == "host_load_pressure"

    healthy = sw._scoring_host_pressure_capacity(
        min_available_memory_mb=4096,
        max_load_per_cpu=4.0,
        available_memory_mb=8192,
        load_per_cpu=2.0,
    )
    assert healthy["available"] is True


def test_scoring_host_pressure_gate_disables_individual_thresholds():
    assert sw._scoring_host_pressure_capacity(
        min_available_memory_mb=0,
        max_load_per_cpu=0.0,
        available_memory_mb=1,
        load_per_cpu=99.0,
    )["available"]
    assert sw._scoring_host_pressure_capacity(
        min_available_memory_mb=4096,
        max_load_per_cpu=4.0,
        available_memory_mb=None,
        load_per_cpu=None,
    )["available"]


def test_restart_orphan_recovery_is_not_blocked_by_stale_owner():
    candidate_id = "candidate:97a10903d96c91880b35a423aa9d44d9a9593c9bc68b2e776fb531a18cb75eb0"
    non_owner_index = (sw._stale_claim_recovery_owner_index(candidate_id, 25) + 1) % 25
    worker_started_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    status_at = worker_started_at - timedelta(seconds=40)
    worker_ref = f"research-lab-scorer-{non_owner_index + 1}"
    row = {
        "current_evaluator_ref": worker_ref,
        "current_status_at": status_at.isoformat(),
    }

    assert (
        sw._candidate_claim_recovery_reason(
            row,
            candidate_id=candidate_id,
            worker_ref=worker_ref,
            worker_index=non_owner_index,
            total_workers=25,
            worker_started_at=worker_started_at,
            stale_after_seconds=900,
            restart_orphan_grace_seconds=30,
        )
        == "restart_orphan"
    )


def test_stale_claim_recovery_is_limited_to_owner_worker():
    candidate_id = "candidate:2eceb2a9574ff577d014ac8a8a285799891dd0470e433af3a88ca6d7e48169e4"
    owner_index = sw._stale_claim_recovery_owner_index(candidate_id, 25)
    non_owner_index = (owner_index + 1) % 25
    row = {
        "current_evaluator_ref": "research-lab-scorer-3",
        "current_status_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }

    common = {
        "row": row,
        "candidate_id": candidate_id,
        "worker_ref": "research-lab-scorer-3",
        "total_workers": 25,
        "worker_started_at": datetime.now(timezone.utc),
        "stale_after_seconds": 900,
        "restart_orphan_grace_seconds": 30,
    }
    assert (
        sw._candidate_claim_recovery_reason(**common, worker_index=owner_index)
        == "stale_claim"
    )
    assert sw._candidate_claim_recovery_reason(**common, worker_index=non_owner_index) is None


# --- bug #37: _baseline_error_is_retryable / _runtime_error_diagnostics ---


@pytest.mark.parametrize(
    "text,expected",
    [
        # Scrapingdog 400 "Something went wrong" is transient/data-shaped.
        ("HTTPError: HTTP Error 400: scrapingdog Something went wrong or profile not found", True),
        # 400 with the scrapingdog provider marker alone.
        ("RuntimeError: scrapingdog request failed status=400", True),
        # 410 is recognized but terminal: retries did not produce usable content.
        ("HTTPError: HTTP Error 410: Gone scrapingdog profile", False),
        ("RuntimeError: provider fetch status=410 gone", False),
        # Rate limits / timeouts / 5xx retry.
        ("HTTPError: HTTP Error 429: Too Many Requests", True),
        ("HTTPError: HTTP Error 408: Request Timeout", True),
        ("HTTPError: HTTP Error 503: Service Unavailable", True),
        ("TimeoutError: read timed out", True),
        ("ConnectionResetError: connection reset by peer", True),
        # OOM / docker infra pressure retries.
        ("RuntimeError: container exit status 137", True),
        ("RuntimeError: Cannot connect to the Docker daemon", True),
        # Genuine auth / request errors stay permanent.
        ("HTTPError: HTTP Error 401: Unauthorized openrouter", False),
        ("HTTPError: HTTP Error 403: Forbidden", False),
        ("HTTPError: HTTP Error 404: Not Found exa", False),
        # Non-scrapingdog 400 without the transient marker stays permanent.
        ("HTTPError: HTTP Error 400: Bad Request openrouter invalid model", False),
        # Unknown text without transient markers is not retryable here
        # (the baseline classifier is deliberately conservative).
        ("ValueError: malformed adapter output", False),
    ],
)
def test_baseline_error_is_retryable(text: str, expected: bool):
    assert sw._baseline_error_is_retryable(text) is expected


def test_baseline_429_backoff_only_for_explicit_429(monkeypatch):
    monkeypatch.setattr(sw, "_PROVIDER_429_RETRY_BACKOFF_SECONDS", 15.0)

    assert sw._baseline_429_retry_backoff_seconds("HTTPError: status=429 too many requests") == 15.0
    assert sw._baseline_429_retry_backoff_seconds("HTTPError: status=408 request timeout") == 0.0
    assert sw._baseline_429_retry_backoff_seconds("HTTPError: status=500 internal error") == 0.0
    assert sw._baseline_429_retry_backoff_seconds("HTTPError: too many requests without status") == 0.0
    assert sw._baseline_429_retry_backoff_seconds("HTTPError: status=410 gone") == 0.0


def test_runtime_error_diagnostics_recognizes_410_and_shape():
    diagnostics = sw._runtime_error_diagnostics("HTTP Error 410: Gone scrapingdog")
    assert diagnostics["status"] == 410
    assert diagnostics["category"] == "provider_http_4xx"
    assert diagnostics["provider"] == "scrapingdog"
    assert set(diagnostics) == {"error_class", "provider", "status", "category"}


# --- bug #36 write side: sanitized event error docs ---


def test_event_error_diagnostics_is_marker_free():
    exc = RuntimeError(
        "pull 123456789.dkr.ecr.us-east-1.amazonaws.com/leadpoet failed; "
        "supabase service_role denied; judge_prompt leaked; status=500"
    )
    doc = sw._event_error_diagnostics(exc)
    # Structured shape only — never raw provider/infra text.
    assert set(doc) == {"error_class", "provider", "status", "category"}
    assert doc["error_class"] == "RuntimeError"
    assert doc["status"] == 500
    rendered = str(doc)
    assert ".dkr.ecr." not in rendered
    assert "service_role" not in rendered
    assert "judge_prompt" not in rendered


def test_safe_event_error_text_redacts_secret_markers():
    exc = RuntimeError(
        "auth sk-or-abc123 failed via https://user:hunter2@db.example.com "
        "with service_role key"
    )
    text = sw._safe_event_error_text(exc)
    assert "sk-or-" not in text
    assert "service_role" not in text
    assert "hunter2" not in text
    assert len(text) <= 500


# --- bug #6: claim-attempt accounting ---


def _rows(*pairs):
    return [{"event_type": event_type, "reason": reason} for event_type, reason in pairs]


def test_count_claim_attempts_single_assignment():
    assert sw._count_claim_attempts(_rows(("assigned", ""))) == 1


def test_count_claim_attempts_requeue_not_double_counted():
    # Old counter charged assigned + requeue = 2; a full cycle is one attempt.
    rows = _rows(("assigned", ""), ("queued", "stale_gateway_scoring_requeued"))
    assert sw._count_claim_attempts(rows) == 1


def test_count_claim_attempts_baseline_wait_refunded():
    rows = _rows(("assigned", ""), ("queued", "baseline_not_ready"))
    assert sw._count_claim_attempts(rows) == 0


def test_count_claim_attempts_mixed_history():
    rows = _rows(
        ("assigned", ""),
        ("queued", "baseline_not_ready"),  # refunded wait cycle
        ("assigned", ""),
        ("queued", "stale_gateway_scoring_requeued"),  # one genuine attempt
        ("assigned", ""),  # in-flight attempt
    )
    assert sw._count_claim_attempts(rows) == 2


def test_count_claim_attempts_never_negative():
    rows = _rows(("queued", "baseline_not_ready"))
    assert sw._count_claim_attempts(rows) == 0


# --- bug #10: unknown scoring exceptions default retryable ---


def test_failure_class_unknown_infra_error_retryable():
    class PostgrestConnectionReset(RuntimeError):
        pass

    _category, retryable = sw._candidate_scoring_failure_class(
        PostgrestConnectionReset("Server disconnected without response")
    )
    assert retryable is True


def test_failure_class_promotion_paused_retryable():
    _category, retryable = sw._candidate_scoring_failure_class(
        PrivateModelLineageUnavailableError("lineage read failed")
    )
    assert retryable is True
    _category, retryable = sw._candidate_scoring_failure_class(
        PromotionPausedError("promotion paused")
    )
    assert retryable is True


def test_failure_class_validation_errors_terminal():
    _category, retryable = sw._candidate_scoring_failure_class(
        ValueError("malformed candidate manifest")
    )
    assert retryable is False


def test_failure_class_timeout_retryable():
    category, retryable = sw._candidate_scoring_failure_class(TimeoutError("timed out"))
    assert category == "adapter_timeout"
    assert retryable is True


def test_failure_class_baseline_not_ready():
    category, retryable = sw._candidate_scoring_failure_class(
        sw.CandidateBaselineNotReady("matching_completed_private_baseline_required")
    )
    assert category == "baseline_not_ready"
    assert retryable is True


def test_failure_class_candidate_window_changed_is_baseline_not_ready():
    category, retryable = sw._candidate_scoring_failure_class(
        sw.CandidateBaselineWindowChanged(
            candidate_window_hash="sha256:" + "1" * 64,
            current_window_hash="sha256:" + "2" * 64,
            progress={"phase": "before_icp", "completed_icp_count": 16},
        )
    )
    assert category == "baseline_not_ready"
    assert retryable is True


def test_failure_class_provider_4xx_uses_baseline_classifier():
    class ProviderError(RuntimeError):
        pass

    # Scrapingdog 400 → retryable via bug #37 semantics.
    _category, retryable = sw._candidate_scoring_failure_class(
        ProviderError("scrapingdog HTTP Error 400: Something went wrong")
    )
    assert retryable is True
    # OpenRouter 403 → permanent.
    _category, retryable = sw._candidate_scoring_failure_class(
        ProviderError("openrouter HTTP Error 403: Forbidden")
    )
    assert retryable is False


# --- §5.2-1: observe-only baseline health diagnostics ---


def _summaries(*runtime_errors):
    return [
        {"icp_ref": f"icp-{i}", "diagnostics": ({"runtime_error": err} if err else {})}
        for i, err in enumerate(runtime_errors)
    ]


def test_build_baseline_health_counts_unresolved_and_gates():
    health = sw._build_baseline_health(
        per_icp_summaries=_summaries(True, True, True, None, None),
        retried=3,
        recovered=1,
        max_unresolved_icps=2,
    )
    assert health["unresolved_provider_errors"] == 3
    assert health["gate_passed"] is False
    assert health["decision"] == "observe_only"
    assert health["retried"] == 3
    assert health["recovered"] == 1
    assert health["max_unresolved_icps"] == 2


def test_build_baseline_health_passes_at_threshold():
    health = sw._build_baseline_health(
        per_icp_summaries=_summaries(True, True, None),
        retried=0,
        recovered=0,
        max_unresolved_icps=2,
    )
    assert health["gate_passed"] is True
    assert health["decision"] == "observe_only"


def test_baseline_health_gate_failure_carries_health():
    health = {"unresolved_provider_errors": 7, "gate_passed": False}
    exc = sw.BaselineHealthGateFailure("gate failed", baseline_health=health)
    assert exc.baseline_health == health


def test_baseline_gate_env_parsing(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS", "5")
    assert sw._baseline_max_unresolved_icps() == 5
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS", "not-a-number")
    assert sw._baseline_max_unresolved_icps() == 2
    monkeypatch.delenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", raising=False)
    assert sw._baseline_max_day_jump_points() is None
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "-4.5")
    assert sw._baseline_max_day_jump_points() == 4.5
    monkeypatch.delenv("RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_BASELINE_MIN_UTC_DAY_DELAY_SECONDS", raising=False)
    assert sw._baseline_min_utc_day_delay_seconds() == 900
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MIN_UTC_DAY_DELAY_SECONDS", "120")
    assert sw._baseline_min_utc_day_delay_seconds() == 120
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS", "180")
    assert sw._baseline_min_utc_day_delay_seconds() == 180
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MIN_UTC_DAY_DELAY_SECONDS", "bad")
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS", "bad")
    assert sw._baseline_min_utc_day_delay_seconds() == 900
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS", "99999")
    assert sw._baseline_min_utc_day_delay_seconds() == 86399
    monkeypatch.delenv("RESEARCH_LAB_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS", raising=False)
    assert sw._candidate_scoring_quiet_start_utc_seconds() == 84600
    monkeypatch.setenv("RESEARCH_LAB_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS", "bad")
    assert sw._candidate_scoring_quiet_start_utc_seconds() == 84600
    monkeypatch.setenv("RESEARCH_LAB_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS", "99999")
    assert sw._candidate_scoring_quiet_start_utc_seconds() == 86399


@pytest.mark.asyncio
async def test_operator_stale_parent_repair_retries_empty_reasoning_content(monkeypatch):
    from research_lab import openrouter_telemetry

    calls = []

    async def fake_call_openrouter_chat_async(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content="" if len(calls) == 1 else '{"candidates":[]}')

    monkeypatch.setattr(openrouter_telemetry, "call_openrouter_chat_async", fake_call_openrouter_chat_async)

    result = await sw._call_operator_openrouter_json(
        api_key="sk-test",
        model_id="model:test",
        messages=[{"role": "user", "content": "repair"}],
        timeout_seconds=30,
    )

    assert result == '{"candidates":[]}'
    assert len(calls) == 2
    assert calls[0]["include_reasoning"] is True
    assert calls[0]["extra_body"]
    assert calls[1]["include_reasoning"] is False
    assert calls[1]["extra_body"] is None
    assert calls[1]["stage"] == "operator_repair_no_reasoning"


@pytest.mark.asyncio
async def test_stale_parent_rebase_failure_auto_routes_to_regeneration(monkeypatch):
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "run_id": "run-1",
        "ticket_id": "ticket-1",
        "candidate_kind": "image_build",
        "candidate_build_doc": {"stale_parent_rebase": {"depth": 3}},
    }
    active_artifact = SimpleNamespace(model_artifact_hash="sha256:" + "a" * 64)
    captured = {"promotion_events": [], "evaluation_events": [], "dispatch_events": [], "recoveries": []}

    async def fake_promotion_event(**kwargs):
        captured["promotion_events"].append(kwargs)
        return kwargs

    async def fake_eval_event(**kwargs):
        captured["evaluation_events"].append(kwargs)
        return kwargs

    async def fake_dispatch_event(**kwargs):
        captured["dispatch_events"].append(kwargs)
        return kwargs

    async def fake_recover(self, candidate):
        captured["recoveries"].append(candidate["candidate_id"])
        return {"ok": True, "recovered": 1, "regenerated": 1}

    monkeypatch.setattr(sw, "create_candidate_promotion_event", fake_promotion_event)
    monkeypatch.setattr(sw, "create_candidate_evaluation_event", fake_eval_event)
    monkeypatch.setattr(sw, "create_scoring_dispatch_event", fake_dispatch_event)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_recover_stale_parent_rebase_failed_candidate",
        fake_recover,
    )

    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(stale_parent_rebase_max_depth=3),
        worker_ref="test-worker",
    )
    result = await worker._queue_stale_parent_rebase(
        candidate,
        active_artifact=active_artifact,
        candidate_parent="sha256:" + "b" * 64,
        evaluation_epoch=23777,
        elapsed_seconds=1.0,
        stage="before_scoring_parent_changed",
    )

    assert result["status"] == "stale_parent_rebase_failed"
    assert result["recovery_result"]["regenerated"] == 1
    assert captured["recoveries"] == [candidate["candidate_id"]]
    assert captured["evaluation_events"][0]["reason"] == "stale_parent_rebase_failed"


@pytest.mark.asyncio
async def test_private_baseline_waits_before_scheduled_utc_start(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 7, 5, 0, 14, 59, tzinfo=timezone.utc)

    monkeypatch.setattr(sw, "datetime", FixedDateTime)
    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(baseline_start_utc_offset_seconds=900)
    )

    result = await worker._maybe_run_private_baseline()

    assert result == {
        "status": "waiting_for_daily_icp_activation",
        "benchmark_date": "2026-07-05",
        "scheduled_start_at": "2026-07-05T00:15:00+00:00",
        "earliest_start_at": "2026-07-05T00:15:00+00:00",
        "start_offset_seconds": 900,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_time",
    (
        datetime(2026, 7, 5, 0, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 5, 0, 45, 0, tzinfo=timezone.utc),
    ),
)
async def test_private_baseline_enters_flow_on_or_after_scheduled_start(
    monkeypatch,
    current_time,
):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return current_time

    async def reached_baseline_flow(self):
        raise RuntimeError("baseline_flow_reached")

    monkeypatch.setattr(sw, "datetime", FixedDateTime)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_resolve_evaluation_epoch",
        reached_baseline_flow,
    )
    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(baseline_start_utc_offset_seconds=900)
    )

    with pytest.raises(RuntimeError, match="baseline_flow_reached"):
        await worker._maybe_run_private_baseline()


@pytest.mark.asyncio
async def test_private_baseline_parallel_resumes_completed_icps(monkeypatch):
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "baseline-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=2,
        private_baseline_retry_concurrency=1,
        private_baseline_provider_retry_rounds=0,
    )
    window = SimpleNamespace(
        benchmark_items=[
            {"icp_ref": "icp-a", "icp_hash": "hash-a"},
            {"icp_ref": "icp-b", "icp_hash": "hash-b"},
        ]
    )
    called: list[str] = []
    checkpointed: list[str] = []

    async def fake_run_baseline_icp(self, *, item, item_index, **kwargs):  # noqa: ANN001
        called.append(item["icp_ref"])
        return {
            "icp_ref": item["icp_ref"],
            "icp_hash": item["icp_hash"],
            "score": float(item_index),
            "company_count": 1,
            "sourced_count": 1,
            "diagnostics": {},
            "_item_index": item_index,
            "_retryable": False,
            "_nonempty": True,
            "_runtime_error": "",
            "_retry_backoff_seconds": 0.0,
        }

    async def checkpoint(row):  # noqa: ANN001
        checkpointed.append(row["icp_ref"])

    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_run_baseline_icp",
        fake_run_baseline_icp,
    )

    rows, stats = await worker._run_baseline_batch_inner(
        runner=object(),
        retry_runner=object(),
        scorer=object(),
        window=window,
        run_start=0.0,
        resume_results=[
            {
                "icp_ref": "icp-a",
                "icp_hash": "hash-a",
                "score": 7.0,
                "company_count": 1,
                "sourced_count": 1,
                "diagnostics": {},
            }
        ],
        icp_checkpoint=checkpoint,
    )

    assert called == ["icp-b"]
    assert checkpointed == ["icp-b"]
    assert [row["icp_ref"] for row in rows] == ["icp-a", "icp-b"]
    assert [row["score"] for row in rows] == [7.0, 2.0]
    assert stats == {"retried": 0, "recovered": 0, "unresolved": 0}


def test_private_baseline_retry_runner_gets_fresh_cost_scope(monkeypatch):
    monkeypatch.setattr(sw.time, "time", lambda: 1234.567)
    base_scope = "sha256:" + "1" * 64
    image_digest = "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "a" * 64
    retry_runner = sw.DockerPrivateModelRunner(
        sw.DockerPrivateModelSpec(
            image_digest=image_digest,
            pull_before_run=False,
            extra_env={sw.PROVIDER_COST_EVALUATION_SCOPE_ENV: base_scope},
        )
    )

    round_one = sw._retry_runner_with_provider_cost_scope(retry_runner, retry_round=1)
    round_two = sw._retry_runner_with_provider_cost_scope(retry_runner, retry_round=2)

    assert retry_runner.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV] == base_scope
    assert round_one.spec.pull_before_run is False
    assert round_two.spec.pull_before_run is False
    assert round_one.spec.image_digest == image_digest
    assert round_two.spec.image_digest == image_digest
    assert round_one.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV] != base_scope
    assert (
        round_one.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV]
        != round_two.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV]
    )


@pytest.mark.asyncio
async def test_private_baseline_retry_round_uses_fresh_cost_scope(monkeypatch):
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "baseline-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=1,
        private_baseline_retry_concurrency=1,
        private_baseline_provider_retry_rounds=1,
    )
    window = SimpleNamespace(benchmark_items=[{"icp_ref": "icp-a", "icp_hash": "hash-a"}])
    image_digest = "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "b" * 64
    first_runner = sw.DockerPrivateModelRunner(
        sw.DockerPrivateModelSpec(
            image_digest=image_digest,
            pull_before_run=False,
            extra_env={sw.PROVIDER_COST_EVALUATION_SCOPE_ENV: "sha256:" + "1" * 64},
        )
    )
    retry_runner = sw.DockerPrivateModelRunner(
        sw.DockerPrivateModelSpec(
            image_digest=image_digest,
            pull_before_run=False,
            extra_env={sw.PROVIDER_COST_EVALUATION_SCOPE_ENV: "sha256:" + "2" * 64},
        )
    )
    seen_scopes: list[str] = []

    async def fake_run_baseline_icp(self, *, runner, item, item_index, **kwargs):  # noqa: ANN001
        seen_scopes.append(runner.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV])
        retryable = len(seen_scopes) == 1
        return {
            "icp_ref": item["icp_ref"],
            "icp_hash": item["icp_hash"],
            "score": 0.0 if retryable else 3.0,
            "company_count": 0 if retryable else 1,
            "sourced_count": 0 if retryable else 1,
            "diagnostics": {},
            "_item_index": item_index,
            "_retryable": retryable,
            "_nonempty": not retryable,
            "_runtime_error": "HTTP 500" if retryable else "",
            "_retry_backoff_seconds": 0.0,
        }

    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_run_baseline_icp",
        fake_run_baseline_icp,
    )

    rows, stats = await worker._run_baseline_batch_inner(
        runner=first_runner,
        retry_runner=retry_runner,
        scorer=object(),
        window=window,
        run_start=0.0,
    )

    assert len(seen_scopes) == 2
    assert seen_scopes[0] == first_runner.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV]
    assert seen_scopes[1] != retry_runner.spec.extra_env[sw.PROVIDER_COST_EVALUATION_SCOPE_ENV]
    assert seen_scopes[1] != seen_scopes[0]
    assert rows[0]["score"] == 3.0
    assert stats == {"retried": 1, "recovered": 1, "unresolved": 0}


def test_private_baseline_checkpoint_rejects_unresolved_and_cost_blocked_rows():
    assert sw._baseline_summary_checkpointable(
        {"icp_ref": "ok", "score": 1.0, "company_count": 1, "diagnostics": {}}
    )
    assert not sw._baseline_summary_checkpointable(
        {
            "icp_ref": "runtime-error",
            "score": 0.0,
            "diagnostics": {"runtime_error": {"category": "provider"}},
        }
    )
    assert not sw._baseline_summary_checkpointable(
        {
            "icp_ref": "cost-blocked",
            "score": 0.0,
            "diagnostics": {"failure_categories": ["provider_cost_cap_blocked"]},
        }
    )
    assert sw._baseline_summary_checkpointable(
        {
            "icp_ref": "cost-blocked-with-output",
            "score": 42.0,
            "company_count": 2,
            "score_breakdowns": [{"final_score": 40.0}, {"final_score": 44.0}],
            "diagnostics": {"failure_categories": ["provider_cost_cap_blocked"]},
        }
    )


def test_private_baseline_cost_cap_is_terminal_and_preserves_partial_scores():
    for message in (
        "HTTPError: HTTP Error 402: Payment Required; status=402",
        "research_lab_provider_cost_cap_exceeded cost_cap_reached",
        "provider_cost_cap_blocked",
    ):
        assert sw._baseline_error_is_retryable(message) is False

    summary = {
        "score": 36.0,
        "company_count": 2,
        "sourced_count": 3,
        "score_breakdowns": [{"final_score": 30.0}, {"final_score": 42.0}],
        "diagnostics": {
            "provider_cost_summary": {
                "cap_blocked": True,
                "tracking_failed_count": 0,
            }
        },
    }
    sw._apply_provider_cost_baseline_outcome(summary)

    assert summary["score"] == 36.0
    assert summary["company_count"] == 2
    assert summary["diagnostics"]["provider_cost_cap_blocked"] is True


def test_private_baseline_tracking_failure_still_fails_closed():
    summary = {
        "score": 36.0,
        "company_count": 2,
        "score_breakdowns": [{"final_score": 30.0}, {"final_score": 42.0}],
        "diagnostics": {
            "provider_cost_summary": {
                "cap_blocked": False,
                "tracking_failed_count": 1,
            }
        },
    }
    sw._apply_provider_cost_baseline_outcome(summary)

    assert summary["score"] == 0.0
    assert summary["company_count"] == 0
    assert summary["diagnostics"]["provider_cost_tracking_failed"] is True


def test_candidate_baseline_target_date_respects_quiet_window():
    quiet_start = 23 * 3600 + 30 * 60

    assert (
        sw._candidate_baseline_target_date(
            datetime(2026, 7, 5, 23, 29, 59, tzinfo=timezone.utc),
            quiet_start_seconds=quiet_start,
        )
        == "2026-07-05"
    )
    assert (
        sw._candidate_baseline_target_date(
            datetime(2026, 7, 5, 23, 30, 0, tzinfo=timezone.utc),
            quiet_start_seconds=quiet_start,
        )
        == "2026-07-06"
    )
    assert (
        sw._candidate_baseline_target_date(
            datetime(2026, 7, 6, 0, 10, 0, tzinfo=timezone.utc),
            quiet_start_seconds=quiet_start,
        )
        == "2026-07-06"
    )


def _artifact(artifact_hash: str = "sha256:" + "1" * 64) -> sw.PrivateModelArtifactManifest:
    return sw.PrivateModelArtifactManifest(
        model_artifact_hash=artifact_hash,
        git_commit_sha="abcdef123456",
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model"
            "@sha256:" + "a" * 64
        ),
        config_hash="sha256:" + "b" * 64,
        component_registry_version="component-registry:v1",
        scoring_adapter_version="scoring-adapter:v1",
        manifest_uri="s3://bucket/manifest.json",
        manifest_hash="sha256:" + "c" * 64,
        signature_ref="s3://bucket/signature.sig",
    )


@pytest.mark.asyncio
async def test_daily_candidate_scoring_uses_same_day_baseline_window(monkeypatch):
    artifact = _artifact()
    baseline_window_hash = "sha256:" + "4" * 64
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig())
    reconstructed_window = SimpleNamespace(
        window_hash=baseline_window_hash,
        benchmark_id=f"research_lab:rolling_icp_window:{baseline_window_hash}",
        split_ref=f"supabase:qualification_private_icp_sets:rolling:{baseline_window_hash}",
        item_refs=("qualification_private_icp_sets:20260705:icp_001",),
        benchmark_items=({"icp_ref": "qualification_private_icp_sets:20260705:icp_001"},),
        public_doc={"rolling_window_hash": baseline_window_hash},
    )
    captured = {}

    async def fake_select_many(table, *, columns, filters, order_by, limit):
        captured["table"] = table
        captured["filters"] = filters
        return [
            {
                "benchmark_bundle_id": "private_benchmark:" + "a" * 64,
                "private_model_manifest_hash": artifact.manifest_hash,
                "rolling_window_hash": baseline_window_hash,
                "current_benchmark_status": "completed",
                "benchmark_quality": "passed",
                "evaluation_epoch": 23766,
                "created_at": "2026-07-05T00:33:46+00:00",
                "score_summary_doc": {
                    "aggregate_score": 12.0,
                    "per_icp_summaries": [{"company_count": 1}],
                    "visibility_split": {
                        "private_count": 1,
                        "items": [
                            {
                                "visibility": "public",
                                "icp_ref": "qualification_private_icp_sets:20260705:icp_001",
                                "score": 10.0,
                            },
                            {
                                "visibility": "private",
                                "icp_ref": "qualification_private_icp_sets:20260705:icp_002",
                                "score": 14.0,
                            },
                        ],
                    },
                },
            }
        ]

    async def fake_reconstruct(self, window_hash):
        captured["reconstruct_window_hash"] = window_hash
        return reconstructed_window

    monkeypatch.setattr(sw, "select_many", fake_select_many)
    monkeypatch.setattr(sw.ResearchLabGatewayScoringWorker, "_reconstruct_rolling_window", fake_reconstruct)

    window, gate = await worker._daily_candidate_scoring_window_and_gate(
        artifact=artifact,
        now=datetime(2026, 7, 5, 16, 0, tzinfo=timezone.utc),
    )

    assert window is reconstructed_window
    assert captured["table"] == "research_lab_private_model_benchmark_current"
    assert ("benchmark_date", "2026-07-05") in captured["filters"]
    assert ("private_model_manifest_hash", artifact.manifest_hash) in captured["filters"]
    assert captured["reconstruct_window_hash"] == baseline_window_hash
    assert gate["rolling_window_hash"] == baseline_window_hash
    assert gate["baseline_private_holdout_icp_count"] == 1


@pytest.mark.asyncio
async def test_candidate_start_gate_allows_before_quiet_when_today_baseline_exists(monkeypatch):
    artifact = _artifact()
    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(candidate_scoring_quiet_start_utc_seconds=84600)
    )
    captured = {}

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):
        captured["target_date"] = now.date().isoformat()
        return (
            SimpleNamespace(window_hash="sha256:" + "7" * 64),
            {"baseline_benchmark_bundle_id": "private_benchmark:test"},
        )

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    gate = await worker._candidate_scoring_start_gate(
        now=datetime(2026, 7, 5, 23, 29, 59, tzinfo=timezone.utc)
    )

    assert gate["available"] is True
    assert gate["target_benchmark_date"] == "2026-07-05"
    assert captured["target_date"] == "2026-07-05"


@pytest.mark.asyncio
async def test_candidate_start_gate_blocks_at_quiet_until_next_baseline(monkeypatch):
    artifact = _artifact()
    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(candidate_scoring_quiet_start_utc_seconds=84600)
    )
    captured = {}

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):
        captured["target_date"] = now.date().isoformat()
        raise sw.CandidateBaselineNotReady("baseline missing")

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    quiet_gate = await worker._candidate_scoring_start_gate(
        now=datetime(2026, 7, 5, 23, 30, 0, tzinfo=timezone.utc)
    )
    midnight_gate = await worker._candidate_scoring_start_gate(
        now=datetime(2026, 7, 6, 0, 10, 0, tzinfo=timezone.utc)
    )

    assert quiet_gate["available"] is False
    assert quiet_gate["reason"] == "candidate_scoring_next_daily_baseline_not_ready"
    assert quiet_gate["target_benchmark_date"] == "2026-07-06"
    assert midnight_gate["available"] is False
    assert midnight_gate["reason"] == "candidate_scoring_daily_baseline_not_ready"
    assert midnight_gate["target_benchmark_date"] == "2026-07-06"
    assert captured["target_date"] == "2026-07-06"


@pytest.mark.asyncio
async def test_candidate_claim_quiet_hold_writes_no_assignment(monkeypatch):
    candidate = {
        "candidate_id": "candidate:" + "8" * 64,
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "current_reason": "",
        "current_status_at": "2026-07-05T23:30:00+00:00",
    }
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig(), worker_ref="test-worker")
    events = []

    async def fake_select_many(table, *, columns, filters, order_by, limit):
        assert table == "research_lab_candidate_evaluation_current"
        return [candidate]

    async def fake_start_gate(self):
        return {
            "available": False,
            "reason": "candidate_scoring_next_daily_baseline_not_ready",
            "now_utc": "2026-07-05T23:30:00+00:00",
            "target_benchmark_date": "2026-07-06",
            "quiet_start_utc_seconds": 84600,
            "private_model_manifest_hash": "sha256:" + "c" * 64,
        }

    async def fake_create_event(**kwargs):  # pragma: no cover - must not assign
        events.append(kwargs)
        raise AssertionError("quiet hold must not write candidate assignment")

    monkeypatch.setattr(sw, "select_many", fake_select_many)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_candidate_scoring_start_gate",
        fake_start_gate,
    )
    monkeypatch.setattr(sw, "create_candidate_evaluation_event", fake_create_event)

    claimed = await worker._claim_next_candidate()

    assert claimed is None
    assert events == []


@pytest.mark.asyncio
async def test_candidate_freshness_passes_when_parent_and_window_match(monkeypatch):
    artifact = _artifact()
    window_hash = "sha256:" + "2" * 64
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig())
    captured = {}

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):
        captured["artifact_hash"] = artifact.manifest_hash
        return SimpleNamespace(window_hash=window_hash), {"baseline_benchmark_bundle_id": "baseline:test"}

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    await worker._check_candidate_scoring_freshness(
        parent_artifact=artifact,
        candidate_window_hash=window_hash,
        progress={"phase": "before_icp", "completed_icp_count": 3},
    )
    assert captured["artifact_hash"] == artifact.manifest_hash


@pytest.mark.asyncio
async def test_candidate_freshness_stale_parent_takes_precedence(monkeypatch):
    artifact = _artifact("sha256:" + "1" * 64)
    active_artifact = _artifact("sha256:" + "9" * 64)
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig())

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=active_artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):  # pragma: no cover - parent check should win first
        raise AssertionError("baseline gate should not run after stale parent")

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    with pytest.raises(sw.StaleParentDuringScoring) as raised:
        await worker._check_candidate_scoring_freshness(
            parent_artifact=artifact,
            candidate_window_hash="sha256:" + "2" * 64,
            progress={"phase": "before_icp", "completed_icp_count": 16},
        )

    assert raised.value.active_artifact == active_artifact
    assert raised.value.candidate_parent == artifact.model_artifact_hash
    assert raised.value.progress["completed_icp_count"] == 16


@pytest.mark.asyncio
async def test_candidate_freshness_waits_when_candidate_baseline_missing(monkeypatch):
    artifact = _artifact()
    candidate_window = "sha256:" + "2" * 64
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig())

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):
        raise sw.CandidateBaselineNotReady("matching baseline missing")

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    with pytest.raises(sw.CandidateBaselineNotReady) as raised:
        await worker._check_candidate_scoring_freshness(
            parent_artifact=artifact,
            candidate_window_hash=candidate_window,
            progress={
                "phase": "before_icp",
                "next_icp_index": 16,
                "completed_icp_count": 16,
                "icp_ref": "icp-16",
            },
        )

    assert "matching baseline missing" in str(raised.value)


@pytest.mark.asyncio
async def test_candidate_freshness_waits_when_current_window_unavailable(monkeypatch):
    artifact = _artifact()
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig())

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):
        raise RuntimeError("baseline lookup unavailable")

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    with pytest.raises(sw.CandidateBaselineNotReady) as raised:
        await worker._check_candidate_scoring_freshness(
            parent_artifact=artifact,
            candidate_window_hash="sha256:" + "2" * 64,
            progress={"phase": "before_icp", "completed_icp_count": 7},
        )

    assert "candidate_daily_baseline_unavailable_during_candidate_scoring" in str(raised.value)


@pytest.mark.asyncio
async def test_candidate_freshness_raises_when_daily_window_changed(monkeypatch):
    artifact = _artifact()
    old_window = "sha256:" + "2" * 64
    current_window = "sha256:" + "3" * 64
    worker = sw.ResearchLabGatewayScoringWorker(sw.ResearchLabGatewayConfig())

    async def fake_load_active(_config, *, register_bootstrap):
        return SimpleNamespace(artifact=artifact)

    async def fake_daily_window_and_gate(self, *, artifact, now=None):
        return SimpleNamespace(window_hash=current_window), {"baseline_benchmark_bundle_id": "baseline:test"}

    monkeypatch.setattr(sw, "load_active_private_model", fake_load_active)
    monkeypatch.setattr(
        sw.ResearchLabGatewayScoringWorker,
        "_daily_candidate_scoring_window_and_gate",
        fake_daily_window_and_gate,
    )

    with pytest.raises(sw.CandidateBaselineWindowChanged) as raised:
        await worker._check_candidate_scoring_freshness(
            parent_artifact=artifact,
            candidate_window_hash=old_window,
            progress={"phase": "before_icp", "completed_icp_count": 16},
        )

    assert raised.value.candidate_window_hash == old_window
    assert raised.value.current_window_hash == current_window
    assert raised.value.progress["completed_icp_count"] == 16


def test_candidate_baseline_wait_event_doc_includes_window_progress():
    exc = sw.CandidateBaselineWindowChanged(
        candidate_window_hash="sha256:" + "2" * 64,
        current_window_hash="sha256:" + "3" * 64,
        progress={
            "phase": "before_icp",
            "next_icp_index": 16,
            "completed_icp_count": 16,
            "icp_ref": "icp-16",
        },
    )

    doc = sw._candidate_baseline_wait_event_doc(exc)

    assert doc["baseline_wait_reason"] == "rolling_window_changed_during_candidate_scoring"
    assert doc["candidate_window_hash"] == "sha256:" + "2" * 64
    assert doc["current_window_hash"] == "sha256:" + "3" * 64
    assert doc["stale_window_progress"] == {
        "phase": "before_icp",
        "completed_icp_count": 16,
        "next_icp_index": 16,
        "icp_ref": "icp-16",
    }


# --- bug #9: audit event fetches are window-scoped ---


def _worker_stub():
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    return worker


def test_audit_event_window_filters_default(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_AUDIT_EVENT_WINDOW_DAYS", raising=False)
    filters = sw.ResearchLabGatewayScoringWorker._audit_event_window_filters(_worker_stub())
    assert len(filters) == 1
    column, op, _cutoff = filters[0]
    assert column == "created_at"
    assert op == "gte"


def test_audit_event_window_filters_opt_out(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_AUDIT_EVENT_WINDOW_DAYS", "0")
    filters = sw.ResearchLabGatewayScoringWorker._audit_event_window_filters(_worker_stub())
    assert filters == ()


def test_audit_select_limits_default_and_env(monkeypatch):
    worker = _worker_stub()
    monkeypatch.delenv("RESEARCH_LAB_AUDIT_SELECT_MAX_ROWS", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_AUDIT_SELECT_BATCH_SIZE", raising=False)
    assert sw.ResearchLabGatewayScoringWorker._audit_select_limits(worker) == (10000, 500)

    monkeypatch.setenv("RESEARCH_LAB_AUDIT_SELECT_MAX_ROWS", "2500")
    monkeypatch.setenv("RESEARCH_LAB_AUDIT_SELECT_BATCH_SIZE", "800")
    assert sw.ResearchLabGatewayScoringWorker._audit_select_limits(worker) == (2500, 800)

    monkeypatch.setenv("RESEARCH_LAB_AUDIT_SELECT_MAX_ROWS", "20")
    monkeypatch.setenv("RESEARCH_LAB_AUDIT_SELECT_BATCH_SIZE", "99999")
    assert sw.ResearchLabGatewayScoringWorker._audit_select_limits(worker) == (1000, 1000)


# --- gate contract: holdout gate carries per-ICP baseline scores (§0-N2) ---


def test_private_holdout_gate_carries_per_icp_baseline_scores():
    row = {
        "benchmark_bundle_id": "bundle-1",
        "rolling_window_hash": "window-hash",
        "private_model_manifest_hash": "manifest-hash",
        "score_summary_doc": {
            "aggregate_score": 12.5,
            "visibility_split": {
                "private_count": 2,
                "items": [
                    {"icp_ref": "icp-a", "visibility": "public", "score": 10.0},
                    {"icp_ref": "icp-b", "visibility": "private", "score": 15.0},
                    {"icp_ref": "icp-c", "visibility": "private", "score": 20.0},
                ],
            },
        },
    }
    gate = sw._private_holdout_gate_from_baseline_row(row)
    assert gate is not None
    assert gate["baseline_per_icp_scores"] == {
        "icp-a": 10.0,
        "icp-b": 15.0,
        "icp-c": 20.0,
    }
    assert gate["baseline_public_score"] == 10.0
    assert gate["baseline_private_holdout_icp_count"] == 2
