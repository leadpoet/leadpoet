"""Semantic restart/weight telemetry contracts.

These tests intentionally replace the SDK boundary with in-memory call
collectors. They exercise stable incident classification and bounded metadata
without network access.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from leadpoet_observability import sentry_operations


@pytest.fixture(autouse=True)
def _reset_operation_state():
    sentry_operations._reset_for_tests()
    yield
    sentry_operations._reset_for_tests()


EXPECTED_FAILURE_CODES = {
    "release.source_tree_mismatch",
    "release.schema_contract_mismatch",
    "release.channel_unavailable",
    "release.builder_resource_exhausted",
    "release.artifact_auth_failed",
    "runtime.enclave_relay_unavailable",
    "authority.dependency_unreadable",
    "restart.stage_deadline_exceeded",
    "weight.allocation_authority_missing",
    "weight.settlement_history_incomplete",
    "weight.ancestry_bounds_exceeded",
    "weight.frontier_source_invalid",
    "weight.gateway_endpoint_unavailable",
    "weight.block_drift_exhausted",
    "weight.authoritative_result_invalid",
    "weight.sdk_response_invalid",
    "weight.chain_transport_poisoned",
    "weight.finalization_missing",
    "release.pcr0_mismatch",
    "weight.bundle_divergence",
    "restart.terminal_failure",
}


def test_failure_code_inventory_is_stable_and_complete():
    assert sentry_operations.INCIDENT_FAILURE_CODES == EXPECTED_FAILURE_CODES


@pytest.mark.parametrize(
    ("signature", "expected"),
    (
        ("worker import preflight failed: staged packages are out of sync", "release.source_tree_mismatch"),
        ("PGRST204 benchmark_attempt column missing from schema cache", "release.schema_contract_mismatch"),
        ("approved V2 release is not published", "release.channel_unavailable"),
        ("No space left on device while builder reclaimed stale mount", "release.builder_resource_exhausted"),
        ("encrypted storage document authentication failed", "release.artifact_auth_failed"),
        ("vsock connection refused because enclave relay stopped", "runtime.enclave_relay_unavailable"),
        ("chain WebSocket connection failed: authority unavailable", "authority.dependency_unreadable"),
        ("stage deadline exceeded after 300 attempts", "restart.stage_deadline_exceeded"),
        ("historical compute fallback lacks finalized allocation authority", "weight.allocation_authority_missing"),
        ("chain realized settlement history is incomplete", "weight.settlement_history_incomplete"),
        ("receipt ancestry frame limit exceeded", "weight.ancestry_bounds_exceeded"),
        ("allocation_frontier_bootstrap_source_invalid", "weight.frontier_source_invalid"),
        ("pinned_gateway_request_failed endpoint=v2_authority", "weight.gateway_endpoint_unavailable"),
        ("block drift is too large", "weight.block_drift_exhausted"),
        ("authoritative weight result fields are invalid", "weight.authoritative_result_invalid"),
        ("cannot unpack non-iterable ExtrinsicResponse object", "weight.sdk_response_invalid"),
        ("Unable to reconnect because there are currently open subscriptions", "weight.chain_transport_poisoned"),
        ("timelocked commitment missing after broadcast", "weight.finalization_missing"),
        ("expected PCR0 mismatch observed PCR0", "release.pcr0_mismatch"),
        ("canonical bundle hash mismatch for auditor", "weight.bundle_divergence"),
    ),
)
def test_each_incident_signature_maps_to_semantic_failure_code(signature, expected):
    error = RuntimeError(signature)
    assert (
        sentry_operations.failure_code_for_exception(
            error, default="restart.terminal_failure"
        )
        == expected
    )


def test_safe_fields_are_allowlisted_bounded_and_do_not_leak_secrets():
    fields = sentry_operations.safe_fields(
        {
            "candidate_sha": "ab" * 20,
            "bundle_hash": "sha256:" + "cd" * 32,
            "blocked_stages": ["signing"] * 100,
            "dependency": "supabase",
            "authorization": "Bearer seeded-secret",
            "provider_payload": "x" * 100_000,
            "query_fingerprint": "person@example.com " + "x" * 1_000,
        }
    )
    encoded = repr(fields)
    assert fields["candidate_sha"] == "ab" * 20
    assert fields["bundle_hash"] == "sha256:" + "cd" * 32
    assert len(fields["blocked_stages"]) == 40
    assert "authorization" not in fields
    assert "provider_payload" not in fields
    assert "person@example.com" not in encoded
    assert len(fields["query_fingerprint"]) <= 256


def test_correlations_are_deterministic_across_process_boundaries():
    runtime_sha = "ab" * 20
    bundle_hash = "sha256:" + "cd" * 32
    expected_release = sentry_operations.release_correlation_id(runtime_sha)
    expected_weight = sentry_operations.weight_correlation_id(
        runtime_sha=runtime_sha,
        netuid=71,
        epoch_id=24307,
        bundle_hash=bundle_hash,
    )
    assert expected_release == sentry_operations.release_correlation_id(runtime_sha)
    assert expected_weight == sentry_operations.weight_correlation_id(
        runtime_sha=runtime_sha,
        netuid="71",
        epoch_id="24307",
        bundle_hash="sha256:" + "ef" * 32,
    )
    assert sentry_operations._trace_id(
        {"weight_correlation_id": expected_weight}
    ) == sentry_operations._trace_id({"weight_correlation_id": expected_weight})


def test_correlations_match_in_independent_python_processes():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import json
from leadpoet_observability.sentry_operations import (
    release_correlation_id,
    weight_correlation_id,
)
sha = 'ab' * 20
print(json.dumps({
    'release': release_correlation_id(sha),
    'weight': weight_correlation_id(runtime_sha=sha, netuid=71, epoch_id=24307),
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
            cwd=repo_root,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    parsed = json.loads(outputs[0])
    assert parsed["release"].startswith("release:")
    assert parsed["weight"].startswith("weight:")


def test_retry_storm_emits_breadcrumbs_then_one_terminal_event(monkeypatch):
    breadcrumbs = []
    events = []
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "add_sentry_breadcrumb",
        lambda **kwargs: breadcrumbs.append(kwargs),
    )
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "capture_sentry_failure",
        lambda **kwargs: events.append(kwargs) or True,
    )
    context = {
        "runtime_sha": "ab" * 20,
        "restart_invocation_id": "restart:test",
    }
    sentry_operations.configure_sentry_context(**context)
    for attempt in range(1, 101):
        sentry_operations.record_retry(
            "release.channel_unavailable",
            component="gateway",
            stage="release_acquisition",
            attempt=attempt,
            attempts=100,
        )
    assert sentry_operations.capture_failure(
        "release.channel_unavailable",
        component="gateway",
        stage="release_acquisition",
        attempts=100,
    )
    assert not sentry_operations.capture_failure(
        "release.channel_unavailable",
        component="gateway",
        stage="restart_wrapper",
        attempts=100,
    )
    assert len(breadcrumbs) == 100
    assert len(events) == 1
    assert events[0]["failure_code"] == "release.channel_unavailable"


def test_terminal_limiter_is_scoped_to_epoch_without_explicit_correlation(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "capture_sentry_failure",
        lambda **kwargs: events.append(kwargs) or True,
    )
    common = {
        "component": "validator",
        "stage": "submission_finalization",
        "runtime_sha": "ab" * 20,
        "netuid": 71,
    }
    assert sentry_operations.capture_failure(
        "weight.finalization_missing", epoch_id=24307, **common
    )
    assert not sentry_operations.capture_failure(
        "weight.finalization_missing", epoch_id=24307, **common
    )
    assert sentry_operations.capture_failure(
        "weight.finalization_missing", epoch_id=24308, **common
    )
    assert len(events) == 2


def test_stage_uses_deterministic_trace_and_preserves_application_error(monkeypatch):
    spans = []

    @contextmanager
    def _span(**kwargs):
        spans.append(kwargs)
        yield object()

    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap, "start_sentry_span", _span
    )
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "add_sentry_breadcrumb",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "record_sentry_distribution",
        lambda *args, **kwargs: None,
    )
    correlation = "weight:" + "ab" * 32
    original = RuntimeError("application failure")
    with pytest.raises(RuntimeError) as raised:
        with sentry_operations.sentry_stage(
            component="validator",
            operation="weight_submission",
            stage="broadcast",
            weight_correlation_id=correlation,
        ):
            raise original
    assert raised.value is original
    assert spans[0]["trace_id"] == sentry_operations._trace_id(
        {"weight_correlation_id": correlation}
    )


def test_completed_stage_emits_one_correlated_sampled_transaction(monkeypatch):
    spans = []

    @contextmanager
    def _span(**kwargs):
        spans.append(kwargs)
        yield object()

    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap, "start_sentry_span", _span
    )
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "add_sentry_breadcrumb",
        lambda **kwargs: None,
    )
    correlation = "weight:" + "ab" * 32
    sentry_operations.record_stage(
        component="auditor-validator",
        stage="submission_success",
        status="passed",
        duration_seconds=1.25,
        weight_correlation_id=correlation,
        bundle_hash="sha256:" + "cd" * 32,
    )
    assert len(spans) == 1
    assert spans[0]["trace_id"] == sentry_operations._trace_id(
        {"weight_correlation_id": correlation}
    )
    assert spans[0]["data"]["bundle_hash"] == "sha256:" + "cd" * 32


def _write_ledger(path: Path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_restart_summary_reports_first_failure_and_missing_milestones(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "restart.jsonl"
    _write_ledger(
        ledger,
        (
            {"stage": "release_ready", "status": "passed", "elapsed_seconds": 2},
            {"stage": "pre_shutdown_checks_complete", "status": "passed", "elapsed_seconds": 5},
            {"stage": "ancestry_postcheckpoint", "status": "failed", "elapsed_seconds": 17},
        ),
    )
    evidence = tmp_path / "ancestry.log"
    evidence.write_text(
        "chain realized settlement history is incomplete", encoding="utf-8"
    )
    failures = []
    monkeypatch.setattr(
        sentry_operations, "capture_failure", lambda *args, **kwargs: failures.append((args, kwargs))
    )
    monkeypatch.setattr(sentry_operations, "record_stage", lambda **kwargs: None)
    sentry_operations.emit_restart_summary(
        component="gateway",
        status="failed",
        stage="ancestry_postcheckpoint",
        ledger_path=ledger,
        restart_invocation_id="restart:test",
        candidate_sha="ab" * 20,
        evidence_paths=[evidence],
    )
    args, fields = failures[0]
    assert args == ("weight.settlement_history_incomplete",)
    assert fields["last_successful_stage"] == "pre_shutdown_checks_complete"
    assert "completed" in fields["missing_milestones"]
    assert fields["duration_seconds"] == 17


def test_successful_but_slow_restart_emits_deadline_alert(tmp_path, monkeypatch):
    ledger = tmp_path / "restart.jsonl"
    _write_ledger(
        ledger,
        (
            {"stage": "release_ready", "status": "passed", "elapsed_seconds": 1},
            {"stage": "completed", "status": "passed", "elapsed_seconds": 20},
        ),
    )
    failures = []
    monkeypatch.setenv("LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS", "5")
    monkeypatch.setattr(
        sentry_operations, "capture_failure", lambda *args, **kwargs: failures.append((args, kwargs))
    )
    monkeypatch.setattr(sentry_operations, "record_stage", lambda **kwargs: None)
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "record_sentry_distribution",
        lambda *args, **kwargs: None,
    )
    sentry_operations.emit_restart_summary(
        component="gateway",
        status="passed",
        stage="completed",
        ledger_path=ledger,
        restart_invocation_id="restart:test",
        candidate_sha="ab" * 20,
    )
    assert failures[0][0] == ("restart.stage_deadline_exceeded",)
    assert failures[0][1]["fail_closed"] is False


def test_release_summary_preserves_stage_ledger_and_reports_first_failure(monkeypatch):
    failures = []
    stages = []
    monkeypatch.setattr(
        sentry_operations, "capture_failure", lambda *args, **kwargs: failures.append((args, kwargs))
    )
    monkeypatch.setattr(
        sentry_operations, "record_stage", lambda **kwargs: stages.append(kwargs)
    )
    monkeypatch.setattr(
        sentry_operations.sentry_bootstrap,
        "record_sentry_distribution",
        lambda *args, **kwargs: None,
    )
    sentry_operations.emit_release_summary(
        component="attested-release",
        physical_role="validator-parent",
        status="failed",
        candidate_sha="ab" * 20,
        stage_statuses=(
            ("source_checkout", "success"),
            ("host_memory_guard", "failure"),
            ("gateway_validator_build", "skipped"),
        ),
        stage_durations={"host_memory_guard": 12.5},
        duration_seconds=42,
    )
    assert [item["stage"] for item in stages] == [
        "source_checkout",
        "host_memory_guard",
        "gateway_validator_build",
    ]
    assert failures[0][0] == ("release.builder_resource_exhausted",)
    assert stages[1]["duration_seconds"] == 12.5
    assert failures[0][1]["last_successful_stage"] == "source_checkout"
    assert failures[0][1]["blocked_stages"] == ["gateway_validator_build"]
