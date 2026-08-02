"""Fail-open command-line bridge contracts for shell and Actions workflows."""

from __future__ import annotations

import json

from leadpoet_observability import sentry_cli


def test_restart_summary_is_correlated_and_flushes(monkeypatch, tmp_path):
    ledger = tmp_path / "restart.jsonl"
    ledger.write_text(
        json.dumps(
            {"stage": "completed", "status": "passed", "elapsed_seconds": 1}
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        sentry_cli, "init_sentry", lambda **kwargs: calls.append(("init", kwargs))
    )
    monkeypatch.setattr(
        sentry_cli,
        "emit_restart_summary",
        lambda **kwargs: calls.append(("summary", kwargs)),
    )
    monkeypatch.setattr(
        sentry_cli, "flush_sentry", lambda **kwargs: calls.append(("flush", kwargs))
    )
    assert sentry_cli.main(
        [
            "restart-summary",
            "--component",
            "gateway",
            "--status",
            "passed",
            "--stage",
            "completed",
            "--ledger",
            str(ledger),
            "--restart-invocation-id",
            "restart:gateway:test",
            "--candidate-sha",
            "ab" * 20,
            "--release-attempts",
            "7",
        ]
    ) == 0
    summary = next(value for kind, value in calls if kind == "summary")
    assert summary["restart_invocation_id"] == "restart:gateway:test"
    assert summary["candidate_sha"] == "ab" * 20
    assert summary["release_attempts"] == 7
    assert calls[-1] == ("flush", {"timeout": 1.0})


def test_invalid_release_sha_never_emits(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        sentry_cli,
        "emit_release_summary",
        lambda **kwargs: emitted.append(kwargs),
    )
    assert sentry_cli.main(
        [
            "release-summary",
            "--component",
            "attested-release",
            "--physical-role",
            "gateway-parent",
            "--status",
            "failed",
            "--candidate-sha",
            "not-a-sha",
        ]
    ) == 0
    assert emitted == []


def test_cli_swallows_internal_telemetry_failure(monkeypatch, tmp_path):
    ledger = tmp_path / "restart.jsonl"
    ledger.write_text("", encoding="utf-8")

    def _boom(**kwargs):
        raise RuntimeError("collector unavailable")

    monkeypatch.setattr(sentry_cli, "emit_restart_summary", _boom)
    assert sentry_cli.main(
        [
            "restart-summary",
            "--component",
            "validator",
            "--status",
            "failed",
            "--stage",
            "release_acquisition",
            "--ledger",
            str(ledger),
            "--restart-invocation-id",
            "restart:validator:test",
        ]
    ) == 0


def test_release_stage_status_parser_is_bounded():
    values = [f"stage_{index}=success" for index in range(100)]
    parsed = sentry_cli._stage_statuses(values)
    assert len(parsed) == 40
    assert parsed[0] == ("stage_0", "success")


def test_actions_stage_timings_are_allowlisted_and_bounded():
    document = {
        "jobs": [
            {
                "name": "Independent gateway-parent builds",
                "steps": [
                    {
                        "name": "Build gateway and validator evidence",
                        "started_at": "2026-08-02T00:00:00Z",
                        "completed_at": "2026-08-02T00:02:30Z",
                    },
                    {
                        "name": "Untrusted future step",
                        "started_at": "2026-08-02T00:00:00Z",
                        "completed_at": "2026-08-02T00:00:01Z",
                    },
                ],
            }
        ]
    }
    assert sentry_cli._stage_durations_from_actions_document(
        document, "Independent gateway-parent builds"
    ) == {"gateway_validator_build": 150.0}


def test_disabled_release_summary_does_not_read_github(monkeypatch):
    calls = []
    monkeypatch.setattr(sentry_cli, "init_sentry", lambda **kwargs: False)
    monkeypatch.setattr(
        sentry_cli,
        "_github_stage_durations",
        lambda job_name: calls.append(job_name) or {},
    )
    monkeypatch.setattr(sentry_cli, "emit_release_summary", lambda **kwargs: None)
    monkeypatch.setattr(sentry_cli, "flush_sentry", lambda **kwargs: None)
    assert sentry_cli.main(
        [
            "release-summary",
            "--component",
            "release-gateway-parent",
            "--physical-role",
            "gateway-parent-builder",
            "--status",
            "passed",
            "--candidate-sha",
            "ab" * 20,
            "--github-job-name",
            "Independent gateway-parent builds",
        ]
    ) == 0
    assert calls == []


def test_unavailable_actions_timing_api_is_an_empty_best_effort_result(monkeypatch):
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.invalid")
    monkeypatch.setenv("GITHUB_REPOSITORY", "leadpoet/leadpoet")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("LEADPOET_GITHUB_JOB_TOKEN", "seeded-job-token")

    def _unavailable(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(sentry_cli.urllib.request, "urlopen", _unavailable)
    assert sentry_cli._github_stage_durations("Independent gateway-parent builds") == {}
