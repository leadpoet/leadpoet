from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lab_arena import telemetry
from scripts import run_lab_arena_service as entrypoint
import leadpoet_observability
import lab_arena.wiring


def _capture_lifecycle(monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple], list[dict]]:
    spans: list[tuple] = []
    stages: list[dict] = []
    monkeypatch.setattr(
        telemetry,
        "record",
        lambda *args, **kwargs: spans.append((*args, kwargs)),
    )
    monkeypatch.setattr(
        leadpoet_observability,
        "record_stage",
        lambda **kwargs: stages.append(kwargs),
    )
    monkeypatch.setattr(entrypoint, "_install_arena_telemetry", lambda app: None)
    monkeypatch.setattr(leadpoet_observability, "init_sentry", lambda **kwargs: False)
    monkeypatch.setitem(
        entrypoint.SOURCE_METADATA, "validator_source_commit", "a" * 40
    )
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("GATEWAY_RESTART_INVOCATION_ID", "gateway-1792000000-1234")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "do-not-export-secret")
    return spans, stages


def test_arena_startup_failure_emits_version_without_started_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)

    def fail_startup(mode: str) -> tuple:
        raise NameError("contracts is not defined")

    monkeypatch.setattr(lab_arena.wiring, "build_service_from_environment", fail_startup)
    with pytest.raises(NameError, match="contracts"):
        entrypoint.main(["--check-only"])
    assert spans == [("service_preflight", "failed", {"source_sha": "a" * 40})]
    assert [stage["status"] for stage in stages] == ["failed"]
    assert stages[0]["candidate_sha"] == "a" * 40
    output = capsys.readouterr().out
    assert "event=preflight outcome=failed" in output
    assert "source_sha=" + "a" * 40 in output
    assert "do-not-export-secret" not in output


def test_arena_clean_exit_emits_start_and_stop_with_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)
    service = SimpleNamespace(startup_checks=lambda: {"database_identity": {"current_user": "service_role"}})
    monkeypatch.setattr(
        lab_arena.wiring,
        "build_service_from_environment",
        lambda mode: (service, object()),
    )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: None))
    assert entrypoint.main(["--no-driver"]) == 0
    assert spans == [
        ("service_startup", "ok", {"source_sha": "a" * 40}),
        ("service_shutdown", "ok", {"source_sha": "a" * 40}),
    ]
    assert [stage["status"] for stage in stages] == ["ready", "completed"]
    assert {stage["candidate_sha"] for stage in stages} == {"a" * 40}
    output = capsys.readouterr().out
    assert "event=startup outcome=ok" in output
    assert "event=shutdown outcome=ok" in output
    assert "do-not-export-secret" not in output


def test_arena_check_only_does_not_claim_a_service_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)
    service = SimpleNamespace(startup_checks=lambda: {"database_identity": {"current_user": "service_role"}})
    monkeypatch.setattr(
        lab_arena.wiring,
        "build_service_from_environment",
        lambda mode: (service, object()),
    )
    assert entrypoint.main(["--check-only"]) == 0
    assert spans == []
    assert stages == []


def test_broken_telemetry_does_not_mask_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_lifecycle(monkeypatch)
    monkeypatch.setattr(
        telemetry,
        "record",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )
    monkeypatch.setattr(
        lab_arena.wiring,
        "build_service_from_environment",
        lambda mode: (_ for _ in ()).throw(NameError("contracts is not defined")),
    )
    with pytest.raises(NameError, match="contracts"):
        entrypoint.main(["--check-only"])
