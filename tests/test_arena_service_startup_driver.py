"""A slow scoring cycle must not prevent the Arena API from starting."""

import threading
from types import SimpleNamespace

import pytest

from lab_arena import wiring
from scripts import run_lab_arena_service as launcher


@pytest.mark.parametrize("bootstrap_fails", [False, True])
def test_sentry_is_initialized_after_config_and_cannot_block_service(
    tmp_path, monkeypatch, capsys, bootstrap_fails
):
    import os
    import leadpoet_observability

    path = tmp_path / "gateway.env"
    path.write_text("LAB_ARENA_MODE=live\nLEADPOET_SENTRY_ENABLED=1\n")
    monkeypatch.setenv("LAB_ARENA_MODE", "")
    monkeypatch.delenv("LAB_ARENA_MODE", raising=False)
    monkeypatch.setenv("LEADPOET_SENTRY_ENABLED", "")
    monkeypatch.delenv("LEADPOET_SENTRY_ENABLED", raising=False)
    calls = []

    def initialize(*, component):
        assert os.environ["LEADPOET_SENTRY_ENABLED"] == "1"
        calls.append(component)
        if bootstrap_fails:
            raise RuntimeError("fixture-private-detail")
        return False

    def build(mode):
        assert calls == ["arena-service"]
        return SimpleNamespace(startup_checks=lambda: {
            "database_identity": {"current_user": "test"},
        }), object()

    monkeypatch.setattr(leadpoet_observability, "init_sentry", initialize)
    monkeypatch.setattr(wiring, "build_service_from_environment", build)
    monkeypatch.setattr(launcher, "_install_arena_telemetry", lambda app: None)
    assert launcher.main(["--environment-file", str(path), "--check-only"]) == 0
    log = capsys.readouterr().err
    assert "fixture-private-detail" not in log
    assert ("RuntimeError" in log) == bootstrap_fails


def test_off_service_does_not_initialize_sentry(monkeypatch):
    import leadpoet_observability

    monkeypatch.setenv("LAB_ARENA_MODE", "off")
    monkeypatch.setattr(leadpoet_observability, "init_sentry",
                        lambda **kwargs: pytest.fail("disabled service initialized"))
    assert launcher.main([]) == 0


@pytest.mark.parametrize("driver_result", ["idle", "failed advance_round"])
def test_api_starts_while_initial_scoring_cycle_is_pending(monkeypatch, driver_result):
    import uvicorn

    started = threading.Event()
    release = threading.Event()
    calls = []
    app = object()
    service = SimpleNamespace(
        startup_checks=lambda: {"database_identity": {"current_user": "test"}},
        review_pending_submissions=lambda: None,
    )
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(wiring, "build_service_from_environment", lambda mode: (service, app))

    def drive(actual):
        assert actual is service
        calls.append("drive")
        started.set()
        assert release.wait(2), "API startup waited for scoring to complete"
        return driver_result

    def serve(actual, **kwargs):
        assert actual is app
        assert started.wait(2)
        calls.append("serve")
        release.set()

    monkeypatch.setattr(launcher, "drive_once", drive)
    monkeypatch.setattr(uvicorn, "run", serve)
    try:
        assert launcher.main([]) == 0
    finally:
        release.set()
    assert calls == ["drive", "serve"]
    assert not any(t.name in {"lab-arena-driver", "lab-arena-code-review"}
                   for t in threading.enumerate())


@pytest.mark.parametrize("arguments,serves", [(["--no-driver"], True), (["--check-only"], False)])
def test_api_only_and_check_only_do_not_start_scoring(monkeypatch, arguments, serves):
    import uvicorn

    calls = []
    service = SimpleNamespace(startup_checks=lambda: {"database_identity": {"current_user": "test"}})
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(wiring, "build_service_from_environment", lambda mode: (service, object()))
    monkeypatch.setattr(launcher, "drive_once", lambda service: pytest.fail("unexpected scoring"))
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append("serve"))
    assert launcher.main(arguments) == 0
    assert calls == (["serve"] if serves else [])
