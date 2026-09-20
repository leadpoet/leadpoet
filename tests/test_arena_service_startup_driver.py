"""A slow scoring cycle must not prevent the Arena API from starting."""

import threading
from types import SimpleNamespace

import pytest

from lab_arena import wiring
from scripts import run_lab_arena_service as launcher


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
