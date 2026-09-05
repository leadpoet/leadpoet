"""The service process driver: one tick advances every active round and never dies."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("run_lab_arena_service", ROOT / "scripts" / "run_lab_arena_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeService:
    def __init__(self, *, current=None, active=None, current_error=None, advance_error=None, ensure=None, ensure_error=None, reward_error=None, activated=0):
        # ``current``: one running round (no status: not open); ``active``: the full active list.
        self._active = list(active) if active is not None else ([current] if current else [])
        self._current_error = current_error
        self._advance_error = advance_error
        self._ensure = ensure or {"status": "disabled"}
        self._ensure_error = ensure_error
        self._reward_error = reward_error
        self._activated = activated
        self.advanced = []
        self.ensured = 0

    def startup_checks(self):
        return {"database_identity": {"current_user": "test"}}

    def activate_pending_rewards(self):
        if self._reward_error is not None:
            raise self._reward_error
        return {"activated": self._activated}

    def ensure_daily_round(self):
        self.ensured += 1
        if self._ensure_error is not None:
            raise self._ensure_error
        return self._ensure

    def active_rounds(self):
        if self._current_error is not None:
            raise self._current_error
        return list(self._active)

    def advance_round(self, round_id):
        if self._advance_error is not None:
            raise self._advance_error
        self.advanced.append(round_id)
        return {"status": "ok"}


def test_scheduler_flags_split_the_driver_from_the_api(script):
    """One scheduler process runs the driver; API replicas run without it; the two flags exclude each other."""

    parser = script.build_parser()
    assert parser.parse_args([]).no_driver is False and parser.parse_args([]).driver_only is False
    assert parser.parse_args(["--no-driver"]).no_driver is True
    assert parser.parse_args(["--driver-only", "--tick-seconds", "5"]).driver_only is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-driver", "--driver-only"])


def _patch_runtime(monkeypatch, script, service):
    wiring = types.ModuleType("lab_arena.wiring")
    wiring.build_service_from_environment = lambda mode: (service, object())
    monkeypatch.setitem(sys.modules, "lab_arena.wiring", wiring)
    monkeypatch.setenv("LAB_ARENA_MODE", "live")


def test_api_only_startup_serves_without_running_a_driver_tick(script, monkeypatch):
    service = FakeService()
    ticks = []
    served = []
    _patch_runtime(monkeypatch, script, service)
    monkeypatch.setattr(script, "drive_once", lambda service: ticks.append(service) or "idle")
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda app, **kwargs: served.append((app, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    assert script.main(["--no-driver"]) == 0
    assert ticks == []
    assert len(served) == 1


def test_normal_startup_keeps_the_initial_driver_tick(script, monkeypatch):
    service = FakeService()
    ticks = []
    _patch_runtime(monkeypatch, script, service)
    monkeypatch.setattr(script, "drive_once", lambda service: ticks.append(service) or "idle")
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda app, **kwargs: None
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    assert script.main([]) == 0
    assert ticks == [service]


def test_driver_only_keeps_the_initial_tick_and_runs_the_loop(script, monkeypatch):
    service = FakeService()
    ticks = []
    _patch_runtime(monkeypatch, script, service)
    monkeypatch.setattr(script, "drive_once", lambda service: ticks.append(service) or "idle")

    class FiniteEvent:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            return self.waits >= 2

        def set(self):
            pass

    monkeypatch.setattr(script.threading, "Event", FiniteEvent)

    assert script.main(["--driver-only", "--tick-seconds", "5"]) == 0
    assert ticks == [service, service]


def test_check_only_performs_startup_checks_without_writes(script, monkeypatch):
    service = FakeService()
    ticks = []
    served = []
    _patch_runtime(monkeypatch, script, service)
    monkeypatch.setattr(script, "drive_once", lambda service: ticks.append(service) or "idle")
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda app, **kwargs: served.append((app, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    assert script.main(["--check-only"]) == 0
    assert ticks == []
    assert served == []


def test_a_tick_advances_the_current_round(script):
    service = FakeService(current={"round_id": "arena-2026-09-03"})
    assert script.drive_once(service) == "advanced arena-2026-09-03" and service.advanced == ["arena-2026-09-03"]
    assert service.ensured == 1  # no round is open for submissions, so the next daily round is offered


def test_a_tick_advances_every_active_round_oldest_first_and_opens_nothing_while_one_is_open(script):
    """Rounds overlap: the running round and the open round both advance; an open round means no new one."""

    service = FakeService(active=[{"round_id": "arena-2026-09-03", "status": "stage1"}, {"round_id": "arena-2026-09-04", "status": "open"}])
    assert script.drive_once(service) == "advanced arena-2026-09-03; advanced arena-2026-09-04"
    assert service.advanced == ["arena-2026-09-03", "arena-2026-09-04"] and service.ensured == 0
    running_only = FakeService(active=[{"round_id": "arena-2026-09-03", "status": "stage1_scoring"}], ensure={"status": "created", "round_id": "arena-2026-09-04"})
    assert script.drive_once(running_only) == "advanced arena-2026-09-03; created arena-2026-09-04"
    one_failing = FakeService(active=[{"round_id": "arena-2026-09-03", "status": "stage1"}, {"round_id": "arena-2026-09-04", "status": "open"}], advance_error=RuntimeError("secret"))
    assert script.drive_once(one_failing) == "failed advance_round arena-2026-09-03: RuntimeError; failed advance_round arena-2026-09-04: RuntimeError"


def test_a_tick_without_a_round_is_idle_unless_it_creates_the_next_daily_round(script):
    disabled = FakeService(current=None)
    assert script.drive_once(disabled) == "idle" and disabled.ensured == 1
    creating = FakeService(current=None, ensure={"status": "created", "round_id": "arena-2026-09-04"})
    assert script.drive_once(creating) == "created arena-2026-09-04"
    assert script.drive_once(FakeService(current=None, ensure_error=ConnectionError("db"))) == "failed ensure_daily_round: ConnectionError"


@pytest.mark.parametrize("kwargs, expected", [
    ({"current_error": ConnectionError("postgrest unreachable")}, "failed active_rounds: ConnectionError"),
    ({"current": {"round_id": "arena-2026-09-03"}, "advance_error": RuntimeError("secret-bearing message never printed")}, "failed advance_round arena-2026-09-03: RuntimeError"),
])
def test_every_tick_failure_is_contained_and_named_by_type_only(script, kwargs, expected):
    """A transient lookup or advance failure is the next tick's retry, never the driver's death."""

    outcome = script.drive_once(FakeService(**kwargs))
    assert outcome == expected
    assert "secret" not in outcome and "postgrest" not in outcome


def test_reward_activation_failure_does_not_undo_publication_or_stop_the_driver(script):
    failed = FakeService(current=None, reward_error=RuntimeError("kms secret"))
    assert script.drive_once(failed) == "failed activate_rewards: RuntimeError"
    retried = FakeService(current=None, activated=1)
    assert script.drive_once(retried) == "activated rewards 1"
