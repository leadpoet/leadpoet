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


@pytest.mark.parametrize("raw", [
    '{"LAB_ARENA_MODE":"live","LAB_ARENA_PORT":"8792","OPENROUTER_API_KEY":"not-imported"}',
    "export LAB_ARENA_MODE=live\nLAB_ARENA_PORT=8792\nOPENROUTER_API_KEY=not-imported\n",
])
def test_scoped_environment_keeps_arena_settings_without_provider_aliases(script, tmp_path, monkeypatch, raw):
    path = tmp_path / "gateway.env"
    path.write_text(raw)
    monkeypatch.setenv("LAB_ARENA_MODE", "shadow")
    monkeypatch.delenv("LAB_ARENA_PORT", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    script.load_scoped_environment(path)
    assert script.os.environ["LAB_ARENA_MODE"] == "shadow"
    assert script.os.environ["LAB_ARENA_PORT"] == "8792"
    assert "OPENROUTER_API_KEY" not in script.os.environ


@pytest.mark.parametrize("raw", [
    "[]",
    "LAB_ARENA_MODE='unterminated",
    "LAB_ARENA_MODE=live\nLAB_ARENA_MODE=shadow",
])
def test_scoped_environment_rejects_invalid_configuration(script, tmp_path, raw):
    path = tmp_path / "gateway.env"
    path.write_text(raw)
    with pytest.raises(ValueError):
        script.load_scoped_environment(path)


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
        self._config = types.SimpleNamespace(defaults=types.SimpleNamespace(daily_cutoff_hour_utc=None))

    def startup_checks(self):
        return {"database_identity": {"current_user": "test"}}

    def review_pending_submissions(self, *, should_continue=None):
        return {"reviewed": 0}

    def activate_pending_rewards(self):
        if self._reward_error is not None:
            raise self._reward_error
        return {"activated": self._activated}

    def promote_pending_baselines(self):
        return {"promoted": 0}

    def reconcile_closed_provider_costs(self):
        return {"status": "idle"}

    def ensure_daily_round(self):
        self.ensured += 1
        if self._ensure_error is not None:
            raise self._ensure_error
        if self._ensure.get("status") == "created" and not any(
            row["round_id"] == self._ensure["round_id"] for row in self._active
        ):
            self._active.append({"round_id": self._ensure["round_id"], "status": "open"})
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
    handlers = []
    app = types.SimpleNamespace(router=types.SimpleNamespace(
        add_event_handler=lambda event, handler: handlers.append((event, handler))))
    wiring.build_service_from_environment = lambda mode: (service, app)
    monkeypatch.setitem(sys.modules, "lab_arena.wiring", wiring)
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(script.signal, "signal", lambda *_: None)
    return app, handlers


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


def test_normal_startup_serves_before_initial_driver_tick_and_joins_workers(script, monkeypatch):
    service = FakeService()
    ticks = []
    app, handlers = _patch_runtime(monkeypatch, script, service)
    workers = {}
    events = []
    class ControlledThread:
        def __init__(self, *, target, name, daemon):
            self.target, self.name = target, name
            assert daemon
            workers[name] = self

        def start(self):
            events.append("start:" + self.name)

        def join(self, timeout=None):
            assert timeout == 5
            events.append("join:" + self.name)

    monkeypatch.setattr(script.threading, "Thread", ControlledThread)
    # Stop after one initial tick. The test controls exactly when the driver
    # starts instead of racing a daemon thread against an instant fake server.
    class OneTickEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            self.set()
            return True

    monkeypatch.setattr(script.threading, "Event", OneTickEvent)
    monkeypatch.setattr(script, "drive_once", lambda service: ticks.append(service) or "idle")
    uvicorn = types.ModuleType("uvicorn")
    def serve(served_app, **kwargs):
        assert served_app is app and ticks == []
        assert [event for event, _ in handlers] == ["shutdown"]
        events.append("serve")
        workers["lab-arena-driver"].target()
        handlers[0][1]()
    uvicorn.run = serve
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    assert script.main([]) == 0
    assert ticks == [service]
    assert events == ["start:lab-arena-code-review", "start:lab-arena-driver", "serve",
                      "join:lab-arena-driver", "join:lab-arena-code-review"]


def test_driver_only_keeps_the_initial_tick_and_runs_the_loop(script, monkeypatch):
    service = FakeService()
    ticks = []
    _patch_runtime(monkeypatch, script, service)
    monkeypatch.setattr(script, "drive_once", lambda service: ticks.append(service) or "idle")

    class FiniteEvent:
        def __init__(self):
            self.waits = 0
            self.stopped = False

        def wait(self, timeout):
            self.waits += 1
            if self.waits >= 2:
                self.set()
            return self.is_set()

        def set(self):
            self.stopped = True

        def is_set(self):
            return self.stopped

    monkeypatch.setattr(script.threading, "Event", FiniteEvent)
    workers = []
    joins = []
    class FakeThread:
        def __init__(self, **kwargs):
            workers.append(kwargs)

        def start(self):
            pass

        def join(self, timeout=None):
            assert timeout is None
            joins.append(timeout)

    monkeypatch.setattr(script.threading, "Thread", FakeThread)

    assert script.main(["--driver-only", "--tick-seconds", "5"]) == 0
    assert ticks == [service, service]
    assert [worker["name"] for worker in workers] == ["lab-arena-code-review"]
    assert joins == [None]


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


def test_a_tick_advances_every_active_round_and_delegates_daily_creation_to_service(script):
    """The driver asks the service to ensure intake before advancing old work."""

    service = FakeService(active=[{"round_id": "arena-2026-09-03", "status": "stage1"}, {"round_id": "arena-2026-09-04", "status": "open"}])
    assert script.drive_once(service) == "advanced arena-2026-09-03; advanced arena-2026-09-04"
    assert service.advanced == ["arena-2026-09-03", "arena-2026-09-04"] and service.ensured == 1
    running_only = FakeService(active=[{"round_id": "arena-2026-09-03", "status": "stage1_scoring"}], ensure={"status": "created", "round_id": "arena-2026-09-04"})
    assert script.drive_once(running_only) == "created arena-2026-09-04; advanced arena-2026-09-03; advanced arena-2026-09-04"
    one_failing = FakeService(active=[{"round_id": "arena-2026-09-03", "status": "stage1"}, {"round_id": "arena-2026-09-04", "status": "open"}], advance_error=RuntimeError("secret"))
    assert script.drive_once(one_failing) == "failed advance_round arena-2026-09-03: RuntimeError; failed advance_round arena-2026-09-04: RuntimeError"


def test_a_tick_without_a_round_is_idle_unless_it_creates_the_next_daily_round(script):
    disabled = FakeService(current=None)
    assert script.drive_once(disabled) == "idle" and disabled.ensured == 1
    creating = FakeService(current=None, ensure={"status": "created", "round_id": "arena-2026-09-04"})
    assert script.drive_once(creating) == "created arena-2026-09-04; advanced arena-2026-09-04"
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
