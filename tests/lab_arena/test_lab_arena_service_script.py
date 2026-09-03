"""The service process driver: one tick advances the current round and never dies."""

from __future__ import annotations

import importlib.util
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
    def __init__(self, *, current=None, current_error=None, advance_error=None, ensure=None, ensure_error=None, release=None, release_error=None):
        self._current = current
        self._current_error = current_error
        self._advance_error = advance_error
        self._ensure = ensure or {"status": "disabled"}
        self._ensure_error = ensure_error
        self._release = release or {"status": "disabled"}
        self._release_error = release_error
        self.advanced = []
        self.ensured = 0
        self.released = 0

    def release_pending(self):
        self.released += 1
        if self._release_error is not None:
            raise self._release_error
        return self._release

    def ensure_daily_round(self):
        self.ensured += 1
        if self._ensure_error is not None:
            raise self._ensure_error
        return self._ensure

    def current_round(self):
        if self._current_error is not None:
            raise self._current_error
        return self._current

    def advance_round(self, round_id):
        if self._advance_error is not None:
            raise self._advance_error
        self.advanced.append(round_id)
        return {"status": "ok"}


def test_a_tick_advances_the_current_round(script):
    service = FakeService(current={"round_id": "arena-2026-09-03"})
    assert script.drive_once(service) == "advanced" and service.advanced == ["arena-2026-09-03"]
    assert service.released == 1  # the published round's release is checked on every tick


def test_a_tick_releases_the_published_king_model_after_the_round_stops_being_current(script):
    """The published round is not current, so its release is a tick step of its own."""

    idle = FakeService(current=None, release={"status": "ok", "round_id": "arena-2026-09-02"})
    assert script.drive_once(idle) == "idle; released arena-2026-09-02" and idle.released == 1
    already = FakeService(current={"round_id": "arena-2026-09-03"}, release={"status": "released", "round_id": "arena-2026-09-02"})
    assert script.drive_once(already) == "advanced"
    failing = FakeService(current={"round_id": "arena-2026-09-03"}, release_error=RuntimeError("github token in message"))
    outcome = script.drive_once(failing)
    assert outcome == "advanced; failed release_pending: RuntimeError" and failing.advanced == ["arena-2026-09-03"]
    assert "token" not in outcome


def test_a_tick_without_a_round_is_idle_unless_it_creates_the_next_daily_round(script):
    disabled = FakeService(current=None)
    assert script.drive_once(disabled) == "idle" and disabled.ensured == 1
    creating = FakeService(current=None, ensure={"status": "created", "round_id": "arena-2026-09-04"})
    assert script.drive_once(creating) == "created arena-2026-09-04"
    assert script.drive_once(FakeService(current=None, ensure_error=ConnectionError("db"))) == "failed ensure_daily_round: ConnectionError"


@pytest.mark.parametrize("kwargs, expected", [
    ({"current_error": ConnectionError("postgrest unreachable")}, "failed current_round: ConnectionError"),
    ({"current": {"round_id": "arena-2026-09-03"}, "advance_error": RuntimeError("secret-bearing message never printed")}, "failed advance_round: RuntimeError"),
])
def test_every_tick_failure_is_contained_and_named_by_type_only(script, kwargs, expected):
    """A transient lookup or advance failure is the next tick's retry, never the driver's death."""

    outcome = script.drive_once(FakeService(**kwargs))
    assert outcome == expected
    assert "secret" not in outcome and "postgrest" not in outcome
