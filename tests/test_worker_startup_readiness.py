"""Cold-start regressions for the gateway worker supervisor."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import threading

import pytest

from gateway.research_lab import worker_autostart
from gateway.research_lab.worker_autostart import (
    ResearchLabWorkerAutoStartPlan,
    ResearchLabWorkerFleetPlan,
    ResearchLabWorkerStartupError,
    ResearchLabWorkerSupervisor,
    start_worker_supervisor_without_blocking_event_loop,
)


def _fleet(kind: str, count: int) -> ResearchLabWorkerFleetPlan:
    return ResearchLabWorkerFleetPlan(
        kind=kind,
        worker_count=count,
        worker_prefix=kind,
        log_level="INFO",
        proxy_refs=tuple(f"proxy-{index}" for index in range(count)),
        enabled=True,
    )


def _plan(*, hosted: int = 1, scoring: int = 1) -> ResearchLabWorkerAutoStartPlan:
    return ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", hosted),
        scoring=_fleet("scoring", scoring),
    )


def _write_readiness_child(tmp_path: Path) -> Path:
    script = tmp_path / "readiness_child.py"
    script.write_text(
        r"""import fcntl
import os
from pathlib import Path
import time

ready_fd = int(os.environ["RESEARCH_LAB_WORKER_READY_FD"])
state_path = Path(os.environ["TEST_WORKER_STARTUP_STATE"])
delay = float(os.environ.get("TEST_WORKER_STARTUP_DELAY", "0"))
mode = os.environ.get("TEST_WORKER_STARTUP_MODE", "partial")


def update_active(delta):
    with state_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip().split()
        active, maximum = (int(raw[0]), int(raw[1])) if len(raw) == 2 else (0, 0)
        active += delta
        maximum = max(maximum, active)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{active} {maximum}\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


update_active(1)
time.sleep(delay)
if mode == "silent":
    time.sleep(60)
elif mode == "eof":
    update_active(-1)
    os.close(ready_fd)
elif mode == "wrong":
    update_active(-1)
    os.write(ready_fd, b"not-ready\n")
    os.close(ready_fd)
    time.sleep(60)
elif mode == "ready_exit":
    update_active(-1)
    os.write(ready_fd, b"ready\n")
    os.close(ready_fd)
else:
    update_active(-1)
    os.write(ready_fd, b"rea")
    time.sleep(0.01)
    os.write(ready_fd, b"dy\n")
    os.close(ready_fd)
    time.sleep(60)
""",
        encoding="utf-8",
    )
    return script


def _configure_real_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    *,
    mode: str,
    delay: float,
    plan: ResearchLabWorkerAutoStartPlan,
) -> tuple[ResearchLabWorkerSupervisor, Path, list[subprocess.Popen[bytes]], list[int]]:
    state_path = tmp_path / "startup-state.txt"
    state_path.write_text("0 0\n", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    monkeypatch.setenv("TEST_WORKER_STARTUP_STATE", str(state_path))
    monkeypatch.setenv("TEST_WORKER_STARTUP_MODE", mode)
    monkeypatch.setenv("TEST_WORKER_STARTUP_DELAY", str(delay))
    monkeypatch.setattr(
        worker_autostart,
        "build_research_lab_worker_environment",
        lambda: dict(os.environ),
    )
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []
    spawn_threads: list[int] = []

    def cleanup_children() -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=1)

    request.addfinalizer(cleanup_children)

    def observed_popen(*args, **kwargs):  # noqa: ANN002,ANN003
        spawn_threads.append(threading.get_ident())
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(worker_autostart.subprocess, "Popen", observed_popen)
    supervisor = ResearchLabWorkerSupervisor(plan)
    supervisor._worker_script = _write_readiness_child(tmp_path)
    supervisor._monitor_children = lambda: None  # type: ignore[method-assign]
    return supervisor, state_path, children, spawn_threads


def test_worker_startup_deadlines_are_derived_and_bounded(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_WORKER_STARTUP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv(
        "RESEARCH_LAB_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS",
        raising=False,
    )
    assert worker_autostart._startup_timeout_seconds() == 120.0
    assert worker_autostart._fleet_startup_timeout_seconds(1) == 120.0
    assert worker_autostart._fleet_startup_timeout_seconds(35) == 420.0
    assert worker_autostart._fleet_startup_timeout_seconds(100) == 480.0

    monkeypatch.setenv("RESEARCH_LAB_WORKER_STARTUP_TIMEOUT_SECONDS", "999")
    monkeypatch.setenv(
        "RESEARCH_LAB_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS",
        "999",
    )
    assert worker_autostart._startup_timeout_seconds() == 180.0
    assert worker_autostart._fleet_startup_timeout_seconds(35) == 480.0


@pytest.mark.asyncio
async def test_real_slow_children_start_sequentially_without_blocking_loop(
    monkeypatch,
    tmp_path,
    request,
):
    supervisor, state_path, children, spawn_threads = _configure_real_child(
        monkeypatch,
        tmp_path,
        request,
        mode="partial",
        delay=0.04,
        plan=_plan(hosted=2, scoring=2),
    )
    monkeypatch.setattr(worker_autostart, "_startup_timeout_seconds", lambda: 0.5)
    monkeypatch.setattr(
        worker_autostart,
        "_fleet_startup_timeout_seconds",
        lambda _count: 2.0,
    )

    async def forbidden_to_thread(*_args, **_kwargs):
        raise AssertionError("real supervisor startup used an executor thread")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)
    ticks = 0
    ticking = True

    async def ticker() -> None:
        nonlocal ticks
        while ticking:
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    caller_thread = threading.get_ident()
    try:
        health = await start_worker_supervisor_without_blocking_event_loop(supervisor)
    finally:
        ticking = False
        await ticker_task

    assert health["status"] == "ready"
    assert health["startup_attempts"] == 1
    assert health["last_startup_failure"] is None
    assert spawn_threads == [caller_thread] * 4
    assert ticks >= 10
    assert state_path.read_text(encoding="utf-8").strip() == "0 1"
    supervisor.stop()
    assert all(child.poll() is not None for child in children)


@pytest.mark.asyncio
async def test_timeout_retries_once_without_spawning_later_workers(
    monkeypatch,
    tmp_path,
    request,
):
    supervisor, _state_path, children, _spawn_threads = _configure_real_child(
        monkeypatch,
        tmp_path,
        request,
        mode="silent",
        delay=0,
        plan=_plan(hosted=2, scoring=1),
    )
    monkeypatch.setattr(worker_autostart, "_startup_timeout_seconds", lambda: 0.03)
    monkeypatch.setattr(
        worker_autostart,
        "_fleet_startup_timeout_seconds",
        lambda _count: 0.3,
    )

    with pytest.raises(ResearchLabWorkerStartupError, match="readiness timed out"):
        await start_worker_supervisor_without_blocking_event_loop(supervisor)

    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
    assert supervisor.children == {}
    assert supervisor._ready_children == set()
    assert supervisor._last_startup_failure == {
        "schema_version": "leadpoet.research_lab_worker_startup_failure.v1",
        "exception_class": "ResearchLabWorkerStartupError",
        "reason": "worker_readiness_timeout",
        "attempt": 2,
        "attempts": 2,
        "retrying": False,
        "fleet": "hosted",
        "worker_index": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "reason"),
    (
        ("wrong", "worker_invalid_readiness_marker"),
        ("eof", "worker_readiness_pipe_closed"),
        ("ready_exit", "worker_exited_before_readiness"),
    ),
)
async def test_invalid_or_terminal_readiness_never_opens_authority(
    monkeypatch,
    tmp_path,
    request,
    mode,
    reason,
):
    supervisor, _state_path, children, _spawn_threads = _configure_real_child(
        monkeypatch,
        tmp_path,
        request,
        mode=mode,
        delay=0,
        plan=_plan(),
    )
    monkeypatch.setattr(worker_autostart, "_startup_timeout_seconds", lambda: 0.5)
    monkeypatch.setattr(
        worker_autostart,
        "_fleet_startup_timeout_seconds",
        lambda _count: 1.0,
    )

    with pytest.raises(ResearchLabWorkerStartupError):
        await start_worker_supervisor_without_blocking_event_loop(supervisor)

    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
    assert supervisor._last_startup_failure is not None
    assert supervisor._last_startup_failure["reason"] == reason


@pytest.mark.asyncio
async def test_cancellation_reaps_current_child_and_stops_later_spawns(
    monkeypatch,
    tmp_path,
    request,
):
    supervisor, _state_path, children, _spawn_threads = _configure_real_child(
        monkeypatch,
        tmp_path,
        request,
        mode="silent",
        delay=0,
        plan=_plan(hosted=2, scoring=1),
    )
    monkeypatch.setattr(worker_autostart, "_startup_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(
        worker_autostart,
        "_fleet_startup_timeout_seconds",
        lambda _count: 2.0,
    )

    startup = asyncio.create_task(
        start_worker_supervisor_without_blocking_event_loop(supervisor)
    )
    for _ in range(100):
        if children:
            break
        await asyncio.sleep(0.005)
    assert len(children) == 1
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(startup, timeout=1)

    assert len(children) == 1
    assert children[0].poll() is not None
    assert supervisor.children == {}


@pytest.mark.asyncio
async def test_retry_diagnostic_is_sanitized_and_retained(monkeypatch, capsys):
    class Child:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.waited = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            self.waited = True
            return self.returncode

    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    supervisor = ResearchLabWorkerSupervisor(_plan())
    first_child = Child()
    replacement_children: list[Child] = []
    calls: list[str] = []

    async def flaky_start(fleet, *, fleet_deadline):  # noqa: ARG001
        calls.append(fleet.kind)
        if len(calls) == 1:
            supervisor.children["hosted:0"] = first_child  # type: ignore[assignment]
            supervisor._child_specs["hosted:0"] = (fleet, 0)
            raise ResearchLabWorkerStartupError(
                "secret://must-not-be-logged",
                reason="worker_readiness_timeout",
                fleet_kind="hosted",
                worker_index=0,
            )
        child = Child()
        replacement_children.append(child)
        key = f"{fleet.kind}:0"
        supervisor.children[key] = child  # type: ignore[assignment]
        supervisor._child_specs[key] = (fleet, 0)
        supervisor._ready_children.add(key)

    monkeypatch.setattr(
        supervisor,
        "_start_fleet_without_blocking_event_loop",
        flaky_start,
    )
    monkeypatch.setattr(supervisor, "_monitor_children", lambda: None)
    monkeypatch.setattr(
        worker_autostart,
        "_fleet_startup_timeout_seconds",
        lambda _count: 2.0,
    )

    health = await supervisor.start_without_blocking_event_loop()

    assert calls == ["hosted", "hosted", "scoring"]
    assert first_child.terminated is True
    assert first_child.waited is True
    assert health["startup_attempts"] == 2
    assert health["last_startup_failure"]["worker_index"] == 1
    output = capsys.readouterr().out
    assert "exception_class=ResearchLabWorkerStartupError" in output
    assert "fleet=hosted" in output
    assert "worker_index=1" in output
    assert "secret://must-not-be-logged" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "timeout", "cancel"))
async def test_async_reader_is_removed_and_fd_is_closed(
    monkeypatch,
    outcome,
):
    class Child:
        def poll(self):
            return None

    supervisor = ResearchLabWorkerSupervisor(_plan())
    read_fd, write_fd = os.pipe()
    loop = asyncio.get_running_loop()
    original_add_reader = loop.add_reader
    original_remove_reader = loop.remove_reader
    added: list[int] = []
    removed: list[int] = []

    def observed_add_reader(fd, callback, *args):  # noqa: ANN001
        added.append(fd)
        return original_add_reader(fd, callback, *args)

    def observed_remove_reader(fd):  # noqa: ANN001
        removed.append(fd)
        return original_remove_reader(fd)

    monkeypatch.setattr(loop, "add_reader", observed_add_reader)
    monkeypatch.setattr(loop, "remove_reader", observed_remove_reader)
    if outcome == "success":
        os.write(write_fd, b"ready\n")
        os.close(write_fd)
        await supervisor._wait_for_spawned_child_ready(
            Child(),  # type: ignore[arg-type]
            read_fd,
            _fleet("hosted", 1),
            0,
            timeout_seconds=0.2,
        )
    elif outcome == "timeout":
        with pytest.raises(ResearchLabWorkerStartupError):
            await supervisor._wait_for_spawned_child_ready(
                Child(),  # type: ignore[arg-type]
                read_fd,
                _fleet("hosted", 1),
                0,
                timeout_seconds=0.01,
            )
        os.close(write_fd)
    else:
        waiter = asyncio.create_task(
            supervisor._wait_for_spawned_child_ready(
                Child(),  # type: ignore[arg-type]
                read_fd,
                _fleet("hosted", 1),
                0,
                timeout_seconds=1,
            )
        )
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        os.close(write_fd)

    assert added == [read_fd]
    assert removed == [read_fd]
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.mark.asyncio
async def test_add_reader_failure_still_closes_read_fd(monkeypatch):
    class Child:
        def poll(self):
            return None

    supervisor = ResearchLabWorkerSupervisor(_plan())
    read_fd, write_fd = os.pipe()
    loop = asyncio.get_running_loop()

    def fail_add_reader(_fd, _callback):
        raise RuntimeError("synthetic add_reader failure")

    monkeypatch.setattr(loop, "add_reader", fail_add_reader)
    with pytest.raises(RuntimeError, match="add_reader failure"):
        await supervisor._wait_for_spawned_child_ready(
            Child(),  # type: ignore[arg-type]
            read_fd,
            _fleet("hosted", 1),
            0,
            timeout_seconds=0.1,
        )
    os.close(write_fd)
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.mark.asyncio
async def test_retry_shares_one_absolute_fleet_deadline(monkeypatch):
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    supervisor = ResearchLabWorkerSupervisor(_plan())
    deadlines: list[float] = []

    async def bounded_failure(_fleet, *, fleet_deadline):
        loop = asyncio.get_running_loop()
        deadlines.append(fleet_deadline)
        if fleet_deadline - loop.time() < 0.035:
            raise ResearchLabWorkerStartupError(
                "fleet deadline exhausted",
                reason="worker_fleet_startup_timeout",
            )
        await asyncio.sleep(0.04)
        raise ResearchLabWorkerStartupError(
            "first attempt failed",
            reason="worker_readiness_timeout",
            fleet_kind="hosted",
            worker_index=0,
        )

    monkeypatch.setattr(
        supervisor,
        "_start_fleet_without_blocking_event_loop",
        bounded_failure,
    )
    monkeypatch.setattr(
        worker_autostart,
        "_fleet_startup_timeout_seconds",
        lambda _count: 0.06,
    )

    with pytest.raises(ResearchLabWorkerStartupError):
        await supervisor.start_without_blocking_event_loop()

    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert supervisor._last_startup_failure is not None
    assert supervisor._last_startup_failure["reason"] == "worker_fleet_startup_timeout"


@pytest.mark.asyncio
async def test_failed_start_cleanup_kills_and_reaps_stubborn_child(monkeypatch):
    class StubbornChild:
        def __init__(self) -> None:
            self.killed = False
            self.terminate_calls = 0
            self.wait_calls = 0

        def poll(self):
            return -9 if self.killed else None

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):  # noqa: ARG002
            self.wait_calls += 1
            return -9

    supervisor = ResearchLabWorkerSupervisor(_plan())
    child = StubbornChild()
    supervisor.children["hosted:0"] = child  # type: ignore[assignment]
    supervisor._child_specs["hosted:0"] = (_fleet("hosted", 1), 0)
    monkeypatch.setattr(
        worker_autostart,
        "WORKER_STARTUP_TERMINATE_GRACE_SECONDS",
        0,
    )
    monkeypatch.setattr(
        worker_autostart,
        "WORKER_STARTUP_KILL_GRACE_SECONDS",
        0,
    )

    await supervisor._cleanup_failed_async_start()

    assert child.terminate_calls == 1
    assert child.killed is True
    assert child.wait_calls == 1
    assert supervisor.children == {}
