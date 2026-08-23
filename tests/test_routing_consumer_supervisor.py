from __future__ import annotations

import asyncio
import os
import threading
import time

from gateway.research_lab.routing_consumer_supervisor import (
    ROUTING_CONSUMER_MODULE,
    RoutingConsumerSupervisorError,
    RoutingExecutionConsumerSupervisor,
)


def test_consumer_supervisor_uses_fixed_child_command() -> None:
    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "false",
        }
    )
    assert supervisor.command[1:] == ("-m", ROUTING_CONSUMER_MODULE)


def test_disabled_supervisor_does_not_spawn_or_report_ready() -> None:
    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "false",
        }
    )
    supervisor.start()
    assert supervisor.health()["status"] == "unavailable"
    supervisor.stop()


def test_async_start_keeps_disabled_path_nonblocking() -> None:
    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "false",
        }
    )
    health = asyncio.run(supervisor.start_without_blocking_event_loop())
    assert health["ready"] is False


def test_supervisor_readiness_stop_and_restart(monkeypatch) -> None:
    class _Child:
        _next_pid = 1000

        def __init__(self):
            self.pid = self._next_pid
            type(self)._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    children = []

    def _popen(_command, *, env, pass_fds, **_kwargs):
        child = _Child()
        children.append(child)
        os.write(pass_fds[0], b"ready\n")
        return child

    monkeypatch.setattr(
        "gateway.research_lab.routing_consumer_supervisor.subprocess.Popen",
        _popen,
    )
    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "true",
            "RESEARCH_LAB_ROUTING_CONSUMER_STARTUP_TIMEOUT_SECONDS": "1",
            "RESEARCH_LAB_ROUTING_CONSUMER_SUPERVISOR_POLL_SECONDS": "0.1",
        }
    )
    supervisor.start()
    assert supervisor.health()["ready"] is True
    children[0].returncode = 1
    supervisor._monitor_thread.join(timeout=2)  # type: ignore[union-attr]
    assert supervisor.health()["restart_count"] >= 1
    assert supervisor.health()["ready"] is True
    supervisor.stop()
    assert supervisor.health()["ready"] is False


def test_health_stays_prompt_while_replacement_waits_for_readiness(monkeypatch) -> None:
    class _Child:
        _next_pid = 3000

        def __init__(self):
            self.pid = type(self)._next_pid
            type(self)._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    replacement_started = threading.Event()
    release_replacement = threading.Event()
    calls = 0
    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "true",
            "RESEARCH_LAB_ROUTING_CONSUMER_STARTUP_TIMEOUT_SECONDS": "300",
            "RESEARCH_LAB_ROUTING_CONSUMER_SUPERVISOR_POLL_SECONDS": "0.1",
        }
    )

    def _start_child():
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement_started.set()
            assert release_replacement.wait(timeout=2)
        child = _Child()
        supervisor._ready = True
        return child

    monkeypatch.setattr(supervisor, "_start_child", _start_child)
    supervisor.start()
    assert supervisor.child is not None
    supervisor.child.returncode = 1
    assert replacement_started.wait(timeout=1)

    started = time.monotonic()
    health = supervisor.health()
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert health["status"] == "unavailable"
    assert health["running"] is False

    release_replacement.set()
    supervisor._monitor_thread.join(timeout=1)  # type: ignore[union-attr]
    assert supervisor.health()["ready"] is True
    supervisor.stop()


def test_stop_reaps_child_published_during_blocked_replacement(monkeypatch) -> None:
    class _Child:
        _next_pid = 4000

        def __init__(self):
            self.pid = type(self)._next_pid
            type(self)._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    replacement_started = threading.Event()
    starting_child: _Child | None = None
    calls = 0
    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "true",
            "RESEARCH_LAB_ROUTING_CONSUMER_STARTUP_TIMEOUT_SECONDS": "300",
            "RESEARCH_LAB_ROUTING_CONSUMER_SUPERVISOR_POLL_SECONDS": "0.1",
        }
    )

    def _start_child():
        nonlocal calls, starting_child
        calls += 1
        child = _Child()
        if calls == 2:
            starting_child = child
            with supervisor._lock:
                supervisor._starting_child = child
            replacement_started.set()
            while child.poll() is None:
                time.sleep(0.01)
            with supervisor._lock:
                if supervisor._starting_child is child:
                    supervisor._starting_child = None
        supervisor._ready = True
        return child

    monkeypatch.setattr(supervisor, "_start_child", _start_child)
    supervisor.start()
    assert supervisor.child is not None
    supervisor.child.returncode = 1
    assert replacement_started.wait(timeout=1)

    supervisor.stop()

    assert starting_child is not None
    assert starting_child.poll() is not None
    assert supervisor._starting_child is None
    assert supervisor._monitor_thread is not None
    assert not supervisor._monitor_thread.is_alive()
    assert supervisor.health()["running"] is False


def test_supervisor_retries_after_a_failed_restart(monkeypatch) -> None:
    class _Child:
        _next_pid = 2000

        def __init__(self):
            self.pid = type(self)._next_pid
            type(self)._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    supervisor = RoutingExecutionConsumerSupervisor(
        environment={
            "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "true",
            "RESEARCH_LAB_ROUTING_CONSUMER_STARTUP_TIMEOUT_SECONDS": "1",
            "RESEARCH_LAB_ROUTING_CONSUMER_SUPERVISOR_POLL_SECONDS": "0.1",
        }
    )
    calls = 0
    replacement_ready = threading.Event()

    def _start_child():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RoutingConsumerSupervisorError("simulated restart failure")
        child = _Child()
        supervisor._ready = True
        if calls == 3:
            replacement_ready.set()
        return child

    monkeypatch.setattr(supervisor, "_start_child", _start_child)
    supervisor.start()
    assert supervisor.child is not None
    supervisor.child.returncode = 1
    assert replacement_ready.wait(timeout=1)
    assert supervisor.health()["ready"] is True
    assert supervisor.health()["restart_count"] == 1
    supervisor.stop()
