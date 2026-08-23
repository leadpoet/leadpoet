"""Dedicated supervisor for the reviewed routing execution consumer.

The routing consumer is a separate process.  It is not part of the hosted or
scoring worker fleets because it owns a durable queue lease and must use the
static reviewed product bootstrap before it can claim work.  The command,
working directory, and module are fixed in this file; configuration cannot
select an import path, endpoint, credential, or executable.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
from typing import Mapping


ROUTING_CONSUMER_READY_FD_ENV = "RESEARCH_LAB_WORKER_READY_FD"
ROUTING_CONSUMER_SUPERVISOR_POLL_ENV = (
    "RESEARCH_LAB_ROUTING_CONSUMER_SUPERVISOR_POLL_SECONDS"
)
ROUTING_CONSUMER_STARTUP_TIMEOUT_ENV = (
    "RESEARCH_LAB_ROUTING_CONSUMER_STARTUP_TIMEOUT_SECONDS"
)
ROUTING_CONSUMER_MODULE = "gateway.research_lab.routing_execution_consumer"


class RoutingConsumerSupervisorError(RuntimeError):
    """The dedicated routing consumer cannot be started safely."""


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _seconds(
    environment: Mapping[str, str], name: str, *, default: float, minimum: float, maximum: float
) -> float:
    try:
        value = float(environment.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise RoutingConsumerSupervisorError(f"{name} must be a number") from exc
    if value < minimum or value > maximum:
        raise RoutingConsumerSupervisorError(
            f"{name} must be {minimum}..{maximum}"
        )
    return value


class RoutingExecutionConsumerSupervisor:
    """Start, monitor, restart, and stop one fixed routing child process."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self.environment = dict(os.environ if environment is None else environment)
        self._startup_timeout = _seconds(
            self.environment,
            ROUTING_CONSUMER_STARTUP_TIMEOUT_ENV,
            default=30.0,
            minimum=1.0,
            maximum=300.0,
        )
        self._poll_seconds = _seconds(
            self.environment,
            ROUTING_CONSUMER_SUPERVISOR_POLL_ENV,
            default=5.0,
            minimum=0.1,
            maximum=300.0,
        )
        self._package_parent = Path(__file__).resolve().parents[2]
        self.child: subprocess.Popen[bytes] | None = None
        self.restart_count = 0
        self._ready = False
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._restart_in_progress = False
        self._starting_child: subprocess.Popen[bytes] | None = None

    @property
    def command(self) -> tuple[str, ...]:
        """Return the immutable child command used by this supervisor."""

        return (
            sys.executable,
            "-m",
            ROUTING_CONSUMER_MODULE,
        )

    def _enabled(self) -> bool:
        return _truthy(
            self.environment.get("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED")
        )

    def start(self) -> None:
        if not self._enabled():
            return
        with self._lock:
            if self._restart_in_progress:
                return
            if self.child is not None and self.child.poll() is None:
                return
            self._stop_event.clear()
            self._restart_in_progress = True

        try:
            child = self._start_child()
        except BaseException:
            with self._lock:
                self._restart_in_progress = False
                self._ready = False
            raise

        terminate_child = False
        with self._lock:
            self._restart_in_progress = False
            if self._stop_event.is_set():
                self._ready = False
                self.child = None
                terminate_child = True
            else:
                self.child = child
                self._monitor_thread = threading.Thread(
                    target=self._monitor,
                    name="routing-execution-consumer-supervisor",
                    daemon=True,
                )
                self._monitor_thread.start()
        if terminate_child:
            self._terminate_and_reap(child)

    async def start_without_blocking_event_loop(self) -> dict[str, object]:
        """Start the child and wait for readiness outside the event loop."""

        startup_task = asyncio.create_task(asyncio.to_thread(self.start))
        try:
            await asyncio.shield(startup_task)
        except asyncio.CancelledError:
            # A worker thread cannot be cancelled.  Serialize shutdown behind
            # the start operation so it cannot publish a child after lifespan
            # cleanup has already inspected the supervisor.
            while not startup_task.done():
                try:
                    await asyncio.shield(startup_task)
                except asyncio.CancelledError:
                    continue
            startup_task.result()
            raise
        return self.health()

    async def stop_without_blocking_event_loop(self) -> None:
        """Stop and reap the child without freezing the gateway event loop."""

        await asyncio.to_thread(self.stop)

    @staticmethod
    def _terminate_and_reap(child: subprocess.Popen[bytes]) -> None:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                child.kill()
            try:
                child.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                # The supervisor cannot safely reuse a child it cannot reap.
                # The next start attempt will remain unavailable.
                pass

    def _start_child(self) -> subprocess.Popen[bytes]:
        read_fd, write_fd = os.pipe()
        child_environment = dict(self.environment)
        child_environment[ROUTING_CONSUMER_READY_FD_ENV] = str(write_fd)
        child_environment.setdefault("PYTHONUNBUFFERED", "1")
        child: subprocess.Popen[bytes] | None = None
        try:
            child = subprocess.Popen(
                list(self.command),
                cwd=str(self._package_parent),
                env=child_environment,
                pass_fds=(write_fd,),
            )
        except Exception as exc:
            os.close(read_fd)
            os.close(write_fd)
            raise RoutingConsumerSupervisorError(
                "routing consumer child could not be started"
            ) from exc
        try:
            with self._lock:
                self._starting_child = child
                startup_cancelled = self._stop_event.is_set()
            os.close(write_fd)
            write_fd = -1
            if startup_cancelled:
                return child
            ready, _, _ = select.select([read_fd], [], [], self._startup_timeout)
            marker = os.read(read_fd, 64) if ready else b""
        except BaseException:
            if child is not None:
                self._terminate_and_reap(child)
            raise
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            if child is not None:
                with self._lock:
                    if self._starting_child is child:
                        self._starting_child = None
        if child is None:  # pragma: no cover - Popen either returns or raises
            raise RoutingConsumerSupervisorError(
                "routing consumer child could not be started"
            )
        if startup_cancelled:
            return child
        if marker != b"ready\n" or child.poll() is not None:
            self._terminate_and_reap(child)
            raise RoutingConsumerSupervisorError(
                "routing consumer child failed to signal readiness"
            )
        self._ready = True
        return child

    def _monitor(self) -> None:
        while not self._stop_event.wait(self._poll_seconds):
            with self._lock:
                if self._stop_event.is_set():
                    return
                child = self.child
                if child is not None and child.poll() is None:
                    continue
                self._ready = False
                if child is not None:
                    # Count one restart for each observed dead child. Retry
                    # attempts after a failed replacement do not double-count
                    # that same child generation.
                    self.restart_count += 1
                # Publish the unavailable state before leaving the lock. This
                # keeps health() prompt while readiness waits for a child.
                self.child = None
                self._restart_in_progress = True

            try:
                replacement = self._start_child()
            except RoutingConsumerSupervisorError:
                # Keep the supervisor unhealthy. Retry on the next poll; no
                # queue claim can occur without the child's ready byte.
                with self._lock:
                    self._restart_in_progress = False
                    self._ready = False
                    if self._stop_event.is_set():
                        return
                continue

            terminate_replacement = False
            with self._lock:
                self._restart_in_progress = False
                if self._stop_event.is_set():
                    self.child = None
                    self._ready = False
                    terminate_replacement = True
                else:
                    self.child = replacement
            if terminate_replacement:
                self._terminate_and_reap(replacement)
                return

    def health(self) -> dict[str, object]:
        with self._lock:
            child = self.child
            running = child is not None and child.poll() is None
            ready = running and self._ready
            return {
                "schema_version": "leadpoet.routing_consumer_supervisor_health.v1",
                "status": "ready" if ready else "unavailable",
                "supervised": True,
                "registered": running,
                "ready": ready,
                "running": running,
                "restart_count": self.restart_count,
                "pid": child.pid if running and child is not None else None,
            }

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            candidates: list[subprocess.Popen[bytes]] = []
            for candidate in (self.child, self._starting_child):
                if candidate is not None and all(
                    candidate is not existing for existing in candidates
                ):
                    candidates.append(candidate)
            # Make readiness unavailable before doing any blocking reap work.
            self.child = None
            self._ready = False
            for candidate in candidates:
                if candidate.poll() is None:
                    candidate.terminate()

        for candidate in candidates:
            self._terminate_and_reap(candidate)
        monitor = self._monitor_thread
        if monitor is not None:
            monitor.join(timeout=2.0)
        with self._lock:
            self.child = None
            self._ready = False
            if self._starting_child is not None and self._starting_child.poll() is not None:
                self._starting_child = None
            if monitor is None or not monitor.is_alive():
                self._restart_in_progress = False


__all__ = [
    "ROUTING_CONSUMER_READY_FD_ENV",
    "ROUTING_CONSUMER_SUPERVISOR_POLL_ENV",
    "ROUTING_CONSUMER_STARTUP_TIMEOUT_ENV",
    "ROUTING_CONSUMER_MODULE",
    "RoutingConsumerSupervisorError",
    "RoutingExecutionConsumerSupervisor",
]
