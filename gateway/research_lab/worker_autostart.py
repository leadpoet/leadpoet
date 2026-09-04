"""Gateway startup supervisor for the daily public-baseline worker fleet."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from typing import Mapping

from gateway.research_lab.config import (
    LEGACY_SCORING_PROXY_PREFIXES,
    SCORING_PROXY_PREFIXES,
    V2_SCORING_PROXY_PREFIXES,
    resolve_worker_process_count,
)
from Leadpoet.utils.subnet_epoch import (
    SubnetEpochError,
    load_subnet_epoch_cutover,
)


TRUTHY = {"1", "true", "yes", "on"}
WORKER_READY_FD_ENV = "RESEARCH_LAB_WORKER_READY_FD"
DEFERRED_WORKER_FLEETS_ENV = "GATEWAY_V2_DEFER_WORKER_FLEETS"
WORKER_FLEET_ROLES = frozenset({"gateway_scoring"})
WORKER_FLEET_ROLE_BY_KIND = {
    "scoring": "gateway_scoring",
}
DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS = 120.0
MIN_WORKER_STARTUP_TIMEOUT_SECONDS = 5.0
MAX_WORKER_STARTUP_TIMEOUT_SECONDS = 180.0
WORKER_FLEET_STARTUP_SECONDS_PER_PROCESS = 12.0
MIN_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS = 120.0
MAX_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS = 480.0
WORKER_STARTUP_ATTEMPTS = 2
WORKER_READY_STABILITY_SECONDS = 0.01
WORKER_STARTUP_TERMINATE_GRACE_SECONDS = 5.0
WORKER_STARTUP_KILL_GRACE_SECONDS = 2.0


class ResearchLabWorkerStartupError(RuntimeError):
    """An authoritative worker failed before entering its poll loop."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "worker_startup_failed",
        fleet_kind: str | None = None,
        worker_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.fleet_kind = fleet_kind
        self.worker_index = worker_index


class DeferredWorkerFleetConfigurationError(ValueError):
    """The explicit one-restart worker deferral setting is invalid."""


async def start_worker_supervisor_without_blocking_event_loop(
    supervisor: "ResearchLabWorkerSupervisor",
) -> dict[str, object]:
    """Start and inspect worker fleets without freezing gateway HTTP I/O."""

    async_start = getattr(
        supervisor,
        "start_without_blocking_event_loop",
        None,
    )
    if callable(async_start):
        return dict(await async_start())

    def _start_and_read_health() -> dict[str, object]:
        supervisor.start()
        return dict(supervisor.health())

    startup_task = asyncio.create_task(asyncio.to_thread(_start_and_read_health))
    try:
        return await asyncio.shield(startup_task)
    except asyncio.CancelledError:
        # asyncio cannot stop a running worker thread.  Serialize cancellation
        # behind startup so lifespan cleanup never calls stop() concurrently
        # with the still-mutating supervisor start path.
        while not startup_task.done():
            try:
                await asyncio.shield(startup_task)
            except asyncio.CancelledError:
                continue
        startup_task.result()
        raise


def deferred_worker_fleet_roles(
    env: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Return explicitly deferred V2 worker roles.

    Deferral is an operator-selected recovery state, never an inferred
    fallback. It suppresses host worker processes only; measured enclave
    topology and proxy transport validation remain unchanged.
    """

    source = os.environ if env is None else env
    raw = str(source.get(DEFERRED_WORKER_FLEETS_ENV) or "").strip().lower()
    if not raw:
        return frozenset()
    if raw == "all":
        return WORKER_FLEET_ROLES
    roles = frozenset(item.strip() for item in raw.split(",") if item.strip())
    unknown = roles - WORKER_FLEET_ROLES
    if not roles or unknown:
        detail = ",".join(sorted(unknown or roles)) or "empty"
        raise DeferredWorkerFleetConfigurationError(
            "invalid deferred V2 worker fleet role(s): %s" % detail
        )
    return roles


def canonical_deferred_worker_fleet_roles(
    roles: frozenset[str],
) -> str:
    return ",".join(sorted(roles))


def _truthy_env(env: Mapping[str, str], name: str, default: str = "false") -> bool:
    return str(env.get(name, default)).strip().lower() in TRUTHY


def _resolve_worker_count(explicit_count: int, proxy_count: int) -> int:
    """Decouple process count from proxy count.

    ``*_PROCESS_COUNT`` (``explicit_count``) is authoritative when set; adding
    proxies must not create processes. When unset (0), default to one worker per
    configured proxy (historical behavior). Delegates to the shared resolver so
    the spawned process count and the in-process partition total apply the same
    clamp and never diverge.
    """
    return resolve_worker_process_count(explicit_count, proxy_count, minimum=0)


def _vmrss_mb(status_path: str) -> int | None:
    """Parse VmRSS from a /proc status file in MB (None off-Linux/on failure)."""
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) < 2:
                        return None
                    return int(parts[1]) // 1024
    except Exception:
        return None
    return None


def _child_rss_mb(pid: int) -> int | None:
    return _vmrss_mb(f"/proc/{pid}/status")


def _supervisor_poll_seconds() -> float:
    try:
        return float(os.getenv("RESEARCH_LAB_WORKER_SUPERVISOR_POLL_SECONDS", "5"))
    except ValueError:
        return 5.0


def _hard_rss_limit_mb() -> int:
    try:
        return int(os.getenv("RESEARCH_LAB_WORKER_HARD_RSS_LIMIT_MB", "16384"))
    except ValueError:
        return 16384


def _rss_telemetry_seconds() -> float:
    try:
        return float(os.getenv("RESEARCH_LAB_WORKER_RSS_TELEMETRY_SECONDS", "300"))
    except ValueError:
        return 300.0


def _startup_timeout_seconds() -> float:
    """Return the bounded per-child cold-start allowance.

    The first child imports the full scoring stack on a CPU-constrained host,
    so 30 seconds is not a realistic cold-start contract.  The whole-fleet
    deadline below remains authoritative; this bound only prevents one broken
    child from consuming all of it.
    """

    try:
        value = float(
            os.getenv(
                "RESEARCH_LAB_WORKER_STARTUP_TIMEOUT_SECONDS",
                str(DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS),
            )
        )
    except ValueError:
        value = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS
    if not math.isfinite(value):
        value = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS
    return min(
        MAX_WORKER_STARTUP_TIMEOUT_SECONDS,
        max(MIN_WORKER_STARTUP_TIMEOUT_SECONDS, value),
    )


def _fleet_startup_timeout_seconds(worker_count: int) -> float:
    """Return one bounded budget shared by initial startup and its retry."""

    configured = os.getenv("RESEARCH_LAB_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS")
    if configured is None or not configured.strip():
        value = max(
            MIN_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS,
            max(1, int(worker_count))
            * WORKER_FLEET_STARTUP_SECONDS_PER_PROCESS,
        )
    else:
        try:
            value = float(configured)
        except ValueError:
            value = max(
                MIN_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS,
                max(1, int(worker_count))
                * WORKER_FLEET_STARTUP_SECONDS_PER_PROCESS,
            )
    if not math.isfinite(value):
        value = MIN_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS
    return min(
        MAX_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS,
        max(MIN_WORKER_FLEET_STARTUP_TIMEOUT_SECONDS, value),
    )


def _int_env(env: Mapping[str, str], name: str, default: int = 0) -> int:
    try:
        return int(str(env.get(name, str(default))).strip())
    except (TypeError, ValueError):
        return default


def build_research_lab_worker_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy and validate the exact environment inherited by worker children."""

    env = dict(os.environ if source is None else source)
    try:
        cutover = load_subnet_epoch_cutover(env)
    except SubnetEpochError as exc:
        raise ResearchLabWorkerStartupError(
            "Research Lab worker epoch authority is missing or invalid"
        ) from exc

    raw_netuid = (
        env.get("BITTENSOR_NETUID")
        or env.get("NETUID")
        or str(cutover.netuid)
    )
    try:
        netuid = int(raw_netuid)
    except (TypeError, ValueError) as exc:
        raise ResearchLabWorkerStartupError(
            "Research Lab worker netuid is invalid"
        ) from exc
    if netuid != cutover.netuid:
        raise ResearchLabWorkerStartupError(
            "Research Lab worker epoch authority targets a different netuid"
        )
    return env


def _configured_proxies(env: Mapping[str, str], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    proxies: list[str] = []
    seen: set[str] = set()
    for index in range(1, 501):
        for prefix in prefixes:
            value = str(env.get(f"{prefix}_{index}", "")).strip()
            if value and value not in seen:
                proxies.append(value)
                seen.add(value)
                break
    for prefix in prefixes:
        value = str(env.get(prefix, "")).strip()
        if value and value not in seen:
            proxies.append(value)
            seen.add(value)
    return tuple(proxies)


def _preferred_proxy_configuration(
    env: Mapping[str, str],
    *,
    v2_prefixes: tuple[str, ...],
    legacy_prefixes: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    v2_values = _configured_proxies(env, v2_prefixes)
    if v2_values:
        return v2_values, "v2_tls"
    legacy_values = _configured_proxies(env, legacy_prefixes)
    if legacy_values:
        return legacy_values, "legacy"
    return (), "none"


def _proxy_ref(proxy_url: str) -> str:
    return "sha256:" + hashlib.sha256(proxy_url.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ResearchLabWorkerFleetPlan:
    kind: str
    worker_count: int
    worker_prefix: str
    log_level: str
    proxy_refs: tuple[str, ...]
    enabled: bool
    reason: str = ""
    proxy_source: str = "none"
    proxy_values: tuple[str, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class ResearchLabWorkerAutoStartPlan:
    auto_start_enabled: bool
    scoring: ResearchLabWorkerFleetPlan


def build_research_lab_worker_autostart_plan(
    env: Mapping[str, str] | None = None,
) -> ResearchLabWorkerAutoStartPlan:
    env = env or os.environ
    auto_start_enabled = _truthy_env(env, "RESEARCH_LAB_AUTO_START_WORKERS", "true")
    scoring_proxies, scoring_proxy_source = _preferred_proxy_configuration(
        env,
        v2_prefixes=V2_SCORING_PROXY_PREFIXES,
        legacy_prefixes=LEGACY_SCORING_PROXY_PREFIXES,
    )
    scoring_legacy_count = _int_env(env, "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT", 0)
    public_baseline_enabled = _truthy_env(
        env,
        "RESEARCH_LAB_PUBLIC_BASELINE_REBENCHMARK_ENABLED",
        "true",
    )
    scoring_enabled = (
        auto_start_enabled
        and _truthy_env(env, "RESEARCH_LAB_AUTO_START_SCORING_WORKERS", "true")
        and public_baseline_enabled
        and (scoring_legacy_count > 0 or bool(scoring_proxies))
    )

    scoring_reason = ""
    if not auto_start_enabled:
        scoring_reason = "auto_start_disabled"
    elif not public_baseline_enabled:
        scoring_reason = "public_baseline_disabled"
    elif scoring_legacy_count <= 0 and not scoring_proxies:
        scoring_reason = "no_qualification_proxies"

    scoring_count = _resolve_worker_count(scoring_legacy_count, len(scoring_proxies))
    return ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=auto_start_enabled,
        scoring=ResearchLabWorkerFleetPlan(
            kind="scoring",
            worker_count=scoring_count if scoring_enabled else 0,
            worker_prefix=str(env.get("RESEARCH_LAB_SCORING_WORKER_PREFIX", "research-lab-scorer")),
            log_level=str(env.get("RESEARCH_LAB_SCORING_WORKER_LOG_LEVEL", "INFO")),
            proxy_refs=tuple(_proxy_ref(proxy) for proxy in scoring_proxies),
            enabled=scoring_enabled,
            reason=scoring_reason,
            proxy_source=scoring_proxy_source,
            proxy_values=scoring_proxies,
        ),
    )


class ResearchLabWorkerSupervisor:
    """Start and stop gateway-owned Research Lab worker child processes."""

    def __init__(
        self,
        plan: ResearchLabWorkerAutoStartPlan | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ):
        source = os.environ if environment is None else environment
        self.plan = plan or build_research_lab_worker_autostart_plan(source)
        try:
            self.deferred_worker_fleet_roles = deferred_worker_fleet_roles(
                source
            )
        except DeferredWorkerFleetConfigurationError as exc:
            raise ResearchLabWorkerStartupError(str(exc)) from exc
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self._child_specs: dict[str, tuple[ResearchLabWorkerFleetPlan, int]] = {}
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._package_parent = Path(__file__).resolve().parents[2]
        self._worker_script = Path(__file__).resolve().parent / "worker_process.py"
        self._ready_children: set[str] = set()
        self._startup_attempts = 0
        self._last_startup_failure: dict[str, object] | None = None

    def _full_topology_required(self) -> bool:
        return os.getenv("GATEWAY_TEE_TOPOLOGY_MODE", "full").strip() == "full"

    def _validate_authoritative_plan(self) -> None:
        if not self._full_topology_required():
            return
        if not self.plan.auto_start_enabled:
            raise ResearchLabWorkerStartupError(
                "authoritative V2 worker autostart cannot be disabled"
            )
        fleet = self.plan.scoring
        if not fleet.enabled or fleet.worker_count <= 0:
            raise ResearchLabWorkerStartupError(
                "scoring worker fleet must contain configured enabled workers; got %d (%s)"
                % (fleet.worker_count, fleet.reason or "enabled")
            )

    def start(self) -> None:
        self._validate_authoritative_plan()
        if not self.plan.auto_start_enabled:
            print("⚠️  Research Lab worker autostart disabled", flush=True)
            return
        print("=" * 80, flush=True)
        print("STARTING RESEARCH LAB PUBLIC BASELINE WORKERS", flush=True)
        print("=" * 80, flush=True)
        self._start_fleet(self.plan.scoring)
        if not self.children:
            print("   No Research Lab worker fleets started", flush=True)
        else:
            self._monitor_thread = threading.Thread(
                target=self._monitor_children,
                name="research-lab-worker-supervisor",
                daemon=True,
            )
            self._monitor_thread.start()
        print("=" * 80 + "\n", flush=True)

    async def start_without_blocking_event_loop(self) -> dict[str, object]:
        """Start workers sequentially without blocking gateway HTTP I/O.

        ``Popen`` uses a fork/exec path because the readiness descriptor is
        passed explicitly. Forking from an executor thread can stall before
        ``Popen`` returns in a multithreaded gateway, while spawning every
        worker at once creates a cold-import storm. Spawn on the event-loop
        owner thread, then use its file-descriptor reader for readiness before
        spawning the next child. Initial startup and one atomic retry share a
        whole-fleet deadline below the outer V2 readiness gate.
        """

        self._validate_authoritative_plan()
        if not self.plan.auto_start_enabled:
            print("⚠️  Research Lab worker autostart disabled", flush=True)
            return dict(self.health())
        print("=" * 80, flush=True)
        print("STARTING RESEARCH LAB PUBLIC BASELINE WORKERS", flush=True)
        print("=" * 80, flush=True)
        configured_workers = sum(
            fleet.worker_count
            for fleet in (self.plan.scoring,)
            if fleet.enabled
            and WORKER_FLEET_ROLE_BY_KIND[fleet.kind]
            not in self.deferred_worker_fleet_roles
        )
        loop = asyncio.get_running_loop()
        fleet_deadline = loop.time() + _fleet_startup_timeout_seconds(
            configured_workers
        )
        for attempt in range(1, WORKER_STARTUP_ATTEMPTS + 1):
            self._startup_attempts = attempt
            try:
                for fleet in (self.plan.scoring,):
                    await self._start_fleet_without_blocking_event_loop(
                        fleet,
                        fleet_deadline=fleet_deadline,
                    )
            except asyncio.CancelledError:
                await self._cleanup_after_startup_cancellation()
                raise
            except ResearchLabWorkerStartupError as exc:
                retrying = attempt < WORKER_STARTUP_ATTEMPTS
                self._record_startup_failure(
                    exc,
                    attempt=attempt,
                    retrying=retrying,
                )
                try:
                    await self._cleanup_failed_async_start()
                except ResearchLabWorkerStartupError as cleanup_exc:
                    self._record_startup_failure(
                        cleanup_exc,
                        attempt=attempt,
                        retrying=False,
                    )
                    raise cleanup_exc from exc
                if not retrying:
                    raise
                if loop.time() >= fleet_deadline:
                    deadline_error = ResearchLabWorkerStartupError(
                        "Research Lab worker fleet startup deadline exhausted",
                        reason="worker_fleet_startup_timeout",
                    )
                    self._record_startup_failure(
                        deadline_error,
                        attempt=attempt + 1,
                        retrying=False,
                    )
                    raise deadline_error from exc
            except BaseException:
                await self._cleanup_failed_async_start()
                raise
            else:
                break
        if not self.children:
            print("   No Research Lab worker fleets started", flush=True)
        else:
            self._monitor_thread = threading.Thread(
                target=self._monitor_children,
                name="research-lab-worker-supervisor",
                daemon=True,
            )
            self._monitor_thread.start()
        print("=" * 80 + "\n", flush=True)
        return dict(self.health())

    def _start_fleet(self, fleet: ResearchLabWorkerFleetPlan) -> None:
        if fleet.kind != "scoring":
            raise ResearchLabWorkerStartupError(
                "only scoring Research Lab workers are supported",
                reason="worker_kind_retired",
                fleet_kind=fleet.kind,
            )
        role = WORKER_FLEET_ROLE_BY_KIND[fleet.kind]
        if role in self.deferred_worker_fleet_roles:
            print(
                "   %s: explicitly deferred for this gateway restart "
                "(configured=%d, running=0)"
                % (fleet.kind, fleet.worker_count),
                flush=True,
            )
            return
        if not fleet.enabled:
            print(f"   {fleet.kind}: skipped ({fleet.reason or 'disabled'})", flush=True)
            return
        print(
            f"   {fleet.kind}: starting {fleet.worker_count} worker(s), "
            f"proxy_refs={list(fleet.proxy_refs)}",
            flush=True,
        )
        for index in range(fleet.worker_count):
            child = self._start_child(fleet, index)
            key = f"{fleet.kind}:{index}"
            self.children[key] = child
            self._child_specs[key] = (fleet, index)
            self._ready_children.add(key)

    async def _start_fleet_without_blocking_event_loop(
        self,
        fleet: ResearchLabWorkerFleetPlan,
        *,
        fleet_deadline: float,
    ) -> None:
        if fleet.kind != "scoring":
            raise ResearchLabWorkerStartupError(
                "only scoring Research Lab workers are supported",
                reason="worker_kind_retired",
                fleet_kind=fleet.kind,
            )
        role = WORKER_FLEET_ROLE_BY_KIND[fleet.kind]
        if role in self.deferred_worker_fleet_roles:
            print(
                "   %s: explicitly deferred for this gateway restart "
                "(configured=%d, running=0)"
                % (fleet.kind, fleet.worker_count),
                flush=True,
            )
            return
        if not fleet.enabled:
            print(f"   {fleet.kind}: skipped ({fleet.reason or 'disabled'})", flush=True)
            return
        print(
            f"   {fleet.kind}: starting {fleet.worker_count} worker(s), "
            f"proxy_refs={list(fleet.proxy_refs)}",
            flush=True,
        )
        loop = asyncio.get_running_loop()
        for index in range(fleet.worker_count):
            remaining_seconds = fleet_deadline - loop.time()
            if remaining_seconds <= 0:
                raise ResearchLabWorkerStartupError(
                    "Research Lab worker fleet startup deadline exhausted",
                    reason="worker_fleet_startup_timeout",
                    fleet_kind=fleet.kind,
                    worker_index=index,
                )
            child, read_fd = self._spawn_child(fleet, index)
            key = f"{fleet.kind}:{index}"
            self.children[key] = child
            self._child_specs[key] = (fleet, index)
            remaining_seconds = fleet_deadline - loop.time()
            if remaining_seconds <= 0:
                try:
                    os.close(read_fd)
                except OSError:
                    pass
                raise ResearchLabWorkerStartupError(
                    "Research Lab worker fleet startup deadline exhausted",
                    reason="worker_fleet_startup_timeout",
                    fleet_kind=fleet.kind,
                    worker_index=index,
                )
            await self._wait_for_spawned_child_ready(
                child,
                read_fd,
                fleet,
                index,
                timeout_seconds=min(
                    _startup_timeout_seconds(),
                    remaining_seconds,
                ),
            )
            self._ready_children.add(key)

    async def _wait_for_spawned_child_ready(
        self,
        child: subprocess.Popen[bytes],
        read_fd: int,
        fleet: ResearchLabWorkerFleetPlan,
        index: int,
        *,
        timeout_seconds: float,
    ) -> subprocess.Popen[bytes]:
        """Await one exact readiness marker; this method solely owns read_fd."""

        loop = asyncio.get_running_loop()
        marker_future: asyncio.Future[tuple[bytes, bool]] = loop.create_future()
        marker = bytearray()
        reader_registered = False

        def _read_marker() -> None:
            if marker_future.done():
                return
            try:
                chunk = os.read(read_fd, max(1, 64 - len(marker)))
            except BlockingIOError:
                return
            except OSError:
                marker_future.set_result((bytes(marker), True))
                return
            marker.extend(chunk)
            if not chunk or b"\n" in marker or len(marker) >= 64:
                marker_future.set_result((bytes(marker), not chunk))

        try:
            os.set_blocking(read_fd, False)
            loop.add_reader(read_fd, _read_marker)
            reader_registered = True
            try:
                ready_marker, marker_eof = await asyncio.wait_for(
                    marker_future,
                    timeout=max(0.001, float(timeout_seconds)),
                )
            except TimeoutError as exc:
                raise ResearchLabWorkerStartupError(
                    "%s worker %d readiness timed out"
                    % (fleet.kind, index + 1),
                    reason="worker_readiness_timeout",
                    fleet_kind=fleet.kind,
                    worker_index=index,
                ) from exc
        finally:
            if reader_registered:
                try:
                    loop.remove_reader(read_fd)
                except Exception:
                    pass
            try:
                os.close(read_fd)
            except OSError:
                pass
        if ready_marker == b"ready\n":
            # A child that writes the marker and exits immediately never owned
            # a usable poll loop. Give that terminal state one event-loop turn
            # plus a tiny bounded scheduling window before accepting it.
            await asyncio.sleep(WORKER_READY_STABILITY_SECONDS)
        child_code = child.poll()
        if ready_marker != b"ready\n" or marker_eof or child_code is not None:
            reason = (
                "worker_exited_before_readiness"
                if child_code is not None
                else (
                    "worker_readiness_pipe_closed"
                    if marker_eof
                    else "worker_invalid_readiness_marker"
                )
            )
            raise ResearchLabWorkerStartupError(
                "%s worker %d failed to signal readiness"
                % (fleet.kind, index + 1),
                reason=reason,
                fleet_kind=fleet.kind,
                worker_index=index,
            )
        return child

    def _record_startup_failure(
        self,
        exc: ResearchLabWorkerStartupError,
        *,
        attempt: int,
        retrying: bool,
    ) -> None:
        diagnostic: dict[str, object] = {
            "schema_version": "leadpoet.research_lab_worker_startup_failure.v1",
            "exception_class": type(exc).__name__,
            "reason": exc.reason,
            "attempt": attempt,
            "attempts": WORKER_STARTUP_ATTEMPTS,
            "retrying": retrying,
        }
        if exc.fleet_kind in WORKER_FLEET_ROLE_BY_KIND:
            diagnostic["fleet"] = exc.fleet_kind
        if exc.worker_index is not None and exc.worker_index >= 0:
            diagnostic["worker_index"] = exc.worker_index + 1
        self._last_startup_failure = diagnostic
        print(
            "research_lab_worker_startup_failed "
            + " ".join(
                "%s=%s" % (key, str(value).lower() if isinstance(value, bool) else value)
                for key, value in diagnostic.items()
                if key != "schema_version"
            ),
            flush=True,
        )

    async def _cleanup_failed_async_start(self) -> None:
        """Atomically terminate and reap every child before a retry or exit."""

        unique_children = list({id(child): child for child in self.children.values()}.values())
        for child in unique_children:
            if child.poll() is None:
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
        pending = await self._wait_for_children_exit(
            unique_children,
            timeout_seconds=WORKER_STARTUP_TERMINATE_GRACE_SECONDS,
        )
        for child in pending:
            try:
                child.kill()
            except ProcessLookupError:
                pass
        pending = await self._wait_for_children_exit(
            pending,
            timeout_seconds=WORKER_STARTUP_KILL_GRACE_SECONDS,
        )
        self.children.clear()
        self._child_specs.clear()
        self._ready_children.clear()
        if pending:
            raise ResearchLabWorkerStartupError(
                "Research Lab worker startup cleanup did not reap every child",
                reason="worker_startup_cleanup_timeout",
            )

    async def _cleanup_after_startup_cancellation(self) -> None:
        """Finish child cleanup even when the owning startup task is cancelled."""

        cleanup_task = asyncio.create_task(self._cleanup_failed_async_start())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        if cleanup_task.cancelled():
            return
        try:
            cleanup_task.result()
        except ResearchLabWorkerStartupError as exc:
            self._record_startup_failure(
                exc,
                attempt=max(1, self._startup_attempts),
                retrying=False,
            )
        except BaseException as exc:
            print(
                "research_lab_worker_startup_cleanup_failed "
                "exception_class=%s" % type(exc).__name__,
                flush=True,
            )

    @staticmethod
    async def _wait_for_children_exit(
        children: list[subprocess.Popen[bytes]],
        *,
        timeout_seconds: float,
    ) -> list[subprocess.Popen[bytes]]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_seconds)
        pending = list(children)
        while pending:
            alive: list[subprocess.Popen[bytes]] = []
            for child in pending:
                if child.poll() is None:
                    alive.append(child)
                    continue
                wait = getattr(child, "wait", None)
                if callable(wait):
                    try:
                        wait(timeout=0)
                    except (subprocess.TimeoutExpired, ProcessLookupError):
                        alive.append(child)
            pending = alive
            if not pending or loop.time() >= deadline:
                break
            await asyncio.sleep(0.05)
        return pending

    def _start_child(self, fleet: ResearchLabWorkerFleetPlan, index: int) -> subprocess.Popen[bytes]:
        child, read_fd = self._spawn_child(fleet, index)
        return self._wait_for_child_ready(child, read_fd, fleet, index)

    def _spawn_child(
        self,
        fleet: ResearchLabWorkerFleetPlan,
        index: int,
    ) -> tuple[subprocess.Popen[bytes], int]:
        if fleet.kind != "scoring":
            raise ResearchLabWorkerStartupError(
                "only scoring Research Lab workers are supported",
                reason="worker_kind_retired",
                fleet_kind=fleet.kind,
                worker_index=index,
            )
        env = build_research_lab_worker_environment()
        env.setdefault("PYTHONUNBUFFERED", "1")
        read_fd, write_fd = os.pipe()
        env[WORKER_READY_FD_ENV] = str(write_fd)
        # Round-robin proxy assignment: identical to positional when the fleet
        # has at least as many proxies as workers (index < len), and reuses
        # proxies deterministically when workers are decoupled to exceed the
        # proxy count.
        proxy_value = (
            fleet.proxy_values[index % len(fleet.proxy_values)]
            if fleet.proxy_values
            else ""
        )
        env.setdefault("RESEARCH_LAB_SCORING_WORKER_ENABLED", "true")
        if proxy_value:
            env["RESEARCH_LAB_SCORING_WORKER_PROXY"] = proxy_value
        command = [
            sys.executable,
            str(self._worker_script),
            "--kind",
            fleet.kind,
            "--worker-index",
            str(index),
            "--total-workers",
            str(fleet.worker_count),
            "--worker-prefix",
            fleet.worker_prefix,
            "--log-level",
            fleet.log_level,
        ]
        try:
            child = subprocess.Popen(
                command,
                cwd=str(self._package_parent),
                env=env,
                pass_fds=(write_fd,),
            )
        except Exception as exc:
            os.close(read_fd)
            raise ResearchLabWorkerStartupError(
                "%s worker %d could not be spawned" % (fleet.kind, index + 1),
                reason="worker_spawn_failed",
                fleet_kind=fleet.kind,
                worker_index=index,
            ) from exc
        finally:
            os.close(write_fd)
        return child, read_fd

    @staticmethod
    def _wait_for_child_ready(
        child: subprocess.Popen[bytes],
        read_fd: int,
        fleet: ResearchLabWorkerFleetPlan,
        index: int,
    ) -> subprocess.Popen[bytes]:
        try:
            ready, _, _ = select.select(
                [read_fd], [], [], _startup_timeout_seconds()
            )
            marker = os.read(read_fd, 64) if ready else b""
        finally:
            os.close(read_fd)
        if marker != b"ready\n" or child.poll() is not None:
            if child.poll() is None:
                child.terminate()
            raise ResearchLabWorkerStartupError(
                "%s worker %d failed to signal readiness"
                % (fleet.kind, index + 1)
            )
        return child

    def health(self) -> dict[str, object]:
        """Return strict live-worker readiness without changing worker state."""
        self._validate_authoritative_plan()
        dead = sorted(key for key, child in self.children.items() if child.poll() is not None)
        running = {key for key, child in self.children.items() if child.poll() is None}
        missing_ready = sorted(running - self._ready_children)
        scoring_running = sum(key.startswith("scoring:") for key in running)
        expected_scoring = (
            0
            if "gateway_scoring" in self.deferred_worker_fleet_roles
            else self.plan.scoring.worker_count
        )
        if dead or missing_ready:
            raise ResearchLabWorkerStartupError(
                "authoritative V2 workers are not healthy: dead=%s missing_ready=%s"
                % (dead, missing_ready)
            )
        if self._full_topology_required() and scoring_running != expected_scoring:
            raise ResearchLabWorkerStartupError(
                "authoritative V2 scoring worker count differs: scoring=%d/%d"
                % (scoring_running, expected_scoring)
            )
        return {
            "schema_version": "leadpoet.research_lab_worker_health.v2",
            "status": "ready",
            "topology_mode": (
                "full" if self._full_topology_required() else "component"
            ),
            "deferred_worker_fleet_roles": sorted(
                self.deferred_worker_fleet_roles
            ),
            "scoring_configured": self.plan.scoring.worker_count,
            "scoring_expected_running": expected_scoring,
            "scoring_running": scoring_running,
            "startup_attempts": getattr(self, "_startup_attempts", 0),
            "last_startup_failure": (
                dict(self._last_startup_failure)
                if getattr(self, "_last_startup_failure", None) is not None
                else None
            ),
        }

    def _monitor_children(self) -> None:
        hard_rss_limit_mb = _hard_rss_limit_mb()
        telemetry_seconds = _rss_telemetry_seconds()
        poll_seconds = _supervisor_poll_seconds()
        last_telemetry = 0.0
        while not self._stop_event.wait(poll_seconds):
            emit_telemetry = (
                telemetry_seconds > 0
                and time.monotonic() - last_telemetry >= telemetry_seconds
            )
            telemetry: list[str] = []
            for key, child in list(self.children.items()):
                code = child.poll()
                if code is None:
                    rss_mb = _child_rss_mb(child.pid)
                    if emit_telemetry and rss_mb is not None:
                        telemetry.append(f"{key}={rss_mb}MB")
                    # Hard backstop: a worker this large is already threatening
                    # host-wide memory pressure (API 500s, refused claims);
                    # losing its in-flight pass to a checkpoint-resume is the
                    # cheaper failure. Normal reclamation is the worker's own
                    # between-pass recycle, which exits long before this.
                    if (
                        hard_rss_limit_mb > 0
                        and rss_mb is not None
                        and rss_mb >= hard_rss_limit_mb
                    ):
                        fleet, index = self._child_specs[key]
                        print(
                            f"   ⚠️  research_lab_worker_hard_rss_limit "
                            f"{fleet.kind} worker {index + 1} rss={rss_mb}MB "
                            f">= {hard_rss_limit_mb}MB; terminating for recycle",
                            flush=True,
                        )
                        child.terminate()
                    continue
                fleet, index = self._child_specs[key]
                if self._stop_event.is_set():
                    return
                print(
                    f"   ⚠️  Research Lab {fleet.kind} worker {index + 1} exited "
                    f"with code {code}; restarting",
                    flush=True,
                )
                self._ready_children.discard(key)
                self.children[key] = self._start_child(fleet, index)
                self._ready_children.add(key)
            if emit_telemetry:
                last_telemetry = time.monotonic()
                if telemetry:
                    print(
                        "   📊 research_lab_worker_rss " + " ".join(telemetry),
                        flush=True,
                    )

    def stop(self) -> None:
        if not self.children:
            return
        self._stop_event.set()
        print("   🛑 Stopping Research Lab worker fleets...", flush=True)
        for child in self.children.values():
            if child.poll() is None:
                child.terminate()
        deadline = time.time() + 15
        for key, child in list(self.children.items()):
            while child.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if child.poll() is None:
                child.kill()
            self.children.pop(key, None)
            self._child_specs.pop(key, None)
            self._ready_children.discard(key)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)
        print("   ✅ Research Lab worker fleets stopped", flush=True)
