"""Gateway startup supervisor for Research Lab worker fleets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from typing import Mapping


TRUTHY = {"1", "true", "yes", "on"}
EXPECTED_HOSTED_WORKERS = 10
EXPECTED_SCORING_WORKERS = 25
WORKER_READY_FD_ENV = "RESEARCH_LAB_WORKER_READY_FD"


class ResearchLabWorkerStartupError(RuntimeError):
    """An authoritative worker failed before entering its poll loop."""

HOSTED_PROXY_PREFIXES = (
    "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
    "RESEARCH_LAB_WORKER_PROXY",
    "RESEARCH_LAB_WORKER_HTTPS_PROXY",
)
SCORING_PROXY_PREFIXES = (
    "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY",
    "QUALIFICATION_WEBSHARE_PROXY",
    "RESEARCH_LAB_SCORING_WORKER_PROXY",
)


def _truthy_env(env: Mapping[str, str], name: str, default: str = "false") -> bool:
    return str(env.get(name, default)).strip().lower() in TRUTHY


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
    try:
        return max(
            1.0,
            float(os.getenv("RESEARCH_LAB_WORKER_STARTUP_TIMEOUT_SECONDS", "30")),
        )
    except ValueError:
        return 30.0


def _int_env(env: Mapping[str, str], name: str, default: int = 0) -> int:
    try:
        return int(str(env.get(name, str(default))).strip())
    except (TypeError, ValueError):
        return default


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
    proxy_values: tuple[str, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class ResearchLabWorkerAutoStartPlan:
    auto_start_enabled: bool
    hosted: ResearchLabWorkerFleetPlan
    scoring: ResearchLabWorkerFleetPlan


def build_research_lab_worker_autostart_plan(
    env: Mapping[str, str] | None = None,
) -> ResearchLabWorkerAutoStartPlan:
    env = env or os.environ
    auto_start_enabled = _truthy_env(env, "RESEARCH_LAB_AUTO_START_WORKERS", "true")
    hosted_proxies = _configured_proxies(env, HOSTED_PROXY_PREFIXES)
    scoring_proxies = _configured_proxies(env, SCORING_PROXY_PREFIXES)
    hosted_legacy_count = _int_env(env, "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT", 0)
    scoring_legacy_count = _int_env(env, "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT", 0)

    hosted_enabled = (
        auto_start_enabled
        and _truthy_env(env, "RESEARCH_LAB_AUTO_START_HOSTED_WORKERS", "true")
        and (
            _truthy_env(env, "RESEARCH_LAB_HOSTED_RUNS_ENABLED")
            or _truthy_env(env, "RESEARCH_LAB_HOSTED_WORKER_ENABLED")
        )
        and (hosted_legacy_count > 0 or bool(hosted_proxies))
    )
    scoring_enabled = (
        auto_start_enabled
        and _truthy_env(env, "RESEARCH_LAB_AUTO_START_SCORING_WORKERS", "true")
        and (
            _truthy_env(env, "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED")
            or _truthy_env(env, "RESEARCH_LAB_SCORING_WORKER_ENABLED")
        )
        and (scoring_legacy_count > 0 or bool(scoring_proxies))
    )

    hosted_reason = ""
    if not auto_start_enabled:
        hosted_reason = "auto_start_disabled"
    elif not (_truthy_env(env, "RESEARCH_LAB_HOSTED_RUNS_ENABLED") or _truthy_env(env, "RESEARCH_LAB_HOSTED_WORKER_ENABLED")):
        hosted_reason = "hosted_runs_disabled"
    elif hosted_legacy_count <= 0 and not hosted_proxies:
        hosted_reason = "no_auto_research_proxies"

    scoring_reason = ""
    if not auto_start_enabled:
        scoring_reason = "auto_start_disabled"
    elif not (_truthy_env(env, "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED") or _truthy_env(env, "RESEARCH_LAB_SCORING_WORKER_ENABLED")):
        scoring_reason = "evaluation_bundles_disabled"
    elif scoring_legacy_count <= 0 and not scoring_proxies:
        scoring_reason = "no_qualification_proxies"

    hosted_count = len(hosted_proxies) if hosted_proxies else max(0, hosted_legacy_count)
    scoring_count = len(scoring_proxies) if scoring_proxies else max(0, scoring_legacy_count)
    return ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=auto_start_enabled,
        hosted=ResearchLabWorkerFleetPlan(
            kind="hosted",
            worker_count=hosted_count if hosted_enabled else 0,
            worker_prefix=str(env.get("RESEARCH_LAB_HOSTED_WORKER_PREFIX", "research-lab-worker")),
            log_level=str(env.get("RESEARCH_LAB_HOSTED_WORKER_LOG_LEVEL", "INFO")),
            proxy_refs=tuple(_proxy_ref(proxy) for proxy in hosted_proxies),
            enabled=hosted_enabled,
            reason=hosted_reason,
            proxy_values=hosted_proxies,
        ),
        scoring=ResearchLabWorkerFleetPlan(
            kind="scoring",
            worker_count=scoring_count if scoring_enabled else 0,
            worker_prefix=str(env.get("RESEARCH_LAB_SCORING_WORKER_PREFIX", "research-lab-scorer")),
            log_level=str(env.get("RESEARCH_LAB_SCORING_WORKER_LOG_LEVEL", "INFO")),
            proxy_refs=tuple(_proxy_ref(proxy) for proxy in scoring_proxies),
            enabled=scoring_enabled,
            reason=scoring_reason,
            proxy_values=scoring_proxies,
        ),
    )


class ResearchLabWorkerSupervisor:
    """Start and stop gateway-owned Research Lab worker child processes."""

    def __init__(self, plan: ResearchLabWorkerAutoStartPlan | None = None):
        self.plan = plan or build_research_lab_worker_autostart_plan()
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self._child_specs: dict[str, tuple[ResearchLabWorkerFleetPlan, int]] = {}
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._package_parent = Path(__file__).resolve().parents[2]
        self._worker_script = Path(__file__).resolve().parent / "worker_process.py"
        self._ready_children: set[str] = set()

    def _full_topology_required(self) -> bool:
        return os.getenv("GATEWAY_TEE_TOPOLOGY_MODE", "full").strip() == "full"

    def _validate_authoritative_plan(self) -> None:
        if not self._full_topology_required():
            return
        if not self.plan.auto_start_enabled:
            raise ResearchLabWorkerStartupError(
                "authoritative V2 worker autostart cannot be disabled"
            )
        expected = {
            "hosted": (self.plan.hosted, EXPECTED_HOSTED_WORKERS),
            "scoring": (self.plan.scoring, EXPECTED_SCORING_WORKERS),
        }
        for kind, (fleet, count) in expected.items():
            if not fleet.enabled or fleet.worker_count != count:
                raise ResearchLabWorkerStartupError(
                    "%s worker fleet must contain exactly %d enabled workers; got %d (%s)"
                    % (kind, count, fleet.worker_count, fleet.reason or "enabled")
                )

    def start(self) -> None:
        self._validate_authoritative_plan()
        if not self.plan.auto_start_enabled:
            print("⚠️  Research Lab worker autostart disabled", flush=True)
            return
        print("=" * 80, flush=True)
        print("🧪 STARTING RESEARCH LAB WORKER FLEETS", flush=True)
        print("=" * 80, flush=True)
        self._start_fleet(self.plan.hosted)
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

    def _start_fleet(self, fleet: ResearchLabWorkerFleetPlan) -> None:
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

    def _start_child(self, fleet: ResearchLabWorkerFleetPlan, index: int) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        read_fd, write_fd = os.pipe()
        env[WORKER_READY_FD_ENV] = str(write_fd)
        if fleet.kind == "hosted":
            env.setdefault("RESEARCH_LAB_HOSTED_WORKER_ENABLED", "true")
            if index < len(fleet.proxy_values):
                env["RESEARCH_LAB_HOSTED_WORKER_PROXY"] = fleet.proxy_values[index]
        else:
            env.setdefault("RESEARCH_LAB_SCORING_WORKER_ENABLED", "true")
            if index < len(fleet.proxy_values):
                env["RESEARCH_LAB_SCORING_WORKER_PROXY"] = fleet.proxy_values[index]
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
        except Exception:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
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
        hosted_running = sum(key.startswith("hosted:") for key in running)
        scoring_running = sum(key.startswith("scoring:") for key in running)
        if dead or missing_ready:
            raise ResearchLabWorkerStartupError(
                "authoritative V2 workers are not healthy: dead=%s missing_ready=%s"
                % (dead, missing_ready)
            )
        if self._full_topology_required() and (
            hosted_running != EXPECTED_HOSTED_WORKERS
            or scoring_running != EXPECTED_SCORING_WORKERS
        ):
            raise ResearchLabWorkerStartupError(
                "authoritative V2 worker count differs: hosted=%d scoring=%d"
                % (hosted_running, scoring_running)
            )
        return {
            "schema_version": "leadpoet.research_lab_worker_health.v2",
            "status": "ready",
            "topology_mode": (
                "full" if self._full_topology_required() else "component"
            ),
            "hosted_configured": self.plan.hosted.worker_count,
            "hosted_running": hosted_running,
            "scoring_configured": self.plan.scoring.worker_count,
            "scoring_running": scoring_running,
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
