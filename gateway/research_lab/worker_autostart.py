"""Gateway startup supervisor for Research Lab worker fleets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Mapping


TRUTHY = {"1", "true", "yes", "on"}

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
    hosted_count_override = _int_env(env, "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT", 0)
    scoring_count_override = _int_env(env, "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT", 0)

    hosted_enabled = (
        auto_start_enabled
        and _truthy_env(env, "RESEARCH_LAB_AUTO_START_HOSTED_WORKERS", "true")
        and (
            _truthy_env(env, "RESEARCH_LAB_HOSTED_RUNS_ENABLED")
            or _truthy_env(env, "RESEARCH_LAB_HOSTED_WORKER_ENABLED")
        )
        and (hosted_count_override > 0 or bool(hosted_proxies))
    )
    scoring_enabled = (
        auto_start_enabled
        and _truthy_env(env, "RESEARCH_LAB_AUTO_START_SCORING_WORKERS", "true")
        and (
            _truthy_env(env, "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED")
            or _truthy_env(env, "RESEARCH_LAB_SCORING_WORKER_ENABLED")
        )
        and (scoring_count_override > 0 or bool(scoring_proxies))
    )

    hosted_reason = ""
    if not auto_start_enabled:
        hosted_reason = "auto_start_disabled"
    elif not (_truthy_env(env, "RESEARCH_LAB_HOSTED_RUNS_ENABLED") or _truthy_env(env, "RESEARCH_LAB_HOSTED_WORKER_ENABLED")):
        hosted_reason = "hosted_runs_disabled"
    elif hosted_count_override <= 0 and not hosted_proxies:
        hosted_reason = "no_auto_research_proxies"

    scoring_reason = ""
    if not auto_start_enabled:
        scoring_reason = "auto_start_disabled"
    elif not (_truthy_env(env, "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED") or _truthy_env(env, "RESEARCH_LAB_SCORING_WORKER_ENABLED")):
        scoring_reason = "evaluation_bundles_disabled"
    elif scoring_count_override <= 0 and not scoring_proxies:
        scoring_reason = "no_qualification_proxies"

    hosted_count = max(0, hosted_count_override or len(hosted_proxies))
    scoring_count = max(0, scoring_count_override or len(scoring_proxies))
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

    def start(self) -> None:
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

    def _start_child(self, fleet: ResearchLabWorkerFleetPlan, index: int) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if fleet.kind == "hosted":
            env.setdefault("RESEARCH_LAB_HOSTED_WORKER_ENABLED", "true")
            if index < len(fleet.proxy_values):
                env["RESEARCH_LAB_HOSTED_WORKER_PROXY"] = fleet.proxy_values[index]
                env["HTTP_PROXY"] = fleet.proxy_values[index]
                env["HTTPS_PROXY"] = fleet.proxy_values[index]
                env["http_proxy"] = fleet.proxy_values[index]
                env["https_proxy"] = fleet.proxy_values[index]
        else:
            env.setdefault("RESEARCH_LAB_SCORING_WORKER_ENABLED", "true")
            if index < len(fleet.proxy_values):
                env["RESEARCH_LAB_SCORING_WORKER_PROXY"] = fleet.proxy_values[index]
                env["HTTP_PROXY"] = fleet.proxy_values[index]
                env["HTTPS_PROXY"] = fleet.proxy_values[index]
                env["http_proxy"] = fleet.proxy_values[index]
                env["https_proxy"] = fleet.proxy_values[index]
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
        return subprocess.Popen(command, cwd=str(self._package_parent), env=env)

    def _monitor_children(self) -> None:
        while not self._stop_event.wait(5):
            for key, child in list(self.children.items()):
                code = child.poll()
                if code is None:
                    continue
                fleet, index = self._child_specs[key]
                if self._stop_event.is_set():
                    return
                print(
                    f"   ⚠️  Research Lab {fleet.kind} worker {index + 1} exited "
                    f"with code {code}; restarting",
                    flush=True,
                )
                self.children[key] = self._start_child(fleet, index)

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
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)
        print("   ✅ Research Lab worker fleets stopped", flush=True)
