"""Behavioural tests for scripts/gateway_liveness_watchdog.sh.

The watchdog's whole value is in the cases where it does NOT restart the
gateway, so those are what these tests pin: the consecutive-failure threshold,
the restart lock held by an in-flight deploy, the cooldown, and the circuit
breaker. Each test drives the real script with a stub recovery command so a
restart attempt is observable as a marker file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = REPO_ROOT / "scripts" / "gateway_liveness_watchdog.sh"

# The health URL points at a closed port so the probe always fails, and the
# interpreter path never matches a running process, so the gateway always reads
# as down unless a test says otherwise.
DEAD_HEALTH_URL = "http://127.0.0.1:9/health"
ABSENT_PYTHON_BIN = "/nonexistent/venv311/bin/python3"


pytestmark = pytest.mark.skipif(
    shutil.which("flock") is None or shutil.which("curl") is None,
    reason="watchdog needs flock and curl",
)


@pytest.fixture()
def harness(tmp_path: Path):
    marker = tmp_path / "restart-ran"
    recovery = tmp_path / "fake_gw_restart.sh"
    recovery.write_text(f'#!/bin/bash\necho ran >> "{marker}"\n')
    recovery.chmod(0o755)

    state_root = tmp_path / "watchdog"
    lock_file = tmp_path / "gateway-restart.lock"
    log_file = tmp_path / "watchdog.log"

    env = {
        **os.environ,
        "GATEWAY_WATCHDOG_ROOT": str(state_root),
        "GATEWAY_WATCHDOG_LOG_FILE": str(log_file),
        "GATEWAY_RESTART_SCRIPT": str(recovery),
        "GATEWAY_RESTART_LOCK_FILE": str(lock_file),
        "GATEWAY_PYTHON_BIN": ABSENT_PYTHON_BIN,
        "GATEWAY_HEALTH_URL": DEAD_HEALTH_URL,
    }

    class Harness:
        state_file = state_root / "state"

        def run(self, times: int = 1) -> None:
            for _ in range(times):
                subprocess.run(
                    ["bash", str(WATCHDOG)], env=env, check=True, capture_output=True
                )

        def write_state(self, failures: int, last_restart_at: int, history: str = "-") -> None:
            state_root.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(f"{failures} {last_restart_at} {history}\n")

        @property
        def restarted(self) -> bool:
            return marker.exists()

        @property
        def log(self) -> str:
            return log_file.read_text() if log_file.exists() else ""

        @property
        def lock_path(self) -> Path:
            return lock_file

    return Harness()


def test_does_not_restart_before_the_failure_threshold(harness) -> None:
    harness.run(times=2)
    assert not harness.restarted
    assert "liveness check failed (2/3)" in harness.log


def test_restarts_after_three_consecutive_failures(harness) -> None:
    harness.run(times=3)
    assert harness.restarted
    assert "running" in harness.log


def test_stands_down_while_a_deploy_holds_the_restart_lock(harness) -> None:
    harness.write_state(failures=2, last_restart_at=0)
    harness.lock_path.touch()
    holder = subprocess.Popen(
        ["flock", str(harness.lock_path), "-c", "sleep 10"],
    )
    try:
        time.sleep(0.5)
        harness.run()
    finally:
        holder.kill()
        holder.wait()
    assert not harness.restarted
    assert "already holds the restart lock" in harness.log


def test_respects_the_cooldown_between_restarts(harness) -> None:
    now = int(time.time())
    harness.write_state(failures=2, last_restart_at=now - 60, history=str(now - 60))
    harness.run()
    assert not harness.restarted
    assert "cooldown" in harness.log


def test_circuit_breaker_opens_after_three_restarts_in_the_window(harness) -> None:
    now = int(time.time())
    history = ",".join(str(now - offset) for offset in (300, 600, 900))
    harness.write_state(failures=2, last_restart_at=0, history=history)
    harness.run()
    assert not harness.restarted
    assert "circuit breaker open" in harness.log


def test_restarts_outside_the_circuit_breaker_window(harness) -> None:
    now = int(time.time())
    # Same three restarts, but all older than the six-hour window.
    history = ",".join(str(now - offset) for offset in (25_000, 30_000, 40_000))
    harness.write_state(failures=2, last_restart_at=0, history=history)
    harness.run()
    assert harness.restarted


def test_clears_the_failure_count_when_the_gateway_is_healthy(tmp_path: Path) -> None:
    """A live process plus a passing health probe resets the counter."""
    import sys

    stub_root = tmp_path / "stub"
    (stub_root / "gateway").mkdir(parents=True)
    (stub_root / "gateway" / "__init__.py").write_text("")
    (stub_root / "gateway" / "main.py").write_text("import time\ntime.sleep(30)\n")

    gateway = subprocess.Popen(
        [sys.executable, "-u", "-m", "gateway.main"], cwd=stub_root
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", "9187", "--bind", "127.0.0.1"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state_root = tmp_path / "watchdog"
    state_root.mkdir()
    (state_root / "state").write_text("2 0 -\n")
    log_file = tmp_path / "watchdog.log"
    try:
        time.sleep(1.0)
        subprocess.run(
            ["bash", str(WATCHDOG)],
            env={
                **os.environ,
                "GATEWAY_WATCHDOG_ROOT": str(state_root),
                "GATEWAY_WATCHDOG_LOG_FILE": str(log_file),
                "GATEWAY_RESTART_SCRIPT": "/bin/false",
                "GATEWAY_RESTART_LOCK_FILE": str(tmp_path / "unused.lock"),
                "GATEWAY_PYTHON_BIN": sys.executable,
                "GATEWAY_HEALTH_URL": "http://127.0.0.1:9187/",
            },
            check=True,
            capture_output=True,
        )
    finally:
        gateway.kill()
        gateway.wait()
        server.kill()
        server.wait()

    assert "healthy again" in log_file.read_text()
    assert (state_root / "state").read_text().split()[0] == "0"
