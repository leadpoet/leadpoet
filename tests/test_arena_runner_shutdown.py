"""Real parent signals must drain the standalone runner's accepted work."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Use the actual claim loop, worker socket, signed result, private completion
# receipt and delivery. Only the sandbox and remote API use local fixtures.
CHILD = r'''
import json
from pathlib import Path
import signal
import sys
import time
from lab_arena import runner, wiring
from scripts import run_lab_arena_runner as entrypoint
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, TrajectoryApi, lease, make_config, valid_company,
)

directory = Path(sys.argv[1])
phase = sys.argv[2]

class Sandbox(BridgingRuntime):
    def run_icp(self, spec, **kwargs):
        result = super().run_icp(spec, **kwargs)
        (directory / "ready").touch()
        deadline = time.monotonic() + 10
        while not (directory / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test release timeout")
            time.sleep(0.01)
        return result

api = TrajectoryApi([lease("active"), lease("must-remain-unclaimed", 1)] if phase == "active" else [])
(directory / "work").mkdir()
worker = runner.Runner(make_config(directory, api, Sandbox(
    output={"companies": [valid_company(1)]}, calls=1,
)))
worker._config.claim_poll_seconds = 0.01
original_run_once = worker.run_once
def run_once(**kwargs):
    result = original_run_once(**kwargs)
    if phase == "idle":
        (directory / "ready").touch()
    return result
worker.run_once = run_once
original_close = worker.close
def close():
    original_close()
    result = {
        "claims": len(api.claims), "provider_calls": len(api.provider_frames),
        "abandoned": worker.abandoned,
        "terminal": [row["body"]["result"]["terminal_status"] for row in api.completions],
        "events": [event["kind"] for event in api.events],
        "pending": len(list((directory / "work" / "pending-completions").glob("*.json"))),
        "remaining": [row["run_id"] for row in api.leases],
    }
    (directory / "outcome.json").write_text(json.dumps(result))
worker.close = close
def build(_):
    if phase == "startup":
        (directory / "ready").touch()
        deadline = time.monotonic() + 10
        while not (directory / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test release timeout")
            time.sleep(0.01)
    return worker
wiring.build_runner_from_environment = build
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
status = entrypoint.main(["--api-base-url", "http://127.0.0.1", "--poll-seconds", "3600"])
assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
raise SystemExit(status)
'''


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("phase", ["active", "idle", "startup"])
@pytest.mark.skipif(os.name != "posix", reason="runner host uses POSIX parent signals")
def test_parent_signal_drains_accepted_work_and_stops_claiming(tmp_path, signum, phase):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(tmp_path), phase],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, process.communicate(timeout=2)
            assert time.monotonic() < deadline, "child did not reach signal boundary"
            time.sleep(0.01)
        process.send_signal(signum)
        if phase == "active":
            time.sleep(0.1)
            assert process.poll() is None, "accepted work was killed before drain"
            assert not (tmp_path / "outcome.json").exists()
            process.send_signal(signum)  # repeated stop must still allow the drain
        (tmp_path / "release").touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        result = json.loads((tmp_path / "outcome.json").read_text())
        assert result["abandoned"] == result["pending"] == 0
        if phase == "active":
            assert result["claims"] == result["provider_calls"] == 1
            assert result["terminal"] == ["accepted"]
            assert "runtime.finished" in result["events"]
            assert result["remaining"] == ["must-remain-unclaimed"]
        else:
            assert result["provider_calls"] == 0
            assert result["terminal"] == []
            if phase == "startup":
                assert result["claims"] == 0
    finally:
        (tmp_path / "release").touch()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_once_keeps_polling_until_idle_and_restores_handlers(monkeypatch):
    from lab_arena import wiring
    from scripts import run_lab_arena_runner

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    events = []

    class Worker:
        def run_once(self, *, stop_event):
            assert not stop_event.is_set()
            events.append(stop_event)
            return 1 if len(events) == 1 else 0

        def close(self):
            assert events[0].is_set()
            events.append("closed")

    monkeypatch.setattr(wiring, "build_runner_from_environment", lambda _: Worker())
    assert run_lab_arena_runner.main(["--api-base-url", "http://127.0.0.1", "--once"]) == 0
    assert events[0] is events[1] and events[2] == "closed"
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
