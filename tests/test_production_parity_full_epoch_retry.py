from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_production_parity_full_host as full_host


CANDIDATE_SHA = "b" * 40
COMMAND = ["bash", str(full_host.ROOT / "gw_restart.sh"), "--commit", CANDIDATE_SHA]


def _emit_real_epoch_rejection(
    log_path: Path,
    *,
    observed: int = 314,
    module_mode: bool = False,
) -> None:
    direct_source = f"""
from Leadpoet.utils import restart_epoch_gate as gate
class Snapshot:
    network_genesis_hash = "0x" + "1" * 64
    netuid = 71
    current_block = 1000
    subnet_epoch_index = 4
    epoch_block = {observed}
    def to_dict(self):
        return {{}}
gate.read_subnet_epoch_snapshot = lambda *_args, **_kwargs: Snapshot()
gate.verify_restart_epoch_window(object())
"""
    module_source = f"""
import runpy
import sys
import types
from Leadpoet.utils import subnet_epoch
class Snapshot:
    network_genesis_hash = "0x" + "1" * 64
    netuid = 71
    current_block = 1000
    subnet_epoch_index = 4
    epoch_block = {observed}
    def to_dict(self):
        return {{}}
subnet_epoch.read_subnet_epoch_snapshot = lambda *_args, **_kwargs: Snapshot()
sys.modules["bittensor"] = types.SimpleNamespace(Subtensor=lambda **_kwargs: object())
runpy.run_module("Leadpoet.utils.restart_epoch_gate", run_name="__main__")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(full_host.ROOT)
    result = subprocess.run(
        [sys.executable, "-c", module_source if module_mode else direct_source],
        cwd=full_host.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 1
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")


def test_restart_epoch_observation_accepts_canonical_module_traceback(tmp_path: Path):
    log_path = tmp_path / "canonical-module.log"
    _emit_real_epoch_rejection(log_path, module_mode=True)

    assert full_host._restart_epoch_gate_observation(log_path) == {
        "marker": "restart_epoch_gate_observation",
        "maximum_restart_epoch_block": 300,
        "observed_epoch_block": 314,
        "reason": "epoch_block_after_restart_deadline",
    }


def _run_retry(
    tmp_path: Path,
    *,
    deadline: float,
):
    return full_host._run_gateway_restart_with_epoch_retry(
        COMMAND,
        env={"GATEWAY_DEPLOY_COMMIT": CANDIDATE_SHA},
        log_path=tmp_path / "gateway-restart.log",
        timing_dir=tmp_path / "restart-timings",
        deadline=deadline,
        candidate_sha=CANDIDATE_SHA,
    )


def test_full_gateway_restart_retries_real_epoch_gate_traceback_then_succeeds(
    monkeypatch,
    tmp_path: Path,
):
    clock = [100.0]
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            _emit_real_epoch_rejection(Path(kwargs["log_path"]))
            return subprocess.CompletedProcess(command, 75)
        Path(kwargs["log_path"]).write_text("gateway ready\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(full_host, "_run", fake_run)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(full_host.time, "sleep", fake_sleep)

    result, timeout, diagnostic, log_path, timing_dir = _run_retry(
        tmp_path,
        deadline=1000.0,
    )

    assert result is not None and result.returncode == 0
    assert timeout is None
    assert sleeps == [60]
    assert [call[1]["timeout"] for call in calls] == [900, 840]
    assert [Path(call[1]["log_path"]).name for call in calls] == [
        "gateway-restart-attempt-1.log",
        "gateway-restart-attempt-2.log",
    ]
    assert [
        Path(call[1]["env"]["GATEWAY_RESTART_TIMING_DIR"]).name for call in calls
    ] == ["attempt-1", "attempt-2"]
    assert log_path.name == "gateway-restart-attempt-2.log"
    assert timing_dir.name == "attempt-2"
    assert diagnostic == {
        "outcome": "exited",
        "returncode": 0,
        "epoch_gate_rejections": [
            {
                "attempt": 1,
                "marker": "restart_epoch_gate_observation",
                "maximum_restart_epoch_block": 300,
                "observed_epoch_block": 314,
                "reason": "epoch_block_after_restart_deadline",
            }
        ],
    }


@pytest.mark.parametrize(
    ("returncode", "body"),
    [
        (1, "unrelated failure\n"),
        (75, "RestartEpochGateError: unrelated\n"),
        (
            75,
            "Traceback (most recent call last):\n"
            '  File "/tmp/other.py", line 1, in verify_restart_epoch_window\n'
            "RestartEpochGateError: production restart may start only at official "
            "subnet epoch block 300 or earlier; observed 314\n",
        ),
        (
            75,
            "RestartEpochGateError: production restart may start only at official "
            "subnet epoch block 300 or earlier; observed 314\n",
        ),
        (
            75,
            "Traceback (most recent call last):\n"
            '  File "/run/candidate/Leadpoet/utils/restart_epoch_gate.py", line 51, '
            "in verify_restart_epoch_window\n"
            "RestartEpochGateError: production restart may start only at official "
            "subnet epoch block 300 or earlier; observed 314\n"
            "RuntimeError: later unrelated failure\n",
        ),
    ],
)
def test_full_gateway_restart_does_not_retry_other_failures(
    monkeypatch,
    tmp_path: Path,
    returncode: int,
    body: str,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(kwargs["log_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["log_path"]).write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(full_host, "_run", fake_run)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        full_host.time,
        "sleep",
        lambda _seconds: pytest.fail("untrusted restart failure was retried"),
    )

    result, timeout, diagnostic, _log, _timing = _run_retry(
        tmp_path,
        deadline=1000.0,
    )

    assert result is not None and result.returncode == returncode
    assert timeout is None
    assert diagnostic == {"outcome": "exited", "returncode": returncode}
    assert len(calls) == 1


def test_full_gateway_restart_does_not_reuse_stale_epoch_rejection(
    monkeypatch,
    tmp_path: Path,
):
    clock = [100.0]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs)
        log_path = Path(kwargs["log_path"])
        if len(calls) == 1:
            _emit_real_epoch_rejection(log_path)
        else:
            log_path.write_text("ERROR: unrelated ancestry failure\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 75)

    monkeypatch.setattr(full_host, "_run", fake_run)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        full_host.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    result, timeout, diagnostic, log_path, _timing = _run_retry(
        tmp_path,
        deadline=1000.0,
    )

    assert result is not None and result.returncode == 75
    assert timeout is None
    assert len(calls) == 2
    assert log_path.name == "gateway-restart-attempt-2.log"
    assert len(diagnostic["epoch_gate_rejections"]) == 1


def test_full_gateway_restart_epoch_retry_respects_existing_deadline(
    monkeypatch,
    tmp_path: Path,
):
    clock = [100.0]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs)
        _emit_real_epoch_rejection(Path(kwargs["log_path"]), observed=315)
        clock[0] = 101.0
        return subprocess.CompletedProcess(command, 75)

    monkeypatch.setattr(full_host, "_run", fake_run)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        full_host.time,
        "sleep",
        lambda _seconds: pytest.fail("expired restart deadline was slept through"),
    )

    result, timeout, diagnostic, _log, _timing = _run_retry(
        tmp_path,
        deadline=101.0,
    )

    assert result is not None and result.returncode == 75
    assert timeout is None
    assert len(calls) == 1
    assert diagnostic["retry_deadline_exhausted"] is True


def test_full_gateway_restart_timeout_after_rejection_keeps_attempt_identity(
    monkeypatch,
    tmp_path: Path,
):
    clock = [100.0]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            _emit_real_epoch_rejection(Path(kwargs["log_path"]))
            return subprocess.CompletedProcess(command, 75)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(full_host, "_run", fake_run)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        full_host.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    result, timeout, diagnostic, log_path, timing_dir = _run_retry(
        tmp_path,
        deadline=20_000.0,
    )

    assert result is None
    assert isinstance(timeout, subprocess.TimeoutExpired)
    assert [call["timeout"] for call in calls] == [10_800, 10_740]
    assert diagnostic["outcome"] == "timed_out"
    assert diagnostic["timeout_seconds"] == 10_740
    assert len(diagnostic["epoch_gate_rejections"]) == 1
    assert log_path.name == "gateway-restart-attempt-2.log"
    assert timing_dir.name == "attempt-2"


def test_full_gateway_restart_retry_wait_cannot_cross_restart_deadline(
    monkeypatch,
    tmp_path: Path,
):
    clock = [100.0]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs)
        _emit_real_epoch_rejection(Path(kwargs["log_path"]))
        return subprocess.CompletedProcess(command, 75)

    monkeypatch.setattr(full_host, "_run", fake_run)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        full_host.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    result, timeout, diagnostic, _log, _timing = _run_retry(
        tmp_path,
        deadline=160.0,
    )

    assert result is not None and result.returncode == 75
    assert timeout is None
    assert len(calls) == 1
    assert clock[0] == 160.0
    assert diagnostic["retry_deadline_exhausted"] is True


def test_full_gateway_restart_epoch_diagnostic_does_not_retain_raw_log(
    monkeypatch,
    tmp_path: Path,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs)
        log_path = Path(kwargs["log_path"])
        if len(calls) == 1:
            _emit_real_epoch_rejection(log_path)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("PRIVATE_TOKEN=must-not-survive\n")
            return subprocess.CompletedProcess(command, 75)
        log_path.write_text("gateway ready\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(full_host, "_run", fake_run)
    clock = [100.0]
    monkeypatch.setattr(full_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        full_host.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    _result, _timeout, diagnostic, _log, _timing = _run_retry(
        tmp_path,
        deadline=1000.0,
    )

    assert diagnostic["outcome"] == "exited"
    assert diagnostic["returncode"] == 0
    assert len(diagnostic["epoch_gate_rejections"]) == 1
    assert "must-not-survive" not in json.dumps(diagnostic)


def test_full_gateway_restart_retry_rejects_other_candidate_identity(tmp_path: Path):
    with pytest.raises(full_host.FullParityError, match="retry identity is invalid"):
        full_host._run_gateway_restart_with_epoch_retry(
            COMMAND,
            env={"GATEWAY_DEPLOY_COMMIT": "c" * 40},
            log_path=tmp_path / "gateway-restart.log",
            timing_dir=tmp_path / "restart-timings",
            deadline=1000.0,
            candidate_sha=CANDIDATE_SHA,
        )
