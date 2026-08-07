"""Scoped shutdown only signals managed processes inside managed roots.

Regression suite for the 2026-08-02 incident: the previous shutdown block in
``gw_restart.sh`` ran global ``pkill -9 -f <pattern>`` and killed the live
production gateway from a CI context on the shared builder host. The scoped
module must (a) kill matching processes under the managed roots, (b) never
signal a matching process outside them, (c) escalate SIGTERM→SIGKILL only for
survivors, and (d) never signal a pid whose identity changed after selection.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

import gateway.tee.scoped_shutdown_v2 as scoped
from gateway.tee.scoped_shutdown_v2 import (
    ScopedShutdownV2Error,
    shutdown_managed_processes,
)


def _write_proc(
    proc_root: Path,
    pid: int,
    *,
    argv: list[str],
    cwd: Path,
    uid: int | None = None,
    start_ticks: int = 7777,
) -> None:
    uid = os.getuid() if uid is None else uid
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "status").write_text(
        f"Name:\t{Path(argv[0]).name}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="utf-8",
    )
    stat_fields = ["0"] * 21 + [str(start_ticks)] + ["0"] * 30
    (process_dir / "stat").write_text(" ".join(stat_fields), encoding="utf-8")
    (process_dir / "cmdline").write_bytes(
        b"\0".join(part.encode() for part in argv) + b"\0"
    )
    cwd.mkdir(parents=True, exist_ok=True)
    (process_dir / "cwd").symlink_to(cwd)


def _remove_proc(proc_root: Path, pid: int) -> None:
    process_dir = proc_root / str(pid)
    for child in process_dir.iterdir():
        child.unlink()
    process_dir.rmdir()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _run(proc_root: Path, roots: list[Path], kill):
    clock = _FakeClock()
    return shutdown_managed_processes(
        roots=roots,
        proc_root=proc_root,
        terminate_timeout_seconds=2.0,
        kill=kill,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_matching_process_outside_managed_roots_survives(tmp_path):
    """THE incident: same argv shape, different owner tree, must survive."""

    proc_root = tmp_path / "proc"
    managed = tmp_path / "leadpoet_repo"
    foreign = tmp_path / "actions-runner" / "_work" / "leadpoet" / "leadpoet"

    _write_proc(
        proc_root, 101,
        argv=["python3", "-u", "-m", "gateway.main"], cwd=managed,
    )
    _write_proc(
        proc_root, 202,
        argv=["python3", "-u", "-m", "gateway.main"], cwd=foreign,
    )

    signalled: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        signalled.append((pid, sig))
        if sig == signal.SIGTERM:
            _remove_proc(proc_root, pid)  # exits promptly on SIGTERM

    report = _run(proc_root, [managed], kill)

    assert [pid for pid, _ in signalled] == [101]
    assert report["terminated"][0]["pid"] == 101
    assert report["terminated"][0]["outcome"] == "terminated"
    assert [entry["pid"] for entry in report["out_of_scope_matches"]] == [202]


def test_second_managed_root_is_also_covered(tmp_path):
    proc_root = tmp_path / "proc"
    repo_root = tmp_path / "leadpoet_repo"
    repo_root.mkdir()
    controller_root = tmp_path / "restart-controller" / "current"

    _write_proc(
        proc_root, 111,
        argv=["python3", "-m", "gateway.utils.tee_egress_forwarder"],
        cwd=controller_root,
    )

    signalled: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        signalled.append((pid, sig))
        if sig == signal.SIGTERM:
            _remove_proc(proc_root, pid)

    report = _run(proc_root, [repo_root, controller_root], kill)

    assert [pid for pid, _ in signalled] == [111]
    assert report["terminated"][0]["component"] == "tee_egress_forwarder"


def test_survivor_is_sigkilled_after_grace(tmp_path):
    proc_root = tmp_path / "proc"
    managed = tmp_path / "repo"
    _write_proc(proc_root, 300, argv=["uvicorn", "app"], cwd=managed)

    signalled: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        signalled.append((pid, sig))  # never exits on its own

    report = _run(proc_root, [managed], kill)

    assert signalled == [(300, signal.SIGTERM), (300, signal.SIGKILL)]
    assert report["terminated"][0]["outcome"] == "sigkill_forced"


def test_identity_mismatch_before_sigterm_skips_signal(tmp_path, monkeypatch):
    """A pid whose identity changed after selection is never signalled."""

    proc_root = tmp_path / "proc"
    managed = tmp_path / "repo"
    _write_proc(
        proc_root, 400,
        argv=["python3", "-m", "gateway.main"], cwd=managed,
    )

    monkeypatch.setattr(scoped, "_same_process", lambda *_a, **_k: False)

    signalled: list[tuple[int, int]] = []
    report = _run(
        proc_root, [managed], lambda pid, sig: signalled.append((pid, sig))
    )

    assert signalled == []
    assert report["terminated"][0]["outcome"] == "exited_before_sigterm"


def test_missing_root_fails_closed(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    with pytest.raises(ScopedShutdownV2Error, match="managed root is unavailable"):
        shutdown_managed_processes(
            roots=[tmp_path / "does-not-exist"],
            proc_root=proc_root,
        )


def test_no_roots_fails_closed(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    with pytest.raises(ScopedShutdownV2Error, match="at least one managed"):
        shutdown_managed_processes(roots=[], proc_root=proc_root)


def test_other_uid_and_unmatched_argv_are_ignored(tmp_path):
    proc_root = tmp_path / "proc"
    managed = tmp_path / "repo"
    _write_proc(
        proc_root, 600,
        argv=["python3", "-m", "gateway.main"], cwd=managed,
        uid=os.getuid() + 1,
    )
    _write_proc(
        proc_root, 601,
        argv=["python3", "-m", "gateway.tasks.epoch_monitor_helper"],
        cwd=managed,
    )

    signalled: list[tuple[int, int]] = []
    report = _run(
        proc_root, [managed], lambda pid, sig: signalled.append((pid, sig))
    )
    assert signalled == []
    assert report["terminated"] == []
    assert report["out_of_scope_matches"] == []


def test_gw_restart_wires_scoped_shutdown_fail_closed():
    root = Path(__file__).resolve().parents[1]
    script = (root / "gw_restart.sh").read_text(encoding="utf-8")
    assert 'gateway/tee/scoped_shutdown_v2.py' in script
    assert '--root $LEADPOET_REPO_ROOT' in script
    assert 'GATEWAY_RESTART_CONTROLLER_CURRENT' in script
    assert 'refusing global pkill fallback' in script
    # The global name-pattern kills must never come back.
    for pattern in (
        'pkill -9 -f "python3 main.py"',
        'pkill -9 -f "python3 -u main.py"',
        'pkill -9 -f "python3 -u -m gateway.main"',
        'pkill -9 -f "uvicorn"',
        'pkill -9 -f "gateway/research_lab/worker_process.py"',
        'pkill -9 -f "run_research_lab_hosted_worker"',
        'pkill -9 -f "run_research_lab_scoring_worker"',
        'pkill -9 -f "gateway.research_lab.provider_evidence_proxy"',
        'pkill -9 -f "provider_evidence_proxy"',
        'pkill -9 -f "gateway.utils.tee_inter_enclave_relay"',
        'pkill -9 -f "gateway.utils.tee_egress_forwarder"',
    ):
        assert pattern not in script, pattern
