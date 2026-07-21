from __future__ import annotations

import os
from pathlib import Path
import shutil

from gateway.tee.host_memory_guard_v2 import (
    ProcessSnapshot,
    available_memory_mib,
    cleanup_disposable_tests,
    inspect_host,
    is_disposable_test_process,
)


def _process(*, cwd: str, argv: tuple[str, ...], uid: int | None = None):
    return ProcessSnapshot(
        pid=123,
        uid=os.getuid() if uid is None else uid,
        rss_kib=1024,
        start_ticks=99,
        cwd=Path(cwd),
        argv=argv,
    )


def test_only_exact_prtest_pytest_process_is_disposable() -> None:
    uid = os.getuid()
    assert is_disposable_test_process(
        _process(cwd="/tmp/prtest", argv=("python3", "-m", "pytest", "tests")),
        expected_uid=uid,
    )
    assert not is_disposable_test_process(
        _process(cwd="/tmp/prtest", argv=("python3", "gateway.main")),
        expected_uid=uid,
    )
    assert not is_disposable_test_process(
        _process(cwd="/home/ec2-user/leadpoet_repo", argv=("pytest", "tests")),
        expected_uid=uid,
    )
    assert not is_disposable_test_process(
        _process(
            cwd="/tmp/prtest",
            argv=("pytest", "tests"),
            uid=uid + 1,
        ),
        expected_uid=uid,
    )


def test_memory_reader_and_inspection_fail_closed(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:       8388608 kB\n", encoding="utf-8")

    assert available_memory_mib(meminfo) == 8192
    report = inspect_host(
        proc_root=proc_root,
        meminfo_path=meminfo,
        minimum_available_mib=16_384,
        cleanup=False,
    )
    assert report["status"] == "blocked"
    assert report["available_memory_mib"] == 8192
    assert report["cleaned_disposable_tests"] == []


def test_cleanup_revalidates_and_terminates_only_matching_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proc_root = tmp_path / "proc"
    process_root = proc_root / "123"
    process_root.mkdir(parents=True)
    (process_root / "status").write_text(
        f"Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n"
        "VmRSS:\t2048 kB\n",
        encoding="utf-8",
    )
    stat_fields = ["0"] * 22
    stat_fields[0] = "123"
    stat_fields[1] = "(python3)"
    stat_fields[21] = "99"
    (process_root / "stat").write_text(" ".join(stat_fields), encoding="utf-8")
    (process_root / "cmdline").write_bytes(b"python3\0-m\0pytest\0tests\0")
    (process_root / "cwd").symlink_to("/tmp/prtest")

    signals = []

    def terminate(pid: int, sent_signal: int) -> None:
        signals.append((pid, sent_signal))
        shutil.rmtree(process_root)

    monkeypatch.setattr("gateway.tee.host_memory_guard_v2.os.kill", terminate)

    cleaned = cleanup_disposable_tests(
        proc_root=proc_root,
        expected_uid=os.getuid(),
        terminate_timeout_seconds=0,
    )

    assert signals == [(123, 15)]
    assert cleaned == [
        {
            "pid": 123,
            "rss_mib": 2.0,
            "cwd": "/tmp/prtest",
            "command": "python3",
            "forced": False,
        }
    ]


def test_gateway_restart_and_attestation_wire_the_memory_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    restart = (root / "gw_restart.sh").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/attested-v2-release.yml").read_text(
        encoding="utf-8"
    )

    assert "start_gateway_host_memory_guard" in restart
    assert "--watch-parent \"$$\"" in restart
    assert "--minimum-available-mib 16384" in restart
    assert "--minimum-available-mib 4096" in restart
    assert "Guard gateway builder host memory" in workflow
    assert "--watch-parent \"$BASHPID\"" in workflow
