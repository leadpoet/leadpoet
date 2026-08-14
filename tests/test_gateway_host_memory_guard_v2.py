from __future__ import annotations

import os
from pathlib import Path
import shutil

from gateway.tee.host_memory_guard_v2 import (
    ProcessSnapshot,
    available_memory_mib,
    cleanup_disposable_tests,
    cleanup_stale_vsock_probes,
    inspect_host,
    is_disposable_test_process,
    stale_vsock_probe_label,
)


def _process(*, cwd: str, argv: tuple[str, ...], uid: int | None = None):
    return ProcessSnapshot(
        pid=123,
        ppid=1,
        uid=os.getuid() if uid is None else uid,
        rss_kib=1024,
        start_ticks=99,
        cpu_ticks=0,
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
    assert report["cleaned_stale_vsock_probes"] == []


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


def _vsock_probe_snapshots(*, child_cpu_ticks: int = 95_000):
    uid = os.getuid()
    command = (
        'python3 - <<\'PY\'\nimport json,socket\n'
        'for method in ("v2_provider_broker_health",'
        '"v2_provider_semantics_health"):\n'
        ' s=socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM); data=b""\n'
        ' while b"\\n" not in data: data+=s.recv(65536)\nPY'
    )
    parent = ProcessSnapshot(
        pid=122,
        ppid=1,
        uid=uid,
        rss_kib=1024,
        start_ticks=100,
        cpu_ticks=0,
        cwd=Path("/home/ec2-user"),
        argv=("bash", "-c", command),
    )
    child = ProcessSnapshot(
        pid=123,
        ppid=parent.pid,
        uid=uid,
        rss_kib=1024,
        start_ticks=100,
        cpu_ticks=child_cpu_ticks,
        cwd=Path("/home/ec2-user"),
        argv=("python3", "-"),
    )
    return child, parent


def test_stale_vsock_probe_requires_exact_loop_identity_age_and_cpu() -> None:
    child, parent = _vsock_probe_snapshots()
    assert stale_vsock_probe_label(
        child,
        parent,
        expected_uid=os.getuid(),
        uptime_seconds=1_000,
        clock_ticks_per_second=100,
    ) == "v2_provider_broker_health+v2_provider_semantics_health"
    assert stale_vsock_probe_label(
        _process(
            cwd="/home/ec2-user/leadpoet_repo",
            argv=("python3", "-u", "-m", "gateway.main"),
        ),
        parent,
        expected_uid=os.getuid(),
        uptime_seconds=1_000,
        clock_ticks_per_second=100,
    ) is None
    low_cpu_child, _ = _vsock_probe_snapshots(child_cpu_ticks=100)
    assert stale_vsock_probe_label(
        low_cpu_child,
        parent,
        expected_uid=os.getuid(),
        uptime_seconds=1_000,
        clock_ticks_per_second=100,
    ) is None


def test_cleanup_stale_vsock_probe_revalidates_child_and_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proc_root = tmp_path / "proc"
    child, parent = _vsock_probe_snapshots()
    for process in (parent, child):
        process_root = proc_root / str(process.pid)
        process_root.mkdir(parents=True)
        (process_root / "status").write_text(
            f"Uid:\t{process.uid}\t{process.uid}\t{process.uid}\t{process.uid}\n"
            "VmRSS:\t1024 kB\n",
            encoding="utf-8",
        )
        fields = ["0"] * 22
        fields[0] = str(process.pid)
        fields[1] = "(python3)"
        fields[3] = str(process.ppid)
        fields[13] = str(process.cpu_ticks)
        fields[21] = str(process.start_ticks)
        (process_root / "stat").write_text(" ".join(fields), encoding="utf-8")
        (process_root / "cmdline").write_bytes(
            b"\0".join(value.encode() for value in process.argv) + b"\0"
        )
        (process_root / "cwd").symlink_to(process.cwd)

    signals = []

    def terminate(pid: int, sent_signal: int) -> None:
        signals.append((pid, sent_signal))
        shutil.rmtree(proc_root / str(pid))

    monkeypatch.setattr("gateway.tee.host_memory_guard_v2.os.kill", terminate)
    cleaned = cleanup_stale_vsock_probes(
        proc_root=proc_root,
        expected_uid=os.getuid(),
        terminate_timeout_seconds=0,
        uptime_seconds=1_000,
        clock_ticks_per_second=100,
    )

    assert signals == [(child.pid, 15)]
    assert cleaned == [
        {
            "pid": child.pid,
            "label": "v2_provider_broker_health+v2_provider_semantics_health",
            "forced": False,
        }
    ]


def test_gateway_restart_uses_bounded_memory_gates_without_async_termination() -> None:
    root = Path(__file__).resolve().parents[1]
    restart = (root / "gw_restart.sh").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/attested-v2-release.yml").read_text(
        encoding="utf-8"
    )

    assert "wait_for_gateway_build_memory" in restart
    assert 'GATEWAY_HOST_MEMORY_GUARD_PATH="${GATEWAY_HOST_MEMORY_GUARD_PATH:-' in restart
    assert 'local guard="$GATEWAY_HOST_MEMORY_GUARD_PATH"' in restart
    assert "--minimum-available-mib 16384" in restart
    assert restart.count("--cleanup-stale-vsock-probes") == 3
    prepared_cleanup = restart.index(
        'echo "Cleaning stale read-only gateway vsock probes before V2 preflight"'
    )
    restart_gate = restart.index(
        'echo "Capturing the official subnet restart window before release acquisition"'
    )
    weight_preflight = restart.index(
        'echo "Preflighting durable V2 validator weight authority before production shutdown"'
    )
    assert prepared_cleanup < restart_gate < weight_preflight
    assert (
        '"$GATEWAY_PREFLIGHT_TREE/gateway/tee/host_memory_guard_v2.py"'
        in restart
    )
    assert "--minimum-available-mib 4096" in workflow
    assert "--watch-parent" not in restart
    assert "--watch-parent" not in (
        root / "gateway" / "tee" / "host_memory_guard_v2.py"
    ).read_text(encoding="utf-8")
    assert "Guard gateway builder host memory" in workflow
    assert "--watch-parent" not in workflow
