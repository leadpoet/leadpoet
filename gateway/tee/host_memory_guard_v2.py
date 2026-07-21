from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import time


DISPOSABLE_TEST_ROOTS = (Path("/tmp/prtest"),)


class HostMemoryGuardV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    uid: int
    rss_kib: int
    start_ticks: int
    cwd: Path
    argv: tuple[str, ...]


def _read_process(proc_root: Path, pid: int) -> ProcessSnapshot | None:
    process_root = proc_root / str(pid)
    try:
        status = (process_root / "status").read_text(encoding="utf-8")
        stat_fields = (process_root / "stat").read_text(encoding="utf-8").split()
        argv = tuple(
            value.decode("utf-8", errors="replace")
            for value in (process_root / "cmdline").read_bytes().split(b"\0")
            if value
        )
        cwd = Path(os.readlink(process_root / "cwd"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None

    fields = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    try:
        uid = int(fields["Uid"].split()[0])
        rss_kib = int(fields.get("VmRSS", "0 kB").split()[0])
        start_ticks = int(stat_fields[21])
    except (KeyError, IndexError, ValueError):
        return None
    return ProcessSnapshot(pid, uid, rss_kib, start_ticks, cwd, argv)


def _processes(proc_root: Path) -> list[ProcessSnapshot]:
    found = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise HostMemoryGuardV2Error(f"cannot inspect process table: {exc}") from exc
    for entry in entries:
        if entry.name.isdigit():
            snapshot = _read_process(proc_root, int(entry.name))
            if snapshot is not None:
                found.append(snapshot)
    return found


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_disposable_test_process(
    process: ProcessSnapshot,
    *,
    expected_uid: int,
) -> bool:
    if process.uid != expected_uid or not process.argv:
        return False
    if not any(_is_under(process.cwd, root) for root in DISPOSABLE_TEST_ROOTS):
        return False
    executable = Path(process.argv[0]).name
    direct_pytest = executable.startswith("pytest")
    module_pytest = any(
        process.argv[index : index + 2] == ("-m", "pytest")
        for index in range(len(process.argv) - 1)
    )
    return direct_pytest or module_pytest


def _same_process(proc_root: Path, expected: ProcessSnapshot) -> bool:
    current = _read_process(proc_root, expected.pid)
    return current is not None and (
        current.start_ticks,
        current.uid,
        current.cwd,
        current.argv,
    ) == (
        expected.start_ticks,
        expected.uid,
        expected.cwd,
        expected.argv,
    )


def cleanup_disposable_tests(
    *,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    terminate_timeout_seconds: float = 10.0,
) -> list[dict[str, object]]:
    uid = os.getuid() if expected_uid is None else expected_uid
    candidates = [
        process
        for process in _processes(proc_root)
        if is_disposable_test_process(process, expected_uid=uid)
    ]
    cleaned = []
    for process in candidates:
        if not _same_process(proc_root, process):
            continue
        os.kill(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + terminate_timeout_seconds
        while _same_process(proc_root, process) and time.monotonic() < deadline:
            time.sleep(0.1)
        forced = _same_process(proc_root, process)
        if forced:
            os.kill(process.pid, signal.SIGKILL)
        cleaned.append(
            {
                "pid": process.pid,
                "rss_mib": round(process.rss_kib / 1024, 1),
                "cwd": str(process.cwd),
                "command": Path(process.argv[0]).name,
                "forced": forced,
            }
        )
    return cleaned


def available_memory_mib(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError) as exc:
        raise HostMemoryGuardV2Error(f"cannot read available memory: {exc}") from exc
    raise HostMemoryGuardV2Error("MemAvailable is missing from /proc/meminfo")


def inspect_host(
    *,
    proc_root: Path = Path("/proc"),
    meminfo_path: Path = Path("/proc/meminfo"),
    minimum_available_mib: int,
    cleanup: bool,
) -> dict[str, object]:
    cleaned = cleanup_disposable_tests(proc_root=proc_root) if cleanup else []
    available_mib = available_memory_mib(meminfo_path)
    top_processes = sorted(
        _processes(proc_root), key=lambda process: process.rss_kib, reverse=True
    )[:10]
    report = {
        "schema_version": "leadpoet.gateway_host_memory_guard.v2",
        "status": "ready" if available_mib >= minimum_available_mib else "blocked",
        "available_memory_mib": available_mib,
        "minimum_available_memory_mib": minimum_available_mib,
        "cleaned_disposable_tests": cleaned,
        "top_processes": [
            {
                "pid": process.pid,
                "rss_mib": round(process.rss_kib / 1024, 1),
                "cwd": str(process.cwd),
                "command": Path(process.argv[0]).name if process.argv else "",
            }
            for process in top_processes
        ],
    }
    return report


def watch_parent(
    parent_pid: int,
    *,
    minimum_available_mib: int,
    interval_seconds: float,
) -> int:
    while Path(f"/proc/{parent_pid}").exists():
        report = inspect_host(
            minimum_available_mib=minimum_available_mib,
            cleanup=True,
        )
        if report["cleaned_disposable_tests"]:
            print(json.dumps(report, sort_keys=True), flush=True)
        if report["status"] != "ready":
            print(json.dumps(report, sort_keys=True), flush=True)
            os.kill(parent_pid, signal.SIGTERM)
            return 2
        time.sleep(interval_seconds)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-disposable-tests", action="store_true")
    parser.add_argument("--minimum-available-mib", type=int, default=16_384)
    parser.add_argument("--watch-parent", type=int)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.minimum_available_mib < 1024:
        parser.error("--minimum-available-mib must be at least 1024")
    if args.watch_parent is not None:
        return watch_parent(
            args.watch_parent,
            minimum_available_mib=args.minimum_available_mib,
            interval_seconds=args.interval_seconds,
        )
    report = inspect_host(
        minimum_available_mib=args.minimum_available_mib,
        cleanup=args.cleanup_disposable_tests,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
