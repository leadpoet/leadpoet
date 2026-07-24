"""Detect foreign Docker builders before shared-host maintenance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Optional


class DockerOperationGuardV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent_pid: int
    command: str
    argv: tuple[str, ...]


def _read_process(proc_root: Path, pid: int) -> Optional[ProcessInfo]:
    root = proc_root / str(pid)
    try:
        status = (root / "status").read_text(encoding="utf-8")
        argv = tuple(
            item.decode("utf-8", errors="replace")
            for item in (root / "cmdline").read_bytes().split(b"\0")
            if item
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    parent_pid = None
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                parent_pid = int(line.split()[1])
            except (IndexError, ValueError):
                return None
            break
    if parent_pid is None or not argv:
        return None
    return ProcessInfo(
        pid=pid,
        parent_pid=parent_pid,
        command=Path(argv[0]).name,
        argv=argv,
    )


def _all_processes(proc_root: Path) -> dict[int, ProcessInfo]:
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise DockerOperationGuardV2Error(
            f"cannot inspect process table: {exc}"
        ) from exc
    processes = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        process = _read_process(proc_root, int(entry.name))
        if process is not None:
            processes[process.pid] = process
    return processes


def _ancestor_pids(
    processes: dict[int, ProcessInfo],
    *,
    current_pid: int,
) -> set[int]:
    ancestors = {current_pid}
    cursor = current_pid
    while cursor in processes:
        parent = processes[cursor].parent_pid
        if parent <= 0 or parent in ancestors:
            break
        ancestors.add(parent)
        cursor = parent
    return ancestors


def _operation_kind(process: ProcessInfo) -> str | None:
    command = process.command
    args = process.argv[1:]
    if command in {"docker-buildx", "buildctl"}:
        return "docker_builder"
    if command == "nitro-cli" and "build-enclave" in args:
        return "nitro_enclave_build"
    if command not in {"docker", "sudo"}:
        return None

    tokens = args
    if command == "sudo":
        try:
            docker_index = next(
                index
                for index, token in enumerate(tokens)
                if Path(token).name == "docker"
            )
        except StopIteration:
            return None
        tokens = tokens[docker_index + 1 :]

    joined = " ".join(tokens)
    if (
        "buildx build" in joined
        or "buildx bake" in joined
        or "compose build" in joined
        or "--build" in tokens
        or (tokens and tokens[0] == "build")
    ):
        return "docker_build"
    if any(
        marker in joined
        for marker in (
            "builder prune",
            "system prune",
            "image prune",
            "container prune",
            "volume prune",
        )
    ):
        return "docker_maintenance"
    return None


def find_foreign_docker_operations(
    *,
    proc_root: Path = Path("/proc"),
    current_pid: Optional[int] = None,
) -> list[dict[str, object]]:
    pid = os.getpid() if current_pid is None else current_pid
    processes = _all_processes(proc_root)
    ancestors = _ancestor_pids(processes, current_pid=pid)
    blockers = []
    for process in processes.values():
        if process.pid in ancestors:
            continue
        kind = _operation_kind(process)
        if kind is None:
            continue
        blockers.append(
            {
                "pid": process.pid,
                "command": process.command,
                "kind": kind,
            }
        )
    return sorted(blockers, key=lambda item: int(item["pid"]))


def wait_for_foreign_docker_operations(
    *,
    timeout_seconds: float,
    interval_seconds: float,
    proc_root: Path = Path("/proc"),
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_blockers: list[dict[str, object]] = []
    while True:
        last_blockers = find_foreign_docker_operations(proc_root=proc_root)
        if not last_blockers:
            return {
                "schema_version": "leadpoet.docker_operation_guard.v2",
                "status": "ready",
                "foreign_operations": [],
            }
        if time.monotonic() >= deadline:
            return {
                "schema_version": "leadpoet.docker_operation_guard.v2",
                "status": "blocked",
                "foreign_operations": last_blockers,
            }
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--interval-seconds", type=float, default=3)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args()
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be non-negative")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    if args.wait:
        report = wait_for_foreign_docker_operations(
            timeout_seconds=args.timeout_seconds,
            interval_seconds=args.interval_seconds,
            proc_root=args.proc_root,
        )
    else:
        blockers = find_foreign_docker_operations(proc_root=args.proc_root)
        report = {
            "schema_version": "leadpoet.docker_operation_guard.v2",
            "status": "ready" if not blockers else "blocked",
            "foreign_operations": blockers,
        }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
