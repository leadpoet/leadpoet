"""Detect foreign Docker builders before shared-host maintenance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Optional


class DockerOperationGuardV2Error(RuntimeError):
    pass


_EXACT_HOST_GATEWAY_ARGS = (b"-u", b"-m", b"gateway.main")
_HOST_GATEWAY_PYTHON_COMMAND = re.compile(r"python(?:3(?:\.\d+)?)?\Z")
_MAX_HOST_GATEWAY_CMDLINE_BYTES = 4096


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


def inspect_exact_host_gateway_runtime(
    *,
    proc_root: Path = Path("/proc"),
    max_process_entries: int = 32_768,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Find the exact host gateway launcher without an unbounded process grep."""

    if max_process_entries < 1:
        raise DockerOperationGuardV2Error("max_process_entries must be positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise DockerOperationGuardV2Error("timeout_seconds must be positive")

    deadline = time.monotonic() + timeout_seconds
    scanned_entry_count = 0
    scanned_process_count = 0
    try:
        with os.scandir(proc_root) as entries:
            for entry in entries:
                scanned_entry_count += 1
                if scanned_entry_count > max_process_entries:
                    raise DockerOperationGuardV2Error(
                        "process table exceeded the bounded host gateway scan"
                    )
                if time.monotonic() >= deadline:
                    raise DockerOperationGuardV2Error(
                        "host gateway process scan exceeded its deadline"
                    )
                if not entry.name.isdigit():
                    continue
                scanned_process_count += 1

                try:
                    with (Path(entry.path) / "cmdline").open("rb") as handle:
                        raw_cmdline = handle.read(_MAX_HOST_GATEWAY_CMDLINE_BYTES + 1)
                except (FileNotFoundError, ProcessLookupError):
                    # Processes can exit between the directory and cmdline reads.
                    continue
                except OSError as exc:
                    raise DockerOperationGuardV2Error(
                        f"cannot inspect process {entry.name} cmdline: {exc}"
                    ) from exc

                # The production gateway argv is short. An oversized argv cannot
                # equal it, so no unbounded read is needed.
                if len(raw_cmdline) > _MAX_HOST_GATEWAY_CMDLINE_BYTES:
                    continue
                argv = tuple(item for item in raw_cmdline.split(b"\0") if item)
                if len(argv) != 4 or argv[1:] != _EXACT_HOST_GATEWAY_ARGS:
                    continue
                try:
                    command = Path(os.fsdecode(argv[0])).name
                except UnicodeError:
                    continue
                if _HOST_GATEWAY_PYTHON_COMMAND.fullmatch(command) is None:
                    continue
                return {
                    "schema_version": "leadpoet.host_gateway_process_guard.v1",
                    "status": "live",
                    "gateway_process": {
                        "pid": int(entry.name),
                        "command": command,
                    },
                    "scanned_entry_count": scanned_entry_count,
                    "scanned_process_count": scanned_process_count,
                }
    except DockerOperationGuardV2Error:
        raise
    except OSError as exc:
        raise DockerOperationGuardV2Error(
            f"cannot inspect process table: {exc}"
        ) from exc

    return {
        "schema_version": "leadpoet.host_gateway_process_guard.v1",
        "status": "absent",
        "gateway_process": None,
        "scanned_entry_count": scanned_entry_count,
        "scanned_process_count": scanned_process_count,
    }


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wait", action="store_true")
    mode.add_argument("--detect-exact-host-gateway", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--interval-seconds", type=float, default=3)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--max-process-entries", type=int, default=32_768)
    parser.add_argument("--scan-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be non-negative")
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.max_process_entries < 1:
        parser.error("--max-process-entries must be positive")
    if not math.isfinite(args.scan_timeout_seconds) or args.scan_timeout_seconds <= 0:
        parser.error("--scan-timeout-seconds must be positive")

    try:
        if args.detect_exact_host_gateway:
            report = inspect_exact_host_gateway_runtime(
                proc_root=args.proc_root,
                max_process_entries=args.max_process_entries,
                timeout_seconds=args.scan_timeout_seconds,
            )
        elif args.wait:
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
    except DockerOperationGuardV2Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.detect_exact_host_gateway:
        return 0
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
