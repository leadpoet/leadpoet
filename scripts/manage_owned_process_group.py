#!/usr/bin/env python3
"""Own and stop one detached Linux process group without broad name matching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import time
from typing import Any, Optional


STATE_VERSION = 1


class OwnershipError(RuntimeError):
    """The requested process group cannot be proved to be ours."""


class ProcessUnreadable(OwnershipError):
    """A present process cannot be inspected safely."""


def _read_process(pid: int) -> Optional[dict[str, Any]]:
    proc = Path("/proc") / str(pid)
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        fields = raw_stat[raw_stat.rfind(")") + 2 :].split()
        cmdline = (proc / "cmdline").read_bytes().split(b"\0")
        argv = [
            item.decode("utf-8", errors="surrogateescape") for item in cmdline if item
        ]
        return {
            "pid": pid,
            "state": fields[0],
            "pgid": int(fields[2]),
            "session": int(fields[3]),
            "start_time_ticks": int(fields[19]),
            "uid": proc.stat().st_uid,
            "cwd": os.path.realpath(os.readlink(proc / "cwd")),
            "argv": argv,
        }
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise ProcessUnreadable(
            f"process identity is unreadable for PID {pid}"
        ) from exc
    except (ValueError, IndexError) as exc:
        raise OwnershipError(f"process identity is invalid for PID {pid}") from exc


def _all_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            try:
                process = _read_process(int(item.name))
            except ProcessUnreadable:
                continue
            if process is not None:
                processes.append(process)
    return processes


def _matching_processes(
    *, argv: list[str], cwd: str, uid: int, pgid: Optional[int] = None
) -> list[dict[str, Any]]:
    expected_cwd = os.path.realpath(cwd)
    matches = []
    for process in _all_processes():
        if (
            process["argv"] == argv
            and process["cwd"] == expected_cwd
            and process["uid"] == uid
            and process["session"] == process["pgid"]
            and (pgid is None or process["pgid"] == pgid)
        ):
            matches.append(process)
    return matches


def _expected_identity(*, argv: list[str], cwd: str, uid: int) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "uid": uid,
        "cwd": os.path.realpath(cwd),
        "argv": argv,
    }


def _state_for(process: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "pid": process["pid"],
        "pgid": process["pgid"],
        "start_time_ticks": process["start_time_ticks"],
        "uid": process["uid"],
        "cwd": process["cwd"],
        "argv": process["argv"],
    }


def _load_state(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise OwnershipError("process state is unavailable") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise OwnershipError(f"process state is not a regular file: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OwnershipError(f"process state is invalid: {path}: {exc}") from exc
    required = {
        "version": int,
        "pid": int,
        "pgid": int,
        "start_time_ticks": int,
        "uid": int,
        "cwd": str,
        "argv": list,
    }
    if not isinstance(state, dict) or any(
        not isinstance(state.get(key), value_type)
        for key, value_type in required.items()
    ):
        raise OwnershipError(f"process state has an invalid schema: {path}")
    if any(state[key] != value for key, value in expected.items()):
        raise OwnershipError(
            f"process state identity differs from the requested instance: {path}"
        )
    if state["pid"] <= 1 or state["pgid"] <= 1 or state["start_time_ticks"] <= 0:
        raise OwnershipError(
            f"process state contains an unsafe process identity: {path}"
        )
    return state


def _write_state(path: Path, process: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(_state_for(process), sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        try:
            os.link(temporary_name, path, follow_symlinks=False)
        except FileExistsError:
            raise OwnershipError(
                f"refusing to replace existing process state: {path}"
            ) from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _same_process(process: Optional[dict[str, Any]], state: dict[str, Any]) -> bool:
    return (
        process is not None
        and all(
            process[key] == state[key]
            for key in ("pid", "pgid", "start_time_ticks", "uid", "cwd", "argv")
        )
        and process["session"] == process["pgid"]
    )


def _active_group_members(pgid: int) -> list[dict[str, Any]]:
    return [
        process
        for process in _all_processes()
        if process["pgid"] == pgid and process["state"] != "Z"
    ]


def _wait_for_group_exit(pgid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _active_group_members(pgid):
            return True
        time.sleep(0.05)
    return not _active_group_members(pgid)


def _stop_state(
    path: Path, state: dict[str, Any], term_seconds: float, kill_seconds: float
) -> None:
    process = _read_process(state["pid"])
    if process is None or process["state"] == "Z":
        if _active_group_members(state["pgid"]):
            raise OwnershipError(
                "owned leader is gone but its process group is still active"
            )
        path.unlink()
        return
    if not _same_process(process, state):
        raise OwnershipError("saved PID was reused or its process identity changed")
    os.killpg(state["pgid"], signal.SIGTERM)
    if not _wait_for_group_exit(state["pgid"], term_seconds):
        os.killpg(state["pgid"], signal.SIGKILL)
        if not _wait_for_group_exit(state["pgid"], kill_seconds):
            raise OwnershipError("owned process group remained active after SIGKILL")
    path.unlink()


def _discover_one(
    args: argparse.Namespace, pgid: Optional[int] = None
) -> dict[str, Any]:
    deadline = time.monotonic() + args.discover_timeout_seconds
    while True:
        matches = _matching_processes(
            argv=args.process_argv, cwd=args.cwd, uid=args.uid, pgid=pgid
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise OwnershipError(
                "more than one exact detached process matches; refusing to choose"
            )
        if time.monotonic() >= deadline:
            raise OwnershipError("no exact detached process matches")
        time.sleep(0.05)


def _record(args: argparse.Namespace) -> None:
    process = _discover_one(args, pgid=args.launch_pgid)
    _write_state(args.state_file, process)


def _stop(args: argparse.Namespace) -> None:
    expected = _expected_identity(argv=args.process_argv, cwd=args.cwd, uid=args.uid)
    if args.state_file.exists() or args.state_file.is_symlink():
        state = _load_state(args.state_file, expected)
        _stop_state(args.state_file, state, args.term_seconds, args.kill_seconds)
        return

    matches = _matching_processes(argv=args.process_argv, cwd=args.cwd, uid=args.uid)
    if not matches:
        return
    if len(matches) != 1:
        raise OwnershipError(
            "more than one exact legacy process matches; refusing to choose"
        )
    _write_state(args.state_file, matches[0])
    state = _load_state(args.state_file, expected)
    _stop_state(args.state_file, state, args.term_seconds, args.kill_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("record", "stop"):
        command = actions.add_parser(action)
        command.add_argument("--state-file", type=Path, required=True)
        command.add_argument("--cwd", required=True)
        command.add_argument("--uid", type=int, required=True)
        command.add_argument("--launch-pgid", type=int)
        command.add_argument("--discover-timeout-seconds", type=float, default=0.0)
        command.add_argument("--term-seconds", type=float, default=5.0)
        command.add_argument("--kill-seconds", type=float, default=2.0)
        command.add_argument("process_argv", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.process_argv[:1] == ["--"]:
        args.process_argv = args.process_argv[1:]
    if not args.process_argv:
        print("ERROR: exact process command is required after --", file=sys.stderr)
        return 2
    if args.uid < 0 or (args.launch_pgid is not None and args.launch_pgid <= 1):
        print("ERROR: unsafe UID or launch process group", file=sys.stderr)
        return 2
    try:
        if args.action == "record":
            if args.launch_pgid is None:
                raise OwnershipError("record requires --launch-pgid")
            _record(args)
        else:
            if args.launch_pgid is not None:
                raise OwnershipError("stop does not accept --launch-pgid")
            _stop(args)
    except (OwnershipError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
