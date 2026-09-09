#!/usr/bin/env python3
"""Own and stop one detached Linux process group without broad name matching."""

from __future__ import annotations

import argparse
import hashlib
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
MAX_VERIFY_STATE_BYTES = 64 * 1024


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


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_verify_file(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise OwnershipError(f"process state is not a regular file: {path}")
    if metadata.st_uid != os.getuid():
        raise OwnershipError("process state is not owned by the current user")
    if metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise OwnershipError("process state is not owner-only")
    if metadata.st_size > MAX_VERIFY_STATE_BYTES:
        raise OwnershipError("process state exceeds the bounded read limit")


def _parse_verify_state(
    raw_state: bytes, path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    try:
        state = json.loads(raw_state.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
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
        type(state.get(key)) is not value_type  # noqa: E721 - reject bool identities
        for key, value_type in required.items()
    ):
        raise OwnershipError(f"process state has an invalid schema: {path}")
    if any(type(item) is not str for item in state["argv"]):
        raise OwnershipError(f"process state has an invalid argv: {path}")
    if any(state[key] != value for key, value in expected.items()):
        raise OwnershipError(
            f"process state identity differs from the requested instance: {path}"
        )
    if state["pid"] <= 1 or state["pgid"] <= 1 or state["start_time_ticks"] <= 0:
        raise OwnershipError(f"process state contains an unsafe process identity: {path}")
    return state


def _read_verify_state(
    path: Path, expected: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    try:
        path_before = os.lstat(path)
    except FileNotFoundError:
        raise OwnershipError("process state is unavailable") from None
    _require_verify_file(path_before, path)

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as exc:
        raise OwnershipError(f"process state could not be opened safely: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        _require_verify_file(opened, path)
        if _file_identity(opened) != _file_identity(path_before):
            raise OwnershipError("process state changed while it was opened")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, MAX_VERIFY_STATE_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_VERIFY_STATE_BYTES:
                raise OwnershipError("process state exceeds the bounded read limit")
        raw_state = b"".join(chunks)

        after = os.fstat(descriptor)
        if _file_identity(after) != _file_identity(opened):
            raise OwnershipError("process state changed while it was read")
        try:
            path_after = os.lstat(path)
        except FileNotFoundError:
            raise OwnershipError("process state was replaced while it was read") from None
        if _file_identity(path_after) != _file_identity(opened):
            raise OwnershipError("process state was replaced while it was read")
    except OSError as exc:
        raise OwnershipError(f"process state could not be read safely: {path}") from exc
    finally:
        os.close(descriptor)
    return raw_state, _parse_verify_state(raw_state, path, expected)


def _verify(args: argparse.Namespace) -> None:
    """Verify one recorded process group without changing local or process state."""
    expected = _expected_identity(argv=args.process_argv, cwd=args.cwd, uid=args.uid)
    raw_state, state = _read_verify_state(args.state_file, expected)
    matches = _matching_processes(
        argv=args.process_argv, cwd=args.cwd, uid=args.uid
    )
    if len(matches) != 1:
        raise OwnershipError(
            f"expected exactly one live detached process, found {len(matches)}"
        )
    process = matches[0]
    if process["state"] == "Z" or not _same_process(process, state):
        raise OwnershipError("saved process identity changed or PID was reused")
    members = _active_group_members(state["pgid"])
    leader = next((member for member in members if member["pid"] == state["pid"]), None)
    if leader is None or not _same_process(leader, state):
        raise OwnershipError("saved process group is not live")

    result = {
        "action": "verify",
        "ok": True,
        "state_version": state["version"],
        "pid": state["pid"],
        "pgid": state["pgid"],
        "start_time_ticks": state["start_time_ticks"],
        "uid": state["uid"],
        "state_sha256": hashlib.sha256(raw_state).hexdigest(),
        "cwd_sha256": hashlib.sha256(state["cwd"].encode("utf-8")).hexdigest(),
        "argv_sha256": _sha256_json(state["argv"]),
        "exact_match_count": len(matches),
        "live_group": True,
        "live_group_member_count": len(members),
    }
    print(json.dumps(result, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record, verify, or stop one exact detached process group"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("record", "stop", "verify"):
        command = actions.add_parser(
            action,
            help=(
                "read-only ownership and liveness check"
                if action == "verify"
                else None
            ),
        )
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
        elif args.action == "stop":
            if args.launch_pgid is not None:
                raise OwnershipError("stop does not accept --launch-pgid")
            _stop(args)
        else:
            if args.launch_pgid is not None:
                raise OwnershipError("verify does not accept --launch-pgid")
            _verify(args)
    except (OwnershipError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
