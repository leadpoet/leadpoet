"""Fail-closed ownership check for the canonical Arena validator restart."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


def _stat(entry: Path) -> List[str]:
    return (entry / "stat").read_text().split()


def _is_descendant(proc_root: Path, pid: int, ancestor: int) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            pid = int(_stat(proc_root / str(pid))[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def find_validator_process(
    source_root: Path, current_root: Path, service_pid: int, proc_root: Path = Path("/proc"),
    ignored_tree_root: int = 0,
) -> Optional[Tuple[int, str]]:
    source_root = source_root.resolve()
    current_root = current_root.resolve(strict=False)
    exact: List[Tuple[int, str]] = []
    suspicious: List[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if ignored_tree_root and _is_descendant(proc_root, pid, ignored_tree_root):
            continue
        try:
            args = [
                value.decode("utf-8", "surrogateescape")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            hits = [
                value
                for value in args
                if value.endswith("neurons/validator.py")
                or value.endswith("scripts/run_arena_validator.py")
            ]
            if not hits:
                continue
            resolved = [
                (cwd / Path(value)).resolve()
                if not Path(value).is_absolute()
                else Path(value).resolve()
                for value in hits
            ]
            source_owned = all(source_root in (path, *path.parents) for path in resolved)
            supervised = pid == service_pid and all(
                current_root in (path, *path.parents) for path in resolved
            )
            if source_owned or supervised:
                exact.append((pid, (entry / "stat").read_text().split()[21]))
            else:
                suspicious.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    if suspicious or len(exact) > 1:
        raise RuntimeError("validator process ownership is ambiguous")
    return exact[0] if exact else None


def find_owned_auxiliary(
    source_root: Path, kind: str, proc_root: Path = Path("/proc")
) -> Optional[Tuple[int, str]]:
    suffix = {
        "runner": "scripts/run_lab_arena_runner.py",
        "relay": "validator_tee.host.chain_relay_v2",
    }[kind]
    source_root = source_root.resolve()
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [
                value.decode("utf-8", "surrogateescape")
                for value in (entry / "cmdline").read_bytes().split(b"\0") if value
            ]
            if kind == "runner":
                selected = any(value.endswith(suffix) for value in args)
            else:
                selected = any(value == suffix for value in args)
            if not selected:
                continue
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            if cwd != source_root:
                raise RuntimeError("%s process ownership is ambiguous" % kind)
            fields = _stat(entry)
            matches.append((int(entry.name), int(fields[4])))
        except (OSError, ValueError, IndexError):
            continue
    if not matches:
        return None
    groups = {group for _, group in matches}
    if len(groups) != 1:
        raise RuntimeError("%s process ownership is ambiguous" % kind)
    owner = groups.pop() if kind == "runner" else matches[0][0]
    if kind == "relay" and len(matches) != 1:
        raise RuntimeError("relay process ownership is ambiguous")
    try:
        start = _stat(proc_root / str(owner))[21]
    except (OSError, IndexError) as exc:
        raise RuntimeError("%s process owner is unavailable" % kind) from exc
    return owner, start


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("current_root", type=Path)
    parser.add_argument("service_pid", type=int)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--ignore-tree-root", type=int, default=0)
    parser.add_argument("--kind", choices=("validator", "runner", "relay"), default="validator")
    args = parser.parse_args()
    found = (find_validator_process(
        args.source_root, args.current_root, args.service_pid, args.proc_root,
        args.ignore_tree_root,
    ) if args.kind == "validator" else find_owned_auxiliary(
        args.source_root, args.kind, args.proc_root
    ))
    print(*(found or ("", "")))
