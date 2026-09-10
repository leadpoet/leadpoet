"""Fail-closed ownership check for the canonical Arena validator restart."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


def find_validator_process(
    source_root: Path, current_root: Path, service_pid: int, proc_root: Path = Path("/proc")
) -> Optional[Tuple[int, str]]:
    source_root = source_root.resolve()
    current_root = current_root.resolve(strict=False)
    exact: List[Tuple[int, str]] = []
    suspicious: List[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("current_root", type=Path)
    parser.add_argument("service_pid", type=int)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args()
    found = find_validator_process(
        args.source_root, args.current_root, args.service_pid, args.proc_root
    )
    print(*(found or ("", "")))
