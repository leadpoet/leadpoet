#!/usr/bin/env python3
"""Launch the normal Arena validator from a protected, non-executable env file."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_environment(path: Path) -> tuple[dict[str, str], str]:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError("Arena validator env file must be a private regular file")
        if metadata.st_uid not in (0, os.geteuid()):
            raise ValueError("Arena validator env file has an unexpected owner")
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Arena validator env file is too large")
    values = {}
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError("Arena validator env file has an invalid assignment")
        if not name.startswith("LAB_ARENA_"):
            raise ValueError("Arena validator env file contains an unrelated setting")
        try:
            parts = shlex.split("VALUE=" + value, comments=True, posix=True)
        except ValueError:
            raise ValueError("Arena validator env file has an invalid value") from None
        if len(parts) != 1 or not parts[0].startswith("VALUE=") or name in values:
            raise ValueError("Arena validator env file has a duplicate or invalid value")
        values[name] = parts[0][6:]
    return values, raw


def load_environment(path: Path) -> None:
    """Parse assignments as data; never source shell commands or print values."""

    values, _raw = _read_environment(path)
    os.environ.update(values)


def _is_proxy_setting(name: str) -> bool:
    from lab_arena.proxy_workers import PROXY_ENVIRONMENT_PREFIXES

    return any(name.startswith(prefix + "_") for prefix in PROXY_ENVIRONMENT_PREFIXES)


def _write_private_snapshot(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            descriptor = -1
            destination.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_environment_snapshot(source: Path, destination: Path) -> None:
    """Materialize a root-ready service env without exposing imported secrets.

    Existing configurations without a secondary proxy file are copied exactly.
    When a secondary file is configured, only its recognized proxy entries are
    imported and all proxy aliases are rewritten to canonical compact indices.
    The source is read as data and the process environment is not modified.
    """

    values, raw = _read_environment(source)
    secondary_path = values.get("LAB_ARENA_PROXY_ENV_FILE", "").strip()
    if not secondary_path:
        _write_private_snapshot(destination, raw)
        return

    from lab_arena.proxy_workers import proxy_workers_from_environment
    from lab_arena.validator_proxy_environment import validator_proxy_environment

    merged = validator_proxy_environment(values)
    inventory = proxy_workers_from_environment(merged)
    snapshot_values = {
        name: value
        for name, value in values.items()
        if name != "LAB_ARENA_PROXY_ENV_FILE" and not _is_proxy_setting(name)
    }
    for worker in inventory.workers:
        snapshot_values[f"LAB_ARENA_WEBSHARE_PROXY_{worker.slot_index}"] = (
            worker.proxy_url
        )
    content = "".join(
        f"{name}={shlex.quote(value)}\n" for name, value in snapshot_values.items()
    )
    _write_private_snapshot(destination, content)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--environment-file", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    load_environment(args.environment_file)
    from lab_arena.validator import main as validator_main

    return validator_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
