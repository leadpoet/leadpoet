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


def load_environment(path: Path) -> None:
    """Parse assignments as data; never source shell commands or print values."""

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
        if not (name.startswith("LAB_ARENA_") or name == "ENCLAVE_CID"):
            raise ValueError("Arena validator env file contains an unrelated setting")
        try:
            parts = shlex.split("VALUE=" + value, comments=True, posix=True)
        except ValueError:
            raise ValueError("Arena validator env file has an invalid value") from None
        if len(parts) != 1 or not parts[0].startswith("VALUE=") or name in values:
            raise ValueError("Arena validator env file has a duplicate or invalid value")
        values[name] = parts[0][6:]
    os.environ.update(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--environment-file", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    load_environment(args.environment_file)
    from lab_arena.validator import main as validator_main

    return validator_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
