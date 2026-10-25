#!/usr/bin/env python3
"""Render one hash-verified private Arena export as readable JSON and Markdown."""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import contracts, training_thinking


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("example_not_regular")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    created = False
    written = []
    names = ("raw-export.json", "readable-trace.json", "readable-trace.md")
    try:
        raw = _read_regular(args.example)
        example = json.loads(raw)
        report = training_thinking.readable_trace(example)
        markdown = training_thinking.readable_trace_markdown(report, raw_export_name=names[0])
        rendered = (raw, (contracts.canonical_json(report) + "\n").encode("utf-8"),
                    markdown.encode("utf-8"))
        args.output.mkdir(mode=0o700)
        created = True
        for name, data in zip(names, rendered):
            _write_private(args.output / name, data)
            written.append(name)
    except Exception:
        if created:
            for name in written:
                (args.output / name).unlink(missing_ok=True)
            try:
                args.output.rmdir()
            except OSError:
                pass
        print("Arena readable trace failed: invalid_example_or_output", file=sys.stderr)
        return 1
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
