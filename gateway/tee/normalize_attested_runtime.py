"""Normalize copied top-level runtime packages before the gateway EIF build."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence


DIRECTORY_MODE = 0o755
FILE_MODE = 0o644
EXECUTABLE_SUFFIXES = frozenset({".sh"})
NORMALIZED_MTIME_NS = 0


def normalized_file_mode(path: Path) -> int:
    return 0o755 if path.suffix.lower() in EXECUTABLE_SUFFIXES else FILE_MODE


def normalize_runtime_tree(root: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("attested runtime root is missing: %s" % root)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("attested runtime must not contain symlinks")

    for path in sorted(root.rglob("*"), key=lambda item: (len(item.parts), item.as_posix()), reverse=True):
        if path.is_file():
            os.chmod(str(path), normalized_file_mode(path))
            os.utime(str(path), ns=(NORMALIZED_MTIME_NS, NORMALIZED_MTIME_NS))
        elif path.is_dir():
            os.chmod(str(path), DIRECTORY_MODE)
            os.utime(str(path), ns=(NORMALIZED_MTIME_NS, NORMALIZED_MTIME_NS))
    os.chmod(str(root), DIRECTORY_MODE)
    os.utime(str(root), ns=(NORMALIZED_MTIME_NS, NORMALIZED_MTIME_NS))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    normalize_runtime_tree(args.root)
    print("attested_runtime_metadata_normalized=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
