#!/usr/bin/env python3.11
"""Install exact commit aliases for content-addressed scoring wheelhouses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def install_aliases(*, root: Path, aliases_path: Path) -> None:
    try:
        aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scoring wheelhouse alias map is unreadable") from exc
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError("scoring wheelhouse alias map must be a nonempty object")

    validated: list[tuple[Path, Path]] = []
    for commit, digest in sorted(aliases.items()):
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise ValueError("scoring wheelhouse alias commit is invalid")
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise ValueError("scoring wheelhouse alias digest is invalid")
        target = root / digest
        alias = root / commit
        if not target.is_dir():
            raise ValueError("scoring wheelhouse alias target is unavailable")
        if alias.exists() or alias.is_symlink():
            raise ValueError("scoring wheelhouse alias already exists")
        validated.append((alias, target))

    for alias, target in validated:
        alias.symlink_to(target, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    args = parser.parse_args()
    install_aliases(root=args.root, aliases_path=args.aliases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
