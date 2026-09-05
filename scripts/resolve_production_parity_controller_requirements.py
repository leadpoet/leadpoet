#!/usr/bin/env python3
"""Resolve the parity controller's dependencies from candidate requirements."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence


REQUIRED_PACKAGES = (
    "arweave-python-client",
    "bittensor",
    "boto3",
    "cryptography",
    "fastapi",
    "httpx",
    "supabase",
    "uvicorn",
)
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")


class ControllerRequirementsError(RuntimeError):
    pass


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def resolve_controller_requirements(requirements_path: Path) -> tuple[str, ...]:
    try:
        raw_lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ControllerRequirementsError(
            "candidate requirements are unreadable"
        ) from exc

    selected: dict[str, list[str]] = {}
    required = {_normalized_name(name) for name in REQUIRED_PACKAGES}
    for raw_line in raw_lines:
        line = re.split(r"\s+#", raw_line, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = NAME_RE.match(line)
        if match is None:
            continue
        name = _normalized_name(match.group(1))
        if name not in required:
            continue
        previous = selected.setdefault(name, [])
        if line in previous:
            continue
        if previous and (
            ";" not in line or any(";" not in item for item in previous)
        ):
            raise ControllerRequirementsError(
                f"candidate requirements define {name} more than once"
            )
        previous.append(line)

    missing = sorted(required.difference(selected))
    if missing:
        raise ControllerRequirementsError(
            "candidate requirements omit parity controller dependencies: "
            + ", ".join(missing)
        )
    return tuple(
        line
        for name in REQUIRED_PACKAGES
        for line in selected[_normalized_name(name)]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        resolved = resolve_controller_requirements(args.requirements)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(resolved) + "\n", encoding="ascii")
    except (OSError, ControllerRequirementsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
