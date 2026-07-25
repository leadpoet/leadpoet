#!/usr/bin/env python3
"""Read only the gateway OTel values without executing the env file."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict


ALLOWED_KEYS = {
    "GATEWAY_OTEL_ENDPOINT",
    "GATEWAY_OTEL_METRICS_ENDPOINT",
    "GATEWAY_OTEL_TOKEN",
}
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse the gateway's newline- or NUL-separated KEY=VALUE format."""
    if not path.is_file():
        return {}

    values: Dict[str, str] = {}
    data = path.read_bytes()
    for raw in re.split(rb"[\n\0]+", data):
        line = raw.strip()
        if not line or line.startswith(b"#") or b"=" not in line:
            continue
        key_raw, value_raw = line.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="ignore").strip()
        if key not in ALLOWED_KEYS or not _ENV_KEY_RE.fullmatch(key):
            continue
        value = value_raw.decode("utf-8", errors="ignore").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values.setdefault(key, value)
    return values


def resolve_value(path: Path, key: str) -> str:
    """Prefer a non-empty explicit environment override, then the env file."""
    ambient = os.environ.get(key, "")
    if ambient:
        return ambient
    return parse_env_file(path).get(key, "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--key", choices=sorted(ALLOWED_KEYS), required=True)
    args = parser.parse_args()
    print(resolve_value(args.env_file, args.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
