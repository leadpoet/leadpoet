#!/usr/bin/env python3
"""Acknowledge an exact migration 203 after the operator verifies its application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import tempfile
from urllib.request import Request, urlopen


_REQUIRED_ENVIRONMENT_NAMES = frozenset(
    {"SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"}
)


def _load_regular(path: Path) -> dict:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.geteuid()
        or not 0 < info.st_size <= 4096
    ):
        raise ValueError("migration barrier is unsafe")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("migration barrier is invalid")
    return value


def _load_environment(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8")
    duplicate_required: set[str] = set()

    def load_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result and name in _REQUIRED_ENVIRONMENT_NAMES:
                duplicate_required.add(name)
            result[name] = value
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=load_object)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        if duplicate_required:
            raise ValueError("gateway environment has duplicate required names")
        return {
            name: str(parsed[name])
            for name in _REQUIRED_ENVIRONMENT_NAMES
            if name in parsed
        }
    result: dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match is None:
            if any(
                re.match(rf"{re.escape(name)}(?:\s|$)", line)
                for name in _REQUIRED_ENVIRONMENT_NAMES
            ):
                raise ValueError("required gateway environment is malformed")
            continue
        if match.group(1) not in _REQUIRED_ENVIRONMENT_NAMES:
            continue
        parts = shlex.split(line, posix=True)
        if len(parts) != 1 or "=" not in parts[0]:
            raise ValueError("required gateway environment is malformed")
        name, value = parts[0].split("=", 1)
        if name in result:
            raise ValueError("gateway environment has duplicate required names")
        result[name] = value
    return result


def _verify_capability(environment: dict[str, str], *, opener=urlopen) -> None:
    url = environment.get("SUPABASE_URL", "").rstrip("/")
    key = environment.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise ValueError("Supabase capability credentials are unavailable")
    request = Request(
        f"{url}/rest/v1/rpc/lab_arena_incentive_retirement_schema_v1",
        data=b"{}",
        headers={"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=10) as response:
            value = json.loads(response.read(4097).decode("utf-8"))
            if int(response.getcode()) != 200:
                raise ValueError("migration 203 capability is unavailable")
    except Exception as exc:
        raise ValueError("migration 203 capability is unavailable") from exc
    expected = {"schema_version": "leadpoet.lab_arena.incentive_retirement_schema.v1", "version": 203}
    if value != expected:
        raise ValueError("migration 203 capability differs")


def complete(*, barrier: Path, completion: Path, sql: Path, commit: str,
             environment: dict[str, str], opener=urlopen) -> dict:
    request = _load_regular(barrier)
    sql_hash = hashlib.sha256(sql.read_bytes()).hexdigest()
    expected = {
        "schema_version": "leadpoet.gateway.migration_203_barrier.v1",
        "candidate_commit": commit,
        "sql_sha256": sql_hash,
        "restart_invocation_id": request.get("restart_invocation_id"),
        "old_producers_stopped": True,
    }
    if request != expected:
        raise ValueError("migration barrier differs from the exact candidate and SQL")
    _verify_capability(environment, opener=opener)
    document = {
        "schema_version": "leadpoet.gateway.migration_203_complete.v1",
        "candidate_commit": commit,
        "sql_sha256": sql_hash,
        "restart_invocation_id": request["restart_invocation_id"],
        "migration_203_verified": True,
    }
    descriptor, name = tempfile.mkstemp(prefix=".migration-203-complete.", dir=completion.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o600)
        os.replace(name, completion)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--sql", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    environment = _load_environment(args.env_file)
    print(json.dumps(complete(barrier=args.barrier, completion=args.completion,
                              sql=args.sql, commit=args.commit,
                              environment=environment), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
