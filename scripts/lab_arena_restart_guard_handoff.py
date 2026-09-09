#!/usr/bin/env python3
"""Create and verify the private validator restart-guard handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

from scripts.lab_arena_restart_claim_guard import (
    GuardError,
    _commitments,
    _identity,
    _require_state,
)


_REQUEST_SCHEMA = "leadpoet.lab_arena.restart_guard_request.v1"
_PERMIT_SCHEMA = "leadpoet.lab_arena.restart_guard_permit.v1"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_INVOCATION = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_NONCE = re.compile(r"^[0-9a-f]{64}$")
_MAX_DOCUMENT_BYTES = 16 * 1024


def _validate_path(path: Path) -> None:
    if (
        not path.is_absolute()
        or path.parts[:2] != ("/", "tmp")
        or ".." in path.parts
        or not re.fullmatch(r"leadpoet-[A-Za-z0-9._-]+\.json", path.name)
    ):
        raise GuardError("Arena guard handoff path is invalid")


def _identity_fields(args: argparse.Namespace) -> dict[str, str]:
    if not _SHA.fullmatch(args.candidate):
        raise GuardError("Arena guard handoff candidate is invalid")
    if not _SHA.fullmatch(args.authority_commit):
        raise GuardError("Arena guard handoff authority is invalid")
    if not _INVOCATION.fullmatch(args.invocation):
        raise GuardError("Arena guard handoff invocation is invalid")
    if args.scope not in {"validator", "all"}:
        raise GuardError("Arena guard handoff scope is invalid")
    if not _NONCE.fullmatch(args.nonce):
        raise GuardError("Arena guard handoff nonce is invalid")
    return {
        "candidate_commit": args.candidate,
        "restart_invocation_id": args.invocation,
        "restart_scope": args.scope,
        "handoff_nonce": args.nonce,
        "authority_commit": args.authority_commit,
    }


def _read_document(path: Path) -> dict[str, Any]:
    _validate_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GuardError("Arena guard handoff file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_size <= 0
            or before.st_size > _MAX_DOCUMENT_BYTES
        ):
            raise GuardError("Arena guard handoff file is unavailable")
        payload = os.read(descriptor, _MAX_DOCUMENT_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or len(payload) > _MAX_DOCUMENT_BYTES
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise GuardError("Arena guard handoff file changed during read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError("Arena guard handoff file is invalid") from exc
    if not isinstance(value, Mapping):
        raise GuardError("Arena guard handoff file is invalid")
    return dict(value)


def _write_document(path: Path, value: Mapping[str, Any]) -> None:
    _validate_path(path)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if len(payload) > _MAX_DOCUMENT_BYTES:
        raise GuardError("Arena guard handoff document is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {"schema_version": _REQUEST_SCHEMA, **_identity_fields(args)}


def _validate_request(args: argparse.Namespace) -> dict[str, Any]:
    expected = _request(args)
    value = _read_document(args.path)
    if value != expected:
        raise GuardError("Arena guard request identity differs")
    return value


def _parse_expiry(value: Any) -> datetime:
    if not isinstance(value, str):
        raise GuardError("Arena restart guard expiry is invalid")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise GuardError("Arena restart guard expiry is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GuardError("Arena restart guard expiry is invalid")
    if parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise GuardError("Arena restart guard permit is expired")
    return parsed


def _permit(args: argparse.Namespace, state: Mapping[str, Any]) -> dict[str, Any]:
    identity = _identity_fields(args)
    normalized = _require_state(state)
    guard, owner = _identity(args.candidate, args.invocation)
    expected_guard, expected_owner = _commitments(guard, owner)
    generation = normalized.get("guard_generation")
    drain = normalized.get("drain")
    if (
        normalized.get("guard_present") is not True
        or normalized.get("guard_active") is not True
        or normalized.get("paused") is not True
        or normalized.get("candidate_commit") != args.candidate
        or normalized.get("restart_scope") != args.scope
        or normalized.get("restart_phase") != "validator_destructive"
        or normalized.get("guard_commitment") != expected_guard
        or normalized.get("owner_commitment") != expected_owner
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
        or (args.expected_generation and generation != args.expected_generation)
        or not isinstance(drain, Mapping)
        or drain.get("preserved") is not True
        or drain.get("still_leased_count") != 0
        or drain.get("lost_or_mutated_count") != 0
        or drain.get("current_leased_count") != 0
        or drain.get("captured_count")
        != drain.get("accepted_receipt_count", -1)
        + drain.get("reported_terminal_receipt_count", -1)
    ):
        raise GuardError("Arena restart guard authorization is invalid")
    expiry = normalized.get("guard_expires_at")
    _parse_expiry(expiry)
    return {
        "schema_version": _PERMIT_SCHEMA,
        **identity,
        "restart_phase": "validator_destructive",
        "guard_generation": generation,
        "guard_expires_at": expiry,
        "guard_commitment": expected_guard,
        "owner_commitment": expected_owner,
    }


def _validate_permit(args: argparse.Namespace) -> dict[str, Any]:
    value = _read_document(args.path)
    identity = _identity_fields(args)
    guard, owner = _identity(args.candidate, args.invocation)
    expected_guard, expected_owner = _commitments(guard, owner)
    expected_keys = {
        "schema_version", *identity,
        "restart_phase", "guard_generation", "guard_expires_at",
        "guard_commitment", "owner_commitment",
    }
    generation = value.get("guard_generation")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != _PERMIT_SCHEMA
        or any(value.get(key) != expected for key, expected in identity.items())
        or value.get("restart_phase") != "validator_destructive"
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
        or value.get("guard_commitment") != expected_guard
        or value.get("owner_commitment") != expected_owner
    ):
        raise GuardError("Arena guard permit identity differs")
    _parse_expiry(value.get("guard_expires_at"))
    return value


def _read_stdin_state() -> dict[str, Any]:
    payload = os.sys.stdin.buffer.read(_MAX_DOCUMENT_BYTES + 1)
    if not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise GuardError("Arena restart guard authorization is invalid")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError("Arena restart guard authorization is invalid") from exc
    if not isinstance(value, Mapping):
        raise GuardError("Arena restart guard authorization is invalid")
    return dict(value)


def _safe_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_version": value.get("schema_version"),
        "candidate_commit": value.get("candidate_commit"),
        "restart_invocation_id": value.get("restart_invocation_id"),
        "restart_scope": value.get("restart_scope"),
        "authority_commit": value.get("authority_commit"),
    }
    nonce = value.get("handoff_nonce")
    if isinstance(nonce, str):
        import hashlib

        summary["handoff_nonce_commitment"] = "sha256:" + hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest()
    for field in (
        "restart_phase", "guard_generation", "guard_expires_at",
        "guard_commitment", "owner_commitment",
    ):
        if field in value:
            summary[field] = value[field]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write-request", "validate-request", "write-permit", "validate-permit"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--authority-commit", required=True)
    parser.add_argument("--expected-generation", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.action == "write-request":
            value = _request(args)
            _write_document(args.path, value)
        elif args.action == "validate-request":
            value = _validate_request(args)
        elif args.action == "write-permit":
            value = _permit(args, _read_stdin_state())
            _write_document(args.path, value)
        else:
            value = _validate_permit(args)
        print(json.dumps(_safe_summary(value), sort_keys=True, separators=(",", ":")))
        return 0
    except (GuardError, OSError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
