#!/usr/bin/env python3
"""Control the durable Lab Arena claim drain for a canonical restart."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import stat
import time
from typing import Any, Mapping
from urllib.parse import urlsplit


class GuardError(RuntimeError):
    pass


_SHA = re.compile(r"^[0-9a-f]{40}$")
_INVOCATION = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_MAX_ENVIRONMENT_BYTES = 1024 * 1024
_SAFE_FIELDS = frozenset(
    {
        "schema_version", "paused", "operator_paused", "guard_present",
        "guard_active", "guard_commitment", "owner_commitment",
        "guard_generation", "guard_expires_at", "candidate_commit",
        "restart_scope", "restart_phase", "drain", "captured_count",
        "accepted_receipt_count", "reported_terminal_receipt_count",
        "still_leased_count", "lost_or_mutated_count", "current_leased_count",
        "pending_retry_count", "snapshot_commitment", "outcome_commitment",
        "preserved", "mode",
    }
)


def _valid_header_value(value: str) -> bool:
    return bool(value) and value.isascii() and all(33 <= ord(char) <= 126 for char in value)


def _identity(candidate: str, invocation: str) -> tuple[str, str]:
    if not _SHA.fullmatch(candidate) or not _INVOCATION.fullmatch(invocation):
        raise GuardError("candidate or restart invocation identity is invalid")
    guard = hashlib.sha256(
        f"leadpoet.lab_arena.restart.guard.v1:{candidate}:{invocation}".encode()
    ).hexdigest()
    owner = hashlib.sha256(
        f"leadpoet.lab_arena.restart.owner.v1:{invocation}".encode()
    ).hexdigest()
    return f"lab_arena_restart_guard:{guard}", f"lab_arena_restart_owner:{owner}"


def _commitments(guard: str, owner: str) -> tuple[str, str]:
    return (
        "sha256:" + hashlib.sha256(guard.encode()).hexdigest(),
        "sha256:" + hashlib.sha256(owner.encode()).hexdigest(),
    )


def _read_scoped_environment(path: Path) -> dict[str, str]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GuardError("Arena restart environment file is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_ENVIRONMENT_BYTES:
            raise GuardError("Arena restart environment file is unavailable")
        raw = b""
        while len(raw) <= _MAX_ENVIRONMENT_BYTES:
            chunk = os.read(
                descriptor, min(65536, _MAX_ENVIRONMENT_BYTES + 1 - len(raw))
            )
            if not chunk:
                break
            raw += chunk
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev != metadata.st_dev
            or final_metadata.st_ino != metadata.st_ino
            or final_metadata.st_size != metadata.st_size
            or final_metadata.st_mtime_ns != metadata.st_mtime_ns
            or len(raw) != metadata.st_size
        ):
            raise GuardError("Arena restart environment file changed during read")
    finally:
        os.close(descriptor)
    if len(raw) > _MAX_ENVIRONMENT_BYTES:
        raise GuardError("Arena restart environment file is unavailable")
    try:
        text = raw.decode("utf-8")
        document = json.loads(text)
    except UnicodeDecodeError as exc:
        raise GuardError("Arena restart environment file is unavailable") from exc
    except json.JSONDecodeError:
        document = None
    if document is not None:
        if not isinstance(document, Mapping):
            raise GuardError("Arena restart environment file is unavailable")
        return {
            str(name): str(value)
            for name, value in document.items()
            if str(name).startswith("LAB_ARENA_")
        }
    values: dict[str, str] = {}
    for raw_line in text.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not name.startswith("LAB_ARENA_"):
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise GuardError("Arena restart environment file is unavailable")
        try:
            parts = shlex.split("VALUE=" + raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise GuardError("Arena restart environment file is unavailable") from exc
        if len(parts) != 1 or not parts[0].startswith("VALUE="):
            raise GuardError("Arena restart environment file is unavailable")
        value = parts[0].split("=", 1)[1]
        if name in values and values[name] != value:
            raise GuardError("Arena restart environment file is unavailable")
        values[name] = value
    return values


def _request(function: str, payload: Mapping[str, Any]) -> Any:
    origin = os.environ.get("LAB_ARENA_SUPABASE_URL", "").strip().rstrip("/")
    anon_key = os.environ.get("LAB_ARENA_SUPABASE_ANON_KEY", "").strip()
    service_key = os.environ.get("LAB_ARENA_SERVICE_KEY", "").strip()
    service_jwt = os.environ.get("LAB_ARENA_SERVICE_JWT", "").strip()
    parsed = urlsplit(origin)
    try:
        port = parsed.port
    except ValueError as exc:
        raise GuardError("Arena restart database authority is unavailable") from exc
    if (
        parsed.scheme != "https" or not parsed.hostname
        or port not in (None, 443) or parsed.path not in ("", "/")
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or not _valid_header_value(anon_key)
        or not (service_key or service_jwt)
        or (service_key and not _valid_header_value(service_key))
        or (not service_key and not _valid_header_value(service_jwt))
        or (service_key and not service_key.startswith("sb_secret_"))
        or (not service_key and service_jwt.count(".") != 2)
    ):
        raise GuardError("Arena restart database authority is unavailable")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    connection = http.client.HTTPSConnection(parsed.hostname, port or 443, timeout=15)
    try:
        headers = {
            "Accept": "application/json", "Content-Type": "application/json",
            "apikey": service_key or anon_key,
            "Content-Length": str(len(body)), "Connection": "close",
        }
        if service_jwt and not service_key:
            headers["Authorization"] = "Bearer " + service_jwt
        try:
            connection.request(
                "POST", f"/rest/v1/rpc/{function}", body=body,
                headers=headers,
            )
        except ValueError as exc:
            raise GuardError("Arena restart guard request is invalid") from exc
        response = connection.getresponse()
        data = response.read(256 * 1024 + 1)
    finally:
        connection.close()
    if len(data) > 256 * 1024:
        raise GuardError("Arena restart guard response is too large")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError("Arena restart guard response is invalid") from exc
    if response.status != 200:
        raise GuardError("Arena restart guard authority rejected the request")
    if not isinstance(value, Mapping):
        raise GuardError("Arena restart guard response is invalid")
    return dict(value)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _safe(item) for key, item in value.items() if key in _SAFE_FIELDS}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    return value


def _require_generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GuardError("Arena restart guard generation is invalid")
    return value


def _require_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "leadpoet.lab_arena.restart_guard_state.v1":
        raise GuardError("Arena restart guard state version is invalid")
    _require_generation(value.get("guard_generation"))
    for field in ("paused", "operator_paused", "guard_present", "guard_active"):
        if not isinstance(value.get(field), bool):
            raise GuardError("Arena restart guard state is invalid")
    drain = value.get("drain")
    if not isinstance(drain, Mapping):
        raise GuardError("Arena restart drain state is invalid")
    _require_drain(drain, "leadpoet.lab_arena.restart_drain_state.v1")
    return dict(value)


def _require_drain(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    if value.get("schema_version") != schema:
        raise GuardError("Arena restart drain state version is invalid")
    for field in (
        "captured_count", "accepted_receipt_count",
        "reported_terminal_receipt_count", "still_leased_count",
        "lost_or_mutated_count", "current_leased_count", "pending_retry_count",
    ):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise GuardError("Arena restart drain counts are invalid")
    if not isinstance(value.get("preserved"), bool):
        raise GuardError("Arena restart drain result is invalid")
    return dict(value)


def _base(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    guard, owner = _identity(args.candidate, args.invocation)
    generation = args.generation
    if generation == 0:
        state = _require_state(_request("lab_arena_restart_guard_state_v1", {}))
        _require_owned_state(args, guard, owner, state)
        generation = _require_generation(state.get("guard_generation"))
    return guard, owner, {
        "p_guard_id": guard,
        "p_owner_id": owner,
        "p_guard_generation": generation,
    }


def _require_owned_state(
    args: argparse.Namespace, guard: str, owner: str, state: Mapping[str, Any],
) -> None:
    expected_guard, expected_owner = _commitments(guard, owner)
    if (
        state.get("guard_present") is not True
        or state.get("guard_commitment") != expected_guard
        or state.get("owner_commitment") != expected_owner
        or state.get("candidate_commit") != args.candidate
    ):
        raise GuardError("Arena restart guard is not owned by this invocation")


def _owned(args: argparse.Namespace) -> dict[str, Any]:
    guard, owner = _identity(args.candidate, args.invocation)
    state = _require_state(_request("lab_arena_restart_guard_state_v1", {}))
    _require_owned_state(args, guard, owner, state)
    return state


def _discover(args: argparse.Namespace) -> dict[str, Any]:
    state = _require_state(_request("lab_arena_restart_guard_state_v1", {}))
    if state.get("guard_present") is not True:
        return state
    current_candidate = state.get("candidate_commit")
    if not isinstance(current_candidate, str) or not _SHA.fullmatch(current_candidate):
        raise GuardError("Arena restart guard candidate is invalid")
    guard, owner = _identity(current_candidate, args.invocation)
    _require_owned_state(
        argparse.Namespace(candidate=current_candidate), guard, owner, state
    )
    return state


def _abort_if_owned_draining(
    args: argparse.Namespace, guard: str, owner: str, generation: int | None,
) -> None:
    """Clear only this exact guard before a destructive phase starts."""
    try:
        state = _require_state(_request("lab_arena_restart_guard_state_v1", {}))
        expected_guard, expected_owner = _commitments(guard, owner)
        if (
            state.get("guard_commitment") != expected_guard
            or state.get("owner_commitment") != expected_owner
            or state.get("candidate_commit") != args.candidate
            or state.get("restart_phase") != "draining"
        ):
            return
        observed_generation = _require_generation(state.get("guard_generation"))
        if generation is not None and observed_generation != generation:
            return
        _request("lab_arena_abort_restart_guard_v1", {
            "p_guard_id": guard,
            "p_owner_id": owner,
            "p_guard_generation": observed_generation,
            "p_actor_ref": f"canonical-active-release:{args.candidate}",
        })
    except (GuardError, OSError, http.client.HTTPException):
        # The caller still fails closed. A later exact-owner canonical retry
        # can renew and either abort or continue the retained guard.
        return


def _acquire(args: argparse.Namespace) -> dict[str, Any]:
    guard, owner = _identity(args.candidate, args.invocation)
    state = _require_state(_request("lab_arena_restart_guard_state_v1", {}))
    generation = _require_generation(state["guard_generation"])
    if state.get("guard_present") is True:
        current_candidate = state.get("candidate_commit")
        if not isinstance(current_candidate, str) or not _SHA.fullmatch(current_candidate):
            raise GuardError("Arena restart guard candidate is invalid")
        current_guard, current_owner = _identity(current_candidate, args.invocation)
        _require_owned_state(
            argparse.Namespace(candidate=current_candidate),
            current_guard,
            current_owner,
            state,
        )
        if state.get("restart_scope") != args.scope:
            raise GuardError("Arena restart guard scope differs")
        if current_candidate != args.candidate:
            if state.get("restart_phase") == "draining":
                _request("lab_arena_abort_restart_guard_v1", {
                    "p_guard_id": current_guard,
                    "p_owner_id": current_owner,
                    "p_guard_generation": generation,
                    "p_actor_ref": f"canonical-active-release:{args.candidate}",
                })
                state = _require_state(
                    _request("lab_arena_restart_guard_state_v1", {})
                )
                if state.get("guard_present") is True:
                    raise GuardError("Arena restart guard abort did not clear ownership")
                generation = _require_generation(state["guard_generation"])
            else:
                state = _require_state(_request(
                    "lab_arena_retarget_restart_guard_v1",
                    {
                        "p_current_guard_id": current_guard,
                        "p_owner_id": current_owner,
                        "p_expected_generation": generation,
                        "p_new_guard_id": guard,
                        "p_new_candidate_commit": args.candidate,
                        "p_restart_scope": args.scope,
                        "p_lease_seconds": args.lease_seconds,
                        "p_actor_ref": f"canonical-active-release:{args.candidate}",
                    },
                ))
                _require_owned_state(args, guard, owner, state)
                return state
    return _require_state(_request("lab_arena_acquire_restart_guard_v1", {
        "p_guard_id": guard, "p_owner_id": owner,
        "p_expected_generation": generation,
        "p_lease_seconds": args.lease_seconds,
        "p_candidate_commit": args.candidate,
        "p_restart_scope": args.scope,
        "p_actor_ref": f"canonical-active-release:{args.candidate}",
    }))


def _drain(args: argparse.Namespace) -> dict[str, Any]:
    guard, owner = _identity(args.candidate, args.invocation)
    generation: int | None = None
    try:
        state = _acquire(args)
        generation = _require_generation(state["guard_generation"])
        deadline = time.monotonic() + args.timeout_seconds
        last_renewal = 0.0
        while True:
            now = time.monotonic()
            if now - last_renewal >= min(30.0, args.lease_seconds / 3):
                state = _acquire(args)
                if _require_generation(state["guard_generation"]) != generation:
                    raise GuardError("Arena restart guard generation changed")
                last_renewal = now
            drain = _require_drain(_request("lab_arena_restart_quiescence_v1", {
                "p_guard_id": guard, "p_owner_id": owner,
                "p_guard_generation": generation,
            }), "leadpoet.lab_arena.restart_quiescence.v1")
            if (
                drain.get("guard_active") is not True
                or drain.get("guard_generation") != generation
                or drain.get("restart_scope") != args.scope
            ):
                raise GuardError("Arena restart guard changed during drain")
            legal_phases = {
                "gateway": {"draining", "gateway_destructive", "gateway_ready"},
                "validator": {"draining", "validator_destructive", "validator_ready"},
                "all": {
                    "draining", "gateway_destructive", "gateway_ready",
                    "validator_destructive", "validator_ready",
                },
            }
            if drain.get("restart_phase") not in legal_phases[args.scope]:
                raise GuardError("Arena restart guard phase is invalid for scope")
            if drain["lost_or_mutated_count"]:
                raise GuardError("Arena lease ended without a durable completion receipt")
            if drain["preserved"] is True:
                return drain
            if now >= deadline:
                raise GuardError("Arena leases did not drain before the restart deadline")
            time.sleep(min(args.poll_seconds, max(0.0, deadline - now)))
    except (GuardError, OSError, http.client.HTTPException):
        _abort_if_owned_draining(args, guard, owner, generation)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=(
            "mode", "state", "discover", "owned", "drain", "authorize", "ready", "abort", "release"
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--scope", choices=("gateway", "validator", "all"))
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--phase")
    parser.add_argument("--lease-seconds", type=int, default=14400)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--environment-file", type=Path)
    args = parser.parse_args()
    try:
        if args.environment_file is not None:
            scoped = _read_scoped_environment(args.environment_file)
            for name, value in scoped.items():
                os.environ.setdefault(name, value)
        if args.action == "mode":
            mode = os.environ.get("LAB_ARENA_MODE", "off").strip().lower()
            if mode not in {"off", "live", "shadow"}:
                raise GuardError("Arena restart mode is invalid")
            value = {
                "schema_version": "leadpoet.lab_arena.restart_mode.v1",
                "mode": mode,
            }
        elif args.action == "state":
            value = _require_state(_request("lab_arena_restart_guard_state_v1", {}))
        elif args.action == "discover":
            value = _discover(args)
        elif args.action == "owned":
            value = _owned(args)
        elif args.action == "drain":
            if args.scope is None:
                raise GuardError("drain requires restart scope")
            value = _drain(args)
        else:
            guard, owner, payload = _base(args)
            del guard, owner
            payload["p_actor_ref"] = f"canonical-active-release:{args.candidate}"
            if args.action in ("authorize", "ready"):
                if not args.phase:
                    raise GuardError(f"{args.action} requires phase")
                payload.pop("p_actor_ref")
                payload["p_phase"] = args.phase
            function = {
                "authorize": "lab_arena_authorize_restart_phase_v1",
                "ready": "lab_arena_mark_restart_ready_v1",
                "abort": "lab_arena_abort_restart_guard_v1",
                "release": "lab_arena_release_restart_guard_v1",
            }[args.action]
            value = _require_state(_request(function, payload))
        print(json.dumps(_safe(value), sort_keys=True, separators=(",", ":")))
        return 0
    except (GuardError, OSError, http.client.HTTPException) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
