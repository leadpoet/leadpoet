#!/usr/bin/env python3.11
"""Persistent role-isolated process for real gateway enclave RPC handlers."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
from typing import Any

from sitecustomize import (
    _external_event,
    _gateway_enclave_socket_path,
    _handle_gateway_enclave_rpc,
    _sha256,
)


MAX_FRAME_BYTES = 128 * 1024 * 1024
ROLES = {
    "gateway_coordinator",
    "gateway_scoring",
}


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    output = bytearray()
    while len(output) < length:
        chunk = connection.recv(min(64 * 1024, length - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    if len(output) != length:
        raise ValueError("persistent gateway enclave frame is incomplete")
    return bytes(output)


def _handle(role: str, gateway_root: Path, body: bytes) -> dict[str, Any]:
    request = json.loads(body)
    if (
        not isinstance(request, dict)
        or set(request) != {"method", "params"}
        or not isinstance(request["params"], dict)
    ):
        raise ValueError("gateway enclave request fields differ")
    expected_merkle = gateway_root / "tee/merkle.py"
    if not expected_merkle.is_file():
        raise ValueError(
            "persistent gateway enclave candidate root became unavailable"
        )
    # A real role enclave has one immutable /app filesystem. Keep the local
    # process pointed at its candidate-backed equivalent even if candidate
    # behavior updates other measured environment values between RPCs.
    os.environ["GATEWAY_ROOT"] = str(gateway_root)
    method = str(request["method"])
    try:
        result = _handle_gateway_enclave_rpc(
            role,
            method,
            request["params"],
        )
        response = {"status": "success", "result": result}
        diagnostic = {
            "status": "ok",
            "error_type": "",
            "error_hash": "",
        }
    except Exception as exc:
        response = {"status": "error", "error": str(exc)}
        diagnostic = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error_hash": _sha256(str(exc)),
        }
    return {"response": response, "diagnostic": diagnostic}


def _prepare_candidate_role_root(role: str) -> Path:
    if role not in ROLES:
        raise ValueError("persistent gateway enclave role differs")
    if "gateway" in sys.modules:
        raise ValueError(
            "gateway modules loaded before candidate role isolation"
        )
    candidate = str(os.environ.get("REHEARSAL_CANDIDATE_SHA") or "")
    configured_source = os.environ.get("REHEARSAL_GATEWAY_CANDIDATE_ROOT")
    if not configured_source:
        raise ValueError("persistent gateway enclave source is unavailable")
    source = Path(configured_source).resolve()
    state_root = Path(
        os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state")
    )
    identity_source = (
        state_root / "gateway-enclave-build-identities" / f"{role}.json"
    )
    attested_runtime_source = state_root / "gateway-attested-runtime"
    release_input = json.loads(
        (state_root / "release-build-input.json").read_text(encoding="utf-8")
    )
    expected_role = release_input.get("gateway_roles", {}).get(role)
    if (
        release_input.get("commit_sha") != candidate
        or not isinstance(expected_role, dict)
        or not source.is_dir()
        or not identity_source.is_file()
        or (
            os.environ.get("REHEARSAL_SCOPE") == "exact"
            and not attested_runtime_source.is_dir()
        )
    ):
        raise ValueError("persistent gateway enclave fixture differs")

    role_parent = state_root / "gateway-enclave-runtimes" / role
    gateway_root = role_parent / "gateway"
    if role_parent.exists():
        raise ValueError("persistent gateway enclave role root already exists")
    role_parent.mkdir(parents=True)
    shutil.copytree(source, gateway_root)
    if attested_runtime_source.is_dir():
        shutil.rmtree(
            gateway_root / "_attested_runtime",
            ignore_errors=True,
        )
        shutil.copytree(
            attested_runtime_source,
            gateway_root / "_attested_runtime",
        )
    identity_target = (
        gateway_root
        / "_attested_runtime/gateway_enclave_build_identity.json"
    )
    identity_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(identity_source, identity_target)

    os.environ["GATEWAY_ROOT"] = str(gateway_root)
    if attested_runtime_source.is_dir():
        source_parent = source.parent
        controller_source = Path("/source").resolve()
        sys.path[:] = [
            item
            for item in sys.path
            if item
            and Path(item).resolve() not in {source_parent, controller_source}
        ]
        sys.path.insert(0, str(gateway_root / "_attested_runtime"))
    sys.path.insert(0, str(role_parent))
    importlib.invalidate_caches()
    from gateway.tee.build_identity import load_identity

    identity = load_identity(
        gateway_root=gateway_root,
        expected_role=role,
    )
    expected = {
        "commit_sha": candidate,
        "identity_hash": expected_role.get("build_identity_hash"),
        "dependency_lock_hash": expected_role.get(
            "dependency_lock_hash"
        ),
        "execution_manifest_hash": expected_role.get(
            "execution_manifest_hash"
        ),
        "topology_hash": expected_role.get("topology_hash"),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("persistent gateway enclave identity differs")
    module_path = Path(
        str(sys.modules["gateway.tee.build_identity"].__file__)
    ).resolve()
    if not module_path.is_relative_to(gateway_root):
        raise ValueError("persistent gateway enclave code root differs")
    return gateway_root


def main() -> int:
    role = str(os.environ.get("REHEARSAL_GATEWAY_ENCLAVE_ROLE") or "")
    if os.environ.get("REHEARSAL_COMPONENT") != "gateway":
        raise SystemExit("persistent gateway enclave requires gateway scope")
    gateway_root = _prepare_candidate_role_root(role)
    socket_path = _gateway_enclave_socket_path(role)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        socket_path.chmod(0o600)
        server.listen(64)
        socket_path.with_suffix(".ready").write_text(
            "ready\n",
            encoding="ascii",
        )
        while True:
            connection, _address = server.accept()
            with connection:
                try:
                    prefix = _recv_exact(connection, 4)
                    size = int.from_bytes(prefix, "big")
                    if size < 2 or size > MAX_FRAME_BYTES:
                        raise ValueError(
                            "persistent gateway enclave request size differs"
                        )
                    body = _recv_exact(connection, size)
                    outer = _handle(role, gateway_root, body)
                except Exception as exc:
                    outer = {
                        "response": {
                            "status": "error",
                            "error": str(exc),
                        },
                        "diagnostic": {
                            "status": "rejected",
                            "error_type": type(exc).__name__,
                            "error_hash": _sha256(str(exc)),
                        },
                    }
                encoded = json.dumps(
                    outer,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                try:
                    connection.sendall(
                        len(encoded).to_bytes(4, "big") + encoded
                    )
                except (
                    BrokenPipeError,
                    ConnectionAbortedError,
                    ConnectionResetError,
                ):
                    # A restarted worker can disappear after the enclave has
                    # accepted its request. Isolate that client lifecycle so
                    # the persistent role remains available to the replacement
                    # worker; all request execution failures still fail closed
                    # through the structured error response above.
                    continue
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        socket_path.with_suffix(".ready").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
