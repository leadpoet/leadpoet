"""Opaque parent relay for mutually attested gateway-enclave TLS channels."""

from __future__ import annotations

import argparse
import json
import re
import select
import socket
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.utils.tee_client import AF_VSOCK, _recv_exact


VMADDR_CID_ANY = 0xFFFFFFFF
DEFAULT_RELAY_PORT = 5002
TARGET_TLS_PORT = 5003
MAX_CONTROL_BYTES = 16 * 1024
MAX_CHANNEL_BYTES_PER_DIRECTION = 512 * 1024 * 1024
RELAY_CHUNK_BYTES = 64 * 1024
IDLE_TIMEOUT_SECONDS = 1800.0
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{32}$")

_CID_BY_ROLE = {role: int(spec["cid"]) for role, spec in ROLE_SPECS.items()}
_COORDINATOR_CID = _CID_BY_ROLE["gateway_coordinator"]
_APPROVED_PAIRS = frozenset(
    {
        (_COORDINATOR_CID, _CID_BY_ROLE["gateway_scoring"]),
        (_COORDINATOR_CID, _CID_BY_ROLE["gateway_autoresearch"]),
        (_CID_BY_ROLE["gateway_scoring"], _COORDINATOR_CID),
        (_CID_BY_ROLE["gateway_autoresearch"], _COORDINATOR_CID),
    }
)


class InterEnclaveRelayError(RuntimeError):
    """A relay request violates the fixed V2 topology or framing contract."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _send_frame(connection: Any, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json(value)
    if len(encoded) < 2 or len(encoded) > MAX_CONTROL_BYTES:
        raise InterEnclaveRelayError("relay control response is outside size limit")
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _read_control(connection: Any) -> dict[str, Any]:
    prefix = _recv_exact(connection, 4)
    if len(prefix) != 4:
        raise InterEnclaveRelayError("relay control prefix is incomplete")
    size = int.from_bytes(prefix, "big")
    if size < 2 or size > MAX_CONTROL_BYTES:
        raise InterEnclaveRelayError("relay control frame is outside size limit")
    encoded = _recv_exact(connection, size)
    if len(encoded) != size:
        raise InterEnclaveRelayError("relay control body is incomplete")
    try:
        value = json.loads(encoded.decode("ascii"))
    except Exception as exc:
        raise InterEnclaveRelayError("relay control JSON is invalid") from exc
    if not isinstance(value, dict):
        raise InterEnclaveRelayError("relay control request must be an object")
    return value


def _validated_target(request: Mapping[str, Any], *, source_cid: int) -> tuple[int, int, str]:
    if set(request) != {
        "schema_version",
        "channel_id",
        "source_cid",
        "target_cid",
        "target_port",
        "topology_hash",
    }:
        raise InterEnclaveRelayError("relay control fields are invalid")
    if request.get("schema_version") != "leadpoet.inter_enclave_relay.v2":
        raise InterEnclaveRelayError("relay control schema is invalid")
    if int(request.get("source_cid", -1)) != int(source_cid):
        raise InterEnclaveRelayError("relay source CID differs from socket peer")
    target_cid = int(request.get("target_cid", -1))
    target_port = int(request.get("target_port", -1))
    if (int(source_cid), target_cid) not in _APPROVED_PAIRS:
        raise InterEnclaveRelayError("relay CID pair is not authorized")
    if target_port != TARGET_TLS_PORT:
        raise InterEnclaveRelayError("relay target port is not authorized")
    if request.get("topology_hash") != topology_hash():
        raise InterEnclaveRelayError("relay topology hash mismatch")
    channel_id = str(request.get("channel_id") or "").lower()
    if not _CHANNEL_ID_RE.fullmatch(channel_id):
        raise InterEnclaveRelayError("relay channel ID is invalid")
    return target_cid, target_port, channel_id


def _relay_opaque(left: Any, right: Any, *, idle_timeout: float) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    totals = {left: 0, right: 0}
    last_activity = time.monotonic()
    while active:
        remaining = idle_timeout - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise InterEnclaveRelayError("relay channel idle timeout")
        readable, _, _ = select.select(list(active), [], [], min(1.0, remaining))
        if not readable:
            continue
        for source in readable:
            data = source.recv(RELAY_CHUNK_BYTES)
            destination = peers[source]
            if not data:
                active.discard(source)
                try:
                    destination.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            totals[source] += len(data)
            if totals[source] > MAX_CHANNEL_BYTES_PER_DIRECTION:
                raise InterEnclaveRelayError("relay channel byte limit exceeded")
            destination.sendall(data)
            last_activity = time.monotonic()


def _connect_target(target_cid: int, target_port: int) -> Any:
    connection = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    connection.connect((target_cid, target_port))
    return connection


def _handle_connection(
    connection: Any,
    *,
    source_cid: int,
    connector: Callable[[int, int], Any] = _connect_target,
    idle_timeout: float = IDLE_TIMEOUT_SECONDS,
) -> None:
    target = None
    try:
        request = _read_control(connection)
        target_cid, target_port, channel_id = _validated_target(
            request,
            source_cid=source_cid,
        )
        target = connector(target_cid, target_port)
        _send_frame(
            connection,
            {
                "result": {
                    "status": "connected",
                    "channel_id": channel_id,
                    "topology_hash": topology_hash(),
                }
            },
        )
        _relay_opaque(connection, target, idle_timeout=idle_timeout)
    except Exception as exc:
        try:
            _send_frame(
                connection,
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                },
            )
        except Exception:
            pass
    finally:
        for candidate in (target, connection):
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass


def serve_forever(*, port: int = DEFAULT_RELAY_PORT) -> None:
    listener = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    listener.bind((VMADDR_CID_ANY, int(port)))
    listener.listen(64)
    while True:
        connection, address = listener.accept()
        source_cid = int(address[0])
        threading.Thread(
            target=_handle_connection,
            kwargs={"connection": connection, "source_cid": source_cid},
            name="gateway-inter-enclave-relay",
            daemon=True,
        ).start()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_RELAY_PORT)
    args = parser.parse_args(argv)
    serve_forever(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
