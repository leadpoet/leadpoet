"""Parent-side raw-byte forwarder for gateway-enclave scoring egress.

The parent validates a bounded connect request, opens only a globally routable
HTTP(S) destination, and then relays opaque bytes.  HTTPS TLS handshakes and
certificate validation happen in the enclave; this process never terminates
TLS or receives provider credentials from the request framing layer.
"""

from __future__ import annotations

import errno
import hashlib
import ipaddress
import json
import logging
import os
import select
import socket
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from gateway.tee.egress_policy import destination_policy_hash, normalize_destination
from gateway.utils.tee_client import AF_VSOCK, _recv_exact


logger = logging.getLogger(__name__)

VMADDR_CID_ANY = 0xFFFFFFFF
DEFAULT_FORWARDER_PORT = 5001
MAX_CONTROL_BYTES = 16 * 1024
MAX_TUNNEL_BYTES_PER_DIRECTION = 256 * 1024 * 1024
DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0
CONNECT_TIMEOUT_SECONDS = 15.0
RELAY_CHUNK_BYTES = 64 * 1024
_PEER_CLOSE_ERRNOS = frozenset(
    value
    for value in (
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ENOTCONN,
        getattr(errno, "ESHUTDOWN", None),
    )
    if value is not None
)


class TEEEgressForwarderError(RuntimeError):
    """The parent could not safely establish or relay an enclave tunnel."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _destination_ref(host: str, port: int) -> str:
    return hashlib.sha256((host + ":" + str(port)).encode("ascii")).hexdigest()[:16]


def _send_response(connection: Any, response: Dict[str, Any]) -> None:
    encoded = _canonical_json(response)
    if len(encoded) > MAX_CONTROL_BYTES:
        raise TEEEgressForwarderError("egress control response exceeds limit")
    connection.sendall(len(encoded).to_bytes(4, byteorder="big") + encoded)


def _global_address_infos(
    host: str,
    port: int,
    *,
    resolver: Callable[..., Iterable[Tuple[Any, ...]]] = socket.getaddrinfo,
) -> List[Tuple[Any, ...]]:
    try:
        infos = list(resolver(host, port, type=socket.SOCK_STREAM))
    except Exception as exc:
        raise TEEEgressForwarderError("egress destination DNS resolution failed") from exc
    usable = []
    observed_addresses = set()
    for info in infos:
        if len(info) != 5:
            continue
        family, socktype, protocol, _canonical_name, sockaddr = info
        if socktype != socket.SOCK_STREAM or not isinstance(sockaddr, tuple) or not sockaddr:
            continue
        address = str(sockaddr[0])
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise TEEEgressForwarderError("egress DNS returned an invalid address") from exc
        if not parsed.is_global:
            raise TEEEgressForwarderError("egress DNS returned a non-global address")
        key = (family, protocol, sockaddr)
        if key not in observed_addresses:
            observed_addresses.add(key)
            usable.append((family, socktype, protocol, "", sockaddr))
    if not usable:
        raise TEEEgressForwarderError("egress destination has no global address")
    return usable


def _connect_public_destination(
    host: str,
    port: int,
    *,
    resolver: Callable[..., Iterable[Tuple[Any, ...]]] = socket.getaddrinfo,
    socket_factory: Callable[..., Any] = socket.socket,
) -> Any:
    last_error = None
    for family, socktype, protocol, _canonical_name, sockaddr in _global_address_infos(
        host,
        port,
        resolver=resolver,
    ):
        candidate = socket_factory(family, socktype, protocol)
        try:
            candidate.settimeout(CONNECT_TIMEOUT_SECONDS)
            candidate.connect(sockaddr)
            candidate.settimeout(None)
            return candidate
        except Exception as exc:
            last_error = exc
            try:
                candidate.close()
            except Exception:
                pass
    raise TEEEgressForwarderError("egress destination connection failed") from last_error


def _relay_bidirectional(
    left: Any,
    right: Any,
    *,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    peers = {left: right, right: left}
    active = {left, right}
    transferred = {left: 0, right: 0}
    first_closed = ""
    write_closed = ""
    last_activity = time.monotonic()
    while active:
        remaining = max(0.0, idle_timeout_seconds - (time.monotonic() - last_activity))
        if remaining <= 0:
            raise TEEEgressForwarderError("egress tunnel idle timeout")
        readable, _writable, _exceptional = select.select(list(active), [], [], min(1.0, remaining))
        if not readable:
            continue
        for source in readable:
            data = source.recv(RELAY_CHUNK_BYTES)
            destination = peers[source]
            if not data:
                if not first_closed:
                    first_closed = "enclave" if source is left else "provider"
                active.discard(source)
                try:
                    destination.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            next_total = transferred[source] + len(data)
            if next_total > MAX_TUNNEL_BYTES_PER_DIRECTION:
                raise TEEEgressForwarderError("egress tunnel byte limit exceeded")
            try:
                destination.sendall(data)
            except OSError as exc:
                if int(getattr(exc, "errno", 0) or 0) not in _PEER_CLOSE_ERRNOS:
                    raise
                # A provider may close after a complete response before the
                # enclave emits its final TLS bytes. Stop only that direction;
                # the authenticated client still decides whether the response
                # it received was complete.
                closed_name = "enclave" if destination is left else "provider"
                if not first_closed:
                    first_closed = closed_name
                if not write_closed:
                    write_closed = closed_name
                active.discard(source)
                continue
            transferred[source] = next_total
            last_activity = time.monotonic()
    result = {
        "enclave_to_provider_bytes": transferred[left],
        "provider_to_enclave_bytes": transferred[right],
        "first_closed": first_closed or "unknown",
    }
    if write_closed:
        result["write_closed"] = write_closed
    return result


def _handle_connection(
    connection: Any,
    *,
    connector: Callable[[str, int], Any] = _connect_public_destination,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
) -> None:
    upstream = None
    connected = False
    destination_ref = "unknown"
    try:
        prefix = _recv_exact(connection, 4)
        if len(prefix) != 4:
            raise TEEEgressForwarderError("egress control frame is incomplete")
        size = int.from_bytes(prefix, byteorder="big")
        if size < 2 or size > MAX_CONTROL_BYTES:
            raise TEEEgressForwarderError("egress control frame size is invalid")
        encoded = _recv_exact(connection, size)
        if len(encoded) != size:
            raise TEEEgressForwarderError("egress control frame body is incomplete")
        request = json.loads(encoded.decode("ascii"))
        if not isinstance(request, dict) or set(request) != {"method", "params"}:
            raise TEEEgressForwarderError("egress control request shape is invalid")
        if request.get("method") != "connect" or not isinstance(request.get("params"), dict):
            raise TEEEgressForwarderError("egress control method is invalid")
        params = request["params"]
        if set(params) != {"host", "port", "policy_hash"}:
            raise TEEEgressForwarderError("egress connect parameters are invalid")
        if params.get("policy_hash") != destination_policy_hash():
            raise TEEEgressForwarderError("egress policy hash mismatch")
        host, port = normalize_destination(params.get("host"), params.get("port"))
        destination_ref = _destination_ref(host, port)
        upstream = connector(host, port)
        _send_response(
            connection,
            {
                "result": {
                    "status": "connected",
                    "policy_hash": destination_policy_hash(),
                }
            },
        )
        connected = True
        relay = _relay_bidirectional(
            connection,
            upstream,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        logger.info(
            "gateway_tee_egress_tunnel_closed destination_ref=%s "
            "enclave_to_provider_bytes=%d provider_to_enclave_bytes=%d "
            "first_closed=%s write_closed=%s",
            destination_ref,
            relay["enclave_to_provider_bytes"],
            relay["provider_to_enclave_bytes"],
            relay["first_closed"],
            relay.get("write_closed", "none"),
        )
    except Exception as exc:
        if not connected:
            try:
                _send_response(
                    connection,
                    {
                        "status": "error",
                        "error": type(exc).__name__,
                    },
                )
            except Exception:
                pass
        logger.warning(
            "gateway_tee_egress_tunnel_failed destination_ref=%s error_type=%s",
            destination_ref,
            type(exc).__name__,
        )
    finally:
        for candidate in (upstream, connection):
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass


class TEEEgressForwarder:
    def __init__(
        self,
        *,
        port: int = DEFAULT_FORWARDER_PORT,
        socket_factory: Callable[..., Any] = socket.socket,
        connector: Callable[[str, int], Any] = _connect_public_destination,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self.port = int(port)
        self._socket_factory = socket_factory
        self._connector = connector
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._listener = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._listener is not None)

    def start(self) -> Dict[str, Any]:
        if self.running:
            return self.status()
        listener = self._socket_factory(AF_VSOCK, socket.SOCK_STREAM)
        try:
            listener.bind((VMADDR_CID_ANY, self.port))
            listener.listen(64)
        except Exception:
            listener.close()
            raise
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="gateway-tee-egress-forwarder",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "status": "running" if self.running else "stopped",
            "port": self.port,
            "policy_hash": destination_policy_hash(),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
        self._listener = None

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._listener.accept()
            except Exception:
                if not self._stop.is_set():
                    logger.exception("gateway_tee_egress_forwarder_accept_failed")
                return
            threading.Thread(
                target=_handle_connection,
                kwargs={
                    "connection": connection,
                    "connector": self._connector,
                    "idle_timeout_seconds": self._idle_timeout_seconds,
                },
                name="gateway-tee-egress-tunnel",
                daemon=True,
            ).start()


_FORWARDER = None  # type: Optional[TEEEgressForwarder]
_FORWARDER_LOCK = threading.Lock()


def _configured_port() -> int:
    try:
        value = int(os.getenv("RESEARCH_LAB_TEE_EGRESS_VSOCK_PORT", str(DEFAULT_FORWARDER_PORT)))
    except ValueError:
        value = DEFAULT_FORWARDER_PORT
    if value <= 1024 or value > 65535:
        raise TEEEgressForwarderError("configured egress vsock port is invalid")
    return value


def ensure_tee_egress_forwarder() -> Dict[str, Any]:
    """Start the process-local forwarder or recognize another worker's bind."""

    global _FORWARDER
    with _FORWARDER_LOCK:
        if _FORWARDER is not None and _FORWARDER.running:
            return _FORWARDER.status()
        candidate = TEEEgressForwarder(port=_configured_port())
        try:
            status = candidate.start()
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            return {
                "status": "owned_by_peer_process",
                "port": candidate.port,
                "policy_hash": destination_policy_hash(),
            }
        _FORWARDER = candidate
        return status


def main() -> int:
    """Run the parent forwarder for the lifetime of the gateway deployment."""

    logging.basicConfig(level=logging.INFO, force=True)
    status = ensure_tee_egress_forwarder()
    if status.get("status") != "running":
        raise TEEEgressForwarderError(
            "gateway TEE egress forwarder port is already owned"
        )
    print(json.dumps(status, sort_keys=True), flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
