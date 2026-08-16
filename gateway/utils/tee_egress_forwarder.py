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

from gateway.tee.egress_policy import (
    destination_policy_hash,
    normalize_destination,
    normalize_proxy_destination,
)
from gateway.tee.egress_framing import (
    TUNNEL_FRAMING_MODE,
    relay_raw_and_framed,
)
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
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)


class TEEEgressForwarderError(RuntimeError):
    """The parent could not safely establish or relay an enclave tunnel."""


class TEEEgressForwarderCleanupError(TEEEgressForwarderError):
    """A destination candidate still has locally owned transport state."""

    def __init__(self, *, primary_error: Exception, resource: Any) -> None:
        super().__init__("egress destination socket cleanup failed")
        self.primary_error = primary_error
        self._resources = (resource,)


def _shutdown_and_close_socket(candidate: Any) -> bool:
    """Attempt full-duplex shutdown and require descriptor release."""

    if candidate is None:
        return True
    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        # A connected peer may already be half-closed and listening sockets
        # commonly reject shutdown.  close() is the ownership boundary.
        pass
    try:
        # socket.close() returns None.  Test doubles and adapters may return
        # False explicitly when they still own the underlying descriptor.
        return candidate.close() is not False
    except Exception:
        return False


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
            if not _shutdown_and_close_socket(candidate):
                raise TEEEgressForwarderCleanupError(
                    primary_error=exc,
                    resource=candidate,
                ) from exc
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
            try:
                data = source.recv(RELAY_CHUNK_BYTES)
            except OSError as exc:
                if exc.errno not in _PEER_CLOSE_ERRNOS:
                    raise
                data = b""
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
    cleanup_failure_callback: Optional[
        Callable[[Any, str, Optional[Exception]], None]
    ] = None,
) -> None:
    upstream = None
    connected = False
    destination_ref = "unknown"
    primary_error = None  # type: Optional[Exception]
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
        base_params = {"host", "port", "policy_hash"}
        allowed_params = base_params | {"purpose", "tunnel_framing"}
        if (
            not base_params.issubset(params)
            or not set(params).issubset(allowed_params)
        ):
            raise TEEEgressForwarderError("egress connect parameters are invalid")
        if params.get("policy_hash") != destination_policy_hash():
            raise TEEEgressForwarderError("egress policy hash mismatch")
        purpose = str(params.get("purpose") or "provider")
        tunnel_framing = str(params.get("tunnel_framing") or "")
        if tunnel_framing and tunnel_framing != TUNNEL_FRAMING_MODE:
            raise TEEEgressForwarderError(
                "egress tunnel framing is invalid"
            )
        if "tunnel_framing" in params and not tunnel_framing:
            raise TEEEgressForwarderError(
                "egress tunnel framing is invalid"
            )
        if purpose == "provider":
            host, port = normalize_destination(
                params.get("host"),
                params.get("port"),
            )
        elif purpose == "upstream_proxy":
            host, port = normalize_proxy_destination(
                params.get("host"),
                params.get("port"),
            )
        else:
            raise TEEEgressForwarderError("egress connect purpose is invalid")
        destination_ref = _destination_ref(host, port)
        upstream = connector(host, port)
        _send_response(
            connection,
            {
                "result": {
                    "status": "connected",
                    "policy_hash": destination_policy_hash(),
                    **(
                        {"tunnel_framing": tunnel_framing}
                        if tunnel_framing
                        else {}
                    ),
                }
            },
        )
        connected = True
        if tunnel_framing == TUNNEL_FRAMING_MODE:
            relay = relay_raw_and_framed(
                upstream,
                connection,
                idle_timeout_seconds=idle_timeout_seconds,
                max_bytes_per_direction=MAX_TUNNEL_BYTES_PER_DIRECTION,
                raw_label="provider",
                framed_label="enclave",
                terminal_initiator=False,
            )
        else:
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
        primary_error = exc
        if (
            isinstance(exc, TEEEgressForwarderCleanupError)
            and cleanup_failure_callback is not None
        ):
            for resource in exc._resources:
                cleanup_failure_callback(
                    resource,
                    "upstream",
                    exc.primary_error,
                )
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
        for endpoint, candidate in (
            ("upstream", upstream),
            ("enclave", connection),
        ):
            if candidate is not None and not _shutdown_and_close_socket(candidate):
                logger.warning(
                    "gateway_tee_egress_endpoint_cleanup_failed "
                    "destination_ref=%s endpoint=%s primary_error_type=%s",
                    destination_ref,
                    endpoint,
                    type(primary_error).__name__ if primary_error else "none",
                )
                if cleanup_failure_callback is not None:
                    try:
                        cleanup_failure_callback(
                            candidate,
                            endpoint,
                            primary_error,
                        )
                    except Exception:
                        # Cleanup accounting must not replace the verified
                        # tunnel failure that led here.
                        logger.exception(
                            "gateway_tee_egress_cleanup_accounting_failed"
                        )


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
        self._lifecycle_lock = threading.RLock()
        self._status_lock = threading.Lock()
        self._pending_endpoint_cleanup_lock = threading.RLock()
        self._pending_endpoint_cleanup_recovery_lock = threading.Lock()
        self._last_failure = None  # type: Optional[Dict[str, Any]]
        self._socket_cleanup_failure_count = 0
        self._pending_endpoint_cleanup = []  # type: List[Any]

    @property
    def running(self) -> bool:
        with self._status_lock:
            endpoint_cleanup_pending = bool(self._pending_endpoint_cleanup)
        return bool(
            not self._stop.is_set()
            and not endpoint_cleanup_pending
            and self._thread
            and self._thread.is_alive()
            and self._listener is not None
        )

    def _record_failure(
        self,
        stage: str,
        *,
        error: Optional[Exception] = None,
        endpoint: str = "none",
        primary_error: Optional[Exception] = None,
        cleanup_failure: bool = False,
    ) -> None:
        with self._status_lock:
            if cleanup_failure:
                self._socket_cleanup_failure_count += 1
            self._last_failure = {
                "stage": str(stage),
                "error_type": (
                    type(error).__name__
                    if error is not None
                    else "TEEEgressForwarderError"
                ),
                "errno": int(getattr(error, "errno", 0) or 0),
                "endpoint": str(endpoint),
                "primary_error_type": (
                    type(primary_error).__name__
                    if primary_error is not None
                    else "none"
                ),
            }

    def _retain_endpoint_cleanup_failure(
        self,
        candidate: Any,
        endpoint: str,
        primary_error: Optional[Exception],
    ) -> None:
        with self._pending_endpoint_cleanup_lock:
            with self._status_lock:
                if not any(
                    pending is candidate
                    for pending in self._pending_endpoint_cleanup
                ):
                    self._pending_endpoint_cleanup.append(candidate)
                self._socket_cleanup_failure_count += 1
                self._last_failure = {
                    "stage": "handler_endpoint_cleanup",
                    "error_type": "TEEEgressForwarderError",
                    "errno": 0,
                    "endpoint": str(endpoint),
                    "primary_error_type": (
                        type(primary_error).__name__
                        if primary_error is not None
                        else "none"
                    ),
                }

    def _retry_pending_endpoint_cleanup_locked(self) -> bool:
        with self._pending_endpoint_cleanup_recovery_lock:
            with self._pending_endpoint_cleanup_lock:
                with self._status_lock:
                    pending = tuple(self._pending_endpoint_cleanup)
            resolved_ids = {
                id(candidate)
                for candidate in pending
                if _shutdown_and_close_socket(candidate)
            }
            with self._pending_endpoint_cleanup_lock:
                with self._status_lock:
                    self._pending_endpoint_cleanup = [
                        candidate
                        for candidate in self._pending_endpoint_cleanup
                        if id(candidate) not in resolved_ids
                    ]
                    return not self._pending_endpoint_cleanup

    @staticmethod
    def _join_thread_bounded(thread: Any, *, timeout: float) -> bool:
        try:
            thread.join(timeout=timeout)
        except RuntimeError:
            # threading.Thread.join() raises before start(). A failed start
            # owns no running thread, but the object remains tracked until the
            # listener cleanup boundary is also proven.
            pass
        return not thread.is_alive()

    def _cleanup_lifecycle_locked(self) -> None:
        self._stop.set()
        listener = self._listener
        thread = self._thread
        listener_clean = True
        if listener is not None:
            listener_clean = _shutdown_and_close_socket(listener)
            if not listener_clean:
                self._record_failure(
                    "listener_cleanup",
                    endpoint="listener",
                    cleanup_failure=True,
                )
        thread_clean = True
        if thread is not None and thread is not threading.current_thread():
            thread_clean = self._join_thread_bounded(thread, timeout=2.0)
            if not thread_clean:
                self._record_failure(
                    "accept_loop_cleanup",
                    error=TEEEgressForwarderError(
                        "gateway TEE egress accept loop did not terminate"
                    ),
                )
        endpoints_clean = self._retry_pending_endpoint_cleanup_locked()
        if listener_clean and thread_clean and endpoints_clean:
            self._listener = None
            self._thread = None
            return
        if not listener_clean:
            raise TEEEgressForwarderError(
                "gateway TEE egress listener cleanup failed"
            )
        if not thread_clean:
            raise TEEEgressForwarderError(
                "gateway TEE egress accept loop did not terminate"
            )
        raise TEEEgressForwarderError(
            "gateway TEE egress endpoint cleanup failed"
        )

    def start(self) -> Dict[str, Any]:
        with self._lifecycle_lock:
            if self.running:
                return self.status()
            if (
                self._listener is not None
                or self._thread is not None
                or self._pending_endpoint_cleanup
            ):
                self._cleanup_lifecycle_locked()
            stop_event = threading.Event()
            self._stop = stop_event
            listener = None
            try:
                listener = self._socket_factory(AF_VSOCK, socket.SOCK_STREAM)
                listener.bind((VMADDR_CID_ANY, self.port))
                listener.listen(64)
            except Exception as exc:
                stop_event.set()
                if listener is not None and not _shutdown_and_close_socket(listener):
                    self._listener = listener
                    self._record_failure(
                        "listener_start_cleanup",
                        endpoint="listener",
                        primary_error=exc,
                        cleanup_failure=True,
                    )
                    raise TEEEgressForwarderError(
                        "gateway TEE egress listener cleanup failed after startup"
                    ) from exc
                raise
            thread = None
            try:
                thread = threading.Thread(
                    target=self._accept_loop,
                    args=(listener, stop_event),
                    name="gateway-tee-egress-forwarder",
                    daemon=True,
                )
                self._listener = listener
                self._thread = thread
                thread.start()
            except Exception as exc:
                stop_event.set()
                listener_clean = _shutdown_and_close_socket(listener)
                thread_clean = bool(
                    thread is None
                    or self._join_thread_bounded(thread, timeout=2.0)
                )
                if not listener_clean or not thread_clean:
                    self._listener = listener
                    self._thread = thread
                    self._record_failure(
                        (
                            "listener_start_cleanup"
                            if not listener_clean
                            else "accept_loop_start_cleanup"
                        ),
                        endpoint="listener",
                        primary_error=exc,
                        cleanup_failure=True,
                    )
                    raise TEEEgressForwarderError(
                        "gateway TEE egress listener cleanup failed after startup"
                        if not listener_clean
                        else "gateway TEE egress accept loop cleanup failed after startup"
                    ) from exc
                self._listener = None
                self._thread = None
                raise
            return self.status()

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            last_failure = dict(self._last_failure or {})
            cleanup_count = self._socket_cleanup_failure_count
            pending_cleanup_count = len(self._pending_endpoint_cleanup)
        cleanup_incomplete = bool(
            pending_cleanup_count
            or (
                self._stop.is_set()
                and (
                    self._listener is not None
                    or self._thread is not None
                )
            )
        )
        accept_loop_failed = bool(
            not self._stop.is_set()
            and self._thread is not None
            and not self._thread.is_alive()
        )
        result = {
            "status": (
                "running"
                if self.running
                else "cleanup_failed"
                if cleanup_incomplete
                else "failed"
                if accept_loop_failed
                else "stopped"
            ),
            "port": self.port,
            "policy_hash": destination_policy_hash(),
            "socket_cleanup_failure_count": cleanup_count,
            "pending_endpoint_cleanup_count": pending_cleanup_count,
        }
        if last_failure:
            result["last_failure"] = last_failure
        return result

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._cleanup_lifecycle_locked()

    def _accept_loop(
        self,
        listener: Any,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                connection, _address = listener.accept()
            except Exception as exc:
                if stop_event.is_set():
                    return
                if int(getattr(exc, "errno", 0) or 0) in _TRANSIENT_ACCEPT_ERRNOS:
                    continue
                self._record_failure("accept_loop", error=exc)
                if not stop_event.is_set():
                    logger.exception("gateway_tee_egress_forwarder_accept_failed")
                return
            admission_blocked = False
            handler_start_error = None  # type: Optional[Exception]
            with self._pending_endpoint_cleanup_lock:
                with self._status_lock:
                    admission_blocked = bool(
                        self._pending_endpoint_cleanup
                    )
                if not admission_blocked:
                    try:
                        threading.Thread(
                            target=self._serve_connection,
                            args=(connection,),
                            name="gateway-tee-egress-tunnel",
                            daemon=True,
                        ).start()
                    except Exception as exc:
                        handler_start_error = exc
            if admission_blocked:
                if not _shutdown_and_close_socket(connection):
                    self._retain_endpoint_cleanup_failure(
                        connection,
                        "enclave",
                        TEEEgressForwarderError(
                            "gateway TEE egress admission is cleanup-blocked"
                        ),
                    )
                continue
            if handler_start_error is not None:
                if not _shutdown_and_close_socket(connection):
                    self._retain_endpoint_cleanup_failure(
                        connection,
                        "enclave",
                        handler_start_error,
                    )
                self._record_failure(
                    "start_handler",
                    error=handler_start_error,
                )

    def _serve_connection(self, connection: Any) -> None:
        _handle_connection(
            connection,
            connector=self._connector,
            idle_timeout_seconds=self._idle_timeout_seconds,
            cleanup_failure_callback=self._retain_endpoint_cleanup_failure,
        )

    def wait_for_accept_loop(self, *, poll_seconds: float = 1.0) -> int:
        """Wait for the listener thread and distinguish expected shutdown."""

        poll_interval = max(0.01, float(poll_seconds))
        while True:
            with self._lifecycle_lock:
                thread = self._thread
                stop_event = self._stop
            with self._status_lock:
                endpoint_cleanup_pending = bool(
                    self._pending_endpoint_cleanup
                )
            if endpoint_cleanup_pending:
                if not self._retry_pending_endpoint_cleanup_locked():
                    time.sleep(poll_interval)
                continue
            if thread is None:
                return 0 if stop_event.is_set() else 1
            thread.join(timeout=poll_interval)
            if not thread.is_alive():
                return 0 if stop_event.is_set() else 1


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
        candidate = _FORWARDER
        if candidate is None:
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
    forwarder = _FORWARDER
    if forwarder is None:
        raise TEEEgressForwarderError(
            "gateway TEE egress forwarder ownership is unavailable"
        )
    try:
        exit_code = forwarder.wait_for_accept_loop()
    except KeyboardInterrupt:
        try:
            forwarder.stop()
        except Exception:
            logger.exception("gateway_tee_egress_forwarder_stop_failed")
            return 1
        return 0
    if exit_code != 0:
        logger.error("gateway_tee_egress_forwarder_accept_loop_exited")
        try:
            forwarder.stop()
        except Exception:
            logger.exception("gateway_tee_egress_forwarder_cleanup_failed")
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
