"""Parent-side opaque relay for validator-enclave chain TLS.

The relay can connect only to the measured Finney live or archive endpoint.  It
never sees an HTTP request or response because the TLS session originates and
terminates in the validator enclave.
"""

from __future__ import annotations

import errno
import ipaddress
import json
import select
import socket
import threading
import time
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PORT,
    chain_source_policy_hash,
)
from leadpoet_observability import capture_failure, record_retry, record_stage


AF_VSOCK = 40
VMADDR_CID_ANY = 0xFFFFFFFF
CHAIN_RELAY_VSOCK_PORT = 5002
MAX_CONTROL_BYTES = 16 * 1024
MAX_BYTES_PER_DIRECTION = 32 * 1024 * 1024
RELAY_CHUNK_BYTES = 64 * 1024
CONNECT_TIMEOUT_SECONDS = 15.0
IDLE_TIMEOUT_SECONDS = 90.0
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)


class ValidatorChainRelayV2Error(RuntimeError):
    """The parent relay request or destination violates the fixed policy."""


def _shutdown_and_close_socket(candidate: Any) -> bool:
    """Attempt full-duplex shutdown and require descriptor release."""

    if candidate is None:
        return True
    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        return candidate.close() is not False
    except Exception:
        return False


def _recv_exact(connection: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _send_control(connection: Any, value: Any) -> None:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if len(encoded) < 2 or len(encoded) > MAX_CONTROL_BYTES:
        raise ValidatorChainRelayV2Error("relay response is outside size limit")
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _read_control(connection: Any) -> dict:
    prefix = _recv_exact(connection, 4)
    if len(prefix) != 4:
        raise ValidatorChainRelayV2Error("relay control prefix is incomplete")
    size = int.from_bytes(prefix, "big")
    if size < 2 or size > MAX_CONTROL_BYTES:
        raise ValidatorChainRelayV2Error("relay control size is invalid")
    body = _recv_exact(connection, size)
    if len(body) != size:
        raise ValidatorChainRelayV2Error("relay control body is incomplete")
    try:
        value = json.loads(body.decode("ascii"))
    except Exception as exc:
        raise ValidatorChainRelayV2Error("relay control JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValidatorChainRelayV2Error("relay control must be an object")
    return value


def _validate_control(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "host",
        "port",
        "policy_hash",
    }:
        raise ValidatorChainRelayV2Error("relay control fields are invalid")
    if value.get("schema_version") != "leadpoet.validator_chain_relay.v2":
        raise ValidatorChainRelayV2Error("relay schema is invalid")
    destination_host = value.get("host")
    if (
        destination_host
        not in (CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST)
        or value.get("port") != CHAIN_ENDPOINT_PORT
    ):
        raise ValidatorChainRelayV2Error("relay destination is not the measured chain")
    if value.get("policy_hash") != chain_source_policy_hash():
        raise ValidatorChainRelayV2Error("relay policy hash differs")
    return str(destination_host)


def _global_addresses(
    destination_host: str,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
) -> list:
    if destination_host not in (CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST):
        raise ValidatorChainRelayV2Error("chain destination is not measured")
    try:
        entries = resolver(
            destination_host,
            CHAIN_ENDPOINT_PORT,
            type=socket.SOCK_STREAM,
        )
    except Exception as exc:
        raise ValidatorChainRelayV2Error("chain DNS resolution failed") from exc
    result = []
    for family, socktype, protocol, _canonical, address in entries:
        if socktype != socket.SOCK_STREAM or not address:
            continue
        try:
            parsed = ipaddress.ip_address(str(address[0]))
        except ValueError as exc:
            raise ValidatorChainRelayV2Error("chain DNS address is invalid") from exc
        if not parsed.is_global:
            raise ValidatorChainRelayV2Error("chain DNS returned a non-global address")
        result.append((family, socktype, protocol, address))
    if not result:
        raise ValidatorChainRelayV2Error("chain endpoint has no global address")
    return result


def _connect_chain(
    destination_host: str,
    *,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
    socket_factory: Callable[..., Any] = socket.socket,
) -> Any:
    last_error = None
    for family, socktype, protocol, address in _global_addresses(
        destination_host,
        resolver,
    ):
        connection = socket_factory(family, socktype, protocol)
        try:
            connection.settimeout(CONNECT_TIMEOUT_SECONDS)
            connection.connect(address)
            connection.settimeout(None)
            return connection
        except Exception as exc:
            last_error = exc
            if not _shutdown_and_close_socket(connection):
                raise ValidatorChainRelayV2Error(
                    "chain connection cleanup failed"
                ) from exc
    raise ValidatorChainRelayV2Error("chain connection failed") from last_error


def _relay(left: Any, right: Any) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    counts = {left: 0, right: 0}
    last_activity = time.monotonic()
    while active:
        remaining = IDLE_TIMEOUT_SECONDS - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise ValidatorChainRelayV2Error("chain relay idle timeout")
        readable, _writable, _exceptional = select.select(
            list(active), [], [], min(1.0, remaining)
        )
        for source in readable:
            data = source.recv(RELAY_CHUNK_BYTES)
            target = peers[source]
            if not data:
                active.discard(source)
                try:
                    target.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            counts[source] += len(data)
            if counts[source] > MAX_BYTES_PER_DIRECTION:
                raise ValidatorChainRelayV2Error("chain relay byte limit exceeded")
            target.sendall(data)
            last_activity = time.monotonic()


def handle_chain_relay_connection(
    connection: Any,
    *,
    connector: Callable[[str], Any] = _connect_chain,
    cleanup_failure_callback: Optional[
        Callable[[Any, str, Optional[Exception]], None]
    ] = None,
) -> None:
    upstream = None
    primary_error = None  # type: Optional[Exception]
    cleanup_failed = False
    try:
        request = _read_control(connection)
        destination_host = _validate_control(request)
        upstream = connector(destination_host)
        _send_control(
            connection,
            {
                "status": "connected",
                "policy_hash": chain_source_policy_hash(),
            },
        )
        _relay(connection, upstream)
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        for endpoint, candidate in (
            ("upstream", upstream),
            ("enclave", connection),
        ):
            if candidate is not None and not _shutdown_and_close_socket(candidate):
                cleanup_failed = True
                if cleanup_failure_callback is not None:
                    try:
                        cleanup_failure_callback(
                            candidate,
                            endpoint,
                            primary_error,
                        )
                    except Exception as cleanup_error:
                        if primary_error is None:
                            raise ValidatorChainRelayV2Error(
                                "chain relay cleanup accounting failed"
                            ) from cleanup_error
        if cleanup_failed and primary_error is None:
            raise ValidatorChainRelayV2Error(
                "chain relay endpoint cleanup failed"
            )


class ValidatorChainRelayV2:
    def __init__(
        self,
        *,
        port: int = CHAIN_RELAY_VSOCK_PORT,
        socket_factory: Callable[..., Any] = socket.socket,
        connector: Callable[[str], Any] = _connect_chain,
    ) -> None:
        self.port = int(port)
        self._socket_factory = socket_factory
        self._connector = connector
        self._listener = None  # type: Optional[Any]
        self._thread = None  # type: Optional[threading.Thread]
        self._stop = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._status_lock = threading.Lock()
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
            and self._thread is not None
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
                    else "ValidatorChainRelayV2Error"
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
        with self._status_lock:
            if not any(
                pending is candidate for pending in self._pending_endpoint_cleanup
            ):
                self._pending_endpoint_cleanup.append(candidate)
            self._socket_cleanup_failure_count += 1
            self._last_failure = {
                "stage": "handler_endpoint_cleanup",
                "error_type": "ValidatorChainRelayV2Error",
                "errno": 0,
                "endpoint": str(endpoint),
                "primary_error_type": (
                    type(primary_error).__name__
                    if primary_error is not None
                    else "none"
                ),
            }

    def _retry_pending_endpoint_cleanup_locked(self) -> bool:
        with self._status_lock:
            pending = list(self._pending_endpoint_cleanup)
        remaining = []
        for candidate in pending:
            if not _shutdown_and_close_socket(candidate):
                remaining.append(candidate)
        with self._status_lock:
            self._pending_endpoint_cleanup = remaining
        return not remaining

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
            thread.join(timeout=2.0)
            thread_clean = not thread.is_alive()
            if not thread_clean:
                self._record_failure(
                    "accept_loop_cleanup",
                    error=ValidatorChainRelayV2Error(
                        "validator chain relay accept loop did not terminate"
                    ),
                )
        endpoints_clean = self._retry_pending_endpoint_cleanup_locked()
        if listener_clean and thread_clean and endpoints_clean:
            self._listener = None
            self._thread = None
            return
        if not listener_clean:
            raise ValidatorChainRelayV2Error(
                "validator chain relay listener cleanup failed"
            )
        if not thread_clean:
            raise ValidatorChainRelayV2Error(
                "validator chain relay accept loop did not terminate"
            )
        raise ValidatorChainRelayV2Error(
            "validator chain relay endpoint cleanup failed"
        )

    def start(self) -> dict:
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
                listener.listen(8)
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
                    raise ValidatorChainRelayV2Error(
                        "validator chain relay listener cleanup failed after startup"
                    ) from exc
                raise
            self._listener = listener
            thread = threading.Thread(
                target=self._accept_loop,
                args=(listener, stop_event),
                name="validator-chain-relay-v2",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except Exception as exc:
                stop_event.set()
                if not _shutdown_and_close_socket(listener):
                    self._record_failure(
                        "listener_start_cleanup",
                        endpoint="listener",
                        primary_error=exc,
                        cleanup_failure=True,
                    )
                    raise ValidatorChainRelayV2Error(
                        "validator chain relay listener cleanup failed after startup"
                    ) from exc
                self._listener = None
                self._thread = None
                raise
            return self.status()

    def status(self) -> dict:
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
            "policy_hash": chain_source_policy_hash(),
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
                    capture_failure(
                        "runtime.enclave_relay_unavailable",
                        component="validator",
                        stage="chain_relay_accept",
                        exception=exc,
                        terminal=True,
                        retryable=False,
                        fail_closed=True,
                        runtime_sha=(
                            os.environ.get("GITHUB_SHA")
                            or os.environ.get("GIT_COMMIT")
                            or ""
                        ),
                    )
                return
            try:
                threading.Thread(
                    target=self._serve_connection,
                    args=(connection,),
                    daemon=True,
                ).start()
            except Exception as exc:
                if not _shutdown_and_close_socket(connection):
                    self._retain_endpoint_cleanup_failure(
                        connection,
                        "enclave",
                        exc,
                    )
                self._record_failure("start_handler", error=exc)

    def _serve_connection(self, connection: Any) -> None:
        started = time.monotonic()
        try:
            handle_chain_relay_connection(
                connection,
                connector=self._connector,
                cleanup_failure_callback=self._retain_endpoint_cleanup_failure,
            )
        except Exception as exc:
            record_retry(
                "authority.dependency_unreadable",
                component="validator",
                stage="chain_relay_request",
                attempt=1,
                attempts=1,
                dependency="finney",
                exception_class=type(exc).__name__,
                runtime_sha=(
                    os.environ.get("GITHUB_SHA")
                    or os.environ.get("GIT_COMMIT")
                    or ""
                ),
            )
            raise
        else:
            record_stage(
                component="validator",
                stage="chain_relay_request",
                status="passed",
                duration_seconds=time.monotonic() - started,
                dependency="finney",
                runtime_sha=(
                    os.environ.get("GITHUB_SHA")
                    or os.environ.get("GIT_COMMIT")
                    or ""
                ),
            )

    def wait_for_accept_loop(self, *, poll_seconds: float = 1.0) -> int:
        """Wait for the listener thread and distinguish expected shutdown."""

        while True:
            with self._lifecycle_lock:
                thread = self._thread
                stop_event = self._stop
            with self._status_lock:
                endpoint_cleanup_pending = bool(
                    self._pending_endpoint_cleanup
                )
            if endpoint_cleanup_pending:
                return 1
            if thread is None:
                return 0 if stop_event.is_set() else 1
            thread.join(timeout=max(0.01, float(poll_seconds)))
            if not thread.is_alive():
                return 0 if stop_event.is_set() else 1


def main() -> int:
    relay = ValidatorChainRelayV2()
    print(json.dumps(relay.start(), sort_keys=True), flush=True)
    try:
        exit_code = relay.wait_for_accept_loop()
    except KeyboardInterrupt:
        relay.stop()
        return 0
    if exit_code != 0:
        relay.stop()
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
