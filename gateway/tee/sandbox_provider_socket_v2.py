"""Measured AF_UNIX bridge from a runsc model sandbox to the provider broker."""

from __future__ import annotations

import base64
import errno
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from leadpoet_canonical.attested_v2 import canonical_json


SANDBOX_PROVIDER_SCHEMA_VERSION = "leadpoet.sandbox_provider_rpc.v2"
MAX_SANDBOX_PROVIDER_FRAME_BYTES = 64 * 1024 * 1024
DEFAULT_SANDBOX_PROVIDER_DRAIN_TIMEOUT_SECONDS = 30.0
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)


class SandboxProviderSocketV2Error(RuntimeError):
    """A sandbox provider request is malformed or outside its measured job."""


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
        chunk = connection.recv(size - len(output))
        if not chunk:
            raise SandboxProviderSocketV2Error("sandbox provider frame is incomplete")
        output.extend(chunk)
    return bytes(output)


def _read_frame(connection: Any) -> Dict[str, Any]:
    size = int.from_bytes(_recv_exact(connection, 4), "big")
    if size < 2 or size > MAX_SANDBOX_PROVIDER_FRAME_BYTES:
        raise SandboxProviderSocketV2Error("sandbox provider frame is outside limit")
    try:
        value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    except Exception as exc:
        raise SandboxProviderSocketV2Error("sandbox provider frame is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SandboxProviderSocketV2Error("sandbox provider request must be an object")
    return value


def _write_frame(connection: Any, value: Mapping[str, Any]) -> None:
    encoded = canonical_json(dict(value)).encode("utf-8")
    if len(encoded) < 2 or len(encoded) > MAX_SANDBOX_PROVIDER_FRAME_BYTES:
        raise SandboxProviderSocketV2Error("sandbox provider response is outside limit")
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _normalize_request(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "method",
        "url",
        "headers",
        "body_b64",
        "timeout_ms",
    }
    if set(value) != fields or value.get("schema_version") != SANDBOX_PROVIDER_SCHEMA_VERSION:
        raise SandboxProviderSocketV2Error("sandbox provider request fields are invalid")
    headers = value.get("headers")
    if not isinstance(headers, Mapping):
        raise SandboxProviderSocketV2Error("sandbox provider headers are invalid")
    try:
        body = base64.b64decode(str(value.get("body_b64") or ""), validate=True)
    except Exception as exc:
        raise SandboxProviderSocketV2Error("sandbox provider body is invalid") from exc
    timeout_ms = value.get("timeout_ms")
    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise SandboxProviderSocketV2Error("sandbox provider timeout is invalid")
    return {
        "method": str(value.get("method") or "").upper(),
        "url": str(value.get("url") or ""),
        "headers": {str(name): str(item) for name, item in headers.items()},
        "body": body,
        "timeout_ms": timeout_ms,
    }


class SandboxProviderSocketServerV2:
    """Serve one job-scoped provider RPC socket with shared retry ordinals."""

    def __init__(
        self,
        *,
        socket_path: Path,
        transport: BrokeredProviderTransportV2,
        execution_scope: Any,
        drain_timeout_seconds: float = DEFAULT_SANDBOX_PROVIDER_DRAIN_TIMEOUT_SECONDS,
    ) -> None:
        if float(drain_timeout_seconds) <= 0:
            raise SandboxProviderSocketV2Error(
                "sandbox provider drain timeout must be positive"
            )
        self.socket_path = Path(socket_path)
        self._transport = transport
        self._execution_scope = execution_scope
        self._drain_timeout_seconds = float(drain_timeout_seconds)
        self._listener = None
        self._thread = None
        self._stop = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._status_lock = threading.Lock()
        self._last_failure = None  # type: Optional[Dict[str, Any]]
        self._socket_cleanup_failure_count = 0
        self._pending_endpoint_cleanup = []  # type: List[Any]
        self._handler_condition = threading.Condition()
        self._handlers: set[threading.Thread] = set()

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
                    else "SandboxProviderSocketV2Error"
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
                "error_type": "SandboxProviderSocketV2Error",
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

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            last_failure = dict(self._last_failure or {})
            cleanup_count = self._socket_cleanup_failure_count
            pending_cleanup_count = len(self._pending_endpoint_cleanup)
        with self._handler_condition:
            active_handler_count = len(self._handlers)
        cleanup_incomplete = bool(
            pending_cleanup_count
            or (
                self._stop.is_set()
                and (
                    self._listener is not None
                    or self._thread is not None
                    or active_handler_count
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
            "active_handler_count": active_handler_count,
            "socket_cleanup_failure_count": cleanup_count,
            "pending_endpoint_cleanup_count": pending_cleanup_count,
        }
        if last_failure:
            result["last_failure"] = last_failure
        return result

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.running:
                return
            with self._handler_condition:
                has_handlers = bool(self._handlers)
            if (
                self._listener is not None
                or self._thread is not None
                or self._pending_endpoint_cleanup
                or has_handlers
            ):
                self._cleanup_lifecycle_locked()
            stop_event = threading.Event()
            self._stop = stop_event
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            self.socket_path.unlink(missing_ok=True)
            listener = None
            try:
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(self.socket_path))
                self.socket_path.chmod(0o600)
                listener.listen(32)
                listener.settimeout(0.25)
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
                    raise SandboxProviderSocketV2Error(
                        "sandbox provider listener cleanup failed after startup"
                    ) from exc
                self.socket_path.unlink(missing_ok=True)
                raise
            self._listener = listener
            thread = threading.Thread(
                target=self._accept_loop,
                args=(listener, stop_event),
                name="leadpoet-sandbox-provider-v2",
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
                    raise SandboxProviderSocketV2Error(
                        "sandbox provider listener cleanup failed after startup"
                    ) from exc
                self._listener = None
                self._thread = None
                self.socket_path.unlink(missing_ok=True)
                raise

    def close(self) -> None:
        with self._lifecycle_lock:
            self._cleanup_lifecycle_locked()

    def _cleanup_lifecycle_locked(self) -> None:
        self._stop.set()
        listener = self._listener
        listener_clean = True
        if listener is not None:
            listener_clean = _shutdown_and_close_socket(listener)
            if not listener_clean:
                self._record_failure(
                    "listener_cleanup",
                    endpoint="listener",
                    cleanup_failure=True,
                )
        accept_thread = self._thread
        thread_clean = True
        if (
            accept_thread is not None
            and accept_thread is not threading.current_thread()
        ):
            # The listener uses a 250ms timeout so platforms where close()
            # does not immediately interrupt accept() still get one bounded
            # wakeup before liveness is judged.
            accept_thread.join(
                timeout=min(
                    1.0,
                    max(0.3, self._drain_timeout_seconds),
                )
            )
            thread_clean = not accept_thread.is_alive()
            if not thread_clean:
                self._record_failure(
                    "accept_loop_cleanup",
                    error=SandboxProviderSocketV2Error(
                        "sandbox provider accept loop did not terminate"
                    ),
                )
        deadline = time.monotonic() + self._drain_timeout_seconds
        handlers_clean = True
        with self._handler_condition:
            while self._handlers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    active_count = len(self._handlers)
                    handlers_clean = False
                    break
                self._handler_condition.wait(timeout=remaining)
        endpoints_clean = self._retry_pending_endpoint_cleanup_locked()
        if listener_clean and thread_clean and handlers_clean and endpoints_clean:
            self._listener = None
            self._thread = None
            self.socket_path.unlink(missing_ok=True)
            return
        if not listener_clean:
            raise SandboxProviderSocketV2Error(
                "sandbox provider listener cleanup failed"
            )
        if not thread_clean:
            raise SandboxProviderSocketV2Error(
                "sandbox provider accept loop did not terminate"
            )
        if not handlers_clean:
            self._record_failure(
                "handler_drain",
                error=SandboxProviderSocketV2Error(
                    "sandbox provider request handlers did not drain"
                ),
            )
            raise SandboxProviderSocketV2Error(
                "sandbox provider request handlers did not drain "
                f"within {self._drain_timeout_seconds:.3f}s "
                f"(active={active_count})"
            )
        raise SandboxProviderSocketV2Error(
            "sandbox provider endpoint cleanup failed"
        )

    def _accept_loop(
        self,
        listener: Any,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if stop_event.is_set():
                    return
                if int(getattr(exc, "errno", 0) or 0) in _TRANSIENT_ACCEPT_ERRNOS:
                    continue
                self._record_failure("accept_loop", error=exc)
                return
            if stop_event.is_set():
                if not _shutdown_and_close_socket(connection):
                    self._retain_endpoint_cleanup_failure(
                        connection,
                        "sandbox",
                        None,
                    )
                return
            handler = threading.Thread(
                target=self._handle_registered,
                args=(connection,),
                name="leadpoet-sandbox-provider-request-v2",
                daemon=True,
            )
            with self._handler_condition:
                self._handlers.add(handler)
            try:
                handler.start()
            except Exception as exc:
                with self._handler_condition:
                    self._handlers.discard(handler)
                    self._handler_condition.notify_all()
                if not _shutdown_and_close_socket(connection):
                    self._retain_endpoint_cleanup_failure(
                        connection,
                        "sandbox",
                        exc,
                    )
                self._record_failure("start_handler", error=exc)

    def _handle_registered(self, connection: Any) -> None:
        try:
            self._handle(connection)
        finally:
            with self._handler_condition:
                self._handlers.discard(threading.current_thread())
                self._handler_condition.notify_all()

    def _handle(self, connection: Any) -> None:
        primary_error = None  # type: Optional[Exception]
        try:
            request = _normalize_request(_read_frame(connection))
            with self._transport.activate_scope(self._execution_scope):
                result = self._transport.execute_http(**request)
            _write_frame(connection, {"result": result})
        except Exception as exc:
            primary_error = exc
            try:
                _write_frame(
                    connection,
                    {
                        "status": "error",
                        "error_code": "sandbox_provider_%s"
                        % type(exc).__name__.lower()[:80],
                    },
                )
            except Exception:
                pass
        finally:
            if not _shutdown_and_close_socket(connection):
                self._retain_endpoint_cleanup_failure(
                    connection,
                    "sandbox",
                    primary_error,
                )
