"""Credential-free sandbox loopback bridge to an Arena host egress socket.

This module uses only the Python standard library so it can be mounted read
only into an isolated sandbox without mounting the Arena service package.
"""

from __future__ import annotations

import argparse
import errno
import os
import select
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional


WEB_EGRESS_SOCKET_ENV = "LAB_ARENA_WEB_EGRESS_SOCKET"
WEB_EGRESS_PROXY_ENV = "LAB_ARENA_WEB_PROXY_URL"
DEFAULT_MAX_CONNECTIONS = 32
DEFAULT_MAX_BYTES_PER_DIRECTION = 64 * 1024 * 1024
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 120.0
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


class LoopbackWebEgressBridgeError(RuntimeError):
    """The sandbox bridge could not start or relay within its limits."""


def _close_socket(candidate: Any) -> None:
    if candidate is None:
        return
    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        candidate.close()
    except Exception:
        pass


def _relay(
    left: Any,
    right: Any,
    *,
    idle_timeout_seconds: float,
    max_bytes_per_direction: int,
) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    transferred = {left: 0, right: 0}
    last_activity = time.monotonic()
    while active:
        remaining = idle_timeout_seconds - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise LoopbackWebEgressBridgeError("bridge idle timeout")
        readable, _writable, _exceptional = select.select(
            list(active), [], [], min(1.0, remaining)
        )
        for source in readable:
            try:
                data = source.recv(RELAY_CHUNK_BYTES)
            except OSError as exc:
                if exc.errno not in _PEER_CLOSE_ERRNOS:
                    raise
                data = b""
            destination = peers[source]
            if not data:
                active.discard(source)
                try:
                    destination.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            transferred[source] += len(data)
            if transferred[source] > max_bytes_per_direction:
                raise LoopbackWebEgressBridgeError("bridge byte limit exceeded")
            destination.sendall(data)
            last_activity = time.monotonic()


class LoopbackWebEgressBridge:
    """Forward a loopback-only TCP proxy endpoint to one host Unix socket."""

    def __init__(
        self,
        uds_path: os.PathLike[str] | str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_bytes_per_direction: int = DEFAULT_MAX_BYTES_PER_DIRECTION,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self.uds_path = Path(uds_path)
        if not self.uds_path.is_absolute() or len(os.fsencode(self.uds_path)) > 103:
            raise LoopbackWebEgressBridgeError("bridge Unix socket path is invalid")
        if host != "127.0.0.1":
            raise LoopbackWebEgressBridgeError("bridge must bind IPv4 loopback")
        if not 0 <= int(port) <= 65535:
            raise LoopbackWebEgressBridgeError("bridge port is invalid")
        if max_connections < 1 or max_bytes_per_direction < 1:
            raise LoopbackWebEgressBridgeError("bridge resource limit is invalid")
        if connect_timeout_seconds <= 0 or idle_timeout_seconds <= 0:
            raise LoopbackWebEgressBridgeError("bridge timeout is invalid")
        self.host = host
        self.port = int(port)
        self._max_connections = int(max_connections)
        self._max_bytes = int(max_bytes_per_direction)
        self._connect_timeout = float(connect_timeout_seconds)
        self._idle_timeout = float(idle_timeout_seconds)
        self._listener = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._connections: set[Any] = set()
        self._workers: set[threading.Thread] = set()
        self._slots = threading.BoundedSemaphore(self._max_connections)
        self._last_failure: Optional[dict[str, Any]] = None

    @property
    def running(self) -> bool:
        return bool(
            not self._stop.is_set()
            and self._thread
            and self._thread.is_alive()
            and self._listener is not None
        )

    @property
    def proxy_url(self) -> str:
        if not self.running or not self.port:
            raise LoopbackWebEgressBridgeError("bridge is not running")
        return "http://127.0.0.1:%d" % self.port

    @property
    def proxy_environment(self) -> dict[str, str]:
        proxy_url = self.proxy_url
        return {
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "http_proxy": proxy_url,
            "https_proxy": proxy_url,
            "NO_PROXY": "",
            "no_proxy": "",
            WEB_EGRESS_PROXY_ENV: proxy_url,
        }

    @property
    def last_failure(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._last_failure) if self._last_failure else None

    def _record_failure(self, stage: str, error: BaseException) -> None:
        with self._lock:
            self._last_failure = {
                "stage": stage,
                "error_type": type(error).__name__,
                "errno": int(getattr(error, "errno", 0) or 0),
            }

    def start(self) -> "LoopbackWebEgressBridge":
        with self._lock:
            if self.running:
                return self
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.host, self.port))
                listener.listen(self._max_connections)
                listener.settimeout(0.5)
            except BaseException:
                _close_socket(listener)
                raise
            self.port = int(listener.getsockname()[1])
            self._listener = listener
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._serve, name="arena-web-egress-bridge", daemon=True
            )
            self._thread.start()
        return self

    def _serve(self) -> None:
        listener = self._listener
        while not self._stop.is_set():
            try:
                client, address = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._record_failure("accept", exc)
                break
            if not isinstance(address, tuple) or address[0] != "127.0.0.1":
                _close_socket(client)
                continue
            if not self._slots.acquire(blocking=False):
                _close_socket(client)
                continue
            worker = threading.Thread(
                target=self._run_client,
                args=(client,),
                name="arena-web-egress-bridge-client",
                daemon=True,
            )
            with self._lock:
                self._connections.add(client)
                self._workers.add(worker)
            worker.start()

    def _run_client(self, client: Any) -> None:
        host_socket = None
        try:
            host_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            host_socket.settimeout(self._connect_timeout)
            host_socket.connect(str(self.uds_path))
            host_socket.settimeout(None)
            with self._lock:
                self._connections.add(host_socket)
            _relay(
                client,
                host_socket,
                idle_timeout_seconds=self._idle_timeout,
                max_bytes_per_direction=self._max_bytes,
            )
        except BaseException as exc:
            self._record_failure("relay", exc)
        finally:
            _close_socket(host_socket)
            _close_socket(client)
            with self._lock:
                self._connections.discard(host_socket)
                self._connections.discard(client)
                self._workers.discard(threading.current_thread())
            self._slots.release()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            listener = self._listener
            thread = self._thread
        _close_socket(listener)
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # End acceptance before taking the final connection snapshot. This
        # prevents a just-accepted relay from surviving the bridge context.
        with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            _close_socket(connection)
        deadline = time.monotonic() + self._connect_timeout + 2.0
        while True:
            with self._lock:
                workers = tuple(self._workers)
            if not workers or time.monotonic() >= deadline:
                break
            for worker in workers:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            workers_remain = bool(self._workers)
            if self._listener is listener:
                self._listener = None
            if self._thread is thread:
                self._thread = None
        if workers_remain:
            raise LoopbackWebEgressBridgeError("bridge transport cleanup is incomplete")

    def __enter__(self) -> "LoopbackWebEgressBridge":
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arena sandbox web egress bridge")
    parser.add_argument(
        "--uds-path",
        default=os.environ.get(WEB_EGRESS_SOCKET_ENV, ""),
        required=not bool(os.environ.get(WEB_EGRESS_SOCKET_ENV)),
    )
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args(argv)
    bridge = LoopbackWebEgressBridge(args.uds_path, port=args.port).start()
    try:
        # The parent chooses the port before launch. Do not print paths or any
        # host configuration from the long-running sandbox process.
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        bridge.stop()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LoopbackWebEgressBridge",
    "LoopbackWebEgressBridgeError",
    "WEB_EGRESS_PROXY_ENV",
    "WEB_EGRESS_SOCKET_ENV",
    "main",
]
