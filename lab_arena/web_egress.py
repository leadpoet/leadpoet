"""Attempt-scoped, host-controlled public web egress for Arena sandboxes.

The sandbox sends ordinary HTTP proxy traffic over an attempt-unique Unix
socket.  This host service validates each destination before it opens either a
direct connection or an authenticated upstream HTTP(S) proxy connection.
Provider credentials and upstream proxy credentials never cross the socket.
Plain HTTP accepts one request with a fixed Content-Length and forces close;
chunked responses remain valid opaque response bytes. HTTPS stays opaque after
the host verifies its initial ClientHello server name.
"""

from __future__ import annotations

import errno
import ipaddress
import os
import re
import select
import socket
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from leadpoet_canonical.proxy_transport import (
    ProxyEndpoint,
    ProxyTransportCleanupError,
    connect_public_destination,
    global_address_infos,
    open_http_connect_tunnel,
    open_http_proxy_connection,
    parse_http_connect_proxy_url,
    shutdown_and_close_socket,
)


MAX_HEADER_BYTES = 64 * 1024
MAX_CLIENT_HELLO_BYTES = 64 * 1024
DEFAULT_MAX_CONNECTIONS = 32
DEFAULT_MAX_TOTAL_CONNECTIONS = 512
DEFAULT_MAX_BYTES_PER_DIRECTION = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 120.0
RELAY_CHUNK_BYTES = 64 * 1024
WEB_EGRESS_SUMMARY_SCHEMA_VERSION = "lab_arena.web_egress_summary.v1"
BLOCKED_EXACT_HOSTS = frozenset(
    {
        "0.0.0.0",
        "instance-data",
        "instance-data.ec2.internal",
        "localhost",
        "metadata.google.internal",
    }
)
BLOCKED_HOST_SUFFIXES = (
    ".internal",
    ".invalid",
    ".local",
    ".localhost",
    ".onion",
    ".test",
)
# These paid services stay behind the Arena broker and its cost ledger.
DEFAULT_PAID_PROVIDER_SUFFIXES = (
    "deepline.com",
    "exa.ai",
    "openrouter.ai",
    "scrapingdog.com",
)
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_HEADER_NAME_RE = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_METHOD_RE = re.compile(rb"^[A-Z]{1,16}$")
_PLAIN_HTTP_METHODS = frozenset(
    {b"DELETE", b"GET", b"HEAD", b"OPTIONS", b"PATCH", b"POST", b"PUT"}
)
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


class WebEgressError(RuntimeError):
    """A web request failed the host egress policy or transport."""


class _TunnelStarted(WebEgressError):
    """A CONNECT response was sent, so no later HTTP error can be emitted."""

    def __init__(self, primary_error: BaseException) -> None:
        super().__init__("web egress tunnel failed after CONNECT response")
        self.primary_error = primary_error


def _close_socket(candidate: Any) -> None:
    shutdown_and_close_socket(candidate)


def _normalize_host(host: Any, *, blocked_suffixes: Sequence[str]) -> str:
    raw = str(host or "").strip().rstrip(".")
    if not raw or any(character.isspace() for character in raw):
        raise WebEgressError("web egress destination host is invalid")
    try:
        normalized = raw.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError) as exc:
        raise WebEgressError("web egress destination host is invalid") from exc
    try:
        ipaddress.ip_address(normalized.strip("[]"))
    except ValueError:
        pass
    else:
        raise WebEgressError("web egress destination cannot be an IP literal")
    if normalized in BLOCKED_EXACT_HOSTS or any(
        normalized.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES
    ):
        raise WebEgressError("web egress destination host is blocked")
    if any(
        normalized == suffix or normalized.endswith("." + suffix)
        for suffix in blocked_suffixes
    ):
        raise WebEgressError("web egress paid provider destination is blocked")
    if not _HOST_RE.fullmatch(normalized):
        raise WebEgressError("web egress destination must be a DNS hostname")
    return normalized


def _read_headers(
    connection: Any, *, limit: int = MAX_HEADER_BYTES
) -> tuple[bytes, bytes]:
    buffered = bytearray()
    while b"\r\n\r\n" not in buffered:
        if len(buffered) >= limit:
            raise WebEgressError("web egress request headers exceed limit")
        chunk = connection.recv(min(4096, limit - len(buffered)))
        if not chunk:
            raise WebEgressError("web egress request headers are incomplete")
        buffered.extend(chunk)
    boundary = buffered.index(b"\r\n\r\n") + 4
    return bytes(buffered[:boundary]), bytes(buffered[boundary:])


def _parse_headers(
    encoded: bytes,
) -> tuple[bytes, bytes, bytes, list[tuple[bytes, bytes]]]:
    lines = encoded[:-4].split(b"\r\n")
    if not lines or len(lines[0]) > 8192:
        raise WebEgressError("web egress request line is invalid")
    parts = lines[0].split(b" ")
    if (
        len(parts) != 3
        or not _METHOD_RE.fullmatch(parts[0])
        or parts[2]
        not in {
            b"HTTP/1.0",
            b"HTTP/1.1",
        }
    ):
        raise WebEgressError("web egress request line is invalid")
    headers = []
    for line in lines[1:]:
        if not line or line[:1] in b" \t" or b":" not in line:
            raise WebEgressError("web egress request headers are malformed")
        name, value = line.split(b":", 1)
        if (
            not _HEADER_NAME_RE.fullmatch(name)
            or any(character < 0x20 and character != 0x09 for character in value)
            or b"\x7f" in value
        ):
            raise WebEgressError("web egress request headers are malformed")
        headers.append((name, value.strip(b" \t")))
    return parts[0], parts[1], parts[2], headers


def _single_header(
    headers: Sequence[tuple[bytes, bytes]], name: bytes
) -> Optional[bytes]:
    values = [value for key, value in headers if key.lower() == name]
    if len(values) > 1:
        raise WebEgressError("web egress request has duplicate authority headers")
    return values[0] if values else None


def _parse_authority(value: bytes, default_port: int) -> tuple[str, int]:
    if not value or b"@" in value or any(character < 0x21 for character in value):
        raise WebEgressError("web egress authority is invalid")
    try:
        parsed = urlsplit("//" + value.decode("ascii"))
        port = parsed.port if parsed.port is not None else default_port
    except (UnicodeDecodeError, ValueError) as exc:
        raise WebEgressError("web egress authority is invalid") from exc
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise WebEgressError("web egress authority is invalid")
    return parsed.hostname, port


def _validate_common_headers(
    headers: Sequence[tuple[bytes, bytes]], *, max_content_length: int
) -> int:
    names = [name.lower() for name, _value in headers]
    if b"proxy-authorization" in names:
        raise WebEgressError("sandbox proxy credentials are forbidden")
    if b"transfer-encoding" in names:
        raise WebEgressError("web egress transfer encoding is forbidden")
    content_lengths = [
        value for name, value in headers if name.lower() == b"content-length"
    ]
    if len(content_lengths) > 1:
        raise WebEgressError("web egress content length is ambiguous")
    if content_lengths:
        try:
            length = int(content_lengths[0])
        except ValueError as exc:
            raise WebEgressError("web egress content length is invalid") from exc
        if length < 0 or length > max_content_length:
            raise WebEgressError("web egress content length is invalid")
        return length
    return 0


def _rewrite_headers(
    headers: Sequence[tuple[bytes, bytes]], *, host: str, port: int
) -> list[tuple[bytes, bytes]]:
    output = []
    for name, value in headers:
        lowered = name.lower()
        if lowered in {b"connection", b"proxy-connection", b"keep-alive"}:
            continue
        output.append((name, value))
    output.append((b"Connection", b"close"))
    output.append((b"Proxy-Connection", b"close"))
    return output


def _encoded_request(
    method: bytes,
    target: bytes,
    version: bytes,
    headers: Sequence[tuple[bytes, bytes]],
) -> bytes:
    return (
        b" ".join((method, target, version))
        + b"\r\n"
        + b"".join(name + b": " + value + b"\r\n" for name, value in headers)
        + b"\r\n"
    )


def _recv_exact(connection: Any, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise WebEgressError("TLS ClientHello is incomplete")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_client_hello(body: bytes) -> str:
    if len(body) < 35:
        raise WebEgressError("TLS ClientHello is malformed")
    cursor = 34
    session_length = body[cursor]
    cursor += 1 + session_length
    if cursor + 2 > len(body):
        raise WebEgressError("TLS ClientHello is malformed")
    cipher_length = int.from_bytes(body[cursor : cursor + 2], "big")
    if cipher_length < 2 or cipher_length % 2:
        raise WebEgressError("TLS ClientHello cipher suites are malformed")
    cursor += 2 + cipher_length
    if cursor >= len(body):
        raise WebEgressError("TLS ClientHello is malformed")
    compression_length = body[cursor]
    if compression_length < 1:
        raise WebEgressError("TLS ClientHello compression methods are malformed")
    cursor += 1 + compression_length
    if cursor + 2 > len(body):
        raise WebEgressError("TLS ClientHello is malformed")
    extensions_length = int.from_bytes(body[cursor : cursor + 2], "big")
    cursor += 2
    if cursor + extensions_length != len(body):
        raise WebEgressError("TLS ClientHello is malformed")
    end = cursor + extensions_length
    server_names = []
    while cursor < end:
        if cursor + 4 > end:
            raise WebEgressError("TLS ClientHello extension is malformed")
        extension_type = int.from_bytes(body[cursor : cursor + 2], "big")
        extension_length = int.from_bytes(body[cursor + 2 : cursor + 4], "big")
        cursor += 4
        extension = body[cursor : cursor + extension_length]
        if len(extension) != extension_length:
            raise WebEgressError("TLS ClientHello extension is malformed")
        cursor += extension_length
        if extension_type != 0:
            continue
        if (
            len(extension) < 2
            or int.from_bytes(extension[:2], "big") != len(extension) - 2
        ):
            raise WebEgressError("TLS server name extension is malformed")
        name_cursor = 2
        while name_cursor < len(extension):
            if name_cursor + 3 > len(extension):
                raise WebEgressError("TLS server name extension is malformed")
            name_type = extension[name_cursor]
            name_length = int.from_bytes(
                extension[name_cursor + 1 : name_cursor + 3], "big"
            )
            name_cursor += 3
            encoded_name = extension[name_cursor : name_cursor + name_length]
            if len(encoded_name) != name_length:
                raise WebEgressError("TLS server name extension is malformed")
            name_cursor += name_length
            if name_type == 0:
                try:
                    server_names.append(
                        encoded_name.decode("ascii").lower().rstrip(".")
                    )
                except UnicodeDecodeError as exc:
                    raise WebEgressError("TLS server name is invalid") from exc
    if len(server_names) != 1:
        raise WebEgressError("TLS ClientHello must contain one server name")
    return server_names[0]


def _read_client_hello(connection: Any) -> tuple[bytes, str]:
    raw = bytearray()
    handshake = bytearray()
    expected_length = None
    while expected_length is None or len(handshake) < expected_length:
        header = _recv_exact(connection, 5)
        raw.extend(header)
        if header[0] != 22 or header[1] != 3 or header[2] > 4:
            raise WebEgressError("HTTPS tunnel must start with a TLS ClientHello")
        record_length = int.from_bytes(header[3:5], "big")
        if record_length < 1 or record_length > 18432:
            raise WebEgressError("TLS record length is invalid")
        record = _recv_exact(connection, record_length)
        raw.extend(record)
        handshake.extend(record)
        if len(raw) > MAX_CLIENT_HELLO_BYTES:
            raise WebEgressError("TLS ClientHello exceeds limit")
        if expected_length is None and len(handshake) >= 4:
            if handshake[0] != 1:
                raise WebEgressError("HTTPS tunnel must start with a TLS ClientHello")
            expected_length = 4 + int.from_bytes(handshake[1:4], "big")
            if expected_length > MAX_CLIENT_HELLO_BYTES:
                raise WebEgressError("TLS ClientHello exceeds limit")
    if len(handshake) != expected_length:
        # A record containing bytes after the first ClientHello would bypass
        # inspection or change replay semantics, so reject it.
        raise WebEgressError("TLS ClientHello record contains trailing data")
    return bytes(raw), _parse_client_hello(bytes(handshake[4:]))


def _relay(
    left: Any,
    right: Any,
    *,
    idle_timeout_seconds: float,
    max_bytes_per_direction: int,
    left_byte_limit: Optional[int] = None,
    left_charge: Optional[Callable[[int], None]] = None,
    right_charge: Optional[Callable[[int], None]] = None,
) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    transferred = {left: 0, right: 0}
    if left_byte_limit == 0:
        active.discard(left)
        try:
            right.shutdown(socket.SHUT_WR)
        except Exception:
            pass
    last_activity = time.monotonic()
    while active:
        remaining = idle_timeout_seconds - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise WebEgressError("web egress tunnel idle timeout")
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
                raise WebEgressError("web egress tunnel byte limit exceeded")
            if source is left and left_byte_limit is not None:
                if transferred[source] > left_byte_limit:
                    raise WebEgressError("plain HTTP body exceeds declared length")
            charge = left_charge if source is left else right_charge
            if charge is not None:
                charge(len(data))
            destination.sendall(data)
            last_activity = time.monotonic()
            if (
                source is left
                and left_byte_limit is not None
                and transferred[source] == left_byte_limit
            ):
                active.discard(left)
                try:
                    right.shutdown(socket.SHUT_WR)
                except Exception:
                    pass


class WebEgressServer:
    """One attempt-owned HTTP proxy endpoint on a host Unix socket."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        proxy_url: Optional[str] = None,
        *,
        blocked_hosts: Sequence[str] = DEFAULT_PAID_PROVIDER_SUFFIXES,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_total_connections: int = DEFAULT_MAX_TOTAL_CONNECTIONS,
        max_bytes_per_direction: int = DEFAULT_MAX_BYTES_PER_DIRECTION,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        resolver: Callable[..., Iterable[tuple[Any, ...]]] = socket.getaddrinfo,
        socket_factory: Callable[..., Any] = socket.socket,
        upstream_connector: Optional[Callable[[str, int], Any]] = None,
    ) -> None:
        self.path = Path(path)
        self._proxy_url = str(proxy_url) if proxy_url else None
        self._proxy_endpoint: Optional[ProxyEndpoint] = None
        self._blocked_hosts = tuple(
            str(value).lower().strip(".") for value in blocked_hosts
        )
        if not self.path.is_absolute() or len(os.fsencode(self.path)) > 103:
            raise WebEgressError("web egress socket path is invalid")
        if (
            max_connections < 1
            or max_total_connections < 1
            or max_bytes_per_direction < 1
            or max_total_bytes < 1
        ):
            raise WebEgressError("web egress resource limit is invalid")
        if connect_timeout_seconds <= 0 or idle_timeout_seconds <= 0:
            raise WebEgressError("web egress timeout is invalid")
        if self._proxy_url:
            self._proxy_endpoint = parse_http_connect_proxy_url(self._proxy_url)
        self._max_connections = int(max_connections)
        self._max_total_connections = int(max_total_connections)
        self._max_bytes = int(max_bytes_per_direction)
        self._max_total_bytes = int(max_total_bytes)
        self._connect_timeout = float(connect_timeout_seconds)
        self._idle_timeout = float(idle_timeout_seconds)
        self._resolver = resolver
        self._socket_factory = socket_factory
        self._upstream_connector = upstream_connector
        self._listener = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._connections: set[Any] = set()
        self._workers: set[threading.Thread] = set()
        self._slots = threading.BoundedSemaphore(self._max_connections)
        self._last_failure: Optional[dict[str, Any]] = None
        self._accepted_connection_count = 0
        self._active_limit_rejection_count = 0
        self._total_limit_rejection_count = 0
        self._client_to_web_bytes = 0
        self._web_to_client_bytes = 0
        self._byte_limit_rejection_count = 0
        self._failure_count = 0
        self._cleanup_block_rejection_count = 0
        self._pending_cleanup: dict[int, Any] = {}

    @property
    def running(self) -> bool:
        return bool(
            not self._stop.is_set()
            and self._thread
            and self._thread.is_alive()
            and self._listener is not None
        )

    @property
    def proxy_url(self) -> Optional[str]:
        """The immutable attempt proxy assignment; only the host can read it."""

        return self._proxy_url

    @property
    def last_failure(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._last_failure) if self._last_failure else None

    @property
    def summary(self) -> dict[str, Any]:
        """Return bounded attempt totals without destinations or credentials."""

        with self._lock:
            return {
                "schema_version": WEB_EGRESS_SUMMARY_SCHEMA_VERSION,
                "running": self.running,
                "accepted_connection_count": self._accepted_connection_count,
                "active_connection_count": len(self._workers),
                "active_limit_rejection_count": self._active_limit_rejection_count,
                "total_limit_rejection_count": self._total_limit_rejection_count,
                "client_to_web_bytes": self._client_to_web_bytes,
                "web_to_client_bytes": self._web_to_client_bytes,
                "byte_limit_rejection_count": self._byte_limit_rejection_count,
                "failure_count": self._failure_count,
                "cleanup_block_rejection_count": self._cleanup_block_rejection_count,
                "pending_cleanup_count": len(self._pending_cleanup),
            }

    def _record_failure(self, stage: str, error: BaseException) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure = {
                "stage": stage,
                "error_type": type(error).__name__,
                "errno": int(getattr(error, "errno", 0) or 0),
            }

    def _charge_bytes(self, direction: str, amount: int) -> None:
        if amount < 0:
            raise WebEgressError("web egress byte accounting is invalid")
        with self._lock:
            total = self._client_to_web_bytes + self._web_to_client_bytes
            if total + amount > self._max_total_bytes:
                self._byte_limit_rejection_count += 1
                raise WebEgressError("web egress attempt byte limit exceeded")
            if direction == "client_to_web":
                self._client_to_web_bytes += amount
            elif direction == "web_to_client":
                self._web_to_client_bytes += amount
            else:
                raise WebEgressError("web egress byte accounting is invalid")

    def _charge_client_to_web(self, amount: int) -> None:
        self._charge_bytes("client_to_web", amount)

    def _charge_web_to_client(self, amount: int) -> None:
        self._charge_bytes("web_to_client", amount)

    def _retain_transport_cleanup(self, error: BaseException) -> None:
        candidate = error.primary_error if isinstance(error, _TunnelStarted) else error
        if not isinstance(candidate, ProxyTransportCleanupError):
            return
        with self._lock:
            for resource in candidate.resources:
                self._pending_cleanup[id(resource)] = resource

    def _retry_pending_cleanup(self) -> bool:
        with self._lock:
            snapshot = tuple(self._pending_cleanup.items())
        cleaned = []
        for resource_id, resource in snapshot:
            if shutdown_and_close_socket(resource):
                cleaned.append((resource_id, resource))
        with self._lock:
            for resource_id, resource in cleaned:
                if self._pending_cleanup.get(resource_id) is resource:
                    self._pending_cleanup.pop(resource_id, None)
            return not self._pending_cleanup

    def _release_owned_socket(self, candidate: Any) -> None:
        if candidate is None:
            return
        cleaned = shutdown_and_close_socket(candidate)
        with self._lock:
            self._connections.discard(candidate)
            if not cleaned:
                self._pending_cleanup[id(candidate)] = candidate

    def start(self) -> "WebEgressServer":
        with self._lock:
            if self.running:
                return self
            if self.path.exists() or self.path.is_symlink():
                raise WebEgressError("web egress socket path already exists")
            parent = self.path.parent
            metadata = parent.stat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise WebEgressError("web egress socket parent is invalid")
            listener = self._socket_factory(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(self.path))
                os.chmod(self.path, 0o600)
                listener.listen(self._max_connections)
                listener.settimeout(0.5)
            except BaseException:
                _close_socket(listener)
                try:
                    if self.path.is_socket():
                        self.path.unlink()
                except OSError:
                    pass
                raise
            self._listener = listener
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._serve, name="arena-web-egress", daemon=True
            )
            self._thread.start()
        return self

    def _serve(self) -> None:
        listener = self._listener
        while not self._stop.is_set():
            try:
                client, _address = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._record_failure("accept", exc)
                break
            if not self._retry_pending_cleanup():
                with self._lock:
                    self._cleanup_block_rejection_count += 1
                try:
                    client.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n"
                    )
                except OSError:
                    pass
                _close_socket(client)
                continue
            with self._lock:
                if self._accepted_connection_count >= self._max_total_connections:
                    self._total_limit_rejection_count += 1
                    total_limit_reached = True
                else:
                    self._accepted_connection_count += 1
                    total_limit_reached = False
            if total_limit_reached:
                try:
                    client.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n"
                    )
                except OSError:
                    pass
                _close_socket(client)
                continue
            if not self._slots.acquire(blocking=False):
                with self._lock:
                    self._active_limit_rejection_count += 1
                try:
                    client.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n"
                    )
                except OSError:
                    pass
                _close_socket(client)
                continue
            worker = threading.Thread(
                target=self._run_client,
                args=(client,),
                name="arena-web-egress-client",
                daemon=True,
            )
            with self._lock:
                self._connections.add(client)
                self._workers.add(worker)
            worker.start()

    def _run_client(self, client: Any) -> None:
        upstream = None
        try:
            client.settimeout(self._connect_timeout)
            upstream = self._handle_client(client)
        except _TunnelStarted as exc:
            self._retain_transport_cleanup(exc)
            self._record_failure("tunnel", exc.primary_error)
        except BaseException as exc:
            self._retain_transport_cleanup(exc)
            self._record_failure("request", exc)
            try:
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
        finally:
            self._release_owned_socket(upstream)
            self._release_owned_socket(client)
            with self._lock:
                self._workers.discard(threading.current_thread())
            self._slots.release()

    def _resolved_ip(self, host: str, port: int) -> str:
        infos = global_address_infos(host, port, resolver=self._resolver)
        return str(infos[0][4][0])

    def _connect_endpoint(self, host: str, port: int) -> Any:
        if self._upstream_connector is not None:
            return self._upstream_connector(host, port)
        return connect_public_destination(
            host,
            port,
            resolver=self._resolver,
            socket_factory=self._socket_factory,
            timeout_seconds=self._connect_timeout,
        )

    def _open_proxy_stream(self) -> tuple[Any, Optional[str]]:
        assert self._proxy_endpoint is not None
        stream, authorization = open_http_proxy_connection(
            self._proxy_endpoint,
            connector=self._connect_endpoint,
            timeout_seconds=self._connect_timeout,
        )
        with self._lock:
            self._connections.add(stream)
        return stream, authorization

    def _proxy_connect(self, destination_ip: str, port: int) -> Any:
        assert self._proxy_endpoint is not None
        stream = open_http_connect_tunnel(
            self._proxy_endpoint,
            destination_ip,
            port,
            connector=self._connect_endpoint,
            timeout_seconds=self._connect_timeout,
        )
        with self._lock:
            self._connections.add(stream)
        return stream

    def _handle_client(self, client: Any) -> Any:
        encoded, remainder = _read_headers(client)
        method, target, version, headers = _parse_headers(encoded)
        content_length = _validate_common_headers(
            headers, max_content_length=self._max_bytes
        )
        if method == b"CONNECT":
            if remainder or content_length:
                raise WebEgressError("CONNECT request contains a body")
            raw_host, port = _parse_authority(target, 443)
            if port != 443:
                raise WebEgressError("web egress CONNECT port is blocked")
            host = _normalize_host(raw_host, blocked_suffixes=self._blocked_hosts)
            host_header = _single_header(headers, b"host")
            if host_header is not None:
                header_host, header_port = _parse_authority(host_header, 443)
                if header_port != port or header_host.lower().rstrip(".") != host:
                    raise WebEgressError("CONNECT Host header does not match authority")
            destination_ip = self._resolved_ip(host, port)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            try:
                raw_hello, server_name = _read_client_hello(client)
                if server_name != host:
                    raise WebEgressError(
                        "TLS server name does not match CONNECT authority"
                    )
                _normalize_host(server_name, blocked_suffixes=self._blocked_hosts)
                upstream = None
                try:
                    self._charge_client_to_web(len(raw_hello))
                    upstream = (
                        self._proxy_connect(destination_ip, port)
                        if self._proxy_url
                        else self._connect_endpoint(destination_ip, port)
                    )
                    if not self._proxy_url:
                        with self._lock:
                            self._connections.add(upstream)
                    upstream.sendall(raw_hello)
                    client.settimeout(None)
                    _relay(
                        client,
                        upstream,
                        idle_timeout_seconds=self._idle_timeout,
                        max_bytes_per_direction=self._max_bytes,
                        left_charge=self._charge_client_to_web,
                        right_charge=self._charge_web_to_client,
                    )
                finally:
                    self._release_owned_socket(upstream)
            except BaseException as exc:
                raise _TunnelStarted(exc) from exc
            return None

        if method not in _PLAIN_HTTP_METHODS:
            raise WebEgressError("plain HTTP method is blocked")
        try:
            if (
                not target
                or b"\\" in target
                or any(character < 0x21 or character > 0x7E for character in target)
            ):
                raise WebEgressError("plain HTTP request target is invalid")
            parsed = urlsplit(target.decode("ascii"))
            parsed_port = parsed.port
        except (UnicodeDecodeError, ValueError) as exc:
            raise WebEgressError("plain HTTP request target is invalid") from exc
        if (
            parsed.scheme.lower() != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise WebEgressError("plain HTTP requires an absolute HTTP URL")
        port = parsed_port if parsed_port is not None else 80
        if port != 80:
            raise WebEgressError("plain HTTP destination port is blocked")
        host = _normalize_host(parsed.hostname, blocked_suffixes=self._blocked_hosts)
        host_header = _single_header(headers, b"host")
        if host_header is None:
            raise WebEgressError("plain HTTP Host header is required")
        header_host, header_port = _parse_authority(host_header, 80)
        if header_port != port or header_host.lower().rstrip(".") != host:
            raise WebEgressError("plain HTTP Host header does not match request URL")
        destination_ip = self._resolved_ip(host, port)
        if len(remainder) > content_length:
            raise WebEgressError("plain HTTP body exceeds declared length")
        rewritten_headers = _rewrite_headers(headers, host=host, port=port)
        upstream = None
        try:
            if self._proxy_url:
                assert self._proxy_endpoint is not None
                authorization = self._proxy_endpoint.authorization_header
                netloc = (
                    ("[%s]" % destination_ip)
                    if ":" in destination_ip
                    else destination_ip
                )
                target_out = urlunsplit(
                    ("http", netloc, parsed.path or "/", parsed.query, "")
                ).encode("ascii")
                if authorization:
                    rewritten_headers.append(
                        (b"Proxy-Authorization", authorization.encode("ascii"))
                    )
            else:
                target_out = urlunsplit(
                    ("", "", parsed.path or "/", parsed.query, "")
                ).encode("ascii")
            outbound_request = (
                _encoded_request(method, target_out, version, rewritten_headers)
                + remainder
            )
            # Account sandbox bytes, not the host-added Proxy-Authorization
            # header length. The aggregate summary must not reveal even an
            # approximate credential length.
            self._charge_client_to_web(len(encoded) + len(remainder))
            if self._proxy_url:
                upstream, _authorization = self._open_proxy_stream()
            else:
                upstream = self._connect_endpoint(destination_ip, port)
                with self._lock:
                    self._connections.add(upstream)
            upstream.sendall(outbound_request)
            client.settimeout(None)
            upstream.settimeout(None)
            _relay(
                client,
                upstream,
                idle_timeout_seconds=self._idle_timeout,
                max_bytes_per_direction=self._max_bytes,
                left_byte_limit=content_length - len(remainder),
                left_charge=self._charge_client_to_web,
                right_charge=self._charge_web_to_client,
            )
        finally:
            self._release_owned_socket(upstream)
        return None

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            listener = self._listener
            thread = self._thread
        self._release_owned_socket(listener)
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # The accept loop must finish before this snapshot. Otherwise it could
        # add a client after the first close pass and let that client retain the
        # attempt's route after this context exits.
        with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            self._release_owned_socket(connection)
        deadline = time.monotonic() + self._connect_timeout + 2.0
        while True:
            with self._lock:
                workers = tuple(self._workers)
            if not workers or time.monotonic() >= deadline:
                break
            for worker in workers:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
        cleanup_complete = self._retry_pending_cleanup()
        with self._lock:
            accept_loop_alive = bool(thread and thread.is_alive())
            workers_remain = bool(self._workers)
            if self._listener is listener:
                self._listener = None
            if self._thread is thread and not accept_loop_alive:
                self._thread = None
        try:
            if self.path.is_socket():
                self.path.unlink()
        except OSError as exc:
            self._record_failure("socket_cleanup", exc)
        if accept_loop_alive or workers_remain or not cleanup_complete:
            error = WebEgressError("web egress transport cleanup is incomplete")
            self._record_failure("transport_cleanup", error)
            raise error

    def __enter__(self) -> "WebEgressServer":
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop()


__all__ = [
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_MAX_TOTAL_CONNECTIONS",
    "DEFAULT_PAID_PROVIDER_SUFFIXES",
    "WEB_EGRESS_SUMMARY_SCHEMA_VERSION",
    "WebEgressError",
    "WebEgressServer",
]
