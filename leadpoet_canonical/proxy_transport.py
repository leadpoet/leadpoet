"""Shared standard-library transport for authenticated public proxy routes."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import ipaddress
import re
import socket
import ssl
import time
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlsplit


DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
DEFAULT_PROXY_TIMEOUT_SECONDS = 5.0
MAX_PROXY_HEADER_BYTES = 64 * 1024
_BLOCKED_EXACT_HOSTS = frozenset(
    (
        "0.0.0.0",
        "instance-data",
        "instance-data.ec2.internal",
        "localhost",
        "metadata.google.internal",
    )
)
_BLOCKED_SUFFIXES = (
    ".internal",
    ".invalid",
    ".local",
    ".localhost",
    ".onion",
    ".test",
)
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)


class ProxyTransportError(RuntimeError):
    """A proxy endpoint or public network operation is invalid."""


class ProxyTransportCleanupError(ProxyTransportError):
    """A failed operation still owns transport resources."""

    def __init__(
        self,
        *,
        stage: str,
        primary_error: BaseException,
        resources: tuple[Any, ...],
    ) -> None:
        super().__init__("proxy transport cleanup failed")
        self.stage = str(stage)
        self.primary_error = primary_error
        self.resources = tuple(
            resource for resource in resources if resource is not None
        )


@dataclass(frozen=True)
class ProxyEndpoint:
    """One parsed proxy endpoint with its authorization hidden from repr."""

    scheme: str
    host: str
    port: int
    authorization_header: str | None = field(default=None, repr=False)


def _normalized_public_host(host: Any) -> str:
    raw_host = str(host or "").strip().rstrip(".")
    if not raw_host or any(character.isspace() for character in raw_host):
        raise ProxyTransportError("egress proxy host is invalid")
    try:
        address = ipaddress.ip_address(raw_host.strip("[]"))
    except ValueError:
        try:
            normalized_host = raw_host.encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError) as exc:
            raise ProxyTransportError("egress proxy host is invalid") from exc
        if normalized_host in _BLOCKED_EXACT_HOSTS:
            raise ProxyTransportError("egress destination host is blocked")
        if any(normalized_host.endswith(suffix) for suffix in _BLOCKED_SUFFIXES):
            raise ProxyTransportError("egress destination suffix is blocked")
        if not _HOST_RE.fullmatch(normalized_host):
            raise ProxyTransportError("egress destination must be a DNS hostname")
        return normalized_host
    if not address.is_global:
        raise ProxyTransportError(
            "egress proxy IP literal is not globally routable"
        )
    return address.compressed.lower()


def normalize_proxy_destination(host: Any, port: Any) -> tuple[str, int]:
    """Return one global DNS or IP destination and a valid TCP port."""

    normalized_host = _normalized_public_host(host)
    try:
        normalized_port = int(port)
    except (TypeError, ValueError) as exc:
        raise ProxyTransportError("egress proxy port is invalid") from exc
    if not 1 <= normalized_port <= 65535:
        raise ProxyTransportError("egress proxy port is invalid")
    return normalized_host, normalized_port


def parse_http_connect_proxy_url(value: str) -> ProxyEndpoint:
    """Parse one HTTP CONNECT or HTTPS proxy URL without retaining the URL."""

    normalized = str(value or "")
    try:
        parsed = urlsplit(normalized)
        scheme = parsed.scheme.lower()
        parsed_port = parsed.port
        port = (
            parsed_port
            if parsed_port is not None
            else (443 if scheme == "https" else 80)
        )
    except ValueError as exc:
        raise ProxyTransportError("worker egress proxy port is invalid") from exc
    if (
        scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ProxyTransportError(
            "worker egress proxy must be an HTTP CONNECT or HTTPS proxy URL"
        )
    try:
        host, port = normalize_proxy_destination(parsed.hostname, port)
    except ProxyTransportError as exc:
        raise ProxyTransportError(
            "worker egress proxy destination is invalid"
        ) from exc
    if (parsed.username is None) != (parsed.password is None):
        raise ProxyTransportError("worker egress proxy credentials are incomplete")

    authorization_header = None
    if parsed.username is not None:
        username = unquote(parsed.username)
        password = unquote(parsed.password or "")
        if any(character in username + password for character in "\x00\r\n"):
            raise ProxyTransportError("worker egress proxy credentials are invalid")
        token = base64.b64encode(
            (username + ":" + password).encode("utf-8")
        ).decode("ascii")
        authorization_header = "Basic " + token
    return ProxyEndpoint(
        scheme=scheme,
        host=host,
        port=port,
        authorization_header=authorization_header,
    )


def validate_http_connect_proxy_url(value: str) -> str:
    """Validate one proxy URL and return it unchanged for secret transport."""

    normalized = str(value or "")
    parse_http_connect_proxy_url(normalized)
    return normalized


def shutdown_and_close_socket(candidate: Any) -> bool:
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


def global_address_infos(
    host: str,
    port: int,
    *,
    resolver: Callable[..., Iterable[tuple[Any, ...]]] = socket.getaddrinfo,
) -> list[tuple[Any, ...]]:
    """Resolve a public endpoint and reject the full set on any private answer."""

    normalized_host, normalized_port = normalize_proxy_destination(host, port)
    try:
        infos = list(
            resolver(normalized_host, normalized_port, type=socket.SOCK_STREAM)
        )
    except Exception as exc:
        raise ProxyTransportError(
            "egress destination DNS resolution failed"
        ) from exc
    usable = []
    observed_addresses = set()
    for info in infos:
        if len(info) != 5:
            continue
        family, socktype, protocol, _canonical_name, sockaddr = info
        if (
            socktype != socket.SOCK_STREAM
            or not isinstance(sockaddr, tuple)
            or not sockaddr
        ):
            continue
        address = str(sockaddr[0])
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ProxyTransportError(
                "egress DNS returned an invalid address"
            ) from exc
        if not parsed.is_global:
            raise ProxyTransportError(
                "egress DNS returned a non-global address"
            )
        key = (family, protocol, sockaddr)
        if key not in observed_addresses:
            observed_addresses.add(key)
            usable.append((family, socktype, protocol, "", sockaddr))
    if not usable:
        raise ProxyTransportError("egress destination has no global address")
    return usable


def connect_public_destination(
    host: str,
    port: int,
    *,
    resolver: Callable[..., Iterable[tuple[Any, ...]]] = socket.getaddrinfo,
    socket_factory: Callable[..., Any] = socket.socket,
    timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> Any:
    """Open one bounded socket only through globally routable DNS answers."""

    timeout = _bounded_timeout(timeout_seconds)
    last_error: BaseException | None = None
    for family, socktype, protocol, _canonical_name, sockaddr in global_address_infos(
        host,
        port,
        resolver=resolver,
    ):
        candidate = socket_factory(family, socktype, protocol)
        try:
            candidate.settimeout(timeout)
            candidate.connect(sockaddr)
            candidate.settimeout(None)
            return candidate
        except BaseException as exc:
            last_error = exc
            if not shutdown_and_close_socket(candidate):
                raise ProxyTransportCleanupError(
                    stage="public_destination_cleanup",
                    primary_error=exc,
                    resources=(candidate,),
                ) from exc
    raise ProxyTransportError("egress destination connection failed") from last_error


def _bounded_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ProxyTransportError("proxy transport timeout is invalid") from exc
    if not 0 < timeout <= 30:
        raise ProxyTransportError("proxy transport timeout is invalid")
    return timeout


def _endpoint(value: str | ProxyEndpoint) -> ProxyEndpoint:
    if isinstance(value, ProxyEndpoint):
        return value
    return parse_http_connect_proxy_url(value)


def open_http_proxy_connection(
    proxy: str | ProxyEndpoint,
    *,
    connector: Callable[[str, int], Any] = connect_public_destination,
    timeout_seconds: float = DEFAULT_PROXY_TIMEOUT_SECONDS,
    tls_context_factory: Callable[[], ssl.SSLContext] = ssl.create_default_context,
) -> tuple[Any, str | None]:
    """Open a proxy stream for CONNECT or absolute-form HTTP requests."""

    endpoint = _endpoint(proxy)
    timeout = _bounded_timeout(timeout_seconds)
    stream = None
    operation_error: BaseException | None = None
    try:
        stream = connector(endpoint.host, endpoint.port)
        stream.settimeout(timeout)
        if endpoint.scheme == "https":
            stream = tls_context_factory().wrap_socket(
                stream,
                server_hostname=endpoint.host,
            )
        return stream, endpoint.authorization_header
    except BaseException as exc:
        operation_error = exc
    if stream is not None and not shutdown_and_close_socket(stream):
        raise ProxyTransportCleanupError(
            stage="proxy_connection_cleanup",
            primary_error=operation_error or ProxyTransportError(
                "proxy connection failed"
            ),
            resources=(stream,),
        ) from operation_error
    if isinstance(operation_error, ProxyTransportError):
        raise operation_error
    if stream is None and operation_error is not None:
        # An injected connector can carry an owner-specific cleanup exception.
        # Preserve it so its caller can recover the retained resource.
        raise operation_error
    raise ProxyTransportError("proxy connection failed") from operation_error


def _read_proxy_headers(connection: Any) -> tuple[bytes, bytes]:
    buffer = bytearray()
    marker = b"\r\n\r\n"
    while marker not in buffer:
        if len(buffer) >= MAX_PROXY_HEADER_BYTES:
            raise ProxyTransportError("upstream proxy response headers exceed limit")
        chunk = connection.recv(
            min(16 * 1024, MAX_PROXY_HEADER_BYTES - len(buffer))
        )
        if not chunk:
            raise ProxyTransportError(
                "upstream proxy closed before CONNECT response"
            )
        buffer.extend(chunk)
    header_end = buffer.index(marker) + len(marker)
    return bytes(buffer[:header_end]), bytes(buffer[header_end:])


def _authority(host: str, port: int) -> str:
    display_host = "[%s]" % host if ":" in host else host
    return "%s:%d" % (display_host, port)


def open_http_connect_tunnel(
    proxy: str | ProxyEndpoint,
    destination_host: str,
    destination_port: int,
    *,
    connector: Callable[[str, int], Any] = connect_public_destination,
    timeout_seconds: float = DEFAULT_PROXY_TIMEOUT_SECONDS,
    tls_context_factory: Callable[[], ssl.SSLContext] = ssl.create_default_context,
) -> Any:
    """Open an authenticated CONNECT stream to a public DNS or IP target."""

    target_host, target_port = normalize_proxy_destination(
        destination_host,
        destination_port,
    )
    stream = None
    operation_error: BaseException | None = None
    try:
        stream, authorization = open_http_proxy_connection(
            proxy,
            connector=connector,
            timeout_seconds=timeout_seconds,
            tls_context_factory=tls_context_factory,
        )
        authority = _authority(target_host, target_port)
        lines = [
            "CONNECT %s HTTP/1.1" % authority,
            "Host: %s" % authority,
            "Proxy-Connection: Keep-Alive",
        ]
        if authorization is not None:
            lines.append("Proxy-Authorization: " + authorization)
        stream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("iso-8859-1"))
        response_headers, remainder = _read_proxy_headers(stream)
        status_line = response_headers.split(b"\r\n", 1)[0]
        parts = status_line.split(b" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise ProxyTransportError("upstream proxy response is malformed")
        status = int(parts[1])
        if status < 200 or status >= 300:
            raise ProxyTransportError(
                "upstream proxy CONNECT failed with HTTP status %d" % status
            )
        if remainder:
            raise ProxyTransportError(
                "upstream proxy returned unexpected CONNECT payload"
            )
        return stream
    except BaseException as exc:
        operation_error = exc
    if stream is not None and not shutdown_and_close_socket(stream):
        raise ProxyTransportCleanupError(
            stage="proxy_connect_cleanup",
            primary_error=operation_error or ProxyTransportError(
                "upstream proxy CONNECT failed"
            ),
            resources=(stream,),
        ) from operation_error
    if isinstance(operation_error, ProxyTransportCleanupError):
        raise operation_error
    if isinstance(operation_error, ProxyTransportError):
        raise operation_error
    if stream is None and operation_error is not None:
        raise operation_error
    raise ProxyTransportError("upstream proxy CONNECT failed") from operation_error


def verify_tls_proxy_connect(
    proxy_url: str,
    *,
    destination_host: str,
    destination_port: int = 443,
    attempts: int = 2,
    timeout_seconds: float = DEFAULT_PROXY_TIMEOUT_SECONDS,
    connector: Callable[[str, int], Any] = connect_public_destination,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Verify an authenticated CONNECT handshake and prove stream cleanup."""

    validate_http_connect_proxy_url(proxy_url)
    normalized_attempts = max(1, int(attempts))
    last_error: BaseException | None = None
    for attempt in range(normalized_attempts):
        stream = None
        try:
            stream = open_http_connect_tunnel(
                proxy_url,
                destination_host,
                destination_port,
                connector=connector,
                timeout_seconds=timeout_seconds,
            )
        except BaseException as exc:
            last_error = exc
        if stream is not None:
            if not shutdown_and_close_socket(stream):
                raise ProxyTransportCleanupError(
                    stage="proxy_connect_verification_cleanup",
                    primary_error=last_error or ProxyTransportError(
                        "proxy CONNECT stream cleanup failed"
                    ),
                    resources=(stream,),
                )
            return
        if isinstance(last_error, ProxyTransportCleanupError):
            raise last_error
        if attempt + 1 < normalized_attempts:
            sleep(0.2)
    raise ProxyTransportError(
        "worker proxy failed authenticated CONNECT preflight"
    ) from last_error
