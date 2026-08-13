"""Enclave-local HTTP proxy for provider and public-web scoring traffic.

HTTPS clients terminate TLS and validate certificates inside the enclave.  The
parent receives only a destination handshake followed by opaque TLS bytes.
External plaintext HTTP is rejected. Loopback HTTP remains available for the
local proxy endpoint and other services inside the measured enclave.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import select
import socket
import ssl
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import unquote, urlsplit

from gateway.tee.egress_framing import (
    TUNNEL_FRAMING_HEADER,
    TUNNEL_FRAMING_MODE,
    relay_raw_and_framed,
    send_tunnel_frame,
)
from gateway.tee.egress_policy import (
    destination_policy_hash,
    normalize_destination,
    normalize_proxy_destination,
)


AF_VSOCK = 40
PARENT_CID = 3
DEFAULT_FORWARDER_PORT = 5001
DEFAULT_LOCAL_PROXY_PORT = 18080
MAX_HEADER_BYTES = 64 * 1024
MAX_CONTROL_BYTES = 16 * 1024
MAX_TUNNEL_BYTES_PER_DIRECTION = 256 * 1024 * 1024
DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0
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
UPSTREAM_PROXY_HEADER = "x-leadpoet-upstream-proxy-b64"
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1

_AIOHTTP_PATCH_LOCK = threading.Lock()
_AIOHTTP_PROXY_URL = ""
_AIOHTTP_ORIGINAL_REQUEST = None


class EnclaveEgressProxyError(RuntimeError):
    """The measured enclave proxy rejected or could not relay a request."""


def _install_aiohttp_proxy(proxy_url: str) -> None:
    """Route aiohttp through the enclave proxy without changing scorer code.

    httpx, requests, and urllib honor the standard proxy environment by
    default. aiohttp deliberately does not, so its lowest request seam is
    patched once inside the measured process to supply the same local proxy.
    """

    global _AIOHTTP_ORIGINAL_REQUEST, _AIOHTTP_PROXY_URL
    with _AIOHTTP_PATCH_LOCK:
        if _AIOHTTP_PROXY_URL:
            if _AIOHTTP_PROXY_URL != proxy_url:
                raise EnclaveEgressProxyError("aiohttp enclave proxy is immutable")
            return
        import aiohttp

        original_request = aiohttp.ClientSession._request

        async def proxied_request(session: Any, method: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("proxy", proxy_url)
            return await original_request(session, method, url, *args, **kwargs)

        _AIOHTTP_ORIGINAL_REQUEST = original_request
        aiohttp.ClientSession._request = proxied_request
        _AIOHTTP_PROXY_URL = proxy_url


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _destination_ref(host: str, port: int) -> str:
    return hashlib.sha256((host + ":" + str(port)).encode("ascii")).hexdigest()[:16]


def _ensure_loopback_interface(
    *,
    ioctl: Optional[Callable[..., bytes]] = None,
    socket_factory: Callable[..., Any] = socket.socket,
) -> None:
    """Bring Linux loopback up before the enclave-local proxy is exposed."""

    if ioctl is None:
        import fcntl

        ioctl = fcntl.ioctl
    control = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    request = struct.pack("16sH22s", b"lo", 0, b"")
    try:
        response = ioctl(control.fileno(), SIOCGIFFLAGS, request)
        _name, flags, _padding = struct.unpack("16sH22s", response)
        if not flags & IFF_UP:
            ioctl(
                control.fileno(),
                SIOCSIFFLAGS,
                struct.pack("16sH22s", b"lo", flags | IFF_UP, b""),
            )
            response = ioctl(control.fileno(), SIOCGIFFLAGS, request)
            _name, flags, _padding = struct.unpack("16sH22s", response)
        if not flags & IFF_UP:
            raise EnclaveEgressProxyError("enclave loopback interface is not up")
    except EnclaveEgressProxyError:
        raise
    except Exception as exc:
        raise EnclaveEgressProxyError(
            "enclave loopback interface initialization failed"
        ) from exc
    finally:
        control.close()


def _verify_loopback_listener(
    listener: Any,
    *,
    local_port: int,
    socket_factory: Callable[..., Any] = socket.socket,
) -> None:
    """Prove the local HTTP client can reach the proxy before provider work."""

    client = None
    accepted = None
    try:
        client = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2.0)
        client.connect(("127.0.0.1", int(local_port)))
        accepted, _address = listener.accept()
    except Exception as exc:
        raise EnclaveEgressProxyError(
            "enclave loopback proxy self-test failed"
        ) from exc
    finally:
        for candidate in (accepted, client):
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass


def _read_headers(connection: Any) -> Tuple[bytes, bytes]:
    buffer = bytearray()
    marker = b"\r\n\r\n"
    while marker not in buffer:
        if len(buffer) >= MAX_HEADER_BYTES:
            raise EnclaveEgressProxyError("proxy request headers exceed limit")
        chunk = connection.recv(min(16 * 1024, MAX_HEADER_BYTES - len(buffer)))
        if not chunk:
            raise EnclaveEgressProxyError("proxy client closed before request headers")
        buffer.extend(chunk)
    header_end = buffer.index(marker) + len(marker)
    return bytes(buffer[:header_end]), bytes(buffer[header_end:])


def _parse_authority(authority: str, default_port: int) -> Tuple[str, int]:
    parsed = urlsplit("//" + str(authority or ""))
    if parsed.username or parsed.password or not parsed.hostname:
        raise EnclaveEgressProxyError("proxy destination authority is invalid")
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise EnclaveEgressProxyError("proxy destination port is invalid") from exc
    return normalize_destination(parsed.hostname, port)


def _parse_proxy_request(header_bytes: bytes) -> Dict[str, Any]:
    try:
        header_text = header_bytes.decode("iso-8859-1")
    except UnicodeDecodeError as exc:
        raise EnclaveEgressProxyError("proxy request headers are invalid") from exc
    lines = header_text.split("\r\n")
    request_parts = lines[0].split(" ", 2)
    if len(request_parts) != 3:
        raise EnclaveEgressProxyError("proxy request line is invalid")
    method, target, version = request_parts
    method = method.upper()
    if version not in ("HTTP/1.0", "HTTP/1.1"):
        raise EnclaveEgressProxyError("proxy HTTP version is unsupported")
    if method == "CONNECT":
        upstream_values = []
        tunnel_framing_values = []
        for line in lines[1:]:
            if not line:
                break
            if ":" not in line:
                raise EnclaveEgressProxyError("proxy header line is invalid")
            name, value = line.split(":", 1)
            if name.strip().lower() == UPSTREAM_PROXY_HEADER:
                upstream_values.append(value.strip())
            if name.strip().lower() == TUNNEL_FRAMING_HEADER:
                tunnel_framing_values.append(value.strip())
        if len(upstream_values) > 1:
            raise EnclaveEgressProxyError("upstream proxy header is duplicated")
        if len(tunnel_framing_values) > 1:
            raise EnclaveEgressProxyError("tunnel framing header is duplicated")
        tunnel_framing = ""
        if tunnel_framing_values:
            if tunnel_framing_values != [TUNNEL_FRAMING_MODE]:
                raise EnclaveEgressProxyError("tunnel framing header is invalid")
            tunnel_framing = TUNNEL_FRAMING_MODE
        upstream_proxy_url = ""
        if upstream_values:
            try:
                upstream_proxy_url = base64.b64decode(
                    upstream_values[0],
                    validate=True,
                ).decode("utf-8")
            except Exception as exc:
                raise EnclaveEgressProxyError(
                    "upstream proxy header is invalid"
                ) from exc
            if len(upstream_proxy_url.encode("utf-8")) > 16 * 1024:
                raise EnclaveEgressProxyError("upstream proxy URL exceeds limit")
        host, port = _parse_authority(target, 443)
        request = {
            "method": method,
            "host": host,
            "port": port,
            "forward_headers": b"",
            "tls_protected": True,
        }
        if tunnel_framing:
            request["tunnel_framing"] = tunnel_framing
        if upstream_proxy_url:
            request["upstream_proxy_url"] = upstream_proxy_url
        return request
    raise EnclaveEgressProxyError(
        "external plaintext HTTP is forbidden; use HTTPS CONNECT"
    )


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
            raise EnclaveEgressProxyError("proxy tunnel idle timeout")
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
                    first_closed = "client" if source is left else "parent"
                active.discard(source)
                try:
                    destination.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            next_total = transferred[source] + len(data)
            if next_total > MAX_TUNNEL_BYTES_PER_DIRECTION:
                raise EnclaveEgressProxyError("proxy tunnel byte limit exceeded")
            try:
                destination.sendall(data)
            except OSError as exc:
                if int(getattr(exc, "errno", 0) or 0) not in _PEER_CLOSE_ERRNOS:
                    raise
                # Preserve the opposite direction after a clean peer close.
                # HTTPX still validates TLS and response completeness inside
                # the enclave, so this cannot turn a partial reply into success.
                closed_name = "client" if destination is left else "parent"
                if not first_closed:
                    first_closed = closed_name
                if not write_closed:
                    write_closed = closed_name
                active.discard(source)
                continue
            transferred[source] = next_total
            last_activity = time.monotonic()
    result = {
        "client_to_parent_bytes": transferred[left],
        "parent_to_client_bytes": transferred[right],
        "first_closed": first_closed or "unknown",
    }
    if write_closed:
        result["write_closed"] = write_closed
    return result


class _FramedParentBridge:
    """Expose one framed parent tunnel as a local raw stream."""

    def __init__(self, framed: Any, *, idle_timeout_seconds: float) -> None:
        application, relay = socket.socketpair()
        self._application = application
        self._relay = relay
        self._framed = framed
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._done = threading.Event()
        self._error = None  # type: Optional[BaseException]
        self._thread = threading.Thread(
            target=self._run,
            name="gateway-enclave-framed-upstream-proxy",
            daemon=True,
        )
        self._thread.start()

    def take_stream(self) -> Any:
        if self._application is None:
            raise EnclaveEgressProxyError("framed parent stream was already claimed")
        application = self._application
        self._application = None
        return application

    def _run(self) -> None:
        try:
            relay_raw_and_framed(
                self._relay,
                self._framed,
                idle_timeout_seconds=self._idle_timeout_seconds,
                max_bytes_per_direction=MAX_TUNNEL_BYTES_PER_DIRECTION,
                raw_label="upstream_proxy",
                framed_label="parent",
                terminal_initiator=True,
            )
        except BaseException as exc:  # noqa: BLE001 - bridge failure closes its stream
            self._error = exc
        finally:
            for candidate in (self._relay, self._framed):
                try:
                    candidate.close()
                except Exception:
                    pass
            self._done.set()

    def close(self) -> None:
        if self._application is not None:
            try:
                self._application.close()
            except Exception:
                pass
            self._application = None
        if not self._done.wait(timeout=1.0):
            for candidate in (self._relay, self._framed):
                try:
                    candidate.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
            self._done.wait(timeout=1.5)

    @property
    def stopped(self) -> bool:
        return self._done.is_set() and not self._thread.is_alive()


class _ManagedProxyStream:
    """Close a TLS/plain proxy stream together with its framing bridge."""

    def __init__(self, stream: Any, bridge: _FramedParentBridge) -> None:
        self._stream = stream
        self._bridge = bridge
        self._closed = False

    def fileno(self) -> int:
        return int(self._stream.fileno())

    def recv(self, size: int) -> bytes:
        return self._stream.recv(size)

    def sendall(self, payload: bytes) -> None:
        self._stream.sendall(payload)

    def shutdown(self, how: int) -> None:
        self._stream.shutdown(how)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            self._bridge.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class EnclaveEgressProxy:
    def __init__(
        self,
        *,
        recv_exact: Callable[[Any, int], bytes],
        local_port: int = DEFAULT_LOCAL_PROXY_PORT,
        forwarder_port: int = DEFAULT_FORWARDER_PORT,
        socket_factory: Callable[..., Any] = socket.socket,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        loopback_initializer: Callable[[], None] = _ensure_loopback_interface,
    ) -> None:
        self.local_port = int(local_port)
        self.forwarder_port = int(forwarder_port)
        self._recv_exact = recv_exact
        self._socket_factory = socket_factory
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._loopback_initializer = loopback_initializer
        self._listener = None
        self._thread = None
        self._stop = threading.Event()
        self._loopback_verified = False
        self._status_lock = threading.Lock()
        self._last_failure = None  # type: Optional[Dict[str, Any]]
        self._last_tunnel = None  # type: Optional[Dict[str, Any]]

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._listener is not None)

    def start(self) -> Dict[str, Any]:
        if self.running:
            return self.status()
        listener = None
        self._loopback_initializer()
        try:
            listener = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", self.local_port))
            listener.listen(64)
            _verify_loopback_listener(
                listener,
                local_port=self.local_port,
                socket_factory=self._socket_factory,
            )
        except Exception:
            if listener is not None:
                try:
                    listener.close()
                except Exception:
                    pass
            raise
        self._listener = listener
        self._loopback_verified = True
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="gateway-enclave-egress-proxy",
            daemon=True,
        )
        self._thread.start()
        self._configure_environment()
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            last_failure = dict(self._last_failure or {})
            last_tunnel = dict(self._last_tunnel or {})
        result = {
            "status": "running" if self.running else "stopped",
            "local_port": self.local_port,
            "forwarder_port": self.forwarder_port,
            "policy_hash": destination_policy_hash(),
            "https_tls_terminates_in_enclave": True,
            "external_plaintext_http_allowed": False,
            "loopback_http_allowed": True,
            "loopback_listener_verified": self._loopback_verified,
            "tls_upstream_proxy_supported": True,
        }
        if last_failure:
            result["last_failure"] = last_failure
        if last_tunnel:
            result["last_tunnel"] = last_tunnel
        return result

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
        self._listener = None
        self._loopback_verified = False

    def _configure_environment(self) -> None:
        proxy_url = "http://127.0.0.1:%s" % self.local_port
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        _install_aiohttp_proxy(proxy_url)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._listener.accept()
            except Exception:
                if not self._stop.is_set():
                    print("[TEE] Egress proxy accept failed", flush=True)
                return
            threading.Thread(
                target=self._handle_client,
                args=(connection,),
                name="gateway-enclave-egress-tunnel",
                daemon=True,
            ).start()

    def _open_parent_tunnel(
        self,
        host: str,
        port: int,
        *,
        purpose: str = "provider",
        tunnel_framing: str = "",
    ) -> Any:
        parent = self._socket_factory(AF_VSOCK, socket.SOCK_STREAM)
        try:
            parent.connect((PARENT_CID, self.forwarder_port))
            params = {
                "host": host,
                "port": port,
                "policy_hash": destination_policy_hash(),
            }
            if purpose == "upstream_proxy":
                params["purpose"] = purpose
            elif purpose != "provider":
                raise EnclaveEgressProxyError("parent egress purpose is invalid")
            if tunnel_framing:
                if tunnel_framing != TUNNEL_FRAMING_MODE:
                    raise EnclaveEgressProxyError(
                        "parent egress tunnel framing is invalid"
                    )
                params["tunnel_framing"] = tunnel_framing
            request = _canonical_json(
                {
                    "method": "connect",
                    "params": params,
                }
            )
            if len(request) > MAX_CONTROL_BYTES:
                raise EnclaveEgressProxyError("proxy control request exceeds limit")
            parent.sendall(len(request).to_bytes(4, byteorder="big") + request)
            prefix = self._recv_exact(parent, 4)
            if len(prefix) != 4:
                raise EnclaveEgressProxyError("parent egress response is incomplete")
            size = int.from_bytes(prefix, byteorder="big")
            if size < 2 or size > MAX_CONTROL_BYTES:
                raise EnclaveEgressProxyError("parent egress response size is invalid")
            encoded = self._recv_exact(parent, size)
            if len(encoded) != size:
                raise EnclaveEgressProxyError("parent egress response body is incomplete")
            response = json.loads(encoded.decode("ascii"))
            result = response.get("result") if isinstance(response, dict) else None
            if not isinstance(result, dict) or result.get("status") != "connected":
                raise EnclaveEgressProxyError("parent refused egress destination")
            if result.get("policy_hash") != destination_policy_hash():
                raise EnclaveEgressProxyError("parent egress policy hash mismatch")
            if str(result.get("tunnel_framing") or "") != tunnel_framing:
                raise EnclaveEgressProxyError("parent egress tunnel framing mismatch")
            return parent
        except Exception:
            try:
                parent.close()
            except Exception:
                pass
            raise

    def _open_upstream_proxy_tunnel(
        self,
        *,
        proxy_url: str,
        destination_host: str,
        destination_port: int,
        tunnel_framing: str = "",
    ) -> Any:
        parsed = urlsplit(str(proxy_url or ""))
        proxy_scheme = parsed.scheme.lower()
        try:
            parsed_port = parsed.port
            proxy_port = (
                parsed_port
                if parsed_port is not None
                else (443 if proxy_scheme == "https" else 80)
            )
        except ValueError as exc:
            raise EnclaveEgressProxyError("upstream proxy port is invalid") from exc
        if (
            proxy_scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise EnclaveEgressProxyError(
                "upstream proxy must be an HTTP CONNECT or HTTPS proxy URL"
            )
        try:
            proxy_host, proxy_port = normalize_proxy_destination(
                parsed.hostname,
                proxy_port,
            )
        except ValueError as exc:
            raise EnclaveEgressProxyError(
                "upstream proxy destination is invalid"
            ) from exc
        if (parsed.username is None) != (parsed.password is None):
            raise EnclaveEgressProxyError("upstream proxy credentials are incomplete")
        parent_kwargs = {"purpose": "upstream_proxy"}
        if tunnel_framing:
            parent_kwargs["tunnel_framing"] = tunnel_framing
        parent = self._open_parent_tunnel(proxy_host, proxy_port, **parent_kwargs)
        bridge = None
        protected = parent
        try:
            if tunnel_framing:
                bridge = _FramedParentBridge(
                    parent,
                    idle_timeout_seconds=self._idle_timeout_seconds,
                )
                protected = bridge.take_stream()
            if proxy_scheme == "https":
                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
                protected = context.wrap_socket(
                    protected,
                    server_hostname=proxy_host,
                )
            lines = [
                "CONNECT %s:%s HTTP/1.1" % (destination_host, destination_port),
                "Host: %s:%s" % (destination_host, destination_port),
                "Proxy-Connection: Keep-Alive",
            ]
            if parsed.username is not None:
                username = unquote(parsed.username)
                password = unquote(parsed.password or "")
                if any(character in username + password for character in "\x00\r\n"):
                    raise EnclaveEgressProxyError(
                        "upstream proxy credentials are invalid"
                    )
                token = base64.b64encode(
                    (username + ":" + password).encode("utf-8")
                ).decode("ascii")
                lines.append("Proxy-Authorization: Basic " + token)
            request = ("\r\n".join(lines) + "\r\n\r\n").encode("iso-8859-1")
            protected.sendall(request)
            response_headers, remainder = _read_headers(protected)
            status_line = response_headers.split(b"\r\n", 1)[0]
            parts = status_line.split(b" ", 2)
            if len(parts) < 2 or not parts[1].isdigit():
                raise EnclaveEgressProxyError("upstream proxy response is malformed")
            status = int(parts[1])
            if status < 200 or status >= 300:
                raise EnclaveEgressProxyError(
                    "upstream proxy CONNECT failed with HTTP status %d" % status
                )
            if remainder:
                raise EnclaveEgressProxyError(
                    "upstream proxy returned unexpected CONNECT payload"
                )
            if bridge is not None:
                return _ManagedProxyStream(protected, bridge)
            return protected
        except Exception:
            try:
                protected.close()
            except Exception:
                pass
            if bridge is not None:
                bridge.close()
            elif protected is not parent:
                try:
                    parent.close()
                except Exception:
                    pass
            raise

    def _handle_client(self, client: Any) -> None:
        parent = None
        destination_ref = "unknown"
        failure_stage = "read_client_headers"
        connect_response_started = False
        try:
            headers, remainder = _read_headers(client)
            failure_stage = "parse_connect_request"
            request = _parse_proxy_request(headers)
            host = str(request["host"])
            port = int(request["port"])
            destination_ref = _destination_ref(host, port)
            upstream_proxy_url = str(request.get("upstream_proxy_url") or "")
            parent_stream_is_framed = bool(
                request.get("tunnel_framing") == TUNNEL_FRAMING_MODE
                and not upstream_proxy_url
            )
            if upstream_proxy_url:
                failure_stage = "open_upstream_proxy_tunnel"
                parent = self._open_upstream_proxy_tunnel(
                    proxy_url=upstream_proxy_url,
                    destination_host=host,
                    destination_port=port,
                    tunnel_framing=str(request.get("tunnel_framing") or ""),
                )
            else:
                failure_stage = "open_parent_tunnel"
                if request.get("tunnel_framing") == TUNNEL_FRAMING_MODE:
                    parent = self._open_parent_tunnel(
                        host,
                        port,
                        tunnel_framing=TUNNEL_FRAMING_MODE,
                    )
                else:
                    parent = self._open_parent_tunnel(host, port)
            failure_stage = "acknowledge_connect"
            if request["method"] == "CONNECT":
                # Once any CONNECT response bytes are emitted the client owns
                # an opaque TLS stream. A later plaintext proxy response would
                # corrupt that stream and can poison otherwise healthy retries.
                connect_response_started = True
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:
                parent.sendall(request["forward_headers"])
            if remainder:
                if parent_stream_is_framed:
                    send_tunnel_frame(parent, remainder)
                else:
                    parent.sendall(remainder)
            failure_stage = "relay_tls_tunnel"
            if parent_stream_is_framed:
                relay = relay_raw_and_framed(
                    client,
                    parent,
                    idle_timeout_seconds=self._idle_timeout_seconds,
                    max_bytes_per_direction=MAX_TUNNEL_BYTES_PER_DIRECTION,
                    raw_label="client",
                    framed_label="parent",
                    terminal_initiator=True,
                )
            else:
                relay = _relay_bidirectional(
                    client,
                    parent,
                    idle_timeout_seconds=self._idle_timeout_seconds,
                )
            with self._status_lock:
                self._last_tunnel = {
                    "stage": failure_stage,
                    "destination_ref": destination_ref,
                    **relay,
                }
        except Exception as exc:
            with self._status_lock:
                self._last_failure = {
                    "stage": failure_stage,
                    "error_type": type(exc).__name__,
                    "errno": int(getattr(exc, "errno", 0) or 0),
                    "destination_ref": destination_ref,
                }
            if not connect_response_started:
                try:
                    client.sendall(
                        b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"
                    )
                except Exception:
                    pass
            print(
                "[TEE] Egress proxy tunnel failed destination_ref=%s error_type=%s"
                % (destination_ref, type(exc).__name__),
                flush=True,
            )
        finally:
            for candidate in (parent, client):
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass


def configured_proxy_ports() -> Tuple[int, int]:
    try:
        local_port = int(
            os.getenv("RESEARCH_LAB_TEE_EGRESS_LOCAL_PORT", str(DEFAULT_LOCAL_PROXY_PORT))
        )
        forwarder_port = int(
            os.getenv("RESEARCH_LAB_TEE_EGRESS_VSOCK_PORT", str(DEFAULT_FORWARDER_PORT))
        )
    except ValueError as exc:
        raise EnclaveEgressProxyError("configured egress proxy port is invalid") from exc
    for port in (local_port, forwarder_port):
        if port <= 1024 or port > 65535:
            raise EnclaveEgressProxyError("configured egress proxy port is invalid")
    return local_port, forwarder_port
