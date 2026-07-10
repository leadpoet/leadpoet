"""Enclave-local HTTP proxy for provider and public-web scoring traffic.

HTTPS clients terminate TLS and validate certificates inside the enclave.  The
parent receives only a destination handshake followed by opaque TLS bytes.
Plain HTTP remains supported for behavior compatibility, but its bytes are not
confidential or tamper-evident and receipts must not describe it as protected
provider evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit

from gateway.tee.egress_policy import destination_policy_hash, normalize_destination


AF_VSOCK = 40
PARENT_CID = 3
DEFAULT_FORWARDER_PORT = 5001
DEFAULT_LOCAL_PROXY_PORT = 18080
MAX_HEADER_BYTES = 64 * 1024
MAX_CONTROL_BYTES = 16 * 1024
MAX_TUNNEL_BYTES_PER_DIRECTION = 256 * 1024 * 1024
DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0
RELAY_CHUNK_BYTES = 64 * 1024

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
        host, port = _parse_authority(target, 443)
        return {
            "method": method,
            "host": host,
            "port": port,
            "forward_headers": b"",
            "tls_protected": True,
        }
    parsed = urlsplit(target)
    if parsed.scheme.lower() != "http" or not parsed.hostname:
        raise EnclaveEgressProxyError("plaintext proxy requests require an absolute http URL")
    try:
        target_port = parsed.port or 80
    except ValueError as exc:
        raise EnclaveEgressProxyError("proxy destination port is invalid") from exc
    host, port = normalize_destination(parsed.hostname, target_port)
    origin_target = parsed.path or "/"
    if parsed.query:
        origin_target += "?" + parsed.query
    forwarded_lines = [method + " " + origin_target + " " + version]
    for line in lines[1:]:
        if not line:
            continue
        name = line.split(":", 1)[0].strip().lower() if ":" in line else ""
        if name in ("proxy-authorization", "proxy-connection"):
            continue
        forwarded_lines.append(line)
    return {
        "method": method,
        "host": host,
        "port": port,
        "forward_headers": ("\r\n".join(forwarded_lines) + "\r\n\r\n").encode("iso-8859-1"),
        "tls_protected": False,
    }


def _relay_bidirectional(
    left: Any,
    right: Any,
    *,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    transferred = {left: 0, right: 0}
    last_activity = time.monotonic()
    while active:
        remaining = max(0.0, idle_timeout_seconds - (time.monotonic() - last_activity))
        if remaining <= 0:
            raise EnclaveEgressProxyError("proxy tunnel idle timeout")
        readable, _writable, _exceptional = select.select(list(active), [], [], min(1.0, remaining))
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
            transferred[source] += len(data)
            if transferred[source] > MAX_TUNNEL_BYTES_PER_DIRECTION:
                raise EnclaveEgressProxyError("proxy tunnel byte limit exceeded")
            destination.sendall(data)
            last_activity = time.monotonic()


class EnclaveEgressProxy:
    def __init__(
        self,
        *,
        recv_exact: Callable[[Any, int], bytes],
        local_port: int = DEFAULT_LOCAL_PROXY_PORT,
        forwarder_port: int = DEFAULT_FORWARDER_PORT,
        socket_factory: Callable[..., Any] = socket.socket,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self.local_port = int(local_port)
        self.forwarder_port = int(forwarder_port)
        self._recv_exact = recv_exact
        self._socket_factory = socket_factory
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
        listener = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.local_port))
        listener.listen(64)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="gateway-enclave-egress-proxy",
            daemon=True,
        )
        self._thread.start()
        self._configure_environment()
        return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "status": "running" if self.running else "stopped",
            "local_port": self.local_port,
            "forwarder_port": self.forwarder_port,
            "policy_hash": destination_policy_hash(),
            "https_tls_terminates_in_enclave": True,
            "plaintext_http_integrity_protected": False,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
        self._listener = None

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

    def _open_parent_tunnel(self, host: str, port: int) -> Any:
        parent = self._socket_factory(AF_VSOCK, socket.SOCK_STREAM)
        try:
            parent.connect((PARENT_CID, self.forwarder_port))
            request = _canonical_json(
                {
                    "method": "connect",
                    "params": {
                        "host": host,
                        "port": port,
                        "policy_hash": destination_policy_hash(),
                    },
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
            return parent
        except Exception:
            try:
                parent.close()
            except Exception:
                pass
            raise

    def _handle_client(self, client: Any) -> None:
        parent = None
        destination_ref = "unknown"
        try:
            headers, remainder = _read_headers(client)
            request = _parse_proxy_request(headers)
            host = str(request["host"])
            port = int(request["port"])
            destination_ref = _destination_ref(host, port)
            parent = self._open_parent_tunnel(host, port)
            if request["method"] == "CONNECT":
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:
                parent.sendall(request["forward_headers"])
            if remainder:
                parent.sendall(remainder)
            _relay_bidirectional(
                client,
                parent,
                idle_timeout_seconds=self._idle_timeout_seconds,
            )
        except Exception as exc:
            try:
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
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
