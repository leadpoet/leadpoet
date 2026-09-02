"""``sitecustomize`` shim for the Lab Arena model image (labarena.md 3.3, 6.1).

The Arena base image writes a ``sitecustomize.py`` containing
``SITECUSTOMIZE_SOURCE`` so this module is installed before any model code
runs. It intercepts the four supported clients (``urllib``, ``requests``,
``httpx`` sync and async, ``aiohttp``), maps each request to exactly one
closed operation through ``lab_arena.operations.match_request``, and sends one
frame over the Unix socket named by ``LAB_ARENA_WORKER_SOCKET``::

    4-byte big-endian length + canonical JSON
    {"schema_version": "leadpoet.lab_arena.operation_frame.v1",
     "operation_id": "...", "parameters": {...}, "timeout_ms": N}

The worker answers one framed JSON document, either
``{"status": int, "headers": {...}, "body_b64": str}`` or
``{"error": "<generic code>"}``, and the shim synthesizes a client-native
response. A request that matches no operation fails inside the client with a
generic connection error; nothing else is reachable and there is no
localhost passthrough. The frame carries only the operation id and validated
parameters: no headers, no raw URL, no credential, and no round, miner,
stage, run, account, or lease identity (those come from the worker's lease).

This module imports only the standard library, ``lab_arena.operations`` and
``lab_arena.contracts`` (whose closure is ``leadpoet_canonical.attested_v2``
and ``leadpoet_canonical.constants``); ``SHIM_IMAGE_MODULES`` lists the exact
module files the base image must ship. Importing it has no side effects;
``install()`` is explicit and idempotent.

The shim is a convenience, not the trust boundary: raw sockets, other HTTP
clients, ``http.client``, or a custom ``urllib`` opener bypass it and reach
nothing because runsc ``--network=none`` provides no interface, and the
socket accepts nothing but operation frames (``validate_operation_frame`` is
the worker-side check).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import socket
import threading
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode
import urllib.error
import urllib.request
import urllib.response

from lab_arena import contracts, operations

OPERATION_FRAME_SCHEMA_VERSION = "leadpoet.lab_arena.operation_frame.v1"
WORKER_SOCKET_ENV = "LAB_ARENA_WORKER_SOCKET"
MAX_FRAME_BYTES = 1_048_576
MAX_RESPONSE_FRAME_BYTES = 4 * 1_048_576
FRAME_FIELDS = ("schema_version", "operation_id", "parameters", "timeout_ms")
RESPONSE_FIELDS = ("status", "headers", "body_b64")
DEFAULT_TIMEOUT_MS = 30_000
SOCKET_GRACE_SECONDS = 15.0
ERROR_PREFIX = "lab arena: "
SITECUSTOMIZE_SOURCE = "import lab_arena.shim as _lab_arena_shim\n_lab_arena_shim.install()\n"
SHIM_IMAGE_MODULES = (
    "lab_arena/__init__.py",
    "lab_arena/contracts.py",
    "lab_arena/operations.py",
    "lab_arena/shim.py",
    "leadpoet_canonical/__init__.py",
    "leadpoet_canonical/constants.py",
    "leadpoet_canonical/attested_v2.py",
)
FRAME_ERROR_CODES = frozenset(
    {
        "invalid_frame",
        "frame_too_large",
        "socket_unavailable",
        "worker_unavailable",
        "invalid_response",
        "invalid_request",
        "invalid_body",
    }
)

_INSTALL_LOCK = threading.Lock()
_ORIGINALS: Dict[str, Any] = {}


_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ShimError(RuntimeError):
    """Base shim failure; ``code`` is a short generic identifier.

    Codes come from ``FRAME_ERROR_CODES``, ``operations.ERROR_CODES``, or the
    worker's own generic refusal vocabulary; the message is always bounded and
    never carries provider, account, credential, or transport detail.
    """

    def __init__(self, code: str) -> None:
        if not isinstance(code, str) or not _CODE_RE.match(code):
            raise ValueError("shim error code is not a generic identifier")
        self.code = code
        super().__init__(ERROR_PREFIX + code)


class ShimRequestError(ShimError):
    """The request matched no operation or violated its schema."""


class ShimTransportError(ShimError):
    """The worker socket was unavailable or answered out of contract."""


class ShimProviderError(ShimError):
    """The worker refused the call with a generic code."""


class OperationFrameError(ShimError):
    """A socket frame is malformed (worker-side check)."""


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def build_operation_frame(operation_id: str, parameters: Mapping[str, Any], timeout_ms: int) -> bytes:
    """Encode one request frame; ``parameters`` must already be normalized."""

    operation = operations.OPERATIONS.get(operation_id)
    if operation is None:
        raise ShimRequestError("no_matching_operation")
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
        raise ShimRequestError("invalid_request")
    frame = {
        "schema_version": OPERATION_FRAME_SCHEMA_VERSION,
        "operation_id": operation_id,
        "parameters": dict(parameters),
        "timeout_ms": min(timeout_ms, operation.timeout_seconds * 1000),
    }
    encoded = contracts.canonical_json(frame).encode("utf-8")
    if len(encoded) > MAX_FRAME_BYTES:
        raise ShimRequestError("request_too_large")
    return encoded


def validate_operation_frame(frame: Any) -> Tuple[str, Dict[str, Any], int]:
    """Worker-side frame check: exact keys, known operation, bounded timeout,
    and a full re-validation of the parameters. Extra keys such as
    ``round_id``, ``lease_token`` or ``miner`` are rejected outright."""

    if not isinstance(frame, Mapping) or set(frame) != set(FRAME_FIELDS):
        raise OperationFrameError("invalid_frame")
    if frame["schema_version"] != OPERATION_FRAME_SCHEMA_VERSION:
        raise OperationFrameError("invalid_frame")
    operation_id = frame["operation_id"]
    operation = operations.OPERATIONS.get(operation_id) if isinstance(operation_id, str) else None
    if operation is None:
        raise OperationFrameError("no_matching_operation")
    timeout_ms = frame["timeout_ms"]
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
        raise OperationFrameError("invalid_frame")
    if not 1 <= timeout_ms <= operation.timeout_seconds * 1000:
        raise OperationFrameError("invalid_frame")
    parameters = operations.validate_operation_request(operation_id, frame["parameters"])
    return operation_id, parameters, timeout_ms


def decode_operation_frame(data: bytes) -> Tuple[str, Dict[str, Any], int]:
    if len(data) > MAX_FRAME_BYTES:
        raise OperationFrameError("frame_too_large")
    try:
        frame = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise OperationFrameError("invalid_frame") from exc
    return validate_operation_frame(frame)


def encode_worker_response(status: int, headers: Mapping[str, str], body: bytes) -> bytes:
    """Worker-side helper producing the framed success document."""

    document = {
        "status": int(status),
        "headers": {str(k): str(v) for k, v in headers.items()},
        "body_b64": base64.b64encode(bytes(body)).decode("ascii"),
    }
    return contracts.canonical_json(document).encode("utf-8")


def encode_worker_error(code: str) -> bytes:
    return contracts.canonical_json({"error": str(code)}).encode("utf-8")


def parse_worker_response(payload: Any) -> Tuple[int, Dict[str, str], bytes]:
    if not isinstance(payload, Mapping):
        raise ShimTransportError("invalid_response")
    if set(payload) == {"error"}:
        code = payload["error"]
        if not isinstance(code, str) or not _CODE_RE.match(code):
            raise ShimTransportError("invalid_response")
        raise ShimProviderError(code)
    if set(payload) != set(RESPONSE_FIELDS):
        raise ShimTransportError("invalid_response")
    status = payload["status"]
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise ShimTransportError("invalid_response")
    raw_headers = payload["headers"]
    if not isinstance(raw_headers, Mapping):
        raise ShimTransportError("invalid_response")
    headers: Dict[str, str] = {}
    for name, value in raw_headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ShimTransportError("invalid_response")
        if name.lower() in ("content-type", "content-length"):
            headers[name.lower()] = value
    try:
        body = base64.b64decode(str(payload["body_b64"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise ShimTransportError("invalid_response") from exc
    headers["content-length"] = str(len(body))
    return status, headers, body


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(65536, size - len(output)))
        if not chunk:
            raise ShimTransportError("worker_unavailable")
        output.extend(chunk)
    return bytes(output)


def worker_socket_path() -> str:
    path = str(os.environ.get(WORKER_SOCKET_ENV) or "").strip()
    if not path.startswith("/"):
        raise ShimTransportError("socket_unavailable")
    return path


def dispatch(operation_id: str, parameters: Mapping[str, Any], timeout_ms: int) -> Tuple[int, Dict[str, str], bytes]:
    """Send one frame and return the worker's ``(status, headers, body)``."""

    encoded = build_operation_frame(operation_id, parameters, timeout_ms)
    path = worker_socket_path()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(timeout_ms / 1000.0 + SOCKET_GRACE_SECONDS)
        try:
            connection.connect(path)
            connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
            size = int.from_bytes(_recv_exact(connection, 4), "big")
            if size < 2 or size > MAX_RESPONSE_FRAME_BYTES:
                raise ShimTransportError("invalid_response")
            raw = _recv_exact(connection, size)
        except OSError as exc:
            raise ShimTransportError("worker_unavailable") from exc
    finally:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ShimTransportError("invalid_response") from exc
    return parse_worker_response(payload)


def execute(
    *,
    method: str,
    url: str,
    headers: Mapping[str, Any],
    body: bytes,
    timeout_ms: int,
) -> Tuple[int, Dict[str, str], bytes]:
    """Match, validate, and dispatch one client request."""

    try:
        operation_id, parameters = operations.match_request(method, url, body, headers)
    except operations.OperationError as exc:
        raise ShimRequestError(exc.code) from None
    return dispatch(operation_id, parameters, max(1, int(timeout_ms)))


def _timeout_ms(value: Any) -> int:
    try:
        if value is None:
            return DEFAULT_TIMEOUT_MS
        if isinstance(value, Mapping):
            values = [float(item) for item in value.values() if item is not None]
            value = max(values) if values else DEFAULT_TIMEOUT_MS / 1000.0
        elif hasattr(value, "total") and getattr(value, "total") is not None:
            value = getattr(value, "total")
        elif hasattr(value, "read") and not isinstance(value, (int, float, str, tuple)):
            candidates = [getattr(value, name, None) for name in ("read", "connect", "write", "pool")]
            values = [float(item) for item in candidates if item is not None]
            value = max(values) if values else DEFAULT_TIMEOUT_MS / 1000.0
        elif isinstance(value, tuple):
            values = [float(item) for item in value if item is not None]
            value = max(values) if values else DEFAULT_TIMEOUT_MS / 1000.0
        return max(1, int(float(value) * 1000))
    except (TypeError, ValueError) as exc:
        raise ShimRequestError("invalid_request") from exc


# ---------------------------------------------------------------------------
# urllib
# ---------------------------------------------------------------------------


def _urlopen(request: Any, *args: Any, **kwargs: Any) -> Any:
    data = kwargs.get("data")
    if args:
        data = args[0]
    timeout = kwargs.get("timeout")
    if len(args) > 1:
        timeout = args[1]
    if isinstance(request, urllib.request.Request):
        url = request.full_url
        method = request.get_method()
        headers = dict(request.header_items())
        body = request.data if data is None else data
    else:
        url = str(request)
        method = "POST" if data is not None else "GET"
        headers = {}
        body = data
    if body is None:
        body = b""
    if isinstance(body, str):
        body = body.encode("utf-8")
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise TypeError("urllib request body must be bytes-like")
    try:
        status, response_headers, response_body = execute(
            method=method, url=url, headers=headers, body=bytes(body), timeout_ms=_timeout_ms(timeout)
        )
    except ShimError as exc:
        raise urllib.error.URLError(str(exc)) from None
    message = Message()
    for name, value in response_headers.items():
        message[name] = value
    if status >= 400:
        raise urllib.error.HTTPError(url, status, "lab arena provider error", message, io.BytesIO(response_body))
    return urllib.response.addinfourl(io.BytesIO(response_body), message, url, status)


# ---------------------------------------------------------------------------
# httpx
# ---------------------------------------------------------------------------


def _httpx_send(httpx: Any) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    def sync_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        extensions = dict(getattr(request, "extensions", {}) or {})
        try:
            status, response_headers, response_body = execute(
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                body=bytes(request.content),
                timeout_ms=_timeout_ms(extensions.get("timeout")),
            )
        except ShimError as exc:
            raise httpx.ConnectError(str(exc), request=request) from None
        return httpx.Response(status_code=status, headers=response_headers, content=response_body, request=request)

    async def async_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(sync_send, client, request, *args, **kwargs)

    return sync_send, async_send


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


def _requests_send(requests: Any) -> Callable[..., Any]:
    def send(session: Any, request: Any, **kwargs: Any) -> Any:
        body = request.body or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        if not isinstance(body, (bytes, bytearray, memoryview)):
            raise requests.exceptions.ConnectionError(ERROR_PREFIX + "invalid_body", request=request)
        try:
            status, response_headers, response_body = execute(
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                body=bytes(body),
                timeout_ms=_timeout_ms(kwargs.get("timeout")),
            )
        except ShimError as exc:
            raise requests.exceptions.ConnectionError(str(exc), request=request) from None
        response = requests.Response()
        response.status_code = status
        response.headers.update(response_headers)
        response._content = response_body
        response.url = str(request.url)
        response.request = request
        response.reason = "OK" if status < 400 else "lab arena provider error"
        response.encoding = "utf-8"
        return response

    return send


# ---------------------------------------------------------------------------
# aiohttp
# ---------------------------------------------------------------------------


class _AiohttpContent:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._offset
        start = self._offset
        self._offset = min(len(self._body), self._offset + max(0, int(size)))
        return self._body[start : self._offset]


class _AiohttpResponse:
    """The subset of ``aiohttp.ClientResponse`` a model can rely on."""

    def __init__(self, *, url: str, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self._body = body
        self.status = int(status)
        self.headers = dict(headers)
        self.url = url
        self.reason = "OK" if status < 400 else "lab arena provider error"
        self.history = ()
        self.request_info = SimpleNamespace(real_url=url)
        self.content = _AiohttpContent(body)

    @property
    def ok(self) -> bool:
        return self.status < 400

    async def __aenter__(self) -> "_AiohttpResponse":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.release()
        await self.wait_for_close()

    async def read(self) -> bytes:
        return self._body

    async def text(self, encoding: Optional[str] = None, errors: str = "strict") -> str:
        return self._body.decode(encoding or "utf-8", errors=errors)

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        return json.loads(self._body.decode(kwargs.get("encoding") or "utf-8"))

    def raise_for_status(self) -> None:
        if self.status < 400:
            return
        import aiohttp

        raise aiohttp.ClientResponseError(
            request_info=self.request_info,
            history=(),
            status=self.status,
            message=self.reason,
            headers=self.headers,
        )

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_for_close(self) -> None:
        return None


def _aiohttp_body(session: Any, kwargs: Mapping[str, Any]) -> Tuple[bytes, Dict[str, str]]:
    headers = {str(k): str(v) for k, v in dict(getattr(session, "headers", {}) or {}).items()}
    headers.update({str(k): str(v) for k, v in dict(kwargs.get("headers") or {}).items()})
    json_body = kwargs.get("json")
    data = kwargs.get("data")
    if json_body is not None and data is not None:
        raise ShimRequestError("invalid_body")
    if json_body is not None:
        serialize = getattr(session, "_json_serialize", json.dumps)
        headers.setdefault("Content-Type", "application/json")
        return str(serialize(json_body)).encode("utf-8"), headers
    if data is None:
        return b"", headers
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), headers
    if isinstance(data, str):
        return data.encode("utf-8"), headers
    if isinstance(data, Mapping):
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        return urlencode(dict(data)).encode("utf-8"), headers
    raise ShimRequestError("invalid_body")


def _aiohttp_request(aiohttp: Any, yarl_url: Any) -> Callable[..., Any]:
    async def request(session: Any, method: Any, str_or_url: Any, *args: Any, **kwargs: Any) -> Any:
        url = yarl_url(str(str_or_url))
        params = kwargs.get("params")
        if params is not None:
            url = url.extend_query(params) if hasattr(url, "extend_query") else url.update_query(params)
        # A per-request timeout wins; otherwise the session's ClientTimeout.
        timeout = kwargs.get("timeout")
        if timeout is None or timeout is getattr(getattr(aiohttp, "helpers", None), "sentinel", object()):
            timeout = getattr(session, "timeout", None)
        try:
            body, headers = _aiohttp_body(session, kwargs)
            status, response_headers, response_body = await asyncio.to_thread(
                execute,
                method=str(method),
                url=str(url),
                headers=headers,
                body=body,
                timeout_ms=_timeout_ms(timeout),
            )
        except ShimError as exc:
            raise aiohttp.ClientConnectionError(str(exc)) from None
        response = _AiohttpResponse(url=str(url), status=status, headers=response_headers, body=response_body)
        raise_for_status = kwargs.get("raise_for_status")
        if raise_for_status is None:
            raise_for_status = getattr(session, "_raise_for_status", False)
        if raise_for_status is True:
            response.raise_for_status()
        elif callable(raise_for_status):
            await raise_for_status(response)
        return response

    return request


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def installed() -> bool:
    return bool(_ORIGINALS)


def install() -> None:
    """Patch the supported clients in place. Idempotent and thread-safe."""

    with _INSTALL_LOCK:
        if _ORIGINALS:
            return
        originals: Dict[str, Any] = {"urllib.request.urlopen": urllib.request.urlopen}
        urllib.request.urlopen = _urlopen
        try:
            import httpx
        except ImportError:
            httpx = None
        if httpx is not None:
            originals["httpx.Client.send"] = httpx.Client.send
            originals["httpx.AsyncClient.send"] = httpx.AsyncClient.send
            sync_send, async_send = _httpx_send(httpx)
            httpx.Client.send = sync_send
            httpx.AsyncClient.send = async_send
        try:
            import requests
        except ImportError:
            requests = None
        if requests is not None:
            originals["requests.Session.send"] = requests.Session.send
            requests.Session.send = _requests_send(requests)
        try:
            import aiohttp
            from yarl import URL as yarl_url
        except ImportError:
            aiohttp = None
            yarl_url = None
        if aiohttp is not None:
            originals["aiohttp.ClientSession._request"] = aiohttp.ClientSession._request
            aiohttp.ClientSession._request = _aiohttp_request(aiohttp, yarl_url)
        _ORIGINALS.update(originals)


def uninstall() -> None:
    """Restore the original client methods (tests only; the image never calls it)."""

    with _INSTALL_LOCK:
        if not _ORIGINALS:
            return
        urllib.request.urlopen = _ORIGINALS["urllib.request.urlopen"]
        if "httpx.Client.send" in _ORIGINALS:
            import httpx

            httpx.Client.send = _ORIGINALS["httpx.Client.send"]
            httpx.AsyncClient.send = _ORIGINALS["httpx.AsyncClient.send"]
        if "requests.Session.send" in _ORIGINALS:
            import requests

            requests.Session.send = _ORIGINALS["requests.Session.send"]
        if "aiohttp.ClientSession._request" in _ORIGINALS:
            import aiohttp

            aiohttp.ClientSession._request = _ORIGINALS["aiohttp.ClientSession._request"]
        _ORIGINALS.clear()


def shim_source_hash() -> str:
    """Hash of this module's source, bound as ``release.shim_hash``."""

    path = Path(__file__)
    if path.suffix != ".py":
        raise ShimTransportError("invalid_request")
    return contracts.hash_bytes(path.read_bytes())


__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "FRAME_ERROR_CODES",
    "FRAME_FIELDS",
    "MAX_FRAME_BYTES",
    "MAX_RESPONSE_FRAME_BYTES",
    "OPERATION_FRAME_SCHEMA_VERSION",
    "OperationFrameError",
    "RESPONSE_FIELDS",
    "SHIM_IMAGE_MODULES",
    "SITECUSTOMIZE_SOURCE",
    "ShimError",
    "ShimProviderError",
    "ShimRequestError",
    "ShimTransportError",
    "WORKER_SOCKET_ENV",
    "build_operation_frame",
    "decode_operation_frame",
    "dispatch",
    "encode_worker_error",
    "encode_worker_response",
    "execute",
    "install",
    "installed",
    "parse_worker_response",
    "shim_source_hash",
    "uninstall",
    "validate_operation_frame",
    "worker_socket_path",
]
