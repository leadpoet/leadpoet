"""Model-sandbox HTTP client shim for the measured AF_UNIX provider broker.

The runsc network namespace has no external interfaces.  Supported Python HTTP
clients are redirected to this socket; raw sockets cannot reach a provider.
"""

from __future__ import annotations

import asyncio
import base64
from email.message import Message
import io
import json
import os
import socket
import threading
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional
import urllib.error
import urllib.request
import urllib.response

from gateway.tee.sandbox_provider_socket_v2 import (
    MAX_SANDBOX_PROVIDER_FRAME_BYTES,
    SANDBOX_PROVIDER_SCHEMA_VERSION,
)
from leadpoet_canonical.attested_v2 import canonical_json


SOCKET_ENV = "LEADPOET_SANDBOX_PROVIDER_SOCKET"
EVIDENCE_CACHE_PATH_ENV = "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH"
EVIDENCE_MODE_ENV = "RESEARCH_LAB_PROVIDER_EVIDENCE_MODE"
EVIDENCE_MISS_SENTINEL = "RESEARCH_LAB_PROVIDER_EVIDENCE_MISS:"
SNAPSHOT_DIR_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_DIR"
PROVIDER_COST_SCOPE_ENV = "RESEARCH_LAB_PROVIDER_COST_SCOPE"
PROVIDER_COST_CAP_MICROUSD_ENV = "RESEARCH_LAB_PROVIDER_COST_CAP_MICROUSD"
PROVIDER_CALL_CAP_ENV = "RESEARCH_LAB_PROVIDER_CALL_CAP"
DEFAULT_TIMEOUT_MS = 30000
_CREDENTIAL_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
    "x-auth-token",
}
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


class SandboxHTTPShimV2Error(RuntimeError):
    """The sandbox could not obtain an authenticated provider terminal."""


def _evidence_mode() -> str:
    mode = str(os.getenv(EVIDENCE_MODE_ENV) or "live").strip().lower()
    if mode not in {"live", "cache_live", "record", "frozen"}:
        raise SandboxHTTPShimV2Error("provider evidence mode is invalid")
    return mode


def _evidence_cache() -> Dict[str, Dict[str, Any]]:
    path = str(os.getenv(EVIDENCE_CACHE_PATH_ENV) or "").strip()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception as exc:
        raise SandboxHTTPShimV2Error("provider evidence cache is unreadable") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "1.1"
        or not isinstance(document.get("entries"), dict)
    ):
        raise SandboxHTTPShimV2Error("provider evidence cache is invalid")
    output: Dict[str, Dict[str, Any]] = {}
    for fingerprint, record in document["entries"].items():
        if (
            isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and isinstance(record, dict)
            and isinstance(record.get("status"), int)
            and isinstance(record.get("body_b64"), str)
        ):
            output[fingerprint] = dict(record)
    return output


def _cached_terminal(
    *, method: str, url: str, body: bytes, mode: str, cache: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    if mode == "live":
        return None
    from research_lab.eval.provider_evidence_cache import (
        canonical_request_fingerprint,
    )

    fingerprint = canonical_request_fingerprint(method, url, body)
    record = cache.get(fingerprint)
    if isinstance(record, Mapping):
        return {
            "terminal_status": "attested_local_response",
            "http_status": int(record["status"]),
            "headers": {"content-type": "application/json"},
            "body_b64": str(record.get("body_b64") or ""),
            "failure_code": None,
            "provider_evidence_fingerprint": fingerprint,
        }
    if mode == "frozen":
        raise SandboxHTTPShimV2Error(EVIDENCE_MISS_SENTINEL + fingerprint)
    return None


def _snapshot_terminal(
    *, method: str, url: str, body: bytes
) -> Optional[Dict[str, Any]]:
    snapshot_dir = str(os.getenv(SNAPSHOT_DIR_ENV) or "").strip()
    if not snapshot_dir:
        return None
    from research_lab.eval.snapshot_store import (
        MISS_POLICY_STRICT,
        MODE_REPLAY,
        ProviderSnapshotStore,
        SnapshotMiss,
    )

    try:
        response = ProviderSnapshotStore(
            snapshot_dir,
            mode=MODE_REPLAY,
            miss_policy=MISS_POLICY_STRICT,
        ).replay(method, url, body=body)
    except SnapshotMiss:
        return None
    return {
        "terminal_status": "attested_local_response",
        "http_status": int(response.get("status") or 0),
        "headers": dict(response.get("headers") or {}),
        "body_b64": base64.b64encode(
            str(response.get("body_text") or "").encode("utf-8")
        ).decode("ascii"),
        "failure_code": None,
        "provider_snapshot_hit": True,
    }


def _recv_exact(connection: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(size - len(output))
        if not chunk:
            raise SandboxHTTPShimV2Error("provider socket response is incomplete")
        output.extend(chunk)
    return bytes(output)


def _headers(headers: Mapping[str, Any]) -> Dict[str, str]:
    return {
        str(name): str(value)
        for name, value in headers.items()
        if str(name).lower() not in _CREDENTIAL_HEADERS
    }


def execute(
    *,
    method: str,
    url: str,
    headers: Mapping[str, Any],
    body: bytes,
    timeout_ms: int,
) -> Dict[str, Any]:
    mode = _evidence_mode()
    snapshot = _snapshot_terminal(
        method=str(method).upper(),
        url=str(url),
        body=bytes(body),
    )
    if snapshot is not None:
        return snapshot
    cached = _cached_terminal(
        method=str(method).upper(),
        url=str(url),
        body=bytes(body),
        mode=mode,
        cache=_evidence_cache(),
    )
    if cached is not None:
        return cached
    socket_path = str(os.getenv(SOCKET_ENV) or "").strip()
    if not socket_path.startswith("/"):
        raise SandboxHTTPShimV2Error("provider socket is not configured")
    normalized_headers = _headers(headers)
    cost_scope = str(os.getenv(PROVIDER_COST_SCOPE_ENV) or "").strip()
    raw_cost_cap = str(os.getenv(PROVIDER_COST_CAP_MICROUSD_ENV) or "").strip()
    raw_call_cap = str(os.getenv(PROVIDER_CALL_CAP_ENV) or "").strip()
    if cost_scope:
        normalized_headers["X-Research-Lab-Cost-Scope"] = cost_scope
    if raw_cost_cap:
        try:
            cost_cap_microusd = int(raw_cost_cap)
        except ValueError as exc:
            raise SandboxHTTPShimV2Error("provider cost cap is invalid") from exc
        if cost_cap_microusd < 0:
            raise SandboxHTTPShimV2Error("provider cost cap is invalid")
        normalized_headers["X-Research-Lab-Cost-Cap-Usd"] = (
            f"{cost_cap_microusd / 1_000_000:.6f}"
        )
    if raw_call_cap:
        try:
            call_cap = int(raw_call_cap)
        except ValueError as exc:
            raise SandboxHTTPShimV2Error("provider call cap is invalid") from exc
        if call_cap < 1:
            raise SandboxHTTPShimV2Error("provider call cap is invalid")
        normalized_headers["X-Research-Lab-Tree-Provider-Call-Cap"] = str(
            call_cap
        )
    request = {
        "schema_version": SANDBOX_PROVIDER_SCHEMA_VERSION,
        "method": str(method).upper(),
        "url": str(url),
        "headers": normalized_headers,
        "body_b64": base64.b64encode(bytes(body)).decode("ascii"),
        "timeout_ms": max(1, int(timeout_ms)),
    }
    encoded = canonical_json(request).encode("utf-8")
    if len(encoded) > MAX_SANDBOX_PROVIDER_FRAME_BYTES:
        raise SandboxHTTPShimV2Error("provider socket request exceeds limit")
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(max(1.0, timeout_ms / 1000.0 + 5.0))
        connection.connect(socket_path)
        connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
        size = int.from_bytes(_recv_exact(connection, 4), "big")
        if size < 2 or size > MAX_SANDBOX_PROVIDER_FRAME_BYTES:
            raise SandboxHTTPShimV2Error("provider socket response exceeds limit")
        response = json.loads(_recv_exact(connection, size).decode("utf-8"))
    finally:
        connection.close()
    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, dict):
        raise SandboxHTTPShimV2Error(
            "provider socket failed: %s"
            % (response.get("error_code") if isinstance(response, dict) else "invalid")
        )
    return result


def _timeout_ms(value: Any) -> int:
    try:
        if value is None:
            return DEFAULT_TIMEOUT_MS
        if hasattr(value, "connect"):
            value = value.connect
        return max(1, int(float(value) * 1000))
    except Exception:
        return DEFAULT_TIMEOUT_MS


def _result_body(result: Mapping[str, Any]) -> bytes:
    if result.get("terminal_status") not in {
        "authenticated_response",
        "attested_local_response",
    }:
        raise SandboxHTTPShimV2Error(
            "attested transport failure: %s" % result.get("failure_code")
        )
    try:
        return base64.b64decode(str(result.get("body_b64") or ""), validate=True)
    except Exception as exc:
        raise SandboxHTTPShimV2Error("provider response body is invalid") from exc


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
    def __init__(self, *, url: str, result: Mapping[str, Any]) -> None:
        self.status = int(result["http_status"])
        self.headers = dict(result.get("headers") or {})
        self.url = url
        self.reason = "authenticated provider response"
        self.history = ()
        self.request_info = SimpleNamespace(real_url=url)
        self._body = _result_body(result)
        self.content = _AiohttpContent(self._body)

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


def _aiohttp_body(aiohttp: Any, session: Any, kwargs: Mapping[str, Any]) -> tuple[bytes, Dict[str, str]]:
    headers = _headers(
        {
            **dict(getattr(session, "headers", {}) or {}),
            **dict(kwargs.get("headers") or {}),
        }
    )
    json_body = kwargs.get("json")
    data = kwargs.get("data")
    if json_body is not None and data is not None:
        raise SandboxHTTPShimV2Error("aiohttp request has both data and json")
    if json_body is not None:
        payload = aiohttp.payload.JsonPayload(
            json_body,
            dumps=getattr(session, "_json_serialize", json.dumps),
        )
    elif data is not None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            payload = aiohttp.payload.BytesPayload(bytes(data))
        elif isinstance(data, str):
            payload = aiohttp.payload.StringPayload(data)
        else:
            payload = aiohttp.FormData(data)()
    else:
        return b"", headers
    value = getattr(payload, "_value", None)
    if not isinstance(value, (bytes, bytearray)):
        raise SandboxHTTPShimV2Error(
            "streaming or multipart aiohttp request bodies are unsupported"
        )
    headers.update(_headers(dict(payload.headers or {})))
    return bytes(value), headers


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        original_urlopen = urllib.request.urlopen

        def urlopen(request, *args, **kwargs):
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
            if str(url).startswith(("http://127.0.0.1", "http://localhost")):
                return original_urlopen(request, *args, **kwargs)
            if body is None:
                body = b""
            if not isinstance(body, (bytes, bytearray, memoryview)):
                raise TypeError("urllib request body must be bytes-like")
            result = execute(
                method=method,
                url=url,
                headers=headers,
                body=bytes(body),
                timeout_ms=_timeout_ms(timeout),
            )
            response_body = _result_body(result)
            message = Message()
            for name, value in dict(result.get("headers") or {}).items():
                message[str(name)] = str(value)
            status = int(result["http_status"])
            if status >= 400:
                raise urllib.error.HTTPError(
                    url,
                    status,
                    "authenticated provider error",
                    message,
                    io.BytesIO(response_body),
                )
            return urllib.response.addinfourl(
                io.BytesIO(response_body), message, url, status
            )

        urllib.request.urlopen = urlopen

        try:
            import httpx

            original_sync_send = httpx.Client.send
            original_async_send = httpx.AsyncClient.send

            def sync_send(client, request, *args, **kwargs):
                if str(request.url).startswith(("http://127.0.0.1", "http://localhost")):
                    return original_sync_send(client, request, *args, **kwargs)
                result = execute(
                    method=request.method,
                    url=str(request.url),
                    headers=dict(request.headers),
                    body=bytes(request.content),
                    timeout_ms=DEFAULT_TIMEOUT_MS,
                )
                return httpx.Response(
                    status_code=int(result["http_status"]),
                    headers=dict(result.get("headers") or {}),
                    content=_result_body(result),
                    request=request,
                )

            async def async_send(client, request, *args, **kwargs):
                return await asyncio.to_thread(
                    sync_send, client, request, *args, **kwargs
                )

            httpx.Client.send = sync_send
            httpx.AsyncClient.send = async_send
        except ImportError:
            pass

        try:
            import requests

            original_requests_send = requests.Session.send

            def requests_send(session, request, **kwargs):
                if str(request.url).startswith(("http://127.0.0.1", "http://localhost")):
                    return original_requests_send(session, request, **kwargs)
                body = request.body or b""
                if isinstance(body, str):
                    body = body.encode("utf-8")
                result = execute(
                    method=request.method,
                    url=str(request.url),
                    headers=dict(request.headers),
                    body=bytes(body),
                    timeout_ms=_timeout_ms(kwargs.get("timeout")),
                )
                response = requests.Response()
                response.status_code = int(result["http_status"])
                response.headers.update(dict(result.get("headers") or {}))
                response._content = _result_body(result)
                response.url = str(request.url)
                response.request = request
                return response

            requests.Session.send = requests_send
        except ImportError:
            pass

        try:
            import aiohttp
            from yarl import URL

            original_aiohttp_request = aiohttp.ClientSession._request

            async def aiohttp_request(session, method, str_or_url, *args, **kwargs):
                url = URL(str(str_or_url))
                if kwargs.get("params") is not None:
                    url = url.extend_query(kwargs["params"])
                if str(url).startswith(("http://127.0.0.1", "http://localhost")):
                    return await original_aiohttp_request(
                        session, method, str_or_url, *args, **kwargs
                    )
                body, headers = _aiohttp_body(aiohttp, session, kwargs)
                result = await asyncio.to_thread(
                    execute,
                    method=str(method),
                    url=str(url),
                    headers=headers,
                    body=body,
                    timeout_ms=_timeout_ms(kwargs.get("timeout")),
                )
                response = _AiohttpResponse(url=str(url), result=result)
                raise_for_status = kwargs.get("raise_for_status")
                if raise_for_status is None:
                    raise_for_status = getattr(session, "_raise_for_status", False)
                if raise_for_status is True:
                    response.raise_for_status()
                elif callable(raise_for_status):
                    await raise_for_status(response)
                return response

            aiohttp.ClientSession._request = aiohttp_request
        except ImportError:
            pass

        _INSTALLED = True
