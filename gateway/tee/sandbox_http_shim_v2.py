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
    socket_path = str(os.getenv(SOCKET_ENV) or "").strip()
    if not socket_path.startswith("/"):
        raise SandboxHTTPShimV2Error("provider socket is not configured")
    request = {
        "schema_version": SANDBOX_PROVIDER_SCHEMA_VERSION,
        "method": str(method).upper(),
        "url": str(url),
        "headers": _headers(headers),
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


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        original_urlopen = urllib.request.urlopen

        def urlopen(request, *args, **kwargs):
            if isinstance(request, urllib.request.Request):
                url = request.full_url
                method = request.get_method()
                headers = dict(request.header_items())
                body = request.data or b""
            else:
                url = str(request)
                method = "GET"
                headers = {}
                body = b""
            if str(url).startswith(("http://127.0.0.1", "http://localhost")):
                return original_urlopen(request, *args, **kwargs)
            result = execute(
                method=method,
                url=url,
                headers=headers,
                body=bytes(body),
                timeout_ms=_timeout_ms(kwargs.get("timeout")),
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

        _INSTALLED = True
