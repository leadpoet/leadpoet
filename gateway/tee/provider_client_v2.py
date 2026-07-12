"""Runner-side HTTP interception for the attested coordinator provider broker."""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
import contextvars
from email.message import Message
import hashlib
import io
import json
import re
import sys
import threading
from typing import Any, Callable, Dict, Mapping, Optional
import urllib.error
import urllib.request
import urllib.response
from urllib.parse import urljoin, urlsplit

from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, validate_transport_attempt
from gateway.tee.source_add_runtime_v2 import (
    source_add_dynamic_retry_policy_hash,
    source_add_route_for_url_v2,
    validate_source_add_runtime_catalog_v2,
)


class ProviderClientV2Error(RuntimeError):
    """A runner request lacks an authenticated coordinator terminal record."""


_ACCEPTED_RESPONSE_TERMINALS = frozenset(
    {"authenticated_response", "attested_local_response"}
)


class _BrokeredAiohttpContent:
    """Small StreamReader-compatible view over an authenticated response body."""

    def __init__(self, body: bytes) -> None:
        self._body = bytes(body)
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or int(size) < 0:
            output = self._body[self._offset :]
            self._offset = len(self._body)
            return output
        end = min(len(self._body), self._offset + int(size))
        output = self._body[self._offset : end]
        self._offset = end
        return output

    async def readany(self) -> bytes:
        return await self.read()

    async def readexactly(self, size: int) -> bytes:
        output = await self.read(size)
        if len(output) != size:
            raise asyncio.IncompleteReadError(output, size)
        return output

    def at_eof(self) -> bool:
        return self._offset >= len(self._body)

    async def iter_chunked(self, size: int):
        while not self.at_eof():
            yield await self.read(size)

    def __aiter__(self):
        return self.iter_chunked(64 * 1024)


class _BrokeredAiohttpResponse:
    """The subset of ClientResponse used by the measured production closure."""

    def __init__(
        self,
        *,
        method: str,
        url: str,
        status: int,
        headers: Mapping[str, Any],
        body: bytes,
        history: tuple = (),
    ) -> None:
        import aiohttp
        from multidict import CIMultiDict, CIMultiDictProxy
        from yarl import URL

        normalized_headers = CIMultiDict(
            (str(name), str(value)) for name, value in headers.items()
        )
        self.method = str(method).upper()
        self.url = URL(str(url))
        self.real_url = self.url
        self.status = int(status)
        self.reason = aiohttp.http.RESPONSES.get(self.status, ("",))[0]
        self.headers = CIMultiDictProxy(normalized_headers)
        self.raw_headers = tuple(
            (str(name).encode("utf-8"), str(value).encode("utf-8"))
            for name, value in normalized_headers.items()
        )
        self._body = bytes(body)
        self.content = _BrokeredAiohttpContent(self._body)
        self.history = tuple(history)
        self.closed = False
        self.connection = None
        self.cookies = {}
        self.request_info = aiohttp.RequestInfo(
            self.url,
            self.method,
            self.headers,
            self.url,
        )

    @property
    def ok(self) -> bool:
        return self.status < 400

    @property
    def content_type(self) -> str:
        value = self.headers.get("content-type", "application/octet-stream")
        return value.split(";", 1)[0].strip().lower()

    @property
    def charset(self) -> Optional[str]:
        value = self.headers.get("content-type", "")
        for item in value.split(";")[1:]:
            name, separator, raw = item.partition("=")
            if separator and name.strip().lower() == "charset":
                return raw.strip().strip('"') or None
        return None

    def get_encoding(self) -> str:
        return self.charset or "utf-8"

    async def read(self) -> bytes:
        return self._body

    async def text(self, encoding: Optional[str] = None, errors: str = "strict") -> str:
        return self._body.decode(encoding or self.get_encoding(), errors=errors)

    async def json(
        self,
        *,
        encoding: Optional[str] = None,
        loads: Callable[[str], Any] = json.loads,
        content_type: Optional[str] = "application/json",
    ) -> Any:
        import aiohttp

        if content_type and content_type.lower() not in self.content_type:
            raise aiohttp.ContentTypeError(
                self.request_info,
                self.history,
                status=self.status,
                message="unexpected mimetype: %s" % self.content_type,
                headers=self.headers,
            )
        return loads(await self.text(encoding=encoding))

    def raise_for_status(self) -> None:
        if self.status < 400:
            return
        import aiohttp

        raise aiohttp.ClientResponseError(
            self.request_info,
            self.history,
            status=self.status,
            message=self.reason,
            headers=self.headers,
        )

    def release(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True

    async def wait_for_close(self) -> None:
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.release()


_PROVIDER_BY_HOST = {
    "openrouter.ai": "openrouter",
    "api.exa.ai": "exa",
    "api.scrapingdog.com": "scrapingdog",
    "code.deepline.com": "deepline",
    "archive.org": "wayback",
    "web.archive.org": "wayback",
    "api.truelist.io": "truelist",
    "cloudflare-dns.com": "dns",
    "rdap.org": "rdap",
    "entrypoint-finney.opentensor.ai": "bittensor_chain",
}
_CREDENTIAL_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "x-auth-token",
}
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _provider_id(url: str) -> str:
    parsed = urlsplit(str(url))
    if parsed.scheme != "https" or not parsed.hostname:
        raise ProviderClientV2Error("runner external request must use HTTPS")
    if parsed.hostname.lower() == "openrouter.ai" and (
        (parsed.path or "").startswith("/api/v1/workspaces")
        or (parsed.path or "").startswith("/api/v1/keys")
    ):
        return "openrouter_management"
    return _PROVIDER_BY_HOST.get(parsed.hostname.lower(), "public_web")


def _loopback_url(url: str) -> bool:
    parsed = urlsplit(str(url))
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _headers_without_credentials(headers: Mapping[str, Any]) -> Dict[str, str]:
    return {
        str(name): str(value)
        for name, value in headers.items()
        if str(name).lower() not in _CREDENTIAL_HEADERS
    }


def _timeout_ms(value: Any, default_ms: int) -> int:
    try:
        if value is None:
            return int(default_ms)
        if hasattr(value, "total"):
            value = value.total
        if isinstance(value, tuple):
            values = [float(item) for item in value if item is not None]
            value = max(values) if values else default_ms / 1000.0
        return max(1, int(float(value) * 1000))
    except Exception:
        return int(default_ms)


class _PayloadWriter:
    def __init__(self) -> None:
        self.chunks = []

    async def write(self, chunk: Any) -> None:
        self.chunks.append(bytes(chunk))


async def _aiohttp_payload(session: Any, *, data: Any, json_value: Any):
    import aiohttp

    if json_value is not None and data is not None:
        raise ValueError("data and json parameters can not be used at the same time")
    if json_value is not None:
        payload = aiohttp.payload.JsonPayload(
            json_value,
            dumps=session._json_serialize,
        )
    elif data is None:
        return b"", {}
    elif isinstance(data, aiohttp.FormData):
        payload = data()
    elif isinstance(data, Mapping) or (
        isinstance(data, (list, tuple))
        and data
        and isinstance(data[0], (list, tuple))
    ):
        payload = aiohttp.FormData(data)()
    else:
        payload = aiohttp.payload.PAYLOAD_REGISTRY.get(data, disposition=None)
    writer = _PayloadWriter()
    await payload.write(writer)
    return b"".join(writer.chunks), dict(payload.headers)


class _ExecutionScope:
    def __init__(
        self,
        *,
        job_id: str,
        purpose: str,
        logical_operation_id: str,
        retry_policy_hashes: Mapping[str, str],
        default_timeout_ms: int,
        terminal_sink: Optional[Callable[[Mapping[str, Any]], None]],
        artifact_sink: Optional[Callable[[str], None]] = None,
        allow_transport_failures: bool = False,
        dynamic_provider_catalog: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.job_id = str(job_id)
        self.purpose = str(purpose)
        self.logical_operation_id = str(logical_operation_id)
        self.retry_policy_hashes = dict(retry_policy_hashes)
        self.default_timeout_ms = int(default_timeout_ms)
        self.terminal_sink = terminal_sink
        self.artifact_sink = artifact_sink
        self.allow_transport_failures = bool(allow_transport_failures)
        self.dynamic_provider_catalog = (
            validate_source_add_runtime_catalog_v2(dynamic_provider_catalog)
            if dynamic_provider_catalog is not None
            else None
        )
        self.attempts = {}  # type: Dict[str, int]
        self.request_intents = set()
        self.terminals = {}
        self.lock = threading.Lock()

    def next_attempt(self, fingerprint: str) -> int:
        with self.lock:
            value = self.attempts.get(fingerprint, 0)
            self.attempts[fingerprint] = value + 1
            return value

    def record_intent(self, logical_operation_id: str, attempt_number: int) -> None:
        with self.lock:
            key = (str(logical_operation_id), int(attempt_number))
            if key in self.request_intents:
                raise ProviderClientV2Error("provider attempt intent is duplicated")
            self.request_intents.add(key)

    def record_terminal(
        self,
        logical_operation_id: str,
        attempt_number: int,
        terminal_status: str,
    ) -> None:
        with self.lock:
            key = (str(logical_operation_id), int(attempt_number))
            if key not in self.request_intents or key in self.terminals:
                raise ProviderClientV2Error("provider terminal does not bind one intent")
            self.terminals[key] = str(terminal_status)

    def assert_accepted_result_is_complete(self) -> None:
        with self.lock:
            missing = self.request_intents - set(self.terminals)
            if missing:
                raise ProviderClientV2Error(
                    "provider request is missing a signed terminal record"
                )
            latest = {}
            for (logical_operation_id, attempt_number), status in self.terminals.items():
                current = latest.get(logical_operation_id)
                if current is None or attempt_number > current[0]:
                    latest[logical_operation_id] = (attempt_number, status)
            failed = sorted(
                logical_operation_id
                for logical_operation_id, (_attempt, status) in latest.items()
                if status not in _ACCEPTED_RESPONSE_TERMINALS
            )
            if failed and not self.allow_transport_failures:
                raise ProviderClientV2Error(
                    "provider transport did not authenticate a terminal response"
                )


class BrokeredProviderTransportV2:
    """Install process-wide low-level hooks, scoped by ContextVar per job."""

    def __init__(
        self,
        execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        terminal_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        self._execute = execute
        self._terminal_sink = terminal_sink
        self._scope = contextvars.ContextVar("leadpoet_provider_scope_v2", default=None)
        self._installed = False
        self._lock = threading.Lock()
        self._restore = []

    @contextmanager
    def scope(
        self,
        *,
        job_id: str,
        purpose: str,
        logical_operation_id: str,
        retry_policy_hashes: Mapping[str, str],
        default_timeout_ms: int = 30000,
        terminal_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        artifact_sink: Optional[Callable[[str], None]] = None,
        allow_transport_failures: bool = False,
        dynamic_provider_catalog: Optional[Mapping[str, Any]] = None,
    ):
        self.install()
        scope = self.create_scope(
            job_id=job_id,
            purpose=purpose,
            logical_operation_id=logical_operation_id,
            retry_policy_hashes=retry_policy_hashes,
            default_timeout_ms=default_timeout_ms,
            terminal_sink=terminal_sink,
            artifact_sink=artifact_sink,
            allow_transport_failures=allow_transport_failures,
            dynamic_provider_catalog=dynamic_provider_catalog,
        )
        with self.activate_scope(scope):
            try:
                yield scope
            except Exception:
                raise
            else:
                scope.assert_accepted_result_is_complete()

    def create_scope(
        self,
        *,
        job_id: str,
        purpose: str,
        logical_operation_id: str,
        retry_policy_hashes: Mapping[str, str],
        default_timeout_ms: int = 30000,
        terminal_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        artifact_sink: Optional[Callable[[str], None]] = None,
        allow_transport_failures: bool = False,
        dynamic_provider_catalog: Optional[Mapping[str, Any]] = None,
    ) -> _ExecutionScope:
        return _ExecutionScope(
            job_id=job_id,
            purpose=purpose,
            logical_operation_id=logical_operation_id,
            retry_policy_hashes=retry_policy_hashes,
            default_timeout_ms=default_timeout_ms,
            terminal_sink=terminal_sink or self._terminal_sink,
            artifact_sink=artifact_sink,
            allow_transport_failures=allow_transport_failures,
            dynamic_provider_catalog=dynamic_provider_catalog,
        )

    @contextmanager
    def activate_scope(self, scope: _ExecutionScope):
        if not isinstance(scope, _ExecutionScope):
            raise ProviderClientV2Error("provider execution scope is invalid")
        token = self._scope.set(scope)
        try:
            yield scope
        finally:
            self._scope.reset(token)

    def execute_http(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, Any],
        body: bytes,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute one request under the currently activated measured scope."""

        return self._execute_request(
            method=method,
            url=url,
            headers=headers,
            body=body,
            timeout_ms=timeout_ms,
        )

    def install(self) -> None:
        with self._lock:
            if self._installed:
                return
            original_run_in_executor = asyncio.BaseEventLoop.run_in_executor

            def _run_in_executor(loop, executor, function, *args):
                # Qualification uses executor threads for WHOIS, DNS, and sync
                # HTTP clients. Carry the per-job ContextVar into those threads
                # so concurrent enclave jobs remain independently attributed.
                context = contextvars.copy_context()

                def _invoke():
                    return context.run(function, *args)

                return original_run_in_executor(loop, executor, _invoke)

            asyncio.BaseEventLoop.run_in_executor = _run_in_executor
            self._restore.append(
                lambda: setattr(
                    asyncio.BaseEventLoop,
                    "run_in_executor",
                    original_run_in_executor,
                )
            )

            original_urlopen = urllib.request.urlopen

            def _urlopen(request, *args, **kwargs):
                scope = self._scope.get()
                if scope is None:
                    url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
                    if _loopback_url(url):
                        return original_urlopen(request, *args, **kwargs)
                    raise ProviderClientV2Error(
                        "runner external HTTP is outside an attested job"
                    )
                return self._urllib_response(
                    request,
                    timeout=kwargs.get("timeout"),
                )

            urllib.request.urlopen = _urlopen
            self._restore.append(
                lambda: setattr(urllib.request, "urlopen", original_urlopen)
            )

            original_opener_open = urllib.request.OpenerDirector.open

            def _opener_open(opener, request, *args, **kwargs):
                scope = self._scope.get()
                url = (
                    request.full_url
                    if isinstance(request, urllib.request.Request)
                    else str(request)
                )
                if scope is None:
                    if _loopback_url(url):
                        return original_opener_open(opener, request, *args, **kwargs)
                    raise ProviderClientV2Error(
                        "runner custom urllib opener is outside an attested job"
                    )
                timeout = kwargs.get("timeout")
                if timeout is None and args:
                    timeout = args[0]
                return self._urllib_response(request, timeout=timeout)

            urllib.request.OpenerDirector.open = _opener_open
            self._restore.append(
                lambda: setattr(
                    urllib.request.OpenerDirector,
                    "open",
                    original_opener_open,
                )
            )

            import httpx

            original_sync_send = httpx.Client.send
            original_async_send = httpx.AsyncClient.send

            def _sync_send(client, request, *args, **kwargs):
                if self._scope.get() is None:
                    if _loopback_url(str(request.url)):
                        return original_sync_send(client, request, *args, **kwargs)
                    raise ProviderClientV2Error(
                        "runner external HTTPX is outside an attested job"
                    )
                return self._httpx_response(request)

            async def _async_send(client, request, *args, **kwargs):
                if self._scope.get() is None:
                    if _loopback_url(str(request.url)):
                        return await original_async_send(client, request, *args, **kwargs)
                    raise ProviderClientV2Error(
                        "runner external HTTPX is outside an attested job"
                    )
                context = contextvars.copy_context()
                return await asyncio.to_thread(context.run, self._httpx_response, request)

            httpx.Client.send = _sync_send
            httpx.AsyncClient.send = _async_send
            self._restore.extend(
                (
                    lambda: setattr(httpx.Client, "send", original_sync_send),
                    lambda: setattr(httpx.AsyncClient, "send", original_async_send),
                )
            )

            import requests

            original_requests_send = requests.Session.send

            def _requests_send(session, request, **kwargs):
                scope = self._scope.get()
                if scope is None:
                    if _loopback_url(str(request.url)):
                        return original_requests_send(session, request, **kwargs)
                    raise ProviderClientV2Error(
                        "runner external requests call is outside an attested job"
                    )
                body = request.body or b""
                if isinstance(body, str):
                    body = body.encode("utf-8")
                if not isinstance(body, (bytes, bytearray, memoryview)):
                    raise ProviderClientV2Error(
                        "streaming requests bodies are unsupported"
                    )
                result = self._execute_request(
                    method=request.method,
                    url=str(request.url),
                    headers=dict(request.headers),
                    body=bytes(body),
                    timeout_ms=_timeout_ms(kwargs.get("timeout"), scope.default_timeout_ms),
                )
                if result.get("terminal_status") not in _ACCEPTED_RESPONSE_TERMINALS:
                    raise requests.exceptions.ConnectionError(
                        "attested transport failure: %s" % result.get("failure_code"),
                        request=request,
                    )
                response = requests.Response()
                response.status_code = int(result["http_status"])
                response.headers.update(dict(result.get("headers") or {}))
                response._content = base64.b64decode(
                    str(result["body_b64"]), validate=True
                )
                response.url = str(request.url)
                response.request = request
                response.raw = io.BytesIO(response._content)
                response.encoding = requests.utils.get_encoding_from_headers(
                    response.headers
                )
                return response

            requests.Session.send = _requests_send
            self._restore.append(
                lambda: setattr(requests.Session, "send", original_requests_send)
            )

            import aiohttp

            original_aiohttp_request = aiohttp.ClientSession._request

            async def _aiohttp_request(session, method, url, *args, **kwargs):
                scope = self._scope.get()
                if scope is None:
                    if _loopback_url(str(url)):
                        return await original_aiohttp_request(
                            session, method, url, *args, **kwargs
                        )
                    raise ProviderClientV2Error(
                        "runner external aiohttp call is outside an attested job"
                    )
                if args:
                    raise ProviderClientV2Error(
                        "positional aiohttp request options are unsupported"
                    )
                return await self._aiohttp_response(
                    session=session,
                    method=str(method),
                    url=str(url),
                    options=kwargs,
                )

            aiohttp.ClientSession._request = _aiohttp_request
            self._restore.append(
                lambda: setattr(
                    aiohttp.ClientSession,
                    "_request",
                    original_aiohttp_request,
                )
            )
            self._installed = True

    def restore(self) -> None:
        with self._lock:
            while self._restore:
                self._restore.pop()()
            self._installed = False

    def _execute_request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, Any],
        body: bytes,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        scope = self._scope.get()
        if scope is None:
            raise ProviderClientV2Error("provider request is outside an attested job")
        dynamic_route = None
        if scope.dynamic_provider_catalog is not None:
            dynamic_route = source_add_route_for_url_v2(
                scope.dynamic_provider_catalog,
                str(method).upper(),
                str(url),
            )
        provider_id = (
            str(dynamic_route["provider_id"])
            if dynamic_route is not None
            else _provider_id(url)
        )
        sanitized_headers = _headers_without_credentials(headers)
        fingerprint = sha256_bytes(
            (
                str(method).upper()
                + "\n"
                + str(url)
                + "\n"
                + sha256_bytes(body)
            ).encode("utf-8")
        )
        attempt_number = scope.next_attempt(fingerprint)
        retry_policy_hash = scope.retry_policy_hashes.get(provider_id)
        if not retry_policy_hash:
            raise ProviderClientV2Error("provider retry policy is not configured")
        if dynamic_route is not None and retry_policy_hash != (
            source_add_dynamic_retry_policy_hash(dynamic_route)
        ):
            raise ProviderClientV2Error(
                "dynamic provider retry policy differs from measured route"
            )
        logical_operation_id = "%s:%s" % (
            scope.logical_operation_id,
            fingerprint.split(":", 1)[1][:16],
        )
        scope.record_intent(logical_operation_id, attempt_number)
        broker_request = {
            "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
            "logical_operation_id": logical_operation_id,
            "job_id": scope.job_id,
            "purpose": scope.purpose,
            "provider_id": provider_id,
            "attempt_number": attempt_number,
            "method": str(method).upper(),
            "url": str(url),
            "headers": sanitized_headers,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "timeout_ms": int(timeout_ms or scope.default_timeout_ms),
            "retry_policy_hash": retry_policy_hash,
        }
        if dynamic_route is not None:
            broker_request["dynamic_route"] = dict(dynamic_route)
        result = dict(self._execute(broker_request))
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise ProviderClientV2Error("coordinator omitted terminal transport record")
        validate_transport_attempt(attempt)
        if (
            attempt["job_id"] != scope.job_id
            or attempt["purpose"] != scope.purpose
            or attempt["provider_id"] != provider_id
            or attempt["attempt_number"] != attempt_number
        ):
            raise ProviderClientV2Error("coordinator terminal record binding mismatch")
        if scope.terminal_sink is not None:
            additional_attempts = result.get("additional_transport_attempts") or []
            if not isinstance(additional_attempts, list):
                raise ProviderClientV2Error(
                    "coordinator additional terminal records are invalid"
                )
            for additional in additional_attempts:
                if not isinstance(additional, Mapping):
                    raise ProviderClientV2Error(
                        "coordinator additional terminal record is invalid"
                    )
                validate_transport_attempt(additional)
                if (
                    additional["job_id"] != scope.job_id
                    or additional["purpose"] != scope.purpose
                ):
                    raise ProviderClientV2Error(
                        "coordinator additional terminal binding mismatch"
                    )
                scope.terminal_sink(dict(additional))
            scope.terminal_sink(dict(attempt))
        evidence_artifacts = result.get("evidence_artifact_hashes") or []
        if not isinstance(evidence_artifacts, list):
            raise ProviderClientV2Error(
                "coordinator evidence artifact commitments are invalid"
            )
        if scope.artifact_sink is not None:
            for artifact_hash in evidence_artifacts:
                if not _HASH_RE.fullmatch(str(artifact_hash or "")):
                    raise ProviderClientV2Error(
                        "coordinator evidence artifact hash is invalid"
                    )
                scope.artifact_sink(str(artifact_hash))
        scope.record_terminal(
            logical_operation_id,
            attempt_number,
            str(attempt["terminal_status"]),
        )
        if (
            attempt["terminal_status"] in _ACCEPTED_RESPONSE_TERMINALS
            and not _HASH_RE.fullmatch(str(result.get("encrypted_artifact_id") or ""))
        ):
            raise ProviderClientV2Error(
                "coordinator omitted encrypted provider artifact"
            )
        if not _HASH_RE.fullmatch(
            str(result.get("encrypted_request_artifact_id") or "")
        ):
            raise ProviderClientV2Error(
                "coordinator omitted encrypted provider request artifact"
            )
        return result

    async def _aiohttp_response(
        self,
        *,
        session: Any,
        method: str,
        url: str,
        options: Mapping[str, Any],
    ) -> _BrokeredAiohttpResponse:
        import aiohttp
        from multidict import CIMultiDict
        from yarl import URL

        if session.closed:
            raise RuntimeError("Session is closed")
        options = dict(options)
        params = options.pop("params", None)
        data = options.pop("data", None)
        json_value = options.pop("json", None)
        request_headers = options.pop("headers", None)
        cookies = options.pop("cookies", None)
        auth = options.pop("auth", None)
        allow_redirects = bool(options.pop("allow_redirects", True))
        max_redirects = int(options.pop("max_redirects", 10))
        timeout = options.pop("timeout", None)
        raise_for_status = options.pop("raise_for_status", None)
        skip_auto_headers = {
            str(item).lower()
            for item in (options.pop("skip_auto_headers", None) or ())
        }
        # Proxy and TLS options belong to the coordinator, not the runner.
        for ignored in (
            "proxy",
            "proxy_auth",
            "proxy_headers",
            "ssl",
            "verify_ssl",
            "ssl_context",
            "fingerprint",
            "server_hostname",
            "trace_request_ctx",
            "read_until_eof",
            "read_bufsize",
            "auto_decompress",
            "max_line_size",
            "max_field_size",
            "compress",
            "chunked",
            "expect100",
        ):
            options.pop(ignored, None)
        if options:
            raise ProviderClientV2Error(
                "unsupported aiohttp request options: %s"
                % ",".join(sorted(options))
            )

        request_url = URL(url)
        if params is not None:
            request_url = request_url.extend_query(params)
        headers = CIMultiDict(session.headers)
        if request_headers:
            headers.update(request_headers)
        if auth is not None and "authorization" not in {
            str(name).lower() for name in headers
        }:
            headers["Authorization"] = auth.encode()
        if cookies:
            from http.cookies import SimpleCookie

            cookie = SimpleCookie()
            cookie.load(cookies)
            headers["Cookie"] = cookie.output(header="", sep=";").strip()
        body, payload_headers = await _aiohttp_payload(
            session,
            data=data,
            json_value=json_value,
        )
        for name, value in payload_headers.items():
            if name not in headers:
                headers[name] = value
        defaults = {
            "accept": "*/*",
            "accept-encoding": "identity",
            "user-agent": "Python/%s.%s aiohttp/%s"
            % (sys.version_info[0], sys.version_info[1], aiohttp.__version__),
        }
        existing_names = {str(name).lower() for name in headers}
        for name, value in defaults.items():
            if name not in skip_auto_headers and name not in existing_names:
                headers[name] = value
        if body and "content-length" not in {
            str(name).lower() for name in headers
        }:
            headers["Content-Length"] = str(len(body))

        scope = self._scope.get()
        if scope is None:  # pragma: no cover - guarded by caller
            raise ProviderClientV2Error("aiohttp request scope disappeared")
        timeout_ms = _timeout_ms(timeout, scope.default_timeout_ms)
        history = []
        current_method = str(method).upper()
        current_url = request_url
        current_body = body
        current_headers = headers
        for redirect_count in range(max_redirects + 1):
            context = contextvars.copy_context()
            result = await asyncio.to_thread(
                context.run,
                self._execute_request,
                method=current_method,
                url=str(current_url),
                headers=dict(current_headers),
                body=current_body,
                timeout_ms=timeout_ms,
            )
            if result.get("terminal_status") not in _ACCEPTED_RESPONSE_TERMINALS:
                raise aiohttp.ClientConnectionError(
                    "attested transport failure: %s" % result.get("failure_code")
                )
            response_body = base64.b64decode(
                str(result["body_b64"]), validate=True
            )
            response = _BrokeredAiohttpResponse(
                method=current_method,
                url=str(current_url),
                status=int(result["http_status"]),
                headers=dict(result.get("headers") or {}),
                body=response_body,
                history=tuple(history),
            )
            location = response.headers.get("location")
            if (
                not allow_redirects
                or response.status not in {301, 302, 303, 307, 308}
                or not location
            ):
                if raise_for_status is True:
                    response.raise_for_status()
                elif callable(raise_for_status):
                    await raise_for_status(response)
                return response
            if redirect_count >= max_redirects:
                raise aiohttp.TooManyRedirects(
                    response.request_info,
                    tuple(history + [response]),
                )
            history.append(response)
            next_url = URL(urljoin(str(current_url), str(location)))
            if next_url.scheme != "https":
                raise ProviderClientV2Error(
                    "authenticated redirect attempted plaintext transport"
                )
            if (
                response.status == 303
                and current_method != "HEAD"
            ) or (
                response.status in {301, 302}
                and current_method == "POST"
            ):
                current_method = "GET"
                current_body = b""
                for name in tuple(current_headers):
                    if str(name).lower() in {
                        "content-length",
                        "content-type",
                        "transfer-encoding",
                    }:
                        del current_headers[name]
            current_url = next_url

        raise ProviderClientV2Error("aiohttp redirect loop did not terminate")

    def _httpx_response(self, request: Any) -> Any:
        import httpx

        try:
            body = bytes(request.content)
        except Exception as exc:
            raise ProviderClientV2Error("streaming HTTPX requests are unsupported") from exc
        result = self._execute_request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            body=body,
        )
        if result.get("terminal_status") not in _ACCEPTED_RESPONSE_TERMINALS:
            raise httpx.TransportError(
                "attested transport failure: %s" % result.get("failure_code"),
                request=request,
            )
        return httpx.Response(
            status_code=int(result["http_status"]),
            headers=dict(result.get("headers") or {}),
            content=base64.b64decode(str(result["body_b64"]), validate=True),
            request=request,
        )

    def _urllib_response(self, request: Any, *, timeout: Any = None) -> Any:
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
        result = self._execute_request(
            method=method,
            url=url,
            headers=headers,
            body=bytes(body),
            timeout_ms=(int(float(timeout) * 1000) if timeout is not None else None),
        )
        if result.get("terminal_status") not in _ACCEPTED_RESPONSE_TERMINALS:
            raise urllib.error.URLError(
                "attested transport failure: %s" % result.get("failure_code")
            )
        response_body = base64.b64decode(str(result["body_b64"]), validate=True)
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
        response = urllib.response.addinfourl(
            io.BytesIO(response_body),
            message,
            url,
            status,
        )
        response.msg = "OK"
        return response
