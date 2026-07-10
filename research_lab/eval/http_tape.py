"""Transient record/replay tape for attested scoring shadow execution.

The authoritative host records responses at the two HTTP seams used by the
qualification scorer (httpx and aiohttp).  The enclave replays those exact
bytes and never falls through to the network on a miss.  Tapes are transported
only inside the bounded scoring job; callers persist only their canonical hash.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import threading
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional

from research_lab.eval.snapshot_store import build_snapshot_request


HTTP_TAPE_SCHEMA_VERSION = "leadpoet.attested_scoring_http_tape.v1"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_TAPE_ENTRIES = 512
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "retry-after",
        "x-openrouter-generation-id",
        "x-openrouter-response-id",
    }
)


class HttpTapeError(RuntimeError):
    """A provider tape was invalid, incomplete, oversized, or replay-missed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _safe_headers(headers: Any) -> Dict[str, str]:
    output = {}
    try:
        items = headers.items()
    except Exception:
        items = ()
    for name, value in items:
        normalized = str(name or "").strip().lower()
        if normalized in _SAFE_RESPONSE_HEADERS:
            output[normalized] = str(value or "")[:4096]
    return dict(sorted(output.items()))


def _request_identity(method: str, url: str, *, params: Any = None, body: Any = None) -> Dict[str, str]:
    snapshot = build_snapshot_request(method, url, params=params, body=body)
    return {
        "request_key": snapshot.request_key,
        "provider": snapshot.provider,
        "method": snapshot.method,
        "endpoint": snapshot.endpoint,
        "params_hash": snapshot.params_hash,
    }


class HttpTapeRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries = []  # type: List[Dict[str, Any]]
        self._total_response_bytes = 0
        self._failure_type = None  # type: Optional[str]

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._failure_type is None:
                self._failure_type = type(exc).__name__

    def begin(
        self,
        method: str,
        url: str,
        *,
        params: Any = None,
        body: Any = None,
        status: int,
        headers: Any,
    ) -> int:
        identity = _request_identity(method, url, params=params, body=body)
        with self._lock:
            if len(self._entries) >= MAX_TAPE_ENTRIES:
                raise HttpTapeError("provider tape entry limit exceeded")
            occurrence = sum(
                1 for entry in self._entries
                if entry["request"]["request_key"] == identity["request_key"]
            )
            self._entries.append(
                {
                    "request": identity,
                    "occurrence": occurrence,
                    "response": {
                        "status": int(status),
                        "headers": _safe_headers(headers),
                        "body_b64": "",
                        "body_sha256": _sha256(b""),
                    },
                }
            )
            return len(self._entries) - 1

    def set_body(self, index: int, body: bytes) -> None:
        raw = bytes(body)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise HttpTapeError("provider tape response exceeds per-response limit")
        with self._lock:
            entry = self._entries[index]
            previous = base64.b64decode(entry["response"]["body_b64"] or "")
            projected = self._total_response_bytes - len(previous) + len(raw)
            if projected > MAX_TOTAL_RESPONSE_BYTES:
                raise HttpTapeError("provider tape response total exceeds limit")
            self._total_response_bytes = projected
            entry["response"]["body_b64"] = base64.b64encode(raw).decode("ascii")
            entry["response"]["body_sha256"] = _sha256(raw)

    def document(self) -> Dict[str, Any]:
        with self._lock:
            if self._failure_type is not None:
                raise HttpTapeError(
                    "provider tape recording failed: %s" % self._failure_type
                )
            body = {
                "schema_version": HTTP_TAPE_SCHEMA_VERSION,
                "entries": json.loads(json.dumps(self._entries)),
                "entry_count": len(self._entries),
                "total_response_bytes": self._total_response_bytes,
            }
        body["tape_hash"] = _sha256(_canonical_json(body))
        return body


def validate_http_tape(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HttpTapeError("provider tape must be an object")
    required = {
        "schema_version",
        "entries",
        "entry_count",
        "total_response_bytes",
        "tape_hash",
    }
    if set(value) != required or value.get("schema_version") != HTTP_TAPE_SCHEMA_VERSION:
        raise HttpTapeError("provider tape schema is invalid")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_TAPE_ENTRIES:
        raise HttpTapeError("provider tape entries are invalid")
    normalized_entries = []
    total_bytes = 0
    occurrence_counts = {}  # type: Dict[str, int]
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"request", "occurrence", "response"}:
            raise HttpTapeError("provider tape entry shape is invalid")
        request = entry.get("request")
        response = entry.get("response")
        if not isinstance(request, Mapping) or set(request) != {
            "request_key",
            "provider",
            "method",
            "endpoint",
            "params_hash",
        }:
            raise HttpTapeError("provider tape request shape is invalid")
        if not isinstance(response, Mapping) or set(response) != {
            "status",
            "headers",
            "body_b64",
            "body_sha256",
        }:
            raise HttpTapeError("provider tape response shape is invalid")
        request_key = str(request.get("request_key") or "")
        expected_occurrence = occurrence_counts.get(request_key, 0)
        if entry.get("occurrence") != expected_occurrence:
            raise HttpTapeError("provider tape occurrence order is invalid")
        occurrence_counts[request_key] = expected_occurrence + 1
        try:
            raw_body = base64.b64decode(str(response.get("body_b64") or ""), validate=True)
        except Exception as exc:
            raise HttpTapeError("provider tape response body is invalid") from exc
        if len(raw_body) > MAX_RESPONSE_BYTES or response.get("body_sha256") != _sha256(raw_body):
            raise HttpTapeError("provider tape response body hash is invalid")
        status = response.get("status")
        if not isinstance(status, int) or status < 100 or status > 599:
            raise HttpTapeError("provider tape response status is invalid")
        headers = response.get("headers")
        if not isinstance(headers, Mapping) or any(
            str(name).lower() not in _SAFE_RESPONSE_HEADERS for name in headers
        ):
            raise HttpTapeError("provider tape response headers are invalid")
        total_bytes += len(raw_body)
        normalized_entries.append(json.loads(json.dumps(entry)))
    if total_bytes > MAX_TOTAL_RESPONSE_BYTES:
        raise HttpTapeError("provider tape response total exceeds limit")
    normalized = {
        "schema_version": HTTP_TAPE_SCHEMA_VERSION,
        "entries": normalized_entries,
        "entry_count": len(normalized_entries),
        "total_response_bytes": total_bytes,
    }
    if value.get("entry_count") != len(entries) or value.get("total_response_bytes") != total_bytes:
        raise HttpTapeError("provider tape summary is invalid")
    if value.get("tape_hash") != _sha256(_canonical_json(normalized)):
        raise HttpTapeError("provider tape hash is invalid")
    normalized["tape_hash"] = value["tape_hash"]
    return normalized


class HttpTapeReplayer:
    def __init__(self, tape: Mapping[str, Any]) -> None:
        self.tape = validate_http_tape(tape)
        self._lock = threading.Lock()
        self._by_key = {}  # type: Dict[str, List[Dict[str, Any]]]
        self._positions = {}  # type: Dict[str, int]
        for entry in self.tape["entries"]:
            self._by_key.setdefault(entry["request"]["request_key"], []).append(entry)

    def take(self, method: str, url: str, *, params: Any = None, body: Any = None) -> Dict[str, Any]:
        identity = _request_identity(method, url, params=params, body=body)
        key = identity["request_key"]
        with self._lock:
            position = self._positions.get(key, 0)
            entries = self._by_key.get(key, [])
            if position >= len(entries):
                raise HttpTapeError("provider tape replay miss")
            entry = entries[position]
            self._positions[key] = position + 1
        if dict(entry["request"]) != identity:
            raise HttpTapeError("provider tape request identity mismatch")
        response = dict(entry["response"])
        response["body"] = base64.b64decode(response.pop("body_b64"), validate=True)
        return response

    def assert_consumed(self) -> None:
        for key, entries in self._by_key.items():
            if self._positions.get(key, 0) != len(entries):
                raise HttpTapeError("provider tape contains unconsumed responses")


_ACTIVE_RECORDER = ContextVar("attested_http_tape_recorder", default=None)
_ACTIVE_REPLAYER = ContextVar("attested_http_tape_replayer", default=None)
_HOOK_LOCK = threading.Lock()
_HOOKS_INSTALLED = False


def _install_hooks() -> None:
    global _HOOKS_INSTALLED
    with _HOOK_LOCK:
        if _HOOKS_INSTALLED:
            return
        _install_httpx_hooks()
        _install_aiohttp_hooks()
        _HOOKS_INSTALLED = True


def _install_httpx_hooks() -> None:
    import httpx

    original_async_send = httpx.AsyncClient.send
    original_sync_send = httpx.Client.send

    async def async_send(client, request, *args, **kwargs):
        replayer = _ACTIVE_REPLAYER.get()
        request_body = bytes(getattr(request, "content", b"") or b"")
        if replayer is not None:
            replay = replayer.take(str(request.method), str(request.url), body=request_body)
            return httpx.Response(
                status_code=replay["status"],
                headers=replay["headers"],
                content=replay["body"],
                request=request,
            )
        response = await original_async_send(client, request, *args, **kwargs)
        recorder = _ACTIVE_RECORDER.get()
        if recorder is not None:
            body = await response.aread()
            try:
                index = recorder.begin(
                    str(request.method),
                    str(request.url),
                    body=request_body,
                    status=response.status_code,
                    headers=response.headers,
                )
                recorder.set_body(index, body)
            except Exception as exc:
                recorder.fail(exc)
        return response

    def sync_send(client, request, *args, **kwargs):
        replayer = _ACTIVE_REPLAYER.get()
        request_body = bytes(getattr(request, "content", b"") or b"")
        if replayer is not None:
            replay = replayer.take(str(request.method), str(request.url), body=request_body)
            return httpx.Response(
                status_code=replay["status"],
                headers=replay["headers"],
                content=replay["body"],
                request=request,
            )
        response = original_sync_send(client, request, *args, **kwargs)
        recorder = _ACTIVE_RECORDER.get()
        if recorder is not None:
            body = response.read()
            try:
                index = recorder.begin(
                    str(request.method),
                    str(request.url),
                    body=request_body,
                    status=response.status_code,
                    headers=response.headers,
                )
                recorder.set_body(index, body)
            except Exception as exc:
                recorder.fail(exc)
        return response

    httpx.AsyncClient.send = async_send
    httpx.Client.send = sync_send


class _RecordingAiohttpContent:
    def __init__(self, inner: Any, recorder: HttpTapeRecorder, index: int) -> None:
        self._inner = inner
        self._recorder = recorder
        self._index = index
        self._body = bytearray()

    async def read(self, size: int = -1) -> bytes:
        chunk = await self._inner.read(size)
        self._body.extend(chunk)
        try:
            self._recorder.set_body(self._index, bytes(self._body))
        except Exception as exc:
            self._recorder.fail(exc)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RecordingAiohttpResponse:
    def __init__(self, inner: Any, recorder: HttpTapeRecorder, index: int) -> None:
        self._inner = inner
        self._recorder = recorder
        self._index = index
        self.content = _RecordingAiohttpContent(inner.content, recorder, index)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
        await self.wait_for_close()

    async def read(self) -> bytes:
        body = await self._inner.read()
        try:
            self._recorder.set_body(self._index, body)
        except Exception as exc:
            self._recorder.fail(exc)
        return body

    async def text(self, *args: Any, **kwargs: Any) -> str:
        value = await self._inner.text(*args, **kwargs)
        body = getattr(self._inner, "_body", None)
        if isinstance(body, bytes):
            try:
                self._recorder.set_body(self._index, body)
            except Exception as exc:
                self._recorder.fail(exc)
        return value

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        value = await self._inner.json(*args, **kwargs)
        body = getattr(self._inner, "_body", None)
        if isinstance(body, bytes):
            try:
                self._recorder.set_body(self._index, body)
            except Exception as exc:
                self._recorder.fail(exc)
        return value

    def release(self) -> Any:
        return self._inner.release()

    def close(self) -> Any:
        return self._inner.close()

    async def wait_for_close(self) -> Any:
        return await self._inner.wait_for_close()


class _ReplayAiohttpContent:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._offset
        start = self._offset
        self._offset = min(len(self._body), self._offset + size)
        return self._body[start:self._offset]


class _ReplayAiohttpResponse:
    def __init__(self, *, url: str, response: Mapping[str, Any]) -> None:
        self.status = int(response["status"])
        self.headers = dict(response["headers"])
        self.url = url
        self._body = bytes(response["body"])
        self.content = _ReplayAiohttpContent(self._body)
        self.history = ()
        self.request_info = SimpleNamespace(real_url=url)

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
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
            message="replayed provider response",
            headers=self.headers,
        )

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_for_close(self) -> None:
        return None


def _install_aiohttp_hooks() -> None:
    import aiohttp

    original_request = aiohttp.ClientSession._request

    async def request(session, method, str_or_url, *args, **kwargs):
        replayer = _ACTIVE_REPLAYER.get()
        params = kwargs.get("params")
        body = kwargs.get("json") if kwargs.get("json") is not None else kwargs.get("data")
        if replayer is not None:
            replay = replayer.take(str(method), str(str_or_url), params=params, body=body)
            return _ReplayAiohttpResponse(url=str(str_or_url), response=replay)
        response = await original_request(session, method, str_or_url, *args, **kwargs)
        recorder = _ACTIVE_RECORDER.get()
        if recorder is None:
            return response
        try:
            index = recorder.begin(
                str(method),
                str(str_or_url),
                params=params,
                body=body,
                status=response.status,
                headers=response.headers,
            )
        except Exception as exc:
            recorder.fail(exc)
            return response
        return _RecordingAiohttpResponse(response, recorder, index)

    aiohttp.ClientSession._request = request


@contextmanager
def record_provider_http_tape() -> Iterator[HttpTapeRecorder]:
    _install_hooks()
    recorder = HttpTapeRecorder()
    replay_token = _ACTIVE_REPLAYER.set(None)
    record_token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(record_token)
        _ACTIVE_REPLAYER.reset(replay_token)


@contextmanager
def replay_provider_http_tape(tape: Mapping[str, Any]) -> Iterator[HttpTapeReplayer]:
    _install_hooks()
    replayer = HttpTapeReplayer(tape)
    record_token = _ACTIVE_RECORDER.set(None)
    replay_token = _ACTIVE_REPLAYER.set(replayer)
    try:
        yield replayer
        replayer.assert_consumed()
    finally:
        _ACTIVE_REPLAYER.reset(replay_token)
        _ACTIVE_RECORDER.reset(record_token)
