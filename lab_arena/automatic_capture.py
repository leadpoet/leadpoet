"""Best-effort copies of HTTP bytes already consumed by Python agent code.

This module uses only the standard library. It does not read a response for the
caller, inspect memory, decrypt TLS, or inspect application variables. Supported
Python HTTP adapters are installed when their modules are imported. Native,
subprocess and custom transports are outside this boundary. The summary reports
these limits; a successful installation does not prove complete agent coverage.
"""
from __future__ import annotations

import builtins
import collections
import contextvars
import functools
import hashlib
import http.client
import importlib
import queue
import re
import sys
import threading
import time
import weakref
from datetime import datetime, timezone
from urllib.parse import urlsplit

_ACTIVE = contextvars.ContextVar("arena_http_capture", default=None)
_INSTALLED = None
_BODY_LIMIT = 6 * 1024 * 1024
_STATE_LIMIT = 2048
_SOURCE = "runtime_http_v1"
_PROVIDER_HOSTS = frozenset({"openrouter.ai", "code.deepline.com", "api.scrapingdog.com"})


class _Body:
    def __init__(self, owner, client, url, status, headers):
        self.owner = owner
        self.client = client
        self.url = str(url)
        self.status = int(status)
        self.content_type = str(headers.get("content-type", ""))
        self.content_encoding = str(headers.get("content-encoding", "")).lower()
        identity = str(headers.get("x-leadpoet-call-identity", ""))
        self.identity = identity if (urlsplit(self.url).hostname in _PROVIDER_HOSTS
                                    and re.fullmatch(r"sha256:[0-9a-f]{64}", identity)) else ""
        self.encoding = "utf-8"
        match = re.search(r"charset\s*=\s*[\"']?([^\s;\"']+)", self.content_type, re.I)
        if match:
            self.encoding = match.group(1)[:80]
        self.retrieved_at = datetime.now(timezone.utc).isoformat()
        self.digest = hashlib.sha256()
        self.size = 0
        self.buffer = bytearray()
        self.truncated = False
        self.done = False
        self.complete = False
        self.provider_wire = False
        self.stream_key = None
        self.lock = threading.RLock()

    def add(self, value):
        if not isinstance(value, (bytes, bytearray, memoryview)):
            return
        with self.lock:
            if self.done:
                return
            self.digest.update(value)
            self.size += len(value)
            if self.identity or not value:
                return
            with self.owner.lock:
                available = min(_BODY_LIMIT - len(self.buffer), self.owner.memory_limit - self.owner.retained)
                amount = min(max(0, available), len(value))
                if amount:
                    self.buffer.extend(memoryview(value)[:amount])
                    self.owner.retained += amount
                if amount != len(value) and not self.truncated:
                    self.truncated = True
                    self.owner.counts["body_buffer_overflow"] += 1

    def finish(self, complete=False):
        with self.lock:
            if self.done:
                return
            self.done = True
            self.complete = bool(complete and not self.truncated)
            with self.owner.lock:
                self.owner.bodies.pop(id(self), None)
                association = self.owner.stream_bodies.get(self.stream_key)
                if association is not None and association[2] is self:
                    self.owner.stream_bodies.pop(self.stream_key, None)
                    reference, via_owner, _ = association
                    holder = reference() if via_owner else None
                    anchor = getattr(holder, "_arena_http_capture_stream", None)
                    if anchor is not None and anchor[0] is self:
                        del holder._arena_http_capture_stream
                self.owner.counts["responses_complete" if self.complete else "responses_partial"] += 1
            self.owner._enqueue(self)


class Recorder:
    """Bounded asynchronous observation writer. ``close`` never drains responses."""
    def __init__(self, checkpoint, *, memory_limit, queue_limit, close_timeout):
        self.checkpoint = checkpoint
        self.memory_limit = max(0, int(memory_limit))
        self.close_timeout = max(0.0, min(float(close_timeout), 0.25))
        self.lock = threading.RLock()
        self.counts = collections.Counter()
        self.bodies = {}
        # StreamReader may prohibit both attributes and weak references. Use
        # its weak-referenceable ClientResponse owner in that case. The recorder
        # does not retain that owner. The owner anchors the exact reader until
        # finish, including if caller code replaces response.content meanwhile.
        self.stream_bodies = {}
        self.retained = 0
        self.inflight = 0
        self.adapters = set()
        self.installed_modules = {}
        self.patches = []
        self.queue = queue.Queue(maxsize=max(1, int(queue_limit)))
        self.closed = False
        self.stopping = threading.Event()
        self.summary = None
        self.worker = threading.Thread(target=self._work, name="arena-http-capture", daemon=True)
        self.worker.start()

    def _count(self, key):
        with self.lock:
            self.counts[key] += 1

    def _safe(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            body = getattr(function, "__self__", None)
            if isinstance(body, _Body):
                body.truncated = True
            self._count("adapter_failure")
            return None

    def _new(self, client, url, status, headers):
        if self.closed or not url or not 100 <= int(status) <= 599:
            return None
        if len(str(url)) > 2000 or len(str(headers.get("content-type", ""))) > 256:
            self._count("metadata_unsupported")
            return None
        with self.lock:
            if len(self.bodies) >= _STATE_LIMIT:
                self.counts["response_limit"] += 1
                return None
            body = _Body(self, client, url, status, headers)
            self.bodies[id(body)] = body
            self.counts["responses_started"] += 1
            self.counts["client_" + client] += 1
            return body

    def _body(self, response, client, metadata):
        key = "_arena_http_capture_body"
        body = getattr(response, key, None)
        if body is None:
            with self.lock:
                association = self.stream_bodies.get(id(response))
                if association is not None:
                    reference, via_owner, candidate = association
                    target = reference()
                    if via_owner:
                        anchor = getattr(target, "_arena_http_capture_stream", None)
                        target = anchor[1] if anchor is not None and anchor[0] is candidate else None
                    if target is response:
                        body = candidate
        if body is not None and body.owner is self:
            return body
        values = metadata(response)
        if values is None:
            return None
        body = self._new(client, *values)
        if body is not None:
            setattr(response, key, body)
        return body

    def _register_stream(self, stream, client, values, *, owner=None):
        with self.lock:
            # A stream is registered only once, after response headers exist.
            # Lookups compare object identity, not just a recyclable integer id.
            existing = self._body(stream, client, lambda _: None)
            if existing is not None:
                return existing
            body = self._new(client, *values)
            if body is None:
                return None
            try:
                reference = weakref.ref(stream, lambda _: self._safe(body.finish, False))
                via_owner = False
            except TypeError:
                try:
                    reference = weakref.ref(owner, lambda _: self._safe(body.finish, False))
                    # The owner already owns this reader. Anchor its exact
                    # identity there, not in the recorder, until finish. This
                    # also prevents id reuse if caller code replaces content.
                    owner._arena_http_capture_stream = (body, stream)
                    via_owner = True
                except (TypeError, AttributeError):
                    self._count("stream_association_unsupported")
                    self._safe(body.finish, False)
                    return None
            body.stream_key = id(stream)
            self.stream_bodies[body.stream_key] = (reference, via_owner, body)
            return body

    def _release(self, body):
        with body.lock:
            with self.lock:
                self.retained -= len(body.buffer)
            body.buffer.clear()

    def _enqueue(self, body):
        try:
            self.queue.put_nowait(body)
        except queue.Full:
            self._count("queue_overflow")
            self._release(body)

    def _checkpoint_result(self, kind, result):
        if result is not None:
            self._count("provider_ack_written" if kind == "response_received" else "web_written")
            return
        self._count("checkpoint_failure")
        getter = getattr(self.checkpoint, "_training_failure_reason", None)
        try:
            reason = getter() if callable(getter) else None
        except Exception:
            reason = None
        if type(reason) is not str or reason not in {"timeout", "host_rejected", "transport_error", "invalid_response",
                          "invalid_frame", "socket_unavailable"}:
            reason = "unknown"
        self._count("checkpoint_failure_" + kind + "_" + reason)

    def _emit(self, body):
        if body.identity:
            if body.complete and 200 <= body.status <= 299:
                if body.content_encoding not in {"", "identity"} and not body.provider_wire:
                    self._count("provider_encoded_body_unsupported")
                    return
                result = self.checkpoint.record_response_received(
                    call_identity=body.identity, status=body.status,
                    body_sha256="sha256:" + body.digest.hexdigest(), body_bytes=body.size)
                self._checkpoint_result("response_received", result)
            else:
                self._count("provider_ack_ineligible")
            return
        if body.truncated:
            # The digest covers all bytes seen; the retained prefix does not.
            # Do not publish a prefix with a digest for a different byte string.
            return
        if not body.size and not body.complete:
            self._count("empty_or_unread_response")
            return
        media = body.content_type.split(";", 1)[0].strip().lower()
        textual = (not media or media.startswith("text/") or media in {
            "application/json", "application/xml", "application/javascript", "application/xhtml+xml",
            "application/x-ndjson", "application/graphql-response+json"} or media.endswith(("+json", "+xml")))
        if not textual:
            self._count("binary_body_unsupported")
            return
        try:
            # Strict round trip also rejects split multibyte partial prefixes.
            text = body.buffer.decode(body.encoding, errors="strict")
            if text.encode(body.encoding, errors="strict") != body.buffer:
                raise ValueError("non-reversible encoding")
            if len(text.encode("utf-8")) > _BODY_LIMIT:
                self._count("text_size_limit")
                return
        except (UnicodeError, LookupError, ValueError):
            self._count("text_encoding_unsupported")
            return
        result = self.checkpoint.record_web_fetch(
            url=body.url, final_url=body.url, status=body.status,
            content_type=body.content_type, body=text, retrieved_at=body.retrieved_at,
            capture={"source": _SOURCE, "complete": body.complete, "encoding": body.encoding,
                     "received_body_sha256": "sha256:" + body.digest.hexdigest(),
                     "received_body_bytes": body.size})
        self._checkpoint_result("web_fetch", result)

    def _work(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                body = self.queue.get(timeout=0.02)
            except queue.Empty:
                continue
            with self.lock:
                self.inflight += 1
            try:
                self._emit(body)
            except Exception:
                self._count("checkpoint_failure")
            finally:
                self._release(body)
                with self.lock:
                    self.inflight -= 1
                self.queue.task_done()

    def _summary(self):
        with self.lock:
            failures = {key: value for key, value in self.counts.items() if value and key in {
                "body_buffer_overflow", "response_limit", "queue_overflow", "checkpoint_failure",
                "binary_body_unsupported", "text_encoding_unsupported", "text_size_limit",
                "provider_encoded_body_unsupported", "responses_partial", "adapter_failure",
                "close_flush_timeout", "metadata_unsupported", "stream_association_unsupported"}}
            return {"source": _SOURCE, "complete": not failures and self.queue.empty() and self.inflight == 0,
                    "pending_queue": self.queue.qsize(), "inflight": self.inflight,
                    "counts": dict(self.counts), "adapters": sorted(self.adapters),
                    "coverage": "supported_python_http_consumption_only",
                    "unverified": ["native_clients", "subprocesses", "custom_transports", "application_variables",
                                   "custom_import_loaders", "unconsumed_responses", "peek_and_custom_response_apis",
                                   "vendored_client_copies"]}

    def close(self):
        """Restore owned patches and request a flush; return within 0.25 seconds."""
        global _INSTALLED
        if self.closed:
            return self.summary or self._summary()
        self.closed = True
        for target, name, previous, replacement in reversed(self.patches):
            if getattr(target, name, None) is replacement:
                setattr(target, name, previous)
        for body in list(self.bodies.values()):
            self._safe(body.finish, False)
        self.stopping.set()
        self.worker.join(self.close_timeout)
        if self.worker.is_alive():
            self._count("close_flush_timeout")
        if _INSTALLED is self:
            _INSTALLED = None
        self.summary = self._summary()
        return self.summary

    def _patch(self, target, name, factory):
        original = getattr(target, name, None)
        if not callable(original) or any(item[0] is target and item[1] == name for item in self.patches):
            return
        replacement = factory(original)
        setattr(target, name, replacement)
        self.patches.append((target, name, original, replacement))

    def _readinto_bytes(self, body, args, kwargs, size):
        try:
            buffer = args[0] if args else kwargs.get("b", kwargs.get("buffer"))
            body.add(memoryview(buffer)[:size])
        except Exception:
            body.truncated = True
            self._count("adapter_failure")

    def _read(self, client, metadata, *, readinto=False, whole=False, eof=None):
        def factory(original):
            @functools.wraps(original)
            def wrapped(response, *args, **kwargs):
                if (_ACTIVE.get() is not None or self.closed
                        or (client == "stdlib" and getattr(response, "_arena_capture_delegated", False))):
                    return original(response, *args, **kwargs)
                try:
                    body = self._body(response, client, metadata)
                except Exception:
                    self._count("adapter_failure")
                    body = None
                if body is None:
                    return original(response, *args, **kwargs)
                token = _ACTIVE.set(body)
                try:
                    value = original(response, *args, **kwargs)
                except BaseException as exc:
                    partial = self._safe(getattr, exc, "partial", None)
                    if isinstance(partial, bytes):
                        self._safe(body.add, partial)
                    self._safe(body.finish, isinstance(exc, StopIteration))
                    raise
                finally:
                    _ACTIVE.reset(token)
                if readinto:
                    if isinstance(value, int) and value > 0:
                        self._safe(self._readinto_bytes, body, args, kwargs, value)
                elif isinstance(value, list):
                    for part in value:
                        self._safe(body.add, part)
                else:
                    self._safe(body.add, value)
                amount = args[0] if args else kwargs.get("amt", kwargs.get("size", -1))
                done = (whole and (amount is None or amount == -1))
                done = done or ((value == b"" or (readinto and value == 0)) and amount != 0)
                if eof is not None:
                    done = done or bool(self._safe(eof, response))
                if done:
                    self._safe(body.finish, True)
                return value
            return wrapped
        return factory

    def _iterator(self, client, metadata, *, provider_raw=False):
        def factory(original):
            @functools.wraps(original)
            def wrapped(response, *args, **kwargs):
                # Do not hold context across yield: agent work may make another request.
                parent = _ACTIVE.get()
                if self.closed or (parent is not None and not provider_raw):
                    yield from original(response, *args, **kwargs)
                    return
                try:
                    body = self._body(response, client, metadata)
                except Exception:
                    self._count("adapter_failure")
                    body = None
                if body is None or (provider_raw and parent is not None and not body.identity):
                    yield from original(response, *args, **kwargs)
                    return
                if provider_raw and body.identity:
                    body.provider_wire = True
                iterator = original(response, *args, **kwargs)
                try:
                    while True:
                        token = _ACTIVE.set(body)
                        try:
                            value = next(iterator)
                        except StopIteration:
                            self._safe(body.finish, True)
                            return
                        finally:
                            _ACTIVE.reset(token)
                        if not body.identity or provider_raw or client != "httpx":
                            self._safe(body.add, value)
                        yield value
                except GeneratorExit:
                    self._safe(body.finish, False)
                    # Only a consumer's explicit generator.close forwards close.
                    # Exhaustion or a read error does not add a cleanup call to
                    # the original iterator, which could replace that error.
                    closer = getattr(iterator, "close", None)
                    if closer is not None:
                        closer()
                    raise
                except BaseException:
                    self._safe(body.finish, False)
                    raise
            return wrapped
        return factory

    def _close(self, client=None, metadata=None):
        def factory(original):
            @functools.wraps(original)
            def wrapped(response, *args, **kwargs):
                try:
                    return original(response, *args, **kwargs)
                finally:
                    body = getattr(response, "_arena_http_capture_body", None)
                    if (body is None and client is not None and _ACTIVE.get() is None
                            and not getattr(response, "_arena_capture_delegated", False)):
                        body = self._safe(self._body, response, client, metadata)
                    if body is not None and body.owner is self and _ACTIVE.get() is None:
                        self._safe(body.finish, False)
            return wrapped
        return factory

    def _install_stdlib(self):
        def getresponse_factory(original):
            @functools.wraps(original)
            def wrapped(connection, *args, **kwargs):
                response = original(connection, *args, **kwargs)
                try:
                    scheme = "https" if isinstance(connection, http.client.HTTPSConnection) else "http"
                    path = getattr(connection, "_arena_capture_path", "")
                    # CONNECT connects to a proxy socket, but the response
                    # belongs to the target named by set_tunnel, not the proxy.
                    host = getattr(connection, "_tunnel_host", None) or connection.host
                    tunnel_port = getattr(connection, "_tunnel_port", None)
                    port = tunnel_port if tunnel_port is not None else connection.port
                    authority = "[" + host + "]" if ":" in host and not host.startswith("[") else host
                    authority += ":" + str(port) if port != (443 if scheme == "https" else 80) else ""
                    url = path if path.startswith(("http://", "https://")) else scheme + "://" + authority + path
                    response._arena_capture_meta = (url, response.status, response.headers)
                except Exception:
                    self._count("adapter_failure")
                return response
            return wrapped
        def request_factory(original):
            @functools.wraps(original)
            def wrapped(connection, method, url, *args, **kwargs):
                connection._arena_capture_path = url
                return original(connection, method, url, *args, **kwargs)
            return wrapped
        self._patch(http.client.HTTPConnection, "putrequest", request_factory)
        self._patch(http.client.HTTPConnection, "getresponse", getresponse_factory)
        metadata = lambda response: getattr(response, "_arena_capture_meta", None)
        for method in ("read", "read1", "readinto", "readline", "readlines", "__next__"):
            self._patch(http.client.HTTPResponse, method, self._read(
                "stdlib", metadata, readinto=method == "readinto", whole=method in {"read", "readlines"},
                eof=lambda response: response.isclosed()))
        self._patch(http.client.HTTPResponse, "close", self._close("stdlib", metadata))
        self.adapters.add("stdlib_http_client_urllib")

    def _install_urllib3(self, module):
        cls = getattr(module, "HTTPResponse", None)
        if cls is None:
            return
        def metadata(response):
            url = response.geturl()
            if not url or not str(url).startswith(("http://", "https://")):
                original = getattr(response, "_original_response", None)
                values = getattr(original, "_arena_capture_meta", None)
                if values:
                    url = values[0]
            return (url, response.status, response.headers) if url else None
        def init_factory(original):
            @functools.wraps(original)
            def wrapped(response, *args, **kwargs):
                token = _ACTIVE.set("urllib3_init")
                try:
                    original(response, *args, **kwargs)
                finally:
                    _ACTIVE.reset(token)
                try:
                    underlying = getattr(response, "_original_response", None)
                    if underlying is not None:
                        underlying._arena_capture_delegated = True
                    body = self._body(response, "urllib3", metadata)
                    content = getattr(response, "_body", None)
                    if body is not None and isinstance(content, bytes):
                        self._safe(body.add, content)
                        self._safe(body.finish, True)
                except Exception:
                    self._count("adapter_failure")
            return wrapped
        self._patch(cls, "__init__", init_factory)
        for method in ("read", "read1", "readinto"):
            self._patch(cls, method, self._read("urllib3", metadata, whole=method == "read", readinto=method == "readinto",
                eof=lambda response: getattr(response, "length_remaining", None) == 0))
        for method in ("stream", "read_chunked", "__iter__"):
            self._patch(cls, method, self._iterator("urllib3", metadata))
        self._patch(cls, "close", self._close())
        self.adapters.add("urllib3_requests")
        return True

    def _install_httpx(self, module):
        cls = getattr(module, "Response", None)
        if cls is None:
            return
        def metadata(response):
            try:
                return str(response.url), response.status_code, response.headers
            except RuntimeError:
                return None
        # iter_bytes delegates to iter_raw; the inner raw boundary supplies exact
        # provider hashes while only the outer decoded boundary stores web text.
        def send_factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                response = original(*args, **kwargs)
                self._safe(self._body, response, "httpx", metadata)
                return response
            return wrapped
        def async_send_factory(original):
            @functools.wraps(original)
            async def wrapped(*args, **kwargs):
                response = await original(*args, **kwargs)
                self._safe(self._body, response, "httpx", metadata)
                return response
            return wrapped
        client = getattr(module, "Client", None)
        async_client = getattr(module, "AsyncClient", None)
        if client is None or async_client is None:
            return
        self._patch(client, "send", send_factory)
        self._patch(async_client, "send", async_send_factory)
        self._patch(cls, "iter_bytes", self._iterator("httpx", metadata))
        self._patch(cls, "iter_raw", self._iterator("httpx_raw", metadata, provider_raw=True))
        self._patch(cls, "aiter_bytes", self._async_iterator("httpx", metadata))
        self._patch(cls, "aiter_raw", self._async_iterator("httpx_raw", metadata, provider_raw=True))
        # Closing from inside iter_raw is an implementation detail; defer partial
        # detection to iterator completion, or to explicit user close.
        def close_factory(original):
            @functools.wraps(original)
            def wrapped(response, *args, **kwargs):
                try:
                    return original(response, *args, **kwargs)
                finally:
                    if _ACTIVE.get() is None:
                        body = getattr(response, "_arena_http_capture_body", None)
                        if body is not None and body.owner is self:
                            self._safe(body.finish, False)
            return wrapped
        self._patch(cls, "close", close_factory)
        self._patch(cls, "aclose", self._async_close())
        self.adapters.add("httpx_sync_async")
        return True

    def _async_iterator(self, client, metadata, *, provider_raw=False):
        def factory(original):
            @functools.wraps(original)
            async def wrapped(response, *args, **kwargs):
                parent = _ACTIVE.get()
                if self.closed or (parent is not None and not provider_raw):
                    async for value in original(response, *args, **kwargs):
                        yield value
                    return
                try:
                    body = self._body(response, client, metadata)
                except Exception:
                    self._count("adapter_failure")
                    body = None
                if body is None or (provider_raw and parent is not None and not body.identity):
                    async for value in original(response, *args, **kwargs):
                        yield value
                    return
                if provider_raw and body.identity:
                    body.provider_wire = True
                iterator = original(response, *args, **kwargs).__aiter__()
                try:
                    while True:
                        token = _ACTIVE.set(body)
                        try:
                            value = await iterator.__anext__()
                        except StopAsyncIteration:
                            self._safe(body.finish, True)
                            return
                        finally:
                            _ACTIVE.reset(token)
                        if not body.identity or provider_raw or client != "httpx":
                            self._safe(body.add, value)
                        yield value
                except GeneratorExit:
                    self._safe(body.finish, False)
                    closer = getattr(iterator, "aclose", None)
                    if closer is not None:
                        await closer()
                    raise
                except BaseException:
                    # Preserve the exact read error or cancellation. The
                    # original __anext__ owns its own cleanup on that path.
                    self._safe(body.finish, False)
                    raise
            return wrapped
        return factory

    def _async_close(self):
        def factory(original):
            @functools.wraps(original)
            async def wrapped(response, *args, **kwargs):
                try:
                    return await original(response, *args, **kwargs)
                finally:
                    if _ACTIVE.get() is None:
                        body = getattr(response, "_arena_http_capture_body", None)
                        if body is not None and body.owner is self:
                            self._safe(body.finish, False)
            return wrapped
        return factory

    def _install_aiohttp(self, module):
        cls = getattr(module, "ClientResponse", None)
        if cls is None:
            return
        # Attach metadata after response headers exist, without replacing streams.
        def start_factory(original):
            @functools.wraps(original)
            async def wrapped(response, *args, **kwargs):
                value = await original(response, *args, **kwargs)
                try:
                    self._register_stream(response.content, "aiohttp",
                                          (str(response.url), response.status, response.headers), owner=response)
                except Exception:
                    self._count("adapter_failure")
                return value
            return wrapped
        self._patch(cls, "start", start_factory)
        streams = sys.modules.get("aiohttp.streams")
        reader = getattr(streams, "StreamReader", None)
        if reader is None:
            return
        metadata = lambda response: None  # Registered after ClientResponse.start.
        def read_factory(original):
            @functools.wraps(original)
            async def wrapped(response, *args, **kwargs):
                if _ACTIVE.get() is not None or self.closed:
                    return await original(response, *args, **kwargs)
                try:
                    body = self._body(response, "aiohttp", metadata)
                except Exception:
                    self._count("adapter_failure")
                    body = None
                if body is None:
                    return await original(response, *args, **kwargs)
                token = _ACTIVE.set(body)
                try:
                    value = await original(response, *args, **kwargs)
                except BaseException as exc:
                    partial = self._safe(getattr, exc, "partial", None)
                    if isinstance(partial, bytes):
                        self._safe(body.add, partial)
                    self._safe(body.finish, False)
                    raise
                finally:
                    _ACTIVE.reset(token)
                self._safe(body.add, value[0] if isinstance(value, tuple) else value)
                amount = args[0] if args else kwargs.get("n", -1)
                if (original.__name__ == "read" and amount == -1) or self._safe(response.at_eof):
                    self._safe(body.finish, True)
                return value
            return wrapped
        for method in ("read", "readany", "readexactly", "readline", "readuntil", "readchunk"):
            self._patch(reader, method, read_factory)
        def nowait_factory(original):
            return self._read("aiohttp", metadata, eof=lambda response: response.at_eof())(original)
        self._patch(reader, "read_nowait", nowait_factory)
        self.adapters.add("aiohttp_stream_reader")
        return True

    def _patch_loaded(self):
        if self.closed:
            return
        for module_name, method in (("urllib3.response", self._install_urllib3),
                                    ("httpx", self._install_httpx), ("aiohttp", self._install_aiohttp)):
            module = sys.modules.get(module_name)
            if module is not None:
                identity = (id(module), id(getattr(module, "Response", getattr(module, "HTTPResponse", getattr(module, "ClientResponse", None)))))
                if self.installed_modules.get(module_name) == identity:
                    continue
                try:
                    with self.lock:
                        if method(module):
                            self.installed_modules[module_name] = identity
                except Exception:
                    self._count("adapter_failure")


def install(checkpoint_module, *, memory_limit=16 * 1024 * 1024, queue_limit=128,
            close_timeout=0.25):
    """Install once before miner imports. No third-party module is imported here."""
    global _INSTALLED
    if _INSTALLED is not None and not _INSTALLED.closed:
        return _INSTALLED
    recorder = Recorder(checkpoint_module, memory_limit=memory_limit,
                        queue_limit=queue_limit, close_timeout=close_timeout)
    _INSTALLED = recorder
    recorder._install_stdlib()
    # Public import entry points retain their exact result and exceptions. Do not
    # proactively import miners or their dependency trees just to install hooks.
    def import_factory(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            value = original(*args, **kwargs)
            recorder._patch_loaded()
            return value
        return wrapped
    recorder._patch(builtins, "__import__", import_factory)
    recorder._patch(importlib, "import_module", import_factory)
    recorder._patch_loaded()
    return recorder
