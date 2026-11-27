"""The lead-order key vault: the only place a miner's provider key is opened.

It runs on Phala Cloud in an Intel TDX VM. Its two private keys come from
dstack's KMS, which releases them only to builds approved on chain, so
Leadpoet cannot read a miner's key, even with full access to its own servers.

Endpoints:

* ``GET  /v1/identity``  public keys, their KMS signature chains, build ids,
                         and this boot's TLS key, signed by the receipts key.
* ``POST /v1/check``     open a newly registered key and validate it.
* ``POST /v1/call``      open a key, attach it, forward the request, and
                         return the response with a signed receipt.

The vault holds keys and proves their use; the gateway decides which models
and tools to pay for. So the vault's rules are about request shapes, not
names, and a new model or Deepline tool needs no vault change. Each tool's
host, key placement, routes and free check are one entry in ``tools.py``:

* OpenRouter: chat completions, Decisions, and the generation cost read-back.
* Deepline: ``POST /api/v2/integrations/<tool>/execute``, any tool.
* Tool market tools: their data endpoints only, never account, billing or
  key management, nor a queued task that bills later.
* The key only where its tool expects it, no redirects, bounded sizes, only
  the broker may call ``check`` and ``call``.

Every forwarded call gets a receipt with a per-key sequence number, so a call
the gateway hides shows up as a gap. No database, files or request logs.

The vault ends TLS itself, with a key made at boot and held only in memory.
Phala's gateway passes the TLS stream through unread, and clients trust the
certificate only because the identity's receipts key signed it.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import http.client
import json
import os
import re
import signal
import socket
import ssl
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote, unquote_to_bytes, urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from . import envelope as env
from .tools import TOOLS, Tool
from .envelope import (CALLER_HEADER, CHECK_TIMEOUT_SECONDS, ENCRYPTION_DOMAIN, HOSTS, IDENTITY_SCHEMA,
                       KEY_ALGORITHM, RECEIPT_SCHEMA, RECEIPTS_DOMAIN, TLS_SCHEMA, canonical, spki_sha256)

# Handshake, headers and body together; a per-read timeout alone would let a
# client that sends one byte at a time hold a connection slot forever.
REQUEST_READ_SECONDS = 10.0
REFUSED_BODY_DISCARD = 65536
MAX_REQUEST_HEAD = 32 * 1024  # request line and headers; http.server alone allows about 6.4 MB

_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}")
METHODS = ("GET", "POST")
DROPPED_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authorization", "proxy-authenticate", "proxy-connection",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length", "cookie", "set-cookie",
    # Bodies pass through unchanged and http.client does not decompress, so
    # ask the provider for plain bytes.
    "accept-encoding",
})
# The only request headers sent on. Anything else the gateway adds, such as a
# method or URL override a provider's front might honor, is dropped.
FORWARDED_HEADERS = frozenset({
    "authorization", "accept", "content-type", "user-agent",
    "x-deepline-execute-response-contract", "x-deepline-execute-response-intent", "x-deepline-tool-error-schema",
})
MAX_REQUEST_BODY = 2 * 1024 * 1024  # vault request JSON; carries a <=1 MiB provider body
MAX_PROVIDER_BODY = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 300.0
MIN_CALLER_TOKEN = 32

OPENROUTER_CURRENT_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/keys/{key_hash}"


class VaultError(Exception):
    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


class UpstreamError(Exception):
    """The provider could not be reached or did not answer in full."""

    def __init__(self, message: str, observed_status: Optional[int] = None,
                 observed_headers: Optional[List[Tuple[str, str]]] = None) -> None:
        super().__init__(message)
        self.observed_status = observed_status
        # Headers that arrived before the failure; they carry the provider's
        # request id, which the broker needs to read the cost back.
        self.observed_headers = observed_headers or []


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    headers: List[Tuple[str, str]]
    body: bytes


Upstream = Callable[[str, str, str, Mapping[str, str], bytes, float, int], UpstreamResponse]


# Provider request ids, kept on a withheld answer so its cost can be read back.
REQUEST_ID_HEADERS = frozenset({"x-generation-id", "x-deepline-request-id", "x-vercel-id"})


def _kept_headers(headers) -> List[List[str]]:
    return [[name.lower(), value] for name, value in headers if name.lower() not in DROPPED_HEADERS]


def echoes(secret: str, data: bytes) -> bool:
    """The key appears in ``data`` literally, URL-encoded, or JSON-escaped,
    in one JSON body or in any ``data:`` line of a streamed one."""

    key = secret.encode("utf-8")
    if key in data or key in unquote_to_bytes(data):
        return True
    if b"\\" not in data:  # without an escape, JSON decoding changes nothing
        return False
    for document in [data] + [line[5:] for line in data.splitlines() if line.startswith(b"data:")]:
        try:
            # Pairs, not a dict: a duplicate member is still seen.
            pending = [json.loads(document, object_pairs_hook=list)]
        except (UnicodeDecodeError, ValueError):
            continue
        except RecursionError:
            return True  # cannot be inspected, so it is withheld
        while pending:
            value = pending.pop()
            if isinstance(value, str):
                text = value.encode("utf-8", errors="surrogatepass")
                if key in text or (b"%" in text and key in unquote_to_bytes(text)):
                    return True
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
    return False


def header_echoes(secret: str, headers) -> bool:
    return any(echoes(secret, ("%s: %s" % (name, value)).encode("utf-8", errors="surrogatepass"))
               for name, value in headers)


def https_upstream(method: str, host: str, target: str, headers: Mapping[str, str], body: bytes,
                   timeout: float, max_bytes: int, *, port: int = 443,
                   context: Optional[ssl.SSLContext] = None) -> UpstreamResponse:
    """One HTTPS/1.1 request, certificate checked, no redirects, done within ``timeout`` in total.

    ``port`` and ``context`` exist for the loopback test; the vault always
    uses 443 and the system's trust store.
    """

    deadline = time.monotonic() + timeout
    connection = http.client.HTTPSConnection(host, port, timeout=timeout,
                                             context=context or ssl.create_default_context())
    status: Optional[int] = None
    seen: List[Tuple[str, str]] = []

    def time_left() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UpstreamError("upstream_timeout", status, seen)
        return remaining

    try:
        connection.request(method, target, body=body if method == "POST" else None, headers=dict(headers))
        # Our own handle: getresponse() drops the connection's socket when
        # the answer says it will close the connection.
        sock = connection.sock
        sock.settimeout(time_left())
        response = connection.getresponse()
        status, seen = response.status, list(response.getheaders())
        chunks: List[bytes] = []
        size = 0
        # The response closes the socket once the body is in.
        while not response.isclosed():
            # Each read gets only the time left, so a provider that trickles
            # its answer cannot outlast the caller's timeout.
            sock.settimeout(time_left())
            chunk = response.read1(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise UpstreamError("response_too_large", status, seen)
            chunks.append(chunk)
        return UpstreamResponse(status, seen, b"".join(chunks))
    except UpstreamError:
        raise
    except TimeoutError:
        raise UpstreamError("upstream_timeout", status, seen) from None
    except (OSError, http.client.HTTPException, ssl.SSLError):
        raise UpstreamError("upstream_unavailable", status, seen) from None
    finally:
        connection.close()


class Vault:
    """Request handling, independent of HTTP and of dstack so it can be tested."""

    def __init__(self, *, encryption_key: ec.EllipticCurvePrivateKey, receipts_key: ec.EllipticCurvePrivateKey,
                 identity: Mapping, caller_token: str, tls_spki_sha256: str, upstream: Upstream = https_upstream,
                 clock: Callable[[], float] = time.time) -> None:
        if not isinstance(caller_token, str) or len(caller_token) < MIN_CALLER_TOKEN:
            raise ValueError("caller token too short")
        self._encryption_key = encryption_key
        self._receipts_key = receipts_key
        self._caller_token = caller_token.encode("utf-8")
        self._upstream = upstream
        self._clock = clock
        self.boot_id = str(uuid.uuid4())
        self.receipts_public_key = env.compressed(receipts_key.public_key()).hex()
        self._identity = dict(identity, schema=IDENTITY_SCHEMA,
                              boot_id=self.boot_id, booted_at=int(clock()))
        # Binds this boot's TLS key to the KMS-issued receipts key.
        self._identity["tls"] = self._signed({"schema": TLS_SCHEMA, "boot_id": self.boot_id,
                                              "spki_sha256": tls_spki_sha256})
        self._lock = threading.Lock()
        self._calls: Dict[str, int] = {}

    # -- public, unauthenticated ------------------------------------------------

    def identity(self) -> Mapping:
        return self._identity

    # -- Leadpoet's broker only ---------------------------------------------------

    def authorize(self, token: Optional[str]) -> None:
        if not token or not hmac.compare_digest(token.encode("utf-8"), self._caller_token):
            raise VaultError("caller_unauthorized", 401)

    def check(self, request: Mapping) -> Mapping:
        """Validate a newly sealed key with its provider, as registration did."""

        if not isinstance(request, Mapping) or not set(request) <= {"provider", "envelope", "management_envelope"}:
            raise VaultError("request_invalid")
        provider = request.get("provider")
        runtime, header = self._open(request.get("envelope"), provider, "runtime")
        key_fingerprint = env.fingerprint(provider, runtime)
        result = {"provider": provider, "miner_hotkey": header.miner_hotkey, "key_fingerprint": key_fingerprint}
        try:
            if provider == "openrouter":
                management, management_header = self._open(request.get("management_envelope"), provider, "management")
                if management_header.miner_hotkey != header.miner_hotkey:
                    raise VaultError("envelope_mismatch")
                self._check_openrouter(runtime, management)
            else:
                if request.get("management_envelope") is not None:
                    raise VaultError("request_invalid")
                tool = TOOLS[provider]
                if tool.balance_url is None:
                    # No free check: opened and named only. Its calls prove it.
                    return dict(result, status="unchecked")
                balance = _field(self._probe(tool, tool.balance_url, runtime, provider + "_api_key_invalid"),
                                 tool.balance_field)
                if balance is not None:
                    if balance < 0:  # ZeroBounce's answer for a key it does not know
                        raise _Invalid(provider + "_api_key_invalid")
                    if balance <= 0:
                        raise _Invalid(provider + "_api_key_no_credit")
                    result["balance"] = balance
        except _Invalid as invalid:
            return dict(result, status="invalid", reason=invalid.code)
        return dict(result, status="valid")

    def call(self, request: Mapping) -> Mapping:
        """Forward one provider request with the miner's key in place of its envelope."""

        method, parts, headers, body, timeout, max_bytes = _parse_call(request)
        provider = HOSTS[parts.hostname]
        sealed = headers.pop("authorization", "")
        if not sealed.startswith("Bearer " + env.PREFIX):
            raise VaultError("credential_missing")
        try:
            secret, header = self._open(sealed[len("Bearer "):], provider, "runtime")
        except VaultError as error:
            if error.code in ("request_invalid", "provider_mismatch"):
                raise
            # The key cannot be used. Answer like the provider would, so the
            # broker retires this key and moves on; nothing was sent or billed.
            return _refusal(401, "vault_key_unusable", None)
        key_fingerprint = env.fingerprint(provider, secret)
        target = parts.path + ("?" + parts.query if parts.query else "")
        # The receipt hashes the target before the key is placed in it.
        outbound, keyed_target = TOOLS[provider].place(secret, headers, target)
        # Every spelling of the key the provider could send back.
        forms = tuple({secret, quote(secret, safe="")})
        key_sequence, at = self._count(key_fingerprint)
        # Per key only: a global count would show each miner everyone's volume.
        receipt = {"schema": RECEIPT_SCHEMA, "boot_id": self.boot_id,
                   "key_sequence": key_sequence, "at": at,
                   "key_fingerprint": key_fingerprint, "miner_hotkey": header.miner_hotkey, "provider": provider,
                   "method": method, "host": parts.hostname, "path": parts.path,
                   "target_sha256": hashlib.sha256(target.encode("ascii")).hexdigest(),
                   # The miner's key is not in this hash: the forwarded
                   # headers as sent, with authorization left out.
                   "headers_sha256": hashlib.sha256(canonical(headers)).hexdigest(),
                   "request_sha256": hashlib.sha256(body).hexdigest()}
        try:
            response = self._upstream(method, parts.hostname, keyed_target, outbound, body, timeout, max_bytes)
        except Exception as caught:  # the call is counted, so every failure must get a receipt
            error = caught if isinstance(caught, UpstreamError) else UpstreamError("upstream_unavailable")
            receipt.update(outcome="transport_error", response_status=error.observed_status, response_sha256=None)
            observed = _kept_headers(error.observed_headers)
            return {"transport_error": str(error), "observed_status": error.observed_status,
                    "observed_headers": [] if any(header_echoes(form, observed) for form in forms) else observed,
                    "receipt": self._signed(receipt)}
        response_headers = _kept_headers(response.headers)
        if any(echoes(form, response.body) or header_echoes(form, response_headers) for form in forms):
            receipt.update(outcome="key_echo_withheld", response_status=response.status, response_sha256=None)
            ids = [[name, value] for name, value in response_headers
                   if name in REQUEST_ID_HEADERS
                   and not any(echoes(form, value.encode("utf-8", errors="surrogatepass")) for form in forms)]
            return _refusal(502, "vault_response_withheld", self._signed(receipt), ids)
        receipt.update(outcome="response", response_status=response.status,
                       response_sha256=hashlib.sha256(response.body).hexdigest())
        return {"status": response.status, "headers": response_headers,
                "body_b64": base64.b64encode(response.body).decode("ascii"), "receipt": self._signed(receipt)}

    # -- internals ----------------------------------------------------------------

    def _open(self, sealed, provider, purpose) -> Tuple[str, env.Header]:
        if provider not in env.PROVIDERS or not isinstance(sealed, str):
            raise VaultError("request_invalid")
        try:
            secret, header = env.open_envelope(sealed, self._encryption_key)
        except env.EnvelopeError as error:
            raise VaultError(str(error)) from None
        if header.provider != provider:
            raise VaultError("provider_mismatch")
        if header.purpose != purpose:
            raise VaultError("purpose_mismatch")
        return secret, header

    def _count(self, key_fingerprint: str) -> Tuple[int, int]:
        """Number a call with this key's next sequence number and the time."""

        with self._lock:
            self._calls[key_fingerprint] = self._calls.get(key_fingerprint, 0) + 1
            return self._calls[key_fingerprint], int(self._clock())

    def _signed(self, document: Mapping) -> Mapping:
        signature = self._receipts_key.sign(canonical(document), ec.ECDSA(hashes.SHA256()))
        return {"document": document, "signature": signature.hex(), "public_key": self.receipts_public_key}

    def _probe(self, tool: Tool, url: str, secret: str, invalid_code: str) -> Mapping:
        parts = urlsplit(url)
        headers, target = tool.place(secret, {"accept": "application/json", "user-agent": "leadpoet-lead-order-vault/1"},
                                     parts.path + ("?" + parts.query if parts.query else ""))
        try:
            response = self._upstream("GET", parts.hostname, target, headers, b"", CHECK_TIMEOUT_SECONDS, 65536)
        except UpstreamError:
            raise VaultError("check_unavailable", 503) from None
        if response.status in tool.invalid_statuses:
            raise _Invalid(invalid_code)
        if response.status == 402:
            raise _Invalid(invalid_code.replace("_invalid", "_no_credit"))
        if response.status != 200:
            raise VaultError("check_unavailable", 503)
        try:
            document = json.loads(response.body)
        except ValueError:
            raise VaultError("check_unavailable", 503) from None
        return document if isinstance(document, Mapping) else {}

    def _check_openrouter(self, runtime: str, management: str) -> None:
        openrouter = TOOLS["openrouter"]
        current = self._probe(openrouter, OPENROUTER_CURRENT_KEY_URL, runtime, "openrouter_api_key_invalid").get("data")
        if not isinstance(current, Mapping) or current.get("disabled") is True or current.get("is_management_key") is not False:
            raise _Invalid("openrouter_api_key_invalid")
        remaining = current.get("limit_remaining")
        if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining <= 0:
            raise _Invalid("openrouter_api_key_no_credit")
        runtime_hash = hashlib.sha256(runtime.encode("utf-8")).hexdigest()
        managed = self._probe(openrouter, OPENROUTER_KEY_URL.format(key_hash=runtime_hash), management,
                              "openrouter_management_key_invalid").get("data")
        if not isinstance(managed, Mapping) or managed.get("hash") != runtime_hash or managed.get("disabled") is True:
            raise _Invalid("openrouter_management_key_invalid")


class _Invalid(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_NUMBER = re.compile(r"-?[0-9]{1,15}(?:\.[0-9]{1,6})?")


def _field(document, path: Tuple):
    """The number at ``path`` in a probe's answer, or None when it is not there."""

    value = document
    for step in path:
        value = value.get(step) if isinstance(value, Mapping) else None
    if isinstance(value, str) and _NUMBER.fullmatch(value):  # ZeroBounce sends its credits as text
        value = float(value)
    if path and type(value) in (int, float) and value == value:
        return value
    return None


def _route_allowed(provider: str, method: str, path: str, query: str) -> bool:
    return TOOLS[provider].route(method, path, query)


def _refusal(status: int, code: str, receipt: Optional[Mapping], headers=()) -> Mapping:
    """An answer the vault gives in the provider's place."""

    return {"status": status, "headers": [["content-type", "application/json"], *headers],
            "body_b64": base64.b64encode(canonical({"error": {"code": code}})).decode("ascii"),
            "receipt": receipt}


def _parse_call(request: Mapping):
    allowed = {"method", "url", "headers", "body_b64", "timeout_seconds", "max_response_bytes"}
    if not isinstance(request, Mapping) or not set(request) <= allowed or not {"method", "url", "headers"} <= set(request):
        raise VaultError("request_invalid")
    method = request["method"]
    if method not in METHODS:
        raise VaultError("method_not_allowed")
    url = request["url"]
    if not isinstance(url, str) or len(url) > 4096:
        raise VaultError("url_invalid")
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        raise VaultError("url_invalid") from None
    if (parts.scheme != "https" or parts.hostname not in HOSTS or port not in (None, 443)
            or parts.username is not None or parts.password is not None or parts.fragment
            or parts.netloc.lower() not in (parts.hostname, parts.hostname + ":443")):
        raise VaultError("host_not_allowed")
    provider = HOSTS[parts.hostname]
    if not _route_allowed(provider, method, parts.path, parts.query) or TOOLS[provider].caller_sets_key(parts.query):
        raise VaultError("path_not_allowed")
    raw_headers = request["headers"]
    if not isinstance(raw_headers, Mapping) or len(raw_headers) > 64:
        raise VaultError("headers_invalid")
    headers: Dict[str, str] = {}
    for name, value in raw_headers.items():
        # Checked here, before the call is counted, so http.client never
        # refuses a header after the count went up.
        if (not isinstance(name, str) or not isinstance(value, str) or not _HEADER_NAME.fullmatch(name)
                or len(value) > 8192 or any(c in value for c in "\r\n\x00") or not value.isascii()):
            raise VaultError("headers_invalid")
        lowered = name.lower()
        if lowered in headers:
            raise VaultError("headers_invalid")
        if lowered in FORWARDED_HEADERS:
            headers[lowered] = value
    body_b64 = request.get("body_b64")
    if body_b64 is None:
        body = b""
    elif isinstance(body_b64, str):
        try:
            body = base64.b64decode(body_b64, validate=True)
        except ValueError:
            raise VaultError("body_invalid") from None
    else:
        raise VaultError("body_invalid")
    if len(body) > MAX_PROVIDER_BODY or (method == "GET" and body):
        raise VaultError("body_invalid")
    timeout = request.get("timeout_seconds", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        raise VaultError("timeout_invalid")
    max_bytes = request.get("max_response_bytes")
    if max_bytes is None:
        max_bytes = MAX_RESPONSE_BYTES
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= MAX_RESPONSE_BYTES:
        raise VaultError("max_response_bytes_invalid")
    return method, parts, headers, body, float(timeout), max_bytes


# -- HTTP -------------------------------------------------------------------------


def make_handler(vault: Vault):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30  # a slow client cannot hold a thread for long
        server_version = "lead-order-vault/1"
        sys_version = ""

        def log_message(self, *args) -> None:  # no request logs, ever
            return

        def setup(self) -> None:
            super().setup()
            self.rfile = _HeadLimit(self.rfile)

        def _reply(self, status: int, document: Mapping) -> None:
            data = canonical(document)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("cache-control", "no-store")
            # One request per connection, so an idle client holds no slot.
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, action: Callable[[], Mapping]) -> None:
            try:
                self._reply(200, action())
            except VaultError as error:
                self._reply(error.status, {"error": error.code})
            except Exception:  # never echo internals; the request is not logged
                self._reply(500, {"error": "vault_error"})

        def do_GET(self) -> None:
            self.server.request_read()
            parts = urlsplit(self.path)
            routes = {"/v1/identity": vault.identity, "/healthz": lambda: {"ok": True}}
            route = routes.get(parts.path)
            self._handle(route if route else _not_found)

        def do_POST(self) -> None:
            routes = {"/v1/call": vault.call, "/v1/check": vault.check}
            route = routes.get(urlsplit(self.path).path)

            def action() -> Mapping:
                length = self.headers.get("content-length", "")
                if not length.isdigit() or int(length) > MAX_REQUEST_BODY:
                    raise VaultError("request_too_large", 413)
                try:
                    if route is None:
                        _not_found()
                    vault.authorize(self.headers.get(CALLER_HEADER))
                except VaultError:
                    # Closing on unread data resets the connection, and the
                    # caller would lose the answer. A stranger's body is not
                    # buffered beyond this.
                    self.rfile.read(min(int(length), REFUSED_BODY_DISCARD))
                    raise
                body = self.rfile.read(int(length))
                self.server.request_read()
                try:
                    request = json.loads(body)
                except ValueError:
                    raise VaultError("request_invalid") from None
                return route(request)

            self._handle(action)

    return Handler


class _HeadLimit:
    """``rfile`` with at most ``MAX_REQUEST_HEAD`` bytes of lines; the body is read as sent."""

    def __init__(self, raw) -> None:
        self._raw = raw
        self._left = MAX_REQUEST_HEAD

    def readline(self, limit: int = -1) -> bytes:
        line = self._raw.readline(self._left + 1 if limit < 0 else min(limit, self._left + 1))
        self._left -= len(line)
        if self._left < 0:
            raise http.client.LineTooLong("request head")
        return line

    def __getattr__(self, name: str):
        return getattr(self._raw, name)


class BoundedServer(ThreadingHTTPServer):
    """The vault's HTTPS server: at most ``MAX_CONNECTIONS`` at once, the rest closed.

    Each connection's TLS handshake runs in its own worker thread, so a slow
    client cannot stall the accept loop. A connection that has not delivered
    its whole request, handshake included, within ``REQUEST_READ_SECONDS``
    is cut; the handler calls ``request_read`` once it has.
    """

    # Not daemons: server_close() waits for calls in flight, so a restart
    # does not cut a billed call or lose its receipt.
    daemon_threads = False
    request_queue_size = 128  # the listen backlog; Python's default of 5 drops bursts
    MAX_CONNECTIONS = 128

    def __init__(self, address, handler, ssl_context: ssl.SSLContext) -> None:
        super().__init__(address, handler)
        self._ssl_context = ssl_context
        self._slots = threading.BoundedSemaphore(self.MAX_CONNECTIONS)
        self._reading = threading.local()

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def finish_request(self, request, client_address) -> None:
        try:
            # No I/O yet: the handshake runs below, under the read deadline.
            tls = self._ssl_context.wrap_socket(request, server_side=True, do_handshake_on_connect=False)
        except OSError:  # the client already left
            return
        timer = threading.Timer(REQUEST_READ_SECONDS, _shutdown, args=(tls,))
        timer.daemon = True
        self._reading.timer = timer
        timer.start()
        try:
            with tls:
                try:
                    tls.do_handshake()
                except OSError:  # includes ssl.SSLError: a failed handshake is just closed
                    return
                super().finish_request(tls, client_address)
        finally:
            timer.cancel()

    def request_read(self) -> None:
        """The whole request has arrived; the read deadline no longer applies."""

        self._reading.timer.cancel()

    def handle_error(self, request, client_address) -> None:
        # socketserver would print a traceback with the client's address;
        # the vault never logs requests. A reset client is not an error.
        pass


def _shutdown(connection: socket.socket) -> None:
    # The base method, so the TLS layer is not torn down under its reader.
    try:
        socket.socket.shutdown(connection, socket.SHUT_RDWR)
    except OSError:
        pass


def tls_server_context() -> Tuple[ssl.SSLContext, str]:
    """A new P-256 TLS key for this boot, never written to disk, and its SPKI hash."""

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "lead-order-vault")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                   .serial_number(x509.random_serial_number())
                   .not_valid_before(now - datetime.timedelta(hours=1))
                   .not_valid_after(now + datetime.timedelta(days=3650))
                   .sign(key, hashes.SHA256()))
    pem = certificate.public_bytes(serialization.Encoding.PEM) + key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    # ssl loads keys only from a path. In the vault that path is an in-memory
    # file; elsewhere (local tests) a temporary file removed right after.
    if sys.platform.startswith("linux"):
        descriptor = os.memfd_create("lead-order-vault-tls")
        try:
            os.write(descriptor, pem)
            context.load_cert_chain("/proc/self/fd/%d" % descriptor)
        finally:
            os.close(descriptor)
    else:
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(pem)
            handle.flush()
            context.load_cert_chain(handle.name)
    return context, spki_sha256(certificate.public_bytes(serialization.Encoding.DER))


def _not_found() -> Mapping:
    raise VaultError("not_found", 404)


# -- dstack ------------------------------------------------------------------------


class Dstack:
    """The dstack guest agent on its Unix socket (API v1)."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path

    def post(self, method: str, body: Mapping) -> Mapping:
        path = self._socket_path

        class Connection(http.client.HTTPConnection):
            def connect(self) -> None:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(30)
                self.sock.connect(path)

        connection = Connection("localhost")
        try:
            data = canonical(body)
            connection.request("POST", "/v1/" + method, body=data,
                               headers={"content-type": "application/json", "content-length": str(len(data))})
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise RuntimeError("dstack %s failed with status %d" % (method, response.status))
            return json.loads(payload)
        finally:
            connection.close()

    def key(self, domain: str) -> Tuple[ec.EllipticCurvePrivateKey, Mapping]:
        reply = self.post("GetKey", {"domain": domain, "algorithm": KEY_ALGORITHM})
        private_key = ec.derive_private_key(int(reply["key"], 16), ec.SECP256K1())
        public_key = env.compressed(private_key.public_key()).hex()
        if reply.get("public_key", public_key).lower() != public_key:
            raise RuntimeError("dstack returned a public key that does not match its private key")
        return private_key, {"domain": domain, "algorithm": KEY_ALGORITHM, "public_key": public_key,
                             "signature_chain": list(reply["signature_chain"])}


def main() -> None:
    caller_token = os.environ.get("LEAD_ORDER_VAULT_CALLER_TOKEN", "")
    dstack = Dstack(os.environ.get("DSTACK_SOCKET", "/var/run/dstack.sock"))
    encryption_key, encryption = dstack.key(ENCRYPTION_DOMAIN)
    receipts_key, receipts = dstack.key(RECEIPTS_DOMAIN)
    info = dstack.post("Info", {})
    identity = {name: info.get(name) for name in ("app_id", "compose_hash", "instance_id", "os_image_hash")}
    identity.update(encryption=encryption, receipts=receipts)
    tls_context, tls_spki = tls_server_context()
    vault = Vault(encryption_key=encryption_key, receipts_key=receipts_key, identity=identity,
                  caller_token=caller_token, tls_spki_sha256=tls_spki)
    server = BoundedServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(vault), tls_context)
    # On stop, refuse new connections and let calls in flight finish; each
    # ends within MAX_TIMEOUT_SECONDS plus REQUEST_READ_SECONDS.
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()
