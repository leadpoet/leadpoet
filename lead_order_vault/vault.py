"""The lead-order key vault: the only place a miner's provider key is opened.

It runs on Phala Cloud in an Intel TDX VM. Its two private keys come from
dstack's KMS, which releases them only to builds approved on chain, so
Leadpoet cannot read a miner's key, even with full access to its own servers.

Endpoints:

* ``GET  /v1/identity``     public keys, their KMS signature chains, build ids.
* ``GET  /v1/attestation``  a TDX quote binding both public keys to a nonce.
* ``GET  /v1/usage``        a signed count of calls made with one key this boot.
* ``POST /v1/check``        open a newly registered key and validate it.
* ``POST /v1/call``         open a key, attach it, forward the request, and
                            return the response with a signed receipt.

Fixed rules: two provider hosts, the key only in the ``authorization``
header, account and key-management paths refused, no redirects, bounded
sizes, only Leadpoet's broker may call ``check`` and ``call``. No database,
no files, no logs of requests or bodies. Counters live in memory and reset on
restart; receipts carry the boot id so a restart is visible.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import socket
import ssl
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from . import envelope as env

ENCRYPTION_DOMAIN = "leadpoet/lead-order-vault/encryption/v1"
RECEIPTS_DOMAIN = "leadpoet/lead-order-vault/receipts/v1"
KEY_ALGORITHM = "secp256k1"
RECEIPT_SCHEMA = "leadpoet.lead_order_vault.receipt.v1"
USAGE_SCHEMA = "leadpoet.lead_order_vault.usage.v1"
ATTESTATION_TAG = b"leadpoet.lead_order_vault.attestation.v1"

HOSTS = {"openrouter.ai": "openrouter", "code.deepline.com": "deepline"}
# Account, key, credit and team management. A runtime key may not reach these
# through the vault, whatever the provider would allow.
DENIED_SEGMENTS = frozenset({
    "keys", "api-keys", "api_keys", "apikeys", "auth", "oauth", "admin", "credits",
    "organization", "organizations", "members", "invites", "team", "teams",
})
METHODS = ("GET", "POST")
DROPPED_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authorization", "proxy-authenticate", "proxy-connection",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length", "cookie", "set-cookie",
    # Bodies pass through unchanged and http.client does not decompress, so
    # ask the provider for plain bytes.
    "accept-encoding",
})
MAX_REQUEST_BODY = 2 * 1024 * 1024  # vault request JSON; carries a <=1 MiB provider body
MAX_PROVIDER_BODY = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 300.0
CHECK_TIMEOUT_SECONDS = 15.0
CALLER_HEADER = "x-lead-order-vault-token"
MIN_CALLER_TOKEN = 32

OPENROUTER_CURRENT_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/keys/{key_hash}"
DEEPLINE_BALANCE_URL = "https://code.deepline.com/api/v2/billing/balance"


class VaultError(Exception):
    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


class UpstreamError(Exception):
    """The provider could not be reached or did not answer in full."""

    def __init__(self, message: str, observed_status: Optional[int] = None) -> None:
        super().__init__(message)
        self.observed_status = observed_status


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    headers: List[Tuple[str, str]]
    body: bytes


Upstream = Callable[[str, str, str, Mapping[str, str], bytes, float, int], UpstreamResponse]


def canonical(document: Mapping) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _compressed_hex(key: ec.EllipticCurvePrivateKey) -> str:
    return key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint).hex()


def https_upstream(method: str, host: str, target: str, headers: Mapping[str, str], body: bytes,
                   timeout: float, max_bytes: int) -> UpstreamResponse:
    """One HTTPS/1.1 request to port 443, certificate checked, no redirects."""

    connection = http.client.HTTPSConnection(host, 443, timeout=timeout, context=ssl.create_default_context())
    status: Optional[int] = None
    try:
        connection.request(method, target, body=body if method == "POST" else None, headers=dict(headers))
        response = connection.getresponse()
        status = response.status
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise UpstreamError("response_too_large", status)
        return UpstreamResponse(status, list(response.getheaders()), data)
    except UpstreamError:
        raise
    except (OSError, http.client.HTTPException, ssl.SSLError):
        raise UpstreamError("upstream_unavailable", status) from None
    finally:
        connection.close()


class Vault:
    """Request handling, independent of HTTP and of dstack so it can be tested."""

    def __init__(self, *, encryption_key: ec.EllipticCurvePrivateKey, receipts_key: ec.EllipticCurvePrivateKey,
                 identity: Mapping, caller_token: str, upstream: Upstream = https_upstream,
                 clock: Callable[[], float] = time.time, attest: Optional[Callable[[bytes], Mapping]] = None) -> None:
        if not isinstance(caller_token, str) or len(caller_token) < MIN_CALLER_TOKEN:
            raise ValueError("caller token too short")
        self._encryption_key = encryption_key
        self._receipts_key = receipts_key
        self._caller_token = caller_token.encode("utf-8")
        self._upstream = upstream
        self._clock = clock
        self._attest = attest
        self.boot_id = str(uuid.uuid4())
        self.booted_at = int(clock())
        self.encryption_public_key = _compressed_hex(encryption_key)
        self.receipts_public_key = _compressed_hex(receipts_key)
        self._identity = dict(identity, schema="leadpoet.lead_order_vault.identity.v1",
                              boot_id=self.boot_id, booted_at=self.booted_at)
        self._lock = threading.Lock()
        self._sequence = 0
        self._calls: Dict[str, int] = {}
        self._checks: Dict[str, int] = {}

    # -- public, unauthenticated ------------------------------------------------

    def identity(self) -> Mapping:
        return self._identity

    def attestation(self, query: Mapping[str, List[str]]) -> Mapping:
        nonce = _single(query, "nonce")
        if len(nonce) != 64 or not _is_hex(nonce):
            raise VaultError("nonce_invalid")
        if self._attest is None:
            raise VaultError("attestation_unavailable", 503)
        report_data = hashlib.sha256(
            ATTESTATION_TAG + bytes.fromhex(self.encryption_public_key)
            + bytes.fromhex(self.receipts_public_key) + bytes.fromhex(nonce)).digest()
        return {"report_data": report_data.hex(), "nonce": nonce,
                "encryption_public_key": self.encryption_public_key,
                "receipts_public_key": self.receipts_public_key, "quote": self._attest(report_data)}

    def usage(self, query: Mapping[str, List[str]]) -> Mapping:
        key_fingerprint = _single(query, "fingerprint")
        if not (key_fingerprint.startswith("sha256:") and len(key_fingerprint) == 71 and _is_hex(key_fingerprint[7:])):
            raise VaultError("fingerprint_invalid")
        with self._lock:
            document = {"schema": USAGE_SCHEMA, "boot_id": self.boot_id, "booted_at": self.booted_at,
                        "key_fingerprint": key_fingerprint, "calls": self._calls.get(key_fingerprint, 0),
                        "checks": self._checks.get(key_fingerprint, 0), "last_sequence": self._sequence,
                        "at": int(self._clock())}
        return self._signed(document)

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
                self._count(self._checks, key_fingerprint)
                self._check_openrouter(runtime, management)
            else:
                if request.get("management_envelope") is not None:
                    raise VaultError("request_invalid")
                self._count(self._checks, key_fingerprint)
                self._probe(DEEPLINE_BALANCE_URL, runtime, "deepline_api_key_invalid")
        except _Invalid as invalid:
            return dict(result, status="invalid", reason=invalid.code)
        return dict(result, status="valid")

    def call(self, request: Mapping) -> Mapping:
        """Forward one provider request with the miner's key in place of its envelope."""

        method, url, headers, body, timeout, max_bytes = _parse_call(request)
        parts = urlsplit(url)
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
            return {"status": 401, "headers": [["content-type", "application/json"]],
                    "body_b64": base64.b64encode(canonical({"error": {"code": "vault_key_unusable"}})).decode(),
                    "receipt": None}
        key_fingerprint = env.fingerprint(provider, secret)
        target = parts.path + ("?" + parts.query if parts.query else "")
        outbound = dict(headers, authorization="Bearer " + secret)
        sequence, key_sequence, at = self._count(self._calls, key_fingerprint)
        receipt = {"schema": RECEIPT_SCHEMA, "boot_id": self.boot_id, "sequence": sequence,
                   "key_sequence": key_sequence, "at": at,
                   "key_fingerprint": key_fingerprint, "miner_hotkey": header.miner_hotkey, "provider": provider,
                   "method": method, "host": parts.hostname, "path": parts.path,
                   "request_sha256": hashlib.sha256(body).hexdigest()}
        try:
            response = self._upstream(method, parts.hostname, target, outbound, body, timeout, max_bytes)
        except UpstreamError as error:
            receipt.update(outcome="transport_error", response_status=error.observed_status, response_sha256=None)
            return {"transport_error": str(error), "observed_status": error.observed_status,
                    "receipt": self._signed(receipt)}
        response_headers = [[name.lower(), value] for name, value in response.headers
                            if name.lower() not in DROPPED_HEADERS]
        secret_bytes = secret.encode("utf-8")
        if secret_bytes in response.body or any(secret in value for _, value in response_headers):
            receipt.update(outcome="key_echo_withheld", response_status=response.status, response_sha256=None)
            return {"status": 502, "headers": [["content-type", "application/json"]],
                    "body_b64": base64.b64encode(canonical({"error": {"code": "vault_response_withheld"}})).decode(),
                    "receipt": self._signed(receipt)}
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

    def _count(self, table: Dict[str, int], key_fingerprint: str) -> Tuple[int, int, int]:
        with self._lock:
            self._sequence += 1
            table[key_fingerprint] = table.get(key_fingerprint, 0) + 1
            return self._sequence, table[key_fingerprint], int(self._clock())

    def _signed(self, document: Mapping) -> Mapping:
        signature = self._receipts_key.sign(canonical(document), ec.ECDSA(hashes.SHA256()))
        return {"document": document, "signature": signature.hex(), "public_key": self.receipts_public_key}

    def _probe(self, url: str, secret: str, invalid_code: str) -> Mapping:
        parts = urlsplit(url)
        headers = {"authorization": "Bearer " + secret, "accept": "application/json",
                   "user-agent": "leadpoet-lead-order-vault/1"}
        try:
            response = self._upstream("GET", parts.hostname, parts.path, headers, b"", CHECK_TIMEOUT_SECONDS, 65536)
        except UpstreamError:
            raise VaultError("check_unavailable", 503) from None
        if response.status in (401, 403, 404):
            raise _Invalid(invalid_code)
        if response.status != 200:
            raise VaultError("check_unavailable", 503)
        try:
            document = json.loads(response.body)
        except ValueError:
            raise VaultError("check_unavailable", 503) from None
        return document if isinstance(document, Mapping) else {}

    def _check_openrouter(self, runtime: str, management: str) -> None:
        current = self._probe(OPENROUTER_CURRENT_KEY_URL, runtime, "openrouter_api_key_invalid").get("data")
        if not isinstance(current, Mapping) or current.get("disabled") is True or current.get("is_management_key") is not False:
            raise _Invalid("openrouter_api_key_invalid")
        remaining = current.get("limit_remaining")
        if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining <= 0:
            raise _Invalid("openrouter_api_key_no_credit")
        runtime_hash = hashlib.sha256(runtime.encode("utf-8")).hexdigest()
        managed = self._probe(OPENROUTER_KEY_URL.format(key_hash=runtime_hash), management,
                              "openrouter_management_key_invalid").get("data")
        if not isinstance(managed, Mapping) or managed.get("hash") != runtime_hash or managed.get("disabled") is True:
            raise _Invalid("openrouter_management_key_invalid")


class _Invalid(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _single(query: Mapping[str, List[str]], name: str) -> str:
    values = query.get(name) or []
    if len(values) != 1:
        raise VaultError(name + "_invalid")
    return values[0]


def _is_hex(value: str) -> bool:
    return bool(value) and all(c in "0123456789abcdef" for c in value)


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
    segments = [segment.lower() for segment in parts.path.split("/")]
    if any(segment in (".", "..") or "%" in segment or segment in DENIED_SEGMENTS for segment in segments):
        raise VaultError("path_not_allowed")
    raw_headers = request["headers"]
    if not isinstance(raw_headers, Mapping) or len(raw_headers) > 64:
        raise VaultError("headers_invalid")
    headers: Dict[str, str] = {}
    for name, value in raw_headers.items():
        if not isinstance(name, str) or not isinstance(value, str) or any(c in name + value for c in "\r\n\x00"):
            raise VaultError("headers_invalid")
        lowered = name.lower()
        if lowered in headers:
            raise VaultError("headers_invalid")
        if lowered not in DROPPED_HEADERS:
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
    return method, url, headers, body, float(timeout), max_bytes


# -- HTTP -------------------------------------------------------------------------


def make_handler(vault: Vault):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "lead-order-vault/1"
        sys_version = ""

        def log_message(self, *args) -> None:  # no request logs, ever
            return

        def _reply(self, status: int, document: Mapping) -> None:
            data = canonical(document)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("cache-control", "no-store")
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
            parts = urlsplit(self.path)
            query = parse_qs(parts.query, keep_blank_values=True)
            routes = {"/v1/identity": vault.identity, "/v1/attestation": lambda: vault.attestation(query),
                      "/v1/usage": lambda: vault.usage(query), "/healthz": lambda: {"ok": True}}
            route = routes.get(parts.path)
            self._handle(route if route else _not_found)

        def do_POST(self) -> None:
            routes = {"/v1/call": vault.call, "/v1/check": vault.check}
            route = routes.get(self.path)
            if route is None:
                self._handle(_not_found)
                return

            def action() -> Mapping:
                vault.authorize(self.headers.get(CALLER_HEADER))
                length = self.headers.get("content-length", "")
                if not length.isdigit() or int(length) > MAX_REQUEST_BODY:
                    raise VaultError("request_too_large", 413)
                try:
                    request = json.loads(self.rfile.read(int(length)))
                except ValueError:
                    raise VaultError("request_invalid") from None
                return route(request)

            self._handle(action)

    return Handler


def _not_found() -> Mapping:
    raise VaultError("not_found", 404)


# -- dstack ------------------------------------------------------------------------


class Dstack:
    """The dstack guest agent on its Unix socket (API v1)."""

    def __init__(self, socket_path: str = "/var/run/dstack.sock") -> None:
        self._socket_path = socket_path

    def _post(self, method: str, body: Mapping) -> Mapping:
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
        reply = self._post("GetKey", {"domain": domain, "algorithm": KEY_ALGORITHM})
        private_key = ec.derive_private_key(int(reply["key"], 16), ec.SECP256K1())
        public_key = _compressed_hex(private_key)
        if reply.get("public_key", public_key).lower() != public_key:
            raise RuntimeError("dstack returned a public key that does not match its private key")
        return private_key, {"domain": domain, "algorithm": KEY_ALGORITHM, "public_key": public_key,
                             "signature_chain": list(reply["signature_chain"])}

    def info(self) -> Mapping:
        return self._post("Info", {})

    def attest(self, report_data: bytes) -> Mapping:
        return self._post("Attest", {"report_data": report_data.hex()})


def main() -> None:
    caller_token = os.environ.get("LEAD_ORDER_VAULT_CALLER_TOKEN", "")
    dstack = Dstack(os.environ.get("DSTACK_SOCKET", "/var/run/dstack.sock"))
    encryption_key, encryption = dstack.key(ENCRYPTION_DOMAIN)
    receipts_key, receipts = dstack.key(RECEIPTS_DOMAIN)
    info = dstack.info()
    identity = {name: info.get(name) for name in ("app_id", "compose_hash", "instance_id", "os_image_hash")}
    identity.update(encryption=encryption, receipts=receipts)
    vault = Vault(encryption_key=encryption_key, receipts_key=receipts_key, identity=identity,
                  caller_token=caller_token, attest=dstack.attest)
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(vault))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
