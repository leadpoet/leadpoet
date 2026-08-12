"""Coordinator-owned HTTPS provider broker with terminal V2 transport records."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import secrets
import socket
import ssl
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from httpx import SyncByteStream

from gateway.tee.egress_framing import (
    TUNNEL_FRAMING_HEADER,
    TUNNEL_FRAMING_MODE,
)
from gateway.tee.egress_policy import normalize_destination, normalize_proxy_destination
from gateway.tee.egress_proxy import DEFAULT_IDLE_TIMEOUT_SECONDS
from gateway.tee.inter_enclave_tls import MAX_FRAME_BYTES, REPLAY_WAIT_SECONDS
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
)
from gateway.tee.source_add_runtime_v2 import (
    source_add_dynamic_job_slot,
    source_add_dynamic_retry_policy_hash,
    validate_source_add_runtime_route_v2,
)


PROVIDER_BROKER_SCHEMA_VERSION = "leadpoet.provider_broker.v2"
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
PROVIDER_RPC_RESPONSE_RESERVE_BYTES = 8 * 1024 * 1024


def _provider_rpc_response_body_limit(
    *,
    frame_bytes: int,
    reserve_bytes: int,
) -> int:
    if frame_bytes <= reserve_bytes or reserve_bytes < 0:
        raise ValueError("provider RPC response frame budget is invalid")
    return ((frame_bytes - reserve_bytes) // 4) * 3


# Provider bodies cross the authenticated RPC as base64. Reserve bounded room
# for the terminal, artifact, cost, and checkpoint evidence added downstream.
MAX_RESPONSE_BODY_BYTES = _provider_rpc_response_body_limit(
    frame_bytes=MAX_FRAME_BYTES,
    reserve_bytes=PROVIDER_RPC_RESPONSE_RESERVE_BYTES,
)
MAX_TRANSPORT_RESPONSE_BODY_BYTES = 96 * 1024 * 1024
MAX_DEDUPLICATION_RECORDS = 10000
TERMINAL_RECORD_RETENTION_SECONDS = 3600.0
MAX_JOB_CREDENTIAL_LEASES = 1024
DIRECT_PROVIDER_KEEPALIVE_EXPIRY_SECONDS = min(
    120.0,
    DEFAULT_IDLE_TIMEOUT_SECONDS / 2.0,
)
EGRESS_PROXY_CREDENTIAL_SLOT = "egress_proxy"
MEASURED_TRANSPORT_REQUEST_HEADERS = (("Accept-Encoding", "identity"),)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_QUERY_NAMES = frozenset(
    {"api_key", "apikey", "key", "token", "access_token"}
)
_SECRET_HEADER_NAMES = frozenset(
    {
        "apikey",
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
    }
)


class ProviderBrokerV2Error(RuntimeError):
    """A request violates the measured provider route or terminal ledger."""


_EGRESS_PROXY_SLOT_REF_HASH = sha256_json(
    {
        "schema_version": "leadpoet.job_credential_slot.v2",
        "credential_slot": EGRESS_PROXY_CREDENTIAL_SLOT,
        "scope": "job",
    }
)


def _job_credential_slot_ref_hash(slot: str) -> str:
    return sha256_json(
        {
            "schema_version": "leadpoet.job_credential_slot.v2",
            "credential_slot": str(slot),
            "scope": "job",
        }
    )


@dataclass(frozen=True)
class ProviderRouteV2:
    provider_id: str
    hosts: Tuple[str, ...]
    path_prefixes: Tuple[str, ...] = ("/",)
    credential_slot: str = ""
    credential_location: str = "none"
    credential_name: str = ""
    credential_prefix: str = ""
    credential_header_aliases: Tuple[Tuple[str, str], ...] = ()
    request_headers: Tuple[Tuple[str, str], ...] = ()
    allowed_methods: Tuple[str, ...] = ()
    allowed_route_pairs: Tuple[Tuple[str, str], ...] = ()
    job_scoped_only: bool = False


BUILTIN_PROVIDER_ROUTES = {
    "openrouter": ProviderRouteV2(
        provider_id="openrouter",
        hosts=("openrouter.ai",),
        path_prefixes=("/api/",),
        credential_slot="openrouter",
        credential_location="header",
        credential_name="Authorization",
        credential_prefix="Bearer ",
    ),
    "openrouter_management": ProviderRouteV2(
        provider_id="openrouter_management",
        hosts=("openrouter.ai",),
        path_prefixes=(
            "/api/v1/generation",
            "/api/v1/workspaces",
            "/api/v1/keys",
        ),
        credential_slot="openrouter_management",
        credential_location="header",
        credential_name="Authorization",
        credential_prefix="Bearer ",
        allowed_methods=("GET", "PATCH"),
        job_scoped_only=True,
    ),
    "exa": ProviderRouteV2(
        provider_id="exa",
        hosts=("api.exa.ai",),
        credential_slot="exa",
        credential_location="header",
        credential_name="x-api-key",
    ),
    "scrapingdog": ProviderRouteV2(
        provider_id="scrapingdog",
        hosts=("api.scrapingdog.com",),
        credential_slot="scrapingdog",
        credential_location="query",
        credential_name="api_key",
    ),
    "deepline": ProviderRouteV2(
        provider_id="deepline",
        hosts=("code.deepline.com",),
        credential_slot="deepline",
        credential_location="header",
        credential_name="Authorization",
        credential_prefix="Bearer ",
    ),
    "supabase": ProviderRouteV2(
        provider_id="supabase",
        hosts=("qplwoislplkcegvdmbim.supabase.co",),
        path_prefixes=("/rest/v1/",),
        credential_slot="supabase_service_role",
        credential_location="header",
        credential_name="Authorization",
        credential_prefix="Bearer ",
        credential_header_aliases=(("apikey", ""),),
    ),
    "truelist": ProviderRouteV2(
        provider_id="truelist",
        hosts=("api.truelist.io",),
        path_prefixes=("/api/v1/",),
        credential_slot="truelist",
        credential_location="header",
        credential_name="Authorization",
        credential_prefix="Bearer ",
    ),
    "dns": ProviderRouteV2(
        provider_id="dns",
        hosts=("cloudflare-dns.com",),
        path_prefixes=("/dns-query",),
    ),
    "rdap": ProviderRouteV2(
        provider_id="rdap",
        hosts=("rdap.org",),
        path_prefixes=("/domain/",),
    ),
    "bittensor_chain": ProviderRouteV2(
        provider_id="bittensor_chain",
        hosts=("entrypoint-finney.opentensor.ai",),
        path_prefixes=("/",),
    ),
    "bittensor_archive": ProviderRouteV2(
        provider_id="bittensor_archive",
        hosts=("archive.chain.opentensor.ai",),
        path_prefixes=("/",),
        allowed_methods=("POST",),
    ),
    "arweave": ProviderRouteV2(
        provider_id="arweave",
        hosts=("arweave.net",),
        path_prefixes=("/",),
        allowed_methods=("GET",),
    ),
    "coingecko": ProviderRouteV2(
        provider_id="coingecko",
        hosts=("api.coingecko.com",),
        path_prefixes=("/api/v3/simple/price",),
        allowed_methods=("GET",),
    ),
    "wayback": ProviderRouteV2(
        provider_id="wayback",
        hosts=("archive.org", "web.archive.org", "arquivo.pt"),
        path_prefixes=("/wayback/available", "/wayback/cdx"),
    ),
    "public_web": ProviderRouteV2(
        provider_id="public_web",
        hosts=(),
    ),
}


def provider_registry_document() -> Dict[str, Any]:
    def route_document(route: ProviderRouteV2) -> Dict[str, Any]:
        document = {
            "hosts": list(route.hosts),
            "path_prefixes": list(route.path_prefixes),
            "credential_slot": route.credential_slot,
            "credential_location": route.credential_location,
            "credential_name": route.credential_name,
            "credential_prefix": route.credential_prefix,
            "credential_header_aliases": [
                {"name": name, "prefix": prefix}
                for name, prefix in route.credential_header_aliases
            ],
        }
        if route.allowed_methods:
            document["allowed_methods"] = list(route.allowed_methods)
        if route.request_headers:
            document["request_headers"] = [
                {"name": name, "value": value}
                for name, value in route.request_headers
            ]
        if route.allowed_route_pairs:
            document["allowed_route_pairs"] = [
                {"method": method, "path": path}
                for method, path in route.allowed_route_pairs
            ]
        if route.job_scoped_only:
            document["job_scoped_only"] = True
        return document

    return {
        "schema_version": "leadpoet.provider_registry.v2",
        "transport": {
            "scheme": "https",
            "port": 443,
            "tls_termination": "gateway_coordinator_enclave",
            "plaintext_external_http": False,
            "request_headers": [
                {"name": name, "value": value}
                for name, value in MEASURED_TRANSPORT_REQUEST_HEADERS
            ],
        },
        "routes": {
            provider_id: route_document(route)
            for provider_id, route in sorted(BUILTIN_PROVIDER_ROUTES.items())
        },
    }


def provider_registry_hash() -> str:
    return sha256_json(provider_registry_document())


def expected_provider_credential_slots() -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                route.credential_slot
                for route in BUILTIN_PROVIDER_ROUTES.values()
                if route.credential_slot and not route.job_scoped_only
            }
        )
    )


def expected_job_credential_slot_ref_hashes() -> Dict[str, str]:
    return {
        EGRESS_PROXY_CREDENTIAL_SLOT: _EGRESS_PROXY_SLOT_REF_HASH,
        "openrouter_management": _job_credential_slot_ref_hash(
            "openrouter_management"
        ),
    }


def measured_retry_policy_hashes(
    protected_workflow_manifest_hash: str,
) -> Dict[str, str]:
    normalized = str(protected_workflow_manifest_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ProviderBrokerV2Error("protected workflow manifest hash is invalid")
    return {
        provider_id: sha256_json(
            {
                "schema_version": "leadpoet.provider_retry_policy.v2",
                "provider_id": provider_id,
                "protected_workflow_manifest_hash": normalized,
                "authority": "measured_caller_retry_logic",
            }
        )
        for provider_id in sorted(BUILTIN_PROVIDER_ROUTES)
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _credential_hash(value: str) -> str:
    return sha256_bytes(("leadpoet-provider-credential-v2:" + value).encode("utf-8"))


def credential_value_hash(value: str) -> str:
    """Match the existing Research Lab key-vault SHA-256 commitment."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProviderBrokerV2Error("provider credential value is invalid")
    return sha256_bytes(value.encode("utf-8"))


def _validated_tls_proxy_url(value: str) -> str:
    normalized = str(value or "")
    parsed = urlsplit(normalized)
    scheme = parsed.scheme.lower()
    try:
        parsed_port = parsed.port
        port = (
            parsed_port
            if parsed_port is not None
            else (443 if scheme == "https" else 80)
        )
    except ValueError as exc:
        raise ProviderBrokerV2Error("worker egress proxy port is invalid") from exc
    if (
        scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderBrokerV2Error(
            "worker egress proxy must be an HTTP CONNECT or HTTPS proxy URL"
        )
    try:
        normalize_proxy_destination(parsed.hostname, port)
    except ValueError as exc:
        raise ProviderBrokerV2Error(
            "worker egress proxy destination is invalid"
        ) from exc
    if (parsed.username is None) != (parsed.password is None):
        raise ProviderBrokerV2Error("worker egress proxy credentials are incomplete")
    if any(
        "\x00" in str(item) or "\r" in str(item) or "\n" in str(item)
        for item in (parsed.username, parsed.password)
        if item is not None
    ):
        raise ProviderBrokerV2Error("worker egress proxy credentials are invalid")
    return normalized


def _nonsecret_headers(headers: Mapping[str, Any]) -> Dict[str, str]:
    output = {}
    for name, value in headers.items():
        normalized_name = str(name).strip().lower()
        if normalized_name in _SECRET_HEADER_NAMES:
            continue
        output[normalized_name] = str(value)
    return dict(sorted(output.items()))


def _decoded_response_headers(headers: Mapping[str, Any]) -> Dict[str, str]:
    output = _nonsecret_headers(headers)
    for name in ("content-encoding", "content-length", "transfer-encoding"):
        output.pop(name, None)
    return output


def _sanitized_path(parsed: Any) -> str:
    query = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name.lower() not in _SECRET_QUERY_NAMES
    ]
    return urlunsplit(("", "", parsed.path or "/", urlencode(query), ""))


def _failure_code(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "response exceeds size limit" in text:
        return "response_too_large"
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in name or "timed out" in text:
        return "timeout"
    if isinstance(exc, ssl.SSLError) or "tls" in name or "certificate" in text:
        return "tls_failure"
    if isinstance(exc, socket.gaierror) or "dns" in name:
        return "dns_failure"
    if isinstance(exc, ConnectionResetError) or "reset" in text:
        return "connection_reset"
    if isinstance(exc, ConnectionRefusedError) or "refused" in text:
        return "connection_refused"
    if "proxy" in name or "proxy" in text:
        return "proxy_failure"
    if "malformed" in text or "protocol" in name:
        return "malformed_reply"
    return "unexpected_eof"


def _extract_tls_metadata(response: Any) -> Tuple[str, str]:
    try:
        stream = response.extensions["network_stream"]
        ssl_object = stream.get_extra_info("ssl_object")
        certificate = ssl_object.getpeercert(True)
        protocol = ssl_object.version()
    except Exception as exc:
        raise ProviderBrokerV2Error("authenticated response lacks TLS evidence") from exc
    if not certificate or not protocol:
        raise ProviderBrokerV2Error("authenticated response TLS evidence is empty")
    return sha256_bytes(bytes(certificate)), str(protocol)


def _make_response_cleanup_nonfatal(response: Any) -> None:
    """Keep stream cleanup from replacing a completed read or its real error."""
    stream = getattr(response, "stream", None)
    if not isinstance(stream, SyncByteStream):
        return

    class CleanupSafeStream(SyncByteStream):
        def __iter__(self):
            yield from stream

        def close(self) -> None:
            # HTTPX closes once when iteration finishes and again when the
            # response context exits. A relay EOF during cleanup is not body
            # evidence; iterator failures above still propagate fail-closed.
            try:
                stream.close()
            except Exception:
                pass

    response.stream = CleanupSafeStream()


def _close_client_nonfatal(client: Any) -> None:
    """Do not let connection cleanup replace authenticated request evidence."""
    try:
        client.close()
    except Exception:
        pass


def _authenticated_body_is_complete_after_stream_error(
    *,
    method: str,
    response: Any,
    byte_count: int,
    body: Optional[bytes] = None,
    error: BaseException,
) -> bool:
    """Accept only an authenticated response whose framing is already complete."""
    # HTTPX/httpcore may wrap a relay EOF in implementation-specific exception
    # types. The exception taxonomy is not an integrity boundary: TLS evidence
    # and response headers have already been authenticated, and callers still
    # verify the exact canonical body, ciphertext hash, and Object Lock state.
    # Recover solely from objective HTTP framing so library wrappers cannot turn
    # an already-complete immutable readback into a permanent availability
    # failure. Incomplete or ambiguous bodies remain fail-closed.
    del error
    normalized_method = str(method or "").upper()
    if normalized_method == "HEAD":
        return byte_count == 0
    response_headers = response.headers
    if str(response_headers.get("content-encoding") or "").strip().lower() not in {
        "",
        "identity",
    }:
        return False
    transfer_encoding = str(
        response_headers.get("transfer-encoding") or ""
    ).strip().lower()
    try:
        raw_values = response_headers.get_list("content-length")
    except AttributeError:
        raw_value = response_headers.get("content-length")
        raw_values = [] if raw_value is None else [raw_value]
    tokens = [
        token.strip()
        for raw_value in raw_values
        for token in str(raw_value).split(",")
    ]
    if not transfer_encoding:
        if tokens:
            if any(not token.isdigit() for token in tokens):
                return False
            declared_lengths = {int(token) for token in tokens}
            return (
                len(declared_lengths) == 1
                and byte_count == next(iter(declared_lengths))
            )
        if normalized_method != "GET":
            return False
        # HTTP/2 carries message completion in DATA-frame END_STREAM rather
        # than HTTP/1 Content-Length or Transfer-Encoding headers. The parent
        # relay can lose that terminal signal after delivering every
        # authenticated TLS byte. A complete JSON document is an objective
        # application boundary: truncating an array or object before its final
        # delimiter cannot still parse, and trailing non-whitespace is
        # rejected by json.loads. Restrict this recovery to authenticated 2xx
        # HTTP/2 JSON reads; unframed HTTP/1 responses remain ambiguous.
        try:
            http_version = str(response.http_version or "").strip().upper()
        except Exception:
            http_version = ""
        if http_version != "HTTP/2":
            return False
        return _authenticated_json_body_is_complete(
            response=response,
            byte_count=byte_count,
            body=body,
        )

    # PostgREST legitimately streams bounded JSON with chunked framing. A
    # relay may preserve the complete authenticated JSON body but lose the
    # terminal zero-length chunk. Recover only when the body itself provides
    # an objective completeness boundary; every other chunked response remains
    # fail-closed.
    if normalized_method != "GET" or transfer_encoding != "chunked" or tokens:
        return False
    return _authenticated_json_body_is_complete(
        response=response,
        byte_count=byte_count,
        body=body,
    )


def _authenticated_json_body_is_complete(
    *,
    response: Any,
    byte_count: int,
    body: Optional[bytes],
) -> bool:
    if not 200 <= int(response.status_code) < 300:
        return False
    content_type = str(response.headers.get("content-type") or "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        return False
    if body is None or len(body) != byte_count:
        return False
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(document, (dict, list))


class HTTPXProviderTransport:
    """TLS and hostname verification run inside the coordinator enclave."""

    def __init__(
        self,
        *,
        proxy_url: str = "http://127.0.0.1:18080",
        ca_bundle: Optional[str] = None,
        response_body_ceiling_bytes: int = MAX_RESPONSE_BODY_BYTES,
        allow_authenticated_complete_body_eof: bool = False,
        parent_tunnel_framing: str = "",
        reuse_direct_connections: bool = False,
    ) -> None:
        if (
            isinstance(response_body_ceiling_bytes, bool)
            or not isinstance(response_body_ceiling_bytes, int)
            or not 1
            <= response_body_ceiling_bytes
            <= MAX_TRANSPORT_RESPONSE_BODY_BYTES
        ):
            raise ProviderBrokerV2Error(
                "provider transport response ceiling is invalid"
            )
        if not isinstance(allow_authenticated_complete_body_eof, bool):
            raise ProviderBrokerV2Error(
                "provider transport complete-body EOF policy is invalid"
            )
        if parent_tunnel_framing not in {"", TUNNEL_FRAMING_MODE}:
            raise ProviderBrokerV2Error(
                "provider transport tunnel framing is invalid"
            )
        if not isinstance(reuse_direct_connections, bool):
            raise ProviderBrokerV2Error(
                "provider transport connection reuse policy is invalid"
            )
        self.proxy_url = proxy_url
        self.ca_bundle = ca_bundle
        self.response_body_ceiling_bytes = response_body_ceiling_bytes
        self.allow_authenticated_complete_body_eof = (
            allow_authenticated_complete_body_eof
        )
        self.parent_tunnel_framing = parent_tunnel_framing
        self.reuse_direct_connections = reuse_direct_connections
        self._direct_client = None
        self._direct_client_lock = threading.Lock()
        self._direct_client_generation = 0
        self._direct_client_leases: Dict[int, int] = {}
        self._retired_direct_clients: Dict[int, Any] = {}

    def _new_client(self, *, proxy_headers: Optional[Mapping[str, str]] = None) -> Any:
        import certifi
        import httpx
        from http.cookiejar import CookieJar, DefaultCookiePolicy

        class RejectAllCookies(DefaultCookiePolicy):
            def set_ok(self, cookie: Any, request: Any) -> bool:
                return False

        verify_path = self.ca_bundle or certifi.where()
        normalized_proxy_headers = dict(proxy_headers or {})
        # The upstream-proxy control path is already an enclave-authenticated
        # TLS tunnel and explicitly rejects parent framing. Applying both
        # headers makes the local proxy reject CONNECT before either the
        # provider or Supabase is contacted.
        if self.parent_tunnel_framing and not normalized_proxy_headers:
            normalized_proxy_headers[TUNNEL_FRAMING_HEADER] = (
                self.parent_tunnel_framing
            )
        return httpx.Client(
            proxy=httpx.Proxy(
                self.proxy_url,
                headers=normalized_proxy_headers or None,
            ),
            verify=verify_path,
            trust_env=False,
            http2=True,
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=32,
                keepalive_expiry=DIRECT_PROVIDER_KEEPALIVE_EXPIRY_SECONDS,
            ),
            cookies=CookieJar(policy=RejectAllCookies()),
            follow_redirects=False,
        )

    @contextmanager
    def _lease_direct_client(self):
        with self._direct_client_lock:
            if self._direct_client is None:
                self._direct_client = self._new_client()
                self._direct_client_generation += 1
            client = self._direct_client
            generation = self._direct_client_generation
            self._direct_client_leases[generation] = (
                self._direct_client_leases.get(generation, 0) + 1
            )
        try:
            yield client, generation
        finally:
            close_client = None
            with self._direct_client_lock:
                remaining = self._direct_client_leases.get(generation, 0) - 1
                if remaining > 0:
                    self._direct_client_leases[generation] = remaining
                else:
                    self._direct_client_leases.pop(generation, None)
                    close_client = self._retired_direct_clients.pop(
                        generation,
                        None,
                    )
            if close_client is not None:
                _close_client_nonfatal(close_client)

    def _retire_direct_client(self, client: Any, generation: int) -> None:
        close_client = None
        with self._direct_client_lock:
            if (
                self._direct_client is client
                and self._direct_client_generation == generation
            ):
                self._direct_client = None
                if self._direct_client_leases.get(generation, 0) > 0:
                    self._retired_direct_clients[generation] = client
                else:
                    close_client = client
        if close_client is not None:
            _close_client_nonfatal(close_client)

    def close(self) -> None:
        with self._direct_client_lock:
            clients = list(self._retired_direct_clients.values())
            if self._direct_client is not None:
                clients.append(self._direct_client)
            self._direct_client = None
            self._direct_client_leases.clear()
            self._retired_direct_clients.clear()
        for client in dict.fromkeys(clients):
            _close_client_nonfatal(client)

    @staticmethod
    def _execute_with_client(
        client: Any,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
        allow_authenticated_complete_body_eof: bool = False,
    ) -> Dict[str, Any]:
        with client.stream(
            method,
            url,
            headers=dict(headers),
            content=body,
            timeout=timeout_seconds,
        ) as response:
            _make_response_cleanup_nonfatal(response)
            # TLS is authenticated before response headers are available.
            # Capture its evidence now because a peer may close the stream
            # immediately after sending a complete bounded response body.
            tls_peer_chain_hash, tls_protocol = _extract_tls_metadata(response)
            chunks = []
            byte_count = 0
            try:
                for chunk in response.iter_bytes():
                    byte_count += len(chunk)
                    if byte_count > max_response_bytes:
                        raise ProviderBrokerV2Error(
                            "provider response exceeds size limit"
                        )
                    chunks.append(chunk)
            except Exception as exc:
                response_body = b"".join(chunks)
                if not (
                    allow_authenticated_complete_body_eof
                    and byte_count <= max_response_bytes
                    and _authenticated_body_is_complete_after_stream_error(
                        method=method,
                        response=response,
                        byte_count=byte_count,
                        body=response_body,
                        error=exc,
                    )
                ):
                    raise
            else:
                response_body = b"".join(chunks)
            return {
                "http_status": int(response.status_code),
                "headers": _decoded_response_headers(response.headers),
                "body": response_body,
                "tls_peer_chain_hash": tls_peer_chain_hash,
                "tls_protocol": tls_protocol,
            }

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_ms: int,
        upstream_proxy_url: Optional[str] = None,
        max_response_bytes: int = MAX_RESPONSE_BODY_BYTES,
    ) -> Dict[str, Any]:
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1
            <= max_response_bytes
            <= self.response_body_ceiling_bytes
        ):
            raise ProviderBrokerV2Error("provider response limit is invalid")

        proxy_headers = None
        if upstream_proxy_url:
            proxy_headers = {
                "X-Leadpoet-Upstream-Proxy-B64": base64.b64encode(
                    upstream_proxy_url.encode("utf-8")
                ).decode("ascii")
            }
        timeout_seconds = max(0.001, timeout_ms / 1000.0)
        if proxy_headers is None and self.reuse_direct_connections:
            with self._lease_direct_client() as (client, generation):
                try:
                    return self._execute_with_client(
                        client,
                        method=method,
                        url=url,
                        headers=headers,
                        body=body,
                        timeout_seconds=timeout_seconds,
                        max_response_bytes=max_response_bytes,
                        allow_authenticated_complete_body_eof=(
                            self.allow_authenticated_complete_body_eof
                        ),
                    )
                except BaseException:
                    # The measured caller owns retry accounting. Retire only
                    # this transport generation so its next measured attempt
                    # cannot inherit a dead relay tunnel. Existing leases may
                    # finish before the failed generation is closed.
                    self._retire_direct_client(client, generation)
                    raise

        # Direct coordinator requests are request-scoped by default. A relay
        # tunnel can expire between otherwise independent jobs or epochs; one
        # stale pooled generation must not make their measured retries share a
        # failure fate. Framed artifact-only callers may explicitly opt into
        # reuse after proving their terminal handshake contract.
        client = self._new_client(proxy_headers=proxy_headers)
        try:
            return self._execute_with_client(
                client,
                method=method,
                url=url,
                headers=headers,
                body=body,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
                allow_authenticated_complete_body_eof=(
                    self.allow_authenticated_complete_body_eof
                ),
            )
        finally:
            _close_client_nonfatal(client)


class ProviderBrokerV2:
    """Execute each logical provider attempt at most once and record a terminal."""

    def __init__(
        self,
        *,
        credential_ref_hashes: Mapping[str, str],
        retry_policy_hashes: Mapping[str, str],
        routes: Mapping[str, ProviderRouteV2] = BUILTIN_PROVIDER_ROUTES,
        transport: Optional[Callable[..., Mapping[str, Any]]] = None,
        artifact_sink: Optional[Callable[..., Mapping[str, Any]]] = None,
        job_credential_slot_ref_hashes: Optional[Mapping[str, str]] = None,
        clock: Callable[[], str] = _timestamp,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.routes = dict(routes)
        self.credential_ref_hashes = {
            str(name): str(value).lower()
            for name, value in credential_ref_hashes.items()
        }
        self.retry_policy_hashes = {
            str(name): str(value).lower()
            for name, value in retry_policy_hashes.items()
        }
        self.job_credential_slot_ref_hashes = {
            str(name): str(value or "").lower()
            for name, value in dict(
                job_credential_slot_ref_hashes
                or expected_job_credential_slot_ref_hashes()
            ).items()
        }
        if any(not _HASH_RE.fullmatch(value) for value in self.credential_ref_hashes.values()):
            raise ProviderBrokerV2Error("provider credential reference hash is invalid")
        if any(not _HASH_RE.fullmatch(value) for value in self.retry_policy_hashes.values()):
            raise ProviderBrokerV2Error("provider retry policy hash is invalid")
        if self.job_credential_slot_ref_hashes != expected_job_credential_slot_ref_hashes():
            raise ProviderBrokerV2Error(
                "job credential slot references differ from measured policy"
            )
        self._transport = (
            transport if transport is not None else HTTPXProviderTransport()
        )
        if artifact_sink is None:
            raise ProviderBrokerV2Error(
                "provider broker requires encrypted artifact persistence"
            )
        self._artifact_sink = artifact_sink
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._credentials = {}  # type: Dict[str, str]
        self._job_credentials = {}  # type: Dict[Tuple[str, str], Dict[str, str]]
        self._records = {}  # type: Dict[Tuple[str, int], Dict[str, Any]]
        self._pending_records = {}  # type: Dict[Tuple[str, int], Dict[str, Any]]
        self._record_keys_by_job = {}  # type: Dict[str, set[Tuple[str, int]]]
        self._inflight = {}  # type: Dict[Tuple[str, int], Tuple[str, threading.Event, object]]
        self._released_terminal_count = 0
        self._expired_terminal_count = 0
        self._lock = threading.Lock()
        self._transaction_state = threading.local()

    @contextmanager
    def transient_terminal_transaction(self):
        """Publish terminal records only when surrounding artifact work commits."""

        state = getattr(self._transaction_state, "current", None)
        if state is None:
            state = {
                "token": object(),
                "depth": 0,
                "created": set(),
                "failed": False,
            }
            self._transaction_state.current = state
        state["depth"] += 1
        try:
            yield
        except BaseException:
            state["failed"] = True
            raise
        finally:
            state["depth"] -= 1
            if state["depth"] == 0:
                wait_events = []
                with self._lock:
                    for key in state["created"]:
                        pending = self._pending_records.get(key)
                        if (
                            pending is None
                            or pending["transaction_token"] is not state["token"]
                        ):
                            continue
                        self._pending_records.pop(key)
                        if not state["failed"]:
                            record = dict(pending["record"])
                            self._records[key] = record
                            self._record_keys_by_job.setdefault(
                                str(record["job_id"]),
                                set(),
                            ).add(key)
                        inflight = self._inflight.get(key)
                        if (
                            inflight is not None
                            and inflight[2] is pending["owner_token"]
                        ):
                            self._inflight.pop(key)
                            wait_events.append(inflight[1])
                del self._transaction_state.current
                for wait_event in wait_events:
                    wait_event.set()

    def _purge_expired_records_locked(self, now: float) -> int:
        cutoff = float(now) - TERMINAL_RECORD_RETENTION_SECONDS
        expired = [
            key
            for key, record in self._records.items()
            if float(record["completed_monotonic"]) <= cutoff
        ]
        for key in expired:
            record = self._records.pop(key)
            job_id = str(record["job_id"])
            job_keys = self._record_keys_by_job.get(job_id)
            if job_keys is not None:
                job_keys.discard(key)
                if not job_keys:
                    self._record_keys_by_job.pop(job_id, None)
        self._expired_terminal_count += len(expired)
        return len(expired)

    def health(self) -> Dict[str, Any]:
        expected_slots = set(expected_provider_credential_slots())
        with self._lock:
            configured_slots = set(self._credentials)
            inflight_count = len(self._inflight)
            terminal_count = len(self._records)
            job_lease_count = len(self._job_credentials)
            released_terminal_count = self._released_terminal_count
            expired_terminal_count = self._expired_terminal_count
        missing = sorted(expected_slots - configured_slots)
        return {
            "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
            "status": "ready" if not missing else "provisioning",
            "credential_slots": sorted(configured_slots),
            "missing_credential_slots": missing,
            "inflight_count": inflight_count,
            "terminal_count": terminal_count,
            "released_terminal_count": released_terminal_count,
            "expired_terminal_count": expired_terminal_count,
            "job_credential_lease_count": job_lease_count,
            "registry_hash": provider_registry_hash(),
            "job_credential_slot_ref_hashes": dict(
                sorted(self.job_credential_slot_ref_hashes.items())
            ),
        }

    def _abandon_inflight(
        self,
        deduplication_key: Tuple[str, int],
        request_fingerprint: str,
    ) -> None:
        with self._lock:
            inflight = self._inflight.get(deduplication_key)
            if inflight is None or inflight[0] != request_fingerprint:
                return
            self._inflight.pop(deduplication_key, None)
            inflight[1].set()

    def _abandon_owned_inflight(self, owner_token: object) -> None:
        with self._lock:
            abandoned = [
                (key, inflight[1])
                for key, inflight in self._inflight.items()
                if inflight[2] is owner_token
                and (
                    key not in self._pending_records
                    or self._pending_records[key]["owner_token"] is not owner_token
                )
            ]
            for key, _ in abandoned:
                self._inflight.pop(key, None)
            for _, wait_event in abandoned:
                wait_event.set()

    def provision_credentials(self, credentials: Mapping[str, str]) -> Dict[str, Any]:
        expected_slots = set(expected_provider_credential_slots())
        if set(credentials) != expected_slots:
            raise ProviderBrokerV2Error("provider credential slots do not match registry")
        normalized = {}
        for slot, value in credentials.items():
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ProviderBrokerV2Error("provider credential value is invalid")
            if _credential_hash(value) != self.credential_ref_hashes.get(slot):
                raise ProviderBrokerV2Error("provider credential hash mismatch")
            normalized[slot] = value
        with self._lock:
            if self._credentials and self._credentials != normalized:
                raise ProviderBrokerV2Error("provider credentials are immutable for boot")
            self._credentials = normalized
        return {
            "status": "ready",
            "credential_slots": sorted(normalized),
        }

    def provision_credential(self, *, slot: str, credential: str) -> Dict[str, Any]:
        """Provision one KMS-unwrapped slot without exposing other credentials."""

        expected_slots = set(expected_provider_credential_slots())
        normalized_slot = str(slot or "")
        if normalized_slot not in expected_slots:
            raise ProviderBrokerV2Error("provider credential slot is not measured")
        if not isinstance(credential, str) or not credential or "\x00" in credential:
            raise ProviderBrokerV2Error("provider credential value is invalid")
        if _credential_hash(credential) != self.credential_ref_hashes.get(
            normalized_slot
        ):
            raise ProviderBrokerV2Error("provider credential hash mismatch")
        with self._lock:
            existing = self._credentials.get(normalized_slot)
            if existing is not None:
                if existing != credential:
                    raise ProviderBrokerV2Error(
                        "provider credential is immutable for boot"
                    )
            else:
                self._credentials[normalized_slot] = credential
            configured = sorted(self._credentials)
        return {
            "status": "ready" if set(configured) == expected_slots else "provisioning",
            "credential_slots": configured,
            "missing_credential_slots": sorted(expected_slots - set(configured)),
        }

    def provision_job_credential(
        self,
        *,
        job_id: str,
        slot: str,
        credential: str,
        credential_value_hash_expected: str,
    ) -> Dict[str, Any]:
        """Lease one miner-owned credential to one attested execution job."""

        normalized_job_id = str(job_id or "")
        normalized_slot = str(slot or "")
        expected_slots = {
            route.credential_slot
            for route in self.routes.values()
            if route.credential_slot
        } | set(self.job_credential_slot_ref_hashes)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", normalized_job_id):
            raise ProviderBrokerV2Error("job credential lease id is invalid")
        if normalized_slot not in expected_slots and not source_add_dynamic_job_slot(
            normalized_slot
        ):
            raise ProviderBrokerV2Error("job credential slot is not measured")
        expected_hash = str(credential_value_hash_expected or "").lower()
        if not _HASH_RE.fullmatch(expected_hash):
            raise ProviderBrokerV2Error("job credential value hash is invalid")
        if credential_value_hash(credential) != expected_hash:
            raise ProviderBrokerV2Error("job credential value hash mismatch")
        lease_key = (normalized_job_id, normalized_slot)
        lease = {
            "credential": credential,
            "credential_ref_hash": expected_hash,
        }
        with self._lock:
            existing = self._job_credentials.get(lease_key)
            if existing is not None and existing != lease:
                raise ProviderBrokerV2Error(
                    "job credential lease is immutable"
                )
            if existing is None and len(self._job_credentials) >= MAX_JOB_CREDENTIAL_LEASES:
                raise ProviderBrokerV2Error("job credential lease capacity is full")
            self._job_credentials[lease_key] = lease
        return {
            "status": "ready",
            "job_id": normalized_job_id,
            "credential_slot": normalized_slot,
            "credential_ref_hash": expected_hash,
        }

    def release_job_credentials(self, job_id: str) -> Dict[str, Any]:
        normalized_job_id = str(job_id or "")
        with self._lock:
            credential_keys = [
                key for key in self._job_credentials if key[0] == normalized_job_id
            ]
            for key in credential_keys:
                del self._job_credentials[key]
            terminal_keys = tuple(
                self._record_keys_by_job.pop(normalized_job_id, ())
            )
            for key in terminal_keys:
                self._records.pop(key, None)
            self._released_terminal_count += len(terminal_keys)
        return {
            "status": "released",
            "job_id": normalized_job_id,
            "released_slot_count": len(credential_keys),
            "released_terminal_count": len(terminal_keys),
        }

    def credential_available(self, *, job_id: str, slot: str) -> bool:
        """Return only credential availability; never expose credential bytes."""

        normalized_job_id = str(job_id or "")
        normalized_slot = str(slot or "")
        with self._lock:
            return (
                (normalized_job_id, normalized_slot) in self._job_credentials
                or normalized_slot in self._credentials
            )

    def transport_reference_hashes(
        self,
        request: Mapping[str, Any],
    ) -> Dict[str, str]:
        """Resolve the measured credential and proxy refs for one job request."""

        provider_id = str(request.get("provider_id") or "")
        parsed = urlsplit(str(request.get("url") or ""))
        method = str(request.get("method") or "").upper()
        route, _dynamic_route = self._route(
            provider_id,
            parsed,
            method=method,
            dynamic_route=request.get("dynamic_route"),
        )
        if route.allowed_methods and method not in route.allowed_methods:
            raise ProviderBrokerV2Error(
                "provider method differs from measured route"
            )
        job_id = str(request.get("job_id") or "")
        credential_ref_hash = sha256_bytes(
            ("leadpoet-no-credential:" + provider_id).encode("ascii")
        )
        with self._lock:
            if route.credential_slot:
                lease = self._job_credentials.get(
                    (job_id, route.credential_slot)
                )
                if lease is not None:
                    credential_ref_hash = str(lease["credential_ref_hash"])
                elif (
                    route.job_scoped_only
                    or route.credential_slot not in self._credentials
                ):
                    raise ProviderBrokerV2Error(
                        "provider credential slot is not provisioned"
                    )
                else:
                    credential_ref_hash = self.credential_ref_hashes[
                        route.credential_slot
                    ]
            proxy_lease = self._job_credentials.get(
                (job_id, EGRESS_PROXY_CREDENTIAL_SLOT)
            )
            egress_proxy_ref_hash = (
                str(proxy_lease["credential_ref_hash"])
                if proxy_lease is not None
                else DIRECT_EGRESS_REF_HASH
            )
        return {
            "credential_ref_hash": credential_ref_hash,
            "egress_proxy_ref_hash": egress_proxy_ref_hash,
        }

    def use_job_credential(
        self,
        *,
        job_id: str,
        slot: str,
        callback: Callable[[str], Any],
    ) -> Any:
        """Invoke measured coordinator code with a leased credential in-enclave."""

        normalized_job_id = str(job_id or "")
        normalized_slot = str(slot or "")
        if not callable(callback):
            raise ProviderBrokerV2Error("job credential callback is invalid")
        with self._lock:
            lease = self._job_credentials.get(
                (normalized_job_id, normalized_slot)
            )
            if lease is None:
                raise ProviderBrokerV2Error(
                    "job credential lease is unavailable"
                )
            credential = str(lease["credential"])
        return callback(credential)

    def _route(
        self,
        provider_id: str,
        parsed: Any,
        *,
        method: str,
        dynamic_route: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ProviderRouteV2, Optional[Dict[str, Any]]]:
        normalized_dynamic = None
        if dynamic_route is not None:
            if provider_id in self.routes:
                raise ProviderBrokerV2Error(
                    "dynamic provider route collides with a measured builtin"
                )
            try:
                normalized_dynamic = validate_source_add_runtime_route_v2(
                    dynamic_route
                )
            except Exception as exc:
                raise ProviderBrokerV2Error(
                    "dynamic provider route is invalid"
                ) from exc
            if normalized_dynamic["provider_id"] != provider_id:
                raise ProviderBrokerV2Error(
                    "dynamic provider identity differs from request"
                )
            auth_kind = normalized_dynamic["auth_kind"]
            credential_location = (
                "header" if auth_kind in {"header", "bearer"} else auth_kind
            )
            credential_name = normalized_dynamic["auth_name"]
            credential_prefix = "Bearer " if auth_kind == "bearer" else ""
            route = ProviderRouteV2(
                provider_id=provider_id,
                hosts=(normalized_dynamic["destination_host"],),
                path_prefixes=(),
                credential_slot=normalized_dynamic["credential_slot"],
                credential_location=(
                    credential_location if auth_kind != "none" else "none"
                ),
                credential_name=credential_name,
                credential_prefix=credential_prefix,
                allowed_methods=tuple(
                    sorted(
                        {
                            item["method"]
                            for item in normalized_dynamic["allowed_routes"]
                        }
                    )
                ),
                allowed_route_pairs=tuple(
                    (item["method"], item["path"])
                    for item in normalized_dynamic["allowed_routes"]
                ),
                job_scoped_only=bool(normalized_dynamic["credential_slot"]),
            )
        else:
            route = self.routes.get(provider_id)
            if route is None:
                raise ProviderBrokerV2Error("provider route is not measured")
        host, port = normalize_destination(parsed.hostname, parsed.port or 443)
        if parsed.scheme != "https" or port != 443:
            raise ProviderBrokerV2Error("provider transport requires HTTPS port 443")
        if route.hosts and host not in route.hosts:
            raise ProviderBrokerV2Error("provider destination differs from measured route")
        if route.allowed_route_pairs and (
            method,
            parsed.path or "/",
        ) not in route.allowed_route_pairs:
            raise ProviderBrokerV2Error(
                "provider method/path differs from measured route"
            )
        if route.path_prefixes and not any(
            (parsed.path or "/").startswith(prefix)
            for prefix in route.path_prefixes
        ):
            raise ProviderBrokerV2Error("provider path differs from measured route")
        return route, normalized_dynamic

    def execute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        owner_token = object()
        try:
            return self._execute(request, owner_token=owner_token)
        finally:
            self._abandon_owned_inflight(owner_token)

    def _execute(
        self,
        request: Mapping[str, Any],
        *,
        owner_token: object,
    ) -> Dict[str, Any]:
        required = {
            "schema_version",
            "logical_operation_id",
            "job_id",
            "purpose",
            "provider_id",
            "attempt_number",
            "method",
            "url",
            "headers",
            "body_b64",
            "timeout_ms",
            "retry_policy_hash",
        }
        request_fields = (
            frozenset(request) if isinstance(request, Mapping) else frozenset()
        )
        if request_fields not in {
            frozenset(required),
            frozenset(required | {"dynamic_route"}),
            frozenset(required | {"max_response_bytes", "artifact_mode"}),
            frozenset(
                required
                | {"dynamic_route", "max_response_bytes", "artifact_mode"}
            ),
        }:
            raise ProviderBrokerV2Error("provider request fields are invalid")
        if request["schema_version"] != PROVIDER_BROKER_SCHEMA_VERSION:
            raise ProviderBrokerV2Error("provider request schema is invalid")
        provider_id = str(request["provider_id"] or "")
        parsed = urlsplit(str(request["url"] or ""))
        method = str(request["method"] or "").upper()
        if not re.fullmatch(r"[A-Z]{3,12}", method):
            raise ProviderBrokerV2Error("provider method is invalid")
        route, dynamic_route = self._route(
            provider_id,
            parsed,
            method=method,
            dynamic_route=(
                request.get("dynamic_route")
                if "dynamic_route" in request
                else None
            ),
        )
        if route.allowed_methods and method not in route.allowed_methods:
            raise ProviderBrokerV2Error("provider method differs from measured route")
        max_response_bytes = MAX_RESPONSE_BODY_BYTES
        artifact_mode = "encrypted_body"
        if "max_response_bytes" in request or "artifact_mode" in request:
            source_add_provenance_summary = (
                dynamic_route is None
                and str(request.get("purpose") or "")
                == "research_lab.source_add_provenance.v2"
                and provider_id in {"scrapingdog", "wayback"}
            )
            if dynamic_route is None and not source_add_provenance_summary:
                raise ProviderBrokerV2Error(
                    "bounded hash-only artifacts require a measured SOURCE_ADD route"
                )
            max_response_bytes = request.get("max_response_bytes")
            artifact_mode = str(request.get("artifact_mode") or "")
            if (
                isinstance(max_response_bytes, bool)
                or not isinstance(max_response_bytes, int)
                or not 1 <= max_response_bytes <= 1024 * 1024
                or artifact_mode != "hash_only"
            ):
                raise ProviderBrokerV2Error(
                    "dynamic provider artifact policy is invalid"
                )
        headers = request["headers"]
        if not isinstance(headers, Mapping):
            raise ProviderBrokerV2Error("provider headers must be an object")
        if any(str(name).lower() in _SECRET_HEADER_NAMES for name in headers):
            raise ProviderBrokerV2Error("runner supplied a credential header")
        try:
            body = base64.b64decode(str(request["body_b64"]), validate=True)
        except Exception as exc:
            raise ProviderBrokerV2Error("provider body is invalid base64") from exc
        if len(body) > MAX_REQUEST_BODY_BYTES:
            raise ProviderBrokerV2Error("provider request body exceeds size limit")
        attempt_number = request["attempt_number"]
        timeout_ms = request["timeout_ms"]
        if not isinstance(attempt_number, int) or attempt_number < 0:
            raise ProviderBrokerV2Error("provider attempt number is invalid")
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ProviderBrokerV2Error("provider timeout is invalid")
        retry_policy_hash = str(request["retry_policy_hash"] or "").lower()
        expected_retry_policy_hash = (
            source_add_dynamic_retry_policy_hash(dynamic_route)
            if dynamic_route is not None
            else self.retry_policy_hashes.get(provider_id)
        )
        if retry_policy_hash != expected_retry_policy_hash:
            raise ProviderBrokerV2Error("provider retry policy hash mismatch")
        logical_operation_id = str(request["logical_operation_id"] or "")
        deduplication_key = (logical_operation_id, attempt_number)
        request_fingerprint = sha256_json(
            {
                **dict(request),
                "headers": _nonsecret_headers(headers),
                "body_b64": base64.b64encode(body).decode("ascii"),
                "url": urlunsplit(
                    (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
                ),
            }
        )
        with self._lock:
            existing = self._records.get(deduplication_key)
            if existing is not None:
                if existing["request_fingerprint"] != request_fingerprint:
                    raise ProviderBrokerV2Error(
                        "logical provider attempt was reused with different request"
                    )
                return dict(existing["result"])
            pending = self._pending_records.get(deduplication_key)
            transaction = getattr(self._transaction_state, "current", None)
            if pending is not None:
                if pending["record"]["request_fingerprint"] != request_fingerprint:
                    raise ProviderBrokerV2Error(
                        "logical provider attempt was reused with different request"
                    )
                if (
                    transaction is not None
                    and pending["transaction_token"] is transaction["token"]
                ):
                    return dict(pending["record"]["result"])
            inflight = self._inflight.get(deduplication_key)
            if inflight is not None:
                if inflight[0] != request_fingerprint:
                    raise ProviderBrokerV2Error(
                        "logical provider attempt is in flight with different request"
                    )
                wait_event = inflight[1]
                owns_attempt = False
            else:
                if (
                    len(self._records) + len(self._pending_records)
                    >= MAX_DEDUPLICATION_RECORDS
                ):
                    self._purge_expired_records_locked(
                        self._monotonic_clock()
                    )
                    if (
                        len(self._records) + len(self._pending_records)
                        >= MAX_DEDUPLICATION_RECORDS
                    ):
                        raise ProviderBrokerV2Error(
                            "provider terminal ledger is full"
                        )
                job_credential_key = (
                    str(request["job_id"] or ""),
                    route.credential_slot,
                )
                if (
                    route.credential_slot
                    and job_credential_key not in self._job_credentials
                    and (
                        route.job_scoped_only
                        or route.credential_slot not in self._credentials
                    )
                ):
                    raise ProviderBrokerV2Error(
                        "provider credential slot is not provisioned"
                    )
                wait_event = threading.Event()
                self._inflight[deduplication_key] = (
                    request_fingerprint,
                    wait_event,
                    owner_token,
                )
                owns_attempt = True
        if not owns_attempt:
            # The owner does not publish this terminal until its surrounding
            # artifact/checkpoint transaction commits.  That lifecycle can
            # legitimately outlive the upstream HTTP timeout.
            if not wait_event.wait(REPLAY_WAIT_SECONDS):
                raise ProviderBrokerV2Error("duplicate provider attempt wait timed out")
            with self._lock:
                completed = self._records.get(deduplication_key)
                if completed is None:
                    raise ProviderBrokerV2Error("duplicate provider attempt did not terminate")
                return dict(completed["result"])

        outbound_headers = {str(k): str(v) for k, v in headers.items()}
        for static_name, static_value in route.request_headers:
            outbound_headers = {
                name: value
                for name, value in outbound_headers.items()
                if name.lower() != static_name.lower()
            }
            outbound_headers[static_name] = static_value
        if dynamic_route is not None:
            static_headers = dynamic_route.get("request_headers") or {}
            for static_name, static_value in static_headers.items():
                outbound_headers = {
                    name: value
                    for name, value in outbound_headers.items()
                    if name.lower() != str(static_name).lower()
                }
                outbound_headers[str(static_name)] = str(static_value)
        # Bind response framing at the measured transport boundary, after
        # route-specific headers, so no caller or dynamic route can opt back
        # into compressed bodies whose decoded bytes cannot prove wire length.
        for static_name, static_value in MEASURED_TRANSPORT_REQUEST_HEADERS:
            outbound_headers = {
                name: value
                for name, value in outbound_headers.items()
                if name.lower() != static_name.lower()
            }
            outbound_headers[static_name] = static_value
        measured_nonsecret_headers = _nonsecret_headers(outbound_headers)
        query = list(parse_qsl(parsed.query, keep_blank_values=True))
        credential_ref_hash = sha256_bytes(
            ("leadpoet-no-credential:" + provider_id).encode("ascii")
        )
        if route.credential_slot:
            with self._lock:
                job_lease = self._job_credentials.get(
                    (str(request["job_id"] or ""), route.credential_slot)
                )
                if job_lease is not None:
                    credential = job_lease["credential"]
                    credential_ref_hash = job_lease["credential_ref_hash"]
                else:
                    if route.job_scoped_only:
                        raise ProviderBrokerV2Error(
                            "provider requires a job-scoped credential"
                        )
                    credential = self._credentials[route.credential_slot]
                    credential_ref_hash = self.credential_ref_hashes[
                        route.credential_slot
                    ]
            if route.credential_location == "header":
                outbound_headers[route.credential_name] = route.credential_prefix + credential
                for alias_name, alias_prefix in route.credential_header_aliases:
                    outbound_headers[alias_name] = alias_prefix + credential
            elif route.credential_location == "query":
                query = [
                    (name, value)
                    for name, value in query
                    if name.lower() != route.credential_name.lower()
                ]
                query.append((route.credential_name, credential))
            else:
                raise ProviderBrokerV2Error("provider credential route is invalid")
        outbound_url = urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), "")
        )
        egress_proxy_url = None
        egress_proxy_ref_hash = DIRECT_EGRESS_REF_HASH
        with self._lock:
            proxy_lease = self._job_credentials.get(
                (str(request["job_id"] or ""), EGRESS_PROXY_CREDENTIAL_SLOT)
            )
        if proxy_lease is not None:
            egress_proxy_url = _validated_tls_proxy_url(proxy_lease["credential"])
            egress_proxy_ref_hash = str(proxy_lease["credential_ref_hash"])
        started_at = self._clock()
        request_id = secrets.token_hex(16)
        request_artifact_doc = {
            "schema_version": "leadpoet.provider_request_artifact.v2",
            "request_id": request_id,
            "logical_operation_id": logical_operation_id,
            "job_id": str(request["job_id"] or ""),
            "purpose": str(request["purpose"] or ""),
            "provider_id": provider_id,
            "attempt_number": attempt_number,
            "method": method,
            "url": (
                urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
                if artifact_mode == "encrypted_body"
                else ""
            ),
            "path_hash": sha256_bytes((parsed.path or "/").encode("utf-8")),
            "query_hash": sha256_bytes(parsed.query.encode("utf-8")),
            "headers": (
                measured_nonsecret_headers
                if artifact_mode == "encrypted_body"
                else {}
            ),
            "nonsecret_headers_hash": sha256_json(measured_nonsecret_headers),
            "body_b64": (
                base64.b64encode(body).decode("ascii")
                if artifact_mode == "encrypted_body"
                else ""
            ),
            "body_hash": sha256_bytes(body),
            "timeout_ms": timeout_ms,
            "max_response_bytes": max_response_bytes,
            "artifact_mode": artifact_mode,
            "retry_policy_hash": retry_policy_hash,
            "egress_proxy_ref_hash": egress_proxy_ref_hash,
            "dynamic_route_hash": (
                str(dynamic_route["route_hash"])
                if dynamic_route is not None
                else ""
            ),
        }
        try:
            request_artifact_bytes = canonical_json(request_artifact_doc).encode("utf-8")
            request_artifact = dict(
                self._artifact_sink(
                    request_artifact_bytes,
                    job_id=str(request["job_id"] or ""),
                    purpose=str(request["purpose"] or ""),
                    artifact_kind="provider_request",
                )
            )
            request_artifact_hash = sha256_bytes(request_artifact_bytes)
            if request_artifact.get("plaintext_hash") != request_artifact_hash:
                raise ProviderBrokerV2Error(
                    "encrypted provider request artifact plaintext hash mismatch"
                )
            request_artifact_id = str(request_artifact.get("artifact_id") or "")
            if not _HASH_RE.fullmatch(request_artifact_id):
                raise ProviderBrokerV2Error(
                    "encrypted provider request artifact ID is invalid"
                )
        except Exception:
            self._abandon_inflight(deduplication_key, request_fingerprint)
            raise
        evidence_artifact_hashes = {
            str(request_artifact[field])
            for field in (
                "artifact_id",
                "plaintext_hash",
                "ciphertext_hash",
                "encryption_context_hash",
            )
            if request_artifact.get(field)
        }
        terminal_kwargs = {}  # type: Dict[str, Any]
        response_payload = None
        failure_stage = "provider_transport"
        try:
            transport_kwargs = {
                "method": method,
                "url": outbound_url,
                "headers": outbound_headers,
                "body": body,
                "timeout_ms": timeout_ms,
            }
            if egress_proxy_url is not None:
                transport_kwargs["upstream_proxy_url"] = egress_proxy_url
            if max_response_bytes != MAX_RESPONSE_BODY_BYTES:
                transport_kwargs["max_response_bytes"] = max_response_bytes
            response = dict(self._transport(**transport_kwargs))
            response_body = bytes(response["body"])
            if len(response_body) > max_response_bytes:
                raise ProviderBrokerV2Error("provider response exceeds size limit")
            response_artifact_body = response_body
            response_artifact_kind = "provider_response"
            if artifact_mode == "hash_only":
                response_artifact_body = canonical_json(
                    {
                        "schema_version": "leadpoet.provider_response_summary.v2",
                        "http_status": int(response["http_status"]),
                        "content_type": str(
                            _nonsecret_headers(response.get("headers", {})).get(
                                "content-type", ""
                            )
                        )[:160],
                        "byte_count": len(response_body),
                        "response_hash": sha256_bytes(response_body),
                    }
                ).encode("utf-8")
                response_artifact_kind = "provider_response_summary"
            failure_stage = "response_artifact_persistence"
            artifact = dict(
                self._artifact_sink(
                    response_artifact_body,
                    job_id=str(request["job_id"] or ""),
                    purpose=str(request["purpose"] or ""),
                    artifact_kind=response_artifact_kind,
                )
            )
            if artifact.get("plaintext_hash") != sha256_bytes(response_artifact_body):
                raise ProviderBrokerV2Error(
                    "encrypted provider artifact plaintext hash mismatch"
                )
            artifact_id = str(artifact.get("artifact_id") or "")
            if not _HASH_RE.fullmatch(artifact_id):
                raise ProviderBrokerV2Error("encrypted provider artifact ID is invalid")
            evidence_artifact_hashes.update(
                str(artifact[field])
                for field in (
                    "artifact_id",
                    "plaintext_hash",
                    "ciphertext_hash",
                    "encryption_context_hash",
                )
                if artifact.get(field)
            )
            terminal_kwargs = {
                "terminal_status": "authenticated_response",
                "http_status": int(response["http_status"]),
                "response_hash": sha256_bytes(response_body),
                "response_artifact_hash": str(artifact["plaintext_hash"]),
                "tls_peer_chain_hash": str(response["tls_peer_chain_hash"]),
                "tls_protocol": str(response["tls_protocol"]),
                "failure_code": None,
            }
            response_payload = {
                "terminal_status": "authenticated_response",
                "http_status": int(response["http_status"]),
                "headers": _nonsecret_headers(response.get("headers", {})),
                "body_b64": base64.b64encode(response_body).decode("ascii"),
                "encrypted_request_artifact_id": request_artifact_id,
                "encrypted_artifact_id": artifact_id,
            }
        except Exception as exc:
            terminal_kwargs = {
                "terminal_status": "transport_failure",
                "http_status": None,
                "response_hash": None,
                "response_artifact_hash": None,
                "tls_peer_chain_hash": None,
                "tls_protocol": None,
                "failure_code": _failure_code(exc),
            }
            response_payload = {
                "terminal_status": "transport_failure",
                "failure_code": terminal_kwargs["failure_code"],
                "failure_stage": failure_stage,
                "failure_error_type": type(exc).__name__,
                "encrypted_request_artifact_id": request_artifact_id,
            }
        attempt = build_transport_attempt(
            request_id=request_id,
            logical_operation_id=logical_operation_id,
            job_id=str(request["job_id"] or ""),
            purpose=str(request["purpose"] or ""),
            provider_id=provider_id,
            attempt_number=attempt_number,
            method=method,
            destination_host=str(parsed.hostname or ""),
            destination_port=parsed.port or 443,
            path_hash=sha256_bytes(_sanitized_path(parsed).encode("utf-8")),
            nonsecret_headers_hash=sha256_json(measured_nonsecret_headers),
            body_hash=sha256_bytes(body),
            credential_ref_hash=credential_ref_hash,
            egress_proxy_ref_hash=egress_proxy_ref_hash,
            retry_policy_hash=retry_policy_hash,
            timeout_ms=timeout_ms,
            started_at=started_at,
            request_artifact_hash=request_artifact_hash,
            completed_at=self._clock(),
            **terminal_kwargs,
        )
        result = {
            **response_payload,
            "transport_attempt": attempt,
            "evidence_artifact_hashes": sorted(evidence_artifact_hashes),
        }
        with self._lock:
            existing = self._records.get(deduplication_key)
            if existing is not None:
                if existing["request_fingerprint"] != request_fingerprint:
                    raise ProviderBrokerV2Error(
                        "logical provider attempt raced with different request"
                    )
                return dict(existing["result"])
            record = {
                "job_id": str(request["job_id"] or ""),
                "request_fingerprint": request_fingerprint,
                "result": dict(result),
                "completed_monotonic": self._monotonic_clock(),
            }
            transaction = getattr(self._transaction_state, "current", None)
            if transaction is not None:
                self._pending_records[deduplication_key] = {
                    "record": record,
                    "owner_token": owner_token,
                    "transaction_token": transaction["token"],
                }
                transaction["created"].add(deduplication_key)
            else:
                self._records[deduplication_key] = record
                self._record_keys_by_job.setdefault(
                    str(request["job_id"] or ""),
                    set(),
                ).add(deduplication_key)
                inflight = self._inflight.pop(deduplication_key, None)
                if inflight is not None:
                    inflight[1].set()
        return result


def credential_reference_hash(value: str) -> str:
    """Build the non-secret boot config commitment for one credential."""

    return _credential_hash(value)
