"""Coordinator-owned HTTPS provider broker with terminal V2 transport records."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import errno
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
import weakref
import zlib

from httpx import SyncByteStream

from gateway.tee.egress_framing import (
    TUNNEL_FRAMING_HEADER,
    TUNNEL_FRAMING_MODE,
)
from gateway.tee.egress_policy import normalize_destination, normalize_proxy_destination
from gateway.tee.egress_proxy import DEFAULT_IDLE_TIMEOUT_SECONDS
from gateway.tee.inter_enclave_tls import MAX_FRAME_BYTES, REPLAY_WAIT_SECONDS
from gateway.tee.rpc_authority import (
    COORDINATOR_ROLE,
    RPCAuthorityError,
    active_enclave_role,
)
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_transport_attempt,
)
from gateway.tee.source_add_runtime_v2 import (
    source_add_dynamic_job_slot,
    source_add_dynamic_retry_policy_hash,
    validate_source_add_runtime_route_v2,
)
PROVIDER_BROKER_SCHEMA_VERSION = "leadpoet.provider_broker.v2"
PROVIDER_TRANSPORT_HEALTH_SCHEMA_VERSION = "leadpoet.provider_transport_health.v2"
PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION = (
    "leadpoet.provider_transport_failure_diagnostic.v2"
)
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
EGRESS_POLICY_JOB_PROXY_ALLOWED = "job_proxy_allowed"
EGRESS_POLICY_DIRECT_ONLY = "direct_only"
_EGRESS_POLICIES = frozenset(
    {EGRESS_POLICY_JOB_PROXY_ALLOWED, EGRESS_POLICY_DIRECT_ONLY}
)
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
_LOCAL_RESOURCE_FAILURES = (
    (
        getattr(errno, "EADDRNOTAVAIL", 99),
        "cannot assign requested address",
        "ephemeral_port_exhausted",
    ),
    (
        getattr(errno, "EADDRINUSE", 98),
        "address already in use",
        "local_address_in_use",
    ),
    (
        getattr(errno, "ENFILE", 23),
        "too many open files in system",
        "system_file_descriptor_exhausted",
    ),
    (
        getattr(errno, "EMFILE", 24),
        "too many open files",
        "process_file_descriptor_exhausted",
    ),
    (
        getattr(errno, "ENOBUFS", 105),
        "no buffer space available",
        "socket_buffer_exhausted",
    ),
    (
        getattr(errno, "ENOMEM", 12),
        "cannot allocate memory",
        "memory_exhausted",
    ),
)
_LOCAL_RESOURCE_ERRNO_KINDS = {
    int(error_number): kind
    for error_number, _token, kind in _LOCAL_RESOURCE_FAILURES
}
_SAFE_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,79}$")
_TRANSPORT_ROUTES = ("direct", "assigned_proxy")
_PROVIDER_TERMINAL_STATUSES = (
    "authenticated_response",
    "transport_failure",
)
_CHAIN_WEIGHT_OBSERVATION_PURPOSE = (
    "research_lab.chain_weight_observation.v1"
)
_EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE = "_leadpoet_explicit_http_transport"
_BROKER_OWNED_HTTPX_CLIENTS_LOCK = threading.Lock()
_BROKER_OWNED_HTTPX_CLIENTS: Dict[
    int,
    Tuple[weakref.ReferenceType[Any], Any],
] = {}
_BROKER_OWNED_HTTPX_SEND_GRANT = contextvars.ContextVar(
    "leadpoet_broker_owned_httpx_send_grant",
    default=None,
)
_PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "provider",
        "request_hash",
        "attempt_number",
        "failure_stage",
        "outer_error_type",
        "primary_error_type",
        "cleanup_error_type",
        "cleanup_errno",
        "cleanup_resource_kind",
    }
)
_PROVIDER_TRANSPORT_FAILURE_STAGES = frozenset(
    {
        "provider_request",
        "response_stream_cleanup",
        "client_transport_cleanup",
    }
)
_CLEANUP_RESOURCE_KIND_BY_STAGE = {
    "response_stream_cleanup": "network_stream",
    "client_transport_cleanup": "client_transport",
}
_MAX_DIAGNOSTIC_ERRNO = 65535


class ProviderBrokerV2Error(RuntimeError):
    """A request violates the measured provider route or terminal ledger."""


class ProviderTransportCleanupError(ProviderBrokerV2Error):
    """A request-owned response stream or explicit pool did not close."""

    def __init__(
        self,
        *,
        stage: str,
        primary_error: Optional[BaseException],
        cleanup_error: BaseException,
    ) -> None:
        super().__init__("provider request-scoped transport cleanup failed")
        self.failure_stage = str(stage)
        self.primary_error_type = (
            _safe_error_type(primary_error) if primary_error is not None else ""
        )
        self.cleanup_error_type = _safe_error_type(cleanup_error)
        self.cleanup_errno, self.cleanup_resource_kind = (
            _local_resource_failure(cleanup_error)
        )
        if not self.cleanup_errno:
            self.cleanup_errno = _exception_errno(cleanup_error)


def _register_broker_owned_httpx_client(
    client: Any,
    explicit_transport: Any,
) -> None:
    """Bind one module-created HTTPX client to its explicit transport."""

    if getattr(client, _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE, None) is not (
        explicit_transport
    ):
        raise ProviderBrokerV2Error(
            "provider HTTPX client transport binding is invalid"
        )
    client_id = id(client)

    def forget(reference: weakref.ReferenceType[Any]) -> None:
        with _BROKER_OWNED_HTTPX_CLIENTS_LOCK:
            current = _BROKER_OWNED_HTTPX_CLIENTS.get(client_id)
            if current is not None and current[0] is reference:
                _BROKER_OWNED_HTTPX_CLIENTS.pop(client_id, None)

    reference = weakref.ref(client, forget)
    with _BROKER_OWNED_HTTPX_CLIENTS_LOCK:
        _BROKER_OWNED_HTTPX_CLIENTS[client_id] = (
            reference,
            explicit_transport,
        )


def is_broker_owned_httpx_client(client: Any) -> bool:
    """Return true only for the exact HTTPX client created by this broker."""

    if _BROKER_OWNED_HTTPX_SEND_GRANT.get() is not client:
        return False
    try:
        if active_enclave_role() != COORDINATOR_ROLE:
            return False
    except RPCAuthorityError:
        return False
    with _BROKER_OWNED_HTTPX_CLIENTS_LOCK:
        current = _BROKER_OWNED_HTTPX_CLIENTS.get(id(client))
        if current is None:
            return False
        reference, explicit_transport = current
        return (
            reference() is client
            and getattr(
                client,
                _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE,
                None,
            )
            is explicit_transport
            and getattr(client, "_transport", None) is explicit_transport
            and getattr(client, "_mounts", None) == {}
            and getattr(client, "follow_redirects", None) is False
        )


@contextmanager
def _broker_owned_httpx_send_scope(client: Any, *args: Any, **kwargs: Any):
    """Grant captured-send access only while the broker opens one stream."""

    if _BROKER_OWNED_HTTPX_SEND_GRANT.get() is not None:
        raise ProviderBrokerV2Error("provider HTTPX send grant is already active")
    token = _BROKER_OWNED_HTTPX_SEND_GRANT.set(client)
    try:
        with client.stream(*args, **kwargs) as response:
            yield response
    finally:
        _BROKER_OWNED_HTTPX_SEND_GRANT.reset(token)


_EGRESS_PROXY_SLOT_REF_HASH = sha256_json(
    {
        "schema_version": "leadpoet.job_credential_slot.v2",
        "credential_slot": EGRESS_PROXY_CREDENTIAL_SLOT,
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
    allow_http2: bool = True
    egress_policy: str = EGRESS_POLICY_JOB_PROXY_ALLOWED


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
    "exa": ProviderRouteV2(
        provider_id="exa",
        hosts=("api.exa.ai",),
        credential_slot="exa",
        credential_location="header",
        credential_name="x-api-key",
        # Exa's assigned-proxy path can deliver the complete authenticated
        # response body while losing only the terminal HTTP marker. A single
        # gzip member supplies an independently checksum-verified boundary.
        request_headers=(("Accept-Encoding", "gzip"),),
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
        # Control, cache, outcome, and weight persistence are global V2
        # authorities. Their authenticated Supabase transport must never
        # inherit the paid provider profile assigned to a scoring job.
        egress_policy=EGRESS_POLICY_DIRECT_ONLY,
        # Retain the global identity profile used by the known-good measured
        # PostgREST path. The bounded JSON document provides the objective
        # completeness boundary if a raw HTTP/2 relay loses only END_STREAM.
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
        path_prefixes=(
            "/wayback/available",
            "/cdx/search/cdx",
            "/wayback/cdx",
        ),
    ),
    "public_web": ProviderRouteV2(
        provider_id="public_web",
        hosts=(),
    ),
}


def provider_routes_for_execution_config(
    execution_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, ProviderRouteV2]:
    from leadpoet_canonical.production_parity_boundary_v2 import (
        configured_boundary_document_v2,
        configured_supabase_origin_v2,
        validate_production_parity_boundary_document_v2,
    )

    if execution_config is None:
        boundary = configured_boundary_document_v2()
        origin = configured_supabase_origin_v2()
    else:
        from gateway.tee.research_lab_runtime_config_v2 import (
            validate_research_lab_execution_config,
        )

        normalized = validate_research_lab_execution_config(execution_config)
        boundary = validate_production_parity_boundary_document_v2(
            normalized["behavior_environment"],
            network=str(normalized["deployment"]["network"]),
            netuid=int(normalized["deployment"]["netuid"]),
        )
        origin = str(boundary["supabase_origin"])
    hostname = str(urlsplit(origin).hostname or "").lower()
    if not hostname:
        raise ProviderBrokerV2Error("Supabase provider origin is invalid")
    routes = dict(BUILTIN_PROVIDER_ROUTES)
    routes["supabase"] = replace(routes["supabase"], hosts=(hostname,))
    routes["bittensor_chain"] = replace(
        routes["bittensor_chain"], hosts=(str(boundary["chain_host"]),)
    )
    routes["bittensor_archive"] = replace(
        routes["bittensor_archive"],
        hosts=(str(boundary["chain_archive_host"]),),
    )
    return routes


def provider_registry_document(
    routes: Optional[Mapping[str, ProviderRouteV2]] = None,
    *,
    execution_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if routes is not None and execution_config is not None:
        raise ProviderBrokerV2Error(
            "provider registry accepts routes or execution config, not both"
        )
    resolved_routes = dict(
        routes
        if routes is not None
        else provider_routes_for_execution_config(execution_config)
    )
    def route_document(route: ProviderRouteV2) -> Dict[str, Any]:
        if route.egress_policy not in _EGRESS_POLICIES:
            raise ProviderBrokerV2Error(
                "provider route egress policy is invalid"
            )
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
            "egress_policy": route.egress_policy,
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
        if not route.allow_http2:
            document["http_versions"] = ["HTTP/1.1"]
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
            for provider_id, route in sorted(resolved_routes.items())
        },
    }


def provider_registry_hash(
    routes: Optional[Mapping[str, ProviderRouteV2]] = None,
    *,
    execution_config: Optional[Mapping[str, Any]] = None,
) -> str:
    return sha256_json(
        provider_registry_document(routes, execution_config=execution_config)
    )


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
    return {EGRESS_PROXY_CREDENTIAL_SLOT: _EGRESS_PROXY_SLOT_REF_HASH}


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


def _exception_chain(exc: BaseException) -> Tuple[BaseException, ...]:
    chain = []
    seen = set()
    current = exc  # type: Optional[BaseException]
    while current is not None and len(chain) < 8 and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        current = (
            cause
            if isinstance(cause, BaseException)
            else context if isinstance(context, BaseException) else None
        )
    return tuple(chain)


def _local_resource_failure(exc: BaseException) -> Tuple[int, str]:
    chain = _exception_chain(exc)
    for current in chain:
        error_number = getattr(current, "errno", None)
        if (
            isinstance(error_number, int)
            and not isinstance(error_number, bool)
            and error_number in _LOCAL_RESOURCE_ERRNO_KINDS
        ):
            return int(error_number), _LOCAL_RESOURCE_ERRNO_KINDS[error_number]
    for current in chain:
        text = str(current).lower()
        for error_number, token, kind in _LOCAL_RESOURCE_FAILURES:
            if token in text:
                return int(error_number), kind
    return 0, ""


def _exception_errno(exc: BaseException) -> int:
    for current in _exception_chain(exc):
        error_number = getattr(current, "errno", None)
        if isinstance(error_number, int) and not isinstance(error_number, bool):
            return int(error_number)
    return 0


def _safe_error_type(exc: BaseException) -> str:
    name = str(type(exc).__name__ or "")
    return name if _SAFE_ERROR_TYPE_RE.fullmatch(name) else "Exception"


def validate_provider_transport_failure_diagnostic(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the exact encrypted-only transport failure projection."""

    if (
        not isinstance(value, Mapping)
        or frozenset(value) != _PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_FIELDS
    ):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic fields are invalid"
        )
    if (
        value.get("schema_version")
        != PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic schema is invalid"
        )
    provider = str(value.get("provider") or "")
    request_hash = str(value.get("request_hash") or "")
    attempt_number = value.get("attempt_number")
    failure_stage = str(value.get("failure_stage") or "")
    outer_error_type = str(value.get("outer_error_type") or "")
    primary_error_type = str(value.get("primary_error_type") or "")
    cleanup_error_type = str(value.get("cleanup_error_type") or "")
    cleanup_errno = value.get("cleanup_errno")
    cleanup_resource_kind = str(value.get("cleanup_resource_kind") or "")
    if not _PROVIDER_ID_RE.fullmatch(provider):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic provider is invalid"
        )
    if not _HASH_RE.fullmatch(request_hash):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic request hash is invalid"
        )
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 0
    ):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic attempt is invalid"
        )
    if failure_stage not in _PROVIDER_TRANSPORT_FAILURE_STAGES:
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic stage is invalid"
        )
    if not _SAFE_ERROR_TYPE_RE.fullmatch(outer_error_type):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic outer error is invalid"
        )
    if primary_error_type and not _SAFE_ERROR_TYPE_RE.fullmatch(
        primary_error_type
    ):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic primary error is invalid"
        )
    if cleanup_error_type and not _SAFE_ERROR_TYPE_RE.fullmatch(
        cleanup_error_type
    ):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic cleanup error is invalid"
        )
    if (
        isinstance(cleanup_errno, bool)
        or not isinstance(cleanup_errno, int)
        or not 0 <= cleanup_errno <= _MAX_DIAGNOSTIC_ERRNO
    ):
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic cleanup errno is invalid"
        )
    expected_resource_kind = _CLEANUP_RESOURCE_KIND_BY_STAGE.get(
        failure_stage,
        "",
    )
    if cleanup_resource_kind != expected_resource_kind:
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic cleanup resource is invalid"
        )
    if failure_stage == "provider_request":
        if (
            not primary_error_type
            or cleanup_error_type
            or cleanup_errno
        ):
            raise ProviderBrokerV2Error(
                "provider transport failure diagnostic request error is invalid"
            )
    elif not cleanup_error_type:
        raise ProviderBrokerV2Error(
            "provider transport failure diagnostic cleanup provenance is invalid"
        )
    return {
        "schema_version": PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION,
        "provider": provider,
        "request_hash": request_hash,
        "attempt_number": attempt_number,
        "failure_stage": failure_stage,
        "outer_error_type": outer_error_type,
        "primary_error_type": primary_error_type,
        "cleanup_error_type": cleanup_error_type,
        "cleanup_errno": cleanup_errno,
        "cleanup_resource_kind": cleanup_resource_kind,
    }


def _provider_transport_failure_diagnostic(
    *,
    provider: str,
    request_hash: str,
    attempt_number: int,
    exc: BaseException,
) -> Dict[str, Any]:
    if isinstance(exc, ProviderTransportCleanupError):
        failure_stage = str(getattr(exc, "failure_stage", "") or "")
        if failure_stage not in _CLEANUP_RESOURCE_KIND_BY_STAGE:
            raise ProviderBrokerV2Error(
                "provider transport cleanup failure stage is invalid"
            )
    else:
        failure_stage = "provider_request"
    if failure_stage == "provider_request":
        primary_error_type = _safe_error_type(exc)
        cleanup_error_type = ""
        cleanup_errno = 0
        cleanup_resource_kind = ""
    else:
        primary_error_type = str(
            getattr(exc, "primary_error_type", "") or ""
        )
        cleanup_error_type = str(
            getattr(exc, "cleanup_error_type", "") or ""
        )
        raw_cleanup_errno = getattr(exc, "cleanup_errno", 0)
        cleanup_errno = (
            int(raw_cleanup_errno)
            if isinstance(raw_cleanup_errno, int)
            and not isinstance(raw_cleanup_errno, bool)
            and 0 <= raw_cleanup_errno <= _MAX_DIAGNOSTIC_ERRNO
            else 0
        )
        cleanup_resource_kind = _CLEANUP_RESOURCE_KIND_BY_STAGE[
            failure_stage
        ]
    return validate_provider_transport_failure_diagnostic(
        {
            "schema_version": (
                PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
            ),
            "provider": str(provider),
            "request_hash": str(request_hash),
            "attempt_number": attempt_number,
            "failure_stage": failure_stage,
            "outer_error_type": _safe_error_type(exc),
            "primary_error_type": primary_error_type,
            "cleanup_error_type": cleanup_error_type,
            "cleanup_errno": cleanup_errno,
            "cleanup_resource_kind": cleanup_resource_kind,
        }
    )


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, ProviderTransportCleanupError):
        return "proxy_failure"
    if _local_resource_failure(exc)[1]:
        return "proxy_failure"
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


class _ExplicitProviderTransportCloseFailure(ProviderBrokerV2Error):
    """A provider transport adapter explicitly retained its resource."""


def _force_close_response_network_stream(
    response: Any,
    *,
    required: bool = False,
) -> None:
    """Close the one request-owned network stream before its client pool."""

    extensions = getattr(response, "extensions", None)
    network_stream = (
        extensions.get("network_stream")
        if isinstance(extensions, Mapping)
        else None
    )
    if network_stream is None:
        if required:
            raise ProviderBrokerV2Error(
                "provider response network stream is unavailable"
            )
        return
    close_error = None  # type: Optional[BaseException]
    try:
        if network_stream.close() is not False:
            return
        close_error = _ExplicitProviderTransportCloseFailure(
            "provider response network stream close was not confirmed"
        )
    except Exception as exc:
        close_error = exc
    try:
        raw_socket = network_stream.get_extra_info("socket")
    except Exception as exc:
        if close_error is None:
            close_error = exc
        raw_socket = None
    if raw_socket is None:
        raise ProviderBrokerV2Error(
            "provider response network stream cleanup failed"
        ) from close_error
    try:
        raw_socket.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        if raw_socket.close() is not False:
            return
        raise _ExplicitProviderTransportCloseFailure(
            "provider response raw socket close was not confirmed"
        )
    except Exception as exc:
        raise ProviderBrokerV2Error(
            "provider response network stream cleanup failed"
        ) from (close_error or exc)


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
                # A cleanup EOF is not response evidence, but swallowing it
                # without closing the underlying relay socket leaks one
                # request-scoped assigned-proxy tunnel per failed cleanup.
                try:
                    _force_close_response_network_stream(
                        response,
                        required=True,
                    )
                except Exception:
                    # Request-scoped execution repeats this as a required
                    # close in its outer finally, where failure is attributed
                    # without replacing an in-flight body exception here.
                    pass

    response.stream = CleanupSafeStream()


def _close_client_transports(
    client: Any,
) -> Tuple[bool, bool, Optional[BaseException]]:
    """Close the client and its one explicit transport independently."""

    client_error = None  # type: Optional[BaseException]
    try:
        if client.close() is False:
            client_error = _ExplicitProviderTransportCloseFailure(
                "provider client close was not confirmed"
            )
    except Exception as exc:
        client_error = exc
    explicit_transport = getattr(
        client,
        _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE,
        None,
    )
    if explicit_transport is None:
        return client_error is None, False, client_error
    try:
        if explicit_transport.close() is False:
            transport_error = _ExplicitProviderTransportCloseFailure(
                "provider explicit transport close was not confirmed"
            )
            return (
                client_error is None,
                False,
                client_error or transport_error,
            )
        return client_error is None, True, client_error
    except Exception as exc:
        return client_error is None, False, exc


def _close_client_nonfatal(client: Any) -> bool:
    """Close all known client transports and report, never hide, failure."""

    client_closed, transport_closed, _error = _close_client_transports(client)
    return client_closed and transport_closed


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
        if normalized_method not in {"GET", "POST"}:
            return False
        # HTTP/2 carries message completion in DATA-frame END_STREAM rather
        # than HTTP/1 Content-Length or Transfer-Encoding headers. The parent
        # relay can lose that terminal signal after delivering every
        # authenticated TLS byte. A complete JSON document is an objective
        # application boundary: truncating an array or object before its final
        # delimiter cannot still parse, and trailing non-whitespace is
        # rejected by json.loads. Restrict this recovery to authenticated 2xx
        # HTTP/2 JSON GET/POST exchanges; unframed HTTP/1 responses and other
        # mutating methods remain ambiguous.
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

    # PostgREST and measured providers legitimately stream bounded JSON with
    # chunked framing. A relay may preserve the complete authenticated JSON
    # body but lose the terminal zero-length chunk. Recover only when the body
    # itself provides an objective completeness boundary; every other chunked
    # response remains fail-closed.
    if (
        normalized_method not in {"GET", "POST"}
        or transfer_encoding != "chunked"
        or tokens
    ):
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


def _append_bounded_gzip_bytes(
    *,
    decoder: Any,
    compressed: bytes,
    chunks: list[bytes],
    byte_count: int,
    max_response_bytes: int,
) -> int:
    """Decode one gzip chunk without allocating beyond the response ceiling."""
    pending = bytes(compressed)
    while pending:
        remaining = max_response_bytes - byte_count
        decoded = decoder.decompress(pending, remaining + 1)
        if len(decoded) > remaining:
            raise ProviderBrokerV2Error("provider response exceeds size limit")
        if decoded:
            chunks.append(decoded)
            byte_count += len(decoded)
        next_pending = decoder.unconsumed_tail
        if decoder.unused_data:
            raise ProviderBrokerV2Error(
                "provider gzip response contains trailing data"
            )
        if next_pending == pending:
            raise ProviderBrokerV2Error(
                "provider gzip response decoder made no progress"
            )
        pending = next_pending
    return byte_count


def _authenticated_gzip_body_is_complete_after_stream_error(
    *,
    method: str,
    response: Any,
    byte_count: int,
    body: bytes,
    decoder: Any,
) -> bool:
    """Recover only after a complete, checksum-verified gzip JSON member."""
    if str(method or "").upper() not in {"GET", "POST"}:
        return False
    try:
        http_version = str(response.http_version or "").strip().upper()
    except Exception:
        http_version = ""
    if http_version not in {"HTTP/1.1", "HTTP/2"}:
        return False
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        return False
    return _authenticated_json_body_is_complete(
        response=response,
        byte_count=byte_count,
        body=body,
    )


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
        upstream_parent_tunnel_framing: Optional[str] = None,
        reuse_direct_connections: bool = False,
        reuse_upstream_proxy_connections: bool = False,
        ensure_egress_ready: Optional[Callable[[], Any]] = None,
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
        if (
            upstream_parent_tunnel_framing is not None
            and upstream_parent_tunnel_framing not in {"", TUNNEL_FRAMING_MODE}
        ):
            raise ProviderBrokerV2Error(
                "provider transport upstream tunnel framing is invalid"
            )
        if not isinstance(reuse_direct_connections, bool):
            raise ProviderBrokerV2Error(
                "provider transport connection reuse policy is invalid"
            )
        if not isinstance(reuse_upstream_proxy_connections, bool):
            raise ProviderBrokerV2Error(
                "provider transport upstream proxy reuse policy is invalid"
            )
        if ensure_egress_ready is not None and not callable(ensure_egress_ready):
            raise ProviderBrokerV2Error(
                "provider transport egress readiness callback is invalid"
            )
        self.proxy_url = proxy_url
        self.ca_bundle = ca_bundle
        self.response_body_ceiling_bytes = response_body_ceiling_bytes
        self.allow_authenticated_complete_body_eof = (
            allow_authenticated_complete_body_eof
        )
        self.parent_tunnel_framing = parent_tunnel_framing
        self.upstream_parent_tunnel_framing = (
            parent_tunnel_framing
            if upstream_parent_tunnel_framing is None
            else upstream_parent_tunnel_framing
        )
        self.reuse_direct_connections = reuse_direct_connections
        self.reuse_upstream_proxy_connections = (
            reuse_upstream_proxy_connections
        )
        self._ensure_egress_ready = ensure_egress_ready
        # Weight reconstruction fans out several independent direct Supabase
        # jobs at once. Admit only one direct TLS exchange at a time whether
        # the caller retains or isolates its raw relay generation, preventing
        # both concurrent handshakes and failure sharing between requests.
        # Assigned-proxy jobs retain their separate immutable-scope clients.
        self._direct_request_slot = threading.Lock()
        self._direct_client = None
        self._direct_client_lock = threading.Lock()
        self._direct_client_generation = 0
        self._direct_client_leases: Dict[int, int] = {}
        self._retired_direct_clients: Dict[int, Any] = {}
        self._proxy_clients: Dict[str, Tuple[Any, int]] = {}
        self._proxy_client_lock = threading.Lock()
        self._proxy_client_generation = 0
        self._proxy_client_leases: Dict[int, int] = {}
        self._retired_proxy_clients: Dict[int, Any] = {}
        self._transport_health_lock = threading.Lock()
        self._transport_attempt_counts = {
            route: {"started": 0, "succeeded": 0, "failed": 0}
            for route in _TRANSPORT_ROUTES
        }
        self._transport_cleanup_counts = {
            route: {
                "attempted": 0,
                "succeeded": 0,
                "client_close_failed": 0,
                "transport_close_failed": 0,
            }
            for route in _TRANSPORT_ROUTES
        }
        self._last_transport_failure = None  # type: Optional[Dict[str, Any]]

    def _new_client(
        self,
        *,
        proxy_headers: Optional[Mapping[str, str]] = None,
        allow_http2: bool = True,
        retain_direct_connections: bool = False,
        retain_assigned_proxy_connections: bool = False,
    ) -> Any:
        if self._ensure_egress_ready is not None:
            # The proxy's listener is process-lifetime authority, but its
            # accept thread can stop after a local resource fault. Re-establish
            # the exact loopback listener before opening a request-owned pool.
            self._ensure_egress_ready()
        import certifi
        import httpx
        from http.cookiejar import CookieJar, DefaultCookiePolicy

        class RejectAllCookies(DefaultCookiePolicy):
            def set_ok(self, cookie: Any, request: Any) -> bool:
                return False

        verify_path = self.ca_bundle or certifi.where()
        normalized_proxy_headers = dict(proxy_headers or {})
        assigned_proxy = bool(normalized_proxy_headers)
        if not isinstance(retain_direct_connections, bool) or not isinstance(
            retain_assigned_proxy_connections,
            bool,
        ):
            raise ProviderBrokerV2Error(
                "provider transport retention policy is invalid"
            )
        if retain_direct_connections and assigned_proxy:
            raise ProviderBrokerV2Error(
                "direct retention cannot apply to an assigned proxy"
            )
        if retain_assigned_proxy_connections and not assigned_proxy:
            raise ProviderBrokerV2Error(
                "assigned proxy retention requires proxy metadata"
            )
        # Direct provider and assigned-proxy tunnels have independent terminal
        # behavior. Keep their framing policies separate while TLS and proxy
        # credentials continue to terminate inside the enclave.
        tunnel_framing = (
            self.upstream_parent_tunnel_framing
            if assigned_proxy
            else self.parent_tunnel_framing
        )
        if tunnel_framing:
            normalized_proxy_headers[TUNNEL_FRAMING_HEADER] = tunnel_framing
        limits = httpx.Limits(
            max_connections=64,
            # An ordinary request-scoped client retains no idle tunnel. The
            # dormant retained modes must opt in explicitly and remain bound
            # to their measured direct or immutable job scope.
            max_keepalive_connections=(
                32
                if (
                    retain_assigned_proxy_connections
                    if assigned_proxy
                    else retain_direct_connections
                )
                else 0
            ),
            keepalive_expiry=DIRECT_PROVIDER_KEEPALIVE_EXPIRY_SECONDS,
        )
        explicit_transport = httpx.HTTPTransport(
            proxy=httpx.Proxy(
                self.proxy_url,
                headers=normalized_proxy_headers or None,
            ),
            verify=verify_path,
            trust_env=False,
            http1=True,
            http2=allow_http2,
            limits=limits,
        )
        client = httpx.Client(
            transport=explicit_transport,
            trust_env=False,
            cookies=CookieJar(policy=RejectAllCookies()),
            follow_redirects=False,
        )
        setattr(
            client,
            _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE,
            explicit_transport,
        )
        _register_broker_owned_httpx_client(client, explicit_transport)
        return client

    @contextmanager
    def _lease_direct_client(self):
        with self._direct_client_lock:
            if self._direct_client is None:
                self._direct_client = self._new_client(
                    retain_direct_connections=True,
                )
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

    @contextmanager
    def _lease_direct_request_slot(self, timeout_seconds: float):
        started = time.monotonic()
        if not self._direct_request_slot.acquire(timeout=timeout_seconds):
            raise TimeoutError("direct provider concurrency slot timed out")
        try:
            yield max(
                0.001,
                timeout_seconds - (time.monotonic() - started),
            )
        finally:
            self._direct_request_slot.release()

    @contextmanager
    def _lease_proxy_client(
        self,
        *,
        connection_scope: str,
        proxy_headers: Mapping[str, str],
        allow_http2: bool,
    ):
        with self._proxy_client_lock:
            current = self._proxy_clients.get(connection_scope)
            if current is None:
                client = self._new_client(
                    proxy_headers=proxy_headers,
                    allow_http2=allow_http2,
                    retain_assigned_proxy_connections=True,
                )
                self._proxy_client_generation += 1
                generation = self._proxy_client_generation
                self._proxy_clients[connection_scope] = (client, generation)
            else:
                client, generation = current
            self._proxy_client_leases[generation] = (
                self._proxy_client_leases.get(generation, 0) + 1
            )
        try:
            yield client, generation
        finally:
            close_client = None
            with self._proxy_client_lock:
                remaining = self._proxy_client_leases.get(generation, 0) - 1
                if remaining > 0:
                    self._proxy_client_leases[generation] = remaining
                else:
                    self._proxy_client_leases.pop(generation, None)
                    close_client = self._retired_proxy_clients.pop(
                        generation,
                        None,
                    )
            if close_client is not None:
                _close_client_nonfatal(close_client)

    def _retire_proxy_client(
        self,
        *,
        connection_scope: str,
        client: Any,
        generation: int,
    ) -> None:
        close_client = None
        with self._proxy_client_lock:
            current = self._proxy_clients.get(connection_scope)
            if (
                current is not None
                and current[0] is client
                and current[1] == generation
            ):
                self._proxy_clients.pop(connection_scope, None)
                if self._proxy_client_leases.get(generation, 0) > 0:
                    self._retired_proxy_clients[generation] = client
                else:
                    close_client = client
        if close_client is not None:
            _close_client_nonfatal(close_client)

    def release_connection_scope(self, connection_scope: str) -> None:
        if not _HASH_RE.fullmatch(str(connection_scope or "")):
            raise ProviderBrokerV2Error(
                "provider transport connection scope is invalid"
            )
        close_client = None
        with self._proxy_client_lock:
            current = self._proxy_clients.pop(connection_scope, None)
            if current is not None:
                client, generation = current
                if self._proxy_client_leases.get(generation, 0) > 0:
                    self._retired_proxy_clients[generation] = client
                else:
                    close_client = client
        if close_client is not None:
            _close_client_nonfatal(close_client)

    @contextmanager
    def _track_transport_attempt(self, route: str):
        if route not in _TRANSPORT_ROUTES:
            raise ProviderBrokerV2Error("provider transport route is invalid")
        with self._transport_health_lock:
            self._transport_attempt_counts[route]["started"] += 1
        try:
            yield
        except BaseException as exc:
            resource_errno, resource_kind = _local_resource_failure(exc)
            resource_errno = int(
                getattr(exc, "cleanup_errno", 0) or resource_errno or 0
            )
            resource_kind = str(
                getattr(exc, "cleanup_resource_kind", "")
                or resource_kind
                or ""
            )
            failure = {
                "route": route,
                "stage": str(
                    getattr(exc, "failure_stage", "provider_request")
                ),
                "failure_code": _failure_code(exc),
                "error_type": _safe_error_type(exc),
                "errno": resource_errno or _exception_errno(exc),
            }
            primary_error_type = str(
                getattr(exc, "primary_error_type", "") or ""
            )
            cleanup_error_type = str(
                getattr(exc, "cleanup_error_type", "") or ""
            )
            if _SAFE_ERROR_TYPE_RE.fullmatch(primary_error_type):
                failure["primary_error_type"] = primary_error_type
            if _SAFE_ERROR_TYPE_RE.fullmatch(cleanup_error_type):
                failure["cleanup_error_type"] = cleanup_error_type
            if resource_kind:
                failure["local_resource_kind"] = resource_kind
            with self._transport_health_lock:
                self._transport_attempt_counts[route]["failed"] += 1
                self._last_transport_failure = failure
            raise
        else:
            with self._transport_health_lock:
                self._transport_attempt_counts[route]["succeeded"] += 1

    def _record_transport_cleanup(
        self,
        route: str,
        *,
        client_closed: bool,
        transport_closed: bool,
    ) -> None:
        if route not in _TRANSPORT_ROUTES:
            raise ProviderBrokerV2Error("provider transport route is invalid")
        with self._transport_health_lock:
            counters = self._transport_cleanup_counts[route]
            counters["attempted"] += 1
            if not client_closed:
                counters["client_close_failed"] += 1
            if not transport_closed:
                counters["transport_close_failed"] += 1
            if transport_closed:
                # The client owns exactly the explicit transport recorded by
                # _new_client. Closing it independently proves the only pool
                # has been torn down even when client.close() itself raises.
                counters["succeeded"] += 1

    def health(self) -> Dict[str, Any]:
        with self._direct_client_lock:
            retired_direct_generations = set(self._retired_direct_clients)
            direct_active_scope_count = int(self._direct_client is not None)
            direct_retired_scope_count = len(self._retired_direct_clients)
            direct_active_lease_count = sum(
                count
                for generation, count in self._direct_client_leases.items()
                if generation not in retired_direct_generations
            )
            direct_retired_lease_count = sum(
                count
                for generation, count in self._direct_client_leases.items()
                if generation in retired_direct_generations
            )
        with self._proxy_client_lock:
            retired_proxy_generations = set(self._retired_proxy_clients)
            assigned_active_scope_count = len(self._proxy_clients)
            assigned_retired_scope_count = len(self._retired_proxy_clients)
            assigned_active_lease_count = sum(
                count
                for generation, count in self._proxy_client_leases.items()
                if generation not in retired_proxy_generations
            )
            assigned_retired_lease_count = sum(
                count
                for generation, count in self._proxy_client_leases.items()
                if generation in retired_proxy_generations
            )
        with self._transport_health_lock:
            counters = {
                route: dict(self._transport_attempt_counts[route])
                for route in _TRANSPORT_ROUTES
            }
            cleanup_counters = {
                route: dict(self._transport_cleanup_counts[route])
                for route in _TRANSPORT_ROUTES
            }
            last_failure = dict(self._last_transport_failure or {})
        result = {
            "schema_version": PROVIDER_TRANSPORT_HEALTH_SCHEMA_VERSION,
            "reuse_direct_connections": self.reuse_direct_connections,
            "reuse_assigned_proxy_connections": (
                self.reuse_upstream_proxy_connections
            ),
            "direct_request_slot_active_count": int(
                self._direct_request_slot.locked()
            ),
            "direct_active_scope_count": direct_active_scope_count,
            "direct_retired_scope_count": direct_retired_scope_count,
            "direct_active_lease_count": direct_active_lease_count,
            "direct_retired_lease_count": direct_retired_lease_count,
            "assigned_active_scope_count": assigned_active_scope_count,
            "assigned_retired_scope_count": assigned_retired_scope_count,
            "assigned_active_lease_count": assigned_active_lease_count,
            "assigned_retired_lease_count": assigned_retired_lease_count,
            "request_counters": counters,
            "cleanup_counters": cleanup_counters,
        }
        if last_failure:
            result["last_failure"] = last_failure
        return result

    def close(self) -> None:
        with self._direct_client_lock:
            clients = list(self._retired_direct_clients.values())
            if self._direct_client is not None:
                clients.append(self._direct_client)
            self._direct_client = None
            self._direct_client_leases.clear()
            self._retired_direct_clients.clear()
        with self._proxy_client_lock:
            clients.extend(
                client for client, _generation in self._proxy_clients.values()
            )
            clients.extend(self._retired_proxy_clients.values())
            self._proxy_clients.clear()
            self._proxy_client_leases.clear()
            self._retired_proxy_clients.clear()
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
        force_close_network_stream: bool = False,
    ) -> Dict[str, Any]:
        with _broker_owned_httpx_send_scope(
            client,
            method,
            url,
            headers=dict(headers),
            content=body,
            timeout=timeout_seconds,
        ) as response:
            _make_response_cleanup_nonfatal(response)
            primary_error = None  # type: Optional[BaseException]
            try:
                # TLS is authenticated before response headers are available.
                # Capture its evidence now because a peer may close the stream
                # immediately after sending a complete bounded response body.
                tls_peer_chain_hash, tls_protocol = _extract_tls_metadata(response)
                chunks = []
                byte_count = 0
                content_encoding = str(
                    response.headers.get("content-encoding") or ""
                ).strip().lower()
                if content_encoding == "gzip":
                    decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
                    try:
                        for chunk in response.iter_raw():
                            byte_count = _append_bounded_gzip_bytes(
                                decoder=decoder,
                                compressed=chunk,
                                chunks=chunks,
                                byte_count=byte_count,
                                max_response_bytes=max_response_bytes,
                            )
                    except Exception:
                        response_body = b"".join(chunks)
                        if not (
                            allow_authenticated_complete_body_eof
                            and _authenticated_gzip_body_is_complete_after_stream_error(
                                method=method,
                                response=response,
                                byte_count=byte_count,
                                body=response_body,
                                decoder=decoder,
                            )
                        ):
                            raise
                    else:
                        remaining = max_response_bytes - byte_count
                        decoded_tail = decoder.flush(remaining + 1)
                        if len(decoded_tail) > remaining:
                            raise ProviderBrokerV2Error(
                                "provider response exceeds size limit"
                            )
                        if decoded_tail:
                            chunks.append(decoded_tail)
                            byte_count += len(decoded_tail)
                        if (
                            not decoder.eof
                            or decoder.unused_data
                            or decoder.unconsumed_tail
                        ):
                            raise ProviderBrokerV2Error(
                                "provider gzip response is incomplete"
                            )
                        response_body = b"".join(chunks)
                else:
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
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                if force_close_network_stream:
                    # This executes before Response.__exit__ and before the
                    # request-scoped client/pool close. It is the hard lifetime
                    # boundary for both successful and failed body reads.
                    try:
                        _force_close_response_network_stream(
                            response,
                            required=True,
                        )
                    except BaseException as cleanup_error:
                        raise ProviderTransportCleanupError(
                            stage="response_stream_cleanup",
                            primary_error=primary_error,
                            cleanup_error=cleanup_error,
                        ) from (primary_error or cleanup_error)

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
        allow_http2: bool = True,
        connection_scope: str = "",
    ) -> Dict[str, Any]:
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1
            <= max_response_bytes
            <= self.response_body_ceiling_bytes
        ):
            raise ProviderBrokerV2Error("provider response limit is invalid")
        if not isinstance(allow_http2, bool):
            raise ProviderBrokerV2Error("provider HTTP/2 policy is invalid")
        normalized_connection_scope = str(connection_scope or "")
        if normalized_connection_scope and not _HASH_RE.fullmatch(
            normalized_connection_scope
        ):
            raise ProviderBrokerV2Error(
                "provider transport connection scope is invalid"
            )

        proxy_headers = None
        if upstream_proxy_url:
            proxy_headers = {
                "X-Leadpoet-Upstream-Proxy-B64": base64.b64encode(
                    upstream_proxy_url.encode("utf-8")
                ).decode("ascii")
            }
        timeout_seconds = max(0.001, timeout_ms / 1000.0)
        if proxy_headers is not None and self.reuse_upstream_proxy_connections:
            if not normalized_connection_scope:
                raise ProviderBrokerV2Error(
                    "provider upstream proxy connection scope is required"
                )
            if allow_http2:
                with self._track_transport_attempt("assigned_proxy"):
                    with self._lease_proxy_client(
                        connection_scope=normalized_connection_scope,
                        proxy_headers=proxy_headers,
                        allow_http2=True,
                    ) as (client, generation):
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
                            self._retire_proxy_client(
                                connection_scope=normalized_connection_scope,
                                client=client,
                                generation=generation,
                            )
                            raise
        if proxy_headers is None and self.reuse_direct_connections and allow_http2:
            with self._track_transport_attempt("direct"):
                with self._lease_direct_request_slot(timeout_seconds) as remaining:
                    with self._lease_direct_client() as (client, generation):
                        try:
                            return self._execute_with_client(
                                client,
                                method=method,
                                url=url,
                                headers=headers,
                                body=body,
                                timeout_seconds=remaining,
                                max_response_bytes=max_response_bytes,
                                allow_authenticated_complete_body_eof=(
                                    self.allow_authenticated_complete_body_eof
                                ),
                            )
                        except BaseException:
                            # The measured caller owns retry accounting. Retire
                            # this serialized generation so the next attempt opens
                            # one fresh tunnel before admitting another request.
                            self._retire_direct_client(client, generation)
                            raise

        # Direct coordinator requests are request-scoped by default. A relay
        # tunnel can expire between otherwise independent jobs or epochs; one
        # stale pooled generation must not make their measured retries share a
        # failure fate. Framed artifact-only callers may explicitly opt into
        # reuse after proving their terminal handshake contract.
        def execute_request_scoped(
            *,
            route: str,
            request_timeout_seconds: float = timeout_seconds,
        ) -> Dict[str, Any]:
            client = self._new_client(
                proxy_headers=proxy_headers,
                allow_http2=allow_http2,
            )
            primary_error = None  # type: Optional[BaseException]
            try:
                return self._execute_with_client(
                    client,
                    method=method,
                    url=url,
                    headers=headers,
                    body=body,
                    timeout_seconds=request_timeout_seconds,
                    max_response_bytes=max_response_bytes,
                    allow_authenticated_complete_body_eof=(
                        self.allow_authenticated_complete_body_eof
                    ),
                    force_close_network_stream=True,
                )
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                client_closed, transport_closed, cleanup_error = (
                    _close_client_transports(client)
                )
                self._record_transport_cleanup(
                    route,
                    client_closed=client_closed,
                    transport_closed=transport_closed,
                )
                if not transport_closed:
                    normalized_cleanup_error = cleanup_error or ProviderBrokerV2Error(
                        "explicit provider transport cleanup failed"
                    )
                    raise ProviderTransportCleanupError(
                        stage="client_transport_cleanup",
                        primary_error=primary_error,
                        cleanup_error=normalized_cleanup_error,
                    ) from (primary_error or normalized_cleanup_error)

        if proxy_headers is None:
            with self._track_transport_attempt("direct"):
                with self._lease_direct_request_slot(timeout_seconds) as remaining:
                    return execute_request_scoped(
                        route="direct",
                        request_timeout_seconds=remaining
                    )
        with self._track_transport_attempt("assigned_proxy"):
            return execute_request_scoped(route="assigned_proxy")


class ProviderBrokerV2:
    """Execute each logical provider attempt at most once and record a terminal."""

    def __init__(
        self,
        *,
        credential_ref_hashes: Mapping[str, str],
        retry_policy_hashes: Mapping[str, str],
        routes: Optional[Mapping[str, ProviderRouteV2]] = None,
        transport: Optional[Callable[..., Mapping[str, Any]]] = None,
        artifact_sink: Optional[Callable[..., Mapping[str, Any]]] = None,
        job_credential_slot_ref_hashes: Optional[Mapping[str, str]] = None,
        clock: Callable[[], str] = _timestamp,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.routes = dict(
            routes if routes is not None else provider_routes_for_execution_config()
        )
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
        self._rolled_back_transport_failure_diagnostics = (
            {}
        )  # type: Dict[str, Dict[str, Any]]
        self._record_keys_by_job = {}  # type: Dict[str, set[Tuple[str, int]]]
        self._inflight = {}  # type: Dict[Tuple[str, int], Tuple[str, threading.Event, object]]
        self._released_terminal_count = 0
        self._expired_terminal_count = 0
        self._provider_terminal_counts = {
            provider_id: {
                route: {
                    terminal_status: 0
                    for terminal_status in _PROVIDER_TERMINAL_STATUSES
                }
                for route in _TRANSPORT_ROUTES
            }
            for provider_id in sorted(BUILTIN_PROVIDER_ROUTES)
        }
        self._provider_2xx_success_counts = {
            provider_id: {route: 0 for route in _TRANSPORT_ROUTES}
            for provider_id in sorted(BUILTIN_PROVIDER_ROUTES)
        }
        self._chain_weight_observation_success_count = 0
        self._lock = threading.Lock()
        self._transaction_state = threading.local()

    def _record_committed_terminal_health_locked(
        self,
        record: Mapping[str, Any],
    ) -> None:
        provider_id = str(record.get("provider_id") or "")
        route = str(record.get("egress_route") or "")
        terminal_status = str(record.get("terminal_status") or "")
        purpose = str(record.get("purpose") or "")
        http_status = int(record.get("http_status") or 0)
        if provider_id not in self._provider_terminal_counts:
            # Dynamic SOURCE_ADD identities are intentionally excluded from
            # this fixed-shape, non-secret production health projection.
            return
        if (
            route not in _TRANSPORT_ROUTES
            or terminal_status not in _PROVIDER_TERMINAL_STATUSES
        ):
            raise ProviderBrokerV2Error(
                "committed provider terminal health identity is invalid"
            )
        self._provider_terminal_counts[provider_id][route][terminal_status] += 1
        healthy_success = (
            terminal_status == "authenticated_response"
            and 200 <= http_status < 300
        )
        if healthy_success:
            self._provider_2xx_success_counts[provider_id][route] += 1
        if (
            provider_id == "supabase"
            and route == "direct"
            and healthy_success
            and purpose == _CHAIN_WEIGHT_OBSERVATION_PURPOSE
        ):
            self._chain_weight_observation_success_count += 1

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
                            self._record_committed_terminal_health_locked(record)
                            self._record_keys_by_job.setdefault(
                                str(record["job_id"]),
                                set(),
                            ).add(key)
                        else:
                            diagnostic = pending["record"].get(
                                "transport_failure_diagnostic"
                            )
                            attempt = pending["record"].get("result", {}).get(
                                "transport_attempt"
                            )
                            attempt_hash = (
                                str(attempt.get("attempt_hash") or "")
                                if isinstance(attempt, Mapping)
                                else ""
                            )
                            if (
                                isinstance(diagnostic, Mapping)
                                and _HASH_RE.fullmatch(attempt_hash)
                            ):
                                if (
                                    len(
                                        self._rolled_back_transport_failure_diagnostics
                                    )
                                    >= MAX_DEDUPLICATION_RECORDS
                                ):
                                    oldest = next(
                                        iter(
                                            self._rolled_back_transport_failure_diagnostics
                                        )
                                    )
                                    self._rolled_back_transport_failure_diagnostics.pop(
                                        oldest,
                                        None,
                                    )
                                self._rolled_back_transport_failure_diagnostics[
                                    attempt_hash
                                ] = dict(diagnostic)
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

    def reseal_transport_failure_diagnostic(
        self,
        *,
        prior_result: Mapping[str, Any],
        outer_error: BaseException,
    ) -> Dict[str, Any] | None:
        """Reseal safe cleanup provenance after a surrounding rollback.

        The broker result exposes only artifact commitments.  The validated
        projection stays coordinator-local and is recovered by the immutable
        attempt hash when semantics persistence rolls the first envelope back.
        """

        if not isinstance(prior_result, Mapping):
            raise ProviderBrokerV2Error(
                "prior provider transport result is invalid"
            )
        if str(prior_result.get("terminal_status") or "") != "transport_failure":
            return None
        attempt = prior_result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise ProviderBrokerV2Error(
                "prior provider transport attempt is invalid"
            )
        try:
            validate_transport_attempt(attempt)
        except Exception as exc:
            raise ProviderBrokerV2Error(
                "prior provider transport attempt is invalid"
            ) from exc
        attempt_hash = str(attempt.get("attempt_hash") or "")
        key = (
            str(attempt.get("logical_operation_id") or ""),
            int(attempt.get("attempt_number")),
        )
        with self._lock:
            diagnostic = self._rolled_back_transport_failure_diagnostics.get(
                attempt_hash
            )
            if diagnostic is not None:
                diagnostic = dict(diagnostic)
            else:
                record = self._records.get(key)
                record_attempt = (
                    record.get("result", {}).get("transport_attempt")
                    if isinstance(record, Mapping)
                    else None
                )
                if (
                    isinstance(record_attempt, Mapping)
                    and str(record_attempt.get("attempt_hash") or "")
                    == attempt_hash
                    and isinstance(
                        record.get("transport_failure_diagnostic"),
                        Mapping,
                    )
                ):
                    diagnostic = dict(
                        record["transport_failure_diagnostic"]
                    )
        if diagnostic is None:
            raise ProviderBrokerV2Error(
                "provider transport failure diagnostic projection is unavailable"
            )
        validated = validate_provider_transport_failure_diagnostic(diagnostic)
        if (
            validated["provider"] != str(attempt.get("provider_id") or "")
            or validated["request_hash"]
            != str(attempt.get("request_hash") or "")
            or validated["attempt_number"] != attempt.get("attempt_number")
        ):
            raise ProviderBrokerV2Error(
                "provider transport failure diagnostic binding differs"
            )
        resealed_document = validate_provider_transport_failure_diagnostic(
            {
                **validated,
                "outer_error_type": _safe_error_type(outer_error),
            }
        )
        payload = canonical_json(resealed_document).encode("utf-8")
        descriptor = dict(
            self._artifact_sink(
                payload,
                job_id=str(attempt.get("job_id") or ""),
                purpose=str(attempt.get("purpose") or ""),
                artifact_kind="provider_transport_failure_diagnostic",
            )
        )
        expected_plaintext_hash = sha256_bytes(payload)
        if descriptor.get("plaintext_hash") != expected_plaintext_hash:
            raise ProviderBrokerV2Error(
                "provider transport failure diagnostic plaintext differs"
            )
        descriptor_hashes = {
            str(descriptor.get(field) or "")
            for field in (
                "artifact_id",
                "plaintext_hash",
                "ciphertext_hash",
                "encryption_context_hash",
            )
            if descriptor.get(field)
        }
        if (
            not _HASH_RE.fullmatch(str(descriptor.get("artifact_id") or ""))
            or any(not _HASH_RE.fullmatch(item) for item in descriptor_hashes)
        ):
            raise ProviderBrokerV2Error(
                "provider transport failure diagnostic descriptor is invalid"
            )
        # Retain this bounded, commitment-keyed projection until its oldest
        # entry is evicted. The artifact sink returns before the surrounding
        # semantics artifact/terminal transaction commits; deleting here would
        # make a second outer commit failure permanently erase the only safe
        # provenance needed for the next retry.
        return descriptor

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
            provider_terminal_counts = {
                provider_id: {
                    route: dict(statuses)
                    for route, statuses in routes.items()
                }
                for provider_id, routes in self._provider_terminal_counts.items()
            }
            provider_2xx_success_counts = {
                provider_id: dict(routes)
                for provider_id, routes in self._provider_2xx_success_counts.items()
            }
            chain_weight_observation_success_count = (
                self._chain_weight_observation_success_count
            )
        missing = sorted(expected_slots - configured_slots)
        result = {
            "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
            "status": "ready" if not missing else "provisioning",
            "credential_slots": sorted(configured_slots),
            "missing_credential_slots": missing,
            "inflight_count": inflight_count,
            "terminal_count": terminal_count,
            "released_terminal_count": released_terminal_count,
            "expired_terminal_count": expired_terminal_count,
            "job_credential_lease_count": job_lease_count,
            "registry_hash": provider_registry_hash(self.routes),
            "job_credential_slot_ref_hashes": dict(
                sorted(self.job_credential_slot_ref_hashes.items())
            ),
            "provider_terminal_counts": provider_terminal_counts,
            "provider_2xx_success_counts": provider_2xx_success_counts,
            "chain_weight_observation_success_count": (
                chain_weight_observation_success_count
            ),
        }
        transport_health = getattr(self._transport, "health", None)
        if callable(transport_health):
            result["transport"] = dict(transport_health())
        return result

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
        proxy_connection_scope = ""
        with self._lock:
            credential_keys = [
                key for key in self._job_credentials if key[0] == normalized_job_id
            ]
            proxy_lease = self._job_credentials.get(
                (normalized_job_id, EGRESS_PROXY_CREDENTIAL_SLOT)
            )
            if proxy_lease is not None:
                proxy_connection_scope = self._proxy_connection_scope(
                    normalized_job_id,
                    str(proxy_lease["credential_ref_hash"]),
                )
            for key in credential_keys:
                del self._job_credentials[key]
            terminal_keys = tuple(
                self._record_keys_by_job.pop(normalized_job_id, ())
            )
            for key in terminal_keys:
                self._records.pop(key, None)
            self._released_terminal_count += len(terminal_keys)
        release_connection_scope = getattr(
            self._transport,
            "release_connection_scope",
            None,
        )
        if proxy_connection_scope and callable(release_connection_scope):
            release_connection_scope(proxy_connection_scope)
        return {
            "status": "released",
            "job_id": normalized_job_id,
            "released_slot_count": len(credential_keys),
            "released_terminal_count": len(terminal_keys),
        }

    @staticmethod
    def _proxy_connection_scope(job_id: str, credential_ref_hash: str) -> str:
        return sha256_json(
            {
                "schema_version": "leadpoet.provider_connection_scope.v2",
                "job_id": str(job_id),
                "egress_proxy_ref_hash": str(credential_ref_hash),
            }
        )

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
            proxy_lease = (
                self._job_credentials.get(
                    (job_id, EGRESS_PROXY_CREDENTIAL_SLOT)
                )
                if route.egress_policy == EGRESS_POLICY_JOB_PROXY_ALLOWED
                else None
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
        if route.egress_policy not in _EGRESS_POLICIES:
            raise ProviderBrokerV2Error(
                "provider route egress policy is invalid"
            )
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
        accepted_fields = {
            frozenset(required),
            frozenset(required | {"dynamic_route"}),
            frozenset(required | {"max_response_bytes", "artifact_mode"}),
            frozenset(
                required
                | {"dynamic_route", "max_response_bytes", "artifact_mode"}
            ),
        }
        if request_fields not in accepted_fields:
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
        if dynamic_route is not None:
            static_headers = dynamic_route.get("request_headers") or {}
            for static_name, static_value in static_headers.items():
                outbound_headers = {
                    name: value
                    for name, value in outbound_headers.items()
                    if name.lower() != str(static_name).lower()
                }
                outbound_headers[str(static_name)] = str(static_value)
        # Bind the default response framing at the measured transport boundary.
        # Built-in routes may select the stricter checksum-delimited gzip
        # profile below; callers and dynamic routes cannot select an encoding.
        for static_name, static_value in MEASURED_TRANSPORT_REQUEST_HEADERS:
            outbound_headers = {
                name: value
                for name, value in outbound_headers.items()
                if name.lower() != static_name.lower()
            }
            outbound_headers[static_name] = static_value
        # A built-in route may select a stricter, statically measured wire
        # profile after the global default. Dynamic routes cannot override it.
        # A compressed response is recoverable only after its complete gzip
        # member and bounded JSON document have both verified.
        for static_name, static_value in route.request_headers:
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
            proxy_lease = (
                self._job_credentials.get(
                    (
                        str(request["job_id"] or ""),
                        EGRESS_PROXY_CREDENTIAL_SLOT,
                    )
                )
                if route.egress_policy == EGRESS_POLICY_JOB_PROXY_ALLOWED
                else None
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
        transport_failure_diagnostic = None  # type: Optional[Dict[str, Any]]
        transport_failure_error = None  # type: Optional[BaseException]
        try:
            transport_kwargs = {
                "method": method,
                "url": outbound_url,
                "headers": outbound_headers,
                "body": body,
                "timeout_ms": timeout_ms,
            }
            if not route.allow_http2:
                transport_kwargs["allow_http2"] = False
            if egress_proxy_url is not None:
                transport_kwargs["upstream_proxy_url"] = egress_proxy_url
                transport_kwargs["connection_scope"] = self._proxy_connection_scope(
                    str(request["job_id"] or ""),
                    egress_proxy_ref_hash,
                )
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
                # One response record represents one physical provider
                # transport.  Composite callers must provide their own
                # signed aggregate count; the Lab must not infer it.
                "call_count": 1,
                "headers": _nonsecret_headers(response.get("headers", {})),
                "body_b64": base64.b64encode(response_body).decode("ascii"),
                "encrypted_request_artifact_id": request_artifact_id,
                "encrypted_artifact_id": artifact_id,
            }
        except Exception as exc:
            if failure_stage == "provider_transport":
                transport_failure_error = exc
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
                "call_count": 1,
                "failure_code": terminal_kwargs["failure_code"],
                "failure_stage": failure_stage,
                "failure_error_type": _safe_error_type(exc),
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
        if transport_failure_error is not None:
            try:
                transport_failure_diagnostic = (
                    _provider_transport_failure_diagnostic(
                        provider=provider_id,
                        request_hash=str(attempt["request_hash"]),
                        attempt_number=attempt_number,
                        exc=transport_failure_error,
                    )
                )
                diagnostic_bytes = canonical_json(
                    transport_failure_diagnostic
                ).encode("utf-8")
                diagnostic_artifact = dict(
                    self._artifact_sink(
                        diagnostic_bytes,
                        job_id=str(request["job_id"] or ""),
                        purpose=str(request["purpose"] or ""),
                        artifact_kind="provider_transport_failure_diagnostic",
                    )
                )
                if diagnostic_artifact.get("plaintext_hash") != sha256_bytes(
                    diagnostic_bytes
                ):
                    raise ProviderBrokerV2Error(
                        "provider transport failure diagnostic plaintext differs"
                    )
                diagnostic_hashes = {
                    str(diagnostic_artifact.get(field) or "")
                    for field in (
                        "artifact_id",
                        "plaintext_hash",
                        "ciphertext_hash",
                        "encryption_context_hash",
                    )
                    if diagnostic_artifact.get(field)
                }
                if (
                    not _HASH_RE.fullmatch(
                        str(diagnostic_artifact.get("artifact_id") or "")
                    )
                    or any(
                        not _HASH_RE.fullmatch(item)
                        for item in diagnostic_hashes
                    )
                ):
                    raise ProviderBrokerV2Error(
                        "provider transport failure diagnostic descriptor is invalid"
                    )
                evidence_artifact_hashes.update(diagnostic_hashes)
            except Exception:
                self._abandon_inflight(
                    deduplication_key,
                    request_fingerprint,
                )
                raise
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
                "provider_id": provider_id,
                "purpose": str(request["purpose"] or ""),
                "egress_route": (
                    "assigned_proxy"
                    if egress_proxy_url is not None
                    else "direct"
                ),
                "terminal_status": str(result["terminal_status"]),
                "http_status": int(result.get("http_status") or 0),
                "request_fingerprint": request_fingerprint,
                "result": dict(result),
                "transport_failure_diagnostic": (
                    dict(transport_failure_diagnostic)
                    if transport_failure_diagnostic is not None
                    else None
                ),
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
                self._record_committed_terminal_health_locked(record)
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
