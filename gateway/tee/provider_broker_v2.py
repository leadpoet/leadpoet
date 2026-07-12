"""Coordinator-owned HTTPS provider broker with terminal V2 transport records."""

from __future__ import annotations

import base64
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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from gateway.tee.egress_policy import normalize_destination
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
MAX_RESPONSE_BODY_BYTES = 64 * 1024 * 1024
MAX_DEDUPLICATION_RECORDS = 10000
MAX_JOB_CREDENTIAL_LEASES = 1024
EGRESS_PROXY_CREDENTIAL_SLOT = "egress_proxy"
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
    "coingecko": ProviderRouteV2(
        provider_id="coingecko",
        hosts=("api.coingecko.com",),
        path_prefixes=("/api/v3/simple/price",),
        allowed_methods=("GET",),
    ),
    "wayback": ProviderRouteV2(
        provider_id="wayback",
        hosts=("archive.org", "web.archive.org"),
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
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ProviderBrokerV2Error("worker egress proxy port is invalid") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or port != 443
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderBrokerV2Error(
            "worker egress proxy must be an HTTPS proxy on port 443"
        )
    normalize_destination(parsed.hostname, port)
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
        certificate = ssl_object.getpeercert(binary_form=True)
        protocol = ssl_object.version()
    except Exception as exc:
        raise ProviderBrokerV2Error("authenticated response lacks TLS evidence") from exc
    if not certificate or not protocol:
        raise ProviderBrokerV2Error("authenticated response TLS evidence is empty")
    return sha256_bytes(bytes(certificate)), str(protocol)


class HTTPXProviderTransport:
    """TLS and hostname verification run inside the coordinator enclave."""

    def __init__(
        self,
        *,
        proxy_url: str = "http://127.0.0.1:18080",
        ca_bundle: Optional[str] = None,
    ) -> None:
        self.proxy_url = proxy_url
        self.ca_bundle = ca_bundle

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_ms: int,
        upstream_proxy_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        import certifi
        import httpx

        verify_path = self.ca_bundle or certifi.where()
        proxy_headers = None
        if upstream_proxy_url:
            proxy_headers = {
                "X-Leadpoet-Upstream-Proxy-B64": base64.b64encode(
                    upstream_proxy_url.encode("utf-8")
                ).decode("ascii")
            }
        with httpx.Client(
            proxy=httpx.Proxy(self.proxy_url, headers=proxy_headers),
            verify=verify_path,
            trust_env=False,
            timeout=max(0.001, timeout_ms / 1000.0),
            follow_redirects=False,
        ) as client:
            response = client.request(
                method,
                url,
                headers=dict(headers),
                content=body,
            )
            response_body = bytes(response.content)
            if len(response_body) > MAX_RESPONSE_BODY_BYTES:
                raise ProviderBrokerV2Error("provider response exceeds size limit")
            tls_peer_chain_hash, tls_protocol = _extract_tls_metadata(response)
            return {
                "http_status": int(response.status_code),
                "headers": _nonsecret_headers(response.headers),
                "body": response_body,
                "tls_peer_chain_hash": tls_peer_chain_hash,
                "tls_protocol": tls_protocol,
            }


class ProviderBrokerV2:
    """Execute each logical provider attempt at most once and record a terminal."""

    def __init__(
        self,
        *,
        credential_ref_hashes: Mapping[str, str],
        retry_policy_hashes: Mapping[str, str],
        routes: Mapping[str, ProviderRouteV2] = BUILTIN_PROVIDER_ROUTES,
        transport: Callable[..., Mapping[str, Any]] = HTTPXProviderTransport(),
        artifact_sink: Optional[Callable[..., Mapping[str, Any]]] = None,
        job_credential_slot_ref_hashes: Optional[Mapping[str, str]] = None,
        clock: Callable[[], str] = _timestamp,
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
        self._transport = transport
        if artifact_sink is None:
            raise ProviderBrokerV2Error(
                "provider broker requires encrypted artifact persistence"
            )
        self._artifact_sink = artifact_sink
        self._clock = clock
        self._credentials = {}  # type: Dict[str, str]
        self._job_credentials = {}  # type: Dict[Tuple[str, str], Dict[str, str]]
        self._records = {}  # type: Dict[Tuple[str, int], Dict[str, Any]]
        self._inflight = {}  # type: Dict[Tuple[str, int], Tuple[str, threading.Event]]
        self._lock = threading.Lock()

    def health(self) -> Dict[str, Any]:
        expected_slots = set(expected_provider_credential_slots())
        with self._lock:
            configured_slots = set(self._credentials)
            inflight_count = len(self._inflight)
            terminal_count = len(self._records)
            job_lease_count = len(self._job_credentials)
        missing = sorted(expected_slots - configured_slots)
        return {
            "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
            "status": "ready" if not missing else "provisioning",
            "credential_slots": sorted(configured_slots),
            "missing_credential_slots": missing,
            "inflight_count": inflight_count,
            "terminal_count": terminal_count,
            "job_credential_lease_count": job_lease_count,
            "registry_hash": provider_registry_hash(),
            "job_credential_slot_ref_hashes": dict(
                sorted(self.job_credential_slot_ref_hashes.items())
            ),
        }

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
            keys = [
                key for key in self._job_credentials if key[0] == normalized_job_id
            ]
            for key in keys:
                del self._job_credentials[key]
        return {
            "status": "released",
            "job_id": normalized_job_id,
            "released_slot_count": len(keys),
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
            inflight = self._inflight.get(deduplication_key)
            if inflight is not None:
                if inflight[0] != request_fingerprint:
                    raise ProviderBrokerV2Error(
                        "logical provider attempt is in flight with different request"
                    )
                wait_event = inflight[1]
                owns_attempt = False
            else:
                if len(self._records) >= MAX_DEDUPLICATION_RECORDS:
                    raise ProviderBrokerV2Error("provider terminal ledger is full")
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
                )
                owns_attempt = True
        if not owns_attempt:
            if not wait_event.wait(max(1.0, timeout_ms / 1000.0 + 5.0)):
                raise ProviderBrokerV2Error("duplicate provider attempt wait timed out")
            with self._lock:
                completed = self._records.get(deduplication_key)
                if completed is None:
                    raise ProviderBrokerV2Error("duplicate provider attempt did not terminate")
                return dict(completed["result"])

        outbound_headers = {str(k): str(v) for k, v in headers.items()}
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
            "url": urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
            ),
            "headers": _nonsecret_headers(headers),
            "body_b64": base64.b64encode(body).decode("ascii"),
            "timeout_ms": timeout_ms,
            "retry_policy_hash": retry_policy_hash,
            "egress_proxy_ref_hash": egress_proxy_ref_hash,
            "dynamic_route_hash": (
                str(dynamic_route["route_hash"])
                if dynamic_route is not None
                else ""
            ),
        }
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
            response = dict(self._transport(**transport_kwargs))
            response_body = bytes(response["body"])
            if len(response_body) > MAX_RESPONSE_BODY_BYTES:
                raise ProviderBrokerV2Error("provider response exceeds size limit")
            artifact = dict(
                self._artifact_sink(
                    response_body,
                    job_id=str(request["job_id"] or ""),
                    purpose=str(request["purpose"] or ""),
                    artifact_kind="provider_response",
                )
            )
            if artifact.get("plaintext_hash") != sha256_bytes(response_body):
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
            nonsecret_headers_hash=sha256_json(_nonsecret_headers(headers)),
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
            self._records[deduplication_key] = {
                "request_fingerprint": request_fingerprint,
                "result": dict(result),
            }
            inflight = self._inflight.pop(deduplication_key, None)
            if inflight is not None:
                inflight[1].set()
        return result


def credential_reference_hash(value: str) -> str:
    """Build the non-secret boot config commitment for one credential."""

    return _credential_hash(value)
