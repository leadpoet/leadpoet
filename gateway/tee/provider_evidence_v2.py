"""Coordinator-owned, signed provider evidence resolution for V2 probes.

The parent only relays mutually attested TLS ciphertext.  Cache misses use the
coordinator's existing HTTPS broker; cache hits are linked to the signed live
resolution that created the entry.  This preserves the loop's record-once,
replay-for-all behavior without trusting the host evidence proxy.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import re
import secrets
import threading
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlencode, urlsplit

from gateway.research_lab.provider_evidence_proxy import _response_is_recordable
from gateway.tee.provider_broker_v2 import (
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
)
from gateway.tee.source_add_runtime_v2 import (
    source_add_dynamic_retry_policy_hash,
    validate_source_add_runtime_route_v2,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint
from research_lab.probe_catalog import (
    ProviderProbeEndpoint,
    validate_probe_catalog,
    validate_probe_params,
)


REQUEST_SCHEMA_VERSION = "leadpoet.provider_probe_resolution_request.v2"
RECORD_SCHEMA_VERSION = "leadpoet.provider_evidence_resolution.v2"
MAX_CACHE_RECORDS = 10000
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PUBKEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")

_PROVIDER_ROUTES = {
    "exa": ("exa", "https://api.exa.ai"),
    "sd": ("scrapingdog", "https://api.scrapingdog.com"),
    "or": ("openrouter", "https://openrouter.ai"),
    "deepline": ("deepline", "https://code.deepline.com"),
}


class ProviderEvidenceV2Error(RuntimeError):
    """Provider evidence is malformed, unsigned, or outside measured routes."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signature_hex(value: Any) -> str:
    text = value.hex() if isinstance(value, bytes) else str(value or "").lower()
    if not _SIGNATURE_RE.fullmatch(text):
        raise ProviderEvidenceV2Error("provider evidence signature is invalid")
    return text


def _record_body(
    *,
    boot_identity_hash: str,
    request_hash: str,
    request_fingerprint: str,
    evidence: str,
    status: int,
    body_hash: str,
    encrypted_request_artifact_id: str,
    encrypted_response_artifact_id: str,
    transport_attempt_hash: str,
    source_record_hash: str,
    issued_at: str,
) -> Dict[str, Any]:
    if evidence not in {
        "recorded",
        "restored",
        "hit",
        "live_unrecorded",
        "replay_miss",
        "transport_failure",
    }:
        raise ProviderEvidenceV2Error("provider evidence outcome is invalid")
    if not isinstance(status, int) or not 0 <= status <= 599:
        raise ProviderEvidenceV2Error("provider evidence status is invalid")
    hashes = {
        "boot_identity_hash": boot_identity_hash,
        "request_hash": request_hash,
        "body_hash": body_hash,
    }
    for field, value in hashes.items():
        if not _HASH_RE.fullmatch(str(value or "")):
            raise ProviderEvidenceV2Error("%s is invalid" % field)
    for field, value in {
        "encrypted_request_artifact_id": encrypted_request_artifact_id,
        "encrypted_response_artifact_id": encrypted_response_artifact_id,
        "transport_attempt_hash": transport_attempt_hash,
        "source_record_hash": source_record_hash,
    }.items():
        if value and not _HASH_RE.fullmatch(str(value)):
            raise ProviderEvidenceV2Error("%s is invalid" % field)
    if not re.fullmatch(r"[0-9a-f]{64}", request_fingerprint):
        raise ProviderEvidenceV2Error("provider request fingerprint is invalid")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "coordinator_boot_identity_hash": str(boot_identity_hash),
        "request_hash": str(request_hash),
        "request_fingerprint": str(request_fingerprint),
        "evidence": evidence,
        "status": status,
        "body_hash": str(body_hash),
        "encrypted_request_artifact_id": str(encrypted_request_artifact_id),
        "encrypted_response_artifact_id": str(encrypted_response_artifact_id),
        "transport_attempt_hash": str(transport_attempt_hash),
        "source_record_hash": str(source_record_hash),
        "issued_at": str(issued_at),
    }


def create_signed_provider_evidence_record(
    *,
    body: Mapping[str, Any],
    coordinator_pubkey: str,
    sign_digest: Callable[[bytes], Any],
) -> Dict[str, Any]:
    normalized = _record_body(
        boot_identity_hash=body["coordinator_boot_identity_hash"],
        request_hash=body["request_hash"],
        request_fingerprint=body["request_fingerprint"],
        evidence=body["evidence"],
        status=body["status"],
        body_hash=body["body_hash"],
        encrypted_request_artifact_id=body["encrypted_request_artifact_id"],
        encrypted_response_artifact_id=body["encrypted_response_artifact_id"],
        transport_attempt_hash=body["transport_attempt_hash"],
        source_record_hash=body["source_record_hash"],
        issued_at=body["issued_at"],
    )
    pubkey = str(coordinator_pubkey or "").lower()
    if not _PUBKEY_RE.fullmatch(pubkey):
        raise ProviderEvidenceV2Error("coordinator public key is invalid")
    record_hash = sha256_json(normalized)
    signature = _signature_hex(
        sign_digest(bytes.fromhex(record_hash.split(":", 1)[1]))
    )
    return {
        **normalized,
        "record_hash": record_hash,
        "coordinator_pubkey": pubkey,
        "coordinator_signature": signature,
    }


def validate_signed_provider_evidence_record(
    record: Mapping[str, Any],
    *,
    boot_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "coordinator_boot_identity_hash",
        "request_hash",
        "request_fingerprint",
        "evidence",
        "status",
        "body_hash",
        "encrypted_request_artifact_id",
        "encrypted_response_artifact_id",
        "transport_attempt_hash",
        "source_record_hash",
        "issued_at",
        "record_hash",
        "coordinator_pubkey",
        "coordinator_signature",
    }
    if not isinstance(record, Mapping) or set(record) != fields:
        raise ProviderEvidenceV2Error("provider evidence record fields are invalid")
    body = _record_body(
        boot_identity_hash=record["coordinator_boot_identity_hash"],
        request_hash=record["request_hash"],
        request_fingerprint=record["request_fingerprint"],
        evidence=record["evidence"],
        status=record["status"],
        body_hash=record["body_hash"],
        encrypted_request_artifact_id=record["encrypted_request_artifact_id"],
        encrypted_response_artifact_id=record["encrypted_response_artifact_id"],
        transport_attempt_hash=record["transport_attempt_hash"],
        source_record_hash=record["source_record_hash"],
        issued_at=record["issued_at"],
    )
    expected_hash = sha256_json(body)
    if record["record_hash"] != expected_hash:
        raise ProviderEvidenceV2Error("provider evidence record hash mismatch")
    if (
        record["coordinator_boot_identity_hash"]
        != boot_identity.get("boot_identity_hash")
        or record["coordinator_pubkey"] != boot_identity.get("signing_pubkey")
    ):
        raise ProviderEvidenceV2Error("provider evidence coordinator identity differs")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(str(record["coordinator_pubkey"]))
        ).verify(
            bytes.fromhex(str(record["coordinator_signature"])),
            bytes.fromhex(expected_hash.split(":", 1)[1]),
        )
    except Exception as exc:
        raise ProviderEvidenceV2Error(
            "provider evidence coordinator signature is invalid"
        ) from exc
    return dict(record)


class ProviderEvidenceAuthorityV2:
    """Resolve typed probe requests and sign every cache/live terminal."""

    def __init__(
        self,
        *,
        broker: ProviderBrokerV2,
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        sign_digest: Callable[[bytes], Any],
        clock: Callable[[], str] = _timestamp,
        cache_store: Optional[Any] = None,
    ) -> None:
        self._broker = broker
        self._boot_identity_supplier = boot_identity_supplier
        self._sign_digest = sign_digest
        self._clock = clock
        self._cache_store = cache_store
        self._cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._attempts: Dict[tuple[str, str], int] = {}
        self._inflight: Dict[tuple[str, str], threading.Event] = {}
        self._cache_day = ""
        self._lock = threading.RLock()

    def resolve(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = self._request(request)
        endpoint = ProviderProbeEndpoint.from_mapping(normalized["endpoint"])
        method = endpoint.method
        dynamic_route = normalized.get("dynamic_route")
        if isinstance(dynamic_route, Mapping):
            provider_id = str(dynamic_route["provider_id"])
            base_url = str(dynamic_route["base_url"])
        else:
            provider_id, base_url = _PROVIDER_ROUTES[endpoint.provider_id]
        url = base_url + endpoint.path
        if normalized["query_params"]:
            url += "?" + urlencode(
                sorted(
                    (str(name), str(value))
                    for name, value in normalized["query_params"].items()
                )
            )
        body = (
            json.dumps(normalized["body_params"], sort_keys=True).encode("utf-8")
            if method == "POST"
            else b""
        )
        fingerprint = canonical_request_fingerprint(method, url, body or None)
        request_hash = sha256_json(normalized)
        utc_day = self._utc_day()
        cache_key = (utc_day, fingerprint)

        while True:
            with self._lock:
                self._ensure_cache_day(utc_day)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return self._cache_hit(
                        normalized=normalized,
                        request_hash=request_hash,
                        fingerprint=fingerprint,
                        cached=cached,
                    )
                event = self._inflight.get(cache_key)
                if event is None:
                    event = threading.Event()
                    self._inflight[cache_key] = event
                    break
            if not event.wait(normalized["timeout_seconds"] + 5):
                raise ProviderEvidenceV2Error(
                    "provider evidence single-flight wait timed out"
                )

        try:
            lookup_attempts = []
            lookup_artifacts = []
            if self._cache_store is not None:
                lookup = self._cache_store.load(
                    utc_day=utc_day,
                    request_fingerprint=fingerprint,
                    job_id=normalized["caller_job_id"],
                    purpose=normalized["purpose"],
                )
                lookup_attempts = list(lookup["transport_attempts"])
                lookup_artifacts = list(lookup["evidence_artifact_hashes"])
                if lookup["found"]:
                    cached = self._restored_cache_source(lookup)
                    with self._lock:
                        self._cache[cache_key] = dict(cached)
                    return self._cache_hit(
                        normalized=normalized,
                        request_hash=request_hash,
                        fingerprint=fingerprint,
                        cached=cached,
                        extra_transport_attempts=lookup_attempts,
                        extra_artifact_hashes=lookup_artifacts,
                    )
            if not normalized["live_enabled"]:
                return self._merge_terminal_evidence(
                    self._terminal(
                        request_hash=request_hash,
                        fingerprint=fingerprint,
                        evidence="replay_miss",
                        status=409,
                        body=b'{"error":"replay_miss"}',
                    ),
                    transport_attempts=lookup_attempts,
                    artifact_hashes=lookup_artifacts,
                )
            attempt_key = (normalized["caller_job_id"], fingerprint)
            with self._lock:
                attempt_number = self._attempts.get(attempt_key, 0)
                self._attempts[attempt_key] = attempt_number + 1
            logical_operation_id = "%s:probe:%s" % (
                normalized["caller_job_id"],
                fingerprint[:16],
            )
            broker_request = {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": logical_operation_id,
                    "job_id": normalized["caller_job_id"],
                    "purpose": normalized["purpose"],
                    "provider_id": provider_id,
                    "attempt_number": attempt_number,
                    "method": method,
                    "url": url,
                    "headers": {"Content-Type": "application/json"},
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "timeout_ms": normalized["timeout_seconds"] * 1000,
                    "retry_policy_hash": (
                        source_add_dynamic_retry_policy_hash(dynamic_route)
                        if isinstance(dynamic_route, Mapping)
                        else self._broker.retry_policy_hashes[provider_id]
                    ),
                }
            if isinstance(dynamic_route, Mapping):
                broker_request["dynamic_route"] = dict(dynamic_route)
            result = self._broker.execute(broker_request)
            broker_artifacts = list(result.get("evidence_artifact_hashes") or [])
            if any(not _HASH_RE.fullmatch(str(item or "")) for item in broker_artifacts):
                raise ProviderEvidenceV2Error(
                    "provider broker artifact commitment is invalid"
                )
            attempt = dict(result["transport_attempt"])
            if result.get("terminal_status") != "authenticated_response":
                return self._merge_terminal_evidence(
                    self._terminal(
                        request_hash=request_hash,
                        fingerprint=fingerprint,
                        evidence="transport_failure",
                        status=502,
                        body=b'{"error":"upstream unreachable"}',
                        transport_attempt=attempt,
                        encrypted_request_artifact_id=str(
                            result.get("encrypted_request_artifact_id") or ""
                        ),
                    ),
                    transport_attempts=lookup_attempts,
                    artifact_hashes=[*lookup_artifacts, *broker_artifacts],
                )
            status = int(result["http_status"])
            response_body = base64.b64decode(str(result["body_b64"]), validate=True)
            evidence = (
                "recorded"
                if _response_is_recordable(
                    endpoint.provider_id,
                    url,
                    status,
                    response_body,
                )
                else "live_unrecorded"
            )
            terminal = self._terminal(
                request_hash=request_hash,
                fingerprint=fingerprint,
                evidence=evidence,
                status=status,
                body=response_body,
                transport_attempt=attempt,
                encrypted_request_artifact_id=str(
                    result.get("encrypted_request_artifact_id") or ""
                ),
                encrypted_response_artifact_id=str(
                    result.get("encrypted_artifact_id") or ""
                ),
            )
            if evidence == "recorded":
                persistence_attempts = []
                persistence_artifacts = []
                if self._cache_store is not None:
                    persisted = self._cache_store.persist_recorded(
                        terminal,
                        utc_day=utc_day,
                        job_id=normalized["caller_job_id"],
                        purpose=normalized["purpose"],
                    )
                    persistence_attempts = list(persisted["transport_attempts"])
                    persistence_artifacts = list(
                        persisted["evidence_artifact_hashes"]
                    )
                with self._lock:
                    if len(self._cache) >= MAX_CACHE_RECORDS:
                        raise ProviderEvidenceV2Error(
                            "provider evidence cache capacity is full"
                        )
                    self._cache.setdefault(cache_key, dict(terminal))
                return self._merge_terminal_evidence(
                    terminal,
                    transport_attempts=[
                        *lookup_attempts,
                        *persistence_attempts,
                    ],
                    artifact_hashes=[
                        *lookup_artifacts,
                        *persistence_artifacts,
                        *broker_artifacts,
                    ],
                )
            return self._merge_terminal_evidence(
                terminal,
                transport_attempts=lookup_attempts,
                artifact_hashes=[*lookup_artifacts, *broker_artifacts],
            )
        finally:
            with self._lock:
                event = self._inflight.pop(cache_key, None)
                if event is not None:
                    event.set()

    def _utc_day(self) -> str:
        timestamp = str(self._clock() or "")
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            timestamp,
        ):
            raise ProviderEvidenceV2Error(
                "provider evidence clock is not canonical UTC"
            )
        return timestamp[:10]

    def _ensure_cache_day(self, utc_day: str) -> None:
        if self._cache_day == utc_day:
            return
        self._cache.clear()
        self._attempts.clear()
        self._cache_day = utc_day

    def _request(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "schema_version",
            "caller_job_id",
            "purpose",
            "endpoint",
            "upstream_base_url",
            "query_params",
            "body_params",
            "live_enabled",
            "timeout_seconds",
        }
        value_fields = frozenset(value) if isinstance(value, Mapping) else frozenset()
        if value_fields not in {
            frozenset(fields),
            frozenset(fields | {"dynamic_route"}),
        }:
            raise ProviderEvidenceV2Error("provider evidence request fields are invalid")
        if value.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise ProviderEvidenceV2Error("provider evidence request schema is invalid")
        endpoint = ProviderProbeEndpoint.from_mapping(value.get("endpoint") or {})
        errors = validate_probe_catalog([endpoint])
        if errors:
            raise ProviderEvidenceV2Error("provider probe endpoint is not measured")
        dynamic_route = None
        if endpoint.provider_id in _PROVIDER_ROUTES:
            if "dynamic_route" in value:
                raise ProviderEvidenceV2Error(
                    "builtin provider cannot use a dynamic route"
                )
            _provider_id, expected_base = _PROVIDER_ROUTES[endpoint.provider_id]
        else:
            try:
                dynamic_route = validate_source_add_runtime_route_v2(
                    value.get("dynamic_route") or {}
                )
            except Exception as exc:
                raise ProviderEvidenceV2Error(
                    "dynamic provider probe route is invalid"
                ) from exc
            if dynamic_route["provider_id"] != endpoint.provider_id:
                raise ProviderEvidenceV2Error(
                    "dynamic provider probe identity differs"
                )
            expected_base = str(dynamic_route["base_url"])
            expected_path = urlsplit(expected_base + endpoint.path).path or "/"
            if not any(
                item["method"] == endpoint.method
                and item["path"] == expected_path
                for item in dynamic_route["allowed_routes"]
            ):
                raise ProviderEvidenceV2Error(
                    "dynamic provider probe route is not measured"
                )
        if str(value.get("upstream_base_url") or "").rstrip("/") != expected_base:
            raise ProviderEvidenceV2Error("provider probe base URL differs")
        query_params = value.get("query_params")
        body_params = value.get("body_params")
        if not isinstance(query_params, Mapping) or not isinstance(body_params, Mapping):
            raise ProviderEvidenceV2Error("provider probe params are invalid")
        if set(query_params) & set(body_params):
            raise ProviderEvidenceV2Error("provider probe param locations overlap")
        normalized_params, param_errors = validate_probe_params(
            endpoint,
            {**dict(query_params), **dict(body_params)},
        )
        locations = {item.name: item.location for item in endpoint.params}
        expected_query = {
            name: item
            for name, item in normalized_params.items()
            if locations.get(name) == "query"
        }
        expected_body = {
            name: item
            for name, item in normalized_params.items()
            if locations.get(name) == "body"
        }
        if param_errors or dict(query_params) != expected_query or dict(body_params) != expected_body:
            raise ProviderEvidenceV2Error("provider probe params differ from catalog")
        timeout = value.get("timeout_seconds")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 5 <= timeout <= 300:
            raise ProviderEvidenceV2Error("provider probe timeout is invalid")
        caller_job_id = str(value.get("caller_job_id") or "")
        purpose = str(value.get("purpose") or "")
        if not caller_job_id or not purpose:
            raise ProviderEvidenceV2Error("provider probe execution scope is invalid")
        normalized = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "caller_job_id": caller_job_id,
            "purpose": purpose,
            "endpoint": endpoint.to_dict(),
            "upstream_base_url": expected_base,
            "query_params": expected_query,
            "body_params": expected_body,
            "live_enabled": bool(value.get("live_enabled")),
            "timeout_seconds": timeout,
        }
        if dynamic_route is not None:
            normalized["dynamic_route"] = dynamic_route
        return normalized

    def _terminal(
        self,
        *,
        request_hash: str,
        fingerprint: str,
        evidence: str,
        status: int,
        body: bytes,
        transport_attempt: Optional[Mapping[str, Any]] = None,
        encrypted_request_artifact_id: str = "",
        encrypted_response_artifact_id: str = "",
        source_record_hash: str = "",
    ) -> Dict[str, Any]:
        boot = dict(self._boot_identity_supplier())
        attempt_hash = str((transport_attempt or {}).get("attempt_hash") or "")
        body_doc = _record_body(
            boot_identity_hash=str(boot["boot_identity_hash"]),
            request_hash=request_hash,
            request_fingerprint=fingerprint,
            evidence=evidence,
            status=status,
            body_hash=sha256_bytes(body),
            encrypted_request_artifact_id=encrypted_request_artifact_id,
            encrypted_response_artifact_id=encrypted_response_artifact_id,
            transport_attempt_hash=attempt_hash,
            source_record_hash=source_record_hash,
            issued_at=self._clock(),
        )
        record = create_signed_provider_evidence_record(
            body=body_doc,
            coordinator_pubkey=str(boot["signing_pubkey"]),
            sign_digest=self._sign_digest,
        )
        artifacts = tuple(
            item
            for item in (
                encrypted_request_artifact_id,
                encrypted_response_artifact_id,
                record["record_hash"],
            )
            if item
        )
        return {
            "status": status,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "evidence": evidence,
            "transport_attempts": (
                [dict(transport_attempt)] if transport_attempt is not None else []
            ),
            "evidence_artifact_hashes": list(artifacts),
            "record": record,
            "source_record": None,
            "source_boot_identity": None,
            "coordinator_boot_identity": boot,
        }

    def _cache_hit(
        self,
        *,
        normalized: Mapping[str, Any],
        request_hash: str,
        fingerprint: str,
        cached: Mapping[str, Any],
        extra_transport_attempts: Optional[list[Mapping[str, Any]]] = None,
        extra_artifact_hashes: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        body = base64.b64decode(str(cached["body_b64"]), validate=True)
        terminal = self._terminal(
            request_hash=request_hash,
            fingerprint=fingerprint,
            evidence="hit",
            status=int(cached["status"]),
            body=body,
            source_record_hash=str(cached["record"]["record_hash"]),
        )
        terminal["source_record"] = dict(cached["record"])
        terminal["source_boot_identity"] = dict(
            cached["coordinator_boot_identity"]
        )
        terminal["transport_attempts"] = [
            *list(extra_transport_attempts or ()),
            *terminal["transport_attempts"],
        ]
        terminal["evidence_artifact_hashes"] = sorted(
            set(terminal["evidence_artifact_hashes"])
            | set(cached["evidence_artifact_hashes"])
            | set(extra_artifact_hashes or ())
            | {
                str(cached["coordinator_boot_identity"]["boot_identity_hash"])
            }
        )
        return terminal

    def _restored_cache_source(
        self,
        lookup: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload = lookup["payload"]
        body = base64.b64decode(str(payload["body_b64"]), validate=True)
        original = payload["source_record"]
        restored = self._terminal(
            request_hash=str(original["request_hash"]),
            fingerprint=str(payload["request_fingerprint"]),
            evidence="restored",
            status=int(payload["status"]),
            body=body,
            encrypted_request_artifact_id=str(
                original["encrypted_request_artifact_id"]
            ),
            encrypted_response_artifact_id=str(
                original["encrypted_response_artifact_id"]
            ),
            source_record_hash=str(lookup["cache_entry_hash"]),
        )
        restored["evidence_artifact_hashes"] = sorted(
            set(restored["evidence_artifact_hashes"])
            | set(lookup["evidence_artifact_hashes"])
            | set(payload["source_evidence_artifact_hashes"])
            | {
                str(payload["source_record"]["record_hash"]),
                str(payload["source_boot_identity"]["boot_identity_hash"]),
                str(lookup["cache_entry_hash"]),
                str(lookup["cache_artifact_id"]),
            }
        )
        return restored

    @staticmethod
    def _merge_terminal_evidence(
        terminal: Mapping[str, Any],
        *,
        transport_attempts: list[Mapping[str, Any]],
        artifact_hashes: list[str],
    ) -> Dict[str, Any]:
        result = dict(terminal)
        result["transport_attempts"] = [
            *[dict(item) for item in transport_attempts],
            *[dict(item) for item in terminal["transport_attempts"]],
        ]
        result["evidence_artifact_hashes"] = sorted(
            set(str(item) for item in terminal["evidence_artifact_hashes"])
            | set(str(item) for item in artifact_hashes)
        )
        return result
