"""Measured preservation layer for existing Research Lab provider semantics."""

from __future__ import annotations

import base64
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import re
import secrets
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import quote, urlsplit

from gateway.research_lab.provider_evidence_proxy import (
    BUDGET_SOFT_STOP_HEADER,
    BUDGET_SOFT_STOP_RESPONSE_HEADER,
    REPLAY_ONLY_HEADER,
    _budget_soft_stop_body,
    _openrouter_chat_completion_path,
    _openrouter_generation_id_from_headers,
    _openrouter_request_with_usage_metadata,
    _response_is_recordable,
)
from gateway.tee.provider_broker_v2 import (
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
    _exception_errno,
    _failure_code,
    _local_resource_failure,
    _nonsecret_headers,
    _safe_error_type,
    _sanitized_path,
)
from gateway.tee.inter_enclave_tls import REPLAY_WAIT_SECONDS
from gateway.tee.provider_evidence_v2 import (
    create_signed_provider_evidence_record,
)
from gateway.tee.source_add_runtime_v2 import (
    validate_source_add_runtime_route_v2,
)
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
)
from research_lab.eval.provider_costs import (
    DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP,
    DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    ProviderCostEstimate,
    ProviderCostLedger,
    decimal_from_env,
    estimate_provider_cost,
    extract_openrouter_cost_dollars,
    redacted_endpoint,
)
from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
)


MAX_CACHE_RECORDS = 10000
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_FIELDS = {
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
_OPTIONAL_REQUEST_FIELDS = {"dynamic_route"}
_LOCAL_RESPONSE_SCHEMA_VERSION = "leadpoet.attested_local_provider_response.v2"
_FAIL_CLOSED_REQUEST_SCHEMA_VERSION = (
    "leadpoet.provider_semantics_fail_closed_request.v2"
)
_PROVIDER_PREFLIGHT_PURPOSE = "research_lab.provider_preflight.v2"
_SEMANTICS_HEALTH_STAGES = (
    "provider_transport",
    "provider_cache_lookup",
    "provider_cache_write",
)
_LEGACY_PROVIDER_IDS = {
    "openrouter": "or",
    "scrapingdog": "sd",
    "exa": "exa",
    "deepline": "deepline",
}
class ProviderSemanticsV2Error(RuntimeError):
    """A provider cache, budget, cost, or authenticated result is invalid."""


logger = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _header(headers: Mapping[str, Any], name: str) -> str:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return ""


def _descriptor_hashes(descriptor: Mapping[str, Any]) -> set[str]:
    output = set()
    for field in (
        "artifact_id",
        "plaintext_hash",
        "ciphertext_hash",
        "encryption_context_hash",
    ):
        value = str(descriptor.get(field) or "")
        if value:
            if not _HASH_RE.fullmatch(value):
                raise ProviderSemanticsV2Error(
                    "local provider artifact descriptor is invalid"
                )
            output.add(value)
    return output


class ProviderSemanticsAuthorityV2:
    """Apply the existing cache/cap/cost rules around authenticated TLS I/O."""

    def __init__(
        self,
        *,
        broker: ProviderBrokerV2,
        cache_store: Any,
        artifact_sink: Callable[..., Mapping[str, Any]],
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        sign_digest: Callable[[bytes], Any],
        artifact_transaction: Optional[Callable[[], Any]] = None,
        clock: Callable[[], str] = _timestamp,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if cache_store is None:
            raise ProviderSemanticsV2Error("provider semantics cache is required")
        terminal_transaction = getattr(
            broker,
            "transient_terminal_transaction",
            None,
        )
        if not callable(terminal_transaction):
            raise ProviderSemanticsV2Error(
                "provider broker terminal transaction is required"
            )
        transport_reference_hashes = getattr(
            broker,
            "transport_reference_hashes",
            None,
        )
        if not callable(transport_reference_hashes):
            raise ProviderSemanticsV2Error(
                "provider broker transport reference resolver is required"
            )
        self._broker = broker
        self._cache_store = cache_store
        self._artifact_sink = artifact_sink
        self._artifact_transaction = artifact_transaction or nullcontext
        self._terminal_transaction = terminal_transaction
        self._boot_identity_supplier = boot_identity_supplier
        self._sign_digest = sign_digest
        self._clock = clock
        self._sleep = sleeper
        self._stage_health_lock = threading.RLock()
        self._stage_counts = {
            stage: {"started": 0, "succeeded": 0, "failed": 0}
            for stage in _SEMANTICS_HEALTH_STAGES
        }
        self._last_stage_failure = None  # type: Optional[Dict[str, Any]]
        self._cache: Dict[tuple[str, ...], Dict[str, Any]] = {}
        self._inflight: Dict[tuple[str, ...], threading.Event] = {}
        self._cost_ledgers: Dict[tuple[str, str], ProviderCostLedger] = {}
        self._live_calls: Dict[tuple[str, str], int] = {}
        self._cache_day = ""
        self._lock = threading.RLock()
        self._semantic_owner_state = threading.local()

    @contextmanager
    def _track_stage(self, stage: str):
        if stage not in _SEMANTICS_HEALTH_STAGES:
            raise ProviderSemanticsV2Error(
                "provider semantics health stage is invalid"
            )
        with self._stage_health_lock:
            self._stage_counts[stage]["started"] += 1
        try:
            yield
        except BaseException as exc:
            resource_errno, resource_kind = _local_resource_failure(exc)
            failure = {
                "stage": stage,
                "failure_code": _failure_code(exc),
                "error_type": _safe_error_type(exc),
                "errno": resource_errno or _exception_errno(exc),
            }
            if resource_kind:
                failure["local_resource_kind"] = resource_kind
            with self._stage_health_lock:
                self._stage_counts[stage]["failed"] += 1
                self._last_stage_failure = failure
            raise
        else:
            with self._stage_health_lock:
                self._stage_counts[stage]["succeeded"] += 1

    def health(self) -> Dict[str, Any]:
        broker = self._broker.health()
        with self._stage_health_lock:
            stage_counts = {
                stage: dict(self._stage_counts[stage])
                for stage in _SEMANTICS_HEALTH_STAGES
            }
            last_stage_failure = dict(self._last_stage_failure or {})
        with self._lock:
            result = {
                "schema_version": "leadpoet.provider_semantics.v2",
                "status": "ready" if broker.get("status") == "ready" else "provisioning",
                "broker_registry_hash": broker.get("registry_hash"),
                "cache_day": self._cache_day,
                "memory_cache_entry_count": len(self._cache),
                "inflight_count": len(self._inflight),
                "cost_scope_count": len(self._cost_ledgers),
                "stage_counters": stage_counts,
            }
        if last_stage_failure:
            result["last_stage_failure"] = last_stage_failure
        return result

    def execute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        prior_result = None  # type: Optional[Dict[str, Any]]
        self._semantic_owner_state.prior_broker_result = None
        try:
            try:
                with self._terminal_transaction(), self._artifact_transaction():
                    result = dict(self._execute(request))
                    prior_result = dict(result)
                    return result
            except Exception as exc:
                # The model may catch a provider transport failure and retry.
                # Give that exact request a measured failure terminal so a
                # later successful attempt can authorize the scope without
                # accepting an incomplete intent.
                logger.warning(
                    "provider_semantics_fail_closed_terminal "
                    "job_id=%s provider_id=%s error_type=%s",
                    str(request.get("job_id") or "")[:256],
                    str(request.get("provider_id") or "")[:128],
                    type(exc).__name__,
                )
                with self._terminal_transaction(), self._artifact_transaction():
                    retained_prior_result = prior_result or getattr(
                        self._semantic_owner_state,
                        "prior_broker_result",
                        None,
                    )
                    return self._fail_closed_terminal(
                        request,
                        exc=exc,
                        prior_result=retained_prior_result,
                    )
        finally:
            # Exact semantic replays are not released until the terminal and
            # encrypted cache work commit.
            try:
                self._release_semantic_owner()
            finally:
                if hasattr(
                    self._semantic_owner_state,
                    "prior_broker_result",
                ):
                    del self._semantic_owner_state.prior_broker_result

    def _fail_closed_terminal(
        self,
        request: Mapping[str, Any],
        *,
        exc: BaseException,
        prior_result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized, body, parsed, _fingerprint = self._request(request)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ProviderSemanticsV2Error(
                "provider semantics failure request is not HTTPS"
            ) from exc
        references = dict(self._broker.transport_reference_hashes(normalized))
        credential_ref_hash = str(references.get("credential_ref_hash") or "")
        egress_proxy_ref_hash = str(
            references.get("egress_proxy_ref_hash") or ""
        )
        if (
            not _HASH_RE.fullmatch(credential_ref_hash)
            or not _HASH_RE.fullmatch(egress_proxy_ref_hash)
        ):
            raise ProviderSemanticsV2Error(
                "provider semantics failure transport references are invalid"
            ) from exc
        diagnostic_artifact = None  # type: Optional[Dict[str, Any]]
        if prior_result is not None:
            reseal_diagnostic = getattr(
                self._broker,
                "reseal_transport_failure_diagnostic",
                None,
            )
            if callable(reseal_diagnostic):
                resealed = reseal_diagnostic(
                    prior_result=prior_result,
                    outer_error=exc,
                )
                if resealed is not None:
                    diagnostic_artifact = dict(resealed)
        diagnostic_plaintext_hash = (
            str(diagnostic_artifact.get("plaintext_hash") or "")
            if diagnostic_artifact is not None
            else ""
        )
        if diagnostic_plaintext_hash and not _HASH_RE.fullmatch(
            diagnostic_plaintext_hash
        ):
            raise ProviderSemanticsV2Error(
                "provider semantics failure diagnostic commitment is invalid"
            ) from exc
        failure_error_type = _safe_error_type(exc)
        request_doc = {
            "schema_version": _FAIL_CLOSED_REQUEST_SCHEMA_VERSION,
            "logical_operation_id": normalized["logical_operation_id"],
            "job_id": normalized["job_id"],
            "purpose": normalized["purpose"],
            "provider_id": normalized["provider_id"],
            "attempt_number": normalized["attempt_number"],
            "method": normalized["method"],
            "destination_host": str(parsed.hostname).lower(),
            "path_hash": sha256_bytes(
                _sanitized_path(parsed).encode("utf-8")
            ),
            "nonsecret_headers_hash": sha256_json(
                _nonsecret_headers(normalized["headers"])
            ),
            "body_hash": sha256_bytes(body),
            "retry_policy_hash": normalized["retry_policy_hash"],
            "timeout_ms": normalized["timeout_ms"],
            "failure_stage": "provider_semantics",
            "failure_error_type": failure_error_type,
            "provider_transport_failure_diagnostic_hash": (
                diagnostic_plaintext_hash
            ),
        }
        request_bytes = canonical_json(request_doc).encode("utf-8")
        request_artifact = dict(
            self._artifact_sink(
                request_bytes,
                job_id=normalized["job_id"],
                purpose=normalized["purpose"],
                artifact_kind="provider_request",
            )
        )
        request_artifact_hash = sha256_bytes(request_bytes)
        if request_artifact.get("plaintext_hash") != request_artifact_hash:
            raise ProviderSemanticsV2Error(
                "provider semantics failure request artifact differs"
            ) from exc
        request_artifact_id = str(request_artifact.get("artifact_id") or "")
        if not _HASH_RE.fullmatch(request_artifact_id):
            raise ProviderSemanticsV2Error(
                "provider semantics failure request artifact is invalid"
            ) from exc
        started_at = self._clock()
        attempt = build_transport_attempt(
            request_id=secrets.token_hex(16),
            logical_operation_id=normalized["logical_operation_id"],
            job_id=normalized["job_id"],
            purpose=normalized["purpose"],
            provider_id=normalized["provider_id"],
            attempt_number=normalized["attempt_number"],
            method=normalized["method"],
            destination_host=str(parsed.hostname),
            destination_port=parsed.port or 443,
            path_hash=request_doc["path_hash"],
            nonsecret_headers_hash=request_doc["nonsecret_headers_hash"],
            body_hash=request_doc["body_hash"],
            credential_ref_hash=credential_ref_hash,
            egress_proxy_ref_hash=egress_proxy_ref_hash,
            retry_policy_hash=normalized["retry_policy_hash"],
            timeout_ms=normalized["timeout_ms"],
            started_at=started_at,
            terminal_status="transport_failure",
            http_status=None,
            response_hash=None,
            request_artifact_hash=request_artifact_hash,
            response_artifact_hash=None,
            tls_peer_chain_hash=None,
            tls_protocol=None,
            failure_code="unexpected_eof",
            completed_at=self._clock(),
        )
        evidence_artifacts = _descriptor_hashes(request_artifact)
        if diagnostic_artifact is not None:
            evidence_artifacts.update(
                _descriptor_hashes(diagnostic_artifact)
            )
        return {
            "terminal_status": "transport_failure",
            "failure_code": "unexpected_eof",
            "failure_stage": "provider_semantics",
            "failure_error_type": failure_error_type,
            "encrypted_request_artifact_id": request_artifact_id,
            "transport_attempt": attempt,
            "evidence_artifact_hashes": sorted(evidence_artifacts),
        }

    def _execute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        normalized, original_body, parsed, fingerprint = self._request(request)
        day = self._utc_day()
        dynamic_route = normalized.get("dynamic_route")
        provider = _LEGACY_PROVIDER_IDS.get(normalized["provider_id"])
        if provider is None and isinstance(dynamic_route, Mapping):
            provider = str(dynamic_route["provider_id"])
        if provider is None:
            # Infrastructure routes such as Supabase retain authenticated
            # transport, but must not recursively read or populate the
            # Supabase-backed paid-provider evidence cache.
            return self._live(
                normalized,
                original_body=original_body,
                parsed=parsed,
                fingerprint=fingerprint,
                ledger=None,
                provider=None,
                lookup_attempts=[],
                lookup_artifacts=[],
                day=day,
                cache_recording_enabled=False,
            )
        bypass_cache = normalized["purpose"] == _PROVIDER_PREFLIGHT_PURPOSE
        cache_key = (
            (day, fingerprint, normalized["job_id"])
            if bypass_cache
            else (day, fingerprint)
        )
        while True:
            with self._lock:
                self._roll_day(day)
                cached = None if bypass_cache else self._cache.get(cache_key)
                if cached is not None:
                    return self._cache_hit(
                        normalized,
                        original_body=original_body,
                        parsed=parsed,
                        fingerprint=fingerprint,
                        cached=cached,
                    )
                event = self._inflight.get(cache_key)
                if event is None:
                    event = threading.Event()
                    self._inflight[cache_key] = event
                    if getattr(self._semantic_owner_state, "owner", None) is not None:
                        raise ProviderSemanticsV2Error(
                            "provider semantics owner state is already active"
                        )
                    self._semantic_owner_state.owner = (cache_key, event)
                    break
            # The owner includes encrypted cache readback after the upstream
            # HTTP call. Wait on the same bounded
            # completion contract as an exact inter-enclave replay, not the
            # shorter provider network timeout.
            if not event.wait(REPLAY_WAIT_SECONDS):
                raise ProviderSemanticsV2Error(
                    "provider semantics single-flight wait timed out"
                )

        try:
            lookup = (
                {
                    "found": False,
                    "payload": {},
                    "transport_attempts": [],
                    "evidence_artifact_hashes": [],
                }
                if bypass_cache
                else self._load_provider_cache(
                    utc_day=day,
                    request_fingerprint=fingerprint,
                    job_id=normalized["job_id"],
                    purpose=normalized["purpose"],
                    attempt_number=normalized["attempt_number"],
                )
            )
            lookup_attempts = list(lookup["transport_attempts"])
            lookup_artifacts = list(lookup["evidence_artifact_hashes"])
            if lookup["found"]:
                cached = dict(lookup["payload"])
                with self._lock:
                    self._cache[cache_key] = dict(cached)
                return self._cache_hit(
                    normalized,
                    original_body=original_body,
                    parsed=parsed,
                    fingerprint=fingerprint,
                    cached=cached,
                    additional_attempts=lookup_attempts,
                    additional_artifacts=lookup_artifacts,
                )

            headers = normalized["headers"]
            ledger = self._cost_ledger(
                day,
                headers,
                default_scope=(
                    normalized["job_id"] if bypass_cache else "unscoped"
                ),
            )
            if isinstance(dynamic_route, Mapping):
                quota = int(dynamic_route["per_day_quota"])
                with self._lock:
                    used = self._live_calls.get((day, provider), 0)
                if quota > 0 and used >= quota:
                    return self._local_response(
                        normalized,
                        parsed=parsed,
                        body=b'{"error":"provider day quota exhausted"}',
                        status=429,
                        evidence="quota_exhausted",
                        cost_event=None,
                        additional_attempts=lookup_attempts,
                        additional_artifacts=lookup_artifacts,
                    )
            if _truthy(_header(headers, REPLAY_ONLY_HEADER)):
                event_doc = ledger.cache_hit_event(
                    provider=provider,
                    endpoint=redacted_endpoint(provider, normalized["url"]),
                    request_fingerprint=fingerprint,
                    status_code=409,
                )
                return self._local_response(
                    normalized,
                    parsed=parsed,
                    body=b'{"error":"replay_miss"}',
                    status=409,
                    evidence="replay_miss",
                    cost_event=event_doc.to_doc(),
                    additional_attempts=lookup_attempts,
                    additional_artifacts=lookup_artifacts,
                )
            if ledger.should_block_paid_call():
                reason = ledger.block_reason()
                soft_stop = reason == "cost_cap_reached" and _truthy(
                    _header(headers, BUDGET_SOFT_STOP_HEADER)
                )
                status = 200 if soft_stop else 402
                evidence = "budget_soft_stop" if soft_stop else "blocked"
                event_doc = ledger.block_event(
                    provider=provider,
                    endpoint=redacted_endpoint(provider, normalized["url"]),
                    request_fingerprint=fingerprint,
                    reason=reason,
                    status_code=status,
                    evidence=evidence,
                ).to_doc()
                body = (
                    _budget_soft_stop_body(provider, normalized["url"])
                    if soft_stop
                    else canonical_json(
                        {
                            "error": (
                                "research_lab_provider_cost_cap_exceeded"
                                if reason == "cost_cap_reached"
                                else "research_lab_provider_cost_tracking_failed"
                            ),
                            "provider": provider,
                            "endpoint": redacted_endpoint(provider, normalized["url"]),
                        }
                    ).encode("utf-8")
                )
                return self._local_response(
                    normalized,
                    parsed=parsed,
                    body=body,
                    status=status,
                    evidence=evidence,
                    cost_event=event_doc,
                    extra_headers=(
                        {BUDGET_SOFT_STOP_RESPONSE_HEADER: "1"}
                        if soft_stop
                        else {}
                    ),
                    additional_attempts=lookup_attempts,
                    additional_artifacts=lookup_artifacts,
                )
            return self._live(
                normalized,
                original_body=original_body,
                parsed=parsed,
                fingerprint=fingerprint,
                ledger=ledger,
                provider=provider,
                lookup_attempts=lookup_attempts,
                lookup_artifacts=lookup_artifacts,
                day=day,
                cache_recording_enabled=not bypass_cache,
            )
        finally:
            # execute() releases the owner after the surrounding terminal and
            # artifact transactions commit or roll back.
            pass

    def _release_semantic_owner(self) -> None:
        owner = getattr(self._semantic_owner_state, "owner", None)
        if owner is None:
            return
        cache_key, owner_event = owner
        try:
            with self._lock:
                event = self._inflight.get(cache_key)
                if event is owner_event:
                    self._inflight.pop(cache_key, None)
                    event.set()
        finally:
            del self._semantic_owner_state.owner

    def _load_provider_cache(self, **kwargs: Any) -> Mapping[str, Any]:
        with self._track_stage("provider_cache_lookup"):
            return self._cache_store.load(**kwargs)

    def _live(
        self,
        normalized: Mapping[str, Any],
        *,
        original_body: bytes,
        parsed: Any,
        fingerprint: str,
        ledger: Optional[ProviderCostLedger],
        provider: Optional[str],
        lookup_attempts: list[Mapping[str, Any]],
        lookup_artifacts: list[str],
        day: str,
        cache_recording_enabled: bool,
    ) -> Dict[str, Any]:
        request = dict(normalized)
        request["headers"] = {
            str(name): str(value)
            for name, value in normalized["headers"].items()
            if not str(name).lower().startswith("x-research-lab-")
            and str(name).lower() != "accept-encoding"
        }
        request["headers"]["Accept-Encoding"] = "identity"
        upstream_body = original_body
        if provider == "or" and _openrouter_chat_completion_path(normalized["url"]):
            upstream_body = _openrouter_request_with_usage_metadata(original_body)
            request["body_b64"] = base64.b64encode(upstream_body).decode("ascii")
            if upstream_body != original_body:
                request["headers"] = {
                    name: value
                    for name, value in request["headers"].items()
                    if name.lower() not in {"content-length", "transfer-encoding"}
                }
        with self._track_stage("provider_transport"):
            result = dict(self._broker.execute(request))
        # Keep the coordinator-only result until the surrounding terminal and
        # artifact transactions commit.  If later cache/outcome persistence
        # rolls them back, execute() can rebind sanitized failure provenance to
        # the fresh fail-closed terminal without exposing diagnostic plaintext.
        self._semantic_owner_state.prior_broker_result = dict(result)
        broker_artifacts = list(result.get("evidence_artifact_hashes") or [])
        additional_attempts = list(lookup_attempts)
        evidence_artifacts = set(lookup_artifacts) | set(broker_artifacts)
        if result.get("terminal_status") != "authenticated_response":
            result["additional_transport_attempts"] = [
                dict(item) for item in additional_attempts
            ]
            result["evidence_artifact_hashes"] = sorted(evidence_artifacts)
            return result

        status = int(result["http_status"])
        body = base64.b64decode(str(result["body_b64"]), validate=True)
        evidence = "live_unrecorded"
        cost_event = None
        if provider is not None and ledger is not None:
            recordable = (
                cache_recording_enabled
                and _response_is_recordable(
                    provider,
                    normalized["url"],
                    status,
                    body,
                )
            )
            evidence = "recorded" if recordable else "live_unrecorded"
            estimate = estimate_provider_cost(
                provider=provider,
                upstream_url=normalized["url"],
                status=status,
                response_body=body,
                request_body=upstream_body or original_body or None,
                scrapingdog_credit_price_usd=decimal_from_env(
                    "RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD",
                    DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
                ),
            )
            estimate, reconciliation_attempts, reconciliation_artifacts = (
                self._reconcile_openrouter(
                    normalized,
                    provider=provider,
                    estimate=estimate,
                    response_headers=result.get("headers") or {},
                )
            )
            additional_attempts.extend(reconciliation_attempts)
            evidence_artifacts.update(reconciliation_artifacts)
            cost_event = ledger.record_live_event(
                provider=provider,
                request_fingerprint=fingerprint,
                status_code=status,
                estimate=estimate,
                evidence=evidence,
            ).to_doc()
            with self._lock:
                key = (day, provider)
                self._live_calls[key] = self._live_calls.get(key, 0) + 1
            if recordable:
                terminal = self._recorded_terminal(
                    normalized,
                    fingerprint=fingerprint,
                    body=body,
                    result=result,
                    source_artifacts=sorted(evidence_artifacts),
                )
                with self._track_stage("provider_cache_write"):
                    persisted = self._cache_store.persist_recorded(
                        terminal,
                        utc_day=day,
                        job_id=normalized["job_id"],
                        purpose=normalized["purpose"],
                    )
                additional_attempts.extend(persisted["transport_attempts"])
                evidence_artifacts.update(persisted["evidence_artifact_hashes"])
                with self._lock:
                    if len(self._cache) >= MAX_CACHE_RECORDS:
                        raise ProviderSemanticsV2Error(
                            "provider semantics cache capacity is full"
                        )
                    self._cache[(day, fingerprint)] = {
                        "schema_version": "leadpoet.provider_evidence_cache_payload.v2",
                        "utc_day": day,
                        "request_fingerprint": fingerprint,
                        "status": status,
                        "body_b64": base64.b64encode(body).decode("ascii"),
                        "source_record": dict(terminal["record"]),
                        "source_boot_identity": dict(
                            terminal["coordinator_boot_identity"]
                        ),
                        "source_transport_attempt": dict(
                            terminal["transport_attempts"][0]
                        ),
                        "source_evidence_artifact_hashes": list(
                            terminal["evidence_artifact_hashes"]
                        ),
                    }
        response_headers = {
            "Content-Type": "application/json",
            "X-Research-Lab-Evidence": evidence,
        }
        if cost_event is not None:
            response_headers.update(self._cost_headers(cost_event))
            evidence_artifacts.add(str(cost_event["event_hash"]))
        response = {
            **result,
            "headers": response_headers,
            "evidence": evidence,
            "additional_transport_attempts": [
                dict(item) for item in additional_attempts
            ],
            "evidence_artifact_hashes": sorted(evidence_artifacts),
        }
        return response

    def _cache_hit(
        self,
        normalized: Mapping[str, Any],
        *,
        original_body: bytes,
        parsed: Any,
        fingerprint: str,
        cached: Mapping[str, Any],
        additional_attempts: Optional[list[Mapping[str, Any]]] = None,
        additional_artifacts: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        body = base64.b64decode(str(cached["body_b64"]), validate=True)
        provider = _LEGACY_PROVIDER_IDS.get(normalized["provider_id"])
        ledger = self._cost_ledger(self._utc_day(), normalized["headers"])
        cost_event = ledger.cache_hit_event(
            provider=provider or normalized["provider_id"],
            endpoint=redacted_endpoint(
                provider or normalized["provider_id"], normalized["url"]
            ),
            request_fingerprint=fingerprint,
            status_code=int(cached["status"]),
        ).to_doc()
        return self._local_response(
            normalized,
            parsed=parsed,
            body=body,
            status=int(cached["status"]),
            evidence="hit",
            cost_event=cost_event,
            source_attempt=cached.get("source_transport_attempt"),
            source_record=cached.get("source_record"),
            source_boot_identity=cached.get("source_boot_identity"),
            additional_attempts=list(additional_attempts or ()),
            additional_artifacts=[
                *list(additional_artifacts or ()),
                *list(cached.get("source_evidence_artifact_hashes") or ()),
            ],
        )

    def _local_response(
        self,
        normalized: Mapping[str, Any],
        *,
        parsed: Any,
        body: bytes,
        status: int,
        evidence: str,
        cost_event: Optional[Mapping[str, Any]],
        source_attempt: Optional[Mapping[str, Any]] = None,
        source_record: Optional[Mapping[str, Any]] = None,
        source_boot_identity: Optional[Mapping[str, Any]] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        additional_attempts: Optional[list[Mapping[str, Any]]] = None,
        additional_artifacts: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        request_doc = {
            "schema_version": _LOCAL_RESPONSE_SCHEMA_VERSION,
            "kind": "request",
            "request": {
                **dict(normalized),
                "headers": _nonsecret_headers(normalized["headers"]),
            },
            "evidence": evidence,
        }
        request_bytes = canonical_json(request_doc).encode("utf-8")
        request_artifact = dict(
            self._artifact_sink(
                request_bytes,
                job_id=normalized["job_id"],
                purpose=normalized["purpose"],
                artifact_kind="provider_request",
            )
        )
        response_artifact = dict(
            self._artifact_sink(
                body,
                job_id=normalized["job_id"],
                purpose=normalized["purpose"],
                artifact_kind="provider_response",
            )
        )
        current_refs = dict(self._broker.transport_reference_hashes(normalized))
        credential_ref_hash = str(current_refs.get("credential_ref_hash") or "")
        egress_proxy_ref_hash = str(
            current_refs.get("egress_proxy_ref_hash") or ""
        )
        if (
            not _HASH_RE.fullmatch(credential_ref_hash)
            or not _HASH_RE.fullmatch(egress_proxy_ref_hash)
        ):
            raise ProviderSemanticsV2Error(
                "attested local response transport references are invalid"
            )
        attempt = build_transport_attempt(
            request_id=secrets.token_hex(16),
            logical_operation_id=normalized["logical_operation_id"],
            job_id=normalized["job_id"],
            purpose=normalized["purpose"],
            provider_id=normalized["provider_id"],
            attempt_number=normalized["attempt_number"],
            method=normalized["method"],
            destination_host=str(parsed.hostname or ""),
            destination_port=443,
            path_hash=sha256_bytes(_sanitized_path(parsed).encode("utf-8")),
            nonsecret_headers_hash=sha256_json(
                _nonsecret_headers(normalized["headers"])
            ),
            body_hash=sha256_bytes(
                base64.b64decode(normalized["body_b64"], validate=True)
            ),
            credential_ref_hash=credential_ref_hash,
            egress_proxy_ref_hash=egress_proxy_ref_hash,
            retry_policy_hash=normalized["retry_policy_hash"],
            timeout_ms=normalized["timeout_ms"],
            started_at=self._clock(),
            terminal_status="attested_local_response",
            http_status=int(status),
            response_hash=sha256_bytes(body),
            request_artifact_hash=sha256_bytes(request_bytes),
            response_artifact_hash=sha256_bytes(body),
            tls_peer_chain_hash=None,
            tls_protocol=None,
            failure_code=None,
            completed_at=self._clock(),
        )
        artifacts = (
            _descriptor_hashes(request_artifact)
            | _descriptor_hashes(response_artifact)
            | set(str(item) for item in (additional_artifacts or ()))
        )
        if cost_event is not None:
            artifacts.add(str(cost_event["event_hash"]))
        if isinstance(source_record, Mapping):
            artifacts.add(str(source_record.get("record_hash") or ""))
        if isinstance(source_boot_identity, Mapping):
            artifacts.add(
                str(source_boot_identity.get("boot_identity_hash") or "")
            )
        if any(not _HASH_RE.fullmatch(item) for item in artifacts):
            raise ProviderSemanticsV2Error(
                "attested local response artifact is invalid"
            )
        headers = {
            "Content-Type": "application/json",
            "X-Research-Lab-Evidence": evidence,
            **dict(extra_headers or {}),
        }
        if cost_event is not None:
            headers.update(self._cost_headers(cost_event))
        return {
            "terminal_status": "attested_local_response",
            "http_status": int(status),
            "headers": headers,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": request_artifact["artifact_id"],
            "encrypted_artifact_id": response_artifact["artifact_id"],
            "transport_attempt": attempt,
            "additional_transport_attempts": [
                dict(item) for item in (additional_attempts or ())
            ],
            "evidence_artifact_hashes": sorted(artifacts),
            "evidence": evidence,
            "source_record": dict(source_record or {}),
            "source_boot_identity": dict(source_boot_identity or {}),
        }

    def _recorded_terminal(
        self,
        normalized: Mapping[str, Any],
        *,
        fingerprint: str,
        body: bytes,
        result: Mapping[str, Any],
        source_artifacts: list[str],
    ) -> Dict[str, Any]:
        boot = dict(self._boot_identity_supplier())
        attempt = dict(result["transport_attempt"])
        record = create_signed_provider_evidence_record(
            body={
                "coordinator_boot_identity_hash": boot["boot_identity_hash"],
                "request_hash": sha256_json(dict(normalized)),
                "request_fingerprint": fingerprint,
                "evidence": "recorded",
                "status": int(result["http_status"]),
                "body_hash": sha256_bytes(body),
                "encrypted_request_artifact_id": str(
                    result["encrypted_request_artifact_id"]
                ),
                "encrypted_response_artifact_id": str(
                    result["encrypted_artifact_id"]
                ),
                "transport_attempt_hash": attempt["attempt_hash"],
                "source_record_hash": "",
                "issued_at": self._clock(),
            },
            coordinator_pubkey=boot["signing_pubkey"],
            sign_digest=self._sign_digest,
        )
        artifacts = sorted(
            set(source_artifacts)
            | {
                record["record_hash"],
                str(boot["boot_identity_hash"]),
            }
        )
        return {
            "status": int(result["http_status"]),
            "body_b64": base64.b64encode(body).decode("ascii"),
            "evidence": "recorded",
            "transport_attempts": [attempt],
            "evidence_artifact_hashes": artifacts,
            "record": record,
            "source_record": None,
            "source_boot_identity": None,
            "coordinator_boot_identity": boot,
        }

    def _reconcile_openrouter(
        self,
        normalized: Mapping[str, Any],
        *,
        provider: str,
        estimate: ProviderCostEstimate,
        response_headers: Mapping[str, Any],
    ) -> tuple[ProviderCostEstimate, list[Dict[str, Any]], set[str]]:
        if provider != "or":
            return estimate, [], set()
        generation_id = estimate.generation_id or _openrouter_generation_id_from_headers(
            response_headers
        )
        if generation_id and not estimate.generation_id:
            estimate = replace(estimate, generation_id=generation_id)
        if not generation_id or (
            estimate.tracking_reason != "missing_openrouter_cost"
            and estimate.cost_source != "openrouter_missing_cost_zero_cost"
        ):
            return estimate, [], set()
        credential_routes = (
            ["openrouter"]
            if self._broker.credential_available(
                job_id=str(normalized["job_id"]),
                slot="openrouter",
            )
            else []
        )
        if not credential_routes:
            return estimate, [], set()
        operation_suffix = sha256_bytes(generation_id.encode("utf-8")).split(
            ":", 1
        )[1][:16]
        attempts = []
        artifacts = set()
        for retry_ordinal, delay in enumerate((0.0, 2.0, 5.0, 10.0, 20.0)):
            if delay:
                self._sleep(delay)
            for credential_ordinal, route in enumerate(credential_routes):
                attempt_number = (
                    retry_ordinal * len(credential_routes) + credential_ordinal
                )
                try:
                    with self._track_stage("provider_transport"):
                        result = dict(
                            self._broker.execute(
                            {
                                "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                                "logical_operation_id": "%s:cost-reconcile:%s"
                                % (
                                    normalized["logical_operation_id"],
                                    operation_suffix,
                                ),
                                "job_id": normalized["job_id"],
                                "purpose": normalized["purpose"],
                                "provider_id": route,
                                "attempt_number": attempt_number,
                                "method": "GET",
                                "url": (
                                    "https://openrouter.ai/api/v1/generation?id="
                                    + quote(generation_id, safe="")
                                ),
                                "headers": {"accept": "application/json"},
                                "body_b64": "",
                                "timeout_ms": 30000,
                                "retry_policy_hash": self._broker.retry_policy_hashes[
                                    route
                                ],
                            }
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "provider_semantics_openrouter_reconcile_attempt_failed "
                        "route=%s retry_ordinal=%d error_type=%s",
                        route,
                        retry_ordinal,
                        type(exc).__name__,
                    )
                    continue
                attempt = result.get("transport_attempt")
                if isinstance(attempt, Mapping):
                    attempts.append(dict(attempt))
                artifacts.update(result.get("evidence_artifact_hashes") or ())
                if result.get("terminal_status") != "authenticated_response":
                    continue
                status = int(result.get("http_status") or 0)
                if 200 <= status < 300:
                    body = base64.b64decode(
                        str(result.get("body_b64") or ""), validate=True
                    )
                    cost, metadata = extract_openrouter_cost_dollars(body)
                    if cost is not None:
                        estimate = ProviderCostEstimate(
                            provider="or",
                            endpoint=estimate.endpoint,
                            model=(
                                estimate.model
                                or str(metadata.get("model") or "")[:160]
                            ),
                            billable=True,
                            cost_usd=cost,
                            cost_source="openrouter_generation_reconciliation",
                            prompt_tokens=int(metadata.get("prompt_tokens") or 0),
                            completion_tokens=int(
                                metadata.get("completion_tokens") or 0
                            ),
                            generation_id=generation_id,
                        )
                    return estimate, attempts, artifacts
                if status in {401, 403} and route == credential_routes[-1]:
                    return estimate, attempts, artifacts
        return estimate, attempts, artifacts

    def _request(
        self,
        request: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], bytes, Any, str]:
        request_fields = (
            frozenset(request) if isinstance(request, Mapping) else frozenset()
        )
        if (
            not isinstance(request, Mapping)
            or not frozenset(_REQUEST_FIELDS).issubset(request_fields)
            or not request_fields.issubset(
                frozenset(_REQUEST_FIELDS | _OPTIONAL_REQUEST_FIELDS)
            )
        ):
            raise ProviderSemanticsV2Error("provider semantics request fields are invalid")
        if request.get("schema_version") != PROVIDER_BROKER_SCHEMA_VERSION:
            raise ProviderSemanticsV2Error("provider semantics schema is invalid")
        headers = request.get("headers")
        if not isinstance(headers, Mapping):
            raise ProviderSemanticsV2Error("provider semantics headers are invalid")
        try:
            body = base64.b64decode(str(request.get("body_b64") or ""), validate=True)
        except Exception as exc:
            raise ProviderSemanticsV2Error(
                "provider semantics body is invalid"
            ) from exc
        parsed = urlsplit(str(request.get("url") or ""))
        fingerprint = canonical_request_fingerprint(
            str(request.get("method") or ""),
            str(request.get("url") or ""),
            body or None,
        )
        normalized = {**dict(request), "headers": dict(headers)}
        if "dynamic_route" in request:
            try:
                dynamic_route = validate_source_add_runtime_route_v2(
                    request["dynamic_route"]
                )
            except Exception as exc:
                raise ProviderSemanticsV2Error(
                    "provider semantics dynamic route is invalid"
                ) from exc
            if dynamic_route["provider_id"] != str(
                request.get("provider_id") or ""
            ):
                raise ProviderSemanticsV2Error(
                    "provider semantics dynamic identity differs"
                )
            normalized["dynamic_route"] = dynamic_route
        return normalized, body, parsed, fingerprint

    def _cost_ledger(
        self,
        day: str,
        headers: Mapping[str, Any],
        *,
        default_scope: str = "unscoped",
    ) -> ProviderCostLedger:
        scope = (
            _header(headers, "X-Research-Lab-Cost-Scope").strip()
            or default_scope
        )
        cap = decimal_from_env(
            "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP",
            DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP,
        )
        header_cap = _header(headers, "X-Research-Lab-Cost-Cap-Usd").strip()
        if header_cap:
            try:
                parsed = Decimal(header_cap)
                if parsed >= 0:
                    cap = parsed
            except Exception:
                pass
        key = (day, scope)
        with self._lock:
            ledger = self._cost_ledgers.get(key)
            if ledger is None:
                ledger = ProviderCostLedger(scope=scope, cap_usd=cap)
                self._cost_ledgers[key] = ledger
            return ledger

    @staticmethod
    def _cost_headers(event_doc: Mapping[str, Any]) -> Dict[str, str]:
        return {
            "X-Research-Lab-Provider-Cost-Event": base64.b64encode(
                canonical_json(dict(event_doc)).encode("utf-8")
            ).decode("ascii")
        }

    def _utc_day(self) -> str:
        timestamp = str(self._clock() or "")
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            timestamp,
        ):
            raise ProviderSemanticsV2Error("provider semantics clock is invalid")
        return timestamp[:10]

    def _roll_day(self, day: str) -> None:
        if self._cache_day == day:
            return
        self._cache.clear()
        self._cost_ledgers.clear()
        self._live_calls.clear()
        self._cache_day = day
