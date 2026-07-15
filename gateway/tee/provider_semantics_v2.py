"""Measured preservation layer for existing Research Lab provider semantics."""

from __future__ import annotations

import base64
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
    _nonsecret_headers,
    _sanitized_path,
)
from gateway.tee.provider_evidence_v2 import (
    create_signed_provider_evidence_record,
)
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from gateway.tee.source_add_runtime_v2 import (
    validate_source_add_runtime_route_v2,
)
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
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
TREE_PROVIDER_CALL_CAP_HEADER = "X-Research-Lab-Tree-Provider-Call-Cap"
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
        clock: Callable[[], str] = _timestamp,
        sleeper: Callable[[float], None] = time.sleep,
        outcome_ledger: Optional[ProviderOutcomeLedgerV2] = None,
        outcome_store: Any = None,
    ) -> None:
        if cache_store is None:
            raise ProviderSemanticsV2Error("provider semantics cache is required")
        self._broker = broker
        self._cache_store = cache_store
        self._artifact_sink = artifact_sink
        self._boot_identity_supplier = boot_identity_supplier
        self._sign_digest = sign_digest
        self._clock = clock
        self._sleep = sleeper
        self._outcome_store = outcome_store
        self._outcome_persist_lock = threading.RLock()
        self._outcome_checkpoint_hash = ""
        self._outcome_checkpoint_day = ""
        self._outcome_restore_attempts: list[Dict[str, Any]] = []
        self._outcome_restore_artifacts: set[str] = set()
        if outcome_ledger is not None:
            self._outcome_ledger = outcome_ledger
        elif outcome_store is not None:
            utc_day = str(clock() or "")[:10]
            restored = outcome_store.load_latest(
                utc_day=utc_day,
                job_id="provider-outcome-restore-%s" % utc_day,
                purpose="research_lab.provider_outcome_state.v2",
            )
            self._outcome_restore_attempts = [
                dict(item) for item in restored.get("transport_attempts") or ()
            ]
            self._outcome_restore_artifacts = set(
                str(item) for item in restored.get("evidence_artifact_hashes") or ()
            )
            initial_document = restored.get("state_document") if restored.get("found") else None
            self._outcome_checkpoint_hash = str(
                restored.get("checkpoint_hash") or ""
            )
            self._outcome_checkpoint_day = utc_day if self._outcome_checkpoint_hash else ""
            self._outcome_ledger = ProviderOutcomeLedgerV2(
                clock=clock,
                initial_document=initial_document,
            )
        else:
            self._outcome_ledger = ProviderOutcomeLedgerV2(clock=clock)
        self._cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._inflight: Dict[tuple[str, str], threading.Event] = {}
        self._cost_ledgers: Dict[tuple[str, str], ProviderCostLedger] = {}
        self._live_calls: Dict[tuple[str, str], int] = {}
        self._tree_live_calls: Dict[tuple[str, str], int] = {}
        self._cache_day = ""
        self._lock = threading.RLock()

    def health(self) -> Dict[str, Any]:
        broker = self._broker.health()
        with self._lock:
            return {
                "schema_version": "leadpoet.provider_semantics.v2",
                "status": "ready" if broker.get("status") == "ready" else "provisioning",
                "broker_registry_hash": broker.get("registry_hash"),
                "cache_day": self._cache_day,
                "memory_cache_entry_count": len(self._cache),
                "inflight_count": len(self._inflight),
                "cost_scope_count": len(self._cost_ledgers),
            }

    def execute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        result = dict(self._execute(request))
        persistence = self._record_provider_outcome(request, result)
        if persistence:
            result["additional_transport_attempts"] = [
                *[dict(item) for item in result.get("additional_transport_attempts") or ()],
                *[dict(item) for item in persistence["transport_attempts"]],
            ]
            result["evidence_artifact_hashes"] = sorted(
                {
                    *[str(item) for item in result.get("evidence_artifact_hashes") or ()],
                    *[str(item) for item in persistence["evidence_artifact_hashes"]],
                }
            )
        return result

    def _execute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        normalized, original_body, parsed, fingerprint = self._request(request)
        day = self._utc_day()
        cache_key = (day, fingerprint)
        timeout_seconds = max(1.0, normalized["timeout_ms"] / 1000.0 + 5.0)
        while True:
            with self._lock:
                self._roll_day(day)
                cached = self._cache.get(cache_key)
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
                    break
            if not event.wait(timeout_seconds):
                raise ProviderSemanticsV2Error(
                    "provider semantics single-flight wait timed out"
                )

        try:
            lookup = self._cache_store.load(
                utc_day=day,
                request_fingerprint=fingerprint,
                job_id=normalized["job_id"],
                purpose=normalized["purpose"],
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
            ledger = self._cost_ledger(day, headers)
            dynamic_route = normalized.get("dynamic_route")
            provider = _LEGACY_PROVIDER_IDS.get(normalized["provider_id"])
            if provider is None and isinstance(dynamic_route, Mapping):
                provider = str(dynamic_route["provider_id"])
            if provider is None:
                # Non-Research-Lab public routes retain authenticated transport
                # but do not invent cost/cache semantics they never had.
                return self._live(
                    normalized,
                    original_body=original_body,
                    parsed=parsed,
                    fingerprint=fingerprint,
                    ledger=None,
                    provider=None,
                    lookup_attempts=lookup_attempts,
                    lookup_artifacts=lookup_artifacts,
                    day=day,
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
            raw_tree_call_cap = _header(
                headers, TREE_PROVIDER_CALL_CAP_HEADER
            ).strip()
            if raw_tree_call_cap:
                try:
                    tree_call_cap = int(raw_tree_call_cap)
                except ValueError as exc:
                    raise ProviderSemanticsV2Error(
                        "tree provider call cap is invalid"
                    ) from exc
                scope = _header(headers, "X-Research-Lab-Cost-Scope").strip()
                if not scope or tree_call_cap < 1 or tree_call_cap > 10_000:
                    raise ProviderSemanticsV2Error(
                        "tree provider call cap scope is invalid"
                    )
                tree_call_key = (day, scope)
                with self._lock:
                    used_tree_calls = self._tree_live_calls.get(tree_call_key, 0)
                    if used_tree_calls < tree_call_cap:
                        self._tree_live_calls[tree_call_key] = used_tree_calls + 1
                if used_tree_calls >= tree_call_cap:
                    event_doc = ledger.block_event(
                        provider=provider,
                        endpoint=redacted_endpoint(provider, normalized["url"]),
                        request_fingerprint=fingerprint,
                        reason="provider_call_cap_reached",
                        status_code=402,
                        evidence="blocked",
                    ).to_doc()
                    return self._local_response(
                        normalized,
                        parsed=parsed,
                        body=b'{"error":"research_lab_tree_provider_call_cap_exceeded"}',
                        status=402,
                        evidence="blocked",
                        cost_event=event_doc,
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
            )
        finally:
            with self._lock:
                event = self._inflight.pop(cache_key, None)
                if event is not None:
                    event.set()

    def provider_outcome_snapshot(self) -> Dict[str, Any]:
        return self._outcome_ledger.snapshot()

    def provider_outcome_snapshot_evidence(self) -> Dict[str, Any]:
        artifacts = set(self._outcome_restore_artifacts)
        if self._outcome_checkpoint_hash:
            artifacts.add(self._outcome_checkpoint_hash)
        return {
            "snapshot": self.provider_outcome_snapshot(),
            "transport_attempts": [
                dict(item) for item in self._outcome_restore_attempts
            ],
            "evidence_artifact_hashes": sorted(artifacts),
        }

    def _record_provider_outcome(
        self,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Dict[str, Any] | None:
        provider_id = str(request.get("provider_id") or "")
        dynamic_route = request.get("dynamic_route")
        provider = _LEGACY_PROVIDER_IDS.get(provider_id)
        if provider is None and isinstance(dynamic_route, Mapping):
            provider = str(dynamic_route.get("provider_id") or "")
        if not provider:
            return None
        terminal_status = str(result.get("terminal_status") or "")
        evidence = str(result.get("evidence") or "")
        status = result.get("http_status")
        if terminal_status == "transport_failure":
            evidence = "error"
            status = 502
        if not evidence:
            evidence = "recorded" if terminal_status == "authenticated_response" else "error"
        try:
            normalized_status = int(status or 0)
        except (TypeError, ValueError) as exc:
            raise ProviderSemanticsV2Error(
                "provider outcome status is invalid"
            ) from exc
        cost_event: Dict[str, Any] = {}
        encoded_cost = _header(
            dict(result.get("headers") or {}),
            "X-Research-Lab-Provider-Cost-Event",
        )
        if encoded_cost:
            try:
                decoded = base64.b64decode(encoded_cost, validate=True)
                parsed_cost = json.loads(decoded.decode("utf-8"))
            except Exception as exc:
                raise ProviderSemanticsV2Error(
                    "provider outcome cost event is invalid"
                ) from exc
            if not isinstance(parsed_cost, Mapping):
                raise ProviderSemanticsV2Error(
                    "provider outcome cost event is not an object"
                )
            cost_event = dict(parsed_cost)
        with self._outcome_persist_lock:
            document = self._outcome_ledger.record(
                provider_id=provider,
                endpoint_class=redacted_endpoint(provider, str(request.get("url") or "")),
                evidence=evidence,
                status=normalized_status,
                live_call=(
                    terminal_status in {"authenticated_response", "transport_failure"}
                    and evidence not in {"hit", "blocked", "budget_soft_stop", "quota_exhausted", "replay_miss"}
                ),
                cost_event=cost_event,
            )
            if self._outcome_store is None:
                return None
            utc_day = str(document["utc_day"])
            previous = (
                self._outcome_checkpoint_hash
                if self._outcome_checkpoint_day == utc_day
                else ""
            )
            persisted = dict(
                self._outcome_store.persist(
                    document,
                    previous_checkpoint_hash=previous,
                    job_id=str(request.get("job_id") or ""),
                    purpose=str(request.get("purpose") or ""),
                )
            )
            self._outcome_checkpoint_hash = str(persisted["checkpoint_hash"])
            self._outcome_checkpoint_day = utc_day
            return persisted

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
        result = dict(self._broker.execute(request))
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
            recordable = _response_is_recordable(
                provider,
                normalized["url"],
                status,
                body,
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
        return {
            **result,
            "headers": response_headers,
            "evidence": evidence,
            "additional_transport_attempts": [
                dict(item) for item in additional_attempts
            ],
            "evidence_artifact_hashes": sorted(evidence_artifacts),
        }

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
        source = dict(source_attempt or {})
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
            credential_ref_hash=str(
                source.get("credential_ref_hash")
                or sha256_bytes(
                    ("leadpoet-attested-local-response:" + evidence).encode("utf-8")
                )
            ),
            egress_proxy_ref_hash=str(
                source.get("egress_proxy_ref_hash") or DIRECT_EGRESS_REF_HASH
            ),
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
        credential_routes = [
            route
            for route in ("openrouter_management", "openrouter")
            if self._broker.credential_available(
                job_id=str(normalized["job_id"]),
                slot=route,
            )
        ]
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
        if request_fields not in {
            frozenset(_REQUEST_FIELDS),
            frozenset(_REQUEST_FIELDS | _OPTIONAL_REQUEST_FIELDS),
        }:
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
    ) -> ProviderCostLedger:
        scope = _header(headers, "X-Research-Lab-Cost-Scope").strip() or "unscoped"
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
        self._tree_live_calls.clear()
        self._cache_day = day
