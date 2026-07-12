"""Measured in-memory replacement for the legacy provider-outcome sidecar."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
import threading
from typing import Any, Dict, Mapping

from gateway.research_lab.provider_outcome_digest import (
    PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION,
    _digest_from_sidecar,
    _empty_outcome_bucket,
    _empty_sidecar_doc,
    _normalize_outcome_bucket,
    _record_outcome_bucket,
    _safe_endpoint_bucket,
    _safe_evidence_bucket,
    _safe_provider_bucket,
    _sidecar_document_hash,
    MAX_PROVIDER_OUTCOME_ENDPOINTS,
    MAX_PROVIDER_OUTCOME_EVIDENCE,
    MAX_PROVIDER_OUTCOME_STATUSES,
)
from leadpoet_canonical.attested_v2 import sha256_json


SNAPSHOT_SCHEMA_VERSION = "leadpoet.provider_outcome_snapshot.v2"
_MAX_PROVIDERS = 12
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DIGEST_FIELDS = {
    "schema_version",
    "utc_day",
    "source",
    "sidecar_sequence",
    "sidecar_document_hash",
    "generated_at",
    "providers",
    "top_error_classes",
    "day_cache_outcomes",
    "aggregate_spend",
    "digest_hash",
}
_MEASURED_PROVIDER_COST_SOURCES = frozenset(
    {
        "exa_cost_dollars",
        "openrouter_response_usage",
        "openrouter_generation_reconciliation",
        "deepline_response_cost",
    }
)
_STATE_FIELDS = {
    "schema_version",
    "utc_day",
    "generated_at",
    "generated_at_epoch",
    "sequence",
    "providers",
    "totals",
    "document_hash",
}
_BUCKET_FIELDS = {
    "call_count",
    "cache_hit_count",
    "live_call_count",
    "blocked_count",
    "quota_exhausted_count",
    "credential_missing_count",
    "replay_miss_count",
    "error_count",
    "measured_spend_microusd",
    "estimated_spend_microusd",
    "status_counts",
    "evidence_counts",
}


class ProviderOutcomeV2Error(RuntimeError):
    """The measured provider-outcome ledger cannot produce a safe snapshot."""


def validate_provider_outcome_snapshot_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the exact measured snapshot consumed by autoresearch."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "provider_outcome_digest",
        "provider_outcome_digest_hash",
        "source_state_hash",
    }:
        raise ProviderOutcomeV2Error("provider outcome snapshot fields are invalid")
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ProviderOutcomeV2Error("provider outcome snapshot schema is invalid")
    digest = value.get("provider_outcome_digest")
    if not isinstance(digest, Mapping) or set(digest) != _DIGEST_FIELDS:
        raise ProviderOutcomeV2Error("provider outcome digest fields are invalid")
    normalized_digest = dict(digest)
    digest_hash = str(normalized_digest.get("digest_hash") or "")
    source_state_hash = str(value.get("source_state_hash") or "")
    if (
        normalized_digest.get("schema_version") != "1.0"
        or normalized_digest.get("source") != "provider_outcome_sidecar"
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(normalized_digest.get("utc_day") or ""))
        or not isinstance(normalized_digest.get("sidecar_sequence"), int)
        or int(normalized_digest["sidecar_sequence"]) < 0
        or not isinstance(normalized_digest.get("providers"), Mapping)
        or not isinstance(normalized_digest.get("top_error_classes"), list)
        or not isinstance(normalized_digest.get("day_cache_outcomes"), Mapping)
        or not isinstance(normalized_digest.get("aggregate_spend"), Mapping)
        or not _HASH_RE.fullmatch(digest_hash)
        or not _HASH_RE.fullmatch(source_state_hash)
        or normalized_digest.get("sidecar_document_hash") != source_state_hash
        or value.get("provider_outcome_digest_hash") != digest_hash
        or digest_hash
        != sha256_json(
            {
                key: item
                for key, item in normalized_digest.items()
                if key != "digest_hash"
            }
        )
    ):
        raise ProviderOutcomeV2Error("provider outcome snapshot commitments differ")
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "provider_outcome_digest": normalized_digest,
        "provider_outcome_digest_hash": digest_hash,
        "source_state_hash": source_state_hash,
    }


def _timestamp_epoch(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except ValueError as exc:
        raise ProviderOutcomeV2Error("provider outcome clock is invalid") from exc


def validate_provider_outcome_state_document_v2(
    value: Mapping[str, Any],
    *,
    expected_utc_day: str,
) -> Dict[str, Any]:
    """Validate one exact encrypted restart checkpoint before restoring it."""

    if not isinstance(value, Mapping) or set(value) != _STATE_FIELDS:
        raise ProviderOutcomeV2Error("provider outcome state fields are invalid")
    utc_day = str(value.get("utc_day") or "")
    if (
        value.get("schema_version") != PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION
        or utc_day != str(expected_utc_day or "")
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", utc_day)
    ):
        raise ProviderOutcomeV2Error("provider outcome state scope is invalid")
    generated_at = str(value.get("generated_at") or "")
    generated_epoch = _timestamp_epoch(generated_at)
    try:
        stored_epoch = float(value.get("generated_at_epoch"))
        sequence = int(value.get("sequence"))
    except (TypeError, ValueError) as exc:
        raise ProviderOutcomeV2Error("provider outcome state counters are invalid") from exc
    if sequence < 0 or stored_epoch <= 0 or abs(stored_epoch - generated_epoch) >= 1.0:
        raise ProviderOutcomeV2Error("provider outcome state timestamp is invalid")
    providers = value.get("providers")
    totals = value.get("totals")
    if (
        not isinstance(providers, Mapping)
        or not isinstance(totals, Mapping)
        or len(providers) > _MAX_PROVIDERS
        or set(totals) != _BUCKET_FIELDS
    ):
        raise ProviderOutcomeV2Error("provider outcome state buckets are invalid")
    normalized_providers: Dict[str, Any] = {}
    for raw_provider, raw_bucket in providers.items():
        provider = _safe_provider_bucket(raw_provider)
        if provider != str(raw_provider) or not isinstance(raw_bucket, Mapping):
            raise ProviderOutcomeV2Error("provider outcome provider bucket is invalid")
        if set(raw_bucket) != _BUCKET_FIELDS | {"endpoints"}:
            raise ProviderOutcomeV2Error("provider outcome provider fields are invalid")
        endpoints = raw_bucket.get("endpoints")
        if not isinstance(endpoints, Mapping) or len(endpoints) > MAX_PROVIDER_OUTCOME_ENDPOINTS:
            raise ProviderOutcomeV2Error("provider outcome endpoint buckets are invalid")
        normalized_endpoints: Dict[str, Any] = {}
        for raw_endpoint, raw_endpoint_bucket in endpoints.items():
            endpoint = _safe_endpoint_bucket(raw_endpoint)
            if (
                endpoint != str(raw_endpoint)
                or not isinstance(raw_endpoint_bucket, Mapping)
                or set(raw_endpoint_bucket) != _BUCKET_FIELDS
            ):
                raise ProviderOutcomeV2Error("provider outcome endpoint bucket is invalid")
            normalized_endpoint = _normalize_outcome_bucket(raw_endpoint_bucket)
            if normalized_endpoint is None:
                raise ProviderOutcomeV2Error("provider outcome endpoint counts are invalid")
            normalized_endpoints[endpoint] = normalized_endpoint
        normalized_bucket = _normalize_outcome_bucket(raw_bucket)
        if normalized_bucket is None:
            raise ProviderOutcomeV2Error("provider outcome provider counts are invalid")
        normalized_bucket["endpoints"] = normalized_endpoints
        normalized_providers[provider] = normalized_bucket
    normalized_totals = _normalize_outcome_bucket(totals)
    if normalized_totals is None:
        raise ProviderOutcomeV2Error("provider outcome total counts are invalid")
    for bucket in [normalized_totals, *normalized_providers.values()]:
        if (
            len(bucket.get("status_counts") or {}) > MAX_PROVIDER_OUTCOME_STATUSES
            or len(bucket.get("evidence_counts") or {}) > MAX_PROVIDER_OUTCOME_EVIDENCE
        ):
            raise ProviderOutcomeV2Error("provider outcome counters exceed limits")
    normalized = {
        "schema_version": PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION,
        "utc_day": utc_day,
        "generated_at": generated_at,
        "generated_at_epoch": stored_epoch,
        "sequence": sequence,
        "providers": normalized_providers,
        "totals": normalized_totals,
    }
    normalized["document_hash"] = _sidecar_document_hash(normalized)
    if normalized["document_hash"] != str(value.get("document_hash") or ""):
        raise ProviderOutcomeV2Error("provider outcome state hash differs")
    return normalized


def _cost_microusd(cost_event: Mapping[str, Any], *, status: int) -> tuple[int, str]:
    if not cost_event or not bool(cost_event.get("billable")) or not 200 <= status < 300:
        return 0, "estimated"
    try:
        cost = Decimal(str(cost_event.get("cost_usd") or "0"))
    except (InvalidOperation, ValueError) as exc:
        raise ProviderOutcomeV2Error("provider outcome cost is invalid") from exc
    if cost < 0:
        raise ProviderOutcomeV2Error("provider outcome cost is negative")
    kind = (
        "measured"
        if str(cost_event.get("cost_source") or "")
        in _MEASURED_PROVIDER_COST_SOURCES
        else "estimated"
    )
    return int(cost * 1_000_000), kind


class ProviderOutcomeLedgerV2:
    """Thread-safe daily aggregates built only from coordinator terminals."""

    def __init__(
        self,
        *,
        clock: Any,
        initial_document: Mapping[str, Any] | None = None,
    ) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        timestamp = str(self._clock() or "")
        _timestamp_epoch(timestamp)
        if initial_document is None:
            self._doc = _empty_sidecar_doc(timestamp[:10])
            self._doc["generated_at"] = timestamp
            self._doc["generated_at_epoch"] = _timestamp_epoch(timestamp)
            self._doc["document_hash"] = _sidecar_document_hash(self._doc)
        else:
            self._doc = validate_provider_outcome_state_document_v2(
                initial_document,
                expected_utc_day=timestamp[:10],
            )

    def record(
        self,
        *,
        provider_id: str,
        endpoint_class: str,
        evidence: str,
        status: int,
        live_call: bool,
        cost_event: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        timestamp = str(self._clock() or "")
        epoch = _timestamp_epoch(timestamp)
        day = timestamp[:10]
        normalized_status = max(0, int(status))
        spend_microusd, spend_kind = _cost_microusd(
            dict(cost_event or {}),
            status=normalized_status,
        )
        with self._lock:
            if str(self._doc.get("utc_day") or "") != day:
                self._doc = _empty_sidecar_doc(day)
            providers = self._doc.setdefault("providers", {})
            provider = _safe_provider_bucket(provider_id)
            if provider not in providers and len(providers) >= max(1, _MAX_PROVIDERS - 1):
                provider = "other"
            provider_bucket = providers.setdefault(provider, _empty_outcome_bucket())
            endpoints = provider_bucket.setdefault("endpoints", {})
            endpoint = _safe_endpoint_bucket(endpoint_class)
            if endpoint not in endpoints and len(endpoints) >= max(
                1, MAX_PROVIDER_OUTCOME_ENDPOINTS - 1
            ):
                endpoint = "other"
            endpoint_bucket = endpoints.setdefault(endpoint, _empty_outcome_bucket())
            safe_evidence = _safe_evidence_bucket(evidence)
            for bucket in (
                provider_bucket,
                endpoint_bucket,
                self._doc.setdefault("totals", _empty_outcome_bucket()),
            ):
                _record_outcome_bucket(
                    bucket,
                    evidence=safe_evidence,
                    status=normalized_status,
                    live_call=bool(live_call),
                    spend_microusd=spend_microusd,
                    spend_kind=spend_kind,
                )
            self._doc["sequence"] = max(0, int(self._doc.get("sequence") or 0)) + 1
            self._doc["generated_at"] = timestamp
            self._doc["generated_at_epoch"] = epoch
            self._doc["document_hash"] = _sidecar_document_hash(self._doc)
            return deepcopy(self._doc)

    def state_document(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._doc)

    def snapshot(self) -> Dict[str, Any]:
        timestamp = str(self._clock() or "")
        epoch = _timestamp_epoch(timestamp)
        with self._lock:
            if str(self._doc.get("utc_day") or "") != timestamp[:10]:
                self._doc = _empty_sidecar_doc(timestamp[:10])
                self._doc["generated_at"] = timestamp
                self._doc["generated_at_epoch"] = epoch
                self._doc["document_hash"] = _sidecar_document_hash(self._doc)
            document = deepcopy(self._doc)
        digest = _digest_from_sidecar(document)
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "provider_outcome_digest": digest,
            "provider_outcome_digest_hash": str(digest["digest_hash"]),
            "source_state_hash": str(document["document_hash"]),
        }
