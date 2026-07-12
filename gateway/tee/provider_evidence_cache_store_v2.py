"""Encrypted, append-only daily provider-evidence cache for the coordinator.

The parent and PostgREST see only authenticated ciphertext plus content hashes.
The persistent artifact key is released only to approved coordinator PCR0s, so
an authenticated cache envelope can be reopened after a coordinator restart
without turning the parent database into a provider-evidence authority.
"""

from __future__ import annotations

import base64
from datetime import date
import json
import re
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlencode

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_broker_v2 import (
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
)
from gateway.tee.provider_evidence_v2 import (
    validate_signed_provider_evidence_record,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_transport_attempt,
)


CACHE_PAYLOAD_SCHEMA_VERSION = "leadpoet.provider_evidence_cache_payload.v2"
CACHE_ROW_SCHEMA_VERSION = "leadpoet.provider_evidence_cache_row.v2"
CACHE_TABLE = "research_lab_provider_evidence_cache_v2"
CACHE_ORIGIN = "https://qplwoislplkcegvdmbim.supabase.co"
CACHE_TIMEOUT_MS = 45_000
MAX_CACHE_RESPONSE_BYTES = 64 * 1024 * 1024
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ProviderEvidenceCacheStoreV2Error(RuntimeError):
    """A cache lookup or write was unauthenticated, altered, or ambiguous."""


def _day(value: Any) -> str:
    normalized = str(value or "")
    if not _DAY_RE.fullmatch(normalized):
        raise ProviderEvidenceCacheStoreV2Error("provider cache UTC day is invalid")
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise ProviderEvidenceCacheStoreV2Error(
            "provider cache UTC day is invalid"
        ) from exc
    if parsed.isoformat() != normalized:
        raise ProviderEvidenceCacheStoreV2Error("provider cache UTC day is invalid")
    return normalized


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ProviderEvidenceCacheStoreV2Error("%s is invalid" % field)
    return normalized


def _fingerprint(value: Any) -> str:
    normalized = str(value or "").lower()
    if not _FINGERPRINT_RE.fullmatch(normalized):
        raise ProviderEvidenceCacheStoreV2Error(
            "provider cache request fingerprint is invalid"
        )
    return normalized


def _decode_body(value: Any) -> bytes:
    try:
        body = base64.b64decode(str(value or ""), validate=True)
    except Exception as exc:
        raise ProviderEvidenceCacheStoreV2Error(
            "provider cache response body is invalid"
        ) from exc
    if len(body) > MAX_CACHE_RESPONSE_BYTES:
        raise ProviderEvidenceCacheStoreV2Error(
            "provider cache response body exceeds limit"
        )
    return body


class ProviderEvidenceCacheStoreV2:
    """Persist and reopen record-once provider evidence through measured TLS."""

    def __init__(
        self,
        *,
        broker: ProviderBrokerV2,
        vault: EncryptedArtifactVaultV2,
        source_boot_verifier: Callable[[Mapping[str, Any]], Any],
    ) -> None:
        self._broker = broker
        self._vault = vault
        self._source_boot_verifier = source_boot_verifier
        retry_hash = str(broker.retry_policy_hashes.get("supabase") or "")
        self._retry_policy_hash = _hash(
            retry_hash,
            "provider cache Supabase retry policy hash",
        )

    def persist_recorded(
        self,
        terminal: Mapping[str, Any],
        *,
        utc_day: str,
        job_id: str,
        purpose: str,
    ) -> Dict[str, Any]:
        """Insert once and authenticate an exact readback before acceptance."""

        normalized_day = _day(utc_day)
        payload = self._cache_payload(terminal, utc_day=normalized_day)
        payload_bytes = canonical_json(payload).encode("utf-8")
        cache_entry_hash = sha256_bytes(payload_bytes)
        cache_job_id = str(job_id or "")
        cache_purpose = str(purpose or "")
        if not cache_job_id or not cache_purpose:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache execution scope is invalid"
            )
        descriptor = self._vault.seal(
            payload_bytes,
            job_id=cache_job_id,
            purpose=cache_purpose,
            artifact_kind="provider_evidence_cache_entry",
        )
        exported = self._vault.export_ciphertext(descriptor["artifact_id"])
        row = self._row(
            payload=payload,
            cache_entry_hash=cache_entry_hash,
            cache_artifact_id=str(descriptor["artifact_id"]),
            storage_document=exported["storage_document"],
        )
        attempts = []
        artifacts = {
            cache_entry_hash,
            str(descriptor["artifact_id"]),
            str(exported["storage_document_hash"]),
        }
        post_result = self._execute(
            method="POST",
            url="%s/rest/v1/%s" % (CACHE_ORIGIN, CACHE_TABLE),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "prefer": "resolution=ignore-duplicates,return=minimal",
            },
            body=canonical_json(row).encode("utf-8"),
            logical_operation_id=(
                "%s:provider-evidence-cache:%s:insert"
                % (cache_job_id, payload["request_fingerprint"][:16])
            ),
            job_id=cache_job_id,
            purpose=cache_purpose,
        )
        self._collect_broker_evidence(post_result, attempts, artifacts)

        rows, read_attempts, read_artifacts = self._read_rows(
            utc_day=normalized_day,
            request_fingerprint=payload["request_fingerprint"],
            job_id=cache_job_id,
            purpose=cache_purpose,
            operation_suffix="readback",
        )
        attempts.extend(read_attempts)
        artifacts.update(read_artifacts)
        if len(rows) != 1 or rows[0] != row:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache durable readback differs"
            )
        self._vault.release_transient(str(descriptor["artifact_id"]))
        return {
            "cache_entry_hash": cache_entry_hash,
            "cache_artifact_id": str(descriptor["artifact_id"]),
            "transport_attempts": attempts,
            "evidence_artifact_hashes": sorted(artifacts),
        }

    def load(
        self,
        *,
        utc_day: str,
        request_fingerprint: str,
        job_id: str,
        purpose: str,
    ) -> Dict[str, Any]:
        """Read one exact daily entry and reopen it inside the coordinator."""

        normalized_day = _day(utc_day)
        normalized_fingerprint = _fingerprint(request_fingerprint)
        rows, attempts, artifacts = self._read_rows(
            utc_day=normalized_day,
            request_fingerprint=normalized_fingerprint,
            job_id=job_id,
            purpose=purpose,
            operation_suffix="lookup",
        )
        if not rows:
            return {
                "found": False,
                "transport_attempts": attempts,
                "evidence_artifact_hashes": sorted(artifacts),
            }
        if len(rows) != 1:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache lookup is ambiguous"
            )
        row = self._validate_row(rows[0])
        if (
            row["utc_day"] != normalized_day
            or row["request_fingerprint"] != normalized_fingerprint
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache lookup scope differs"
            )
        plaintext = self._vault.decrypt_storage_document(
            row["encrypted_cache_doc"]
        )
        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache plaintext is invalid"
            ) from exc
        if canonical_json(payload).encode("utf-8") != plaintext:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache plaintext is not canonical"
            )
        normalized_payload = self._validate_payload(payload)
        if (
            normalized_payload["utc_day"] != normalized_day
            or normalized_payload["request_fingerprint"]
            != normalized_fingerprint
            or sha256_bytes(plaintext) != row["cache_entry_hash"]
            or normalized_payload["source_record"]["record_hash"]
            != row["source_record_hash"]
            or normalized_payload["source_boot_identity"]["boot_identity_hash"]
            != row["source_boot_identity_hash"]
            or normalized_payload["source_record"]["body_hash"]
            != row["response_body_hash"]
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache row commitments differ"
            )
        aad = self._storage_aad(row["encrypted_cache_doc"])
        if (
            aad.get("boot_identity_hash") != row["source_boot_identity_hash"]
            or aad.get("artifact_kind") != "provider_evidence_cache_entry"
            or aad.get("plaintext_hash") != row["cache_entry_hash"]
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache encryption context differs"
            )
        artifacts.update(
            {
                row["cache_entry_hash"],
                row["cache_artifact_id"],
                row["source_record_hash"],
                row["source_boot_identity_hash"],
            }
        )
        artifacts.update(normalized_payload["source_evidence_artifact_hashes"])
        return {
            "found": True,
            "payload": normalized_payload,
            "cache_entry_hash": row["cache_entry_hash"],
            "cache_artifact_id": row["cache_artifact_id"],
            "transport_attempts": attempts,
            "evidence_artifact_hashes": sorted(artifacts),
        }

    def _cache_payload(
        self,
        terminal: Mapping[str, Any],
        *,
        utc_day: str,
    ) -> Dict[str, Any]:
        if not isinstance(terminal, Mapping) or terminal.get("evidence") != "recorded":
            raise ProviderEvidenceCacheStoreV2Error(
                "only recorded provider evidence can enter the daily cache"
            )
        if terminal.get("source_record") is not None:
            raise ProviderEvidenceCacheStoreV2Error(
                "recorded provider evidence unexpectedly has cache ancestry"
            )
        record = terminal.get("record")
        boot = terminal.get("coordinator_boot_identity")
        artifacts = terminal.get("evidence_artifact_hashes")
        source_attempts = terminal.get("transport_attempts")
        if not isinstance(record, Mapping) or not isinstance(boot, Mapping):
            raise ProviderEvidenceCacheStoreV2Error(
                "recorded provider evidence identity is missing"
            )
        if not isinstance(artifacts, list):
            raise ProviderEvidenceCacheStoreV2Error(
                "recorded provider evidence artifacts are invalid"
            )
        if not isinstance(source_attempts, list) or len(source_attempts) != 1:
            raise ProviderEvidenceCacheStoreV2Error(
                "recorded provider evidence terminal is missing"
            )
        source_attempt = dict(source_attempts[0])
        validate_transport_attempt(source_attempt)
        self._source_boot_verifier(boot)
        normalized_record = validate_signed_provider_evidence_record(
            record,
            boot_identity=boot,
        )
        if (
            normalized_record["evidence"] != "recorded"
            or normalized_record["source_record_hash"]
            or normalized_record["record_hash"] not in artifacts
            or source_attempt["attempt_hash"]
            != normalized_record["transport_attempt_hash"]
            or source_attempt["terminal_status"] != "authenticated_response"
            or source_attempt["http_status"] != normalized_record["status"]
            or source_attempt["response_hash"] != normalized_record["body_hash"]
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "recorded provider evidence is not a cache source"
            )
        body = _decode_body(terminal.get("body_b64"))
        if (
            sha256_bytes(body) != normalized_record["body_hash"]
            or int(terminal.get("status")) != normalized_record["status"]
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "recorded provider response differs from signed source"
            )
        normalized_artifacts = sorted(
            {_hash(item, "provider cache source artifact") for item in artifacts}
        )
        return self._validate_payload(
            {
                "schema_version": CACHE_PAYLOAD_SCHEMA_VERSION,
                "utc_day": utc_day,
                "request_fingerprint": normalized_record["request_fingerprint"],
                "status": normalized_record["status"],
                "body_b64": base64.b64encode(body).decode("ascii"),
                "source_record": dict(normalized_record),
                "source_boot_identity": dict(boot),
                "source_transport_attempt": source_attempt,
                "source_evidence_artifact_hashes": normalized_artifacts,
            }
        )

    def _validate_payload(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "schema_version",
            "utc_day",
            "request_fingerprint",
            "status",
            "body_b64",
            "source_record",
            "source_boot_identity",
            "source_transport_attempt",
            "source_evidence_artifact_hashes",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache payload fields are invalid"
            )
        if value.get("schema_version") != CACHE_PAYLOAD_SCHEMA_VERSION:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache payload schema is invalid"
            )
        utc_day = _day(value.get("utc_day"))
        fingerprint = _fingerprint(value.get("request_fingerprint"))
        status = value.get("status")
        if not isinstance(status, int) or isinstance(status, bool) or not 0 <= status <= 599:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache status is invalid"
            )
        body = _decode_body(value.get("body_b64"))
        boot = value.get("source_boot_identity")
        record = value.get("source_record")
        artifacts = value.get("source_evidence_artifact_hashes")
        source_attempt = value.get("source_transport_attempt")
        if not isinstance(boot, Mapping) or not isinstance(record, Mapping):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache source identity is invalid"
            )
        if not isinstance(artifacts, list):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache source artifacts are invalid"
            )
        if not isinstance(source_attempt, Mapping):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache source terminal is invalid"
            )
        validate_transport_attempt(source_attempt)
        self._source_boot_verifier(boot)
        normalized_record = validate_signed_provider_evidence_record(
            record,
            boot_identity=boot,
        )
        normalized_artifacts = sorted(
            {_hash(item, "provider cache source artifact") for item in artifacts}
        )
        if (
            normalized_record["evidence"] != "recorded"
            or normalized_record["source_record_hash"]
            or normalized_record["request_fingerprint"] != fingerprint
            or normalized_record["status"] != status
            or normalized_record["body_hash"] != sha256_bytes(body)
            or normalized_record["record_hash"] not in normalized_artifacts
            or source_attempt["attempt_hash"]
            != normalized_record["transport_attempt_hash"]
            or source_attempt["terminal_status"] != "authenticated_response"
            or source_attempt["http_status"] != status
            or source_attempt["response_hash"] != sha256_bytes(body)
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache source commitments differ"
            )
        return {
            "schema_version": CACHE_PAYLOAD_SCHEMA_VERSION,
            "utc_day": utc_day,
            "request_fingerprint": fingerprint,
            "status": status,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "source_record": normalized_record,
            "source_boot_identity": dict(boot),
            "source_transport_attempt": dict(source_attempt),
            "source_evidence_artifact_hashes": normalized_artifacts,
        }

    def _row(
        self,
        *,
        payload: Mapping[str, Any],
        cache_entry_hash: str,
        cache_artifact_id: str,
        storage_document: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self._validate_row(
            {
                "schema_version": CACHE_ROW_SCHEMA_VERSION,
                "utc_day": payload["utc_day"],
                "request_fingerprint": payload["request_fingerprint"],
                "cache_entry_hash": cache_entry_hash,
                "cache_artifact_id": cache_artifact_id,
                "source_record_hash": payload["source_record"]["record_hash"],
                "source_boot_identity_hash": payload["source_boot_identity"][
                    "boot_identity_hash"
                ],
                "response_body_hash": payload["source_record"]["body_hash"],
                "encrypted_cache_doc": dict(storage_document),
            }
        )

    def _validate_row(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "schema_version",
            "utc_day",
            "request_fingerprint",
            "cache_entry_hash",
            "cache_artifact_id",
            "source_record_hash",
            "source_boot_identity_hash",
            "response_body_hash",
            "encrypted_cache_doc",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache row fields are invalid"
            )
        if value.get("schema_version") != CACHE_ROW_SCHEMA_VERSION:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache row schema is invalid"
            )
        storage_document = value.get("encrypted_cache_doc")
        if not isinstance(storage_document, Mapping):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache encrypted document is invalid"
            )
        return {
            "schema_version": CACHE_ROW_SCHEMA_VERSION,
            "utc_day": _day(value.get("utc_day")),
            "request_fingerprint": _fingerprint(value.get("request_fingerprint")),
            "cache_entry_hash": _hash(value.get("cache_entry_hash"), "cache_entry_hash"),
            "cache_artifact_id": _hash(value.get("cache_artifact_id"), "cache_artifact_id"),
            "source_record_hash": _hash(value.get("source_record_hash"), "source_record_hash"),
            "source_boot_identity_hash": _hash(
                value.get("source_boot_identity_hash"),
                "source_boot_identity_hash",
            ),
            "response_body_hash": _hash(value.get("response_body_hash"), "response_body_hash"),
            "encrypted_cache_doc": dict(storage_document),
        }

    def _read_rows(
        self,
        *,
        utc_day: str,
        request_fingerprint: str,
        job_id: str,
        purpose: str,
        operation_suffix: str,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], set[str]]:
        query = urlencode(
            [
                (
                    "select",
                    "schema_version,utc_day,request_fingerprint,cache_entry_hash,"
                    "cache_artifact_id,source_record_hash,source_boot_identity_hash,"
                    "response_body_hash,encrypted_cache_doc",
                ),
                ("utc_day", "eq.%s" % _day(utc_day)),
                (
                    "request_fingerprint",
                    "eq.%s" % _fingerprint(request_fingerprint),
                ),
                ("limit", "2"),
            ]
        )
        result = self._execute(
            method="GET",
            url="%s/rest/v1/%s?%s" % (CACHE_ORIGIN, CACHE_TABLE, query),
            headers={"accept": "application/json"},
            body=b"",
            logical_operation_id=(
                "%s:provider-evidence-cache:%s:%s"
                % (job_id, request_fingerprint[:16], operation_suffix)
            ),
            job_id=job_id,
            purpose=purpose,
        )
        attempts = []
        artifacts = set()
        self._collect_broker_evidence(result, attempts, artifacts)
        if (
            result.get("terminal_status") != "authenticated_response"
            or not 200 <= int(result.get("http_status") or 0) < 300
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache authenticated read failed"
            )
        try:
            body = base64.b64decode(str(result.get("body_b64") or ""), validate=True)
            parsed = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache read response is invalid"
            ) from exc
        attempt = result["transport_attempt"]
        if sha256_bytes(body) != attempt.get("response_hash"):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache read response hash differs"
            )
        if not isinstance(parsed, list) or any(
            not isinstance(item, Mapping) for item in parsed
        ):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache read response is not a row array"
            )
        return [dict(item) for item in parsed], attempts, artifacts

    def _execute(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        logical_operation_id: str,
        job_id: str,
        purpose: str,
    ) -> Dict[str, Any]:
        return dict(
            self._broker.execute(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": logical_operation_id,
                    "job_id": str(job_id),
                    "purpose": str(purpose),
                    "provider_id": "supabase",
                    "attempt_number": 0,
                    "method": method,
                    "url": url,
                    "headers": dict(headers),
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "timeout_ms": CACHE_TIMEOUT_MS,
                    "retry_policy_hash": self._retry_policy_hash,
                }
            )
        )

    @staticmethod
    def _collect_broker_evidence(
        result: Mapping[str, Any],
        attempts: list[Dict[str, Any]],
        artifacts: set[str],
    ) -> None:
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache terminal attempt is missing"
            )
        attempts.append(dict(attempt))
        for field in ("encrypted_request_artifact_id", "encrypted_artifact_id"):
            value = str(result.get(field) or "")
            if value:
                artifacts.add(_hash(value, field))
        evidence_hashes = result.get("evidence_artifact_hashes") or []
        if not isinstance(evidence_hashes, list):
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache broker artifacts are invalid"
            )
        artifacts.update(
            _hash(value, "provider cache broker artifact")
            for value in evidence_hashes
        )

    @staticmethod
    def _storage_aad(storage_document: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            aad = base64.b64decode(
                str(storage_document.get("aad_b64") or ""),
                validate=True,
            )
            parsed = json.loads(aad.decode("utf-8"))
        except Exception as exc:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache encryption context is invalid"
            ) from exc
        if not isinstance(parsed, Mapping) or canonical_json(dict(parsed)).encode(
            "utf-8"
        ) != aad:
            raise ProviderEvidenceCacheStoreV2Error(
                "provider cache encryption context is not canonical"
            )
        return dict(parsed)
