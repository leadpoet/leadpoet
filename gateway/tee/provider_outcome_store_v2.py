"""Encrypted append-only restart checkpoints for measured provider outcomes."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, Mapping
from urllib.parse import urlencode

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_broker_v2 import (
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
)
from gateway.tee.provider_outcome_v2 import (
    validate_provider_outcome_state_document_v2,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


CHECKPOINT_SCHEMA_VERSION = "leadpoet.provider_outcome_checkpoint.v2"
CHECKPOINT_ROW_SCHEMA_VERSION = "leadpoet.provider_outcome_checkpoint_row.v2"
CHECKPOINT_TABLE = "research_lab_provider_outcome_checkpoints_v2"
CHECKPOINT_ORIGIN = "https://qplwoislplkcegvdmbim.supabase.co"
CHECKPOINT_TIMEOUT_MS = 45_000
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ProviderOutcomeStoreV2Error(RuntimeError):
    """A provider-outcome checkpoint is missing, altered, or ambiguous."""


def _hash(value: Any, field: str, *, optional: bool = False) -> str:
    normalized = str(value or "").lower()
    if optional and not normalized:
        return ""
    if not _HASH_RE.fullmatch(normalized):
        raise ProviderOutcomeStoreV2Error("%s is invalid" % field)
    return normalized


def _day(value: Any) -> str:
    normalized = str(value or "")
    if not _DAY_RE.fullmatch(normalized):
        raise ProviderOutcomeStoreV2Error("provider outcome checkpoint day is invalid")
    return normalized


class ProviderOutcomeStoreV2:
    """Persist every accepted aggregate transition before returning it to a job."""

    def __init__(
        self,
        *,
        broker: ProviderBrokerV2,
        vault: EncryptedArtifactVaultV2,
    ) -> None:
        self._broker = broker
        self._vault = vault
        self._retry_policy_hash = _hash(
            broker.retry_policy_hashes.get("supabase"),
            "provider outcome Supabase retry policy hash",
        )

    def persist(
        self,
        document: Mapping[str, Any],
        *,
        previous_checkpoint_hash: str,
        job_id: str,
        purpose: str,
    ) -> Dict[str, Any]:
        utc_day = _day(document.get("utc_day"))
        normalized_document = validate_provider_outcome_state_document_v2(
            document,
            expected_utc_day=utc_day,
        )
        sequence = int(normalized_document["sequence"])
        if sequence <= 0:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint sequence must be positive"
            )
        previous_hash = _hash(
            previous_checkpoint_hash,
            "previous provider outcome checkpoint hash",
            optional=True,
        )
        payload_body = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "utc_day": utc_day,
            "sequence": sequence,
            "previous_checkpoint_hash": previous_hash,
            "state_document": normalized_document,
        }
        payload = {
            **payload_body,
            "checkpoint_hash": sha256_json(payload_body),
        }
        payload_bytes = canonical_json(payload).encode("utf-8")
        descriptor = self._vault.seal(
            payload_bytes,
            job_id=str(job_id),
            purpose=str(purpose),
            artifact_kind="provider_outcome_checkpoint",
        )
        exported = self._vault.export_ciphertext(descriptor["artifact_id"])
        row = self._row(
            payload=payload,
            artifact_id=str(descriptor["artifact_id"]),
            storage_document=exported["storage_document"],
        )
        attempts: list[Dict[str, Any]] = []
        artifacts = {
            payload["checkpoint_hash"],
            normalized_document["document_hash"],
            str(descriptor["artifact_id"]),
            str(exported["storage_document_hash"]),
        }
        result = self._execute(
            method="POST",
            url="%s/rest/v1/%s" % (CHECKPOINT_ORIGIN, CHECKPOINT_TABLE),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "prefer": "resolution=ignore-duplicates,return=minimal",
            },
            body=canonical_json(row).encode("utf-8"),
            logical_operation_id="%s:provider-outcome:%d:insert"
            % (job_id, sequence),
            job_id=job_id,
            purpose=purpose,
        )
        self._collect_evidence(result, attempts, artifacts)
        rows, read_attempts, read_artifacts = self._read_rows(
            utc_day=utc_day,
            sequence=sequence,
            order_latest=False,
            job_id=job_id,
            purpose=purpose,
            operation_suffix="readback",
        )
        attempts.extend(read_attempts)
        artifacts.update(read_artifacts)
        if len(rows) != 1 or rows[0] != row:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint durable readback differs"
            )
        self._vault.release_transient(str(descriptor["artifact_id"]))
        return {
            "checkpoint_hash": payload["checkpoint_hash"],
            "state_document_hash": normalized_document["document_hash"],
            "transport_attempts": attempts,
            "evidence_artifact_hashes": sorted(artifacts),
        }

    def load_latest(
        self,
        *,
        utc_day: str,
        job_id: str,
        purpose: str,
    ) -> Dict[str, Any]:
        normalized_day = _day(utc_day)
        rows, attempts, artifacts = self._read_rows(
            utc_day=normalized_day,
            sequence=None,
            order_latest=True,
            job_id=job_id,
            purpose=purpose,
            operation_suffix="restore",
        )
        if not rows:
            return {
                "found": False,
                "transport_attempts": attempts,
                "evidence_artifact_hashes": sorted(artifacts),
            }
        if len(rows) != 1:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome latest checkpoint is ambiguous"
            )
        row = self._validate_row(rows[0])
        plaintext = self._vault.decrypt_storage_document(
            row["encrypted_checkpoint_doc"]
        )
        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint plaintext is invalid"
            ) from exc
        if canonical_json(payload).encode("utf-8") != plaintext:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint plaintext is not canonical"
            )
        normalized_payload = self._validate_payload(payload)
        if (
            normalized_payload["utc_day"] != normalized_day
            or normalized_payload["sequence"] != row["sequence"]
            or normalized_payload["checkpoint_hash"] != row["checkpoint_hash"]
            or normalized_payload["previous_checkpoint_hash"]
            != row["previous_checkpoint_hash"]
            or normalized_payload["state_document"]["document_hash"]
            != row["state_document_hash"]
            or sha256_bytes(plaintext)
            != str(row["encrypted_checkpoint_doc"].get("plaintext_hash") or "")
        ):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint row commitments differ"
            )
        aad = self._storage_aad(row["encrypted_checkpoint_doc"])
        if (
            aad.get("artifact_kind") != "provider_outcome_checkpoint"
            or aad.get("plaintext_hash") != sha256_bytes(plaintext)
        ):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint encryption context differs"
            )
        artifacts.update(
            {
                row["checkpoint_hash"],
                row["state_document_hash"],
                row["checkpoint_artifact_id"],
            }
        )
        return {
            "found": True,
            "checkpoint_hash": row["checkpoint_hash"],
            "state_document": normalized_payload["state_document"],
            "transport_attempts": attempts,
            "evidence_artifact_hashes": sorted(artifacts),
        }

    def _validate_payload(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "schema_version",
            "utc_day",
            "sequence",
            "previous_checkpoint_hash",
            "state_document",
            "checkpoint_hash",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint fields are invalid"
            )
        if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint schema is invalid"
            )
        utc_day = _day(value.get("utc_day"))
        try:
            sequence = int(value.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint sequence is invalid"
            ) from exc
        if sequence <= 0:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint sequence is invalid"
            )
        state_document = validate_provider_outcome_state_document_v2(
            value.get("state_document"),
            expected_utc_day=utc_day,
        )
        if int(state_document["sequence"]) != sequence:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint state sequence differs"
            )
        previous_hash = _hash(
            value.get("previous_checkpoint_hash"),
            "previous provider outcome checkpoint hash",
            optional=True,
        )
        body = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "utc_day": utc_day,
            "sequence": sequence,
            "previous_checkpoint_hash": previous_hash,
            "state_document": state_document,
        }
        checkpoint_hash = _hash(value.get("checkpoint_hash"), "checkpoint_hash")
        if checkpoint_hash != sha256_json(body):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint hash differs"
            )
        return {**body, "checkpoint_hash": checkpoint_hash}

    def _row(
        self,
        *,
        payload: Mapping[str, Any],
        artifact_id: str,
        storage_document: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self._validate_row(
            {
                "schema_version": CHECKPOINT_ROW_SCHEMA_VERSION,
                "utc_day": payload["utc_day"],
                "sequence": payload["sequence"],
                "checkpoint_hash": payload["checkpoint_hash"],
                "previous_checkpoint_hash": payload["previous_checkpoint_hash"],
                "state_document_hash": payload["state_document"]["document_hash"],
                "checkpoint_artifact_id": artifact_id,
                "encrypted_checkpoint_doc": dict(storage_document),
            }
        )

    def _validate_row(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "schema_version",
            "utc_day",
            "sequence",
            "checkpoint_hash",
            "previous_checkpoint_hash",
            "state_document_hash",
            "checkpoint_artifact_id",
            "encrypted_checkpoint_doc",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint row fields are invalid"
            )
        if value.get("schema_version") != CHECKPOINT_ROW_SCHEMA_VERSION:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint row schema is invalid"
            )
        try:
            sequence = int(value.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint row sequence is invalid"
            ) from exc
        encrypted = value.get("encrypted_checkpoint_doc")
        if sequence <= 0 or not isinstance(encrypted, Mapping):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint row is invalid"
            )
        return {
            "schema_version": CHECKPOINT_ROW_SCHEMA_VERSION,
            "utc_day": _day(value.get("utc_day")),
            "sequence": sequence,
            "checkpoint_hash": _hash(value.get("checkpoint_hash"), "checkpoint_hash"),
            "previous_checkpoint_hash": _hash(
                value.get("previous_checkpoint_hash"),
                "previous checkpoint hash",
                optional=True,
            ),
            "state_document_hash": _hash(
                value.get("state_document_hash"), "state document hash"
            ),
            "checkpoint_artifact_id": _hash(
                value.get("checkpoint_artifact_id"), "checkpoint artifact ID"
            ),
            "encrypted_checkpoint_doc": dict(encrypted),
        }

    def _read_rows(
        self,
        *,
        utc_day: str,
        sequence: int | None,
        order_latest: bool,
        job_id: str,
        purpose: str,
        operation_suffix: str,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], set[str]]:
        query_items = [
            (
                "select",
                "schema_version,utc_day,sequence,checkpoint_hash,"
                "previous_checkpoint_hash,state_document_hash,"
                "checkpoint_artifact_id,encrypted_checkpoint_doc",
            ),
            ("utc_day", "eq.%s" % _day(utc_day)),
        ]
        if sequence is not None:
            query_items.append(("sequence", "eq.%d" % int(sequence)))
        if order_latest:
            query_items.append(("order", "sequence.desc"))
        query_items.append(("limit", "1" if order_latest else "2"))
        result = self._execute(
            method="GET",
            url="%s/rest/v1/%s?%s"
            % (CHECKPOINT_ORIGIN, CHECKPOINT_TABLE, urlencode(query_items)),
            headers={"accept": "application/json"},
            body=b"",
            logical_operation_id="%s:provider-outcome:%s" % (job_id, operation_suffix),
            job_id=job_id,
            purpose=purpose,
        )
        attempts: list[Dict[str, Any]] = []
        artifacts: set[str] = set()
        self._collect_evidence(result, attempts, artifacts)
        if (
            result.get("terminal_status") != "authenticated_response"
            or not 200 <= int(result.get("http_status") or 0) < 300
        ):
            attempt = result.get("transport_attempt")
            failure_code = (
                str(attempt.get("failure_code") or "none")
                if isinstance(attempt, Mapping)
                else "missing_attempt"
            )
            failure_stage = str(result.get("failure_stage") or "unknown")
            failure_error_type = str(
                result.get("failure_error_type") or "unknown"
            )
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint authenticated read failed "
                "(terminal_status=%s http_status=%d failure_code=%s "
                "failure_stage=%s failure_error_type=%s)"
                % (
                    str(result.get("terminal_status") or "missing"),
                    int(result.get("http_status") or 0),
                    failure_code,
                    failure_stage,
                    failure_error_type,
                )
            )
        try:
            body = base64.b64decode(str(result.get("body_b64") or ""), validate=True)
            parsed = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint response is invalid"
            ) from exc
        if (
            sha256_bytes(body)
            != str(result["transport_attempt"].get("response_hash") or "")
            or not isinstance(parsed, list)
            or any(not isinstance(item, Mapping) for item in parsed)
        ):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint response commitments differ"
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
                    "timeout_ms": CHECKPOINT_TIMEOUT_MS,
                    "retry_policy_hash": self._retry_policy_hash,
                }
            )
        )

    @staticmethod
    def _collect_evidence(
        result: Mapping[str, Any],
        attempts: list[Dict[str, Any]],
        artifacts: set[str],
    ) -> None:
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint terminal is missing"
            )
        attempts.append(dict(attempt))
        for field in ("encrypted_request_artifact_id", "encrypted_artifact_id"):
            value = str(result.get(field) or "")
            if value:
                artifacts.add(_hash(value, field))
        evidence = result.get("evidence_artifact_hashes") or []
        if not isinstance(evidence, list):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint artifacts are invalid"
            )
        artifacts.update(_hash(item, "checkpoint artifact") for item in evidence)

    @staticmethod
    def _storage_aad(storage_document: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            encoded = base64.b64decode(
                str(storage_document.get("aad_b64") or ""), validate=True
            )
            parsed = json.loads(encoded.decode("utf-8"))
        except Exception as exc:
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint encryption context is invalid"
            ) from exc
        if (
            not isinstance(parsed, Mapping)
            or canonical_json(dict(parsed)).encode("utf-8") != encoded
        ):
            raise ProviderOutcomeStoreV2Error(
                "provider outcome checkpoint encryption context is not canonical"
            )
        return dict(parsed)
