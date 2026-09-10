"""Measured coordinator entrypoints for unchanged Research Lab decisions."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, Optional

from gateway.tee.artifact_vault_v2 import (
    ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD,
    ARTIFACT_PERSISTENCE_RETRYABLE_HTTP_STATUSES,
)
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionJobV2Error,
    ExecutionResultV2,
)
from leadpoet_canonical.attested_v2 import sha256_json, transport_root, validate_transport_attempt

OP_ATTEST_ARTIFACT_PERSISTENCE = "attest_artifact_persistence"
OP_ATTEST_QUALIFICATION_ADMISSION = "attest_qualification_admission"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_PERSISTENCE_PURPOSE = "leadpoet.artifact_persistence.v2"
_ARTIFACT_PERSISTENCE_PROVIDER = "aws_s3_object_lock"
_PROVIDER_CREDENTIAL_REFS_FIELD = "_v2_provider_credential_ref_hashes"
_PROVIDER_CREDENTIAL_PROFILE_FIELD = "_v2_provider_credential_profile"
COORDINATOR_OPERATIONS_V2 = {
    OP_ATTEST_ARTIFACT_PERSISTENCE: frozenset({_ARTIFACT_PERSISTENCE_PURPOSE}),
    OP_ATTEST_QUALIFICATION_ADMISSION: frozenset({"research_lab.admission.v2"}),
}


def coordinator_receipt_output_v2(operation, output):
    return dict(output)


def _validated_artifact_persistence_attempts(
    *,
    artifact_id: str,
    attempts: Any,
    context: ExecutionContextV2,
    expected_transport_root: Any,
) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        raise ValueError("artifact persistence transport evidence is incomplete")
    normalized = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("artifact persistence transport evidence is invalid")
        validate_transport_attempt(attempt)
        normalized.append(dict(attempt))

    methods = [item["method"] for item in normalized]
    try:
        first_head = methods.index("HEAD")
    except ValueError:
        first_head = -1
    get_count = first_head
    head_count = len(normalized) - first_head if first_head >= 0 else 0
    maximum = ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD
    if (
        not 1 <= get_count <= maximum
        or not 1 <= head_count <= maximum
        or any(method != "GET" for method in methods[:first_head])
        or any(method != "HEAD" for method in methods[first_head:])
    ):
        raise ValueError("artifact persistence transport evidence is incomplete")

    stable_fields = (
        "destination_host",
        "destination_port",
        "path_hash",
        "nonsecret_headers_hash",
        "body_hash",
        "credential_ref_hash",
        "egress_proxy_ref_hash",
        "retry_policy_hash",
        "timeout_ms",
    )
    expected_stable = {
        field: normalized[0][field]
        for field in stable_fields
    }
    for ordinal, attempt in enumerate(normalized):
        method = attempt["method"]
        if (
            attempt["job_id"] != context.job_id
            or attempt["purpose"] != _ARTIFACT_PERSISTENCE_PURPOSE
            or attempt["provider_id"] != _ARTIFACT_PERSISTENCE_PROVIDER
            or attempt["attempt_number"] != ordinal
            or attempt["logical_operation_id"]
            != "%s:%s" % (artifact_id, method.lower())
            or any(
                attempt[field] != expected_stable[field]
                for field in stable_fields
            )
        ):
            raise ValueError("artifact persistence transport binding is invalid")

    for method_attempts in (
        normalized[:first_head],
        normalized[first_head:],
    ):
        for attempt in method_attempts[:-1]:
            if attempt["terminal_status"] == "transport_failure":
                continue
            if (
                attempt["terminal_status"] == "authenticated_response"
                and attempt["http_status"]
                in ARTIFACT_PERSISTENCE_RETRYABLE_HTTP_STATUSES
            ):
                continue
            raise ValueError("artifact persistence retry sequence is invalid")
        final_attempt = method_attempts[-1]
        if (
            final_attempt["terminal_status"] != "authenticated_response"
            or final_attempt["http_status"] != 200
        ):
            raise ValueError("artifact persistence transport evidence is incomplete")

    observed_root = transport_root(normalized)
    if observed_root != expected_transport_root:
        raise ValueError("artifact persistence transport root differs")
    return normalized

def coordinator_failed_parent_graph_policy_v2(
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[str, ...]:
    """Permit failed source ancestry only when persisting its artifacts."""

    root_hash = str(graph.get("root_receipt_hash") or "")
    receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
    }
    failed_hashes = {
        receipt_hash
        for receipt_hash, receipt in receipts.items()
        if receipt.get("status") == "failed"
    }
    if not failed_hashes:
        return ()
    root = receipts.get(root_hash)
    if not isinstance(root, Mapping):
        raise ValueError("failed receipt graph root is missing")

    operation = str(manifest.get("operation") or "")
    if operation == OP_ATTEST_ARTIFACT_PERSISTENCE:
        if str(payload.get("source_receipt_hash") or "") != root_hash:
            raise ValueError("artifact persistence failed source differs")
        return tuple(sorted(failed_hashes))

    raise ValueError("failed receipt ancestry is unauthorized for operation")


class CoordinatorExecutorV2:
    """Protect qualification admission and artifact persistence."""

    def __init__(self, *, artifact_evidence_supplier=None, qualification_admission_resolver=None):
        self._artifact_evidence_supplier = artifact_evidence_supplier
        self._qualification_admission_resolver = qualification_admission_resolver
    async def __call__(
        self,
        operation: str,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if operation not in COORDINATOR_OPERATIONS_V2:
            raise ValueError("unsupported V2 coordinator operation")
        payload = dict(payload)
        credential_profile = payload.pop(
            _PROVIDER_CREDENTIAL_PROFILE_FIELD,
            None,
        )
        if credential_profile is None:
            credential_profile = "default"
        if credential_profile != context.provider_credential_profile:
            raise ValueError(
                "V2 provider credential profile differs from job manifest"
            )
        if credential_profile != "default":
            raise ValueError(
                "V2 provider credential profile is not allowed for coordinator"
            )
        credential_refs = payload.pop(_PROVIDER_CREDENTIAL_REFS_FIELD, None)
        if credential_refs is None and context.provider_credential_ref_hashes:
            raise ValueError("V2 provider credential profile is missing")
        if credential_refs is None:
            credential_refs = {}
        if not isinstance(credential_refs, Mapping):
            raise ValueError("V2 provider credential profile is invalid")
        if dict(credential_refs) != dict(
            context.provider_credential_ref_hashes
        ):
            raise ValueError(
                "V2 provider credential profile differs from job manifest"
            )
        if operation == OP_ATTEST_ARTIFACT_PERSISTENCE:
            return self._attest_artifact_persistence(payload, context)
        if operation == OP_ATTEST_QUALIFICATION_ADMISSION:
            return self._attest_qualification_admission(payload, context)
        raise ValueError("unsupported V2 coordinator operation")

    def _attest_artifact_persistence(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if set(payload) != {
            "source_receipt_hash",
            "artifact_ids",
            "artifact_plaintext_hashes",
        }:
            raise ValueError("artifact persistence payload fields are invalid")
        source_receipt_hash = str(payload.get("source_receipt_hash") or "")
        artifact_ids = payload.get("artifact_ids")
        plaintext_hashes = payload.get("artifact_plaintext_hashes")
        if not _HASH_RE.fullmatch(source_receipt_hash):
            raise ValueError("artifact source receipt hash is invalid")
        if (
            not isinstance(artifact_ids, list)
            or not artifact_ids
            or len(set(artifact_ids)) != len(artifact_ids)
            or any(not _HASH_RE.fullmatch(str(item or "")) for item in artifact_ids)
        ):
            raise ValueError("artifact persistence IDs are invalid")
        if (
            not isinstance(plaintext_hashes, list)
            or len(plaintext_hashes) != len(artifact_ids)
            or any(not _HASH_RE.fullmatch(str(item or "")) for item in plaintext_hashes)
        ):
            raise ValueError("artifact plaintext commitments are invalid")
        if self._artifact_evidence_supplier is None:
            raise ValueError("artifact persistence evidence is unavailable")
        evidence = [
            dict(item)
            for item in self._artifact_evidence_supplier(artifact_ids, context)
        ]
        if sorted(item.get("artifact_id") for item in evidence) != sorted(artifact_ids):
            raise ValueError("artifact persistence evidence set differs")
        if sorted(item.get("plaintext_hash") for item in evidence) != sorted(
            plaintext_hashes
        ):
            raise ValueError("artifact plaintext commitments differ")
        if any(not item.get("persisted") for item in evidence):
            raise ValueError("artifact persistence evidence is incomplete")
        transport_attempts = []
        artifact_hashes = []
        output_artifacts = []
        for item in sorted(evidence, key=lambda value: value["artifact_id"]):
            attempts = _validated_artifact_persistence_attempts(
                artifact_id=item["artifact_id"],
                attempts=item.get("transport_attempts"),
                context=context,
                expected_transport_root=item.get("transport_root"),
            )
            transport_attempts.extend(attempts)
            output_artifacts.append(
                {
                    key: item[key]
                    for key in (
                        "artifact_id",
                        "plaintext_hash",
                        "ciphertext_hash",
                        "artifact_ref",
                        "storage_document_hash",
                        "encryption_context_hash",
                        "object_lock_mode",
                        "retain_until",
                        "transport_root",
                    )
                }
            )
            artifact_hashes.extend(
                (
                    item["plaintext_hash"],
                    item["ciphertext_hash"],
                    item["storage_document_hash"],
                    item["transport_root"],
                )
            )
        output = {
            "source_receipt_hash": source_receipt_hash,
            "artifacts": output_artifacts,
            "artifact_set_root": sha256_json(output_artifacts),
        }
        return ExecutionResultV2(
            output=output,
            transport_attempts=tuple(transport_attempts),
            artifact_hashes=tuple(artifact_hashes),
        )

    def _attest_qualification_admission(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if context.purpose != "research_lab.admission.v2":
            raise ValueError("qualification admission purpose is incorrect")
        if self._qualification_admission_resolver is None:
            raise ValueError("measured qualification admission source is unavailable")
        document = dict(self._qualification_admission_resolver(payload, context))
        if int(document.get("epoch_id", -1)) != int(context.epoch_id):
            raise ValueError("qualification admission epoch differs")
        leads = document.get("leads")
        if not isinstance(leads, list):
            raise ValueError("qualification admission leads are invalid")
        return ExecutionResultV2(
            output=document,
            artifact_hashes=(sha256_json(leads),),
        )
