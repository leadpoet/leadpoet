"""Durable encrypted persistence for one V2 execution transport ledger."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.tee.coordinator_executor_v2 import OP_ATTEST_ARTIFACT_PERSISTENCE
from gateway.utils.tee_artifact_store_v2 import (
    ATTESTED_V2_ARTIFACT_KEY_PREFIX,
    persist_enclave_artifact_v2,
)
from gateway.utils.tee_client import coordinator_tee_client
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    merkle_root,
    validate_receipt_graph,
)


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class AttestedArtifactPersistenceV2Error(RuntimeError):
    """A required hidden request/response artifact was not durably retained."""


def _validate_transport_artifact_commitments(
    *,
    expected_hashes: Sequence[str],
    observed_hashes: Sequence[str],
    committed_hashes: Sequence[str],
) -> None:
    """Require each distinct content commitment in the encrypted artifact vault."""

    expected = {str(item or "") for item in expected_hashes}
    observed = {str(item or "") for item in observed_hashes}
    committed = {str(item or "") for item in committed_hashes}
    if (
        any(not _HASH_RE.fullmatch(item) for item in expected)
        or not expected.issubset(observed)
        or not observed.issubset(committed)
    ):
        raise AttestedArtifactPersistenceV2Error(
            "coordinator artifacts differ from execution commitments"
        )


def _select_committed_encrypted_artifacts(
    artifacts: Any,
    *,
    committed_hashes: Sequence[str],
    require_descriptor_commitments: bool = False,
) -> list[dict[str, Any]]:
    """Exclude transient coordinator envelopes absent from the source receipt."""

    if not isinstance(artifacts, list):
        raise AttestedArtifactPersistenceV2Error(
            "coordinator encrypted artifact list is invalid"
        )
    committed = {str(item or "") for item in committed_hashes}
    selected = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise AttestedArtifactPersistenceV2Error(
                "coordinator encrypted artifact descriptor is invalid"
            )
        plaintext_hash = str(artifact.get("plaintext_hash") or "")
        if not _HASH_RE.fullmatch(plaintext_hash):
            raise AttestedArtifactPersistenceV2Error(
                "coordinator encrypted artifact plaintext hash is invalid"
            )
        descriptor_hashes = {
            str(artifact.get(field) or "")
            for field in (
                "artifact_id",
                "ciphertext_hash",
                "encryption_context_hash",
            )
            if artifact.get(field)
        }
        if (
            plaintext_hash in committed
            and (
                not require_descriptor_commitments
                or descriptor_hashes.issubset(committed)
            )
        ):
            selected.append(dict(artifact))
    return selected


async def _list_committed_transport_artifacts(
    *,
    client: Any,
    root_job_id: str,
    root_purpose: str,
    transport_attempts: Sequence[Mapping[str, Any]],
    committed_hashes: Sequence[str],
) -> list[dict[str, Any]]:
    scopes = [(str(root_job_id), str(root_purpose))]
    seen_scopes = set(scopes)
    for attempt in transport_attempts:
        if attempt.get("provider_id") == "aws_s3_object_lock":
            continue
        scope = (
            str(attempt.get("job_id") or ""),
            str(attempt.get("purpose") or ""),
        )
        if not all(scope):
            raise AttestedArtifactPersistenceV2Error(
                "transport artifact scope is invalid"
            )
        if scope not in seen_scopes:
            scopes.append(scope)
            seen_scopes.add(scope)

    by_artifact_id: dict[str, dict[str, Any]] = {}
    for scoped_job_id, scoped_purpose in scopes:
        listed = await client.v2_list_encrypted_artifacts(
            job_id=scoped_job_id,
            purpose=scoped_purpose,
        )
        selected = _select_committed_encrypted_artifacts(
            listed.get("artifacts"),
            committed_hashes=committed_hashes,
        )
        for artifact in selected:
            artifact_id = str(artifact.get("artifact_id") or "")
            if not _HASH_RE.fullmatch(artifact_id):
                raise AttestedArtifactPersistenceV2Error(
                    "coordinator encrypted artifact ID is invalid"
                )
            previous = by_artifact_id.get(artifact_id)
            if previous is not None and previous != artifact:
                raise AttestedArtifactPersistenceV2Error(
                    "coordinator encrypted artifact descriptor differs"
                )
            by_artifact_id[artifact_id] = artifact
    return [by_artifact_id[item] for item in sorted(by_artifact_id)]


async def persist_execution_transport_artifacts_v2(
    *,
    job_id: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    source_receipt: Mapping[str, Any],
    source_graph: Mapping[str, Any],
    transport_attempts: Sequence[Mapping[str, Any]],
    execution_artifact_hashes: Sequence[str] = (),
    release_manifest: Mapping[str, Any],
    client: Any = coordinator_tee_client,
    bucket: str | None = None,
    key_prefix: str = ATTESTED_V2_ARTIFACT_KEY_PREFIX,
    source_ancestry_compact_proof: Mapping[str, Any],
    persist_graph: Any = None,
    load_ancestry_proofs: Any = None,
    persist_ancestry_checkpoint: Any = None,
    boot_verifier: Any = None,
) -> dict[str, Any]:
    source_allowed_failed = {
        str(item.get("receipt_hash") or "")
        for item in source_graph.get("receipts") or ()
        if isinstance(item, Mapping) and item.get("status") != "succeeded"
    }
    validate_receipt_graph(
        source_graph,
        allowed_failed_receipt_hashes=source_allowed_failed,
    )
    committed_hashes = [str(item or "") for item in execution_artifact_hashes]
    if any(not _HASH_RE.fullmatch(item) for item in committed_hashes):
        raise AttestedArtifactPersistenceV2Error(
            "execution artifact commitment is invalid"
        )
    expected_artifact_root = (
        merkle_root(committed_hashes, domain="leadpoet-artifact-v2")
        if committed_hashes
        else EMPTY_ARTIFACT_ROOT
    )
    if source_receipt.get("artifact_root") != expected_artifact_root:
        raise AttestedArtifactPersistenceV2Error(
            "execution artifact root differs from receipt"
        )
    expected_hashes = sorted(
        [
            str(item.get("request_artifact_hash") or "")
            for item in transport_attempts
            if item.get("provider_id") != "aws_s3_object_lock"
        ]
        + [
            str(item.get("response_artifact_hash") or "")
            for item in transport_attempts
            if item.get("terminal_status")
            in {"authenticated_response", "attested_local_response"}
            and item.get("provider_id") != "aws_s3_object_lock"
        ]
    )
    artifacts = await _list_committed_transport_artifacts(
        client=client,
        root_job_id=str(job_id),
        root_purpose=str(purpose),
        transport_attempts=transport_attempts,
        committed_hashes=committed_hashes,
    )
    observed_hashes = sorted(
        str(item.get("plaintext_hash") or "")
        for item in artifacts
        if isinstance(item, Mapping)
    )
    _validate_transport_artifact_commitments(
        expected_hashes=expected_hashes,
        observed_hashes=observed_hashes,
        committed_hashes=committed_hashes,
    )
    reuse_persisted_artifacts = bool(artifacts) and all(
        item.get("persisted") is True for item in artifacts
    )
    resume_persisted_artifacts = any(
        item.get("persisted") is True for item in artifacts
    )
    lineage_payload = {
        "source_receipt_hash": str(source_receipt["receipt_hash"]),
        "artifact_ids": [str(item["artifact_id"]) for item in artifacts],
        "artifact_plaintext_hashes": observed_hashes,
    }
    # The coordinator operation derives and signs the persistence job ID. The
    # same deterministic source identity is used for each S3 readback proof.
    from gateway.research_lab.attested_scoring_v2 import (
        _gateway_ancestry_lineage_id,
        _persist_graph_then_ancestry_checkpoint_v2,
        _persist_ancestry_checkpoint_after_graph_v2,
        derive_execution_job_id_v2,
    )
    from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes

    lineage_job_id = derive_execution_job_id_v2(
        operation=OP_ATTEST_ARTIFACT_PERSISTENCE,
        purpose="leadpoet.artifact_persistence.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload_sha256=sha256_bytes(
            canonical_json(lineage_payload).encode("utf-8")
        ),
        parent_receipt_hashes=(str(source_graph["root_receipt_hash"]),),
        input_artifact_hashes=(),
        release_hash=str(release_manifest["release_hash"]),
        physical_role="gateway_coordinator",
    )
    persisted = []
    if not reuse_persisted_artifacts:
        target_bucket = str(
            bucket
            or os.getenv("RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET", "")
            or ""
        ).strip()
        if not target_bucket:
            raise AttestedArtifactPersistenceV2Error(
                "V2 encrypted artifact bucket is not configured"
            )
        for artifact in artifacts:
            if artifact.get("persisted") is True:
                continue
            result = await persist_enclave_artifact_v2(
                str(artifact["artifact_id"]),
                bucket=target_bucket,
                key_prefix=key_prefix,
                client=client,
                attestation_job_id=lineage_job_id,
            )
            if result.get("status") != "persisted":
                raise AttestedArtifactPersistenceV2Error(
                    "V2 encrypted artifact persistence failed closed"
                )
            persisted.append(dict(result))
    if not callable(boot_verifier):
        raise AttestedArtifactPersistenceV2Error(
            "V2 artifact ancestry boot verifier is unavailable"
        )
    lineage_id = _gateway_ancestry_lineage_id()
    _, source_checkpoint_persistence = (
        await _persist_graph_then_ancestry_checkpoint_v2(
            source_graph,
            source_ancestry_compact_proof,
            expected_root_receipt_hash=str(source_receipt["receipt_hash"]),
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=boot_verifier,
            persist_graph=persist_graph,
            persist_ancestry_checkpoint=persist_ancestry_checkpoint,
            allowed_failed_receipt_hashes=source_allowed_failed,
        )
    )
    outcome = await execute_coordinator_v2(
        operation=OP_ATTEST_ARTIFACT_PERSISTENCE,
        purpose="leadpoet.artifact_persistence.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload=lineage_payload,
        parent_graphs=(dict(source_graph),),
        parent_ancestry_proofs=(dict(source_ancestry_compact_proof),),
        allowed_failed_parent_receipt_hashes=source_allowed_failed,
        input_artifact_hashes=(),
        release_manifest=release_manifest,
        client=client,
        persist_graph=persist_graph,
        load_ancestry_proofs=load_ancestry_proofs,
        persist_ancestry_checkpoint=persist_ancestry_checkpoint,
        boot_verifier=boot_verifier,
    )
    graph = outcome.get("receipt_graph")
    receipt = outcome.get("receipt")
    ancestry_compact_proof = outcome.get("ancestry_compact_proof")
    if (
        not isinstance(graph, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(ancestry_compact_proof, Mapping)
    ):
        raise AttestedArtifactPersistenceV2Error(
            "V2 artifact persistence receipt is unavailable"
        )
    if receipt.get("job_id") != lineage_job_id:
        raise AttestedArtifactPersistenceV2Error(
            "V2 artifact persistence job binding differs"
        )
    child_allowed_failed = {
        str(item.get("receipt_hash") or "")
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping) and item.get("status") != "succeeded"
    }
    if (
        outcome.get("status") != "succeeded"
        or receipt.get("status") != "succeeded"
        or child_allowed_failed
    ):
        raise AttestedArtifactPersistenceV2Error(
            "V2 artifact persistence lineage did not succeed"
        )
    validate_receipt_graph(
        graph,
        required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
        allowed_failed_receipt_hashes=child_allowed_failed,
    )
    final_checkpoint_persistence = (
        await _persist_ancestry_checkpoint_after_graph_v2(
            ancestry_compact_proof,
            checkpointed_graph=graph,
            expected_root_receipt_hash=str(receipt["receipt_hash"]),
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=boot_verifier,
            persist_ancestry_checkpoint=persist_ancestry_checkpoint,
        )
    )
    if resume_persisted_artifacts:
        lineage_result = outcome.get("result")
        persisted_evidence = (
            lineage_result.get("artifacts")
            if isinstance(lineage_result, Mapping)
            else None
        )
        descriptor_by_id = {
            str(item["artifact_id"]): item for item in artifacts
        }
        if (
            not isinstance(persisted_evidence, list)
            or {
                str(item.get("artifact_id") or "")
                for item in persisted_evidence
                if isinstance(item, Mapping)
            }
            != set(descriptor_by_id)
            or any(not isinstance(item, Mapping) for item in persisted_evidence)
        ):
            raise AttestedArtifactPersistenceV2Error(
                "V2 persisted artifact lineage output is invalid"
            )
        persisted = [
            {
                "status": "persisted",
                "artifact_id": str(item["artifact_id"]),
                "artifact_ref": str(item["artifact_ref"]),
                "artifact_kind": str(
                    descriptor_by_id[str(item["artifact_id"])]["artifact_kind"]
                ),
                "artifact_hash": str(item["ciphertext_hash"]),
                "encryption_context_hash": str(item["encryption_context_hash"]),
                "object_lock_mode": str(item["object_lock_mode"]),
                "retain_until": str(item["retain_until"]),
                "storage_document_hash": str(item["storage_document_hash"]),
                "transport_root": str(item["transport_root"]),
            }
            for item in persisted_evidence
        ]
    from gateway.research_lab.attested_v2_store import persist_execution_sidecars_v2

    sidecars = await persist_execution_sidecars_v2(
        artifact_receipt_hash=str(receipt["receipt_hash"]),
        artifacts=persisted,
        transitions=(),
    )
    return {
        **dict(outcome),
        "artifacts": persisted,
        "sidecar_persistence": dict(sidecars),
        "execution_ancestry_checkpoint_persistence": dict(
            source_checkpoint_persistence
        ),
        "ancestry_checkpoint_persistence": dict(
            final_checkpoint_persistence
        ),
    }
