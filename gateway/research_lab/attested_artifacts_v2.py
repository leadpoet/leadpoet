"""Durable encrypted persistence for one V2 execution transport ledger."""

from __future__ import annotations

from collections import Counter
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
) -> dict[str, Any]:
    allowed_failed = {
        str(item.get("receipt_hash") or "")
        for item in source_graph.get("receipts") or ()
        if isinstance(item, Mapping) and item.get("status") == "failed"
    }
    validate_receipt_graph(
        source_graph,
        allowed_failed_receipt_hashes=allowed_failed,
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
        [str(item.get("request_artifact_hash") or "") for item in transport_attempts]
        + [
            str(item.get("response_artifact_hash") or "")
            for item in transport_attempts
            if item.get("terminal_status")
            in {"authenticated_response", "attested_local_response"}
        ]
    )
    listed = await client.v2_list_encrypted_artifacts(
        job_id=str(job_id),
        purpose=str(purpose),
    )
    artifacts = listed.get("artifacts")
    if not isinstance(artifacts, list):
        raise AttestedArtifactPersistenceV2Error(
            "coordinator encrypted artifact list is invalid"
        )
    observed_hashes = sorted(
        str(item.get("plaintext_hash") or "")
        for item in artifacts
        if isinstance(item, Mapping)
    )
    observed_descriptor_hashes = {
        str(item.get(field) or "")
        for item in artifacts
        if isinstance(item, Mapping)
        for field in (
            "artifact_id",
            "plaintext_hash",
            "ciphertext_hash",
            "encryption_context_hash",
        )
        if item.get(field)
    }
    expected_counts = Counter(expected_hashes)
    observed_counts = Counter(observed_hashes)
    if (
        any(observed_counts[key] < count for key, count in expected_counts.items())
        or not observed_descriptor_hashes.issubset(set(committed_hashes))
    ):
        raise AttestedArtifactPersistenceV2Error(
            "coordinator artifacts differ from execution commitments"
        )
    target_bucket = str(
        bucket
        or os.getenv("RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET", "")
        or ""
    ).strip()
    if not target_bucket:
        raise AttestedArtifactPersistenceV2Error(
            "V2 encrypted artifact bucket is not configured"
        )
    lineage_payload = {
        "source_receipt_hash": str(source_receipt["receipt_hash"]),
        "artifact_ids": [str(item["artifact_id"]) for item in artifacts],
        "artifact_plaintext_hashes": observed_hashes,
    }
    # The coordinator operation derives and signs the persistence job ID. The
    # same deterministic source identity is used for each S3 readback proof.
    from gateway.research_lab.attested_scoring_v2 import derive_execution_job_id_v2
    from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes

    lineage_job_id = derive_execution_job_id_v2(
        operation=OP_ATTEST_ARTIFACT_PERSISTENCE,
        purpose="leadpoet.artifact_persistence.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload_sha256=sha256_bytes(canonical_json(lineage_payload).encode("utf-8")),
        parent_receipt_hashes=(str(source_receipt["receipt_hash"]),),
        input_artifact_hashes=(),
        release_hash=str(release_manifest["release_hash"]),
        physical_role="gateway_coordinator",
    )
    persisted = []
    for artifact in artifacts:
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
    outcome = await execute_coordinator_v2(
        operation=OP_ATTEST_ARTIFACT_PERSISTENCE,
        purpose="leadpoet.artifact_persistence.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload=lineage_payload,
        parent_graphs=(dict(source_graph),),
        allowed_failed_parent_receipt_hashes=allowed_failed,
        input_artifact_hashes=(),
        release_manifest=release_manifest,
        client=client,
    )
    graph = outcome.get("receipt_graph")
    receipt = outcome.get("receipt")
    if not isinstance(graph, Mapping) or not isinstance(receipt, Mapping):
        raise AttestedArtifactPersistenceV2Error(
            "V2 artifact persistence receipt is unavailable"
        )
    validate_receipt_graph(
        graph,
        required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
        allowed_failed_receipt_hashes=allowed_failed,
    )
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
    }
