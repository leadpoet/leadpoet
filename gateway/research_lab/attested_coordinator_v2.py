"""Strict V2 bridge for measured ranking, promotion, and allocation decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from gateway.research_lab.attested_scoring_v2 import (
    DEFAULT_POLL_SECONDS,
    DEFAULT_RELEASE_MANIFEST_PATH,
    DEFAULT_TIMEOUT_SECONDS,
    execute_scoring_v2,
)
from gateway.tee.coordinator_executor_v2 import (
    COORDINATOR_OPERATIONS_V2,
    coordinator_receipt_output_v2,
)
from gateway.utils.tee_client import coordinator_tee_client


async def execute_coordinator_v2(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    payload: Mapping[str, Any],
    parent_graphs: Sequence[Mapping[str, Any]] = (),
    parent_ancestry_proofs: Sequence[Mapping[str, Any]] = (),
    allowed_failed_parent_receipt_hashes: Iterable[str] = (),
    input_artifact_hashes: Iterable[str] = (),
    provider_credential_ref_hashes: Optional[Mapping[str, str]] = None,
    internally_provisioned_credential_slots: Iterable[str] = (),
    require_egress_proxy: Optional[bool] = None,
    provider_profile_loader: Any = None,
    additional_job_credential_envelope_builder: Any = None,
    job_credential_provisioner: Any = None,
    credential_coordinator_client: Any = coordinator_tee_client,
    artifact_coordinator_client: Any = coordinator_tee_client,
    release_manifest: Optional[Mapping[str, Any]] = None,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    client: Any = coordinator_tee_client,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    persist_graph: Any = None,
    load_ancestry_proofs: Any = None,
    persist_ancestry_checkpoint: Any = None,
    boot_verifier: Any = None,
) -> dict[str, Any]:
    if provider_profile_loader is None:
        provider_profile_loader = _empty_coordinator_provider_profile
    return await execute_scoring_v2(
        operation=operation,
        purpose=purpose,
        epoch_id=epoch_id,
        sequence=sequence,
        payload=payload,
        worker_index=0,
        parent_graphs=parent_graphs,
        parent_ancestry_proofs=parent_ancestry_proofs,
        allowed_failed_parent_receipt_hashes=allowed_failed_parent_receipt_hashes,
        input_artifact_hashes=input_artifact_hashes,
        provider_credential_ref_hashes=provider_credential_ref_hashes,
        internally_provisioned_credential_slots=(
            internally_provisioned_credential_slots
        ),
        require_egress_proxy=require_egress_proxy,
        provider_profile_loader=provider_profile_loader,
        additional_job_credential_envelope_builder=(
            additional_job_credential_envelope_builder
        ),
        job_credential_provisioner=job_credential_provisioner,
        credential_coordinator_client=credential_coordinator_client,
        release_manifest=release_manifest,
        release_manifest_path=release_manifest_path,
        client=client,
        artifact_coordinator_client=artifact_coordinator_client,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        persist_graph=persist_graph,
        load_ancestry_proofs=load_ancestry_proofs,
        persist_ancestry_checkpoint=persist_ancestry_checkpoint,
        boot_verifier=boot_verifier,
        operation_registry=COORDINATOR_OPERATIONS_V2,
        physical_role_override="gateway_coordinator",
        expected_service_role="gateway_coordinator",
        rpc_namespace="coordinator_v2",
        receipt_output_projector=coordinator_receipt_output_v2,
        allow_persistence_bound_artifact_descriptors=True,
    )


def _empty_coordinator_provider_profile(
    profile: str,
    **_kwargs: Any,
) -> dict[str, Any]:
    return {
        "profile": str(profile or "default"),
        "credential_ref_hashes": {},
        "envelopes": [],
    }


async def load_provider_outcome_snapshot_v2(
    *,
    epoch_id: int,
    sequence: int = 0,
    execute: Any = execute_coordinator_v2,
) -> dict[str, Any]:
    from gateway.tee.coordinator_executor_v2 import OP_PROVIDER_OUTCOME_SNAPSHOT_V2

    return await execute(
        operation=OP_PROVIDER_OUTCOME_SNAPSHOT_V2,
        purpose="research_lab.provider_outcome_snapshot.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload={
            "schema_version": "leadpoet.provider_outcome_snapshot_request.v2",
        },
    )
