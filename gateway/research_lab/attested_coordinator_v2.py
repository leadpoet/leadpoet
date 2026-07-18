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
from gateway.tee.coordinator_executor_v2 import COORDINATOR_OPERATIONS_V2
from gateway.utils.tee_client import coordinator_tee_client


async def execute_coordinator_v2(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    payload: Mapping[str, Any],
    parent_graphs: Sequence[Mapping[str, Any]] = (),
    allowed_failed_parent_receipt_hashes: Iterable[str] = (),
    input_artifact_hashes: Iterable[str] = (),
    provider_credential_ref_hashes: Optional[Mapping[str, str]] = None,
    internally_provisioned_credential_slots: Iterable[str] = (),
    require_egress_proxy: Optional[bool] = None,
    provider_profile_loader: Any = None,
    additional_job_credential_envelope_builder: Any = None,
    job_credential_provisioner: Any = None,
    release_manifest: Optional[Mapping[str, Any]] = None,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    client: Any = coordinator_tee_client,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    persist_graph: Any = None,
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
        release_manifest=release_manifest,
        release_manifest_path=release_manifest_path,
        client=client,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        persist_graph=persist_graph,
        boot_verifier=boot_verifier,
        operation_registry=COORDINATOR_OPERATIONS_V2,
        physical_role_override="gateway_coordinator",
        expected_service_role="gateway_coordinator",
        rpc_namespace="coordinator_v2",
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


async def register_openrouter_credentials_v2(
    *,
    miner_hotkey: str,
    key_label: Optional[str],
    runtime_credential: Mapping[str, Any],
    management_credential: Mapping[str, Any],
    epoch_id: int,
    sequence: int = 0,
    execute: Any = execute_coordinator_v2,
) -> dict[str, Any]:
    from gateway.tee.coordinator_executor_v2 import (
        OP_REGISTER_OPENROUTER_CREDENTIAL_V2,
    )
    from gateway.tee.openrouter_credential_v2 import (
        OPENROUTER_REGISTRATION_REQUEST_SCHEMA_VERSION,
        validate_openrouter_ingress_envelope_v2,
    )

    runtime = validate_openrouter_ingress_envelope_v2(runtime_credential)
    management = validate_openrouter_ingress_envelope_v2(management_credential)
    if (
        runtime["credential_kind"] != "runtime"
        or management["credential_kind"] != "management"
        or runtime["miner_hotkey_hash"] != management["miner_hotkey_hash"]
    ):
        raise ValueError("OpenRouter ingress credential pair is invalid")
    refs = {
        "openrouter": str(runtime["credential_value_hash"]),
        "openrouter_management": str(management["credential_value_hash"]),
    }
    return await execute(
        operation=OP_REGISTER_OPENROUTER_CREDENTIAL_V2,
        purpose="research_lab.openrouter_credential.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload={
            "schema_version": OPENROUTER_REGISTRATION_REQUEST_SCHEMA_VERSION,
            "miner_hotkey": str(miner_hotkey),
            "key_label": key_label,
            "runtime_credential": dict(runtime),
            "management_credential": dict(management),
        },
        input_artifact_hashes=(
            str(runtime["envelope_hash"]),
            str(runtime["ciphertext_blob_hash"]),
            str(management["envelope_hash"]),
            str(management["ciphertext_blob_hash"]),
        ),
        provider_credential_ref_hashes=refs,
        internally_provisioned_credential_slots=tuple(sorted(refs)),
        require_egress_proxy=False,
        provider_profile_loader=_empty_coordinator_provider_profile,
    )


async def preflight_openrouter_key_ref_v2(
    *,
    key_ref: str,
    miner_hotkey: str,
    epoch_id: int,
    sequence: int = 0,
    execute: Any = execute_coordinator_v2,
) -> dict[str, Any]:
    from gateway.research_lab.v2_credential_envelopes import (
        load_openrouter_credential_commitments_v2,
        load_openrouter_job_credential_envelope_v2,
    )
    from gateway.tee.coordinator_executor_v2 import (
        OP_PREFLIGHT_OPENROUTER_CREDENTIAL_V2,
    )
    from gateway.tee.openrouter_credential_v2 import (
        OPENROUTER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    )
    from leadpoet_canonical.attested_v2 import sha256_bytes

    commitments = await load_openrouter_credential_commitments_v2(
        key_ref=str(key_ref)
    )
    miner_hash = sha256_bytes(str(miner_hotkey).encode("utf-8"))
    if commitments.get("miner_hotkey_hash") != miner_hash:
        raise ValueError("OpenRouter credential does not belong to miner")

    async def envelope_builder(job_id: str):
        return (
            await load_openrouter_job_credential_envelope_v2(
                key_ref=str(key_ref),
                credential_kind="runtime",
                job_id=str(job_id),
            ),
        )

    return await execute(
        operation=OP_PREFLIGHT_OPENROUTER_CREDENTIAL_V2,
        purpose="research_lab.openrouter_credit_preflight.v2",
        epoch_id=int(epoch_id),
        sequence=int(sequence),
        payload={
            "schema_version": OPENROUTER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
            "key_ref_hash": str(commitments["key_ref_hash"]),
            "miner_hotkey_hash": miner_hash,
            "credential_value_hash": str(
                commitments["runtime_credential_value_hash"]
            ),
        },
        input_artifact_hashes=tuple(sorted(commitments.values())),
        provider_credential_ref_hashes={
            "openrouter": str(commitments["runtime_credential_value_hash"]),
        },
        require_egress_proxy=False,
        provider_profile_loader=_empty_coordinator_provider_profile,
        additional_job_credential_envelope_builder=envelope_builder,
    )


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
