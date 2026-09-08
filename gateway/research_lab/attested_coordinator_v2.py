"""Strict V2 bridge for measured ranking, promotion, and allocation decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
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


_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS = (
    "LEADPOET_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS"
)
_TEMPORARY_TESTNET401_PRIOR_CHANNEL = Path(
    "/run/leadpoet-testnet401/prior-release-channel-v2.json"
)
_TEMPORARY_TESTNET401_CURRENT_VALIDATOR_RELEASE = Path(
    "/run/leadpoet-testnet401/validator-release.json"
)
_TEMPORARY_TESTNET401_RELEASE_LINEAGE = Path(
    "/run/leadpoet-testnet401/gateway-lineage.json"
)


def _bounded_json(path: Path, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or not 0 < metadata.st_size <= 4 * 1024 * 1024
        ):
            raise ValueError
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is unavailable or invalid")
    return value


def _temporary_testnet401_release_channel_loader(
    *,
    current_release_path: Path,
) -> Any:
    enabled = os.environ.get(_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS)
    if enabled is None:
        return None
    if enabled != "true" or any(
        os.environ.get(name) != value
        for name, value in (
            ("BITTENSOR_NETWORK", "test"),
            ("BITTENSOR_NETUID", "401"),
            ("ALLOWED_NETUIDS", "401"),
        )
    ):
        raise RuntimeError("temporary testnet401 release scope is invalid")
    if Path(current_release_path) != DEFAULT_RELEASE_MANIFEST_PATH:
        raise RuntimeError("temporary testnet401 current release path differs")

    from gateway.tee.release_channel_v2 import (
        build_release_channel_v2,
        build_release_lineage_v2,
        validate_prior_release_channel_v2,
        validate_release_channel_v2,
    )
    from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2
    from gateway.tee.release_manifest_v2 import validate_release_manifest
    from validator_tee.host.release_v2 import validate_validator_release_manifest

    current_gateway = validate_release_manifest(
        _bounded_json(current_release_path, "current gateway release manifest")
    )
    current_commit = str(current_gateway["commit_sha"])
    current_channel = validate_release_channel_v2(
        build_release_channel_v2(
            gateway_release_manifest=current_gateway,
            validator_release_manifest=validate_validator_release_manifest(
                _bounded_json(
                    _TEMPORARY_TESTNET401_CURRENT_VALIDATOR_RELEASE,
                    "current validator release manifest",
                )
            ),
        ),
        expected_commit=current_commit,
    )
    prior_raw = _bounded_json(
        _TEMPORARY_TESTNET401_PRIOR_CHANNEL,
        "prior release channel",
    )
    prior_commit = str(prior_raw.get("commit_sha") or "").lower()
    prior_channel = validate_prior_release_channel_v2(
        prior_raw,
        expected_commit=prior_commit,
    )
    if not prior_commit or prior_commit == current_commit:
        raise RuntimeError("temporary testnet401 prior release identity differs")
    lineage = validate_compact_release_lineage_v2(
        _bounded_json(
            _TEMPORARY_TESTNET401_RELEASE_LINEAGE,
            "temporary release lineage",
        ),
        expected_current_commit=current_commit,
    )
    expected_lineage = build_release_lineage_v2(
        [prior_channel, current_channel],
        current_commit=current_commit,
    )
    if lineage != expected_lineage:
        raise RuntimeError("temporary testnet401 release lineage differs")
    channels = {prior_commit: prior_channel, current_commit: current_channel}

    def load(commit: str) -> dict[str, Any]:
        channel = channels.get(str(commit or "").lower())
        if channel is None:
            raise RuntimeError("temporary testnet401 release is not approved")
        return dict(channel)

    return load


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
    release_channel_loader = None
    if boot_verifier is None and release_manifest is None:
        release_channel_loader = _temporary_testnet401_release_channel_loader(
            current_release_path=release_manifest_path,
        )
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
        release_channel_loader=release_channel_loader,
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
