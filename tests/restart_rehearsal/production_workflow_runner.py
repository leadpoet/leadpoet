#!/usr/bin/env python3.11
"""Execute the real V2 canonical, signing, SDK, receipt, and auditor path.

Input generation is test-only.  Every security-sensitive output is produced or
validated by candidate production modules.  The irreversible chain broadcast
and production database are replaced by :mod:`local_services`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SOURCE_ROOT = Path(os.environ.get("REHEARSAL_SOURCE_ROOT", "/source")).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from leadpoet_canonical.attested_v2 import (  # noqa: E402
    EMPTY_HOST_OPERATION_ROOT,
    build_execution_receipt_body,
    build_receipt_graph,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.auditor_v2 import (  # noqa: E402
    verify_attested_weight_authority_v2,
    verify_attested_weight_bundle_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (  # noqa: E402
    build_weight_extrinsic_authorization_v2,
    chain_signing_profiles,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (  # noqa: E402
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)
from local_services import (  # noqa: E402
    LocalBoundaryServices,
    LocalEnclaveSigningBoundary,
    LocalSDKSubstrateBoundary,
    local_enclave_backed_wallet,
)
from sanitized_weight_fixture import (  # noqa: E402
    EMPTY_ARTIFACT_ROOT,
    EMPTY_TRANSPORT_ROOT,
    SanitizedWeightFixture,
)
from validator_tee.enclave.hotkey_authority_v2 import (  # noqa: E402
    _Sr25519Backend,
)
from validator_tee.host.weight_authority_v2 import (  # noqa: E402
    build_authoritative_weight_bundle_v2,
)
from gateway.tee.rehearsal_behavior_contract_v2 import (  # noqa: E402
    build_rehearsal_behavior_contract_v2,
    validate_rehearsal_behavior_contract_v2,
)
from validator_tee.host.enclave_hotkey_v2 import (  # noqa: E402
    AuthoritativeSetWeightsContextV2,
    _weight_extrinsic_module,
)


NOW = "2026-07-25T00:00:00Z"
GENESIS_HASH = (
    "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
)
def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _run_workflow_stage(
    *,
    stage: str,
    action: Callable[[], Any],
    stages: list[dict[str, Any]],
) -> tuple[bool, Any]:
    """Fail one stage while allowing independent downstream probes to run."""

    try:
        value = action()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        result = {
            "error": str(exc)[:2000],
            "error_type": type(exc).__name__,
            "stage": stage,
            "status": "failed",
            "traceback": traceback.format_exc(limit=20)[-12000:],
        }
        stages.append(result)
        print(
            "PRODUCTION_WORKFLOW_STAGE_FAILED_CONTINUING "
            f"stage={stage} error_type={result['error_type']} "
            f"error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
        return False, None
    stages.append({"stage": stage, "status": "passed"})
    print(f"PRODUCTION_WORKFLOW_STAGE_PASSED stage={stage}", flush=True)
    return True, value


def _mark_workflow_stage_unexercised(
    *,
    stage: str,
    blocked_by: list[str],
    stages: list[dict[str, Any]],
) -> None:
    stages.append(
        {
            "blocked_by": list(blocked_by),
            "stage": stage,
            "status": "unexercised",
        }
    )
    print(
        "PRODUCTION_WORKFLOW_STAGE_UNEXERCISED "
        f"stage={stage} blocked_by={','.join(blocked_by)}",
        file=sys.stderr,
        flush=True,
    )


def _require_equal(left: Any, right: Any, message: str) -> Any:
    if left != right:
        raise RuntimeError(message)
    return left


class _AuditorScaleValue:
    def __init__(self, value: Any):
        self.value = value


class _AuditorLocalSubstrate:
    """Exact-hash chain boundary consumed by the production auditor."""

    def __init__(self, *, epoch_id: int, block: int):
        self.epoch_id = int(epoch_id)
        self.block = int(block)
        self.last_epoch_block = self.epoch_id * 360
        self.last_update = 0
        self.weights: list[tuple[int, int]] = []

    @staticmethod
    def _hash(block: int) -> str:
        return "0x" + hashlib.sha256(
            f"leadpoet-auditor-local-block:{block}".encode("ascii")
        ).hexdigest()

    def get_block_hash(self, block: int) -> str:
        return GENESIS_HASH if int(block) == 0 else self._hash(int(block))

    def get_chain_finalised_head(self) -> str:
        return self._hash(self.block)

    def get_chain_head(self) -> str:
        return self._hash(self.block)

    def get_block_number(self, block_hash: str) -> int:
        if block_hash == GENESIS_HASH:
            return 0
        if block_hash != self._hash(self.block):
            raise RuntimeError("auditor local chain received an unknown hash")
        return self.block

    def query(
        self,
        *,
        module: str,
        storage_function: str,
        params: list[Any],
        block_hash: str,
    ) -> _AuditorScaleValue:
        if block_hash != self._hash(self.block):
            raise RuntimeError("auditor local query is not exact-hash pinned")
        if module == "Timestamp" and storage_function == "Now" and params == []:
            return _AuditorScaleValue(
                int(datetime(2026, 7, 25, tzinfo=timezone.utc).timestamp())
                * 1000
            )
        if module != "SubtensorModule":
            raise RuntimeError("auditor local query module differs")
        if params == [71]:
            scheduler = {
                "Tempo": 360,
                "LastEpochBlock": self.last_epoch_block,
                "PendingEpochAt": self.last_epoch_block + 360,
                "SubnetEpochIndex": self.epoch_id,
                "BlocksSinceLastStep": self.block - self.last_epoch_block,
                "RevealPeriodEpochs": 1,
                "LastUpdate": [self.last_update],
            }
            if storage_function not in scheduler:
                raise RuntimeError("auditor local scheduler field differs")
            return _AuditorScaleValue(scheduler[storage_function])
        if params == [71, 0] and storage_function == "Weights":
            return _AuditorScaleValue(list(self.weights))
        raise RuntimeError("auditor local query shape differs")


class _AuditorLocalSubtensor:
    def __init__(self, substrate: _AuditorLocalSubstrate):
        self.substrate = substrate

    def get_subnet_hyperparameters(
        self, netuid: int, block: int | None = None
    ) -> Any:
        if int(netuid) != 71 or block is not None:
            raise RuntimeError("auditor local hyperparameter request differs")
        return SimpleNamespace(tempo=360, commit_reveal_period=1)

    def set_weights(
        self,
        *,
        netuid: int,
        wallet: Any,
        uids: list[int],
        weights: list[float],
        wait_for_finalization: bool,
        mechid: int,
    ) -> tuple[bool, str]:
        del wallet
        if (
            int(netuid) != 71
            or wait_for_finalization is not True
            or int(mechid) != 0
            or len(uids) != len(weights)
        ):
            raise RuntimeError("auditor local set_weights contract differs")
        from leadpoet_canonical.weights import normalize_to_u16

        self.substrate.weights = list(
            zip(
                [int(uid) for uid in uids],
                normalize_to_u16(
                    [int(uid) for uid in uids],
                    [float(weight) for weight in weights],
                ),
            )
        )
        self.substrate.last_update = self.substrate.block
        return True, "local finalized chain boundary accepted"


def _run_production_auditor(
    *,
    authority: Mapping[str, Any],
    identity_cache: Mapping[str, Any],
    epoch_id: int,
    block: int,
) -> dict[str, Any]:
    """Run the real auditor verifier, exact-block gate, and submit loop."""

    import neurons.auditor_validator as auditor_module
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover

    substrate = _AuditorLocalSubstrate(epoch_id=epoch_id, block=block)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="local"),
    )
    auditor.epoch_cutover = SubnetEpochCutover(
        network_genesis_hash=GENESIS_HASH,
        netuid=71,
        cutover_block=30_000 * 360,
        cutover_block_hash=_AuditorLocalSubstrate._hash(30_000 * 360),
        first_subnet_epoch_index=30_000,
        first_settlement_epoch_id=30_000,
        last_legacy_epoch_id=29_999,
    )
    auditor.epoch_archive_endpoint = "local://archive-boundary"
    auditor.epoch_archive_subtensor = _AuditorLocalSubtensor(substrate)
    auditor.subtensor = _AuditorLocalSubtensor(substrate)
    auditor.uid = 0
    auditor.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
        )
    )
    auditor.last_submitted_epoch = None
    auditor.last_authority_epoch = None

    original = auditor_module.verify_attested_weight_authority_v2

    def verify_with_local_nitro(
        value: Mapping[str, Any],
        *,
        identity_cache: Mapping[str, Any],
        chain_signing_profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        return original(
            value,
            identity_cache=identity_cache,
            chain_signing_profile=chain_signing_profile,
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "verified": True,
                "pcr0": expected_pcr0,
                "boundary": "local_nitro_attestation",
            },
        )

    auditor_module.verify_attested_weight_authority_v2 = (
        verify_with_local_nitro
    )
    try:
        verified = auditor.verify_attested_weights_v2(
            dict(authority),
            identity_cache=dict(identity_cache),
        )
    finally:
        auditor_module.verify_attested_weight_authority_v2 = original
    if verified is None:
        raise RuntimeError("production auditor rejected local authority")
    submitted = auditor.submit_weights_to_chain(
        epoch_id,
        verified,
        submission_epoch_id=epoch_id,
    )
    if not submitted:
        raise RuntimeError("production auditor did not finalize local weights")
    return verified


def _file_identity(path: str, candidate_sha: str) -> dict[str, str]:
    source = SOURCE_ROOT / path
    if not source.is_file():
        raise RuntimeError(f"candidate production source is absent: {path}")
    import subprocess

    expected = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{candidate_sha}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    observed = source.read_bytes()
    if observed != expected:
        raise RuntimeError(f"candidate production source differs: {path}")
    return {
        "path": path,
        "sha256": hashlib.sha256(observed).hexdigest(),
        "commit_sha": candidate_sha,
    }


def _receipt(
    *,
    epoch_id: int,
    candidate_sha: str,
    role: str,
    purpose: str,
    job_id: str,
    private_key: Ed25519PrivateKey,
    boot: Mapping[str, Any],
    config_hash: str,
    input_root: str,
    output_root: str,
    parents: list[str],
    sequence: int,
    transport_attempts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    attempts = list(transport_attempts or [])
    public_key = private_key.public_key().public_bytes_raw().hex()
    artifact_hashes = [
        item[key]
        for item in attempts
        for key in ("request_artifact_hash", "response_artifact_hash")
    ]
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        sequence=sequence,
        commit_sha=candidate_sha,
        pcr0=str(boot["pcr0"]),
        build_manifest_hash=str(boot["build_manifest_hash"]),
        dependency_lock_hash=str(boot["dependency_lock_hash"]),
        config_hash=config_hash,
        boot_identity_hash=str(boot["boot_identity_hash"]),
        input_root=input_root,
        output_root=output_root,
        transport_root_hash=(
            merkle_root(
                [str(item["attempt_hash"]) for item in attempts],
                domain="leadpoet-transport-v2",
            )
            if attempts
            else EMPTY_TRANSPORT_ROOT
        ),
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=(
            merkle_root(artifact_hashes, domain="leadpoet-artifact-v2")
            if artifact_hashes
            else EMPTY_ARTIFACT_ROOT
        ),
        parent_receipt_hashes=parents,
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


def _exercise_sdk_bridge(
    *,
    epoch_id: int,
    uids: list[int],
    weights_u16: list[int],
    submission_event_hash: str,
) -> dict[str, Any]:
    """Run the production Bittensor SDK interception with strict boundaries."""

    client = LocalEnclaveSigningBoundary()
    substrate = LocalSDKSubstrateBoundary()
    wallet = local_enclave_backed_wallet(client)
    mechanism = _weight_extrinsic_module()
    with AuthoritativeSetWeightsContextV2(
        substrate=substrate,
        wallet=wallet,
        weight_authorization_id=sha256_json(
            {"epoch_id": epoch_id, "kind": "sdk-weight-authorization"}
        ),
        weight_submission_event_hash=submission_event_hash,
        expected_era_period=8,
    ) as context:
        mechanism.get_encrypted_commit_v2(
            uids=uids,
            weights=weights_u16,
            version_key=10005000,
            last_epoch_block=epoch_id * 360,
            pending_epoch_at=0,
            subnet_epoch_index=epoch_id,
            tempo=360,
            blocks_since_last_step=22,
            current_block=epoch_id * 360 + 22,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey=wallet.hotkey.public_key,
        )
        signed = substrate.create_signed_extrinsic(
            call=object(),
            keypair=wallet.hotkey,
            era={"period": 8},
            nonce=None,
        )
    commit_requests = [
        request for kind, request in client.requests if kind == "commit"
    ]
    extrinsic_requests = [
        request for kind, request in client.requests if kind == "extrinsic"
    ]
    if (
        len(commit_requests) != 1
        or len(extrinsic_requests) != 1
        or commit_requests[0]["uids"] != uids
        or commit_requests[0]["weights_u16"] != weights_u16
        or len(context.extrinsic_signature_results) != 1
    ):
        raise RuntimeError("production SDK signing bridge evidence differs")
    return {
        "verified": True,
        "commit_request_hash": sha256_json(commit_requests[0]),
        "extrinsic_request_hash": sha256_json(extrinsic_requests[0]),
        "signature_hex": bytes(signed.signature).hex(),
    }


def _recompose_candidate_bundle(
    *,
    epoch_fixture: SanitizedWeightFixture,
    bundle: Mapping[str, Any],
    epoch_id: int,
) -> dict[str, Any]:
    binding_receipt = next(
        receipt
        for receipt in bundle["receipt_graph"]["receipts"]
        if receipt["purpose"] == "validator.hotkey_signature.v2"
    )
    weight_boot_for_handoff = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    enclave_graph = build_receipt_graph(
        root_receipt_hash=binding_receipt["parent_receipt_hashes"][0],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            receipt
            for receipt in bundle["receipt_graph"]["receipts"]
            if receipt["receipt_hash"] != binding_receipt["receipt_hash"]
        ],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
        host_operations=bundle["receipt_graph"]["host_operations"],
    )
    return build_authoritative_weight_bundle_v2(
        enclave_response={
            "weight_snapshot": bundle["weight_snapshot"],
            "weight_result": bundle["weight_result"],
            "weights_signature": bundle["weights_signature"],
            "receipt_graph": enclave_graph,
            "boot_identity": weight_boot_for_handoff,
            "weight_authorization_id": sha256_json(
                {"epoch_id": epoch_id, "kind": "local-authorization"}
            ),
            "source_artifacts": [],
        },
        validator_hotkey=bundle["validator_hotkey"],
        binding_message=bundle["binding_message"],
        binding_signature_result={
            "purpose": "validator.gateway_binding.v2",
            "validator_hotkey": bundle["validator_hotkey"],
            "signature": bundle["validator_hotkey_signature"],
            "receipt": binding_receipt,
        },
    )


def _run_independent_epoch_diagnostics(
    *,
    candidate_sha: str,
    epoch_id: int,
    stages: list[dict[str, Any]],
) -> None:
    """Exercise independent downstream contracts before the joined epoch."""

    epoch_fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    bundle_passed, bundle = _run_workflow_stage(
        stage="diagnostic:candidate-bundle-generation",
        action=epoch_fixture.bundle,
        stages=stages,
    )
    dependent_stages = (
        "diagnostic:host-bundle-composition",
        "diagnostic:primary-bundle-verification",
        "diagnostic:auditor-bundle-verification",
        "diagnostic:primary-auditor-vector-equality",
        "diagnostic:sdk-signing-bridge",
    )
    if not bundle_passed:
        for stage in dependent_stages:
            _mark_workflow_stage_unexercised(
                stage=stage,
                blocked_by=["diagnostic:candidate-bundle-generation"],
                stages=stages,
            )
        return

    _run_workflow_stage(
        stage="diagnostic:host-bundle-composition",
        action=lambda: _require_equal(
            _recompose_candidate_bundle(
                epoch_fixture=epoch_fixture,
                bundle=bundle,
                epoch_id=epoch_id,
            ),
            bundle,
            "production host bundle composition differs from canonical fixture",
        ),
        stages=stages,
    )
    primary_passed, primary = _run_workflow_stage(
        stage="diagnostic:primary-bundle-verification",
        action=lambda: validate_published_weight_bundle_v2(bundle),
        stages=stages,
    )
    auditor_passed, auditor = _run_workflow_stage(
        stage="diagnostic:auditor-bundle-verification",
        action=lambda: verify_attested_weight_bundle_v2(
            bundle,
            identity_cache=epoch_fixture.identity_cache(bundle),
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "verified": True,
                "pcr0": expected_pcr0,
                "boundary": "local_nitro_attestation",
            },
        ),
        stages=stages,
    )
    if primary_passed and auditor_passed:
        _run_workflow_stage(
            stage="diagnostic:primary-auditor-vector-equality",
            action=lambda: _require_equal(
                {
                    "uids": list(primary["uids"]),
                    "weights_u16": list(primary["weights_u16"]),
                },
                {
                    "uids": list(auditor["uids"]),
                    "weights_u16": list(auditor["weights_u16"]),
                },
                "primary and auditor canonical vectors differ",
            ),
            stages=stages,
        )
    else:
        blocked_by = []
        if not primary_passed:
            blocked_by.append("diagnostic:primary-bundle-verification")
        if not auditor_passed:
            blocked_by.append("diagnostic:auditor-bundle-verification")
        _mark_workflow_stage_unexercised(
            stage="diagnostic:primary-auditor-vector-equality",
            blocked_by=blocked_by,
            stages=stages,
        )
    _run_workflow_stage(
        stage="diagnostic:sdk-signing-bridge",
        action=lambda: _exercise_sdk_bridge(
            epoch_id=epoch_id,
            uids=[
                int(value)
                for value in bundle["weight_result"]["sparse_uids"]
            ],
            weights_u16=[
                int(value)
                for value in bundle["weight_result"]["sparse_weights_u16"]
            ],
            submission_event_hash=sha256_json(
                {"epoch_id": epoch_id, "kind": "diagnostic-publication"}
            ),
        ),
        stages=stages,
    )


def _run_epoch(
    *,
    services: LocalBoundaryServices,
    fixture: Mapping[str, Any],
    candidate_sha: str,
    epoch_id: int,
) -> dict[str, Any]:
    epoch_fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    coordinator_key = epoch_fixture.coordinator_key
    weight_key = epoch_fixture.weight_key
    bundle = epoch_fixture.bundle()
    assembled_bundle = _recompose_candidate_bundle(
        epoch_fixture=epoch_fixture,
        bundle=bundle,
        epoch_id=epoch_id,
    )
    if assembled_bundle != bundle:
        raise RuntimeError(
            "production host bundle composition differs from canonical fixture"
        )
    verified_bundle = validate_published_weight_bundle_v2(bundle)
    identity_cache = epoch_fixture.identity_cache(bundle)
    auditor_bundle = verify_attested_weight_bundle_v2(
        bundle,
        identity_cache=identity_cache,
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "verified": True,
            "pcr0": expected_pcr0,
            "boundary": "local_nitro_attestation",
        },
    )
    primary_vector = {
        "uids": list(verified_bundle["uids"]),
        "weights_u16": list(verified_bundle["weights_u16"]),
    }
    auditor_vector = {
        "uids": list(auditor_bundle["uids"]),
        "weights_u16": list(auditor_bundle["weights_u16"]),
    }
    if primary_vector != auditor_vector:
        raise RuntimeError("primary and auditor canonical vectors differ")

    persisted_bundle = services.request(
        "POST",
        "/database/insert",
        {
            "kind": "published_weight_bundle_v2",
            "epoch_id": epoch_id,
            "body": bundle,
        },
    )
    coordinator_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    weight_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "durable_readback_hash": persisted_bundle["evidence_hash"],
        "transparency_event_hash": sha256_json(
            {"epoch_id": epoch_id, "kind": "transparency"}
        ),
    }
    publication_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="gateway_coordinator",
        purpose="gateway.weights.publication.v2",
        job_id=f"weight-publication-{epoch_id}",
        private_key=coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json({"publication": "input", "epoch_id": epoch_id}),
        output_root=sha256_json(publication_doc),
        parents=[verified_bundle["root_receipt_hash"]],
        sequence=200,
    )
    publication_graph = build_receipt_graph(
        root_receipt_hash=publication_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=bundle["receipt_graph"]["receipts"] + [publication_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
    )
    submission_event_hash = sha256_json(
        {
            "bundle_hash": verified_bundle["bundle_hash"],
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc["transparency_event_hash"],
            "durable_readback_hash": publication_doc["durable_readback_hash"],
        }
    )
    sdk_bridge = _exercise_sdk_bridge(
        epoch_id=epoch_id,
        uids=primary_vector["uids"],
        weights_u16=primary_vector["weights_u16"],
        submission_event_hash=submission_event_hash,
    )

    profile_manifest = json.loads(
        (
            SOURCE_ROOT
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    profile = next(
        item
        for item in chain_signing_profiles(profile_manifest)
        if int(item["spec_version"])
        == int(profile_manifest["spec_version"])
    )
    seed = hashlib.sha256(
        b"hotkey-seed:" + candidate_sha.encode("ascii")
    ).digest()
    sr25519 = _Sr25519Backend()
    public_key, secret_key = sr25519.pair_from_seed(seed)
    commitment = hashlib.sha512(
        b"timelocked:" + epoch_id.to_bytes(8, "big") + _canonical(primary_vector)
    ).digest()
    block = int(verified_bundle["block"])
    authorization = build_weight_extrinsic_authorization_v2(
        profile=profile,
        validator_hotkey=verified_bundle["validator_hotkey"],
        hotkey_public_key_hex=public_key.hex(),
        epoch_id=epoch_id,
        netuid=int(verified_bundle["netuid"]),
        subnet_epoch_index=epoch_id,
        weight_receipt_hash=verified_bundle["weight_receipt_hash"],
        weight_submission_event_hash=submission_event_hash,
        weights_hash=verified_bundle["weights_hash"],
        sparse_uids=primary_vector["uids"],
        sparse_weights_u16=primary_vector["weights_u16"],
        commitment=commitment,
        reveal_round=epoch_id + 1,
        era_current=block,
        nonce=epoch_id,
        block_hash=hashlib.sha256(f"block:{block}".encode("ascii")).hexdigest(),
    )
    signature = sr25519.sign(
        (public_key, secret_key),
        bytes.fromhex(authorization["signed_message_hex"]),
    )
    signed_extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex=public_key.hex(),
        signature_hex=signature.hex(),
        era_period=int(authorization["era_period"]),
        era_current=int(authorization["era_current"]),
        nonce=int(authorization["nonce"]),
        call_data_hex=str(authorization["call_data_hex"]),
    )
    extrinsic_hash = signed_extrinsic_hash_v2(signed_extrinsic)
    services.request(
        "POST",
        "/chain/submit_extrinsic",
        {
            "epoch_id": epoch_id,
            "extrinsic_hash": extrinsic_hash,
            "extrinsic_hex": signed_extrinsic.hex(),
            "bundle_hash": verified_bundle["bundle_hash"],
            "weights_hash": verified_bundle["weights_hash"],
            **primary_vector,
        },
    )
    finalized = services.request(
        "POST",
        "/chain/finalize",
        {
            "epoch_id": epoch_id,
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": block + 1,
        },
    )

    extrinsic_output = {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "signature": signature.hex(),
        "extrinsic_hash": extrinsic_hash,
    }
    extrinsic_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="validator_weights",
        purpose="validator.set_weights_extrinsic.v2",
        job_id=f"set-weights-{epoch_id}",
        private_key=weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=authorization["authorization_hash"],
        output_root=sha256_json(extrinsic_output),
        parents=[verified_bundle["weight_receipt_hash"]],
        sequence=201,
    )
    finalization_job = f"weight-finalization-{epoch_id}"
    attempts = [
        epoch_fixture.source_attempt(
            category="weight-finalization",
            job_id=finalization_job,
            purpose="validator.weights.finalized.v2",
            sequence=300,
            provider_id="bittensor_chain",
            host="entrypoint-finney.opentensor.ai",
            method="POST",
        ),
        epoch_fixture.source_attempt(
            category="weight-finalization-archive",
            job_id=finalization_job,
            purpose="validator.weights.finalized.v2",
            sequence=301,
            provider_id="bittensor_archive",
            host="archive.chain.opentensor.ai",
            method="POST",
        ),
    ]
    finalization_doc = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "netuid": int(verified_bundle["netuid"]),
        "epoch_id": epoch_id,
        "weights_hash": verified_bundle["weights_hash"],
        "weight_receipt_hash": verified_bundle["weight_receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization": authorization,
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "extrinsic_signature": signature.hex(),
        "extrinsic_receipt_hash": extrinsic_receipt["receipt_hash"],
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": int(finalized["finalized_block"]),
        "finalized_block_hash": str(finalized["finalized_block_hash"]),
        "state_transition_hash": str(finalized["state_transition_hash"]),
    }
    final_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="validator_weights",
        purpose="validator.weights.finalized.v2",
        job_id=finalization_job,
        private_key=weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "extrinsic_receipt_hashes": [extrinsic_receipt["receipt_hash"]],
            }
        ),
        output_root=sha256_json(finalization_doc),
        parents=[extrinsic_receipt["receipt_hash"]],
        sequence=202,
        transport_attempts=attempts,
    )
    final_graph = build_receipt_graph(
        root_receipt_hash=final_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            item
            for item in bundle["receipt_graph"]["receipts"]
            if item["purpose"] != "validator.hotkey_signature.v2"
        ]
        + [extrinsic_receipt, final_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"]
        + attempts,
    )
    finalization_submission = {
        "schema_version": "leadpoet.weight_finalization_submission.v2",
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "weight_submission_event_hash": submission_event_hash,
        "finalization": finalization_doc,
        "receipt_graph": final_graph,
    }
    verified_finalization = validate_weight_finalization_submission_v2(
        finalization_submission,
        chain_signing_profile=profile_manifest,
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": submission_event_hash,
            "bundle_hash": verified_bundle["bundle_hash"],
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": authorization["authorization_hash"],
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": finalization_doc["finalized_block"],
            "finalized_block_hash": finalization_doc["finalized_block_hash"],
            "state_transition_hash": finalization_doc["state_transition_hash"],
        }
    )
    authority = {
        "schema_version": "leadpoet.published_weight_authority.v2",
        "bundle": bundle,
        "publication": {
            "weight_submission_event_hash": submission_event_hash,
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "publication_doc": publication_doc,
            "receipt_graph": publication_graph,
        },
        "finalization": {
            "weight_finalization_event_hash": finalization_event_hash,
            "submission": finalization_submission,
        },
    }
    auditor_authority = verify_attested_weight_authority_v2(
        authority,
        identity_cache=identity_cache,
        chain_signing_profile=profile_manifest,
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "verified": True,
            "pcr0": expected_pcr0,
            "boundary": "local_nitro_attestation",
        },
    )
    if auditor_authority["extrinsic_hash"] != extrinsic_hash:
        raise RuntimeError("auditor finalization differs from local chain")
    production_auditor_authority = _run_production_auditor(
        authority=authority,
        identity_cache=identity_cache,
        epoch_id=epoch_id,
        block=int(verified_bundle["block"]),
    )
    if production_auditor_authority["extrinsic_hash"] != extrinsic_hash:
        raise RuntimeError(
            "production auditor finalization differs from local chain"
        )

    reveal = services.request(
        "POST",
        "/chain/reveal",
        {"epoch_id": epoch_id, **primary_vector},
    )
    last_update = services.request(
        "GET", f"/chain/epoch/{epoch_id}/last_update"
    )
    revealed = services.request("GET", f"/chain/epoch/{epoch_id}/reveal")
    if revealed["reveal"]["vector_hash"] != reveal["vector_hash"]:
        raise RuntimeError("revealed vector readback differs")
    return {
        "epoch_id": epoch_id,
        "pcr0": weight_boot["pcr0"],
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "publication_receipt_hash": publication_receipt["receipt_hash"],
        "finalization_receipt_hash": verified_finalization[
            "finalization_receipt_hash"
        ],
        "receipt_ancestry_verified": True,
        "canonical_vector_hash": sha256_json(primary_vector),
        "canonical_vector_equal": True,
        "weights_hash": verified_bundle["weights_hash"],
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "signed_extrinsic_hash": extrinsic_hash,
        "sdk_bridge_verified": sdk_bridge["verified"],
        "sdk_commit_request_hash": sdk_bridge["commit_request_hash"],
        "sdk_extrinsic_request_hash": sdk_bridge["extrinsic_request_hash"],
        "finalized_block": finalized["finalized_block"],
        "last_update": last_update["last_update"],
        "reveal_vector_hash": reveal["vector_hash"],
        "auditor_verified": True,
        "auditor_runtime_verified": True,
    }


def _exercise_fault(
    services: LocalBoundaryServices,
    *,
    fault: str,
    ordinal: int,
) -> dict[str, Any]:
    services.inject(fault)
    status = {
        "http_400": 400,
        "http_403": 403,
        "http_429": 429,
        "http_500": 500,
        "duplicate_response": 409,
        "malformed_json": 502,
        "partial_body": 502,
        "unexpected_eof": 502,
        "timeout": 504,
    }.get(fault, 503)
    response = services.request(
        "POST",
        "/database/insert",
        {
            "kind": "fault_probe",
            "epoch_id": -1,
            "body": {"fault": fault, "ordinal": ordinal},
        },
        expected_status=status,
    )
    if response.get("fault") != fault:
        raise RuntimeError(f"fault response differs for {fault}")
    return {"fault": fault, "status": "fail_closed"}


def _exercise_concurrency(services: LocalBoundaryServices) -> int:
    def insert(ordinal: int) -> str:
        response = services.request(
            "POST",
            "/database/insert",
            {
                "kind": "concurrency_probe",
                "epoch_id": -2,
                "body": {"caller": ordinal},
            },
        )
        return str(response["evidence_hash"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(insert, range(32)))
    if len(set(hashes)) != 32:
        raise RuntimeError("concurrent durable writes were not isolated")
    return len(hashes)


async def _exercise_chain_settlement_state_space_async() -> dict[str, Any]:
    """Exercise every prefix topology through the production bootstrap gate."""

    from gateway.research_lab import champion_settlement_v2 as settlement
    from gateway.research_lab import store

    netuid = 71
    activation_epoch = 40_000
    target_epoch = activation_epoch + 4
    source_bundle_hash = sha256_json(
        {"kind": "rehearsal-settlement-source", "epoch": activation_epoch}
    )
    activation = {
        "netuid": netuid,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "first_epoch_id": activation_epoch,
        "source_bundle_hash": source_bundle_hash,
        "source_bundle_epoch_id": activation_epoch,
        "source_finalized_block": 8_700_039,
    }
    state: dict[str, Any] = {"rows": []}
    validated_ranges: list[tuple[int, int]] = []

    async def select_many(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
            return [dict(activation)]
        if table == settlement.FINALIZED_ALLOCATION_VIEW_V2:
            return [
                {
                    "bundle_hash": source_bundle_hash,
                    "netuid": netuid,
                    "epoch_id": activation_epoch,
                    "finalized_block": activation["source_finalized_block"],
                    "finalization_receipt_hash": sha256_json(
                        {"kind": "finalization", "epoch": activation_epoch}
                    ),
                }
            ]
        raise AssertionError(f"unexpected settlement select_many table: {table}")

    async def select_all(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if table != settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
            raise AssertionError(
                f"unexpected settlement select_all table: {table}"
            )
        return [dict(row) for row in state["rows"]]

    async def load_chain_history(
        *,
        netuid: int,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        if netuid != 71 or start_epoch != activation_epoch:
            raise AssertionError("settlement prefix validation range differs")
        validated_ranges.append((start_epoch, end_epoch))
        return [
            {"epoch": epoch}
            for epoch in range(start_epoch, end_epoch + 1)
        ]

    async def load_finalized_history(
        *,
        netuid: int,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        if (
            netuid != 71
            or start_epoch != activation_epoch
            or end_epoch != target_epoch
        ):
            raise AssertionError("finalized source validation range differs")
        return [
            {
                "epoch": activation_epoch,
                "finalized_bundle_hashes": [source_bundle_hash],
            }
        ]

    originals = (
        store.select_many,
        store.select_all,
        settlement.load_chain_realized_allocation_history_v1,
        settlement.load_finalized_allocation_history_v2,
    )
    store.select_many = select_many
    store.select_all = select_all
    settlement.load_chain_realized_allocation_history_v1 = load_chain_history
    settlement.load_finalized_allocation_history_v2 = load_finalized_history
    try:
        accepted: list[dict[str, Any]] = []
        total_epochs = target_epoch - activation_epoch + 1
        for prefix_length in range(total_epochs + 1):
            validated_ranges.clear()
            state["rows"] = [
                {
                    "netuid": netuid,
                    "epoch_id": epoch,
                    "settlement_hash": sha256_json(
                        {"kind": "settlement", "epoch": epoch}
                    ),
                }
                for epoch in range(
                    activation_epoch,
                    activation_epoch + prefix_length,
                )
            ]
            result = (
                await settlement.validate_chain_realized_settlement_bootstrap_v1(
                    netuid=netuid,
                    target_epoch=target_epoch,
                    maximum_backlog=total_epochs,
                )
            )
            expected_status = (
                "pristine_bootstrap_pending"
                if prefix_length == 0
                else "resumable_bootstrap_pending"
            )
            if (
                result["status"] != expected_status
                or result["backlog_epoch_count"]
                != total_epochs - prefix_length
                or result["validated_chain_realized_epochs"]
                != [
                    activation_epoch + offset
                    for offset in range(prefix_length)
                ]
                or (
                    prefix_length > 0
                    and validated_ranges
                    != [
                        (
                            activation_epoch,
                            activation_epoch + prefix_length - 1,
                        )
                    ]
                )
                or (prefix_length == 0 and validated_ranges)
            ):
                raise RuntimeError(
                    "chain settlement prefix behavior differs from contract"
                )
            accepted.append(
                {
                    "prefix_length": prefix_length,
                    "status": result["status"],
                    "backlog_epoch_count": result["backlog_epoch_count"],
                }
            )

        invalid_states = {
            "duplicate": [activation_epoch, activation_epoch],
            "gap": [activation_epoch, activation_epoch + 2],
            "missing-first": [activation_epoch + 1],
            "ahead": list(range(activation_epoch, target_epoch + 2)),
        }
        rejected: list[str] = []
        for name, epochs in invalid_states.items():
            state["rows"] = [
                {
                    "netuid": netuid,
                    "epoch_id": epoch,
                    "settlement_hash": sha256_json(
                        {"kind": "invalid-settlement", "name": name, "epoch": epoch}
                    ),
                }
                for epoch in epochs
            ]
            try:
                await settlement.validate_chain_realized_settlement_bootstrap_v1(
                    netuid=netuid,
                    target_epoch=target_epoch,
                    maximum_backlog=total_epochs,
                )
            except settlement.ChampionSettlementV2Error:
                rejected.append(name)
            else:
                raise RuntimeError(
                    f"invalid settlement topology was accepted: {name}"
                )

        state["rows"] = []
        try:
            await settlement.validate_chain_realized_settlement_bootstrap_v1(
                netuid=netuid,
                target_epoch=target_epoch,
                maximum_backlog=total_epochs - 1,
            )
        except settlement.ChampionSettlementV2Error:
            rejected.append("backlog-exceeds-policy")
        else:
            raise RuntimeError("excessive settlement backlog was accepted")
        return {
            "accepted_prefixes": accepted,
            "accepted_count": len(accepted),
            "rejected_state_classes": sorted(rejected),
        }
    finally:
        (
            store.select_many,
            store.select_all,
            settlement.load_chain_realized_allocation_history_v1,
            settlement.load_finalized_allocation_history_v2,
        ) = originals


def _exercise_chain_settlement_state_space() -> dict[str, Any]:
    return asyncio.run(_exercise_chain_settlement_state_space_async())


def _exercise_conditional_icp_policy() -> dict[str, Any]:
    """Validate configured tails and center through the production selector."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from research_lab.eval.conditional_validation import (
        build_conditional_category_assignment,
    )

    policy = (
        ResearchLabGatewayConfig.from_env().conditional_validation_policy()
    )
    policy_doc = policy.to_dict()
    if not policy.enabled:
        try:
            build_conditional_category_assignment(
                rolling_window_hash=sha256_json({"window": "disabled"}),
                benchmark_items=[],
                per_icp_summaries=[],
                policy=policy,
                baseline_serving_model_version_hash=sha256_json(
                    {"model": "disabled"}
                ),
            )
        except ValueError:
            return {
                "mode": policy.mode,
                "policy_hash": policy_doc["policy_hash"],
                "assignment_status": "disabled_fail_closed",
            }
        raise RuntimeError("disabled conditional policy accepted an assignment")

    items = []
    summaries = []
    for index in range(policy.total_icps):
        ref = f"rehearsal-icp-{index:04d}"
        items.append(
            {
                "icp_ref": ref,
                "icp_hash": sha256_json({"icp": index}),
                "intent_signal_signature": sha256_json(
                    {"intent": index}
                ),
                "set_id": index // max(1, policy.fresh_icp_count),
                "day_index": index,
                "day_rank": index + 1,
                "cohort": (
                    "fresh"
                    if index < policy.fresh_icp_count
                    else "retained"
                ),
            }
        )
        score = (
            50.0
            if policy.total_icps == 1
            else (100.0 * index) / (policy.total_icps - 1)
        )
        summaries.append({"icp_ref": ref, "score": score})

    kwargs = {
        "rolling_window_hash": sha256_json({"window": "configured"}),
        "benchmark_items": items,
        "per_icp_summaries": summaries,
        "policy": policy,
        "baseline_serving_model_version_hash": sha256_json(
            {"model": "configured"}
        ),
    }
    assignment = build_conditional_category_assignment(**kwargs)
    replay = build_conditional_category_assignment(**kwargs)
    if assignment != replay:
        raise RuntimeError("conditional ICP assignment is not deterministic")
    rows = sorted(assignment["items"], key=lambda row: float(row["score"]))
    low_refs = {
        row["icp_ref"] for row in rows[: policy.low_tail_count]
    }
    conditional_refs = {
        row["icp_ref"]
        for row in rows[
            policy.low_tail_count : (
                policy.low_tail_count + policy.conditional_total_icps
            )
        ]
    }
    high_refs = {
        row["icp_ref"]
        for row in rows[
            policy.low_tail_count + policy.conditional_total_icps :
        ]
    }
    assigned = assignment["items"]
    actual_conditional = {
        row["icp_ref"]
        for row in assigned
        if row["category"] == "conditional"
    }
    initial_refs = {
        row["icp_ref"]
        for row in assigned
        if row["category"] in {"public", "private"}
    }
    if (
        actual_conditional != conditional_refs
        or initial_refs != low_refs | high_refs
        or assignment["category_counts"]
        != {
            "public": policy.public_total_icps,
            "private": policy.private_total_icps,
            "conditional": policy.conditional_total_icps,
        }
        or sum(
            row["category"] == "public"
            and row["strength_label"] == "weak"
            for row in assigned
        )
        != policy.public_weak_total
        or sum(
            row["category"] == "private"
            and row["strength_label"] == "weak"
            for row in assigned
        )
        != policy.private_weak_total
    ):
        raise RuntimeError(
            "conditional ICP assignment differs from configured tail policy"
        )
    return {
        "policy_hash": policy_doc["policy_hash"],
        "assignment_hash": assignment["assignment_hash"],
        "category_counts": assignment["category_counts"],
        "low_tail_count": policy.low_tail_count,
        "high_tail_count": policy.high_tail_count,
        "conditional_count": policy.conditional_total_icps,
    }


def _exercise_conditional_candidate_gate() -> dict[str, Any]:
    """Prove conditional work runs only after the configured initial gate."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from research_lab.eval.evaluator import build_holdout_gate_result

    policy = ResearchLabGatewayConfig.from_env().conditional_validation_policy()
    if not policy.enabled:
        return {
            "mode": policy.mode,
            "policy_hash": policy.to_dict()["policy_hash"],
            "advancement_status": "disabled_fail_closed",
        }

    def rows(prefix: str, count: int, score: float) -> list[dict[str, Any]]:
        return [
            {
                "icp_ref": f"{prefix}-{index:04d}",
                "candidate_company_scores": [float(score)],
            }
            for index in range(count)
        ]

    gate = {
        "conditional_validation_required": True,
        "baseline_benchmark_bundle_id": "private_benchmark:" + "1" * 64,
        "baseline_benchmark_hash": sha256_json({"baseline": "candidate-gate"}),
        "category_assignment_hash": sha256_json({"assignment": "candidate-gate"}),
        "conditional_validation_policy_hash": policy.to_dict()["policy_hash"],
        "baseline_public_score": 0.0,
        "baseline_private_score": 0.0,
        "baseline_conditional_score": 0.0,
        "baseline_preliminary_score": 0.0,
        "baseline_aggregate_score": 0.0,
        "threshold_points": float(policy.threshold_points),
    }
    passing_score = float(policy.threshold_points)
    public = rows("public", policy.public_total_icps, passing_score)
    private = rows("private", policy.private_total_icps, passing_score)
    conditional = rows(
        "conditional",
        policy.conditional_total_icps,
        passing_score,
    )

    preliminary_rows, preliminary = build_holdout_gate_result(
        public_results=public,
        private_results=private,
        conditional_results=(),
        public_icp_count=policy.public_total_icps,
        private_icp_count=policy.private_total_icps,
        conditional_icp_count=policy.conditional_total_icps,
        gate=gate,
    )
    final_rows, final = build_holdout_gate_result(
        public_results=public,
        private_results=private,
        conditional_results=conditional,
        public_icp_count=policy.public_total_icps,
        private_icp_count=policy.private_total_icps,
        conditional_icp_count=policy.conditional_total_icps,
        gate=gate,
    )
    rejected_rows, rejected = build_holdout_gate_result(
        public_results=rows(
            "public-rejected",
            policy.public_total_icps,
            0.0,
        ),
        private_results=rows(
            "private-rejected",
            policy.private_total_icps,
            0.0,
        ),
        conditional_results=conditional,
        public_icp_count=policy.public_total_icps,
        private_icp_count=policy.private_total_icps,
        conditional_icp_count=policy.conditional_total_icps,
        gate={
            **gate,
            "baseline_preliminary_score": 50.0,
        },
    )
    initial_count = policy.public_total_icps + policy.private_total_icps
    if (
        preliminary.get("decision") != "conditional_validation_required"
        or preliminary.get("conditional_holdout_evaluated") is not False
        or len(preliminary_rows) != initial_count
        or final.get("decision") != "conditional_validation_approved"
        or final.get("conditional_holdout_evaluated") is not True
        or len(final_rows) != policy.total_icps
        or rejected.get("decision")
        != "rejected_before_conditional_validation"
        or rejected.get("conditional_holdout_evaluated") is not False
        or len(rejected_rows) != initial_count
    ):
        raise RuntimeError(
            "conditional candidate advancement differs from configured gate"
        )
    return {
        "policy_hash": policy.to_dict()["policy_hash"],
        "initial_count": initial_count,
        "conditional_count": policy.conditional_total_icps,
        "final_count": len(final_rows),
        "preliminary_decision": preliminary["decision"],
        "final_decision": final["decision"],
        "rejected_decision": rejected["decision"],
    }


def _exercise_git_tree_replacement() -> dict[str, Any]:
    """Validate deterministic replacement ancestry using configured tree policy."""

    from gateway.research_lab.git_tree_models import (
        TreePolicy,
        TreeReplacement,
        derive_tree_id,
    )

    policy = TreePolicy.from_env(os.environ)
    run_id = "00000000-0000-4000-8000-000000000001"
    roots = [
        sha256_json({"root": ordinal})
        for ordinal in range(3)
    ]
    manifests = [
        sha256_json({"manifest": ordinal})
        for ordinal in range(3)
    ]
    initial_tree_id = derive_tree_id(
        run_id=run_id,
        root_artifact_hash=roots[0],
        policy=policy,
    )
    first = TreeReplacement(
        generation=1,
        replaces_tree_id=initial_tree_id,
        cancellation_event_hash=sha256_json({"cancel": 0}),
        prior_root_artifact_hash=roots[0],
        prior_root_manifest_hash=manifests[0],
        prior_policy_hash=policy.policy_hash,
        root_artifact_hash=roots[1],
        root_manifest_hash=manifests[1],
        policy_hash=policy.policy_hash,
    )
    first_tree_id = derive_tree_id(
        run_id=run_id,
        root_artifact_hash=roots[1],
        policy=policy,
        replacement=first,
    )
    second = TreeReplacement(
        generation=2,
        replaces_tree_id=first_tree_id,
        cancellation_event_hash=sha256_json({"cancel": 1}),
        prior_root_artifact_hash=roots[1],
        prior_root_manifest_hash=manifests[1],
        prior_policy_hash=policy.policy_hash,
        root_artifact_hash=roots[2],
        root_manifest_hash=manifests[2],
        policy_hash=policy.policy_hash,
        reason="replacement_target_advanced",
    )
    second_tree_id = derive_tree_id(
        run_id=run_id,
        root_artifact_hash=roots[2],
        policy=policy,
        replacement=second,
    )
    if (
        len({initial_tree_id, first_tree_id, second_tree_id}) != 3
        or TreeReplacement.from_mapping(first.to_dict()) != first
        or TreeReplacement.from_mapping(second.to_dict()) != second
        or derive_tree_id(
            run_id=run_id,
            root_artifact_hash=roots[2],
            policy=policy,
            replacement=second,
        )
        != second_tree_id
    ):
        raise RuntimeError("Git-tree replacement identity is not deterministic")
    return {
        "policy_hash": policy.policy_hash,
        "max_nodes": policy.max_nodes,
        "tree_ids": [
            initial_tree_id,
            first_tree_id,
            second_tree_id,
        ],
        "replacement_hashes": [
            first.replacement_hash,
            second.replacement_hash,
        ],
    }


def _exercise_historical_metagraph_layouts() -> dict[str, Any]:
    """Exercise every candidate-declared archive layout through production."""

    from fixture_contract import (
        load_rehearsal_metagraph_account_ids,
        load_rehearsal_metagraph_hotkeys,
    )
    from gateway.tee.coordinator_chain_source_v2 import (
        CHAIN_ARCHIVE_ENDPOINT_URL,
        CoordinatorChainSourceV2,
    )
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from leadpoet_canonical.attested_v2 import (
        build_transport_attempt,
        sha256_bytes,
    )
    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_ARCHIVE_ENDPOINT_HOST,
        CHAIN_RPC_METHOD,
        ChainSourceV2Error,
        chain_source_policy_document,
        chain_source_policy_hash,
        encode_selective_metagraph_params,
        weights_storage_key,
    )

    policy = chain_source_policy_document()
    layouts = tuple(
        int(value) for value in policy["selective_result_last_fields"]
    )
    if (
        not layouts
        or tuple(sorted(set(layouts))) != layouts
        or any(value <= 52 for value in layouts)
    ):
        raise RuntimeError(
            "candidate chain-source result layouts are invalid"
        )
    account_ids = load_rehearsal_metagraph_account_ids(SOURCE_ROOT)
    hotkeys = load_rehearsal_metagraph_hotkeys(SOURCE_ROOT)
    validator_hotkey = hotkeys[0]
    cutover = json.loads(
        (
            SOURCE_ROOT
            / "config"
            / "stateful-epoch-cutover-sn71.json"
        ).read_text(encoding="utf-8")
    )
    netuid = int(cutover["netuid"])
    epoch_id = int(cutover["last_legacy_epoch_id"])
    target_block = (epoch_id + 1) * 360 - 1
    retry_hashes = {
        "bittensor_chain": sha256_json({"retry": "chain"}),
        "bittensor_archive": sha256_json({"retry": "archive"}),
        "coingecko": sha256_json({"retry": "coingecko"}),
    }
    def selective_fixture(last_field: int) -> str:
        if netuid < 1 << 6:
            compact_netuid = bytes((netuid << 2,))
        elif netuid < 1 << 14:
            compact_netuid = ((netuid << 2) | 1).to_bytes(2, "little")
        else:
            compact_netuid = ((netuid << 2) | 2).to_bytes(4, "little")
        encoded = bytearray(b"\x01" + compact_netuid)
        encoded.extend(b"\x00" * 4)
        encoded.extend(b"\x01" + account_ids[0])
        encoded.extend(b"\x00")
        encoded.extend(
            b"\x01"
            + ((target_block << 2) | 2).to_bytes(4, "little")
        )
        encoded.extend(b"\x00" * 44)
        encoded.extend(
            b"\x01"
            + bytes((len(account_ids) << 2,))
            + b"".join(account_ids)
        )
        encoded.extend(b"\x00" * (int(last_field) - 52))
        return "0x" + bytes(encoded).hex()

    class StrictArchiveBoundary:
        def __init__(self, *, last_field: int) -> None:
            self.last_field = int(last_field)
            self.calls: list[dict[str, Any]] = []

        def execute(
            self,
            request: Mapping[str, Any],
        ) -> dict[str, Any]:
            if (
                request.get("provider_id") != "bittensor_archive"
                or request.get("method") != "POST"
                or request.get("url") != CHAIN_ARCHIVE_ENDPOINT_URL
                or request.get("retry_policy_hash")
                != retry_hashes["bittensor_archive"]
            ):
                raise RuntimeError(
                    "historical layout probe crossed an undeclared boundary"
                )
            request_body = base64.b64decode(
                str(request["body_b64"]),
                validate=True,
            )
            rpc = json.loads(request_body)
            if set(rpc) != {"jsonrpc", "id", "method", "params"} or (
                rpc.get("jsonrpc") != "2.0"
            ):
                raise RuntimeError(
                    "historical layout probe received malformed JSON-RPC"
                )
            method = rpc.get("method")
            call_number = len(self.calls) + 1
            self.calls.append(
                {
                    "method": method,
                    "params": rpc.get("params"),
                }
            )
            if method == "chain_getFinalizedHead":
                if rpc.get("params") != []:
                    raise RuntimeError(
                        "historical finalized-head request differs"
                    )
                value: Any = "0x" + "a" * 64
            elif method == "chain_getBlockHash":
                if rpc.get("params") != [target_block]:
                    raise RuntimeError(
                        "historical layout probe requested another block"
                    )
                value = "0x" + "b" * 64
            elif method == "chain_getHeader":
                at_hash = str((rpc.get("params") or [""])[0])
                is_target = at_hash == "0x" + "b" * 64
                if at_hash not in {
                    "0x" + "a" * 64,
                    "0x" + "b" * 64,
                }:
                    raise RuntimeError(
                        "historical layout probe requested another hash"
                    )
                value = {
                    "number": hex(
                        target_block if is_target else target_block + 20
                    ),
                    "stateRoot": "0x" + "c" * 64,
                    "parentHash": "0x" + "d" * 64,
                    "extrinsicsRoot": "0x" + "e" * 64,
                    "digest": {"logs": []},
                }
            elif method == "state_call":
                if rpc.get("params") != [
                    CHAIN_RPC_METHOD,
                    encode_selective_metagraph_params(netuid=netuid),
                    "0x" + "b" * 64,
                ]:
                    raise RuntimeError(
                        "historical selective metagraph request differs"
                    )
                value = selective_fixture(self.last_field)
            elif method == "state_getStorage":
                if rpc.get("params") != [
                    weights_storage_key(
                        netuid=netuid,
                        validator_uid=0,
                    ),
                    "0x" + "b" * 64,
                ]:
                    raise RuntimeError(
                        "historical weight-storage request differs"
                    )
                value = "0x" + (
                    b"\x08"
                    + (1).to_bytes(2, "little")
                    + (1000).to_bytes(2, "little")
                    + (4).to_bytes(2, "little")
                    + (2000).to_bytes(2, "little")
                ).hex()
            else:
                raise RuntimeError(
                    "historical layout probe received an unknown RPC"
                )
            response_body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rpc.get("id"),
                    "result": value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            artifact_hash = sha256_json(
                {
                    "call": call_number,
                    "layout": self.last_field,
                    "method": method,
                }
            )
            attempt = build_transport_attempt(
                request_id=("%032x" % call_number),
                logical_operation_id=str(
                    request["logical_operation_id"]
                ),
                job_id=str(request["job_id"]),
                purpose=str(request["purpose"]),
                provider_id="bittensor_archive",
                attempt_number=int(request["attempt_number"]),
                method="POST",
                destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
                destination_port=443,
                path_hash=sha256_json({"path": "/"}),
                nonsecret_headers_hash=sha256_json(
                    {"headers": "application/json"}
                ),
                body_hash=sha256_bytes(request_body),
                credential_ref_hash=sha256_json(
                    {"credential": "public-archive"}
                ),
                retry_policy_hash=str(request["retry_policy_hash"]),
                timeout_ms=int(request["timeout_ms"]),
                started_at=NOW,
                terminal_status="authenticated_response",
                http_status=200,
                response_hash=sha256_bytes(response_body),
                request_artifact_hash=artifact_hash,
                response_artifact_hash=sha256_bytes(response_body),
                tls_peer_chain_hash=sha256_json(
                    {"tls": "archive-rehearsal"}
                ),
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at=NOW,
            )
            return {
                "terminal_status": "authenticated_response",
                "http_status": 200,
                "body_b64": base64.b64encode(response_body).decode(
                    "ascii"
                ),
                "transport_attempt": attempt,
            }

    def execute_layout(last_field: int) -> tuple[dict[str, Any], int]:
        boundary = StrictArchiveBoundary(last_field=last_field)
        source = CoordinatorChainSourceV2(
            execute_provider=boundary.execute,
            retry_policy_hashes=retry_hashes,
            epoch_authority={
                "mode": "stateful_v1",
                "cutover": cutover,
            },
            sleep=lambda _seconds: None,
        )
        context = ExecutionContextV2(
            job_id=f"rehearsal:historical-layout:{last_field}",
            purpose="research_lab.legacy_finalized_allocation.v2",
            epoch_id=epoch_id,
        )
        result = source.read_historical_finalized_weights(
            netuid=netuid,
            epoch_id=epoch_id,
            validator_hotkey=validator_hotkey,
            context=context,
        )
        return result, len(boundary.calls)

    accepted: list[int] = []
    call_counts: dict[str, int] = {}
    for last_field in layouts:
        result, call_count = execute_layout(last_field)
        if (
            result["target_block"] != target_block
            or result["validator_uid"] != 0
            or result["weights"] != [[1, 1000], [4, 2000]]
            or call_count != 6
        ):
            raise RuntimeError(
                "historical archive layout produced different authority"
            )
        accepted.append(last_field)
        call_counts[str(last_field)] = call_count

    rejected_layout = next(
        (
            value
            for value in range(53, max(layouts) + 1)
            if value not in layouts
        ),
        max(layouts) + 1,
    )
    try:
        execute_layout(rejected_layout)
    except ChainSourceV2Error:
        pass
    else:
        raise RuntimeError(
            "undeclared historical archive layout did not fail closed"
        )
    return {
        "policy_hash": chain_source_policy_hash(),
        "accepted_layouts": accepted,
        "rejected_layout": rejected_layout,
        "rpc_call_counts": call_counts,
    }


def _exercise_research_lab_allocation_conservation() -> dict[str, Any]:
    """Exercise the configured no-burn and compatibility allocation modes."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from leadpoet_verifier.economics import allocate_research_lab_epoch

    policy = ResearchLabGatewayConfig.from_env().reimbursement_policy_doc(
        enabled=True
    )
    policy_hash = sha256_json(policy)
    epoch = 30_000
    cap = Decimal(str(policy["research_lab_emission_percent"]))
    if (
        cap <= 0
        or policy.get("enable_conservative") is not False
        or policy.get("enable_champ_cap") is not False
        or Decimal(
            str(
                policy[
                    "reimbursement_max_cost_multiplier_with_champions"
                ]
            )
        )
        != Decimal("2")
    ):
        raise RuntimeError(
            "Research Lab default allocation policy differs from no-burn V2"
        )

    def reimbursement(
        uid: int,
        compute_microusd: int,
    ) -> dict[str, Any]:
        return {
            "uid": uid,
            "miner_hotkey": "reimbursement-%d" % uid,
            "source_id": "reimbursement_schedule:rehearsal-%d" % uid,
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reimbursement_epochs"]),
            "target_reimbursement_microusd": compute_microusd,
            "eligible_compute_microusd": compute_microusd,
        }

    current = allocate_research_lab_epoch(
        epoch,
        policy,
        [reimbursement(1, 1_000_000), reimbursement(2, 3_000_000)],
        [],
    )
    current_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in current["reimbursement_allocations"]
    }
    if (
        sum(current_paid.values()) != cap
        or current_paid[2] != current_paid[1] * Decimal("3")
        or Decimal(str(current["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "current reimbursements did not conserve the Lab cap by compute"
        )

    source_hash = sha256_json({"fixture": "historical-compute"})

    def fallback(uid: int, compute_microusd: int) -> dict[str, Any]:
        return {
            "uid": uid,
            "miner_hotkey": "fallback-%d" % uid,
            "source_id": "historical_compute_fallback:%064d" % uid,
            "island": "historical_compute",
            "status": "active",
            "target_reimbursement_microusd": compute_microusd,
            "fallback_window_start_epoch": epoch - 20,
            "fallback_window_end_epoch": epoch - 1,
            "source_allocation_epoch": epoch - 1,
            "source_allocation_hash": source_hash,
            "contribution_count": 1,
            "contribution_hash": sha256_json(
                {"uid": uid, "compute_microusd": compute_microusd}
            ),
        }

    historical = allocate_research_lab_epoch(
        epoch,
        policy,
        [],
        [],
        fallback_reimbursement_obligations=[
            fallback(3, 1_000_000),
            fallback(4, 3_000_000),
        ],
    )
    historical_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in historical["reimbursement_allocations"]
    }
    if (
        sum(historical_paid.values()) != cap
        or historical_paid[4] != historical_paid[3] * Decimal("3")
        or historical.get("historical_compute_fallback_source_epoch")
        != epoch - 1
        or Decimal(str(historical["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "historical compute fallback did not conserve the Lab cap"
        )

    champions = [
        {
            "uid": 5,
            "miner_hotkey": "champion-5",
            "source_id": "champion_reward:rehearsal-5",
            "champion_reward_id": "champion_reward:rehearsal-5",
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reward_epochs"]),
            "improvement_points": 1.0,
            "desired_alpha_percent": 7.0,
        },
        {
            "uid": 6,
            "miner_hotkey": "champion-6",
            "source_id": "champion_reward:rehearsal-6",
            "champion_reward_id": "champion_reward:rehearsal-6",
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reward_epochs"]),
            "improvement_points": 2.0,
            "desired_alpha_percent": 14.0,
        },
    ]
    champion_allocation = allocate_research_lab_epoch(
        epoch,
        policy,
        [],
        champions,
    )
    champion_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in [
            *champion_allocation["champion_allocations"],
            *champion_allocation["queued_champion_allocations"],
        ]
    }
    if (
        sum(champion_paid.values()) != cap
        or champion_paid[6] != champion_paid[5] * Decimal("2")
        or Decimal(str(champion_allocation["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "champions did not split the remaining Lab cap by configured reward"
        )

    valuation_microusd = int(
        (
            Decimal(str(policy["usd_per_0_1_percent_epoch"]))
            * Decimal(1_000_000)
        ).to_integral_value()
    )
    capped = allocate_research_lab_epoch(
        epoch,
        policy,
        [
            reimbursement(
                7,
                valuation_microusd * int(policy["reimbursement_epochs"]),
            )
        ],
        [champions[0]],
    )
    capped_reimbursement = Decimal(
        str(capped["reimbursement_allocations"][0]["paid_alpha_percent"])
    )
    if (
        capped_reimbursement != Decimal("0.2")
        or Decimal(str(capped["champion_alpha_percent"]))
        != cap - capped_reimbursement
        or Decimal(str(capped["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "active-champion reimbursement cap or remainder differs"
        )

    conservative_policy = dict(policy)
    conservative_policy["enable_conservative"] = True
    conservative = allocate_research_lab_epoch(
        epoch,
        conservative_policy,
        [],
        [],
    )
    if (
        Decimal(str(conservative["unallocated_percent"])) != cap
        or conservative["reimbursement_allocations"]
        or conservative["champion_allocations"]
    ):
        raise RuntimeError(
            "conservative compatibility mode no longer preserves burn"
        )
    return {
        "policy_hash": policy_hash,
        "lab_cap_percent": float(cap),
        "current_reimbursement_alpha_percent": float(
            current["reimbursement_alpha_percent"]
        ),
        "historical_reimbursement_alpha_percent": float(
            historical["reimbursement_alpha_percent"]
        ),
        "champion_alpha_percent": float(
            champion_allocation["champion_alpha_percent"]
        ),
        "active_champion_reimbursement_alpha_percent": float(
            capped["reimbursement_alpha_percent"]
        ),
        "conservative_unallocated_percent": float(
            conservative["unallocated_percent"]
        ),
        "conserved": True,
    }


def _exercise_receipt_graph_aggregate_pagination() -> dict[str, Any]:
    """Exercise aggregate evidence paging through the candidate store helper."""

    from gateway.research_lab import attested_v2_store

    row_limit = int(attested_v2_store._MAX_GRAPH_ROWS)
    query_chunk = int(attested_v2_store._GRAPH_QUERY_CHUNK)
    if row_limit < 1 or query_chunk < 1 or query_chunk > row_limit:
        raise RuntimeError("candidate V2 receipt graph limits are invalid")

    row_count = row_limit + 1
    width = len(str(row_count))
    expected_rows = [
        {
            "attempt_hash": (
                f"rehearsal-aggregate-attempt-{index:0{width}d}"
            )
        }
        for index in range(row_count)
    ]
    expected_by_key = {
        str(row["attempt_hash"]): dict(row) for row in expected_rows
    }
    expected_keys = set(expected_by_key)
    observed_queries: list[dict[str, Any]] = []
    original_select_all = attested_v2_store.select_all

    async def strict_select_all(
        table: str,
        *,
        filters: tuple[tuple[str, str, Any], ...],
        order_by: tuple[tuple[str, bool], ...],
        max_rows: int,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if (
            table != attested_v2_store.TRANSPORT_TABLE
            or len(filters) != 1
            or filters[0][0] != "attempt_hash"
            or filters[0][1] != "in"
            or order_by != (("attempt_hash", False),)
            or int(max_rows) != row_limit
        ):
            raise RuntimeError(
                "receipt graph rehearsal received an unknown store operation"
            )
        values = [str(value) for value in filters[0][2]]
        if not values or len(values) > query_chunk:
            raise RuntimeError(
                "receipt graph rehearsal query exceeded candidate chunk limit"
            )
        unknown = sorted(set(values) - expected_keys)
        if unknown:
            raise RuntimeError(
                "receipt graph rehearsal queried undeclared evidence"
            )
        observed_queries.append(
            {
                "count": len(values),
                "first": values[0],
                "last": values[-1],
            }
        )
        return [dict(expected_by_key[value]) for value in values]

    async def exercise() -> tuple[set[str], bool]:
        attested_v2_store.select_all = strict_select_all
        try:
            existing = await attested_v2_store._existing_exact_rows(
                attested_v2_store.TRANSPORT_TABLE,
                key_field="attempt_hash",
                expected_rows=expected_rows,
            )
            try:
                await attested_v2_store._select_by_values(
                    attested_v2_store.RECEIPT_TABLE,
                    field="receipt_hash",
                    values=(
                        f"rehearsal-receipt-{index:0{width}d}"
                        for index in range(row_count)
                    ),
                    key_fields=("receipt_hash",),
                )
            except attested_v2_store.AttestedV2StoreError as exc:
                if str(exc) != "V2 receipt graph exceeds row limit":
                    raise
                structural_limit_enforced = True
            else:
                structural_limit_enforced = False
            return existing, structural_limit_enforced
        finally:
            attested_v2_store.select_all = original_select_all

    existing, structural_limit_enforced = asyncio.run(exercise())
    if existing != expected_keys:
        raise RuntimeError("aggregate V2 receipt evidence was not exact")
    if len(observed_queries) < 2:
        raise RuntimeError("aggregate V2 receipt evidence was not paged")
    if (
        max(int(query["count"]) for query in observed_queries) > query_chunk
        or not structural_limit_enforced
    ):
        raise RuntimeError("V2 receipt graph safety bounds were weakened")
    return {
        "aggregate_rows": row_count,
        "aggregate_evidence_paged": True,
        "per_query_row_limit": row_limit,
        "query_chunk": query_chunk,
        "query_count": len(observed_queries),
        "structural_limit_enforced": True,
    }


BEHAVIOR_ACTIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "chain-settlement-state-space": _exercise_chain_settlement_state_space,
    "conditional-icp-policy": _exercise_conditional_icp_policy,
    "conditional-candidate-gate": _exercise_conditional_candidate_gate,
    "git-tree-replacement": _exercise_git_tree_replacement,
    "historical-metagraph-layouts": _exercise_historical_metagraph_layouts,
    "receipt-graph-aggregate-pagination": (
        _exercise_receipt_graph_aggregate_pagination
    ),
    "research-lab-allocation-conservation": (
        _exercise_research_lab_allocation_conservation
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("prepush", "release"), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--boundary-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate_sha) != 40 or any(
        value not in "0123456789abcdef" for value in args.candidate_sha
    ):
        parser.error("--candidate-sha must be a full lowercase Git SHA")
    expected_epochs = 1 if args.profile == "prepush" else 100
    if args.epochs != expected_epochs:
        parser.error(f"{args.profile} requires exactly {expected_epochs} epochs")

    stages: list[dict[str, Any]] = []
    fixture: dict[str, Any] | None = None
    boundary_contract: dict[str, Any] | None = None
    behavior_contract: dict[str, Any] | None = None

    def load_inputs() -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        loaded_fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        loaded_boundary_contract = json.loads(
            args.boundary_contract.read_text(encoding="utf-8")
        )
        if loaded_fixture["sanitization"]["contains_production_credentials"]:
            raise RuntimeError("rehearsal fixture contains production credentials")
        if set(loaded_boundary_contract["forbidden_substitutions"]) != {
            "gateway",
            "validator",
            "auditor",
            "canonical_bundle",
            "receipt_graph",
            "signature",
            "sdk_extrinsic",
            "verification",
        }:
            raise RuntimeError("rehearsal substitution policy is incomplete")
        loaded_behavior_contract = validate_rehearsal_behavior_contract_v2(
            build_rehearsal_behavior_contract_v2(
                source_root=SOURCE_ROOT,
                candidate_sha=args.candidate_sha,
                profile=args.profile,
                epoch_count=args.epochs,
            )
        )
        if args.profile == "release" and list(
            loaded_fixture.get("fault_matrix") or []
        ) != loaded_behavior_contract["fault_ids"]:
            raise RuntimeError(
                "mounted fault matrix differs from candidate contract"
            )
        return (
            loaded_fixture,
            loaded_boundary_contract,
            loaded_behavior_contract,
        )

    inputs_passed, inputs = _run_workflow_stage(
        stage="input-contract",
        action=load_inputs,
        stages=stages,
    )
    if inputs_passed:
        fixture, boundary_contract, behavior_contract = inputs

    identities: list[dict[str, str]] = []
    source_paths = (
        list(behavior_contract["production_source_paths"])
        if behavior_contract is not None
        else []
    )
    for path in source_paths:
        passed, identity = _run_workflow_stage(
            stage=f"source-identity:{path}",
            action=lambda path=path: _file_identity(path, args.candidate_sha),
            stages=stages,
        )
        if passed:
            identities.append(identity)

    behavior_evidence: dict[str, Any] = {}
    behavior_scenarios = (
        list(behavior_contract["behavior_scenarios"])
        if behavior_contract is not None
        else []
    )
    for scenario in behavior_scenarios:
        action = BEHAVIOR_ACTIONS.get(scenario)
        if action is None:
            _run_workflow_stage(
                stage=f"behavior:{scenario}",
                action=lambda scenario=scenario: (_ for _ in ()).throw(
                    RuntimeError(
                        f"candidate behavior scenario has no runner: {scenario}"
                    )
                ),
                stages=stages,
            )
            continue
        passed, result = _run_workflow_stage(
            stage=f"behavior:{scenario}",
            action=action,
            stages=stages,
        )
        if passed:
            behavior_evidence[scenario] = result

    _run_independent_epoch_diagnostics(
        candidate_sha=args.candidate_sha,
        epoch_id=30_000,
        stages=stages,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    service_root = args.output.parent / "local-services"
    faults: list[dict[str, Any]] = []
    concurrent_writes = 0
    epochs: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    cleanup = {
        "pending_faults": 0,
        "boundary_thread_alive_before_close": False,
        "boundary_thread_alive_after_close": False,
        "local_chain_epochs": 0,
    }

    if fixture is None:
        if args.profile == "release":
            _mark_workflow_stage_unexercised(
                stage="fault-matrix",
                blocked_by=["input-contract"],
                stages=stages,
            )
            _mark_workflow_stage_unexercised(
                stage="concurrency",
                blocked_by=["input-contract"],
                stages=stages,
            )
        for ordinal in range(args.epochs):
            _mark_workflow_stage_unexercised(
                stage=f"epoch-{30_000 + ordinal}",
                blocked_by=["input-contract"],
                stages=stages,
            )
        _mark_workflow_stage_unexercised(
            stage="boundary-cleanup",
            blocked_by=["input-contract"],
            stages=stages,
        )
    else:
        if args.profile == "release":
            for ordinal, fault in enumerate(fixture["fault_matrix"]):
                def run_fault(
                    *,
                    ordinal: int = ordinal,
                    fault: str = str(fault),
                ) -> dict[str, Any]:
                    with LocalBoundaryServices(
                        root=service_root / f"fault-{ordinal:02d}",
                        fixture=fixture,
                    ) as fault_services:
                        return _exercise_fault(
                            fault_services,
                            fault=fault,
                            ordinal=ordinal,
                        )

                passed, result = _run_workflow_stage(
                    stage=f"fault:{ordinal}:{fault}",
                    action=run_fault,
                    stages=stages,
                )
                if passed:
                    faults.append(result)

            def run_concurrency() -> int:
                with LocalBoundaryServices(
                    root=service_root / "concurrency",
                    fixture=fixture,
                ) as concurrency_services:
                    return _exercise_concurrency(concurrency_services)

            passed, result = _run_workflow_stage(
                stage="concurrency",
                action=run_concurrency,
                stages=stages,
            )
            if passed:
                concurrent_writes = result

        services = LocalBoundaryServices(
            root=service_root / "epochs",
            fixture=fixture,
        )
        services_started, _ = _run_workflow_stage(
            stage="boundary-start",
            action=services.__enter__,
            stages=stages,
        )
        if services_started:
            try:
                first_epoch = 30_000
                for ordinal in range(args.epochs):
                    epoch_id = first_epoch + ordinal
                    passed, epoch = _run_workflow_stage(
                        stage=f"epoch-{epoch_id}",
                        action=lambda epoch_id=epoch_id: _run_epoch(
                            services=services,
                            fixture=fixture,
                            candidate_sha=args.candidate_sha,
                            epoch_id=epoch_id,
                        ),
                        stages=stages,
                    )
                    if passed:
                        epochs.append(epoch)
                boundary_events = list(services.state.events)
                cleanup = {
                    "pending_faults": len(services.state.faults),
                    "boundary_thread_alive_before_close": (
                        services.thread.is_alive()
                    ),
                    "boundary_thread_alive_after_close": True,
                    "local_chain_epochs": len(services.state.chain),
                }
            finally:
                cleanup_passed, _ = _run_workflow_stage(
                    stage="boundary-cleanup",
                    action=lambda: services.__exit__(None, None, None),
                    stages=stages,
                )
                cleanup["boundary_thread_alive_after_close"] = (
                    services.thread.is_alive()
                )
                cleanup["local_chain_epochs"] = len(services.state.chain)
                if not cleanup_passed:
                    cleanup["cleanup_failed"] = True
        else:
            for ordinal in range(args.epochs):
                _mark_workflow_stage_unexercised(
                    stage=f"epoch-{30_000 + ordinal}",
                    blocked_by=["boundary-start"],
                    stages=stages,
                )
            _mark_workflow_stage_unexercised(
                stage="boundary-cleanup",
                blocked_by=["boundary-start"],
                stages=stages,
            )

    validation_dependencies = [
        item["stage"] for item in stages if item.get("status") != "passed"
    ]
    stage_status = {
        str(item.get("stage")): str(item.get("status"))
        for item in stages
        if isinstance(item, Mapping)
    }
    duplicate_stage_ids = len(stage_status) != len(stages)
    expected_before_validation = (
        set(behavior_contract["required_stage_ids"])
        - {"workflow-evidence-validation"}
        if behavior_contract is not None
        else set()
    )
    observed_before_validation = set(stage_status)

    epoch_authority_complete = (
        len(epochs) == expected_epochs
        and all(
            epoch.get("canonical_vector_equal") is True
            and epoch.get("receipt_ancestry_verified") is True
            and epoch.get("auditor_verified") is True
            and epoch.get("auditor_runtime_verified") is True
            and epoch.get("sdk_bridge_verified") is True
            and bool(epoch.get("signed_extrinsic_hash"))
            and epoch.get("last_update") == epoch.get("finalized_block")
            for epoch in epochs
        )
    )
    identity_paths = [str(item.get("path")) for item in identities]
    identity_commits = {
        str(item.get("commit_sha")) for item in identities
    }
    boundary_definitions = (
        boundary_contract.get("boundaries")
        if isinstance(boundary_contract, Mapping)
        else None
    )
    unknown_boundaries_rejected = (
        isinstance(boundary_definitions, Mapping)
        and bool(boundary_definitions)
        and all(
            isinstance(definition, Mapping)
            and definition.get("reject_unknown") is True
            for definition in boundary_definitions.values()
        )
    )
    behavioral_invariants = {
        "candidate_identity_exact": (
            behavior_contract is not None
            and behavior_contract.get("candidate_sha") == args.candidate_sha
        ),
        "protected_source_identity_exact": (
            behavior_contract is not None
            and sorted(identity_paths)
            == sorted(behavior_contract["production_source_paths"])
            and identity_commits == {args.candidate_sha}
        ),
        "chain_settlement_state_space_complete": (
            "chain-settlement-state-space" in behavior_evidence
        ),
        "conditional_icp_policy_config_bound": (
            "conditional-icp-policy" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["conditional-icp-policy"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["conditional_icp"].get(
                "policy_hash"
            )
        ),
        "conditional_candidate_advancement_exact": (
            "conditional-candidate-gate" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["conditional-candidate-gate"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["conditional_icp"].get(
                "policy_hash"
            )
        ),
        "git_tree_replacement_deterministic": (
            "git-tree-replacement" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["git-tree-replacement"].get("policy_hash")
            == behavior_contract["policy_commitments"]["git_tree"].get(
                "policy_hash"
            )
        ),
        "historical_metagraph_layouts_policy_bound": (
            "historical-metagraph-layouts" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["historical-metagraph-layouts"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["chain_source"].get(
                "policy_hash"
            )
            and behavior_evidence["historical-metagraph-layouts"].get(
                "accepted_layouts"
            )
            == behavior_contract["policy_commitments"]["chain_source"][
                "policy"
            ].get("selective_result_last_fields")
        ),
        "receipt_graph_aggregate_evidence_paged": (
            behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("aggregate_evidence_paged")
            is True
            and behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("structural_limit_enforced")
            is True
        ),
        "research_lab_allocation_policy_config_bound": (
            "research-lab-allocation-conservation" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence[
                "research-lab-allocation-conservation"
            ].get("policy_hash")
            == behavior_contract["policy_commitments"][
                "research_lab_allocation"
            ].get("policy_hash")
        ),
        "research_lab_allocation_conserved": (
            behavior_evidence.get(
                "research-lab-allocation-conservation",
                {},
            ).get("conserved")
            is True
        ),
        "canonical_vector_primary_auditor_equal": (
            epoch_authority_complete
            and all(
                epoch.get("canonical_vector_equal") is True
                for epoch in epochs
            )
        ),
        "receipt_ancestry_verified": (
            epoch_authority_complete
            and all(
                epoch.get("receipt_ancestry_verified") is True
                for epoch in epochs
            )
        ),
        "sdk_signing_bridge_verified": (
            epoch_authority_complete
            and all(
                epoch.get("sdk_bridge_verified") is True
                for epoch in epochs
            )
        ),
        "submission_finalized": (
            epoch_authority_complete
            and all(bool(epoch.get("signed_extrinsic_hash")) for epoch in epochs)
        ),
        "last_update_readback_equal": (
            epoch_authority_complete
            and all(
                epoch.get("last_update") == epoch.get("finalized_block")
                for epoch in epochs
            )
        ),
        "boundary_cleanup_complete": (
            cleanup["pending_faults"] == 0
            and cleanup["boundary_thread_alive_after_close"] is False
            and cleanup["local_chain_epochs"] == expected_epochs
        ),
        "unknown_boundaries_rejected": unknown_boundaries_rejected,
    }

    def validate_workflow_evidence() -> None:
        if behavior_contract is None:
            raise RuntimeError("candidate behavior contract is unavailable")
        if duplicate_stage_ids:
            raise RuntimeError("workflow emitted duplicate stage evidence")
        if observed_before_validation != expected_before_validation:
            missing = sorted(
                expected_before_validation - observed_before_validation
            )
            unexpected = sorted(
                observed_before_validation - expected_before_validation
            )
            raise RuntimeError(
                "workflow stage contract differs "
                f"missing={missing} unexpected={unexpected}"
            )
        required_invariants = set(
            behavior_contract["required_invariant_ids"]
        )
        if set(behavioral_invariants) != required_invariants:
            raise RuntimeError("workflow invariant contract differs")
        failed_invariants = sorted(
            name
            for name, passed in behavioral_invariants.items()
            if passed is not True
        )
        if failed_invariants:
            raise RuntimeError(
                "joined V2 workflow invariants failed: "
                + ",".join(failed_invariants)
            )
        if args.profile == "release" and (
            len(faults) != len(behavior_contract["fault_ids"])
            or concurrent_writes != 32
        ):
            raise RuntimeError("release fault or concurrency evidence is incomplete")

    if validation_dependencies:
        _mark_workflow_stage_unexercised(
            stage="workflow-evidence-validation",
            blocked_by=validation_dependencies,
            stages=stages,
        )
    else:
        _run_workflow_stage(
            stage="workflow-evidence-validation",
            action=validate_workflow_evidence,
            stages=stages,
        )

    status = (
        "passed"
        if all(item.get("status") == "passed" for item in stages)
        else "failed"
    )
    manifest = {
        "schema_version": "leadpoet.local_v2_workflow_evidence.v1",
        "status": status,
        "profile": args.profile,
        "release_sha": args.candidate_sha,
        "fixture_hash": sha256_json(fixture) if fixture is not None else None,
        "boundary_contract_hash": (
            sha256_json(boundary_contract)
            if boundary_contract is not None
            else None
        ),
        "behavior_contract": behavior_contract,
        "behavior_contract_hash": (
            behavior_contract.get("contract_hash")
            if behavior_contract is not None
            else None
        ),
        "behavior_evidence": behavior_evidence,
        "behavioral_invariants": behavioral_invariants,
        "production_source_identities": identities,
        "epoch_count": len(epochs),
        "epochs": epochs,
        "fault_matrix": faults,
        "concurrent_write_count": concurrent_writes,
        "boundary_event_count": len(boundary_events),
        "cleanup": cleanup,
        "stages": stages,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_bytes(_canonical(manifest) + b"\n")
    if status != "passed":
        failed = sum(item.get("status") == "failed" for item in stages)
        unexercised = sum(
            item.get("status") == "unexercised" for item in stages
        )
        print(
            "PRODUCTION_WORKFLOW_REHEARSAL_FAILED "
            f"profile={args.profile} failed={failed} "
            f"unexercised={unexercised} evidence={args.output}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"PRODUCTION_WORKFLOW_REHEARSAL_SUCCESS profile={args.profile} "
        f"epochs={len(epochs)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
