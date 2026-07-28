#!/usr/bin/env python3.11
"""Execute the real V2 canonical, signing, SDK, receipt, and auditor path.

Input generation is test-only.  Every security-sensitive output is produced or
validated by candidate production modules.  The irreversible chain broadcast
and production database are replaced by :mod:`local_services`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
from validator_tee.host.enclave_hotkey_v2 import (  # noqa: E402
    AuthoritativeSetWeightsContextV2,
    _weight_extrinsic_module,
)


NOW = "2026-07-25T00:00:00Z"
GENESIS_HASH = (
    "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
)
PRODUCTION_SOURCE_PATHS = (
    "leadpoet_canonical/attested_v2.py",
    "leadpoet_canonical/auditor_v2.py",
    "leadpoet_canonical/hotkey_authority_v2.py",
    "leadpoet_canonical/weight_authority_v2.py",
    "leadpoet_canonical/weight_computation.py",
    "neurons/auditor_validator.py",
    "validator_tee/enclave/hotkey_authority_v2.py",
    "validator_tee/host/enclave_hotkey_v2.py",
    "validator_tee/host/weight_authority_v2.py",
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
    contract: dict[str, Any] | None = None

    def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
        loaded_fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        loaded_contract = json.loads(
            args.boundary_contract.read_text(encoding="utf-8")
        )
        if loaded_fixture["sanitization"]["contains_production_credentials"]:
            raise RuntimeError("rehearsal fixture contains production credentials")
        if set(loaded_contract["forbidden_substitutions"]) != {
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
        return loaded_fixture, loaded_contract

    inputs_passed, inputs = _run_workflow_stage(
        stage="input-contract",
        action=load_inputs,
        stages=stages,
    )
    if inputs_passed:
        fixture, contract = inputs

    identities: list[dict[str, str]] = []
    for path in PRODUCTION_SOURCE_PATHS:
        passed, identity = _run_workflow_stage(
            stage=f"source-identity:{path}",
            action=lambda path=path: _file_identity(path, args.candidate_sha),
            stages=stages,
        )
        if passed:
            identities.append(identity)

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

    def validate_workflow_evidence() -> None:
        if (
            cleanup["pending_faults"] != 0
            or cleanup["boundary_thread_alive_after_close"]
            or len(epochs) != expected_epochs
            or any(
                not epoch["canonical_vector_equal"]
                or not epoch["receipt_ancestry_verified"]
                or not epoch["auditor_verified"]
                or not epoch["auditor_runtime_verified"]
                or epoch["last_update"] != epoch["finalized_block"]
                for epoch in epochs
            )
        ):
            raise RuntimeError("joined V2 workflow evidence is incomplete")
        if args.profile == "release" and (
            len(faults) < 15 or concurrent_writes != 32
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
            sha256_json(contract) if contract is not None else None
        ),
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
