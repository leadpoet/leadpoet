from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab import attested_coordinator_v2, attested_scoring_v2
from gateway.tee import release_lineage_v2
from gateway.tee.coordinator_executor_v2 import OP_ATTEST_WEIGHT_PUBLICATION
from gateway.tee.release_channel_v2 import (
    ReleaseChannelV2Error,
    build_release_channel_v2,
    build_release_lineage_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
)
from scripts import verify_temporary_testnet_weights as proof_verifier
from tests.test_release_channel_v2 import (
    _local_gateway_manifest,
    _local_validator_manifest,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
LINEAGE_ID = "sha256:" + "c" * 64
NOW = "2026-09-08T00:00:00Z"
COMMITS = tuple(str(index) * 40 for index in range(1, 5))


def _private_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _channels():
    return [
        build_release_channel_v2(
            gateway_release_manifest=_local_gateway_manifest(commit),
            validator_release_manifest=_local_validator_manifest(commit),
        )
        for commit in COMMITS
    ]


def _install_scope(monkeypatch, tmp_path, *, channels=None):
    channels = list(channels or _channels())
    current = channels[-1]
    lineage = build_release_lineage_v2(channels, current_commit=COMMITS[-1])
    gateway_path = tmp_path / "gateway-v2-release-manifest.json"
    validator_path = tmp_path / "validator-release.json"
    lineage_path = tmp_path / "gateway-lineage.json"
    channels_path = tmp_path / "release-channels-v2.json"
    _private_json(gateway_path, current["gateway_release_manifest"])
    _private_json(validator_path, current["validator_release_manifest"])
    _private_json(lineage_path, lineage)
    _private_json(
        channels_path,
        {channel["commit_sha"]: channel for channel in channels},
    )
    monkeypatch.setenv("LEADPOET_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS", "true")
    monkeypatch.setenv("BITTENSOR_NETWORK", "test")
    monkeypatch.setenv("BITTENSOR_NETUID", "401")
    monkeypatch.setenv("ALLOWED_NETUIDS", "401")
    monkeypatch.setenv("GATEWAY_V2_RELEASE_MANIFEST", str(gateway_path))
    monkeypatch.setattr(
        attested_coordinator_v2,
        "DEFAULT_RELEASE_MANIFEST_PATH",
        gateway_path,
    )
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_TEMPORARY_TESTNET401_CURRENT_VALIDATOR_RELEASE",
        validator_path,
    )
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_TEMPORARY_TESTNET401_RELEASE_LINEAGE",
        lineage_path,
    )
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_TEMPORARY_TESTNET401_RELEASE_CHANNELS",
        channels_path,
    )
    monkeypatch.setattr(
        attested_scoring_v2,
        "DEFAULT_RELEASE_MANIFEST_PATH",
        gateway_path,
    )
    monkeypatch.setitem(
        attested_coordinator_v2.execute_coordinator_v2.__kwdefaults__,
        "release_manifest_path",
        gateway_path,
    )
    return {
        "channels": channels,
        "lineage": lineage,
        "gateway_path": gateway_path,
        "validator_path": validator_path,
        "lineage_path": lineage_path,
        "channels_path": channels_path,
    }


def _identity(*, lineage, commit, physical_role, private_key):
    expectation = lineage["releases"][commit]["roles"][physical_role]
    public_key = private_key.public_key().public_bytes_raw().hex()
    body = build_boot_identity_body(
        role=physical_role,
        physical_role=physical_role,
        commit_sha=expectation["commit_sha"],
        pcr0=expectation["pcr0"],
        build_manifest_hash=expectation["build_manifest_hash"],
        dependency_lock_hash=expectation["dependency_lock_hash"],
        config_hash=HASH_A,
        boot_nonce=(commit[0] + ("1" if physical_role == "gateway_coordinator" else "2")) * 16,
        signing_pubkey=public_key,
        transport_pubkey=(commit[0] * 64),
        transport_certificate_hash=HASH_B,
        attestation_user_data_hash=HASH_A,
        issued_at=NOW,
    )
    return create_boot_identity(
        body=body,
        attestation_document_b64=base64.b64encode(b"test-attestation").decode(),
    )


def _parent_graph(*, lineage, commit, physical_role, index):
    private_key = Ed25519PrivateKey.from_private_bytes(bytes([index + 1]) * 32)
    boot = _identity(
        lineage=lineage,
        commit=commit,
        physical_role=physical_role,
        private_key=private_key,
    )
    purpose = (
        "research_lab.allocation.v2"
        if physical_role == "gateway_coordinator"
        else "validator.hotkey_signature.v2"
    )
    body = build_execution_receipt_body(
        role=physical_role,
        purpose=purpose,
        job_id=f"release-transition-{index}",
        epoch_id=22_064,
        sequence=index,
        commit_sha=commit,
        pcr0=boot["pcr0"],
        build_manifest_hash=boot["build_manifest_hash"],
        dependency_lock_hash=boot["dependency_lock_hash"],
        config_hash=HASH_A,
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=HASH_A,
        output_root=HASH_B,
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    receipt = create_signed_execution_receipt(
        body=body,
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=private_key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
    ), boot


class BoundaryReached(RuntimeError):
    pass


class _Client:
    def __init__(self, boot):
        self.boot = boot

    async def coordinator_v2_health(self):
        return {
            "authority": "v2_only",
            "role": "gateway_coordinator",
            "physical_role": "gateway_coordinator",
            "worker_count": 1,
            "configured_worker_count": 0,
            "workers_alive": True,
            "ancestry_checkpoints": True,
            "ancestry_lineage_id": LINEAGE_ID,
            "boot_identity_hash": self.boot["boot_identity_hash"],
        }

    async def v2_get_boot_identity(self):
        return self.boot

    async def coordinator_v2_cancel_job(self, _job_id):
        return {"state": "cancelled"}

    async def coordinator_v2_submit_job(self, *_args, **_kwargs):
        raise AssertionError("upload boundary replacement was not used")

    async def coordinator_v2_put_chunk(self, *_args, **_kwargs):
        raise AssertionError("upload boundary replacement was not used")

    async def coordinator_v2_seal_job(self, *_args, **_kwargs):
        raise AssertionError("upload boundary replacement was not used")

    async def coordinator_v2_get_status(self, *_args, **_kwargs):
        raise AssertionError("upload boundary replacement was not used")


def _recording_nitro(calls):
    def verify(identity, *, expected_pcr0, certificate_validity_at_attestation_time):
        assert identity["pcr0"] == expected_pcr0
        assert certificate_validity_at_attestation_time is True
        calls.append((identity["commit_sha"], identity["physical_role"]))
        return dict(identity)

    return verify


@pytest.mark.asyncio
async def test_real_coordinator_scoring_boundary_verifies_four_release_parents(
    monkeypatch, tmp_path
):
    scope = _install_scope(monkeypatch, tmp_path)
    role_by_commit = {
        COMMITS[0]: "gateway_coordinator",
        COMMITS[1]: "validator_weights",
        COMMITS[2]: "gateway_coordinator",
        COMMITS[3]: "validator_weights",
    }
    graphs = []
    parent_boots = []
    for index, commit in enumerate(COMMITS):
        graph, boot = _parent_graph(
            lineage=scope["lineage"],
            commit=commit,
            physical_role=role_by_commit[commit],
            index=index,
        )
        graphs.append(graph)
        parent_boots.append(boot)
    current_key = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)
    current_boot = _identity(
        lineage=scope["lineage"],
        commit=COMMITS[-1],
        physical_role="gateway_coordinator",
        private_key=current_key,
    )
    calls = []
    nitro = _recording_nitro(calls)
    monkeypatch.setattr(attested_scoring_v2, "verify_boot_identity_nitro", nitro)
    monkeypatch.setattr(release_lineage_v2, "verify_boot_identity_nitro", nitro)
    monkeypatch.setattr(attested_scoring_v2, "_gateway_ancestry_lineage_id", lambda: LINEAGE_ID)
    monkeypatch.setattr(
        release_lineage_v2,
        "_fetch_historical_release",
        lambda _commit: (_ for _ in ()).throw(AssertionError("canonical S3 fallback used")),
    )

    async def stop_at_upload(**_kwargs):
        raise BoundaryReached("parent verification completed")

    monkeypatch.setattr(
        attested_scoring_v2,
        "upload_attested_execution_job_v2",
        stop_at_upload,
    )
    client = _Client(current_boot)
    with pytest.raises(BoundaryReached, match="parent verification completed"):
        await attested_coordinator_v2.execute_coordinator_v2(
            operation=OP_ATTEST_WEIGHT_PUBLICATION,
            purpose="gateway.weights.publication.v2",
            epoch_id=22_064,
            sequence=0,
            payload={"bundle_hash": HASH_A},
            parent_graphs=tuple(graphs),
            load_ancestry_proofs=lambda *_args, **_kwargs: {},
            client=client,
        )
    assert {(boot["commit_sha"], boot["physical_role"]) for boot in parent_boots} <= set(calls)
    assert (COMMITS[-1], "gateway_coordinator") in calls


def test_actual_weight_helpers_and_proof_use_the_same_full_channel_set(
    monkeypatch, tmp_path
):
    scope = _install_scope(monkeypatch, tmp_path)
    from gateway.api import weights
    from gateway.utils import pcr0_builder

    calls = []
    monkeypatch.setattr(
        release_lineage_v2,
        "verify_boot_identity_nitro",
        _recording_nitro(calls),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "verify_pcr0",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dynamic historical validator build used")
        ),
    )
    identities = []
    for index, commit in enumerate(COMMITS):
        for role_offset, role in enumerate(("gateway_coordinator", "validator_weights")):
            identities.append(
                _identity(
                    lineage=scope["lineage"],
                    commit=commit,
                    physical_role=role,
                    private_key=Ed25519PrivateKey.from_private_bytes(
                        bytes([20 + 2 * index + role_offset]) * 32
                    ),
                )
            )
    verifier = weights._build_authoritative_v2_receipt_boot_verifier(
        {"boot_identities": identities}
    )
    for identity in identities:
        assert verifier(identity)["boot_identity_hash"] == identity["boot_identity_hash"]
        assert weights._verify_authoritative_v2_boot(identity)["boot_identity_hash"] == identity["boot_identity_hash"]
    approved = proof_verifier.build_approved_release_lineage(
        candidate=COMMITS[-1],
        gateway_release=scope["channels"][-1]["gateway_release_manifest"],
        validator_release=scope["channels"][-1]["validator_release_manifest"],
        runtime_lineage=scope["lineage"],
        release_channels_path=scope["channels_path"],
    )
    assert approved == scope["lineage"]
    assert {(commit, role) for commit in COMMITS for role in ("gateway_coordinator", "validator_weights")} <= set(calls)


@pytest.mark.parametrize("mutation", ("missing", "extra", "tampered", "current_validator"))
def test_full_channel_store_mutations_fail_closed(monkeypatch, tmp_path, mutation):
    scope = _install_scope(monkeypatch, tmp_path)
    store = {channel["commit_sha"]: channel for channel in scope["channels"]}
    if mutation == "missing":
        store.pop(COMMITS[0])
    elif mutation == "extra":
        extra = build_release_channel_v2(
            gateway_release_manifest=_local_gateway_manifest("5" * 40),
            validator_release_manifest=_local_validator_manifest("5" * 40),
        )
        store[extra["commit_sha"]] = extra
    elif mutation == "tampered":
        store[COMMITS[0]] = {**store[COMMITS[0]], "channel_hash": HASH_A}
    else:
        _private_json(
            scope["validator_path"],
            _local_validator_manifest("5" * 40),
        )
    _private_json(scope["channels_path"], store)
    with pytest.raises(
        (RuntimeError, ReleaseChannelV2Error),
        match="channel|release",
    ):
        attested_coordinator_v2._temporary_testnet401_release_channel_loader(
            current_release_path=scope["gateway_path"]
        )


def test_task_channel_files_are_not_read_without_exact_testnet_scope(
    monkeypatch, tmp_path
):
    scope = _install_scope(monkeypatch, tmp_path)
    monkeypatch.delenv("LEADPOET_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS")
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_bounded_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("task channel file was read")
        ),
    )
    assert (
        attested_coordinator_v2._temporary_testnet401_release_channel_loader(
            current_release_path=scope["gateway_path"]
        )
        is None
    )
