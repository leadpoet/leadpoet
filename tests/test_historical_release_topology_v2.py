import copy
import json
import subprocess
from pathlib import Path

import pytest

from gateway import deploy_readiness
from gateway.tee.prepare_active_release_lineage_v2 import (
    _fetch_exact_release_lineage_v2,
)
from gateway.tee.release_channel_v2 import (
    build_historical_release_channel_v2,
    build_historical_release_lineage_v2,
    build_release_channel_v2,
    release_channel_key,
)
from gateway.tee.release_manifest_v2 import (
    HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    PROTECTED_BASELINE_COMMIT,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    ReleaseManifestV2Error,
    validate_historical_release_manifest,
    validate_release_manifest,
)
from gateway.tee.release_lineage_v2 import (
    ReleaseLineageV2Error,
    validate_historical_compact_release_lineage_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json
from tests.test_release_channel_v2 import _gateway_manifest, _validator_manifest


ROOT = Path(__file__).resolve().parents[1]
F90_COMMIT = "f90b5eb3739eb3871a0d7bde0a3a1c41c62016ea"
LEGACY_ROLES = (
    "gateway_autoresearch",
    "gateway_coordinator",
    "gateway_scoring",
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _historical_gateway_manifest(commit: str = F90_COMMIT) -> dict:
    roles = {}
    for role, character in zip(LEGACY_ROLES, "abc"):
        roles[role] = {
            "physical_role": role,
            "service_role": role,
            "commit_sha": commit,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "source_manifest_hash": _hash("1"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("2"),
            "dockerfile_hash": _hash("3"),
            "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
            "eif_hashes": [_hash(character)],
            "verified_build_count": 6,
            "builder_domains": ["gateway", "validator"],
        }
    body = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "commit_sha": commit,
        "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        "protected_baseline_commit": PROTECTED_BASELINE_COMMIT,
        "acceptance_signer_pubkey_hash": _hash("f"),
        "roles": roles,
        "build_evidence_root": _hash("4"),
        "verified_build_count": 18,
    }
    return {**body, "release_hash": sha256_json(body)}


def _historical_channel(commit: str = F90_COMMIT) -> dict:
    return build_historical_release_channel_v2(
        gateway_release_manifest=_historical_gateway_manifest(commit),
        validator_release_manifest=_validator_manifest(commit),
        expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    )


class _Body:
    def __init__(self, value: bytes):
        self.value = value

    def read(self) -> bytes:
        return self.value


class _S3:
    def __init__(self, channels: list[dict]):
        self.objects = {
            release_channel_key(channel["commit_sha"]): (
                json.dumps(channel, sort_keys=True, separators=(",", ":"))
            ).encode("ascii")
            for channel in channels
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        del Bucket
        return {"Body": _Body(self.objects[Key])}


def _boot(role: str, summary: dict, character: str) -> dict:
    return {
        "role": role,
        "physical_role": role,
        "commit_sha": summary["commit_sha"],
        "pcr0": summary["pcr0"],
        "build_manifest_hash": summary["execution_manifest_hash"],
        "dependency_lock_hash": summary["dependency_lock_hash"],
        "boot_identity_hash": _hash(character),
        "config_hash": _hash(character),
    }


def test_historical_manifest_requires_exact_opt_in_and_known_shape() -> None:
    frozen_topology = json.loads(
        subprocess.check_output(
            ["git", "show", f"{F90_COMMIT}:gateway/tee/topology.json"],
            cwd=ROOT,
            text=True,
        )
    )
    assert frozen_topology["topology_hash"] == HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
    assert set(frozen_topology["roles"]) == set(LEGACY_ROLES)
    manifest = _historical_gateway_manifest()
    assert validate_historical_release_manifest(
        manifest,
        expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    ) == manifest
    with pytest.raises(ReleaseManifestV2Error, match="roles are incomplete"):
        validate_release_manifest(manifest)
    with pytest.raises(ReleaseManifestV2Error, match="unsupported"):
        validate_historical_release_manifest(
            manifest,
            expected_topology_hash=_hash("9"),
        )


def test_historical_lineage_rejects_mixed_current_topology() -> None:
    historical = _historical_channel()
    lineage = build_historical_release_lineage_v2(
        [historical],
        current_commit=F90_COMMIT,
        expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    )
    assert validate_historical_compact_release_lineage_v2(
        lineage,
        expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        expected_current_commit=F90_COMMIT,
    ) == lineage

    mixed = copy.deepcopy(lineage)
    mixed["releases"][F90_COMMIT]["roles"].pop("gateway_autoresearch")
    body = {key: value for key, value in mixed.items() if key != "lineage_hash"}
    mixed["lineage_hash"] = sha256_json(body)
    with pytest.raises(ReleaseLineageV2Error, match="roles are incomplete"):
        validate_historical_compact_release_lineage_v2(
            mixed,
            expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        )


def test_active_lineage_fetch_accepts_f90_and_rejects_mixed_release() -> None:
    authority = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    historical = _historical_channel()
    lineage = _fetch_exact_release_lineage_v2(
        candidate_commit_sha=F90_COMMIT,
        authority_commit_sha=authority,
        required_commits=[F90_COMMIT],
        repository=ROOT,
        bucket="test",
        prefix="attested-v2/releases",
        historical_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        s3_client=_S3([historical]),
    )
    assert set(lineage["releases"]) == {F90_COMMIT}

    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(authority),
        validator_release_manifest=_validator_manifest(authority),
    )
    with pytest.raises(ReleaseManifestV2Error, match="roles are incomplete"):
        _fetch_exact_release_lineage_v2(
            candidate_commit_sha=F90_COMMIT,
            authority_commit_sha=authority,
            required_commits=sorted((F90_COMMIT, authority)),
            repository=ROOT,
            bucket="test",
            prefix="attested-v2/releases",
            historical_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
            s3_client=_S3([historical, current]),
        )


def test_historical_deploy_readiness_is_exact_and_opt_in() -> None:
    gateway_release = _historical_gateway_manifest()
    validator_release = _validator_manifest(F90_COMMIT)
    channel = _historical_channel()
    lineage = build_historical_release_lineage_v2(
        [channel],
        current_commit=F90_COMMIT,
        expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    )
    boots = {
        role: _boot(role, gateway_release["roles"][role], character)
        for role, character in zip(LEGACY_ROLES, "567")
    }
    runtime_readiness = {
        "schema_version": "leadpoet.gateway_v2_runtime_readiness.v2",
        "status": "ready",
        "provider_registry_hash": _hash("8"),
        "roles": [
            {
                "physical_role": role,
                "role": role,
                "worker_count": 1,
                "configured_worker_count": 1,
                "boot_identity_hash": boots[role]["boot_identity_hash"],
            }
            for role in LEGACY_ROLES
        ],
    }
    verify = lambda identity, **_kwargs: identity
    gateway_evidence = deploy_readiness.build_gateway_v2_readiness_evidence(
        expected_commit=F90_COMMIT,
        source_commit=F90_COMMIT,
        build_commit=F90_COMMIT,
        gateway_release_manifest=gateway_release,
        validator_release_manifest=validator_release,
        compact_lineage=lineage,
        boot_identities=boots,
        expected_role_config_hashes={
            role: boot["config_hash"] for role, boot in boots.items()
        },
        runtime_readiness=runtime_readiness,
        coordinator_attestation_pcr0=boots["gateway_coordinator"]["pcr0"],
        boot_verifier=verify,
        expected_historical_topology_hash=(
            HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
        ),
    )
    validator_summary = validator_release["release"]
    validator_boot = {
        "role": "validator_weights",
        "physical_role": "validator_weights",
        "commit_sha": F90_COMMIT,
        "pcr0": validator_summary["pcr0"],
        "build_manifest_hash": validator_summary["app_manifest_hash"],
        "dependency_lock_hash": validator_summary["dependency_lock_hash"],
        "boot_identity_hash": _hash("9"),
        "config_hash": _hash("9"),
    }
    validator_evidence = deploy_readiness.build_validator_v2_readiness_evidence(
        expected_commit=F90_COMMIT,
        host_commit=F90_COMMIT,
        gateway_release_manifest=gateway_release,
        validator_release_manifest=validator_release,
        compact_lineage=lineage,
        boot_identity=validator_boot,
        expected_config_hash=validator_boot["config_hash"],
        boot_verifier=verify,
        expected_historical_topology_hash=(
            HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
        ),
    )
    ready = deploy_readiness.build_v2_deploy_readiness_manifest(
        expected_commit=F90_COMMIT,
        gateway_evidence=gateway_evidence,
        validator_evidence=validator_evidence,
        expected_historical_topology_hash=(
            HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
        ),
    )
    assert deploy_readiness.validate_v2_deploy_readiness_manifest(
        ready,
        runtime_source_commit=F90_COMMIT,
        runtime_build_commit=F90_COMMIT,
        expected_historical_topology_hash=(
            HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
        ),
    )["ok"] is True
    with pytest.raises(ReleaseManifestV2Error, match="roles are incomplete"):
        deploy_readiness.build_gateway_v2_readiness_evidence(
            expected_commit=F90_COMMIT,
            source_commit=F90_COMMIT,
            build_commit=F90_COMMIT,
            gateway_release_manifest=gateway_release,
            validator_release_manifest=validator_release,
            compact_lineage=lineage,
            boot_identities=boots,
            expected_role_config_hashes={
                role: boot["config_hash"] for role, boot in boots.items()
            },
            runtime_readiness=runtime_readiness,
            coordinator_attestation_pcr0=boots["gateway_coordinator"]["pcr0"],
            boot_verifier=verify,
        )
