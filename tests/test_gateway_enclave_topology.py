import json
import asyncio
from pathlib import Path

import pytest

from gateway.tee.topology import (
    AUTORESEARCH_ROLE,
    COORDINATOR_ROLE,
    HOST_RESERVED_MEMORY_MIB,
    HOST_RESERVED_VCPUS,
    ROLE_SPECS,
    SCORING_ROLE,
    TopologyError,
    manifest_document,
    validate_manifest,
    validate_production_capacity,
    validate_worker_partition,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee import verify_topology


ROOT = Path(__file__).resolve().parents[1]


def test_approved_three_enclave_topology_is_exact():
    validate_worker_partition()
    assert ROLE_SPECS[COORDINATOR_ROLE]["cid"] == 16
    assert ROLE_SPECS[SCORING_ROLE]["cid"] == 17
    assert ROLE_SPECS[AUTORESEARCH_ROLE]["cid"] == 18
    assert ROLE_SPECS[SCORING_ROLE]["worker_assignment"] == "all_configured"
    assert ROLE_SPECS[AUTORESEARCH_ROLE]["worker_assignment"] == "all_configured"
    assert HOST_RESERVED_VCPUS == 4
    assert HOST_RESERVED_MEMORY_MIB == 40 * 1024


def test_full_topology_requires_r7i_4xlarge_capacity_floor():
    with pytest.raises(TopologyError, match="16 vCPUs"):
        validate_production_capacity(parent_vcpus=8, parent_memory_mib=65536)
    with pytest.raises(TopologyError, match="131072 MiB"):
        validate_production_capacity(parent_vcpus=16, parent_memory_mib=65536)
    capacity = validate_production_capacity(
        parent_vcpus=16,
        parent_memory_mib=128 * 1024,
    )
    assert capacity["host_vcpus"] == 4
    assert capacity["host_memory_mib"] == 40 * 1024


def test_checked_in_topology_manifest_matches_code():
    manifest = json.loads(
        (ROOT / "gateway" / "tee" / "topology.json").read_text(encoding="utf-8")
    )
    assert validate_manifest(manifest) == manifest_document()


def test_topology_manifest_rejects_resource_drift():
    manifest = manifest_document()
    manifest["roles"][SCORING_ROLE]["memory_mib"] -= 1
    with pytest.raises(TopologyError, match="not canonical"):
        validate_manifest(manifest)


def test_restart_allocator_matches_exact_full_topology():
    allocator = (ROOT / "gateway" / "tee" / "configure_allocator.sh").read_text(
        encoding="utf-8"
    )
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert 'REQUIRED_CPUS" -ne 12' in allocator
    assert 'REQUIRED_MEMORY_MIB" -ne 90112' in allocator
    assert "nitro-enclaves-allocator.service" in allocator
    assert 'sudo nitro-cli terminate-enclave --all' in allocator
    assert 'sudo systemctl restart "$ALLOCATOR_SERVICE"' in allocator
    assert 'bash "$GATEWAY_ROOT/tee/configure_allocator.sh"' in restart
    assert restart.index("configure_allocator.sh") < restart.index(
        "bash ./start_enclave.sh"
    )


def _release():
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items()), start=1):
        character = "%x" % index
        values = {
            "commit_sha": "1" * 40,
            "pcr0": character * 96,
            "normalized_image_hash": "sha256:" + character * 64,
            "eif_hash": "sha256:" + character * 64,
            "source_manifest_hash": "sha256:" + "a" * 64,
            "build_identity_hash": "sha256:" + character * 64,
            "execution_manifest_hash": "sha256:" + character * 64,
            "dependency_lock_hash": "sha256:" + "b" * 64,
            "dockerfile_hash": "sha256:" + "c" * 64,
            "topology_hash": manifest_document()["topology_hash"],
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **values,
                    }
                )
    return build_release_manifest(
        rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )


def test_topology_health_matches_exact_approved_role_release(monkeypatch):
    release = _release()
    by_cid = {spec["cid"]: role for role, spec in ROLE_SPECS.items()}

    class Client:
        def __init__(self, cid):
            self.role = by_cid[cid]

        async def role_health(self):
            expected = release["roles"][self.role]
            return {
                "status": "healthy",
                "role": self.role,
                "service_role": ROLE_SPECS[self.role]["service_role"],
                "topology_hash": manifest_document()["topology_hash"],
                "commit_sha": expected["commit_sha"],
                "pcr0": expected["pcr0"],
                "build_identity_hash": expected["build_identity_hash"],
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    result = asyncio.run(
        verify_topology.verify_roles(
            list(ROLE_SPECS),
            release_manifest=release,
        )
    )
    assert {item["role"] for item in result} == set(ROLE_SPECS)


def test_topology_health_rejects_running_pcr_not_in_release(monkeypatch):
    release = _release()
    role = COORDINATOR_ROLE

    class Client:
        def __init__(self, cid):
            assert cid == ROLE_SPECS[role]["cid"]

        async def role_health(self):
            expected = release["roles"][role]
            return {
                "status": "healthy",
                "role": role,
                "service_role": ROLE_SPECS[role]["service_role"],
                "topology_hash": manifest_document()["topology_hash"],
                "commit_sha": expected["commit_sha"],
                "pcr0": "f" * 96,
                "build_identity_hash": expected["build_identity_hash"],
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(verify_topology.TopologyHealthError, match="approved release at pcr0"):
        asyncio.run(verify_topology.verify_roles([role], release_manifest=release))


def test_startup_proves_coordinator_release_before_starting_runners():
    script = (ROOT / "gateway" / "tee" / "start_enclave.sh").read_text(
        encoding="utf-8"
    )
    coordinator = script.index("start_role gateway_coordinator")
    coordinator_health = script.index("verify_roles gateway_coordinator")
    runners = script.index(
        "for role in gateway_scoring gateway_autoresearch"
    )
    final_health = script.index('verify_roles "${ROLES[@]}"')
    assert coordinator < coordinator_health < runners < final_health
    assert '--release-manifest "$RELEASE_MANIFEST"' in script
