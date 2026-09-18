import asyncio
import json
from pathlib import Path

import pytest

from gateway.tee import verify_topology
from gateway.tee.release_manifest_v2 import BUILD_EVIDENCE_SCHEMA_VERSION, build_release_manifest
from gateway.tee.topology import (
    COORDINATOR_ROLE,
    HOST_RESERVED_MEMORY_MIB,
    HOST_RESERVED_VCPUS,
    ROLE_SPECS,
    TopologyError,
    manifest_document,
    validate_manifest,
    validate_production_capacity,
    validate_worker_partition,
)


ROOT = Path(__file__).resolve().parents[1]


def _release():
    rows = []
    values = {
        "commit_sha": "1" * 40,
        "pcr0": "1" * 96,
        "normalized_image_hash": "sha256:" + "1" * 64,
        "eif_hash": "sha256:" + "2" * 64,
        "source_manifest_hash": "sha256:" + "3" * 64,
        "build_identity_hash": "sha256:" + "4" * 64,
        "execution_manifest_hash": "sha256:" + "5" * 64,
        "dependency_lock_hash": "sha256:" + "6" * 64,
        "dockerfile_hash": "sha256:" + "7" * 64,
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
                    "physical_role": COORDINATOR_ROLE,
                    "service_role": COORDINATOR_ROLE,
                    **values,
                }
            )
    return build_release_manifest(rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64)


def _health(release, *, runtime_status="not_configured", transport_status="healthy"):
    expected = release["roles"][COORDINATOR_ROLE]
    return {
        "status": "healthy",
        "role": COORDINATOR_ROLE,
        "service_role": COORDINATOR_ROLE,
        "topology_hash": manifest_document()["topology_hash"],
        "commit_sha": expected["commit_sha"],
        "pcr0": expected["pcr0"],
        "build_identity_hash": expected["build_identity_hash"],
        "v2_runtime": {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "status": runtime_status,
            "physical_role": COORDINATOR_ROLE,
            "service_role": COORDINATOR_ROLE,
        },
        "parent_rpc_transport": {
            "schema_version": "leadpoet.gateway_vsock_rpc_transport_health.v2",
            "status": transport_status,
        },
    }


def test_approved_coordinator_topology_is_exact():
    validate_worker_partition()
    assert set(ROLE_SPECS) == {COORDINATOR_ROLE}
    assert ROLE_SPECS[COORDINATOR_ROLE] == {
        "cid": 16,
        "vcpus": 2,
        "memory_mib": 8 * 1024,
        "service_role": COORDINATOR_ROLE,
    }
    assert HOST_RESERVED_VCPUS == 14
    assert HOST_RESERVED_MEMORY_MIB == 120 * 1024


def test_capacity_and_checked_in_manifest_match_coordinator_policy():
    with pytest.raises(TopologyError, match="16 vCPUs"):
        validate_production_capacity(parent_vcpus=8, parent_memory_mib=128 * 1024)
    capacity = validate_production_capacity(parent_vcpus=16, parent_memory_mib=128 * 1024)
    assert capacity["host_vcpus"] == 14
    assert capacity["host_memory_mib"] == 120 * 1024
    checked_in = json.loads((ROOT / "gateway/tee/topology.json").read_text(encoding="utf-8"))
    assert validate_manifest(checked_in) == manifest_document()


def test_measured_coordinator_release_and_prebootstrap_state_are_required(monkeypatch):
    release = _release()

    class Client:
        def __init__(self, cid):
            assert cid == 16

        async def role_health(self):
            return _health(release)

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    result = asyncio.run(
        verify_topology.verify_roles(
            [COORDINATOR_ROLE], release_manifest=release, prebootstrap_launch_readiness=True
        )
    )
    assert [item["role"] for item in result] == [COORDINATOR_ROLE]


@pytest.mark.parametrize(
    ("runtime_status", "transport_status", "error"),
    (("error", "healthy", "pre-bootstrap"), ("not_configured", "error", "parent_rpc_transport")),
)
def test_coordinator_readiness_fails_closed(monkeypatch, runtime_status, transport_status, error):
    release = _release()

    class Client:
        def __init__(self, cid):
            assert cid == 16

        async def role_health(self):
            return _health(release, runtime_status=runtime_status, transport_status=transport_status)

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(verify_topology.TopologyHealthError, match=error):
        asyncio.run(
            verify_topology.verify_roles(
                [COORDINATOR_ROLE], release_manifest=release, prebootstrap_launch_readiness=True
            )
        )


def test_restart_build_and_launch_have_no_retired_worker_roles():
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    build = (ROOT / "gateway/tee/build_role_enclaves.sh").read_text(encoding="utf-8")
    start = (ROOT / "gateway/tee/start_enclave.sh").read_text(encoding="utf-8")
    combined = restart + build + start
    assert "gateway_scoring" not in combined
    assert "gateway_autoresearch" not in combined
    assert "tee_inter_enclave_relay" not in combined
    assert "tee_egress_forwarder" not in combined
    assert "FULL_LAUNCH_ORDER=(\n  gateway_coordinator\n)" in start
    assert '--release-manifest "$RELEASE_MANIFEST"' in start
