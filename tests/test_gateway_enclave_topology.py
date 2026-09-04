import json
import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from gateway.tee.topology import (
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


def test_approved_two_enclave_topology_is_exact():
    validate_worker_partition()
    assert set(ROLE_SPECS) == {COORDINATOR_ROLE, SCORING_ROLE}
    assert ROLE_SPECS[COORDINATOR_ROLE]["cid"] == 16
    assert ROLE_SPECS[SCORING_ROLE]["cid"] == 17
    assert ROLE_SPECS[SCORING_ROLE]["worker_assignment"] == "all_configured"
    assert HOST_RESERVED_VCPUS == 8
    assert HOST_RESERVED_MEMORY_MIB == 64 * 1024


def test_full_topology_requires_r7i_4xlarge_capacity_floor():
    with pytest.raises(TopologyError, match="16 vCPUs"):
        validate_production_capacity(parent_vcpus=8, parent_memory_mib=65536)
    with pytest.raises(TopologyError, match="131072 MiB"):
        validate_production_capacity(parent_vcpus=16, parent_memory_mib=65536)
    capacity = validate_production_capacity(
        parent_vcpus=16,
        parent_memory_mib=128 * 1024,
    )
    assert capacity["host_vcpus"] == 8
    assert capacity["host_memory_mib"] == 64 * 1024


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
    assert 'REQUIRED_CPUS" -ne 8' in allocator
    assert 'REQUIRED_MEMORY_MIB" -ne 65536' in allocator
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


def _healthy_inter_enclave_transport():
    child = {
        "schema_version": "leadpoet.inter_enclave_transport_health.v2",
        "status": "healthy",
    }
    return {
        "schema_version": "leadpoet.inter_enclave_role_transport_health.v2",
        "status": "healthy",
        "server": dict(child),
        "client": dict(child),
    }


def _prebootstrap_role_health(*, role, release):
    expected = release["roles"][role]
    return {
        "status": "healthy",
        "role": role,
        "service_role": ROLE_SPECS[role]["service_role"],
        "topology_hash": manifest_document()["topology_hash"],
        "commit_sha": expected["commit_sha"],
        "pcr0": expected["pcr0"],
        "build_identity_hash": expected["build_identity_hash"],
        "v2_runtime": {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "status": "not_configured",
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
        },
        "parent_rpc_transport": {
            "schema_version": (
                "leadpoet.gateway_vsock_rpc_transport_health.v2"
            ),
            "status": "healthy",
        },
        "inter_enclave_transport": {
            "schema_version": (
                "leadpoet.inter_enclave_role_transport_health.v2"
            ),
            "status": "error",
            "server": {"status": "unavailable"},
            "client": {"status": "unavailable"},
        },
    }


def test_topology_prebootstrap_readiness_accepts_only_pristine_launch_state(
    monkeypatch,
):
    release = _release()
    by_cid = {spec["cid"]: role for role, spec in ROLE_SPECS.items()}

    class Client:
        def __init__(self, cid):
            self.role = by_cid[cid]

        async def role_health(self):
            return _prebootstrap_role_health(
                role=self.role,
                release=release,
            )

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    result = asyncio.run(
        verify_topology.verify_roles(
            list(ROLE_SPECS),
            release_manifest=release,
            prebootstrap_launch_readiness=True,
        )
    )
    assert {item["role"] for item in result} == set(ROLE_SPECS)


def test_strict_topology_readiness_rejects_prebootstrap_transport_state(
    monkeypatch,
):
    release = _release()
    role = COORDINATOR_ROLE

    class Client:
        def __init__(self, cid):
            assert cid == ROLE_SPECS[role]["cid"]

        async def role_health(self):
            return _prebootstrap_role_health(role=role, release=release)

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(
        verify_topology.TopologyHealthError,
        match="inter_enclave_transport health failed",
    ):
        asyncio.run(
            verify_topology.verify_roles([role], release_manifest=release)
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("v2_runtime", "status"), "error"),
        (("v2_runtime", "schema_version"), "unknown"),
        (("v2_runtime", "error_type"), "RetainedBootstrapError"),
        (("parent_rpc_transport", "status"), "error"),
        (("inter_enclave_transport", "status"), "healthy"),
        (("inter_enclave_transport", "server", "status"), "error"),
        (
            (
                "inter_enclave_transport",
                "server",
                "retained_cleanup_failures",
            ),
            1,
        ),
        (("inter_enclave_transport", "client", "status"), "healthy"),
    ),
)
def test_topology_prebootstrap_readiness_rejects_non_pristine_state(
    monkeypatch,
    path,
    replacement,
):
    release = _release()
    role = COORDINATOR_ROLE
    health = _prebootstrap_role_health(role=role, release=release)
    target = health
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    class Client:
        def __init__(self, cid):
            assert cid == ROLE_SPECS[role]["cid"]

        async def role_health(self):
            return deepcopy(health)

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(
        verify_topology.TopologyHealthError,
        match="pre-bootstrap|parent_rpc_transport health failed",
    ):
        asyncio.run(
            verify_topology.verify_roles(
                [role],
                release_manifest=release,
                prebootstrap_launch_readiness=True,
            )
        )


def test_topology_cli_selects_prebootstrap_launch_readiness(monkeypatch):
    observed = {}

    async def fake_verify_roles(
        roles,
        *,
        release_manifest=None,
        prebootstrap_launch_readiness=False,
    ):
        observed.update(
            roles=list(roles),
            release_manifest=release_manifest,
            prebootstrap_launch_readiness=prebootstrap_launch_readiness,
        )
        return []

    monkeypatch.setattr(verify_topology, "verify_roles", fake_verify_roles)
    assert (
        verify_topology.main(
            ["--prebootstrap-launch-readiness", COORDINATOR_ROLE]
        )
        == 0
    )
    assert observed == {
        "roles": [COORDINATOR_ROLE],
        "release_manifest": None,
        "prebootstrap_launch_readiness": True,
    }


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
                "parent_rpc_transport": {
                    "schema_version": (
                        "leadpoet.gateway_vsock_rpc_transport_health.v2"
                    ),
                    "status": "healthy",
                },
                "inter_enclave_transport": _healthy_inter_enclave_transport(),
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    result = asyncio.run(
        verify_topology.verify_roles(
            list(ROLE_SPECS),
            release_manifest=release,
        )
    )
    assert {item["role"] for item in result} == set(ROLE_SPECS)


def test_topology_health_rejects_latched_rpc_transport_failure(monkeypatch):
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
                "pcr0": expected["pcr0"],
                "build_identity_hash": expected["build_identity_hash"],
                "parent_rpc_transport": {
                    "schema_version": (
                        "leadpoet.gateway_vsock_rpc_transport_health.v2"
                    ),
                    "status": "error",
                },
                "inter_enclave_transport": _healthy_inter_enclave_transport(),
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(
        verify_topology.TopologyHealthError,
        match="parent_rpc_transport health failed",
    ):
        asyncio.run(
            verify_topology.verify_roles([role], release_manifest=release)
        )


@pytest.mark.parametrize(
    "invalid_transport_health",
    (
        None,
        "healthy",
        {"schema_version": "unknown", "status": "healthy"},
        {
            "schema_version": (
                "leadpoet.gateway_vsock_rpc_transport_health.v2"
            ),
            "status": "unknown",
        },
    ),
)
def test_topology_health_rejects_missing_or_unknown_transport_projection(
    monkeypatch,
    invalid_transport_health,
):
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
                "pcr0": expected["pcr0"],
                "build_identity_hash": expected["build_identity_hash"],
                "parent_rpc_transport": invalid_transport_health,
                "inter_enclave_transport": _healthy_inter_enclave_transport(),
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(
        verify_topology.TopologyHealthError,
        match="parent_rpc_transport health failed",
    ):
        asyncio.run(
            verify_topology.verify_roles([role], release_manifest=release)
        )


@pytest.mark.parametrize(
    "invalid_child_health",
    (
        None,
        "healthy",
        {"schema_version": "unknown", "status": "healthy"},
        {
            "schema_version": "leadpoet.inter_enclave_transport_health.v2",
            "status": "unknown",
        },
    ),
)
def test_topology_health_rejects_invalid_inter_enclave_child(
    monkeypatch,
    invalid_child_health,
):
    release = _release()
    role = COORDINATOR_ROLE

    class Client:
        def __init__(self, cid):
            assert cid == ROLE_SPECS[role]["cid"]

        async def role_health(self):
            expected = release["roles"][role]
            transport = _healthy_inter_enclave_transport()
            transport["client"] = invalid_child_health
            return {
                "status": "healthy",
                "role": role,
                "service_role": ROLE_SPECS[role]["service_role"],
                "topology_hash": manifest_document()["topology_hash"],
                "commit_sha": expected["commit_sha"],
                "pcr0": expected["pcr0"],
                "build_identity_hash": expected["build_identity_hash"],
                "parent_rpc_transport": {
                    "schema_version": (
                        "leadpoet.gateway_vsock_rpc_transport_health.v2"
                    ),
                    "status": "healthy",
                },
                "inter_enclave_transport": transport,
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(
        verify_topology.TopologyHealthError,
        match="inter_enclave_transport client health failed",
    ):
        asyncio.run(
            verify_topology.verify_roles([role], release_manifest=release)
        )


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
                "parent_rpc_transport": {
                    "schema_version": (
                        "leadpoet.gateway_vsock_rpc_transport_health.v2"
                    ),
                    "status": "healthy",
                },
                "inter_enclave_transport": _healthy_inter_enclave_transport(),
            }

    monkeypatch.setattr(verify_topology, "TEEClient", Client)
    with pytest.raises(verify_topology.TopologyHealthError, match="approved release at pcr0"):
        asyncio.run(verify_topology.verify_roles([role], release_manifest=release))


def test_startup_allocates_largest_roles_first_before_aggregate_release_check():
    script = (ROOT / "gateway" / "tee" / "start_enclave.sh").read_text(
        encoding="utf-8"
    )
    scoring_order = script.index(
        "FULL_LAUNCH_ORDER=(\n  gateway_scoring\n  gateway_coordinator"
    )
    cleanup = script.index("sudo nitro-cli terminate-enclave --all")
    launch_loop = script.index('for role in "${FULL_LAUNCH_ORDER[@]}"')
    launch_all = script.index(
        'for role in "${FULL_LAUNCH_ORDER[@]}"; do\n'
        '    start_role "$role"\n'
        "  done",
        launch_loop,
    )
    final_health = script.index('wait_for_roles "${ROLES[@]}"')
    final_state = script.index("sudo nitro-cli describe-enclaves", final_health)
    assert (
        scoring_order
        < cleanup
        < launch_loop
        == launch_all
        < final_health
        < final_state
    )
    assert 'wait_for_roles "$role"' not in script[launch_loop:final_health]
    assert "set -euo pipefail" in script
    assert '--release-manifest "$RELEASE_MANIFEST"' in script


def test_startup_retries_strict_role_health_during_enclave_cold_start():
    script = (ROOT / "gateway" / "tee" / "start_enclave.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'ROLE_READY_TIMEOUT_SECONDS="${GATEWAY_TEE_ROLE_READY_TIMEOUT_SECONDS:-180}"'
        in script
    )
    assert (
        'ROLE_READY_RETRY_SECONDS="${GATEWAY_TEE_ROLE_READY_RETRY_SECONDS:-5}"'
        in script
    )
    assert 'if output="$(verify_roles "$@" 2>&1)"; then' in script
    assert 'local verify_args=(--prebootstrap-launch-readiness)' in script
    assert 'if [ "$SECONDS" -ge "$deadline" ]; then' in script
    assert "sudo nitro-cli describe-enclaves >&2 || true" in script
    assert "sleep 15" not in script
