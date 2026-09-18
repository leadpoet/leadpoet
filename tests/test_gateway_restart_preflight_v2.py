import pytest

from gateway.tee import restart_preflight_v2 as preflight
from gateway.tee.release_manifest_v2 import BUILD_EVIDENCE_SCHEMA_VERSION, build_release_manifest
from gateway.tee.topology import COORDINATOR_ROLE, manifest_document, topology_hash


COMMIT = "1" * 40


def _hash(character):
    return "sha256:" + character * 64


def _release(commit=COMMIT):
    rows = []
    values = {
        "commit_sha": commit,
        "pcr0": "1" * 96,
        "normalized_image_hash": _hash("1"),
        "eif_hash": _hash("2"),
        "source_manifest_hash": _hash("3"),
        "build_identity_hash": _hash("4"),
        "execution_manifest_hash": _hash("5"),
        "dependency_lock_hash": _hash("6"),
        "dockerfile_hash": _hash("7"),
        "topology_hash": topology_hash(),
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
    return build_release_manifest(rows, acceptance_signer_pubkey_hash=_hash("f"))


def _verify(**overrides):
    values = {
        "deploy_commit": COMMIT,
        "release_manifest": _release(),
        "topology_manifest": manifest_document(),
        "topology_mode": "full",
        "instance_type": "r7i.4xlarge",
        "parent_vcpus": 16,
        "parent_memory_mib": 125000,
    }
    values.update(overrides)
    return preflight.verify_gateway_restart_preflight_v2(**values)


def test_full_restart_preflight_accepts_exact_coordinator_release():
    result = _verify()
    assert result["status"] == "ready"
    assert result["deploy_commit"] == COMMIT
    assert result["role_count"] == 1
    assert result["topology_hash"] == topology_hash()


def test_restart_preflight_rejects_release_for_another_commit():
    with pytest.raises(preflight.GatewayRestartPreflightV2Error, match="another commit"):
        _verify(deploy_commit="2" * 40)


@pytest.mark.parametrize(
    "overrides",
    (
        {"instance_type": "r7i.2xlarge"},
        {"parent_vcpus": 8},
        {"parent_memory_mib": 64000},
    ),
)
def test_full_restart_preflight_rejects_wrong_or_undersized_host(overrides):
    with pytest.raises(preflight.GatewayRestartPreflightV2Error, match="requires|insufficient"):
        _verify(**overrides)


def test_component_preflight_keeps_exact_release_without_full_host_floor():
    result = _verify(
        topology_mode="component",
        instance_type="r7i.2xlarge",
        parent_vcpus=8,
        parent_memory_mib=64000,
    )
    assert result["status"] == "ready"
    assert result["role_count"] == 1


def test_capacity_detection_counts_cpus_reserved_by_nitro(monkeypatch):
    monkeypatch.setattr(preflight.os, "sysconf", lambda name: 16)
    monkeypatch.setattr(preflight.os, "cpu_count", lambda: 14)
    assert preflight._configured_processor_count() == 16
