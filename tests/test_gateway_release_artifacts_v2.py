from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gateway.tee import verify_release_artifacts_v2 as artifact_verifier
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.tee.verify_release_artifacts_v2 import (
    ReleaseArtifactVerificationError,
    source_manifest_hash,
    verify_release_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _describe_fixture_eif(monkeypatch):
    def _read(path: Path) -> str:
        measurement = path.with_name(
            path.name.replace("tee-enclave-", "enclave-build-").replace(
                ".eif", ".json"
            )
        )
        return artifact_verifier._pcr0_from_build_output(measurement)

    monkeypatch.setattr(artifact_verifier, "_pcr0_from_eif", _read)


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path):
    gateway_root = tmp_path / "gateway"
    eif_root = tmp_path / "eifs"
    context = gateway_root / "_enclave_source"
    context.mkdir(parents=True)
    (context / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    dockerfile = gateway_root / "tee" / "Dockerfile.enclave"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    eif_root.mkdir()

    rows = []
    observed = {}
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        pcr0 = ("abcdef0123456789"[index]) * 96
        eif_bytes = ("eif:" + role).encode("ascii")
        image_id = _sha256(("image:" + role).encode("ascii"))
        identity = {
            "commit_sha": "1" * 40,
            "identity_hash": _sha256(("identity:" + role).encode("ascii")),
            "execution_manifest_hash": _sha256(
                ("execution:" + role).encode("ascii")
            ),
            "dependency_lock_hash": _sha256(b"dependency-lock"),
            "topology_hash": topology_hash(),
        }
        identity_path = (
            gateway_root
            / "_attested_runtime"
            / "gateway_enclave_build_identities"
            / (role + ".json")
        )
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        (eif_root / ("tee-enclave-%s.eif" % role)).write_bytes(eif_bytes)
        (eif_root / ("enclave-image-%s.txt" % role)).write_text(
            image_id + "\n", encoding="utf-8"
        )
        (eif_root / ("enclave-build-%s.json" % role)).write_text(
            json.dumps({"Measurements": {"PCR0": pcr0}}), encoding="utf-8"
        )
        values = {
            "commit_sha": "1" * 40,
            "pcr0": pcr0,
            "normalized_image_hash": image_id,
            "eif_hash": _sha256(eif_bytes),
            "source_manifest_hash": source_manifest_hash(context),
            "build_identity_hash": identity["identity_hash"],
            "execution_manifest_hash": identity["execution_manifest_hash"],
            "dependency_lock_hash": identity["dependency_lock_hash"],
            "dockerfile_hash": _sha256(dockerfile.read_bytes()),
            "topology_hash": topology_hash(),
        }
        observed[role] = values
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
    return (
        gateway_root,
        eif_root,
        build_release_manifest(
            rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
        ),
        observed,
    )


def test_local_role_artifacts_must_match_approved_six_build_release(tmp_path):
    gateway_root, eif_root, release, observed = _fixture(tmp_path)
    result = verify_release_artifacts(
        release_manifest=release,
        gateway_root=gateway_root,
        eif_root=eif_root,
    )
    assert result["release_hash"] == release["release_hash"]
    assert {item["physical_role"] for item in result["roles"]} == set(ROLE_SPECS)
    assert result["roles"][0]["eif_hash"] == observed[
        result["roles"][0]["physical_role"]
    ]["eif_hash"]


def test_local_eif_hash_is_recorded_even_when_build_metadata_differs(tmp_path):
    gateway_root, eif_root, release, _observed = _fixture(tmp_path)
    (eif_root / "tee-enclave-gateway_scoring.eif").write_bytes(b"tampered")
    result = verify_release_artifacts(
        release_manifest=release,
        gateway_root=gateway_root,
        eif_root=eif_root,
    )
    role = next(
        item for item in result["roles"] if item["physical_role"] == "gateway_scoring"
    )
    assert role["eif_hash"] == _sha256(b"tampered")


def test_local_eif_pcr0_must_match_its_build_output(tmp_path, monkeypatch):
    gateway_root, eif_root, release, _observed = _fixture(tmp_path)
    monkeypatch.setattr(artifact_verifier, "_pcr0_from_eif", lambda _path: "f" * 96)
    with pytest.raises(
        ReleaseArtifactVerificationError,
        match="EIF PCR0 differs from its build output",
    ):
        verify_release_artifacts(
            release_manifest=release,
            gateway_root=gateway_root,
            eif_root=eif_root,
        )


def test_role_build_archives_only_after_local_release_verification():
    script = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    verify_offset = script.index("verify_release_artifacts_v2.py")
    archive_offset = script.index("gateway.tee.release_archive_v2")
    assert verify_offset < archive_offset
    assert '--retain 3' in script
    assert "docker_image_normalizer_v2" in script
    assert 'RELEASE_ARCHIVE_ROOT="${GATEWAY_V2_RELEASE_ARCHIVE_ROOT:-$EIF_ROOT/releases-v2}"' in script
