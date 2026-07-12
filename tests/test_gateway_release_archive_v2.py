from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gateway.tee.release_archive_v2 import (
    ReleaseArchiveV2Error,
    archive_verified_release,
    select_release_manifest,
    verify_archive_directory,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.tee.verify_release_artifacts_v2 import (
    source_manifest_hash,
    verify_release_artifacts,
)


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _release_fixture(root: Path, commit_character: str):
    gateway_root = root / "gateway"
    eif_root = root / "eifs"
    context = gateway_root / "_enclave_source"
    context.mkdir(parents=True, exist_ok=True)
    (context / "runtime.py").write_text(
        "RELEASE = %r\n" % commit_character,
        encoding="utf-8",
    )
    dockerfile = gateway_root / "tee" / "Dockerfile.enclave"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    eif_root.mkdir(exist_ok=True)
    rows = []
    for role_index, (role, spec) in enumerate(sorted(ROLE_SPECS.items()), start=1):
        pcr_character = "%x" % role_index
        pcr0 = pcr_character * 96
        eif_bytes = ("eif:%s:%s" % (commit_character, role)).encode("ascii")
        image_id = _sha(("image:%s:%s" % (commit_character, role)).encode("ascii"))
        identity = {
            "commit_sha": commit_character * 40,
            "identity_hash": _sha(("identity:" + role).encode("ascii")),
            "execution_manifest_hash": _sha(("execution:" + role).encode("ascii")),
            "dependency_lock_hash": _sha(b"dependency-lock"),
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
            "nitro output\n" + json.dumps({"Measurements": {"PCR0": pcr0}}),
            encoding="utf-8",
        )
        deterministic = {
            "commit_sha": commit_character * 40,
            "pcr0": pcr0,
            "normalized_image_hash": image_id,
            "eif_hash": _sha(eif_bytes),
            "source_manifest_hash": source_manifest_hash(context),
            "build_identity_hash": identity["identity_hash"],
            "execution_manifest_hash": identity["execution_manifest_hash"],
            "dependency_lock_hash": identity["dependency_lock_hash"],
            "dockerfile_hash": _sha(dockerfile.read_bytes()),
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
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    release = build_release_manifest(rows)
    release_path = eif_root / "gateway-v2-release-manifest.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    verification = verify_release_artifacts(
        release_manifest=release,
        gateway_root=gateway_root,
        eif_root=eif_root,
    )
    (eif_root / "gateway-v2-local-verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    return gateway_root, eif_root, release_path, release


def test_verified_gateway_release_is_archived_as_complete_immutable_set(tmp_path):
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    archive_root = tmp_path / "archive"
    result = archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
        archived_at="2026-07-10T12:00:00Z",
    )
    archived = verify_archive_directory(Path(result["archive_path"]))
    assert archived["release_hash"] == release["release_hash"]
    assert len(archived["files"]) == 18
    assert result["retained_release_count"] == 1


def test_gateway_archive_rejects_artifact_tampering(tmp_path):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "b"
    )
    result = archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=tmp_path / "archive",
    )
    archive = Path(result["archive_path"])
    (archive / "tee-enclave-gateway_scoring_a.eif").write_bytes(b"tampered")
    with pytest.raises(ReleaseArchiveV2Error, match="size mismatch|hash mismatch"):
        verify_archive_directory(archive)


def test_gateway_archive_retains_current_plus_two_predecessors(tmp_path):
    archive_root = tmp_path / "archive"
    releases = []
    for character in ("a", "b", "c", "d"):
        gateway_root, eif_root, release_path, release = _release_fixture(
            tmp_path / ("build-" + character), character
        )
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
        )
        releases.append(release)
    index = json.loads((archive_root / "index.json").read_text())
    assert [item["release_hash"] for item in index["releases"]] == [
        release["release_hash"] for release in reversed(releases[-3:])
    ]
    assert not (archive_root / releases[0]["release_hash"].split(":", 1)[1]).exists()


def test_gateway_rollback_selection_exports_only_a_verified_release(tmp_path):
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "e"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    selected = tmp_path / "selected" / "release.json"
    result = select_release_manifest(
        archive_root=archive_root,
        release_hash=release["release_hash"],
        output=selected,
    )
    assert result["release_hash"] == release["release_hash"]
    assert json.loads(selected.read_text()) == release
    assert selected.stat().st_mode & 0o777 == 0o600
