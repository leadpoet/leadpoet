from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gateway.tee import verify_release_artifacts_v2 as artifact_verifier
from gateway.tee.release_archive_v2 import (
    ReleaseArchiveV2Error,
    archive_verified_release,
    load_last_good_release,
    select_release_manifest,
    verify_archive_index,
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
from leadpoet_canonical.attested_v2 import sha256_json


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _role_pcr0s(release):
    return {
        role: release["roles"][role]["pcr0"] for role in sorted(ROLE_SPECS)
    }


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
    release = build_release_manifest(
        rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )
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
    assert len(archived["files"]) == 14
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
    (archive / "tee-enclave-gateway_scoring.eif").write_bytes(b"tampered")
    with pytest.raises(ReleaseArchiveV2Error, match="size mismatch|hash mismatch"):
        verify_archive_directory(archive)


def test_gateway_archive_rejects_eif_rehashed_after_tampering(tmp_path):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "f"
    )
    result = archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=tmp_path / "archive",
    )
    archive = Path(result["archive_path"])
    relative_path = "tee-enclave-gateway_scoring.eif"
    eif_path = archive / relative_path
    eif_path.write_bytes(b"tampered-and-rehashed")

    archive_doc_path = archive / "archive.json"
    archive_doc = json.loads(archive_doc_path.read_text(encoding="utf-8"))
    archive_doc["files"][relative_path] = {
        "sha256": _sha(eif_path.read_bytes()),
        "size_bytes": eif_path.stat().st_size,
    }
    body = {key: value for key, value in archive_doc.items() if key != "archive_hash"}
    archive_doc["archive_hash"] = sha256_json(body)
    archive_doc_path.write_text(json.dumps(archive_doc), encoding="utf-8")

    with pytest.raises(
        ReleaseArchiveV2Error,
        match="gateway EIF differs from local verification",
    ):
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


def test_gateway_archive_pins_verified_last_good_across_failed_builds(tmp_path):
    archive_root = tmp_path / "archive"
    releases = []
    marker = tmp_path / "gateway-last-good.json"
    for character in ("a", "b", "c", "d"):
        gateway_root, eif_root, release_path, release = _release_fixture(
            tmp_path / ("build-" + character), character
        )
        if character == "b":
            marker.write_text(
                json.dumps(
                    {
                        "status": "succeeded",
                        "target_sha": "a" * 40,
                        "role_pcr0s": _role_pcr0s(releases[0]),
                    }
                ),
                encoding="utf-8",
            )
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
            last_good_manifest_path=marker if marker.exists() else None,
        )
        releases.append(release)

    index = verify_archive_index(
        archive_root=archive_root,
        required_commit_sha="a" * 40,
        required_role_pcr0s=_role_pcr0s(releases[0]),
        minimum_releases=3,
        maximum_releases=3,
    )
    assert index["releases"][0]["release_hash"] == releases[-1]["release_hash"]
    assert any(item["commit_sha"] == "a" * 40 for item in index["releases"])


def test_gateway_archive_tolerates_legacy_missing_last_good(tmp_path):
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

    marker = tmp_path / "gateway-last-good.json"
    marker.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "target_sha": "a" * 40,
                "role_pcr0s": _role_pcr0s(releases[0]),
            }
        ),
        encoding="utf-8",
    )
    gateway_root, eif_root, release_path, newest = _release_fixture(
        tmp_path / "build-e", "e"
    )
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
        last_good_manifest_path=marker,
    )
    index = verify_archive_index(
        archive_root=archive_root,
        minimum_releases=3,
        maximum_releases=3,
    )
    assert index["releases"][0]["release_hash"] == newest["release_hash"]
    with pytest.raises(ReleaseArchiveV2Error, match="last-good commit"):
        verify_archive_index(
            archive_root=archive_root,
            required_commit_sha="a" * 40,
            required_role_pcr0s=_role_pcr0s(releases[0]),
            minimum_releases=3,
            maximum_releases=3,
        )


def test_gateway_archive_rejects_corrupt_present_last_good_before_index_change(
    tmp_path,
):
    archive_root = tmp_path / "archive"
    releases = []
    for character in ("a", "b", "c"):
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
    marker = tmp_path / "gateway-last-good.json"
    wrong_pcr0s = _role_pcr0s(releases[0])
    wrong_pcr0s[sorted(ROLE_SPECS)[0]] = "f" * 96
    marker.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "target_sha": "a" * 40,
                "role_pcr0s": wrong_pcr0s,
            }
        ),
        encoding="utf-8",
    )
    original_index = (archive_root / "index.json").read_bytes()
    original_directories = {
        path.name for path in archive_root.iterdir() if path.is_dir()
    }
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build-d", "d"
    )
    with pytest.raises(ReleaseArchiveV2Error, match="role PCR0s"):
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
            last_good_manifest_path=marker,
        )
    assert (archive_root / "index.json").read_bytes() == original_index
    assert {
        path.name for path in archive_root.iterdir() if path.is_dir()
    } == original_directories


def test_gateway_archive_tolerates_genuinely_absent_last_good_marker(tmp_path):
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    archive_root = tmp_path / "archive"
    result = archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
        last_good_manifest_path=tmp_path / "missing-last-good.json",
    )
    assert result["release_hash"] == release["release_hash"]


def test_gateway_cleanup_verifier_binds_complete_last_good_identity(tmp_path):
    archive_root = tmp_path / "archive"
    releases = []
    for character in ("a", "b", "c"):
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

    marker = tmp_path / "gateway-last-good.json"
    marker.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "target_sha": "a" * 40,
                "role_pcr0s": _role_pcr0s(releases[0]),
            }
        ),
        encoding="utf-8",
    )
    last_good = load_last_good_release(marker)
    index = verify_archive_index(
        archive_root=archive_root,
        required_commit_sha=last_good["commit_sha"],
        required_role_pcr0s=last_good["role_pcr0s"],
        minimum_releases=3,
        maximum_releases=3,
    )
    assert len(index["releases"]) == 3

    wrong_pcr0s = dict(last_good["role_pcr0s"])
    wrong_pcr0s[sorted(ROLE_SPECS)[0]] = "f" * 96
    with pytest.raises(ReleaseArchiveV2Error, match="role PCR0s"):
        verify_archive_index(
            archive_root=archive_root,
            required_commit_sha=last_good["commit_sha"],
            required_role_pcr0s=wrong_pcr0s,
            minimum_releases=3,
            maximum_releases=3,
        )


def test_gateway_cleanup_verifier_rejects_symlink_last_good(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    marker = tmp_path / "gateway-last-good.json"
    marker.symlink_to(target)
    with pytest.raises(ReleaseArchiveV2Error, match="non-regular"):
        load_last_good_release(marker)


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
