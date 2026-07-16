from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import validator_tee.host.release_archive_v2 as release_archive

from validator_tee.host.release_archive_v2 import (
    ValidatorReleaseArchiveV2Error,
    archive_verified_release,
    select_release_manifest,
    verify_archive_directory,
)
from validator_tee.host.release_v2 import (
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
)


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def _describe_fixture_eif(monkeypatch):
    monkeypatch.setattr(
        release_archive,
        "describe_eif_pcr0",
        lambda path: release_archive.parse_pcr0(
            (Path(path).parent / "enclave_build_output.txt").read_text(
                encoding="utf-8"
            )
        ),
    )


def _fixture(root: Path, character: str):
    tee_root = root / "validator_tee"
    tee_root.mkdir(parents=True)
    eif = ("validator-eif:" + character).encode("ascii")
    dockerfile = ("FROM validator-base\n# " + character + "\n").encode("ascii")
    base = b"FROM amazonlinux:2\n"
    pcr0 = character * 96
    (tee_root / "validator-enclave.eif").write_bytes(eif)
    (tee_root / "enclave_build_output.txt").write_text(
        "nitro output\n" + json.dumps({"Measurements": {"PCR0": pcr0}}),
        encoding="utf-8",
    )
    (tee_root / "Dockerfile.enclave").write_bytes(dockerfile)
    (tee_root / "Dockerfile.base").write_bytes(base)
    (tee_root / "runtime-artifacts-v2.lock.json").write_text(
        json.dumps({"release": character}), encoding="utf-8"
    )
    release = build_validator_release(
        commit_sha=character * 40,
        pcr0=pcr0,
        app_manifest_hash=_sha(("app:" + character).encode("ascii")),
        dependency_lock_hash=_sha(("deps:" + character).encode("ascii")),
        normalized_image_hash=_sha(("image:" + character).encode("ascii")),
        eif_hash=_sha(eif),
        dockerfile_hash=_sha(dockerfile),
        base_dockerfile_hash=_sha(base),
    )
    (tee_root / "validator-v2-release.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    evidence = [
        build_validator_build_evidence(
            release,
            builder_domain=domain,
            builder_id=domain + "-parent",
            build_ordinal=ordinal,
        )
        for domain in ("gateway", "validator")
        for ordinal in (1, 2, 3)
    ]
    manifest = build_validator_release_manifest(evidence)
    manifest_path = root / "validator-v2-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tee_root, manifest_path, manifest


def test_validator_archive_requires_and_retains_complete_verified_eif(tmp_path):
    tee_root, manifest_path, manifest = _fixture(tmp_path / "build", "a")
    result = archive_verified_release(
        release_manifest_path=manifest_path,
        validator_tee_root=tee_root,
        archive_root=tmp_path / "archive",
        archived_at="2026-07-10T12:00:00Z",
    )
    archived = verify_archive_directory(Path(result["archive_path"]))
    assert archived["release_hash"] == manifest["release"]["release_hash"]
    assert len(archived["files"]) == 7
    assert result["retained_release_count"] == 1


def test_validator_archive_rejects_tampered_eif(tmp_path):
    tee_root, manifest_path, _manifest = _fixture(tmp_path / "build", "b")
    result = archive_verified_release(
        release_manifest_path=manifest_path,
        validator_tee_root=tee_root,
        archive_root=tmp_path / "archive",
    )
    archive = Path(result["archive_path"])
    (archive / "validator-enclave.eif").write_bytes(b"tampered")
    with pytest.raises(
        ValidatorReleaseArchiveV2Error,
        match="size mismatch|hash mismatch",
    ):
        verify_archive_directory(archive)


def test_validator_archive_rejects_eif_with_different_measured_pcr0(
    tmp_path, monkeypatch
):
    tee_root, manifest_path, _manifest = _fixture(tmp_path / "build", "b")
    result = archive_verified_release(
        release_manifest_path=manifest_path,
        validator_tee_root=tee_root,
        archive_root=tmp_path / "archive",
    )
    monkeypatch.setattr(release_archive, "describe_eif_pcr0", lambda _path: "f" * 96)

    with pytest.raises(
        ValidatorReleaseArchiveV2Error,
        match="EIF PCR0 differs",
    ):
        verify_archive_directory(Path(result["archive_path"]))


def test_validator_archive_retains_current_and_two_predecessors(tmp_path):
    archive_root = tmp_path / "archive"
    manifests = []
    for character in ("a", "b", "c", "d"):
        tee_root, manifest_path, manifest = _fixture(
            tmp_path / ("build-" + character), character
        )
        archive_verified_release(
            release_manifest_path=manifest_path,
            validator_tee_root=tee_root,
            archive_root=archive_root,
        )
        manifests.append(manifest)
    index = json.loads((archive_root / "index.json").read_text())
    assert [item["release_hash"] for item in index["releases"]] == [
        item["release"]["release_hash"] for item in reversed(manifests[-3:])
    ]
    first = manifests[0]["release"]["release_hash"].split(":", 1)[1]
    assert not (archive_root / first).exists()


def test_validator_rollback_selection_exports_verified_manifest(tmp_path):
    tee_root, manifest_path, manifest = _fixture(tmp_path / "build", "e")
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=manifest_path,
        validator_tee_root=tee_root,
        archive_root=archive_root,
    )
    output = tmp_path / "selected" / "validator-release.json"
    selected = select_release_manifest(
        archive_root=archive_root,
        release_hash=manifest["release"]["release_hash"],
        output=output,
    )
    assert selected["commit_sha"] == "e" * 40
    assert json.loads(output.read_text()) == manifest
    assert output.stat().st_mode & 0o777 == 0o600
