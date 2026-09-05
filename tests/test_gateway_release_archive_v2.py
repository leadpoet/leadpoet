from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from gateway.tee import release_archive_v2 as release_archive
from gateway.tee import verify_release_artifacts_v2 as artifact_verifier
from gateway.tee.release_archive_v2 import (
    ReleaseArchiveCacheMiss,
    ReleaseArchiveV2Error,
    archive_verified_release,
    load_last_good_release,
    main as archive_main,
    restore_verified_release,
    select_release_manifest,
    verify_archive_index,
    verify_archive_directory,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.tee.verify_release_artifacts_v2 import (
    source_manifest_hash,
    verify_release_artifacts,
)
from leadpoet_canonical.attested_v2 import sha256_json


ROOT = Path(__file__).resolve().parents[1]


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


def _convert_archive_to_historical_three_role(archive_root: Path, result):
    archive = Path(result["archive_path"])
    release_path = archive / "gateway-v2-release-manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    roles = copy.deepcopy(release["roles"])
    for summary in roles.values():
        summary["topology_hash"] = HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
    autoresearch = copy.deepcopy(roles["gateway_scoring"])
    autoresearch.update(
        {
            "physical_role": "gateway_autoresearch",
            "service_role": "gateway_autoresearch",
            "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        }
    )
    roles["gateway_autoresearch"] = autoresearch
    body = {
        **{key: value for key, value in release.items() if key != "release_hash"},
        "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        "roles": roles,
        "verified_build_count": 18,
    }
    historical = {**body, "release_hash": sha256_json(body)}
    release_path.write_text(json.dumps(historical), encoding="utf-8")

    for template in (
        "tee-enclave-%s.eif",
        "enclave-build-%s.json",
        "enclave-image-%s.txt",
        "build-identities/%s.json",
    ):
        source = archive / (template % "gateway_scoring")
        destination = archive / (template % "gateway_autoresearch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    verification_path = archive / "gateway-v2-local-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    row = copy.deepcopy(verification["roles"][1])
    row["physical_role"] = "gateway_autoresearch"
    verification["roles"].append(row)
    verification["release_hash"] = historical["release_hash"]
    verification_path.write_text(json.dumps(verification), encoding="utf-8")

    archive_document_path = archive / "archive.json"
    archive_document = json.loads(archive_document_path.read_text(encoding="utf-8"))
    inventory = {}
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        if path == archive_document_path:
            continue
        relative = str(path.relative_to(archive))
        inventory[relative] = {
            "sha256": _sha(path.read_bytes()),
            "size_bytes": path.stat().st_size,
        }
    archive_body = {
        **{
            key: value
            for key, value in archive_document.items()
            if key != "archive_hash"
        },
        "release_hash": historical["release_hash"],
        "files": inventory,
    }
    archived = {**archive_body, "archive_hash": sha256_json(archive_body)}
    archive_document_path.write_text(json.dumps(archived), encoding="utf-8")
    historical_archive = archive_root / historical["release_hash"].split(":", 1)[1]
    archive.rename(historical_archive)
    index_path = archive_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["current_release_hash"] = historical["release_hash"]
    index["releases"][0] = {
        "release_hash": historical["release_hash"],
        "commit_sha": historical["commit_sha"],
        "archive_hash": archived["archive_hash"],
        "archived_at": archived["archived_at"],
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return historical, historical_archive


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
    assert len(archived["files"]) == 2 + 4 * len(ROLE_SPECS)
    assert result["retained_release_count"] == 1


def test_exact_gateway_release_restores_from_verified_archive(tmp_path, monkeypatch):
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    archive_root = tmp_path / "archive"
    archived = archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    (restored_root / "tee-enclave-gateway_scoring.eif").write_bytes(b"stale")
    verified_roots = []
    verifier = release_archive.verify_release_artifacts

    def record_verification(**kwargs):
        verified_roots.append(Path(kwargs["eif_root"]))
        return verifier(**kwargs)

    monkeypatch.setattr(
        release_archive, "verify_release_artifacts", record_verification
    )

    result = restore_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=restored_root,
        archive_root=archive_root,
    )

    assert result["status"] == "restored"
    assert result["release_hash"] == release["release_hash"]
    assert result["commit_sha"] == release["commit_sha"]
    archive_path = Path(archived["archive_path"])
    for relative in release_archive._restored_runtime_files():
        assert (restored_root / relative).read_bytes() == (
            archive_path / relative
        ).read_bytes()
    assert len(verified_roots) == 3
    assert verified_roots[0] == restored_root
    assert verified_roots[-1] == restored_root
    assert not list(tmp_path.glob(".gateway-eif-restore.*"))


def test_current_release_restores_with_historical_three_role_archive_retained(
    tmp_path,
):
    archive_root = tmp_path / "archive"
    old_gateway, old_eifs, old_manifest, _old_release = _release_fixture(
        tmp_path / "old-build", "a"
    )
    old_result = archive_verified_release(
        release_manifest_path=old_manifest,
        gateway_root=old_gateway,
        eif_root=old_eifs,
        archive_root=archive_root,
    )
    historical, historical_archive = _convert_archive_to_historical_three_role(
        archive_root, old_result
    )
    assert verify_archive_directory(historical_archive)["release_hash"] == historical[
        "release_hash"
    ]
    last_good = tmp_path / "last-good.json"
    last_good.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "target_sha": historical["commit_sha"],
                "role_pcr0s": {
                    role: summary["pcr0"]
                    for role, summary in historical["roles"].items()
                },
            }
        ),
        encoding="utf-8",
    )

    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "current-build", "b"
    )
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
        last_good_manifest_path=last_good,
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    result = restore_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=restored_root,
        archive_root=archive_root,
    )
    assert result["release_hash"] == release["release_hash"]
    assert historical_archive.is_dir()

    archive_document_path = historical_archive / "archive.json"
    archive_document = json.loads(archive_document_path.read_text(encoding="utf-8"))
    archive_document["files"]["unknown-role.eif"] = {
        "sha256": _sha(b"unknown"),
        "size_bytes": 7,
    }
    archive_document["archive_hash"] = sha256_json(
        {
            key: value
            for key, value in archive_document.items()
            if key != "archive_hash"
        }
    )
    archive_document_path.write_text(json.dumps(archive_document), encoding="utf-8")
    with pytest.raises(ReleaseArchiveV2Error, match="inventory is incomplete"):
        verify_archive_directory(historical_archive)


def test_exact_gateway_release_skips_reinstall_when_live_set_is_verified(
    tmp_path, monkeypatch
):
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "1"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )

    def unexpected_install(**_kwargs):
        raise AssertionError("verified live EIFs must not be reinstalled")

    monkeypatch.setattr(
        release_archive, "_install_restored_runtime", unexpected_install
    )
    result = restore_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )

    assert result["status"] == "already_installed"
    assert result["release_hash"] == release["release_hash"]


def test_gateway_restore_copies_read_only_live_rollback_backups(
    tmp_path, monkeypatch
):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "2"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    sentinel = restored_root / "tee-enclave-gateway_scoring.eif"
    sentinel.write_bytes(b"old-live-eif")
    sentinel.chmod(0o444)
    copies = []
    copier = release_archive._copy_regular_for_rollback

    def record_copy(source, destination):
        copier(source, destination)
        copies.append((Path(source), Path(destination)))
        assert Path(source).stat().st_ino != Path(destination).stat().st_ino
        assert Path(destination).read_bytes() == b"old-live-eif"

    monkeypatch.setattr(
        release_archive, "_copy_regular_for_rollback", record_copy
    )
    restore_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=restored_root,
        archive_root=archive_root,
    )

    assert any(source == sentinel for source, _destination in copies)


def test_gateway_restore_backup_failure_leaves_live_set_unchanged(
    tmp_path, monkeypatch
):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build-backup-failure", "7"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    original = {}
    for relative in release_archive._restored_runtime_files():
        payload = ("old:" + relative).encode("utf-8")
        (restored_root / relative).write_bytes(payload)
        original[relative] = payload

    copies = 0
    copier = release_archive._copy_regular_for_rollback

    def fail_second_copy(source, destination):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("simulated rollback copy failure")
        copier(source, destination)

    installed = []
    monkeypatch.setattr(
        release_archive, "_copy_regular_for_rollback", fail_second_copy
    )
    monkeypatch.setattr(
        release_archive,
        "_install_replace",
        lambda source, destination: installed.append((source, destination)),
    )

    with pytest.raises(
        ReleaseArchiveV2Error, match="cannot be retained atomically"
    ):
        restore_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=restored_root,
            archive_root=archive_root,
        )

    assert installed == []
    for relative, payload in original.items():
        assert (restored_root / relative).read_bytes() == payload
    assert not list(tmp_path.glob(".gateway-eif-restore.*"))


def test_restore_cli_returns_three_only_for_genuine_cache_miss(
    tmp_path, capsys
):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "b"
    )
    missing_archive = tmp_path / "missing-archive"

    status = archive_main(
        [
            "--restore",
            "--release-manifest",
            str(release_path),
            "--gateway-root",
            str(gateway_root),
            "--eif-root",
            str(eif_root),
            "--archive-root",
            str(missing_archive),
        ]
    )

    assert status == 3
    assert json.loads(capsys.readouterr().out)["status"] == "cache_miss"


def test_exact_gateway_restore_fails_closed_on_present_archive_tamper(tmp_path):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "c"
    )
    archive_root = tmp_path / "archive"
    archived = archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    sentinel = restored_root / "tee-enclave-gateway_scoring.eif"
    sentinel.write_bytes(b"still-live")
    (Path(archived["archive_path"]) / sentinel.name).write_bytes(b"tampered")

    with pytest.raises(ReleaseArchiveV2Error) as error:
        archive_main(
            [
                "--restore",
                "--release-manifest",
                str(release_path),
                "--gateway-root",
                str(gateway_root),
                "--eif-root",
                str(restored_root),
                "--archive-root",
                str(archive_root),
            ]
        )

    assert not isinstance(error.value, ReleaseArchiveCacheMiss)
    assert sentinel.read_bytes() == b"still-live"


def test_exact_gateway_restore_rolls_back_an_atomic_install_failure(
    tmp_path, monkeypatch
):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "d"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    original = {}
    for relative in release_archive._restored_runtime_files():
        payload = ("old:" + relative).encode("utf-8")
        (restored_root / relative).write_bytes(payload)
        original[relative] = payload

    replacements = 0

    def fail_second_replace(source, destination):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("simulated atomic replacement failure")
        os.replace(source, destination)

    monkeypatch.setattr(release_archive, "_install_replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        restore_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=restored_root,
            archive_root=archive_root,
        )

    for relative, payload in original.items():
        assert (restored_root / relative).read_bytes() == payload
    assert not list(tmp_path.glob(".gateway-eif-restore.*"))


def test_exact_gateway_restore_rejects_symlinked_archive_root(tmp_path):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "e"
    )
    real_archive = tmp_path / "real-archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=real_archive,
    )
    linked_archive = tmp_path / "linked-archive"
    linked_archive.symlink_to(real_archive, target_is_directory=True)

    with pytest.raises(ReleaseArchiveV2Error, match="symlink ancestry"):
        restore_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=linked_archive,
        )


def test_gateway_archive_writer_rejects_symlinked_root_and_lock(tmp_path):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "3"
    )
    real_archive = tmp_path / "real-archive"
    real_archive.mkdir()
    linked_archive = tmp_path / "linked-archive"
    linked_archive.symlink_to(real_archive, target_is_directory=True)

    with pytest.raises(ReleaseArchiveV2Error, match="symlink ancestry"):
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=linked_archive,
        )

    lock_target = tmp_path / "lock-target"
    lock_target.write_text("do-not-follow", encoding="utf-8")
    (real_archive / ".archive.lock").symlink_to(lock_target)
    with pytest.raises(ReleaseArchiveV2Error, match="lock is unavailable"):
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=real_archive,
        )
    assert lock_target.read_text(encoding="utf-8") == "do-not-follow"


def test_gateway_role_builder_cold_builds_only_on_exact_archive_miss():
    script = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    restore = script.index("--restore")
    miss = script.index('[ "$restore_status" -ne 3 ]', restore)
    cold_gate = script.index('[ "$RESTORED_EXACT_RELEASE" != "1" ]', miss)
    cold_build = script.index("sudo docker build", cold_gate)
    assert restore < miss < cold_gate < cold_build
    assert "exit \"$restore_status\"" in script[miss:cold_gate]


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


@pytest.mark.parametrize("already_installed", [True, False])
def test_restore_promotes_verified_retained_release_to_archive_head(
    tmp_path, already_installed
):
    archive_root = tmp_path / "archive"
    fixtures = []
    for character in ("a", "b", "c"):
        fixture = _release_fixture(tmp_path / ("build-" + character), character)
        gateway_root, eif_root, release_path, _release = fixture
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
        )
        fixtures.append(fixture)

    gateway_root, built_eif_root, release_path, release = fixtures[1]
    if already_installed:
        eif_root = built_eif_root
    else:
        eif_root = tmp_path / "restored"
        eif_root.mkdir()
    result = restore_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )

    assert result["status"] == (
        "already_installed" if already_installed else "restored"
    )
    index = json.loads((archive_root / "index.json").read_text(encoding="utf-8"))
    assert index["current_release_hash"] == release["release_hash"]
    assert index["releases"][0]["release_hash"] == release["release_hash"]
    assert len(index["releases"]) == 3


def test_failed_restore_does_not_promote_archive_head(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    fixtures = []
    for character in ("a", "b", "c"):
        fixture = _release_fixture(tmp_path / ("build-" + character), character)
        gateway_root, eif_root, release_path, _release = fixture
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
        )
        fixtures.append(fixture)
    original_index = (archive_root / "index.json").read_bytes()
    gateway_root, _eif_root, release_path, _release = fixtures[0]
    restored_root = tmp_path / "restored"
    restored_root.mkdir()

    def fail_install(_source, _destination):
        raise OSError("simulated install failure")

    monkeypatch.setattr(release_archive, "_install_replace", fail_install)
    with pytest.raises(OSError, match="simulated install failure"):
        restore_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=restored_root,
            archive_root=archive_root,
        )

    assert (archive_root / "index.json").read_bytes() == original_index


def test_archive_writer_rejects_noncanonical_retention_bound(tmp_path):
    gateway_root, eif_root, release_path, _release = _release_fixture(
        tmp_path / "build", "8"
    )
    with pytest.raises(ReleaseArchiveV2Error, match="exactly current plus two"):
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=tmp_path / "archive",
            retain_releases=4,
        )


def test_restore_and_archive_reject_oversized_index_before_mutation(tmp_path):
    archive_root = tmp_path / "archive"
    fixtures = []
    for character in ("a", "b", "c"):
        fixture = _release_fixture(tmp_path / ("build-" + character), character)
        gateway_root, eif_root, release_path, _release = fixture
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
        )
        fixtures.append(fixture)
    index_path = archive_root / "index.json"
    oversized = json.loads(index_path.read_text(encoding="utf-8"))
    oversized["releases"].append(dict(oversized["releases"][-1]))
    index_path.write_text(json.dumps(oversized), encoding="utf-8")
    original_index = index_path.read_bytes()

    gateway_root, eif_root, release_path, _release = fixtures[0]
    with pytest.raises(ReleaseArchiveV2Error, match="unbounded"):
        restore_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
        )
    assert index_path.read_bytes() == original_index

    gateway_root, eif_root, release_path, newest = _release_fixture(
        tmp_path / "build-d", "d"
    )
    with pytest.raises(ReleaseArchiveV2Error, match="unbounded"):
        archive_verified_release(
            release_manifest_path=release_path,
            gateway_root=gateway_root,
            eif_root=eif_root,
            archive_root=archive_root,
        )
    assert index_path.read_bytes() == original_index
    assert not (archive_root / newest["release_hash"].split(":", 1)[1]).exists()


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
