from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

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
    archive_offset = script.index("--archive", verify_offset)
    assert verify_offset < archive_offset
    assert '--retain 3' in script
    assert '--last-good-manifest "$LAST_GOOD_MANIFEST"' in script


def test_cold_role_build_matches_attested_pcr_builder_inputs():
    script = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    build = script[script.index("sudo env") : script.index("sudo python3")]

    assert "DOCKER_BUILDKIT=1" in build
    assert "BUILDX_NO_DEFAULT_ATTESTATIONS=1" in build
    assert "docker build" in build
    assert "--pull" in build
    assert '--build-arg "SOURCE_DATE_EPOCH=0"' in build
    assert '--build-arg "LEADPOET_ENCLAVE_ROLE=${role}"' in build


def test_cold_build_verifies_complete_staging_set_before_transactional_install():
    script = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    staging = script.index(
        'COLD_BUILD_ROOT="$(mktemp -d "$EIF_ROOT/.gateway-eif-cold-build.XXXXXXXX")"'
    )
    build = script.index("sudo nitro-cli build-enclave", staging)
    verify = script.index("verify_release_artifacts_v2.py", build)
    archive = script.index("--archive", verify)
    restore = script.index("--restore", archive)
    cleanup = script.index("cleanup_cold_build_root", restore)

    assert staging < build < verify < archive < restore < cleanup
    cold_slice = script[staging:cleanup]
    assert '--eif-root "$BUILD_EIF_ROOT"' in cold_slice
    assert '--eif-root "$EIF_ROOT"' in script[restore:cleanup]
    assert 'output="$BUILD_EIF_ROOT/tee-enclave-${role}.eif"' in cold_slice
    assert 'output="$EIF_ROOT/tee-enclave-${role}.eif"' not in cold_slice


def test_cold_build_publishes_root_created_eif_to_unprivileged_verifier(
    tmp_path,
):
    script = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    function_start = script.index("publish_built_eif_for_verification() {")
    function_end = script.index("\n}\n", function_start) + len("\n}\n")
    function_source = script[function_start:function_end]
    build_offset = script.index("sudo nitro-cli build-enclave")
    publish_offset = script.index(
        'publish_built_eif_for_verification "$output"', build_offset
    )
    describe_offset = script.index(
        'nitro-cli describe-eif --eif-path "$output"', publish_offset
    )
    verify_offset = script.index("verify_release_artifacts_v2.py", describe_offset)
    assert build_offset < publish_offset < describe_offset < verify_offset

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo_log = tmp_path / "sudo.log"
    sudo = fake_bin / "sudo"
    sudo.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$SUDO_LOG\"\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    sudo.chmod(0o700)
    artifact = tmp_path / "role.eif"
    artifact.write_bytes(b"complete-eif")
    artifact.chmod(0o000)

    result = subprocess.run(
        [
            "bash",
            "-c",
            function_source
            + '\npublish_built_eif_for_verification "$1"\n',
            "publish-eif-test",
            str(artifact),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "SUDO_LOG": str(sudo_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert artifact.read_bytes() == b"complete-eif"
    assert sudo_log.read_text(encoding="utf-8").startswith(
        "chown --no-dereference -- "
    )


def test_cold_build_rejects_symlink_before_privileged_eif_publication(tmp_path):
    script = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    function_start = script.index("publish_built_eif_for_verification() {")
    function_end = script.index("\n}\n", function_start) + len("\n}\n")
    function_source = script[function_start:function_end]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo_log = tmp_path / "sudo.log"
    sudo = fake_bin / "sudo"
    sudo.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$SUDO_LOG\"\n"
        "exit 91\n",
        encoding="utf-8",
    )
    sudo.chmod(0o700)
    target = tmp_path / "target.eif"
    target.write_bytes(b"do-not-publish")
    linked = tmp_path / "linked.eif"
    linked.symlink_to(target)

    result = subprocess.run(
        [
            "bash",
            "-c",
            function_source
            + '\npublish_built_eif_for_verification "$1"\n',
            "publish-eif-test",
            str(linked),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "SUDO_LOG": str(sudo_log),
        },
    )

    assert result.returncode != 0
    assert "unavailable or unsafe" in result.stderr
    assert not sudo_log.exists()
    assert target.read_bytes() == b"do-not-publish"


def test_restart_preserves_exact_role_artifacts_until_verified_restore_or_build():
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    builder = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )

    assert 'rm -f "$GATEWAY_TEE_EIF_ROOT"/enclave-build-*.json' not in restart
    assert 'rm -f "$output" "$measurements"' in builder
    assert restart.index('bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"') < (
        restart.index('bash "$GATEWAY_ROOT/tee/build_role_enclaves.sh"')
    )
    assert "docker_image_normalizer_v2" in builder
    assert (
        'RELEASE_ARCHIVE_ROOT="${GATEWAY_V2_RELEASE_ARCHIVE_ROOT:-$EIF_ROOT/releases-v2}"'
        in builder
    )
