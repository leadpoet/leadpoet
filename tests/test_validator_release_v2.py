from __future__ import annotations

import copy
import json

import pytest

from validator_tee.host.release_v2 import (
    ValidatorReleaseV2Error,
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
    parse_pcr0,
    validate_validator_release_manifest,
    validate_validator_release,
)


def _hash(character):
    return "sha256:" + character * 64


def _release():
    return build_validator_release(
        commit_sha="1" * 40,
        pcr0="2" * 96,
        app_manifest_hash=_hash("3"),
        dependency_lock_hash=_hash("4"),
        normalized_image_hash=_hash("5"),
        eif_hash=_hash("6"),
        dockerfile_hash=_hash("7"),
        base_dockerfile_hash=_hash("8"),
    )


def test_validator_release_binds_every_deterministic_build_output():
    value = _release()
    assert validate_validator_release(value) == value
    assert value["release_hash"].startswith("sha256:")


def test_validator_release_identity_excludes_non_measured_eif_bytes():
    first = _release()
    second = build_validator_release(
        commit_sha=first["commit_sha"],
        pcr0=first["pcr0"],
        app_manifest_hash=first["app_manifest_hash"],
        dependency_lock_hash=first["dependency_lock_hash"],
        normalized_image_hash=first["normalized_image_hash"],
        eif_hash=_hash("9"),
        dockerfile_hash=first["dockerfile_hash"],
        base_dockerfile_hash=first["base_dockerfile_hash"],
    )

    assert first["release_hash"] == second["release_hash"]


def test_validator_release_rejects_tampering_and_debug_pcr():
    value = copy.deepcopy(_release())
    value["app_manifest_hash"] = _hash("9")
    with pytest.raises(ValidatorReleaseV2Error, match="hash mismatch"):
        validate_validator_release(value)
    with pytest.raises(ValidatorReleaseV2Error, match="PCR0"):
        build_validator_release(
            commit_sha="1" * 40,
            pcr0="0" * 96,
            app_manifest_hash=_hash("3"),
            dependency_lock_hash=_hash("4"),
            normalized_image_hash=_hash("5"),
            eif_hash=_hash("6"),
            dockerfile_hash=_hash("7"),
            base_dockerfile_hash=_hash("8"),
        )


def test_validator_release_parses_nitro_measurement_output():
    output = "build log\n" + json.dumps(
        {"Measurements": {"PCR0": "a" * 96, "PCR1": "b" * 96}}
    )
    assert parse_pcr0(output) == "a" * 96


def _manifest(release=None):
    release = release or _release()
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
    return build_validator_release_manifest(evidence)


def test_validator_release_manifest_requires_six_identical_builds():
    manifest = _manifest()
    assert validate_validator_release_manifest(manifest) == manifest
    assert manifest["verified_build_count"] == 6
    assert manifest["release"] == _release()


def test_validator_release_manifest_rejects_one_divergent_gateway_build():
    release = _release()
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
    divergent = copy.deepcopy(evidence[-1])
    divergent["release"] = build_validator_release(
        commit_sha="1" * 40,
        pcr0="f" * 96,
        app_manifest_hash=_hash("3"),
        dependency_lock_hash=_hash("4"),
        normalized_image_hash=_hash("5"),
        eif_hash=_hash("6"),
        dockerfile_hash=_hash("7"),
        base_dockerfile_hash=_hash("8"),
    )
    evidence[-1] = divergent
    with pytest.raises(ValidatorReleaseV2Error, match="diverged"):
        build_validator_release_manifest(evidence)


def test_validator_release_manifest_commits_divergent_raw_eif_hashes():
    release = _release()
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
    alternate = build_validator_release(
        commit_sha=release["commit_sha"],
        pcr0=release["pcr0"],
        app_manifest_hash=release["app_manifest_hash"],
        dependency_lock_hash=release["dependency_lock_hash"],
        normalized_image_hash=release["normalized_image_hash"],
        eif_hash=_hash("9"),
        dockerfile_hash=release["dockerfile_hash"],
        base_dockerfile_hash=release["base_dockerfile_hash"],
    )
    evidence[-1] = build_validator_build_evidence(
        alternate,
        builder_domain="validator",
        builder_id="validator-parent",
        build_ordinal=3,
    )

    manifest = build_validator_release_manifest(evidence)

    assert manifest["eif_hashes"] == sorted({_hash("6"), _hash("9")})
    assert manifest["release"]["release_hash"] == release["release_hash"]
