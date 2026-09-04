import json
from pathlib import Path

import pytest

from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
    fetch_release_channel_v2,
    fetch_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    build_local_release_identity,
    validate_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.auditor_v2 import fetch_locked_release_identity_cache
from validator_tee.host.release_v2 import (
    build_local_validator_release_identity,
    build_validator_release,
    validate_validator_release_manifest,
)


COMMIT = "a" * 40


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _gateway_results(commit: str = COMMIT) -> list[dict]:
    results = []
    for index, role in enumerate(sorted(ROLE_SPECS), start=1):
        character = format(index, "x")
        results.append(
            {
                "role": role,
                "commit_sha": commit,
                "pcr0": character * 96,
                "image_id": _hash(character),
                "source_manifest_hash": _hash(character),
                "build_identity_hash": _hash(character),
                "execution_manifest_hash": _hash(character),
                "dependency_lock_hash": _hash(character),
                "dockerfile_hash": _hash(character),
                "topology_hash": topology_hash(),
            }
        )
    return results


def _channel(commit: str = COMMIT) -> dict:
    gateway = build_local_release_identity(_gateway_results(commit))
    release = build_validator_release(
        commit_sha=commit,
        pcr0="e" * 96,
        app_manifest_hash=_hash("1"),
        dependency_lock_hash=_hash("2"),
        normalized_image_hash=_hash("3"),
        eif_hash=_hash("4"),
        dockerfile_hash=_hash("5"),
        base_dockerfile_hash=_hash("6"),
    )
    validator = build_local_validator_release_identity(release)
    return build_release_channel_v2(
        gateway_release_manifest=gateway,
        validator_release_manifest=validator,
    )


def _channel_with_eif_hash(eif_hash: str) -> dict:
    gateway = build_local_release_identity(_gateway_results())
    release = build_validator_release(
        commit_sha=COMMIT,
        pcr0="e" * 96,
        app_manifest_hash=_hash("1"),
        dependency_lock_hash=_hash("2"),
        normalized_image_hash=_hash("3"),
        eif_hash=eif_hash,
        dockerfile_hash=_hash("5"),
        base_dockerfile_hash=_hash("6"),
    )
    return build_release_channel_v2(
        gateway_release_manifest=gateway,
        validator_release_manifest=build_local_validator_release_identity(release),
    )


def test_local_release_identities_validate_without_external_evidence() -> None:
    channel = _channel()
    gateway = validate_release_manifest(channel["gateway_release_manifest"])
    validator = validate_validator_release_manifest(
        channel["validator_release_manifest"]
    )

    assert gateway["commit_sha"] == COMMIT
    assert gateway["verified_build_count"] == len(ROLE_SPECS)
    assert {row["verified_build_count"] for row in gateway["roles"].values()} == {
        1
    }
    assert validator["verified_build_count"] == 1


def test_local_release_channel_precedes_s3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel = _channel()
    gateway_path = tmp_path / "gateway.json"
    validator_path = tmp_path / "validator.json"
    gateway_path.write_text(
        json.dumps(channel["gateway_release_manifest"]), encoding="utf-8"
    )
    validator_path.write_text(
        json.dumps(channel["validator_release_manifest"]), encoding="utf-8"
    )
    monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", COMMIT)
    monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
    monkeypatch.setenv("LEADPOET_LOCAL_VALIDATOR_RELEASE", str(validator_path))

    class NoS3:
        def get_object(self, **_kwargs):
            raise AssertionError("S3 must not be called for the local commit")

    assert fetch_release_channel_v2(
        bucket="unused", commit_sha=COMMIT, s3_client=NoS3()
    ) == channel


def test_installed_lineage_removes_s3_from_next_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    next_commit = "b" * 40
    prior = build_release_lineage_v2(
        [_channel()],
        current_commit=COMMIT,
    )
    current = _channel(next_commit)
    prior_path = tmp_path / "prior-lineage.json"
    gateway_path = tmp_path / "gateway.json"
    validator_path = tmp_path / "validator.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    gateway_path.write_text(
        json.dumps(current["gateway_release_manifest"]), encoding="utf-8"
    )
    validator_path.write_text(
        json.dumps(current["validator_release_manifest"]), encoding="utf-8"
    )
    monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", next_commit)
    monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
    monkeypatch.setenv("LEADPOET_LOCAL_VALIDATOR_RELEASE", str(validator_path))
    monkeypatch.setenv(
        "LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE", str(prior_path)
    )

    class NoS3:
        def get_object(self, **_kwargs):
            raise AssertionError("S3 must not be called for installed lineage")

    lineage = fetch_release_lineage_v2(
        bucket="unused",
        current_commit=next_commit,
        allowed_commits=[COMMIT, next_commit],
        required_commits=[COMMIT, next_commit],
        s3_client=NoS3(),
    )

    assert lineage["current_commit_sha"] == next_commit
    assert set(lineage["releases"]) == {COMMIT, next_commit}


def test_local_channel_identity_ignores_raw_validator_eif_bytes() -> None:
    first = _channel_with_eif_hash(_hash("4"))
    second = _channel_with_eif_hash(_hash("5"))

    assert first["validator_release_manifest"] != second[
        "validator_release_manifest"
    ]
    assert first["channel_hash"] == second["channel_hash"]


def test_auditor_accepts_inline_local_release_identity() -> None:
    cache = fetch_locked_release_identity_cache(
        {
            "schema_version": "leadpoet.auditor_local_release_evidence.v1",
            "commit_sha": COMMIT,
            "release_channel": _channel(),
        }
    )
    assert len(cache["entries"]) == len(ROLE_SPECS) + 1
    assert {entry["verified_build_count"] for entry in cache["entries"]} == {1}


def test_restart_scripts_do_not_wait_for_github_release() -> None:
    root = Path(__file__).resolve().parents[1]
    gateway = (root / "gw_restart.sh").read_text(encoding="utf-8")
    validator = (root / "validator_restart.sh").read_text(encoding="utf-8")
    operator = (root / "scripts/restart_attested_release_local.sh").read_text(
        encoding="utf-8"
    )

    assert "Approved V2 release is not published yet" not in gateway
    assert "Approved V2 release is not published yet" not in validator
    assert "--ensure" not in gateway
    assert "--ensure" not in validator
    assert "build_local_release_v2.sh" in gateway
    assert "build_local_release_v2.sh" in validator
    assert "fetch_release_channel_v2" not in operator


def test_local_release_builder_runs_modules_from_candidate_root() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "gateway/tee/build_local_release_v2.sh").read_text(
        encoding="utf-8"
    )

    candidate_root = script.index('cd "$CANDIDATE_ROOT"')
    gateway_builder = script.index(
        "python3 -m validator_tee.host.gateway_pcr0_builder"
    )
    release_builder = script.index("python3 -m gateway.tee.local_release_v2")
    assert candidate_root < gateway_builder < release_builder
