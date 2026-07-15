import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    ReleaseManifestV2Error,
    build_release_manifest,
    role_expectation,
    validate_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash


ROOT = Path(__file__).resolve().parents[1]


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _evidence():
    rows = []
    for role_index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        role_character = "abcdef0123456789"[role_index]
        deterministic = {
            "commit_sha": "1" * 40,
            "pcr0": role_character * 96,
            "normalized_image_hash": _hash(role_character),
            "eif_hash": _hash(role_character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(role_character),
            "execution_manifest_hash": _hash(role_character),
            "dependency_lock_hash": _hash("3"),
            "dockerfile_hash": _hash("4"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": "%s-parent" % domain,
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    return rows


def _release(rows=None):
    return build_release_manifest(
        rows or _evidence(),
        acceptance_signer_pubkey_hash=_hash("f"),
    )


def test_release_requires_six_matching_builds_for_every_role():
    release = _release()
    assert validate_release_manifest(release) == release
    assert release["verified_build_count"] == 24
    assert all(value["verified_build_count"] == 6 for value in release["roles"].values())
    expectation = role_expectation(release, "gateway_scoring_a")
    assert expectation["service_role"] == "gateway_scoring"
    assert expectation["release_hash"] == release["release_hash"]


def test_release_rejects_cross_host_pcr_or_image_divergence():
    rows = _evidence()
    rows[0]["pcr0"] = "f" * 96
    with pytest.raises(ReleaseManifestV2Error, match="diverged at pcr0"):
        _release(rows)

    rows = _evidence()
    rows[0]["normalized_image_hash"] = _hash("f")
    with pytest.raises(ReleaseManifestV2Error, match="normalized_image_hash"):
        _release(rows)


def test_release_rejects_missing_or_duplicate_build_evidence():
    rows = _evidence()
    with pytest.raises(ReleaseManifestV2Error, match="exactly 24"):
        _release(rows[:-1])
    rows[-1] = copy.deepcopy(rows[-2])
    with pytest.raises(ReleaseManifestV2Error, match="duplicated"):
        _release(rows)


def test_release_hash_detects_role_summary_tampering():
    release = _release()
    release["roles"]["gateway_coordinator"]["pcr0"] = "f" * 96
    with pytest.raises(ReleaseManifestV2Error, match="hash mismatch"):
        validate_release_manifest(release)


def test_release_assembly_script_accepts_exact_two_parent_evidence(tmp_path):
    rows = _evidence()
    gateway = tmp_path / "gateway.json"
    validator = tmp_path / "validator.json"
    output = tmp_path / "release" / "manifest.json"
    gateway.write_text(
        json.dumps([row for row in rows if row["builder_domain"] == "gateway"]),
        encoding="utf-8",
    )
    validator.write_text(
        json.dumps([row for row in rows if row["builder_domain"] == "validator"]),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "bash",
            str(ROOT / "gateway" / "tee" / "assemble_release_manifest_v2.sh"),
            str(gateway),
            str(validator),
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GATEWAY_V2_ACCEPTANCE_SIGNER_PUBKEY_HASH": _hash("f"),
        },
    )
    assert validate_release_manifest(json.loads(output.read_text()))["commit_sha"] == "1" * 40
    assert output.stat().st_mode & 0o777 == 0o600


def test_gateway_release_evidence_script_requires_three_builds_for_every_role():
    script = (ROOT / "gateway" / "tee" / "build_release_evidence_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "--repetitions 3" in script
    assert "--all-roles" in script
    assert "--builder-domain \"$BUILDER_DOMAIN\"" in script
    assert "--builder-id \"$BUILDER_ID\"" in script
    assert "status --porcelain --untracked-files=no" in script
    assert "normalize_build_evidence" in script
    assert "role/domain/ordinal evidence coverage is incomplete" in script
