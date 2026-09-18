from pathlib import Path
import subprocess
import sys

import pytest

from gateway.tee import release_lineage_v2
from gateway.tee.release_lineage_v2 import (
    ReleaseLineageV2Error,
    build_compact_release_lineage_boot_verifier_v2,
    validate_compact_release_lineage_v2,
    validate_prior_compact_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    ReleaseManifestV2Error,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash


def test_lineage_import_does_not_require_validator_package():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['validator_tee'] = None; "
                "import gateway.tee.release_lineage_v2"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _hash(character):
    return "sha256:" + character * 64


def _release(commit_character):
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        values = {
            "commit_sha": commit_character * 40,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
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
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **values,
                    }
                )
    return build_release_manifest(
        rows,
        acceptance_signer_pubkey_hash=_hash("f"),
    )


def _identity(release, role="gateway_coordinator"):
    expectation = release["roles"][role]
    return {
        "physical_role": role,
        "commit_sha": expectation["commit_sha"],
        "pcr0": expectation["pcr0"],
        "build_manifest_hash": expectation["execution_manifest_hash"],
        "dependency_lock_hash": expectation["dependency_lock_hash"],
    }


def _compact_lineage(current, *historical):
    releases = {}
    for gateway in (current, *historical):
        commit = gateway["commit_sha"]
        roles = {
            role: {
                "commit_sha": summary["commit_sha"],
                "pcr0": summary["pcr0"],
                "build_manifest_hash": summary["execution_manifest_hash"],
                "dependency_lock_hash": summary["dependency_lock_hash"],
            }
            for role, summary in gateway["roles"].items()
        }
        releases[commit] = {
            "channel_hash": _hash(commit[0]),
            "gateway_release_hash": gateway["release_hash"],
            "roles": roles,
        }
    body = {
        "schema_version": "leadpoet.attested_release_lineage.v1",
        "current_commit_sha": current["commit_sha"],
        "current_gateway_release_hash": current["release_hash"],
        "releases": {commit: releases[commit] for commit in sorted(releases)},
    }
    return {**body, "lineage_hash": release_lineage_v2.sha256_json(body)}


def test_compact_lineage_verifies_historical_gateway_boots():
    current = _release("1")
    historical = _release("2")
    lineage = validate_compact_release_lineage_v2(
        _compact_lineage(current, historical),
        expected_current_commit=current["commit_sha"],
        expected_current_gateway_release_hash=current["release_hash"],
    )
    observed = []
    verifier = build_compact_release_lineage_boot_verifier_v2(
        lineage,
        boot_verifier=lambda identity, **kwargs: observed.append(
            (identity["physical_role"], kwargs["expected_pcr0"])
        )
        or identity,
    )
    gateway_boot = _identity(historical)
    assert verifier(gateway_boot) == gateway_boot
    assert observed == [("gateway_coordinator", gateway_boot["pcr0"])]


def test_compact_lineage_accepts_installed_two_role_release_only_as_history():
    current = _release("1")
    historical = _release("2")
    lineage = _compact_lineage(current, historical)
    historical_entry = lineage["releases"][historical["commit_sha"]]
    historical_entry["roles"]["gateway_scoring"] = {
        **historical_entry["roles"]["gateway_coordinator"],
        "pcr0": "e" * 96,
        "build_manifest_hash": _hash("e"),
    }
    body = {key: value for key, value in lineage.items() if key != "lineage_hash"}
    lineage["lineage_hash"] = release_lineage_v2.sha256_json(body)

    validated = validate_compact_release_lineage_v2(
        lineage,
        expected_current_commit=current["commit_sha"],
        expected_current_gateway_release_hash=current["release_hash"],
    )
    verifier = build_compact_release_lineage_boot_verifier_v2(
        validated,
        boot_verifier=lambda identity, **_: identity,
    )
    old_scoring_boot = {
        "physical_role": "gateway_scoring",
        "commit_sha": historical["commit_sha"],
        "pcr0": "e" * 96,
        "build_manifest_hash": _hash("e"),
        "dependency_lock_hash": _hash("3"),
    }
    assert verifier(old_scoring_boot) == old_scoring_boot

    invalid_current = _compact_lineage(current)
    current_entry = invalid_current["releases"][current["commit_sha"]]
    current_entry["roles"]["gateway_scoring"] = {
        **current_entry["roles"]["gateway_coordinator"],
        "pcr0": "e" * 96,
    }
    body = {
        key: value for key, value in invalid_current.items() if key != "lineage_hash"
    }
    invalid_current["lineage_hash"] = release_lineage_v2.sha256_json(body)
    with pytest.raises(ReleaseLineageV2Error, match="roles are incomplete"):
        validate_compact_release_lineage_v2(invalid_current)




def test_compact_lineage_fails_closed_on_hash_role_and_pcr_drift():
    current = _release("1")
    historical = _release("2")
    lineage = _compact_lineage(current, historical)
    with pytest.raises(ReleaseLineageV2Error, match="hash differs"):
        validate_compact_release_lineage_v2(
            {**lineage, "lineage_hash": _hash("9")}
        )

    verifier = build_compact_release_lineage_boot_verifier_v2(
        lineage,
        boot_verifier=lambda identity, **_: identity,
    )
    with pytest.raises(ReleaseLineageV2Error, match="pcr0"):
        verifier({**_identity(historical), "pcr0": "9" * 96})
    with pytest.raises(ReleaseLineageV2Error, match="role is absent"):
        verifier({**_identity(historical), "physical_role": "unknown"})


def test_compact_lineage_rejects_more_than_512_attested_releases():
    current = _release("1")
    lineage = _compact_lineage(current)
    template = next(iter(lineage["releases"].values()))
    releases = {}
    for index in range(513):
        commit = f"{index:040x}"
        releases[commit] = {
            **template,
            "roles": {
                role: {**expectation, "commit_sha": commit}
                for role, expectation in template["roles"].items()
            },
        }
    body = {
        "schema_version": lineage["schema_version"],
        "current_commit_sha": "0" * 40,
        "current_gateway_release_hash": template["gateway_release_hash"],
        "releases": releases,
    }
    with pytest.raises(ReleaseLineageV2Error, match="lineage is invalid"):
        validate_compact_release_lineage_v2(
            {**body, "lineage_hash": release_lineage_v2.sha256_json(body)}
        )
