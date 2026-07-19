from pathlib import Path
import subprocess
import sys

import pytest

from gateway.tee import release_lineage_v2
from gateway.tee.release_lineage_v2 import (
    ReleaseLineageV2Error,
    build_release_lineage_boot_verifier_v2,
    load_approved_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
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


def _identity(release, role="gateway_scoring"):
    expectation = release["roles"][role]
    return {
        "physical_role": role,
        "commit_sha": expectation["commit_sha"],
        "pcr0": expectation["pcr0"],
        "build_manifest_hash": expectation["execution_manifest_hash"],
        "dependency_lock_hash": expectation["dependency_lock_hash"],
    }


def test_lineage_loads_missing_commit_from_exact_release_channel():
    current = _release("1")
    historical = _release("2")
    calls = []

    def load(commit):
        calls.append(commit)
        return {"gateway_release_manifest": historical}

    releases = load_approved_release_lineage_v2(
        current_release=current,
        parent_graphs=({"boot_identities": [_identity(historical)]},),
        release_channel_loader=load,
    )
    assert set(releases) == {"1" * 40, "2" * 40}
    assert calls == ["2" * 40]


def test_lineage_rejects_channel_for_another_commit():
    current = _release("1")
    historical = _release("2")
    with pytest.raises(
        ReleaseLineageV2Error,
        match="channel commit differs",
    ):
        load_approved_release_lineage_v2(
            current_release=current,
            parent_graphs=({"boot_identities": [_identity(historical)]},),
            release_channel_loader=lambda _commit: {
                "gateway_release_manifest": _release("3")
            },
        )


def test_lineage_verifier_accepts_historical_release_and_rejects_drift(
    monkeypatch,
):
    current = _release("1")
    historical = _release("2")
    observed = []
    monkeypatch.setattr(
        release_lineage_v2,
        "verify_boot_identity_nitro",
        lambda identity, *, expected_pcr0: observed.append(
            (identity["commit_sha"], expected_pcr0)
        )
        or identity,
    )
    verifier = build_release_lineage_boot_verifier_v2(
        {
            current["commit_sha"]: current,
            historical["commit_sha"]: historical,
        }
    )
    identity = _identity(historical)
    assert verifier(identity) == identity
    assert observed == [(historical["commit_sha"], identity["pcr0"])]

    with pytest.raises(ReleaseLineageV2Error, match="dependency_lock_hash"):
        verifier({**identity, "dependency_lock_hash": _hash("9")})
