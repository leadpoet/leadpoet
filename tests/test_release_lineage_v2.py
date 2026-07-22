from pathlib import Path
import subprocess
import types
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
        lambda identity, *, expected_pcr0,
        certificate_validity_at_attestation_time: observed.append(
            (
                identity["commit_sha"],
                expected_pcr0,
                certificate_validity_at_attestation_time,
            )
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
    assert observed == [
        (historical["commit_sha"], identity["pcr0"], True)
    ]

    with pytest.raises(ReleaseLineageV2Error, match="dependency_lock_hash"):
        verifier({**identity, "dependency_lock_hash": _hash("9")})


def _validator_identity(commit_char="a"):
    # Validator boots are dynamically built from Git; they carry no gateway
    # release role and are verified via the gateway's PCR0 build cache.
    return {
        "physical_role": "validator_weights",
        "commit_sha": commit_char * 40,
        "pcr0": "9" * 96,
        "boot_identity_hash": "sha256:" + "7" * 64,
        "build_manifest_hash": "sha256:" + "b" * 64,
        "dependency_lock_hash": "sha256:" + "c" * 64,
    }


def test_required_commits_excludes_validator_boots():
    # Regression: an allocation graph embeds finalized weight receipt
    # ancestry, which carries a validator boot. That validator commit must
    # not require a gateway release channel (it has none), or the whole
    # gateway wedges at startup with "unknown release role".
    gateway = _release("1")
    graphs = (
        {
            "boot_identities": [
                _identity(gateway, role="gateway_scoring"),
                _validator_identity(),
            ]
        },
    )
    commits = release_lineage_v2._required_commits(graphs)
    assert gateway["roles"]["gateway_scoring"]["commit_sha"] in commits
    assert _validator_identity()["commit_sha"] not in commits


def test_load_lineage_ignores_validator_commit(monkeypatch):
    gateway = _release("1")
    calls = []

    def load(commit):
        calls.append(commit)
        raise AssertionError("validator commit must never hit the channel loader")

    releases = load_approved_release_lineage_v2(
        current_release=gateway,
        parent_graphs=({"boot_identities": [_validator_identity()]},),
        release_channel_loader=load,
    )
    assert set(releases) == {"1" * 40}
    assert calls == []


def test_verifier_routes_validator_boot_to_dynamic_pcr0(monkeypatch):
    gateway = _release("1")
    validator = _validator_identity()
    seen = {}

    def _verify_pcr0(pcr0):
        seen["pcr0"] = pcr0
        return {"valid": True, "commit_hash": validator["commit_sha"]}

    fake_pcr0 = types.ModuleType("gateway.utils.pcr0_builder")
    fake_pcr0.verify_pcr0 = _verify_pcr0
    monkeypatch.setitem(sys.modules, "gateway.utils.pcr0_builder", fake_pcr0)

    nitro = []
    monkeypatch.setattr(
        release_lineage_v2,
        "verify_boot_identity_nitro",
        lambda identity, *, expected_pcr0, certificate_validity_at_attestation_time: nitro.append(
            (expected_pcr0, certificate_validity_at_attestation_time)
        )
        or {"verified": True},
    )

    verifier = build_release_lineage_boot_verifier_v2(
        {gateway["commit_sha"]: gateway}
    )
    assert verifier(validator) == {"verified": True}
    assert seen["pcr0"] == validator["pcr0"]
    assert nitro == [(validator["pcr0"], True)]


def test_verifier_validator_boot_fails_closed_when_pcr0_absent(monkeypatch):
    gateway = _release("1")
    fake_pcr0 = types.ModuleType("gateway.utils.pcr0_builder")
    fake_pcr0.verify_pcr0 = lambda _pcr0: {"valid": False}
    monkeypatch.setitem(sys.modules, "gateway.utils.pcr0_builder", fake_pcr0)

    verifier = build_release_lineage_boot_verifier_v2(
        {gateway["commit_sha"]: gateway}
    )
    with pytest.raises(ReleaseLineageV2Error, match="dynamic Git build cache"):
        verifier(_validator_identity())


def test_verifier_validator_boot_rejects_commit_mismatch(monkeypatch):
    gateway = _release("1")
    validator = _validator_identity()
    fake_pcr0 = types.ModuleType("gateway.utils.pcr0_builder")
    fake_pcr0.verify_pcr0 = lambda _pcr0: {
        "valid": True,
        "commit_hash": "f" * 40,
    }
    monkeypatch.setitem(sys.modules, "gateway.utils.pcr0_builder", fake_pcr0)

    verifier = build_release_lineage_boot_verifier_v2(
        {gateway["commit_sha"]: gateway}
    )
    with pytest.raises(ReleaseLineageV2Error, match="commit differs"):
        verifier(validator)
