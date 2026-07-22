"""Gateway weight endpoints accept boots from the approved release lineage.

Weight bundles embed upstream receipt ancestry produced by earlier approved
gateway deployments. Verifying every gateway boot against only the currently
deployed release manifest rejects legitimate bundles with
"gateway boot differs from approved release at commit_sha" the moment the
gateway redeploys. Non-validator boots from other commits must instead be
verified against the immutable approved release lineage, and unapproved
commits must keep failing closed.
"""

from __future__ import annotations

import pytest

from tests.test_release_lineage_v2 import _identity, _release


@pytest.fixture()
def weights_module(monkeypatch):
    from gateway.api import weights

    return weights


def _stub_nitro(calls):
    def verify(identity, *, expected_pcr0, certificate_validity_at_attestation_time):
        calls.append(
            {
                "commit": identity.get("commit_sha"),
                "expected_pcr0": expected_pcr0,
                "issuance_time_validity": certificate_validity_at_attestation_time,
            }
        )
        return {"verified": True, "pcr0": expected_pcr0}

    return verify


def test_current_release_boot_keeps_direct_manifest_path(
    weights_module, monkeypatch
):
    current = _release("1")
    calls = []
    monkeypatch.setattr(
        weights_module, "_gateway_v2_release_manifest", lambda: current
    )
    monkeypatch.setattr(
        weights_module, "verify_boot_identity_nitro", _stub_nitro(calls)
    )

    identity = _identity(current)
    result = weights_module._verify_authoritative_v2_boot(dict(identity))

    assert result["verified"] is True
    assert calls[0]["commit"] == identity["commit_sha"]
    assert calls[0]["issuance_time_validity"] is True


def test_historical_release_boot_verifies_via_approved_lineage(
    weights_module, monkeypatch
):
    from gateway.tee import release_lineage_v2

    current = _release("1")
    historical = _release("2")
    fetched = []
    nitro_calls = []

    monkeypatch.setattr(
        weights_module, "_gateway_v2_release_manifest", lambda: current
    )

    def fetch(commit):
        fetched.append(commit)
        return {"gateway_release_manifest": historical}

    monkeypatch.setattr(release_lineage_v2, "_fetch_historical_release", fetch)
    monkeypatch.setattr(
        release_lineage_v2, "verify_boot_identity_nitro", _stub_nitro(nitro_calls)
    )

    identity = _identity(historical)
    result = weights_module._verify_authoritative_v2_boot(dict(identity))

    assert result["verified"] is True
    assert fetched == ["2" * 40]
    assert nitro_calls[0]["expected_pcr0"] == identity["pcr0"]
    assert nitro_calls[0]["issuance_time_validity"] is True


def test_unapproved_historical_commit_fails_closed(weights_module, monkeypatch):
    from gateway.tee import release_lineage_v2

    current = _release("1")
    historical = _release("2")

    monkeypatch.setattr(
        weights_module, "_gateway_v2_release_manifest", lambda: current
    )

    def fetch(commit):
        raise release_lineage_v2.ReleaseLineageV2Error(
            "historical release channel is unavailable or invalid"
        )

    monkeypatch.setattr(release_lineage_v2, "_fetch_historical_release", fetch)

    with pytest.raises(release_lineage_v2.ReleaseLineageV2Error):
        weights_module._verify_authoritative_v2_boot(dict(_identity(historical)))


def test_lineage_rejects_role_drift_against_its_own_release(
    weights_module, monkeypatch
):
    from gateway.tee import release_lineage_v2

    current = _release("1")
    historical = _release("2")

    monkeypatch.setattr(
        weights_module, "_gateway_v2_release_manifest", lambda: current
    )
    monkeypatch.setattr(
        release_lineage_v2,
        "_fetch_historical_release",
        lambda commit: {"gateway_release_manifest": historical},
    )

    drifted = dict(_identity(historical))
    drifted["build_manifest_hash"] = "sha256:" + "e" * 64
    with pytest.raises(release_lineage_v2.ReleaseLineageV2Error):
        weights_module._verify_authoritative_v2_boot(drifted)


def test_validator_boot_keeps_dynamic_build_verification(
    weights_module, monkeypatch
):
    from gateway.utils import pcr0_builder

    pcr0_calls = []
    nitro_calls = []

    def verify_pcr0(pcr0, *, expected_commit):
        pcr0_calls.append((pcr0, expected_commit))
        return {"valid": True, "commit_hash": expected_commit}

    monkeypatch.setattr(
        pcr0_builder,
        "verify_pcr0",
        verify_pcr0,
    )
    monkeypatch.setattr(
        weights_module, "verify_boot_identity_nitro", _stub_nitro(nitro_calls)
    )

    identity = {
        "physical_role": "validator_weights",
        "commit_sha": "f" * 40,
        "pcr0": "9" * 96,
    }
    result = weights_module._verify_authoritative_v2_boot(dict(identity))

    assert result["verified"] is True
    assert nitro_calls[0]["commit"] == "f" * 40
    assert pcr0_calls == [("9" * 96, "f" * 40)]


def test_validator_boot_selects_exact_commit_when_pcr0_is_shared(
    weights_module, monkeypatch
):
    from gateway.utils import pcr0_builder

    shared_pcr0 = "9" * 96
    current_commit = "a" * 40
    historical_commit = "f" * 40
    monkeypatch.setattr(
        pcr0_builder,
        "_pcr0_cache",
        {
            "current": {
                "pcr0": shared_pcr0,
                "commit_hash": current_commit,
            },
            "historical": {
                "pcr0": shared_pcr0,
                "commit_hash": historical_commit,
            },
        },
    )
    monkeypatch.setattr(
        weights_module,
        "verify_boot_identity_nitro",
        _stub_nitro([]),
    )

    result = weights_module._verify_authoritative_v2_boot(
        {
            "physical_role": "validator_weights",
            "commit_sha": historical_commit,
            "pcr0": shared_pcr0,
        }
    )

    assert result["verified"] is True


def test_validator_boot_rejects_shared_pcr0_without_exact_commit(
    weights_module, monkeypatch
):
    from gateway.utils import pcr0_builder

    monkeypatch.setattr(
        pcr0_builder,
        "_pcr0_cache",
        {
            "other": {
                "pcr0": "9" * 96,
                "commit_hash": "a" * 40,
            }
        },
    )

    with pytest.raises(ValueError, match="commit differs"):
        weights_module._verify_authoritative_v2_boot(
            {
                "physical_role": "validator_weights",
                "commit_sha": "f" * 40,
                "pcr0": "9" * 96,
            }
        )


def test_validator_boot_accepts_exact_commit_alias_for_identical_inputs(
    weights_module, monkeypatch
):
    from gateway.utils import pcr0_builder

    shared_pcr0 = "9" * 96
    original_commit = "a" * 40
    aliased_commit = "f" * 40
    entry = {
        "pcr0": shared_pcr0,
        "commit_hash": original_commit,
        "commit_hashes": [original_commit],
        "commit_timestamp": "1",
    }
    assert pcr0_builder._register_commit_alias(entry, aliased_commit, "2") is True
    monkeypatch.setattr(pcr0_builder, "_pcr0_cache", {"shared": entry})
    monkeypatch.setattr(
        weights_module,
        "verify_boot_identity_nitro",
        _stub_nitro([]),
    )

    result = weights_module._verify_authoritative_v2_boot(
        {
            "physical_role": "validator_weights",
            "commit_sha": aliased_commit,
            "pcr0": shared_pcr0,
        }
    )

    assert result["verified"] is True
    verification = pcr0_builder.verify_pcr0(
        shared_pcr0,
        expected_commit=aliased_commit,
    )
    assert verification["commit_hash"] == aliased_commit
    assert verification["commit_hashes"] == [original_commit, aliased_commit]
