from __future__ import annotations

import json

import pytest

from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
)
from gateway.tee.release_lineage_v2 import (
    build_compact_release_lineage_boot_verifier_v2,
)
from scripts import verify_temporary_testnet_weights as verifier
from tests.test_release_channel_v2 import (
    _gateway_manifest,
    _validator_manifest,
)


CURRENT_COMMIT = "a" * 40
EXTRA_COMMIT = "b" * 40


def _channel(commit):
    return build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(commit),
        validator_release_manifest=_validator_manifest(commit),
    )


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _approved_lineage(tmp_path, *, with_prior):
    current = _channel(CURRENT_COMMIT)
    prior_channel_path = tmp_path / "prior-channel.json"
    prior_lineage_path = tmp_path / "prior-lineage.json"
    channels = []
    if with_prior:
        prior = _channel(verifier.PRIOR_RELEASE_COMMIT)
        _write_json(prior_channel_path, prior)
        _write_json(
            prior_lineage_path,
            build_release_lineage_v2(
                [prior], current_commit=verifier.PRIOR_RELEASE_COMMIT
            ),
        )
        channels.append(prior)
    channels.append(current)
    runtime_lineage = build_release_lineage_v2(
        channels, current_commit=CURRENT_COMMIT
    )
    approved = verifier.build_approved_release_lineage(
        candidate=CURRENT_COMMIT,
        gateway_release=current["gateway_release_manifest"],
        validator_release=current["validator_release_manifest"],
        runtime_lineage=runtime_lineage,
        prior_release_channel_path=prior_channel_path,
        prior_release_lineage_path=prior_lineage_path,
    )
    return current, approved, prior_channel_path, prior_lineage_path


def test_current_only_lineage_is_accepted_when_prior_artifacts_are_absent(
    tmp_path,
):
    _current, approved, _prior_channel, _prior_lineage = _approved_lineage(
        tmp_path, with_prior=False
    )

    assert set(approved["releases"]) == {CURRENT_COMMIT}


def test_exact_prior_and_current_lineage_verifies_a_prior_boot(tmp_path):
    _current, approved, _prior_channel, _prior_lineage = _approved_lineage(
        tmp_path, with_prior=True
    )
    prior_role = approved["releases"][verifier.PRIOR_RELEASE_COMMIT][
        "roles"
    ]["gateway_scoring"]
    prior_boot = {
        "physical_role": "gateway_scoring",
        **prior_role,
    }
    observed = []
    boot_verifier = build_compact_release_lineage_boot_verifier_v2(
        approved,
        boot_verifier=lambda identity, **kwargs: observed.append(kwargs) or identity,
    )

    assert boot_verifier(prior_boot) == prior_boot
    assert observed == [
        {
            "expected_pcr0": prior_role["pcr0"],
            "certificate_validity_at_attestation_time": True,
        }
    ]


@pytest.mark.parametrize("missing_name", ("channel", "lineage"))
def test_partial_prior_artifacts_are_rejected(tmp_path, missing_name):
    current = _channel(CURRENT_COMMIT)
    prior = _channel(verifier.PRIOR_RELEASE_COMMIT)
    prior_channel_path = tmp_path / "prior-channel.json"
    prior_lineage_path = tmp_path / "prior-lineage.json"
    if missing_name != "channel":
        _write_json(prior_channel_path, prior)
    if missing_name != "lineage":
        _write_json(
            prior_lineage_path,
            build_release_lineage_v2(
                [prior], current_commit=verifier.PRIOR_RELEASE_COMMIT
            ),
        )

    with pytest.raises(RuntimeError, match="prior release artifacts are incomplete"):
        verifier.build_approved_release_lineage(
            candidate=CURRENT_COMMIT,
            gateway_release=current["gateway_release_manifest"],
            validator_release=current["validator_release_manifest"],
            runtime_lineage=build_release_lineage_v2(
                [current], current_commit=CURRENT_COMMIT
            ),
            prior_release_channel_path=prior_channel_path,
            prior_release_lineage_path=prior_lineage_path,
        )


def test_prior_lineage_with_an_unapproved_extra_release_is_rejected(tmp_path):
    current = _channel(CURRENT_COMMIT)
    prior = _channel(verifier.PRIOR_RELEASE_COMMIT)
    extra = _channel(EXTRA_COMMIT)
    prior_channel_path = tmp_path / "prior-channel.json"
    prior_lineage_path = tmp_path / "prior-lineage.json"
    _write_json(prior_channel_path, prior)
    _write_json(
        prior_lineage_path,
        build_release_lineage_v2(
            [extra, prior], current_commit=verifier.PRIOR_RELEASE_COMMIT
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="prior release lineage differs from its approved channel",
    ):
        verifier.build_approved_release_lineage(
            candidate=CURRENT_COMMIT,
            gateway_release=current["gateway_release_manifest"],
            validator_release=current["validator_release_manifest"],
            runtime_lineage=build_release_lineage_v2(
                [prior, current], current_commit=CURRENT_COMMIT
            ),
            prior_release_channel_path=prior_channel_path,
            prior_release_lineage_path=prior_lineage_path,
        )


def test_prior_channel_for_an_unapproved_commit_is_rejected(tmp_path):
    current = _channel(CURRENT_COMMIT)
    wrong_prior = _channel(EXTRA_COMMIT)
    prior_channel_path = tmp_path / "prior-channel.json"
    prior_lineage_path = tmp_path / "prior-lineage.json"
    _write_json(prior_channel_path, wrong_prior)
    _write_json(
        prior_lineage_path,
        build_release_lineage_v2(
            [wrong_prior], current_commit=EXTRA_COMMIT
        ),
    )

    with pytest.raises(RuntimeError, match="release channel is for another commit"):
        verifier.build_approved_release_lineage(
            candidate=CURRENT_COMMIT,
            gateway_release=current["gateway_release_manifest"],
            validator_release=current["validator_release_manifest"],
            runtime_lineage=build_release_lineage_v2(
                [current], current_commit=CURRENT_COMMIT
            ),
            prior_release_channel_path=prior_channel_path,
            prior_release_lineage_path=prior_lineage_path,
        )


def test_runtime_lineage_with_an_unapproved_extra_release_is_rejected(tmp_path):
    current = _channel(CURRENT_COMMIT)
    prior = _channel(verifier.PRIOR_RELEASE_COMMIT)
    extra = _channel(EXTRA_COMMIT)
    prior_channel_path = tmp_path / "prior-channel.json"
    prior_lineage_path = tmp_path / "prior-lineage.json"
    _write_json(prior_channel_path, prior)
    _write_json(
        prior_lineage_path,
        build_release_lineage_v2(
            [prior], current_commit=verifier.PRIOR_RELEASE_COMMIT
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="runtime release lineage differs from the approved release set",
    ):
        verifier.build_approved_release_lineage(
            candidate=CURRENT_COMMIT,
            gateway_release=current["gateway_release_manifest"],
            validator_release=current["validator_release_manifest"],
            runtime_lineage=build_release_lineage_v2(
                [prior, extra, current], current_commit=CURRENT_COMMIT
            ),
            prior_release_channel_path=prior_channel_path,
            prior_release_lineage_path=prior_lineage_path,
        )
