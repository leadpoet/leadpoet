"""Auditor V2 hardening: attestation-time certs, epoch binding, loud failures.

Boot identities are verified for days after issuance while Nitro leaf
certificates live for hours, so the auditor must validate certificates at
attestation time like every internal verifier. A verified-but-stale
authority must never be accepted for a different epoch, a missing PCR0
cache must fail loudly, and the auto-restart wrapper must survive a
nonzero exit under ``set -e``.
"""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

import leadpoet_canonical.auditor_v2 as auditor_v2
import neurons.auditor_validator as auditor_module


def _auditor(netuid=71):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(netuid=netuid)
    auditor._last_v2_authority_was_absent = False
    return auditor


def test_default_boot_verifier_checks_certificates_at_attestation_time():
    # Every verify path must bind the real Nitro verifier with
    # attestation-time certificate validity; an injected test verifier keeps
    # its own signature untouched. The publication/finalization checks are
    # shared, so both authority entrypoints must provably route through the
    # helper that binds the verifier.
    for fn in (
        auditor_v2.verify_attested_weight_bundle_v2,
        auditor_v2._verify_publication_and_finalization,
    ):
        source = inspect.getsource(fn)
        assert "certificate_validity_at_attestation_time=True" in source, fn.__name__
    for fn in (
        auditor_v2.verify_attested_weight_authority_v2,
        auditor_v2.verify_published_weight_authority_stage_v2,
    ):
        source = inspect.getsource(fn)
        assert "_verify_publication_and_finalization" in source, fn.__name__


def test_verified_authority_must_match_requested_epoch_and_netuid(monkeypatch):
    auditor = _auditor(netuid=71)

    async def fake_fetch(_epoch):
        return {"bundle": "raw"}

    async def fake_identity_cache(_authority):
        return {"schema_version": "leadpoet.independent_pcr0_identities.v2", "entries": []}

    stale = {"epoch_id": 24060, "netuid": 71, "uids": [], "weights_u16": []}
    monkeypatch.setattr(auditor, "fetch_attested_weights_v2", fake_fetch)
    monkeypatch.setattr(auditor, "_fetch_release_identity_cache", fake_identity_cache)
    monkeypatch.setattr(
        auditor,
        "verify_attested_weights_v2",
        lambda _b, **_kwargs: dict(stale),
    )
    verified, status = asyncio.run(
        auditor.fetch_verified_weight_authority(24065)
    )
    assert verified is None
    assert status == "v2_invalid"

    fresh = {"epoch_id": 24065, "netuid": 71, "uids": [1], "weights_u16": [65535]}
    monkeypatch.setattr(
        auditor,
        "verify_attested_weights_v2",
        lambda _b, **_kwargs: dict(fresh),
    )
    verified, status = asyncio.run(
        auditor.fetch_verified_weight_authority(24065)
    )
    assert status == "v2_verified"
    assert verified["epoch_id"] == 24065

    wrong_net = {"epoch_id": 24065, "netuid": 72, "uids": [], "weights_u16": []}
    monkeypatch.setattr(
        auditor,
        "verify_attested_weights_v2",
        lambda _b, **_kwargs: dict(wrong_net),
    )
    verified, status = asyncio.run(
        auditor.fetch_verified_weight_authority(24065)
    )
    assert verified is None
    assert status == "v2_invalid"


def test_absent_authority_reports_absent_not_unavailable(monkeypatch):
    auditor = _auditor()

    async def fake_fetch_absent(_epoch):
        auditor._last_v2_authority_was_absent = True
        return None

    monkeypatch.setattr(auditor, "fetch_attested_weights_v2", fake_fetch_absent)
    verified, status = asyncio.run(
        auditor.fetch_verified_weight_authority(24065)
    )
    assert verified is None
    assert status == "v2_absent"

    async def fake_fetch_error(_epoch):
        auditor._last_v2_authority_was_absent = False
        return None

    monkeypatch.setattr(auditor, "fetch_attested_weights_v2", fake_fetch_error)
    verified, status = asyncio.run(
        auditor.fetch_verified_weight_authority(24065)
    )
    assert status == "v2_unavailable"


def test_missing_pcr0_cache_logs_error(monkeypatch, caplog):
    auditor = _auditor()
    monkeypatch.delenv("AUDITOR_INDEPENDENT_PCR0_CACHE_FILE", raising=False)
    with caplog.at_level("ERROR"):
        result = auditor.verify_attested_weights_v2({"bundle": {}})
    assert result is None
    assert any(
        "auditor_v2_pcr0_cache_missing" in record.message
        for record in caplog.records
    )


def test_wrapper_restart_loop_survives_nonzero_exit():
    source = inspect.getsource(auditor_module)
    assert 'python3 neurons/auditor_validator.py "$@" || EXIT_CODE=$?' in source


def test_wrapper_activation_failures_cannot_start_stale_consensus_code():
    source = inspect.getsource(auditor_module)
    assert "Continuing without auto-updates" not in source
    assert (
        "Auditor startup refused: update wrapper could not be created"
        in source
    )
    assert (
        "Auditor startup refused: update wrapper could not be executed"
        in source
    )
