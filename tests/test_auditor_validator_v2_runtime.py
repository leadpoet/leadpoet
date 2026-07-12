from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import neurons.auditor_validator as auditor_module
from leadpoet_canonical.weights import bundle_weights_hash


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("http://52.91.135.79:8000/", "http://52.91.135.79:8000"),
        ("https://gateway.example.com", "https://gateway.example.com"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    ),
)
def test_auditor_accepts_current_http_and_future_https_gateway(value, expected):
    assert auditor_module._normalize_gateway_url(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        "",
        "ftp://gateway.example.com",
        "https://user:password@gateway.example.com",
        "https://gateway.example.com?redirect=attacker",
        "https://gateway.example.com#fragment",
    ),
)
def test_auditor_rejects_non_origin_gateway_urls(value):
    with pytest.raises(RuntimeError, match=r"HTTP\(S\) origin"):
        auditor_module._normalize_gateway_url(value)


def _auditor_for_one_verification(verified_result):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.should_exit = False
    auditor.subtensor = SimpleNamespace(get_current_block=lambda: 345)
    auditor.config = SimpleNamespace(netuid=71)
    auditor.last_submitted_epoch = None
    auditor.consecutive_errors = 0
    auditor.max_consecutive_errors = 5
    auditor.metagraph = object()

    async def fetch(_epoch):
        return {"authority": "fixture"}

    auditor.fetch_attested_weights_v2 = fetch
    auditor.verify_attested_weights_v2 = lambda _bundle: verified_result
    return auditor


def test_failed_auditor_verification_never_submits_fallback(monkeypatch, capsys):
    auditor = _auditor_for_one_verification(None)
    sleeps = []

    async def stop_after_wait(seconds):
        sleeps.append(seconds)
        auditor.should_exit = True

    monkeypatch.setattr(auditor_module.asyncio, "sleep", stop_after_wait)
    auditor.submit_weights_to_chain = lambda *_args, **_kwargs: pytest.fail(
        "verification failure must not submit any vector"
    )

    async def forbidden_v1_fetch(_epoch):
        pytest.fail("invalid V2 authority must not fall back to V1")

    auditor.fetch_attested_weights_v1 = forbidden_v1_fetch

    asyncio.run(auditor.run())

    output = capsys.readouterr().out
    assert output.count("❌ Auditor verification failed") == 1
    assert "Trust level" not in output
    assert "BURN" not in output
    assert sleeps == [30]


def test_v2_404_uses_verified_v1_fallback(monkeypatch, capsys):
    verified = {
        "uids": [1],
        "weights_u16": [65535],
        "validator_hotkey": "5" * 48,
    }
    auditor = _auditor_for_one_verification(None)

    async def missing_v2(_epoch):
        auditor._last_v2_route_was_missing = True
        return None

    async def fetch_v1(_epoch):
        return {"legacy": "fixture"}

    auditor.fetch_attested_weights_v2 = missing_v2
    auditor.fetch_attested_weights_v1 = fetch_v1
    auditor.verify_attested_weights_v1 = lambda *_args, **_kwargs: verified
    auditor.save_pending_equivocation_check = lambda *_args: None

    def submit(_epoch, bundle):
        assert bundle is verified
        auditor.should_exit = True
        return True

    auditor.submit_weights_to_chain = submit

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(auditor_module.asyncio, "sleep", no_wait)
    asyncio.run(auditor.run())

    output = capsys.readouterr().out
    assert output.count("✅ Auditor V1 compatibility verification passed") == 1


def test_v2_transport_failure_does_not_use_v1():
    auditor = _auditor_for_one_verification(None)

    async def unavailable_v2(_epoch):
        auditor._last_v2_route_was_missing = False
        return None

    async def forbidden_v1_fetch(_epoch):
        pytest.fail("transport failure must not downgrade authority")

    auditor.fetch_attested_weights_v2 = unavailable_v2
    auditor.fetch_attested_weights_v1 = forbidden_v1_fetch

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23912))

    assert result is None
    assert status == "v2_unavailable"


@pytest.mark.parametrize(
    ("detail", "route_missing"),
    (
        ("Not Found", True),
        ("Authoritative V2 weights not found", False),
    ),
)
def test_v2_fetch_distinguishes_missing_route_from_pending_bundle(
    monkeypatch,
    detail,
    route_missing,
):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.gateway_url = "https://gateway.example.com"
    auditor.config = SimpleNamespace(netuid=71)

    class Response:
        status = 404

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {"detail": detail}

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(auditor_module.aiohttp, "ClientSession", Session)

    assert asyncio.run(auditor.fetch_attested_weights_v2(23912)) is None
    assert auditor._last_v2_route_was_missing is route_missing


def test_pending_v2_bundle_does_not_use_v1():
    auditor = _auditor_for_one_verification(None)

    async def pending_v2(_epoch):
        auditor._last_v2_route_was_missing = False
        return None

    async def forbidden_v1_fetch(_epoch):
        pytest.fail("an installed V2 route with a pending bundle must not downgrade")

    auditor.fetch_attested_weights_v2 = pending_v2
    auditor.fetch_attested_weights_v1 = forbidden_v1_fetch

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23912))

    assert result is None
    assert status == "v2_unavailable"


def _signed_v1_bundle():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    bundle = {
        "netuid": 71,
        "epoch_id": 23912,
        "block": 23912 * 360 + 345,
        "uids": [1, 7],
        "weights_u16": [65535, 1234],
        "validator_enclave_pubkey": public_key,
        "validator_attestation_b64": "fixture",
        "validator_pcr0": (
            "8b56c0be4cfc55131ce299a6c7f9f2dde56d0cc2e75ffcef5558af5efc60ee3"
            "bd0c5e8b3db822659b50c175ed906a009"
        ),
        "pcr0_commit_hash": "a41e0ad3f440a58893333cef4cf556dbe0d2d3f1",
    }
    bundle["weights_hash"] = bundle_weights_hash(
        bundle["netuid"],
        bundle["epoch_id"],
        bundle["block"],
        list(zip(bundle["uids"], bundle["weights_u16"])),
    )
    bundle["validator_signature"] = private_key.sign(
        bytes.fromhex(bundle["weights_hash"])
    ).hex()
    return bundle


def test_v1_fallback_verifies_hash_signature_and_pcr_binding(monkeypatch):
    auditor = _auditor_for_one_verification(None)
    bundle = _signed_v1_bundle()
    monkeypatch.setattr(
        auditor,
        "_verify_v1_nitro_attestation",
        lambda *_args, **_kwargs: True,
    )

    assert auditor.verify_attested_weights_v1(
        bundle,
        expected_epoch_id=23912,
    ) == bundle

    bundle["weights_u16"][1] += 1
    assert auditor.verify_attested_weights_v1(
        bundle,
        expected_epoch_id=23912,
    ) is None


def test_verifier_failure_emits_no_diagnostic_rows(monkeypatch, caplog):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    monkeypatch.delenv("AUDITOR_INDEPENDENT_PCR0_CACHE_FILE", raising=False)

    with caplog.at_level("DEBUG"):
        assert auditor.verify_attested_weights_v2({"authority": "fixture"}) is None

    assert caplog.text == ""


def test_successful_auditor_verification_has_one_status_line(monkeypatch, capsys):
    verified = {
        "uids": [1],
        "weights_u16": [65535],
        "validator_hotkey": "5" * 48,
        "independent_receipt_identities": ["fixture"],
    }
    auditor = _auditor_for_one_verification(verified)
    auditor.save_pending_equivocation_check = lambda *_args: None

    def submit(*_args, **_kwargs):
        auditor.should_exit = True
        return True

    auditor.submit_weights_to_chain = submit

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(auditor_module.asyncio, "sleep", no_wait)
    asyncio.run(auditor.run())

    output = capsys.readouterr().out
    assert output.count("✅ Auditor verification passed") == 1
    assert "Trust level" not in output
    assert "independent PCR0 identities" not in output


def test_auditor_source_has_no_verification_burn_or_trust_banner():
    source = (
        Path(__file__).resolve().parents[1]
        / "neurons"
        / "auditor_validator.py"
    ).read_text(encoding="utf-8")
    assert "submit_burn_weights_to_uid0" not in source
    assert "AUDITOR VERIFICATION MODE" not in source
    assert "Trust level:" not in source
