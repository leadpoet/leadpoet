from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import neurons.auditor_validator as auditor_module


@pytest.mark.parametrize(
    ("environ", "expected"),
    (
        ({}, "http://52.91.135.79:8000"),
        ({"GATEWAY_URL": ""}, "http://52.91.135.79:8000"),
        (
            {"GATEWAY_URL": " https://gateway.example.com/ "},
            "https://gateway.example.com/",
        ),
    ),
)
def test_auditor_gateway_default_and_override(environ, expected):
    assert auditor_module._default_gateway_url(environ) == expected


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

    asyncio.run(auditor.run())

    output = capsys.readouterr().out
    assert output.count("❌ Auditor verification failed") == 1
    assert "Trust level" not in output
    assert "BURN" not in output
    assert sleeps == [30]


def test_v2_404_fails_closed_without_submission(monkeypatch, capsys):
    auditor = _auditor_for_one_verification(None)

    async def missing_v2(_epoch):
        return None

    auditor.fetch_attested_weights_v2 = missing_v2
    auditor.submit_weights_to_chain = lambda *_args, **_kwargs: pytest.fail(
        "missing V2 authority must not submit any vector"
    )

    async def stop_after_wait(_seconds):
        auditor.should_exit = True

    monkeypatch.setattr(auditor_module.asyncio, "sleep", stop_after_wait)
    asyncio.run(auditor.run())

    output = capsys.readouterr().out
    assert "Weights not yet published" in output
    assert "Auditor verification passed" not in output


def test_v2_transport_failure_is_unavailable():
    auditor = _auditor_for_one_verification(None)

    async def unavailable_v2(_epoch):
        return None

    auditor.fetch_attested_weights_v2 = unavailable_v2

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23912))

    assert result is None
    assert status == "v2_unavailable"


@pytest.mark.parametrize(
    ("detail", "warning_expected"),
    (
        ("Not Found", False),
        ("v2 weight bundle not found", False),
        ("finalized v2 weight authority not found", False),
        ("Authoritative V2 weights not found", True),
    ),
)
def test_v2_fetch_recognizes_only_known_absent_authority_responses(
    monkeypatch,
    caplog,
    detail,
    warning_expected,
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

    with caplog.at_level("WARNING"):
        assert asyncio.run(auditor.fetch_attested_weights_v2(23912)) is None
    assert ("auditor_v2_fetch_failed" in caplog.text) is warning_expected


def test_pending_v2_bundle_is_unavailable():
    auditor = _auditor_for_one_verification(None)

    async def pending_v2(_epoch):
        return None

    auditor.fetch_attested_weights_v2 = pending_v2

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23912))

    assert result is None
    assert status == "v2_unavailable"


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
    # The verified V1 path is intentionally present (selected via
    # AUDITOR_WEIGHT_PROTOCOL) so auditors keep submitting while the gateway
    # serves the legacy weight path; it must stay fully verified.
    assert "AUDITOR_WEIGHT_PROTOCOL" in source
    assert "verify_attested_weights_v1" in source


def _protocol_probe_auditor(monkeypatch, protocol):
    if protocol is None:
        monkeypatch.delenv("AUDITOR_WEIGHT_PROTOCOL", raising=False)
    else:
        monkeypatch.setenv("AUDITOR_WEIGHT_PROTOCOL", protocol)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.calls = []

    async def v2_absent(_epoch):
        auditor.calls.append("v2")
        auditor._last_v2_authority_was_absent = True
        return None

    async def v1_bundle(_epoch):
        auditor.calls.append("v1")
        return {"bundle": "v1"}

    auditor.fetch_attested_weights_v2 = v2_absent
    auditor.fetch_attested_weights_v1 = v1_bundle
    auditor.verify_attested_weights_v1 = (
        lambda bundle, *, expected_epoch_id: dict(bundle)
    )
    return auditor


def test_forced_v1_mode_never_requests_the_v2_endpoint(monkeypatch):
    auditor = _protocol_probe_auditor(monkeypatch, "legacy_v1_compat")

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23973))

    assert auditor.calls == ["v1"]
    assert status == "v1_verified"
    assert result == {"bundle": "v1"}


def test_forced_v2_mode_never_falls_back_to_v1(monkeypatch):
    auditor = _protocol_probe_auditor(monkeypatch, "authoritative_v2")

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23973))

    assert auditor.calls == ["v2"]
    assert result is None
    assert status == "v2_unavailable"


def test_auto_mode_uses_verified_v1_only_when_v2_authority_absent(monkeypatch):
    auditor = _protocol_probe_auditor(monkeypatch, None)

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(23973))

    assert auditor.calls == ["v2", "v1"]
    assert status == "v1_verified"
    assert result == {"bundle": "v1"}


def test_unknown_protocol_value_warns_and_runs_auto(monkeypatch, caplog):
    auditor = _protocol_probe_auditor(monkeypatch, "banana")

    with caplog.at_level("WARNING"):
        result, status = asyncio.run(
            auditor.fetch_verified_weight_authority(23973)
        )

    assert "auditor_weight_protocol_invalid" in caplog.text
    assert auditor.calls == ["v2", "v1"]
    assert status == "v1_verified"
