from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import neurons.auditor_validator as auditor_module


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
