from __future__ import annotations

import asyncio
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import neurons.auditor_validator as auditor_module


def test_auditor_cli_config_enables_sdk_argument_parsing(monkeypatch):
    observed = {}

    def fake_config(parser, *, args):
        observed["parser"] = parser
        observed["args"] = args
        observed["parse_flag"] = os.environ.get("BT_NO_PARSE_CLI_ARGS")
        return SimpleNamespace()

    monkeypatch.delenv("BT_NO_PARSE_CLI_ARGS", raising=False)
    monkeypatch.setattr(auditor_module.bt, "Config", fake_config)
    parser = argparse.ArgumentParser()

    auditor_module._build_bittensor_cli_config(
        parser,
        ["--wallet.name", "auditor_wallet"],
    )

    assert observed == {
        "parser": parser,
        "args": ["--wallet.name", "auditor_wallet"],
        "parse_flag": "false",
    }
    assert "BT_NO_PARSE_CLI_ARGS" not in os.environ


@pytest.mark.parametrize(
    ("environ", "expected"),
    (
        ({}, "https://gateway.subnet71.com"),
        ({"GATEWAY_URL": ""}, "https://gateway.subnet71.com"),
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


@pytest.mark.parametrize(
    ("environ", "expected"),
    (
        ({}, auditor_module.OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT),
        (
            {"AUDITOR_BITTENSOR_ARCHIVE_ENDPOINT": "wss://archive.example:443/"},
            "wss://archive.example:443",
        ),
        (
            {"BITTENSOR_ARCHIVE_ENDPOINT": "ws://10.0.0.10:9944"},
            "ws://10.0.0.10:9944",
        ),
    ),
)
def test_auditor_archive_endpoint_policy(environ, expected):
    assert auditor_module._auditor_archive_endpoint(environ) == expected


@pytest.mark.parametrize(
    "endpoint",
    (
        "ws://8.8.8.8:9944",
        "https://archive.example",
        "wss://user:password@archive.example",
        "wss://archive.example/path",
        "wss://archive.example?redirect=attacker",
    ),
)
def test_auditor_archive_endpoint_rejects_unsafe_origins(endpoint):
    with pytest.raises(auditor_module.SubnetEpochError):
        auditor_module._auditor_archive_endpoint(
            {"AUDITOR_BITTENSOR_ARCHIVE_ENDPOINT": endpoint}
        )


def test_auditor_archive_endpoint_rejects_conflicting_aliases():
    with pytest.raises(auditor_module.SubnetEpochError, match="conflicting"):
        auditor_module._auditor_archive_endpoint(
            {
                "AUDITOR_BITTENSOR_ARCHIVE_ENDPOINT": "wss://one.example",
                "BITTENSOR_ARCHIVE_ENDPOINT": "wss://two.example",
            }
        )


def test_archive_connection_retries_only_the_selected_endpoint(monkeypatch):
    calls = []
    sleeps = []

    def connect(*, network):
        calls.append(network)
        if len(calls) < 3:
            raise TimeoutError("fixture TLS timeout")
        return "archive-subtensor"

    monkeypatch.setattr(auditor_module.bt, "Subtensor", connect)
    monkeypatch.setattr(auditor_module.time, "sleep", sleeps.append)

    assert auditor_module._connect_epoch_archive_subtensor(
        endpoint=auditor_module.OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
        retry_delay_seconds=0.25
    ) == "archive-subtensor"
    assert calls == [auditor_module.OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT] * 3
    assert sleeps == [0.25, 0.25]


def test_archive_connection_fails_closed(monkeypatch):
    def fail(**_kwargs):
        raise TimeoutError("fixture TLS timeout")

    monkeypatch.setattr(auditor_module.bt, "Subtensor", fail)
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(
        auditor_module.SubnetEpochError,
        match="selected trusted epoch archive",
    ):
        auditor_module._connect_epoch_archive_subtensor(attempts=2)


def _auditor_for_one_verification(verified_result):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.should_exit = False
    epoch_state = SimpleNamespace(
        current_block=345,
        workflow_epoch_id=101,
        epoch_block=345,
        blocks_remaining=15,
        identity=23_928,
        deadline_reached=lambda threshold: 345 >= threshold,
    )
    auditor._read_epoch_state = lambda: epoch_state
    auditor.epoch_cutover = object()
    auditor.subtensor = SimpleNamespace(
        get_current_block=lambda: 345,
        metagraph=lambda _netuid: object(),
    )
    auditor.config = SimpleNamespace(netuid=71)
    auditor.last_submitted_epoch = None
    auditor.last_authority_epoch = None
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
        auditor._last_v2_authority_was_absent = True
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


def test_stateful_bundle_epoch_verification_uses_archive_subtensor(monkeypatch):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.epoch_cutover = object()
    auditor.subtensor = object()
    auditor.epoch_archive_subtensor = object()
    auditor.config = SimpleNamespace(netuid=71)
    observed = []

    class Snapshot:
        def settlement_epoch_id(self, cutover):
            assert cutover is auditor.epoch_cutover
            return 101

    def read_snapshot(subtensor, **kwargs):
        observed.append((subtensor, kwargs))
        return Snapshot()

    monkeypatch.setattr(
        auditor_module,
        "read_subnet_epoch_snapshot",
        read_snapshot,
    )

    auditor._verify_stateful_bundle_epoch(
        {"block": 8_637_160, "epoch_id": 101}
    )

    assert observed == [
        (
            auditor.epoch_archive_subtensor,
            {"netuid": 71, "block_number": 8_637_160},
        )
    ]


def test_stateful_bundle_epoch_reconnects_same_archive_after_read_failure(
    monkeypatch,
):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.epoch_cutover = object()
    auditor.epoch_archive_endpoint = "wss://operator-archive.example:443"
    stale = object()
    refreshed = object()
    auditor.epoch_archive_subtensor = stale
    auditor.config = SimpleNamespace(netuid=71)
    observed = []

    class Snapshot:
        def settlement_epoch_id(self, cutover):
            assert cutover is auditor.epoch_cutover
            return 101

    def read_snapshot(subtensor, **kwargs):
        observed.append(("read", subtensor, kwargs))
        if subtensor is stale:
            raise TimeoutError("stale archive socket")
        return Snapshot()

    monkeypatch.setattr(
        auditor_module,
        "read_subnet_epoch_snapshot",
        read_snapshot,
    )
    monkeypatch.setattr(
        auditor_module,
        "_connect_epoch_archive_subtensor",
        lambda *, endpoint: observed.append(("connect", endpoint)) or refreshed,
    )
    monkeypatch.setattr(
        auditor_module,
        "validate_subnet_epoch_cutover_anchor",
        lambda source, cutover, *, expected_archive_endpoint: observed.append(
            ("anchor", source, cutover, expected_archive_endpoint)
        ),
    )

    auditor._verify_stateful_bundle_epoch(
        {"block": 8_637_160, "epoch_id": 101}
    )

    assert auditor.epoch_archive_subtensor is refreshed
    assert observed == [
        (
            "read",
            stale,
            {"netuid": 71, "block_number": 8_637_160},
        ),
        ("connect", "wss://operator-archive.example:443"),
        (
            "anchor",
            refreshed,
            auditor.epoch_cutover,
            "wss://operator-archive.example:443",
        ),
        (
            "read",
            refreshed,
            {"netuid": 71, "block_number": 8_637_160},
        ),
    ]


def test_stateful_bundle_epoch_mismatch_does_not_reconnect(monkeypatch):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.epoch_cutover = object()
    auditor.epoch_archive_endpoint = "wss://archive.example:443"
    auditor.epoch_archive_subtensor = object()
    auditor.config = SimpleNamespace(netuid=71)

    class Snapshot:
        def settlement_epoch_id(self, _cutover):
            return 100

    monkeypatch.setattr(
        auditor_module,
        "read_subnet_epoch_snapshot",
        lambda *_args, **_kwargs: Snapshot(),
    )
    monkeypatch.setattr(
        auditor_module,
        "_connect_epoch_archive_subtensor",
        lambda **_kwargs: pytest.fail("semantic mismatch must not reconnect"),
    )

    with pytest.raises(
        auditor_module.SubnetEpochError,
        match="differs from official chain state",
    ):
        auditor._verify_stateful_bundle_epoch(
            {"block": 8_637_160, "epoch_id": 101}
        )


def test_verifier_failure_emits_no_diagnostic_rows(monkeypatch, caplog):
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    monkeypatch.delenv("AUDITOR_INDEPENDENT_PCR0_CACHE_FILE", raising=False)

    with caplog.at_level("DEBUG"):
        assert auditor.verify_attested_weights_v2({"authority": "fixture"}) is None

    # A missing PCR0 cache is an operator misconfiguration and must be loud;
    # verification failures still emit no per-receipt diagnostic rows.
    assert any(
        "auditor_v2_pcr0_cache_missing" in record.message
        for record in caplog.records
    )
    assert all(
        "auditor_v2_pcr0_cache_missing" in record.message
        for record in caplog.records
    )


def test_successful_auditor_verification_has_one_status_line(monkeypatch, capsys):
    verified = {
        "epoch_id": 101,
        "netuid": 71,
        "uids": [1],
        "weights_u16": [65535],
        "validator_hotkey": "5" * 48,
        "independent_receipt_identities": ["fixture"],
    }
    auditor = _auditor_for_one_verification(verified)
    auditor.save_pending_equivocation_check = lambda *_args: None

    async def fetch_verified(_epoch):
        return verified, "v2_verified"

    auditor.fetch_verified_weight_authority = fetch_verified

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
    assert "AUDITOR_WEIGHT_PROTOCOL" in source
    assert "verify_attested_weights_v1" not in source
    assert "fetch_attested_weights_v1" not in source


@pytest.mark.parametrize("protocol", ["legacy_v1_compat", "auto", "banana"])
def test_non_v2_protocol_is_rejected(monkeypatch, protocol):
    monkeypatch.setenv("AUDITOR_WEIGHT_PROTOCOL", protocol)
    with pytest.raises(RuntimeError, match="must be authoritative_v2"):
        auditor_module.AuditorValidator.auditor_weight_protocol()


def test_authority_candidates_are_current_epoch_only():
    auditor = SimpleNamespace(last_authority_epoch=None)
    assert auditor_module.AuditorValidator._authority_candidate_epochs(
        auditor, 24083
    ) == [24083]

    auditor.last_authority_epoch = 24082
    assert auditor_module.AuditorValidator._authority_candidate_epochs(
        auditor, 24083
    ) == [24083]

    auditor.last_authority_epoch = 24083
    assert auditor_module.AuditorValidator._authority_candidate_epochs(
        auditor, 24083
    ) == []
