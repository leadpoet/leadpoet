from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_miner_testnet", ROOT / "scripts" / "validate_miner_testnet.py"
)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_testnet_server_uses_real_gateway_validator_scope():
    source = """
import json
from types import SimpleNamespace
from scripts.validate_miner_testnet import _configure_testnet_registry
_configure_testnet_registry()
from gateway.utils import registry
from lab_arena.service import _gateway_validator_authorizer
def unexpected_global_registry(_hotkey):
    raise AssertionError('standalone Arena must use its own finalized snapshot')
registry.is_registered_hotkey = unexpected_global_registry
snapshot = SimpleNamespace(netuid=401, hotkeys=('test-validator',),
                           active=(True,), validator_permit=(False,), stake=(0.0,))
result = _gateway_validator_authorizer('test-validator', network_name='test', netuid=401,
                                      metagraph=snapshot)
print(json.dumps({'result':result,'network':registry.BITTENSOR_NETWORK,'netuid':registry.BITTENSOR_NETUID}))
"""
    env = dict(os.environ, BITTENSOR_NETWORK="finney", BITTENSOR_NETUID="71", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "result": [True, "validator"], "network": "test", "netuid": 401,
    }


@pytest.mark.parametrize("module_name", ["gateway.config", "gateway.utils.registry"])
def test_testnet_server_refuses_loaded_mainnet_registry(monkeypatch, module_name):
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(
        BITTENSOR_NETWORK="finney", BITTENSOR_NETUID=71,
    ))
    with pytest.raises(SCRIPT.ConfigurationError, match="different chain"):
        SCRIPT._configure_testnet_registry()
    assert os.environ["BITTENSOR_NETWORK"] == "finney"
    assert os.environ["BITTENSOR_NETUID"] == "71"


def test_database_guard_accepts_only_named_loopback_target():
    pytest.importorskip("psycopg2")
    accepted = SCRIPT._database_parameters(
        "host=127.0.0.1 port=55432 dbname=miner_testnet user=postgres",
        expected_database="miner_testnet",
        expected_port=55432,
    )
    assert accepted["host"] == "127.0.0.1"

    for dsn in (
        "host=db.example port=55432 dbname=miner_testnet user=postgres",
        "host=127.0.0.1 port=5432 dbname=miner_testnet user=postgres",
        "host=127.0.0.1 port=55432 dbname=postgres user=postgres",
        "host=127.0.0.1 port=55432 dbname=miner_testnet user=arena",
        "host=127.0.0.1 port=55432 dbname=miner_testnet user=postgres options='-c search_path=bad'",
    ):
        with pytest.raises(SCRIPT.ConfigurationError):
            SCRIPT._database_parameters(
                dsn, expected_database="miner_testnet", expected_port=55432
            )


def test_s3_guard_requires_one_unique_testnet_component():
    assert SCRIPT._validate_s3_prefix("miner-testnet-20260904") == "miner-testnet-20260904"
    for prefix in ("", "miner-testnet", "arena", "miner-testnet-run/child", "../miner-testnet-run"):
        with pytest.raises(SCRIPT.ConfigurationError):
            SCRIPT._validate_s3_prefix(prefix)


def test_gateway_secret_parser_reports_names_without_values():
    secret = SCRIPT._parse_environment_document(
        '{"LAB_ARENA_OPENROUTER_API_KEY":" private-value ","SUPABASE_URL":"https://x.supabase.co","unrelated.lowercase-key":"ignored"}'
    )
    assert secret["LAB_ARENA_OPENROUTER_API_KEY"] == " private-value "
    assert secret["SUPABASE_URL"] == "https://x.supabase.co"
    assert secret["unrelated.lowercase-key"] == "ignored"
    with pytest.raises(SCRIPT.ConfigurationError) as error:
        SCRIPT._require_secret_names(secret, tuple(SCRIPT.ORGANIZER_KEY_ALIASES))
    assert "private-value" not in str(error.value)


def test_provider_key_aliases_accept_legacy_gateway_names_without_exposing_values():
    assert SCRIPT._secret_alias(
        {"OPENROUTER_API_KEY": "private-value"},
        "LAB_ARENA_OPENROUTER_API_KEY",
    ) == "private-value"
    with pytest.raises(SCRIPT.ConfigurationError) as error:
        SCRIPT._secret_alias(
            {
                "LAB_ARENA_OPENROUTER_API_KEY": "first",
                "OPENROUTER_API_KEY": "second",
            },
            "LAB_ARENA_OPENROUTER_API_KEY",
        )
    assert "first" not in str(error.value) and "second" not in str(error.value)


def test_parser_fixes_shadow_limits_and_requires_explicit_resources():
    args = SCRIPT.build_parser().parse_args(
        [
            "serve",
            "--gateway-secret-id",
            "gateway-secret",
            "--chain-endpoint",
            "wss://test.invalid",
            "--cutoff",
            "2026-09-05T04:20:00Z",
            "--kms-key-id",
            "arn:aws:kms:us-east-1:493765492819:key/00000000-0000-0000-0000-000000000000",
            "--s3-prefix",
            "miner-testnet-20260904",
            "--scorer-image",
            "registry.example/repository@sha256:" + "a" * 64,
        ]
    )
    assert args.execution_cap_usd == 5_000_000
    assert args.scoring_cap_usd == 10_000_000
    assert args.runner_hotkey == SCRIPT.DEFAULT_RUNNER
    assert args.miner_hotkey == SCRIPT.DEFAULT_MINER


@pytest.mark.parametrize("service_key", ("", "sb_secret_example"))
def test_managed_transport_uses_arena_environment_without_network(monkeypatch, service_key):
    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_ANON_KEY", "anon")
    monkeypatch.setenv("LAB_ARENA_SERVICE_JWT", "header.payload.signature")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", service_key)
    args = type("Args", (), {"arena_environment_file": None})()

    transport = SCRIPT._managed_postgrest_transport(args)
    try:
        assert "header.payload.signature" not in repr(transport)
        assert "example.supabase.co" in repr(transport)
        assert ("Authorization" in transport._headers) is not bool(service_key)
        assert transport._headers["apikey"] == (service_key or "anon")
    finally:
        transport.close()


def test_managed_driver_is_pinned_and_does_not_create_daily_rounds():
    calls = []

    class Service:
        def advance_round(self, round_id):
            calls.append(("advance", round_id))
            return {"status": "waiting"}

        def ensure_daily_round(self):
            calls.append(("ensure",))
            raise AssertionError("managed driver must not create a daily round")

    assert SCRIPT._advance_pinned(Service(), "arena-2026-09-07-e2e1") == (
        "advanced arena-2026-09-07-e2e1:waiting"
    )
    assert calls == [("advance", "arena-2026-09-07-e2e1")]


def test_serve_pins_the_runtime_service_to_the_requested_round():
    source = inspect.getsource(SCRIPT._serve)
    assert "pinned_round_id=round_id" in source


def test_testnet_database_uses_current_arena_schema_and_review_migration():
    from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS

    assert SCRIPT.EXPECTED_SCHEMA_VERSION == 197
    assert SCRIPT.MIGRATIONS == tuple(
        "scripts/" + migration for migration in DEFAULT_MIGRATIONS
    )


def test_testnet_service_wires_a_separate_full_code_review_worker():
    source = inspect.getsource(SCRIPT._serve)
    assert "SubmissionCodeReviewer(" in source
    assert "credential_for=submission_keys.code_review_key" in source
    assert "code_reviewer=SubmissionCodeReviewer(" in source
    assert "service.review_pending_submissions()" in source
    assert 'name="testnet-arena-code-review"' in source
    assert "review_thread.start()" in source
    assert "review_thread.join(timeout=5)" in source
