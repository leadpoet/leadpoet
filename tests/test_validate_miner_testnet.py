from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_miner_testnet", ROOT / "scripts" / "validate_miner_testnet.py"
)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


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
        SCRIPT._require_secret_names(secret, SCRIPT.REQUIRED_ORGANIZER_KEYS)
    assert "private-value" not in str(error.value)


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
