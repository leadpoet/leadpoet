"""Regression coverage for weight-path event-loop safety changes."""

from __future__ import annotations

import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


class _Query:
    def __init__(self, rows):
        self.rows = rows
        self.execute_thread = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def range(self, *_args, **_kwargs):
        return self

    def execute(self):
        self.execute_thread = threading.get_ident()
        return SimpleNamespace(data=self.rows)


class _Client:
    def __init__(self, query):
        self.query = query

    def table(self, _name):
        return self.query


@pytest.mark.asyncio
async def test_public_weight_read_executes_off_event_loop(monkeypatch):
    from gateway.api import weights
    from gateway.db import client

    bundle = {
        "netuid": 71,
        "epoch_id": 24103,
        "block": 1,
        "uids": [0],
        "weights_u16": [65535],
        "weights_hash": "sha256:" + "1" * 64,
        "validator_hotkey": "validator",
        "validator_enclave_pubkey": "pubkey",
        "validator_signature": "signature",
        "validator_attestation_b64": "attestation",
        "validator_code_hash": "sha256:" + "2" * 64,
    }
    query = _Query([bundle])
    monkeypatch.setattr(client, "get_read_client", lambda: _Client(query))

    result = await weights.get_latest_weights(71, 24103)

    assert result["weights_hash"] == bundle["weights_hash"]
    assert query.execute_thread is not None
    assert query.execute_thread != threading.get_ident()


@pytest.mark.asyncio
async def test_fulfillment_consensus_read_executes_off_event_loop(monkeypatch):
    from gateway.fulfillment import consensus

    query = _Query([])
    monkeypatch.setattr(consensus, "_get_supabase", lambda: _Client(query))

    assert await consensus._fetch_request_scores("request") == []
    assert query.execute_thread is not None
    assert query.execute_thread != threading.get_ident()


def test_fulfillment_rewards_malformed_200_retries_then_fails(monkeypatch):
    from Leadpoet.utils import cloud_db

    calls = 0

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"unexpected": {}}

    def get(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _Response()

    monkeypatch.setattr(cloud_db.requests, "get", get)
    monkeypatch.setattr(cloud_db.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        cloud_db.gateway_get_all_fulfillment_rewards(
            SimpleNamespace(),
            24103,
        )
    assert calls == 3


def _has_to_thread_call(source_path: Path, target_name: str) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "asyncio"
            and function.attr == "to_thread"
        ):
            continue
        target = node.args[0]
        if isinstance(target, ast.Attribute) and target.attr == target_name:
            return True
    return False


def test_validator_fulfillment_fetch_is_offloaded():
    assert _has_to_thread_call(
        Path("neurons/validator.py"),
        "_get_fulfillment_emission_share",
    )


def test_validation_nonce_check_uses_async_path():
    source = Path("gateway/api/validate.py").read_text(encoding="utf-8")
    assert "await check_and_store_nonce_async(" in source
    assert "check_and_store_nonce," not in source
