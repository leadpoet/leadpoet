from __future__ import annotations

import base64

import pytest

from gateway.research_lab.key_vault import openrouter_kms_encryption_context
from gateway.research_lab.store import canonical_hash
from gateway.research_lab.v2_credential_envelopes import (
    build_openrouter_credential_envelope_v2,
    persist_openrouter_credential_envelope_v2,
)


KEY_REF = "encrypted_ref:openrouter:" + "1" * 32
RAW_KEY = "sk-or-v1-" + "a" * 32


def _encrypted(ciphertext: bytes = b"kms-ciphertext"):
    context = openrouter_kms_encryption_context(
        miner_hotkey="miner-hotkey",
        key_ref=KEY_REF,
    )
    return {
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "kms_key_id": "arn:aws:kms:us-east-1:123456789012:key/key-id",
        "encryption_context_hash": canonical_hash(context),
    }


def test_v2_credential_envelope_commits_secret_without_storing_plaintext():
    envelope = build_openrouter_credential_envelope_v2(
        key_ref=KEY_REF,
        miner_hotkey="miner-hotkey",
        credential_kind="runtime",
        raw_credential=RAW_KEY,
        encrypted=_encrypted(),
    )

    assert envelope["credential_value_hash"].startswith("sha256:")
    assert envelope["ciphertext_blob_hash"].startswith("sha256:")
    assert envelope["key_ref_hash"].startswith("sha256:")
    assert RAW_KEY not in str(envelope)
    assert base64.b64decode(envelope["ciphertext_blob_b64"]) == b"kms-ciphertext"


def test_management_envelope_uses_distinct_job_only_slot():
    envelope = build_openrouter_credential_envelope_v2(
        key_ref=KEY_REF,
        miner_hotkey="miner-hotkey",
        credential_kind="management",
        raw_credential=RAW_KEY,
        encrypted=_encrypted(),
    )
    assert envelope["credential_slot"] == "openrouter_management"
    assert RAW_KEY not in str(envelope)


@pytest.mark.asyncio
async def test_v2_credential_envelope_reregistration_keeps_first_kms_ciphertext(
    monkeypatch,
):
    first = build_openrouter_credential_envelope_v2(
        key_ref=KEY_REF,
        miner_hotkey="miner-hotkey",
        credential_kind="runtime",
        raw_credential=RAW_KEY,
        encrypted=_encrypted(b"first-random-kms-ciphertext"),
    )
    second = build_openrouter_credential_envelope_v2(
        key_ref=KEY_REF,
        miner_hotkey="miner-hotkey",
        credential_kind="runtime",
        raw_credential=RAW_KEY,
        encrypted=_encrypted(b"second-random-kms-ciphertext"),
    )
    stored = {**first, "created_at": "2026-07-10T00:00:00Z"}
    monkeypatch.setattr(
        "gateway.research_lab.v2_credential_envelopes.select_one",
        lambda *_args, **_kwargs: _async_value(stored),
    )
    monkeypatch.setattr(
        "gateway.research_lab.v2_credential_envelopes.insert_row",
        lambda *_args, **_kwargs: pytest.fail("existing envelope must be reused"),
    )

    result = await persist_openrouter_credential_envelope_v2(second)
    assert result["ciphertext_blob_hash"] == first["ciphertext_blob_hash"]


async def _async_value(value):
    return value
