"""Regression tests for the Langfuse redaction boundary (plan §28.1).

The redactor is the thing that makes any payload exportable to Langfuse —
these tests pin fail-closed blocking for every secret marker and protected
key, hashing of sensitive scalars, and the miner-identity discipline
(raw hotkeys never pass through; ``miner_hotkey_hash`` is the dashboard key).
"""

from __future__ import annotations

import pytest

from research_lab.observability.redaction import (
    BLOCKED_KEYS,
    HASH_VALUE_KEYS,
    SECRET_MARKERS,
    RedactionBlocked,
    assert_langfuse_safe,
    miner_hotkey_hash,
    redact_for_langfuse,
)


# ---------------------------------------------------------------------------
# Fail-closed blocking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", sorted(SECRET_MARKERS))
def test_every_secret_marker_blocks_scalar_values(marker: str) -> None:
    with pytest.raises(RedactionBlocked):
        redact_for_langfuse({"note": f"prefix {marker} suffix"})


@pytest.mark.parametrize("marker", ["sk-or-abc123", "SERVICE_ROLE=x", "Hidden_ICP payload"])
def test_marker_matching_is_case_insensitive(marker: str) -> None:
    with pytest.raises(RedactionBlocked):
        redact_for_langfuse({"note": marker})


@pytest.mark.parametrize("key", sorted(BLOCKED_KEYS))
def test_every_blocked_key_is_rejected(key: str) -> None:
    with pytest.raises(RedactionBlocked):
        redact_for_langfuse({key: "anything"})


def test_blocked_key_rejected_in_nested_structures() -> None:
    with pytest.raises(RedactionBlocked):
        redact_for_langfuse({"outer": [{"deep": {"prompt": "raw text"}}]})


def test_secret_marker_blocked_inside_nested_list() -> None:
    with pytest.raises(RedactionBlocked):
        redact_for_langfuse({"items": [["ok", {"v": "has sk-or-key inside"}]]})


def test_assert_langfuse_safe_raises_on_unsafe() -> None:
    with pytest.raises(RedactionBlocked):
        assert_langfuse_safe({"raw_icp": "sealed content"})


# ---------------------------------------------------------------------------
# Sensitive scalars hash instead of passing through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(HASH_VALUE_KEYS))
def test_hash_value_keys_never_pass_raw_values(key: str) -> None:
    raw = "raw-sensitive-value-xyz"
    result = redact_for_langfuse({key: raw})
    assert raw not in str(result[key])
    assert str(result[key]).startswith("[REDACTED:")


def test_miner_hotkey_key_is_auto_hashed() -> None:
    raw = "5FminerHotkeyRawValue1234567890"
    result = redact_for_langfuse({"miner_hotkey": raw})
    assert raw not in str(result)
    assert str(result["miner_hotkey"]).startswith("[REDACTED:miner_hotkey:")


def test_miner_hotkey_hash_key_passes_through() -> None:
    hashed = miner_hotkey_hash("5FminerHotkeyRawValue1234567890")
    result = redact_for_langfuse({"miner_hotkey_hash": hashed})
    assert result["miner_hotkey_hash"] == hashed


def test_miner_hotkey_hash_helper_is_stable_and_safe() -> None:
    a = miner_hotkey_hash("hotkey-a")
    assert a.startswith("sha256:")
    assert a == miner_hotkey_hash("hotkey-a")
    assert a != miner_hotkey_hash("hotkey-b")
    assert "hotkey-a" not in a


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------


def test_email_redacted_in_prod_mode() -> None:
    result = redact_for_langfuse({"contact": "reach me at jane.doe@example.com"}, mode="prod")
    assert "jane.doe@example.com" not in str(result)
    assert "[REDACTED:email" in str(result["contact"])


def test_email_redacted_inline_in_debug_mode() -> None:
    result = redact_for_langfuse({"contact": "reach jane.doe@example.com now"}, mode="debug")
    assert "jane.doe@example.com" not in str(result)


def test_long_text_summarized_to_hash_in_prod() -> None:
    result = redact_for_langfuse({"blob": "x" * 3000}, mode="prod")
    assert isinstance(result["blob"], dict)
    assert result["blob"]["redacted_length"] == 3000
    assert str(result["blob"]["redacted_summary_hash"]).startswith("sha256:")


# ---------------------------------------------------------------------------
# Safe material passes through untouched
# ---------------------------------------------------------------------------


def test_refs_hashes_counts_and_flags_pass_through() -> None:
    payload = {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "score_bundle_hash": "sha256:" + "a" * 64,
        "execution_trace_ref": "execution_trace:22222222-2222-4222-8222-222222222222",
        "hard_failure_count": 0,
        "candidate_win": True,
        "note": None,
        "labels": ["schema_valid", "build_passed"],
    }
    assert redact_for_langfuse(payload) == payload
