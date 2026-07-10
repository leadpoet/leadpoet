"""Free ticket-intake validation: path allowlist checks, exact-direction
refusal dedup with exploration allowance, and signature parity with the
loop engine's refusal events."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from gateway.research_lab import ticket_intake_validation as tiv

ALLOWED_PREFIXES = ("gateway/", "qualification/", "sourcing_model/", "validator_models/")
ALLOWED_EXACT = ("research_lab_adapter.py",)
ALLOWED_SUFFIXES = (".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".md")


def _validate(brief, **kwargs):
    return asyncio.run(
        tiv.validate_ticket_direction(
            brief_public_summary=brief,
            allowed_prefixes=ALLOWED_PREFIXES,
            allowed_exact=ALLOWED_EXACT,
            allowed_suffixes=ALLOWED_SUFFIXES,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# focus-signature parity with the loop engine
# ---------------------------------------------------------------------------


def test_focus_signature_matches_engine_hash():
    from gateway.research_lab.code_loop_engine import _focus_signature_hash

    brief = "  Improve   Exa  sourcing\nfor SaaS ICPs  "
    ticket = {"ticket_doc": {"brief_public_summary": brief}}
    assert tiv.focus_signature_hash_for_brief(brief) == _focus_signature_hash(ticket)


def test_focus_signature_normalizes_case_and_whitespace():
    a = tiv.focus_signature_hash_for_brief("Try DIFFERENT   prompts")
    b = tiv.focus_signature_hash_for_brief("try different prompts")
    c = tiv.focus_signature_hash_for_brief("try different prompts for exa")
    assert a == b  # same direction, different formatting
    assert a != c  # similar-but-different direction hashes differently


# ---------------------------------------------------------------------------
# path reference extraction + allowlist
# ---------------------------------------------------------------------------


def test_path_references_extracted_from_prose():
    brief = "Edit sourcing_model/discovery.py and gateway/foo/bar.json to add retries."
    refs = tiv.path_references_from_brief(brief)
    assert "sourcing_model/discovery.py" in refs
    assert "gateway/foo/bar.json" in refs


@pytest.mark.parametrize(
    "path,expected",
    [
        ("sourcing_model/discovery.py", ""),
        ("research_lab_adapter.py", ""),  # bare filename: never rejected
        ("discovery.py", ""),  # bare filename: planner resolves at run time
        ("../secrets/keys.py", "path_escapes_repository"),
        ("/etc/passwd.txt", "path_escapes_repository"),
        ("private_repo/model.py", "path_outside_editable_roots"),
        ("gateway/binary.exe", "path_suffix_not_editable"),
    ],
)
def test_invalid_path_reference_reasons(path, expected):
    reason = tiv.invalid_path_reference_reason(
        path,
        allowed_prefixes=ALLOWED_PREFIXES,
        allowed_exact=ALLOWED_EXACT,
        allowed_suffixes=ALLOWED_SUFFIXES,
    )
    assert reason == expected


def test_direction_with_uneditable_path_rejected():
    with mock.patch.object(tiv, "direction_refusal_count", side_effect=AssertionError):
        rejection = _validate("Please patch secret_vault/keys.py to rotate credentials")
    assert rejection is not None
    assert rejection["reason"] == "direction_references_uneditable_path"
    assert "secret_vault/keys.py" in rejection["detail"]


def test_direction_with_editable_path_passes_path_check(monkeypatch):
    async def zero_refusals(**kwargs):
        return 0

    async def fake_select_many(table, **kwargs):
        return [{"current_status_at": "2026-07-10T01:19:12"}]

    monkeypatch.setattr(tiv, "direction_refusal_count", zero_refusals)
    with mock.patch("gateway.research_lab.store.select_many", fake_select_many):
        rejection = _validate("Tighten retries in sourcing_model/discovery.py")
    assert rejection is None


# ---------------------------------------------------------------------------
# refusal dedup with exploration allowance
# ---------------------------------------------------------------------------


def _events(signature, n, created="2026-07-10T12:00:00"):
    return [
        {"created_at": created, "event_doc": {"focus_signature_hash": signature}}
        for _ in range(n)
    ]


def test_direction_rejected_after_repeated_refusals(monkeypatch):
    brief = "route exa search to another surface"
    signature = tiv.focus_signature_hash_for_brief(brief)

    async def fake_select_many(table, **kwargs):
        if table == "research_lab_private_model_version_current":
            return [{"current_status_at": "2026-07-10T01:19:12"}]
        return _events(signature, 3)

    with mock.patch("gateway.research_lab.store.select_many", fake_select_many):
        rejection = _validate(brief)
    assert rejection is not None
    assert rejection["reason"] == "direction_repeatedly_refused_against_current_source"
    assert rejection["refusal_count"] == 3


def test_direction_allowed_below_refusal_threshold(monkeypatch):
    brief = "route exa search to another surface"
    signature = tiv.focus_signature_hash_for_brief(brief)

    async def fake_select_many(table, **kwargs):
        if table == "research_lab_private_model_version_current":
            return [{"current_status_at": "2026-07-10T01:19:12"}]
        return _events(signature, 2)  # threshold default 3

    with mock.patch("gateway.research_lab.store.select_many", fake_select_many):
        rejection = _validate(brief)
    assert rejection is None


def test_refusals_against_previous_source_do_not_count(monkeypatch):
    # Refusals recorded before the active model switched are stale evidence.
    brief = "route exa search to another surface"
    signature = tiv.focus_signature_hash_for_brief(brief)

    async def fake_select_many(table, **kwargs):
        if table == "research_lab_private_model_version_current":
            return [{"current_status_at": "2026-07-10T01:19:12"}]
        return _events(signature, 5, created="2026-07-09T23:00:00")  # pre-switch

    with mock.patch("gateway.research_lab.store.select_many", fake_select_many):
        rejection = _validate(brief)
    assert rejection is None


def test_similar_direction_never_deduplicated(monkeypatch):
    # Exploration allowance: a reworded/experimental direction hashes
    # differently and must pass even when a sibling direction was refused.
    refused = tiv.focus_signature_hash_for_brief("route exa search to another surface")

    async def fake_select_many(table, **kwargs):
        if table == "research_lab_private_model_version_current":
            return [{"current_status_at": "2026-07-10T01:19:12"}]
        return _events(refused, 10)

    with mock.patch("gateway.research_lab.store.select_many", fake_select_many):
        rejection = _validate("experiment with different prompts for Exa company sourcing")
    assert rejection is None


def test_validation_fails_open_on_store_errors(monkeypatch):
    async def boom(table, **kwargs):
        raise RuntimeError("postgrest down")

    with mock.patch("gateway.research_lab.store.select_many", boom):
        rejection = _validate("route exa search to another surface")
    assert rejection is None  # infra failure must never block intake


def test_validation_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INTAKE_DIRECTION_VALIDATION", "false")
    rejection = _validate("Please patch secret_vault/keys.py to rotate credentials")
    assert rejection is None
