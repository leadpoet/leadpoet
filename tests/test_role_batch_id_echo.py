"""Role-batch judge id echo robustness.

The batch judge embeds lead ids in prose and parses them back from LLM JSON,
so a numeric id can come back quoted ("0" for 0) or a string id can come back
as a number. An unmatched echo silently keeps the fail-closed False default,
which rejects a lead the judge actually accepted (role_mismatch) — the same
symptom as the lost batch-decision handoff bug. These tests pin the id
type-coercion on the result mapping.
"""

from __future__ import annotations

import pytest

from qualification.scoring import role_batch_check
from qualification.scoring.role_batch_check import _coerce_result_id, batch_check


@pytest.fixture
def judged(monkeypatch):
    """Run batch_check with a stubbed judge; returns (results_for(parsed))."""

    def _run(leads, parsed):
        async def fake_judge_chunk(http, key, target_roles, chunk):
            return parsed

        monkeypatch.setenv("OPENROUTER_KEY", "test-key")
        monkeypatch.setattr(role_batch_check, "_judge_chunk", fake_judge_chunk)
        import asyncio

        return asyncio.run(batch_check(leads, ["Head of Marketing"]))

    return _run


def test_quoted_numeric_echo_still_lands(judged):
    leads = [{"id": 0, "role": "Head of Communications and Marketing"}]
    results = judged(leads, [{"id": "0", "match": True}])
    assert results == {0: True}


def test_numeric_echo_of_string_id_still_lands(judged):
    leads = [{"id": "3", "role": "VP Marketing"}]
    results = judged(leads, [{"id": 3, "match": True}])
    assert results == {"3": True}


def test_exact_id_echo_unchanged(judged):
    leads = [
        {"id": 0, "role": "Head of Communications and Marketing"},
        {"id": 1, "role": "VP Sales"},
    ]
    results = judged(leads, [{"id": 0, "match": True}, {"id": 1, "match": False}])
    assert results == {0: True, 1: False}


def test_unknown_echo_stays_fail_closed(judged):
    leads = [{"id": 0, "role": "Head of Marketing Ops"}]
    results = judged(leads, [{"id": "not-a-lead", "match": True}])
    assert results == {0: False}


def test_coerce_result_id_edges():
    results = {0: False, "a@b.com": False, "7": False}
    assert _coerce_result_id("0", results) == 0
    assert _coerce_result_id(7, results) == "7"
    assert _coerce_result_id(True, results) is None
    assert _coerce_result_id(None, results) is None
    assert _coerce_result_id("missing", results) is None
