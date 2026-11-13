"""Spend existing company attempts in ordered passes, with intrinsic checkpoints."""
from collections import Counter
from copy import deepcopy

import pytest

pytestmark = pytest.mark.usefixtures("scorer_process_environment")

from qualification.scoring import lead_scorer
from tests.lab_arena.legacy_company_judgments_test import _input, _score
from tests.test_arena_score_integrity import _positive_breakdown


def test_quick_companies_checkpoint_before_earlier_company_retries(monkeypatch):
    calls, checkpoints, cache_ids = [], [], {}

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        cache = kwargs["verified_company_fit_retry_cache"]
        assert cache is not None
        cache_ids.setdefault(name, id(cache))
        assert cache_ids[name] == id(cache)
        cache["attempts"] = cache.get("attempts", 0) + 1
        if name == "A":
            raise RuntimeError("provider unavailable")
        return _positive_breakdown(name, name.lower() + ".example")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    output = _score(_input(("A", "B", "C")), on_company_judgments=lambda rows: checkpoints.append((list(calls), deepcopy(rows))))
    assert output["failure"] == "judge_error"
    assert calls == ["A", "B", "C", "A", "A"]
    assert [seen for seen, _rows in checkpoints] == [["A", "B"], ["A", "B", "C"]]
    assert [row["company_index"] for row in output["company_judgments"]] == [1, 2]
    assert output["detail"].startswith("judge_exhausted attempts=3 ")


def test_late_earlier_alias_keeps_original_order_and_intrinsic_authorities(monkeypatch):
    calls = []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        if name == "A" and calls.count("A") == 1:
            raise RuntimeError("provider unavailable")
        return _positive_breakdown("Canonical", "canonical.example", "canonical")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    document = _input(("A", "B", "C"))
    # B is judged before A recovers. C is a submitted alias and must never
    # reach a provider, in either pass. On recovery A owns the first slot.
    for company in document["companies"][1:]:
        company["company_website"] = "https://canonical.example"
        company["company_linkedin"] = "https://www.linkedin.com/company/canonical"
    document["companies"][2]["company_name"] = "Canonical"
    output = _score(document)
    assert "failure" not in output
    assert calls == ["A", "B", "A"]
    assert [row["final_score"] for row in output["breakdowns"]] == [60, 0, 0]
    assert [row["duplicate_company"] for row in output["breakdowns"]] == [False, True, True]
    raw = output["company_judgments"]
    assert [row["company_index"] for row in raw] == [0, 1]
    assert all(row["raw_judgment"]["final_score"] == 60 for row in raw)
    assert all("duplicate_company" not in row["raw_judgment"] for row in raw)
    # An all-hit resume uses the same intrinsic authorities and ordered gates.
    saved = {row["cache_key"]: row["raw_judgment"] for row in raw}
    resumed = _score(document, saved)
    assert resumed["breakdowns"] == output["breakdowns"]
    assert calls == ["A", "B", "A"]


def test_interruption_after_quick_company_retains_its_real_checkpoint(monkeypatch):
    calls, saved = [], {}
    failed = True

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        if name == "A" and failed:
            raise RuntimeError("provider unavailable")
        return _positive_breakdown(name, name.lower() + ".example")

    def checkpoint(rows):
        saved.update({row["cache_key"]: deepcopy(row["raw_judgment"]) for row in rows})
        raise SystemExit("sandbox interruption")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    document = _input(("A", "B", "C"))
    with pytest.raises(SystemExit, match="sandbox interruption"):
        _score(document, on_company_judgments=checkpoint)
    assert calls == ["A", "B"]
    failed = False
    resumed = _score(document, saved)
    assert "failure" not in resumed
    assert calls == ["A", "B", "A", "C"]
    assert [row["company_index"] for row in resumed["company_judgments"]] == [0, 2]


@pytest.mark.parametrize("fault", ["provider", "malformed"])
def test_each_unfinished_key_gets_three_attempts_and_no_completed_replay(monkeypatch, fault):
    calls = []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        if name in {"A", "B"}:
            raise RuntimeError("provider unavailable") if fault == "provider" else ValueError("malformed")
        row = _positive_breakdown(name, name.lower() + ".example")
        row.update(final_score=0, intent_signals_detail=[])
        row["verifier_gate_receipts"][0]["decision"] = "mismatch"
        return row

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    output = _score(_input(("A", "B", "A", "C")))
    assert output["failure"] == "judge_error"
    assert calls == ["A", "B", "C", "A", "B", "A", "B"]
    assert Counter(calls) == {"A": 3, "B": 3, "C": 1}
    assert [row["company_index"] for row in output["company_judgments"]] == [3]
    assert output["company_judgments"][0]["raw_judgment"]["final_score"] == 0


@pytest.mark.parametrize("recovers", [False, True])
def test_source_local_failure_uses_final_attempt_semantics(monkeypatch, recovers):
    calls = []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        row = _positive_breakdown(name, name.lower() + ".example")
        if name == "A" and not (recovers and calls.count("A") == 3):
            row.update(final_score=0, intent_signals_detail=[], failure_reason="Company fit unavailable: source blocked")
            row["verifier_gate_receipts"][0].update(
                decision="unavailable", failure_reason_code="source_blocked",
            )
        return row

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    output = _score(_input())
    assert "failure" not in output
    assert calls == ["A", "B", "A", "A"]
    raw = output["company_judgments"]
    assert [row["company_index"] for row in raw] == [0, 1]
    assert raw[0]["raw_judgment"]["final_score"] == (60 if recovers else 0)
