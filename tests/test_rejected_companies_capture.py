"""Rigorous local tests for the sourced-but-rejected-company capture
(_persist_rejected_companies) — false-rejection (FN) monitoring.

No DB / network: store.insert_row is monkeypatched to a collector. Async tests
run under pytest.ini `asyncio_mode = auto`.
"""
from __future__ import annotations

import pytest

from gateway.research_lab import scoring_worker as sw


class _Collector:
    """Stand-in for store.insert_row: records rows, or raises to test best-effort."""

    def __init__(self, fail: bool = False):
        self.rows: list[dict] = []
        self.fail = fail

    async def __call__(self, table, row):
        assert table == "research_lab_rejected_companies"
        if self.fail:
            raise RuntimeError("db unavailable")
        self.rows.append(row)
        return {"ok": True}


@pytest.fixture
def collector(monkeypatch):
    c = _Collector()
    monkeypatch.setattr(sw, "insert_row", c)
    monkeypatch.delenv("RESEARCH_LAB_REJECTED_COMPANIES_CAPTURE", raising=False)  # default on
    return c


def _outputs():
    return [
        {
            "company_name": "Alpha", "company_website": "alpha.com", "company_linkedin": "alpha",
            "industry": "Software", "sub_industry": "CRM", "employee_count": "51-200",
            "country": "United States",
        },
        {"company_name": "Beta", "company_website": "beta.com", "company_linkedin": "beta"},
        {"company_name": "Gamma"},
    ]


def _breakdowns():
    return [
        {"final_score": 43.2},  # accepted -> must be skipped
        {"final_score": 0.0, "failure_reason": "company_stage_mismatch", "stage_failed": "pre_checks",
         "fit_passed": False, "attribute_passed": True, "icp_fit": 0.6},
        {"final_score": 0.0, "failure_reason": "intent_fabricated", "intent_passed": False,
         "intent_signal_final": 0.1},
    ]


# ---------- helper unit tests ----------

def test_optional_bool():
    assert sw._optional_bool(True) is True
    assert sw._optional_bool(False) is False
    assert sw._optional_bool(None) is None
    assert sw._optional_bool("true") is True
    assert sw._optional_bool("FALSE") is False
    assert sw._optional_bool("1") is True
    assert sw._optional_bool("0") is False
    assert sw._optional_bool("maybe") is None


def test_optional_score():
    assert sw._optional_score(1.5) == 1.5
    assert sw._optional_score("2") == 2.0
    assert sw._optional_score(None) is None
    assert sw._optional_score("abc") is None


# ---------- core capture behavior ----------

async def test_only_rejections_are_stored(collector):
    n = await sw._persist_rejected_companies(
        context_ref="baseline", icp_ref="icp-1", icp_hash="sha256:abc",
        is_reference_model=True, outputs=_outputs(), breakdowns=_breakdowns())
    assert n == 2
    assert [r["company_name"] for r in collector.rows] == ["Beta", "Gamma"]  # Alpha accepted, skipped


async def test_field_mapping(collector):
    await sw._persist_rejected_companies(
        context_ref="baseline", icp_ref="icp-1", icp_hash="sha256:abc",
        is_reference_model=True, outputs=_outputs(), breakdowns=_breakdowns())
    beta = collector.rows[0]
    assert beta["failure_reason"] == "company_stage_mismatch"
    assert beta["failure_stage"] == "pre_checks"
    assert beta["fit_passed"] is False
    assert beta["attribute_passed"] is True
    assert beta["intent_passed"] is None          # not present in breakdown -> None
    assert beta["icp_fit"] == 0.6
    assert beta["final_score"] == 0.0
    assert beta["is_reference_model"] is True
    assert beta["candidate_id"] is None
    assert beta["context_ref"] == "baseline"
    assert beta["icp_ref"] == "icp-1"
    assert beta["icp_hash"] == "sha256:abc"
    assert beta["company_linkedin"] == "beta"
    assert isinstance(beta["dedup_key"], str) and beta["dedup_key"]
    gamma = collector.rows[1]
    assert gamma["failure_reason"] == "intent_fabricated"
    assert gamma["intent_signal"] == 0.1
    assert gamma["intent_passed"] is False


async def test_failure_reason_overrides_positive_score(collector):
    # positive score BUT a failure_reason => still a rejection
    n = await sw._persist_rejected_companies(
        context_ref="r", icp_ref="i", icp_hash="h", is_reference_model=True,
        outputs=[{"company_name": "X"}],
        breakdowns=[{"final_score": 10.0, "failure_reason": "some_flag"}])
    assert n == 1


# ---------- dedup key ----------

async def test_dedup_key_stable_across_runs(collector):
    args = dict(icp_ref="icp-1", icp_hash="h1", is_reference_model=True,
                outputs=[{"company_linkedin": "beta"}],
                breakdowns=[{"final_score": 0.0, "failure_reason": "r1"}])
    await sw._persist_rejected_companies(context_ref="run-day1", **args)
    await sw._persist_rejected_companies(context_ref="run-day2", **args)  # different run, same rejection
    assert collector.rows[0]["dedup_key"] == collector.rows[1]["dedup_key"]


async def test_dedup_key_differs_by_reason(collector):
    base = dict(context_ref="r", icp_ref="i", icp_hash="h1", is_reference_model=True,
                outputs=[{"company_linkedin": "beta"}])
    await sw._persist_rejected_companies(breakdowns=[{"final_score": 0.0, "failure_reason": "r1"}], **base)
    await sw._persist_rejected_companies(breakdowns=[{"final_score": 0.0, "failure_reason": "r2"}], **base)
    assert collector.rows[0]["dedup_key"] != collector.rows[1]["dedup_key"]


async def test_dedup_key_same_baseline_and_candidate(collector):
    # model is NOT part of the key: same company + error + icp collapses
    # baseline and candidate into one dedup_key (one row).
    out = [{"company_name": "Beta"}]
    bd = [{"final_score": 0.0, "failure_reason": "r1"}]
    await sw._persist_rejected_companies(context_ref="r", icp_ref="i", icp_hash="h1",
                                         is_reference_model=True, outputs=out, breakdowns=bd)
    await sw._persist_rejected_companies(context_ref="cand", icp_ref="i", icp_hash="h1",
                                         is_reference_model=False, outputs=out, breakdowns=bd,
                                         candidate_id="cand-1")
    assert collector.rows[0]["dedup_key"] == collector.rows[1]["dedup_key"]


async def test_identity_drift_does_not_split(collector):
    # same company reported with case / punctuation / linkedin variation across
    # runs must NOT split into two rows (name-normalized key).
    await sw._persist_rejected_companies(context_ref="r", icp_ref="i", icp_hash="h",
                                         is_reference_model=True, outputs=[{"company_name": "Open AI"}],
                                         breakdowns=[{"final_score": 0.0, "failure_reason": "r"}])
    await sw._persist_rejected_companies(context_ref="r", icp_ref="i", icp_hash="h",
                                         is_reference_model=True,
                                         outputs=[{"company_name": "open ai!", "company_linkedin": "openai-slug"}],
                                         breakdowns=[{"final_score": 0.0, "failure_reason": "r"}])
    assert collector.rows[0]["dedup_key"] == collector.rows[1]["dedup_key"]


# ---------- candidate path fields ----------

async def test_candidate_fields_populated(collector):
    await sw._persist_rejected_companies(
        context_ref="cand-1", icp_ref="i", icp_hash="h1", is_reference_model=False,
        outputs=[{"company_name": "X"}], breakdowns=[{"final_score": 0.0, "failure_reason": "r"}],
        candidate_id="cand-1", model_manifest_hash="mh-1")
    row = collector.rows[0]
    assert row["candidate_id"] == "cand-1"
    assert row["model_manifest_hash"] == "mh-1"
    assert row["is_reference_model"] is False


# ---------- robustness / gating ----------

async def test_best_effort_never_raises_on_db_failure(monkeypatch):
    c = _Collector(fail=True)
    monkeypatch.setattr(sw, "insert_row", c)
    monkeypatch.delenv("RESEARCH_LAB_REJECTED_COMPANIES_CAPTURE", raising=False)
    n = await sw._persist_rejected_companies(
        context_ref="r", icp_ref="i", icp_hash="h", is_reference_model=True,
        outputs=[{"company_name": "X"}], breakdowns=[{"final_score": 0.0, "failure_reason": "r"}])
    assert n == 0  # swallowed; no exception propagates


async def test_disabled_via_env(collector, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_REJECTED_COMPANIES_CAPTURE", "false")
    n = await sw._persist_rejected_companies(
        context_ref="r", icp_ref="i", icp_hash="h", is_reference_model=True,
        outputs=_outputs(), breakdowns=_breakdowns())
    assert n == 0
    assert collector.rows == []


async def test_empty_inputs(collector):
    assert await sw._persist_rejected_companies(
        context_ref="r", icp_ref="i", icp_hash="h", is_reference_model=True,
        outputs=[], breakdowns=[]) == 0
    assert await sw._persist_rejected_companies(
        context_ref="r", icp_ref="i", icp_hash="h", is_reference_model=True,
        outputs=None, breakdowns=None) == 0


async def test_non_mapping_breakdown_skipped(collector):
    n = await sw._persist_rejected_companies(
        context_ref="r", icp_ref="i", icp_hash="h", is_reference_model=True,
        outputs=[{"company_name": "X"}, {"company_name": "Y"}],
        breakdowns=["not-a-dict", {"final_score": 0.0, "failure_reason": "r"}])
    assert n == 1  # index-0 pair skipped (non-mapping), index-1 stored
    assert collector.rows[0]["company_name"] == "Y"
