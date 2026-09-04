"""One malformed record must not take down the whole scoring feed.

On 2026-09-04 `GET /fulfillment/scoring` returned 500 on every call from
09:57 UTC onwards. There was no deploy behind it and the database was
answering normally at its usual speed — the payload assembly was throwing on
a stored row whose shape it did not expect, and because assembly ran as one
loop over every request, one bad record cost every validator the entire feed.
These tests pin the two properties that stop that: a request that cannot be
assembled is skipped, and the call only fails when the feed can serve nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gateway.fulfillment import api as fulfillment_api


class _Query:
    """Minimal postgrest-style query stub: every filter is a no-op."""

    def __init__(self, rows):
        self._rows = rows
        self._range = (0, 999)

    def select(self, _fields):
        return self

    def eq(self, _field, _value):
        return self

    def in_(self, _field, _values):
        return self

    def or_(self, _filters):
        return self

    def order(self, _field):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        page = self._range[0] // 1000
        return SimpleNamespace(data=self._rows if page == 0 else [])


class _Supabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _Query(self._tables.get(name, []))


def _request(rid, **overrides):
    row = {
        "request_id": rid,
        "status": "scoring",
        "icp_details": {"role": "engineer"},
    }
    row.update(overrides)
    return row


def _submission(rid, sid, lead_data):
    return {
        "request_id": rid,
        "submission_id": sid,
        "miner_hotkey": f"hk-{sid}",
        "revealed": True,
        "lead_data": lead_data,
    }


def _collect(monkeypatch, requests_rows, submission_rows):
    supabase = _Supabase({
        "fulfillment_requests": requests_rows,
        "fulfillment_scores": [],
        "fulfillment_submissions": submission_rows,
    })
    monkeypatch.setattr(fulfillment_api, "_get_supabase", lambda: supabase)
    return fulfillment_api._collect_scoring_requests_sync("validator-hotkey")


def test_one_unassemblable_request_does_not_cost_the_others(monkeypatch):
    """The 2026-09-04 outage shape: a poisoned row alongside healthy ones."""
    out = _collect(
        monkeypatch,
        [_request("good-request-id"), _request("bad-request-id")],
        [
            _submission("good-request-id", "s-good", [{"lead_id": "l1", "data": {"a": 1}}]),
            # `lead_data` is JSONB. A stored scalar is not iterable, so
            # assembling this one raises — the shape of failure that used to
            # escape as a 500 for the whole call.
            _submission("bad-request-id", "s-bad", 7),
        ],
    )

    assert [r["request_id"] for r in out["requests"]] == ["good-request-id"]
    assert out["skipped"] == 1


def test_total_assembly_failure_is_reported_not_served_as_empty(monkeypatch):
    """An empty feed means "nothing to score" — it must not mean "all broken"."""
    with pytest.raises(HTTPException) as excinfo:
        _collect(
            monkeypatch,
            [_request("bad-request-id")],
            [_submission("bad-request-id", "s-bad", 7)],
        )

    assert excinfo.value.status_code == 500
    assert "bad-requ" in excinfo.value.detail


def test_nothing_to_score_is_still_an_empty_success(monkeypatch):
    out = _collect(monkeypatch, [_request("quiet-request-id")], [])
    assert out == {"requests": [], "skipped": 0}


def test_stray_non_map_lead_entry_keeps_leads_and_ids_aligned():
    """`leads` and `lead_ids` are zipped onto scores — they must stay aligned."""
    entry = fulfillment_api._assemble_scoring_request(
        _request("rid"),
        [_submission("rid", "s1", [
            {"lead_id": "l1", "data": {"a": 1}},
            "a stray string where a map belongs",
            {"lead_id": "l2", "data": {"a": 2}},
        ])],
        set(),
    )

    submission = entry["submissions"][0]
    assert submission["lead_ids"] == ["l1", "l2"]
    assert len(submission["leads"]) == len(submission["lead_ids"])


def test_already_scored_submissions_are_still_omitted():
    assert fulfillment_api._assemble_scoring_request(
        _request("rid"),
        [_submission("rid", "s1", [{"lead_id": "l1", "data": {}}])],
        {"s1"},
    ) is None


def test_required_attributes_are_still_merged_into_the_icp():
    entry = fulfillment_api._assemble_scoring_request(
        _request("rid", required_attributes=["email"]),
        [_submission("rid", "s1", [{"lead_id": "l1", "data": {}}])],
        set(),
    )
    assert entry["icp"] == {"role": "engineer", "required_attributes": ["email"]}
