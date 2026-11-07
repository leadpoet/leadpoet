"""Admission explanations use saved, typed evidence and never reviewer prose."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from fastapi.testclient import TestClient

from lab_arena import code_review, public_dashboard
from lab_arena.api import create_app
from lab_arena.service import ArenaService
from tests.lab_arena.test_lab_arena_public_dashboard import (
    DashboardService, MINER_HOTKEY, _now, _published_round,
)


def _row(status="error", **document):
    return {"submission_id": "sub-miner", "round_id": "arena-2026-09-09",
            "miner_hotkey": MINER_HOTKEY, "status": "rejected", "is_king": False,
            "accepted_at": "2026-09-08T17:50:08Z", "rejection_rule": "code_review_incomplete",
            "source_archive_sha256": "sha256:" + "a" * 64,
            "code_review_status": status, "code_review_doc": document}


def _workflow(row):
    return {"classification": "workflow", "policy_id": code_review.RESEARCH_POLICY_ID,
            "policy_hash": code_review.RESEARCH_POLICY_HASH,
            "reviewer_version": code_review.RESEARCH_REVIEWER_VERSION,
            "source_archive_sha256": row["source_archive_sha256"],
            "source_manifest_sha256": "sha256:" + "b" * 64,
            "evidence_claims": ["fixed_control"], "evidence_count": 1}


@pytest.mark.parametrize("status,document,reason", [
    ("rejected", {"categories": ["fabricated_evidence"]}, "source_policy_violation"),
    ("passed", {}, "source_review_unverified"),
    ("error", {"error_code": "review_response_invalid", "error_reason": "research_evidence_mismatch"}, "review_evidence_invalid"),
    ("error", {"error_code": "review_response_invalid", "error_reason": "coverage"}, "review_evidence_invalid"),
    ("error", {"error_code": "review_response_invalid", "error_reason": "research_binding"}, "review_evidence_invalid"),
    *[("error", {"error_code": "review_response_invalid", "error_reason": reason}, "review_response_incomplete")
      for reason in ("truncated", "truncated_at_limit", "not_finished")],
    *[("error", {"error_code": "review_response_invalid", "error_reason": reason}, "review_error")
      for reason in ("refusal_or_tools", "content_json", "envelope", "unknown private prose")],
    ("error", {"error_code": "code_review_provider_refused"}, "review_error"),
    ("error", {"error_code": "code_review_provider_unavailable"}, "review_error"),
    ("rejected", {}, "review_incomplete"),
    ("pending", {}, "review_incomplete"),
])
def test_admission_reason_preserves_distinct_saved_outcomes(status, document, reason):
    row = _row(status, **document)
    before = deepcopy(row)
    projected = public_dashboard.code_review_summary(row)
    assert projected["admission_reason"] == reason
    assert row == before
    assert json.loads(json.dumps(projected)) == projected


@pytest.mark.parametrize("categories", ["fabricated_evidence", None, {}, [None, {}, ["malicious_behavior"], "private-secret"]])
def test_invalid_categories_cannot_create_or_publish_a_violation(categories):
    projected = public_dashboard.code_review_summary(_row("rejected", categories=categories))
    assert projected["categories"] == []
    assert projected["admission_reason"] == "review_incomplete"
    assert "private-secret" not in json.dumps(projected)


@pytest.mark.parametrize("status", ["error", "passed", "pending", "reviewing"])
def test_stale_categories_are_not_source_verdicts(status):
    view = public_dashboard.code_review_summary(_row(status, categories=["malicious_behavior"]))
    assert view["categories"] == []
    assert view.get("admission_reason") != "source_policy_violation"


@pytest.mark.parametrize("code", ["private-secret", {}, ["code_review_provider_refused"]])
def test_unknown_or_malformed_error_codes_are_not_public(code):
    view = public_dashboard.code_review_summary(_row("error", error_code=code))
    assert view["admission_reason"] == "review_incomplete"
    assert view["error_code"] is None
    assert "private-secret" not in json.dumps(view)


def test_known_categories_are_sorted_and_filtered_without_changing_saved_row():
    row = _row("rejected", categories=["malicious_behavior", {}, "private-secret", "fabricated_evidence", "malicious_behavior"])
    before = deepcopy(row)
    view = public_dashboard.code_review_summary(row)
    assert view["categories"] == ["fabricated_evidence", "malicious_behavior"]
    assert view["admission_reason"] == "source_policy_violation"
    assert "private-secret" not in json.dumps(view)
    assert row == before


def test_workflow_requires_typed_frozen_policy_and_matching_source():
    row = _row("rejected", categories=[])
    row["code_review_doc"]["research_control"] = _workflow(row)
    round_row = {"configuration_doc": {"reasoning_capture_policy": "full_generated_thinking_v1"}}
    assert public_dashboard.code_review_summary(row, round_row=round_row)["admission_reason"] == "model_directed_research_required"
    for field, value in [("policy_hash", "private-secret"), ("source_archive_sha256", "sha256:" + "c" * 64),
                         ("source_manifest_sha256", None), ("evidence_count", True),
                         ("evidence_claims", ["uncertainty"])]:
        changed = deepcopy(row)
        changed["code_review_doc"]["research_control"][field] = value
        view = public_dashboard.code_review_summary(changed, round_row=round_row)
        assert view["admission_reason"] == "review_incomplete"
        assert "private-secret" not in json.dumps(view)
    assert public_dashboard.code_review_summary(row)["admission_reason"] == "review_incomplete"
    del row["source_archive_sha256"]
    assert public_dashboard.code_review_summary(row, round_row=round_row)["admission_reason"] == "review_incomplete"


@pytest.mark.parametrize("status", ["passed", "pending", "reviewing"])
def test_nonterminal_reviews_do_not_receive_an_exclusion_reason(status):
    row = _row(status)
    row.update(status="accepted", rejection_rule=None)
    assert "admission_reason" not in public_dashboard.code_review_summary(row)


def test_round_and_submission_http_views_share_safe_reason():
    row = _row("error", error_code="review_response_invalid", error_reason="research_evidence_mismatch",
               findings=[{"evidence": "private-secret"}], provider_response="private-secret")
    round_row = _published_round()
    selected = []
    def records(*_args, **kwargs):
        selected.append(kwargs["columns"])
        return [row]
    service = DashboardService(_round=lambda _: round_row, now=_now,
        _store=SimpleNamespace(get_submission=lambda _: row, list_submissions=records,
                             list_runs=lambda *_a, **_k: [], list_runtime_starts=lambda *_a, **_k: []),
        _public_code_review=ArenaService._public_code_review)
    service.submission_status = lambda sid: ArenaService.submission_status(service, sid)
    service.public_submissions = lambda rid: public_dashboard.submissions_snapshot(service, rid)
    with TestClient(create_app(service)) as client:
        item = client.get("/arena/v1/submissions/sub-miner")
        collection = client.get("/arena/v1/rounds/arena-2026-09-09/submissions")
    assert item.status_code == collection.status_code == 200
    assert item.json()["code_review"]["admission_reason"] == "review_evidence_invalid"
    assert collection.json()["submissions"][0]["code_review"] == item.json()["code_review"]
    assert "source_archive_sha256" in selected[0]
    assert "private-secret" not in item.text + collection.text
