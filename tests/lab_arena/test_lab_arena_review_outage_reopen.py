"""The review-hold rerun trigger also reopens a round already published (702).

Regression for arena-2026-11-19: it published no_king with only the baseline
because every review failed with code_review_provider_unavailable, and the
owner trigger refused published rounds outright.
"""
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStoreError
from scripts import lab_arena_admin

NOW = datetime(2026, 11, 19, 8, 0, 0, tzinfo=timezone.utc)
HASH = "sha256:" + "a" * 64


def reopen_service(*, next_cutoff="2026-11-20T00:00:00Z", refuse=None):
    calls = []
    published = {"round_id": "arena-2026-11-19", "status": "published",
                 "configuration_doc": {"schedule": {"submission_cutoff": "2026-11-19T00:00:00Z"}}}
    later = [] if next_cutoff is None else [{"round_id": "arena-2026-11-20", "status": "open",
             "configuration_doc": {"schedule": {"submission_cutoff": next_cutoff}}}]

    def reopen(round_id, **arguments):
        calls.append(arguments)
        if refuse:
            raise ArenaStoreError("rpc lab_arena_reopen_review_outage_v1 failed: %s {PL/pgSQL}" % refuse)
        if arguments["dry_run"]:
            return {"status": "dry_run", "round_sha256": HASH, "submissions": [{"submission_id": "m0"}]}
        return {"status": "reopened", "idempotent": False}
    service = object.__new__(ArenaService)
    service._clock = lambda: NOW
    service._round = lambda _: published
    service._invalidate_hot_round = lambda: None
    service.active_rounds = lambda: later
    service._config = SimpleNamespace(defaults=SimpleNamespace(stage_minutes={"stage_2": 180}))
    service._store = SimpleNamespace(reopen_review_outage=reopen)
    return service, calls


def test_published_rerun_dry_run_bounds_deadlines_by_the_next_round():
    service, calls = reopen_service()
    result = service.decide_review_hold("arena-2026-11-19", "rerun", dry_run=True)
    assert result["status"] == "dry_run" and len(calls) == 1
    call = calls[0]
    assert call["dry_run"] is True and call["round_sha256"] is None and call["submissions"] is None
    assert call["review_deadline"] == "2026-11-19T09:00:00Z"
    assert call["stage_2_close"] == "2026-11-19T12:00:00Z"
    # Republish before next cutoff - 2h; SQL cancels at next cutoff - 1h.
    assert call["final_scoring_close"] == "2026-11-19T21:59:59Z"
    assert call["publication_deadline"] == "2026-11-19T22:00:00Z"
    assert call["request_id"] == "review-outage-reopen:arena-2026-11-19"


def test_published_rerun_applies_only_the_confirmed_round_with_approval():
    service, calls = reopen_service()
    with pytest.raises(ServiceError, match="review_reopen_approval_required"):
        service.decide_review_hold("arena-2026-11-19", "rerun")
    with pytest.raises(ServiceError, match="review_reopen_round_changed"):
        service.decide_review_hold("arena-2026-11-19", "rerun", approval_ref="ok",
                                   expect_round_sha256="sha256:" + "b" * 64)
    assert all(call["dry_run"] for call in calls)
    result = service.decide_review_hold("arena-2026-11-19", "rerun", approval_ref="ok", expect_round_sha256=HASH)
    assert result["status"] == "reopened"
    applied = calls[-1]
    assert applied["dry_run"] is False and applied["round_sha256"] == HASH
    assert applied["submissions"] == [{"submission_id": "m0"}] and applied["approval_ref"] == "ok"


@pytest.mark.parametrize("next_cutoff,code", [
    (None, "review_reopen_next_round_unknown"),
    ("2026-11-19T11:00:00Z", "review_reopen_window_too_short"),
])
def test_published_rerun_refuses_up_front_when_the_next_round_is_unbounded_or_too_close(next_cutoff, code):
    service, calls = reopen_service(next_cutoff=next_cutoff)
    with pytest.raises(ServiceError, match=code):
        service.decide_review_hold("arena-2026-11-19", "rerun", dry_run=True)
    assert calls == []


def test_published_rerun_reports_the_sql_refusal_reason():
    service, _ = reopen_service(refuse="review_outage_reopen: participants or runs are not baseline-only")
    with pytest.raises(ServiceError, match=r"review_reopen_refused \(review_outage_reopen: participants or runs"):
        service.decide_review_hold("arena-2026-11-19", "rerun", dry_run=True)


def driver_service(status, outcome):
    calls = []
    row = {"round_id": "arena-2026-11-19", "status": status, "configuration_doc": {"schedule": {
        "submission_cutoff": "2026-11-19T00:00:00Z", "publication_deadline": "2026-11-19T22:00:00Z"}}}

    def advance(round_id):
        calls.append(round_id)
        if isinstance(outcome, Exception):
            raise outcome
        return dict(outcome)
    service = object.__new__(ArenaService)
    service._lock = threading.RLock()
    service._clock = lambda: NOW
    service._round = lambda _: row
    service._invalidate_hot_round = lambda: None
    service._store = SimpleNamespace(advance_review_outage=advance)
    service.open_stage = lambda round_id, stage: {"status": "ok", "opened": stage}
    return service, calls


@pytest.mark.parametrize("outcome", [{"status": "waiting", "reason": "review_pending"},
                                     {"status": "admitted", "admitted": ["m0"]},
                                     {"status": "cancelled", "reason": "review_outage_reopen_expired"}])
def test_driver_holds_admits_or_cancels_a_reopened_round(outcome):
    service, calls = driver_service("stage1_scored", outcome)
    result = service._advance_round_locked("arena-2026-11-19")
    assert calls == ["arena-2026-11-19"] and result["status"] == outcome["status"]


@pytest.mark.parametrize("outcome", [{"status": "not_reopened"},
                                     ArenaStoreError("missing", operation="rpc", resource="lab_arena_advance_review_outage_v1", code="42883")])
def test_driver_continues_normally_without_a_reopen_or_before_702(outcome):
    service, calls = driver_service("stage1_scored", outcome)
    assert service._advance_round_locked("arena-2026-11-19") == {"status": "ok", "opened": 2}
    assert calls == ["arena-2026-11-19"]


def test_admin_rerun_on_a_published_round_reopens_with_confirmation(capsys):
    decisions = []
    row = {"round_id": "arena-2026-11-19", "status": "published"}

    def decide(round_id, action, **options):
        decisions.append((action, options))
        return {"status": "dry_run" if options["dry_run"] else "reopened"}
    service = SimpleNamespace(store=SimpleNamespace(get_round=lambda _: row), decide_review_hold=decide)
    parse = lab_arena_admin.build_parser().parse_args
    assert lab_arena_admin.run(parse(["review-hold", "--round", "arena-2026-11-19", "--action", "rerun",
                                      "--dry-run"]), service) == 0
    assert lab_arena_admin.run(parse(["review-hold", "--round", "arena-2026-11-19", "--action", "rerun",
                                      "--approval-ref", "gavin", "--expect-round-sha256", HASH]), service) == 0
    assert decisions == [
        ("rerun", {"dry_run": True, "expect_round_sha256": None, "approval_ref": None}),
        ("rerun", {"dry_run": False, "expect_round_sha256": HASH, "approval_ref": "gavin"})]
    assert lab_arena_admin.run(parse(["review-hold", "--round", "arena-2026-11-19", "--action", "release"]),
                               service) == 2
