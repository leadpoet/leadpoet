"""The one-time legacy worker cutover requires a provably idle review window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.arena_first_cutover_idle import require_idle


def _service(now, row, *, round_status="stage2", cutoff_seconds=10800):
    return SimpleNamespace(
        active_rounds=lambda: [{"round_id": "r"}],
        _round=lambda _id: {"round_id": "r", "status": round_status},
        _submission_replacement_cutoff=lambda _round: now + timedelta(seconds=cutoff_seconds),
        _is_daily_baseline=lambda _row, _round: False,
        _store=SimpleNamespace(
            code_review_schema=lambda: {"schema_version": "leadpoet.lab_arena.code_review.v1",
                                        "atomic_restart_guard": True},
            list_submissions=lambda _id, *, status: [row] if row["status"] == status else []),
    )


@pytest.mark.parametrize("decision", ["passed", "rejected"])
@pytest.mark.parametrize("status", ["accepted", "frozen"])
def test_completed_review_retains_its_audit_claim_without_live_work(decision, status):
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    row = {"status": status, "code_review_status": decision,
           "code_review_claim": "retained-finish-token", "code_review_expires_at": None,
           "code_review_started_at": "2026-10-14T23:00:00+00:00",
           "code_review_attempts": 1, "code_review_doc": {"decision": decision}}
    require_idle(_service(now, row), now=now)
    assert row["code_review_claim"] == "retained-finish-token"


@pytest.mark.parametrize("review", [
    {"code_review_status": "pending", "code_review_expires_at": None},
    {"code_review_status": "reviewing", "code_review_expires_at": None},
    {"code_review_status": "error", "code_review_expires_at": None},
    {"code_review_status": None, "code_review_expires_at": None},
    {"code_review_status": "passed"},
    {"code_review_status": "rejected", "code_review_expires_at": "2026-10-15T01:00:00+00:00"},
    {"code_review_status": "passed", "code_review_expires_at": "2026-10-14T01:00:00+00:00"},
    {"code_review_status": "passed", "code_review_claim": None,
     "code_review_expires_at": "2026-10-15T01:00:00+00:00"},
])
def test_active_nonterminal_or_ambiguous_review_still_blocks_cutover(review):
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    row = {"status": "frozen", "code_review_claim": "retained-or-active-token", **review}
    with pytest.raises(ValueError, match="claimed|lease"):
        require_idle(_service(now, row), now=now)


@pytest.mark.parametrize("decision", ["pending", "running", "error", None])
def test_unclaimed_nonterminal_review_can_still_start_after_cutoff(decision):
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    row = {"status": "frozen", "code_review_claim": None,
           "code_review_expires_at": None, "code_review_status": decision}
    with pytest.raises(ValueError, match="can still start"):
        require_idle(_service(now, row), now=now)


@pytest.mark.parametrize("seconds", [7199, 7200, 7201])
def test_completed_audit_claim_does_not_bypass_cutoff_safety(seconds):
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    row = {"status": "accepted", "code_review_status": "passed",
           "code_review_claim": "retained-finish-token", "code_review_expires_at": None}
    service = _service(now, row, round_status="open", cutoff_seconds=seconds)
    if seconds <= 7200:
        with pytest.raises(ValueError, match="too near"):
            require_idle(service, now=now)
    else:
        require_idle(service, now=now)


def test_first_cutover_allows_only_distant_open_cutoff_without_claims():
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    round_row = {"round_id": "r", "status": "open"}
    claims = []
    service = SimpleNamespace(
        active_rounds=lambda: [{"round_id": "r"}],
        _round=lambda _id: round_row,
        _submission_replacement_cutoff=lambda _round: now + timedelta(hours=3),
        _is_daily_baseline=lambda _row, _round: False,
        _store=SimpleNamespace(list_submissions=lambda _id, *, status: claims if status == "accepted" else []),
    )
    require_idle(service, now=now)
    claims.append({"code_review_claim": "in-flight"})
    with pytest.raises(ValueError, match="claimed"):
        require_idle(service, now=now)
    claims.clear()
    round_row["status"] = "stage2"
    claims.append({"code_review_claim": None, "code_review_status": "error"})
    with pytest.raises(ValueError, match="can still start"):
        require_idle(service, now=now)
    claims[0]["code_review_status"] = "passed"
    require_idle(service, now=now)


def test_first_cutover_rejects_near_review_cutoff():
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    service = SimpleNamespace(
        active_rounds=lambda: [{"round_id": "r"}],
        _round=lambda _id: {"round_id": "r", "status": "open"},
        _submission_replacement_cutoff=lambda _round: now + timedelta(minutes=9),
    )
    with pytest.raises(ValueError, match="too near"):
        require_idle(service, now=now)


def _guard(monkeypatch, **changes):
    from scripts import lab_arena_restart_claim_guard as guard
    args = SimpleNamespace(candidate="a" * 40, invocation="idle-recovery", scope="gateway", generation=7)
    guard_id, owner = guard._identity(args.candidate, args.invocation)
    guard_commitment, owner_commitment = guard._commitments(guard_id, owner)
    state = {"schema_version": "leadpoet.lab_arena.restart_guard_state.v1",
             "guard_present": True, "guard_active": True, "paused": True, "operator_paused": False,
             "drain": {"schema_version": "leadpoet.lab_arena.restart_drain_state.v1",
                       "captured_count": 0, "accepted_receipt_count": 0,
                       "reported_terminal_receipt_count": 0, "still_leased_count": 0,
                       "lost_or_mutated_count": 0, "current_leased_count": 0,
                       "pending_retry_count": 0, "preserved": True},
             "guard_commitment": guard_commitment, "owner_commitment": owner_commitment,
             "candidate_commit": args.candidate, "restart_scope": "gateway",
             "restart_phase": "gateway_destructive", "guard_generation": 7, **changes}
    monkeypatch.setattr(guard, "_request", lambda *_: state)
    return args


@pytest.mark.parametrize("round_status", ["open", "stage2"])
@pytest.mark.parametrize("seconds", [-1, 0, 7200, 7201])
@pytest.mark.parametrize("scope", ["gateway", "all"])
def test_owned_guard_blocks_new_review_even_at_cutoff(monkeypatch, round_status, seconds, scope):
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    row = {"status": "accepted", "code_review_claim": None,
           "code_review_expires_at": None, "code_review_status": "pending"}
    service = _service(now, row, round_status=round_status, cutoff_seconds=seconds)
    args = _guard(monkeypatch, restart_scope=scope)
    args.scope = scope
    require_idle(service, now=now, restart_guard=args)


@pytest.mark.parametrize("changes", [
    {"schema_version": "unknown"}, {"guard_present": False}, {"guard_active": False},
    {"owner_commitment": "foreign"}, {"guard_commitment": "foreign"},
    {"candidate_commit": "b" * 40}, {"restart_scope": "validator"},
    {"restart_phase": "gateway_ready"}, {"guard_generation": 8},
    {"guard_generation": True},
])
def test_guarded_idle_fails_closed_without_exact_active_owner(monkeypatch, changes):
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    with pytest.raises((ValueError, RuntimeError)):
        require_idle(_service(now, {"status": "accepted"}, round_status="open", cutoff_seconds=0),
                     now=now, restart_guard=_guard(monkeypatch, **changes))


@pytest.mark.parametrize("review", [
    {"code_review_claim": "paid", "code_review_expires_at": None, "code_review_status": "reviewing"},
    {"code_review_claim": None, "code_review_expires_at": "2026-10-28T01:00:00Z"},
])
def test_owned_guard_does_not_abandon_paid_review(monkeypatch, review):
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="claimed|lease"):
        require_idle(_service(now, {"status": "accepted", **review}, round_status="open", cutoff_seconds=0),
                     now=now, restart_guard=_guard(monkeypatch))


def test_guard_state_read_failure_blocks_idle_recovery(monkeypatch):
    from scripts import lab_arena_restart_claim_guard as guard
    args = _guard(monkeypatch)
    def unavailable(*_):
        raise guard.GuardError("guard state unavailable")
    monkeypatch.setattr(guard, "_request", unavailable)
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    with pytest.raises(guard.GuardError, match="unavailable"):
        require_idle(_service(now, {"status": "accepted"}, round_status="open", cutoff_seconds=0),
                     now=now, restart_guard=args)


@pytest.mark.parametrize("schema", [{}, {"schema_version": "unknown", "atomic_restart_guard": True},
    {"schema_version": "leadpoet.lab_arena.code_review.v1", "atomic_restart_guard": False},
    {"schema_version": "leadpoet.lab_arena.code_review.v1", "atomic_restart_guard": 1}])
def test_guarded_idle_requires_installed_atomic_reservation_gate(monkeypatch, schema):
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    service = _service(now, {"status": "accepted"}, round_status="open", cutoff_seconds=0)
    service._store.code_review_schema = lambda: schema
    with pytest.raises(ValueError, match="atomic review admission"):
        require_idle(service, now=now, restart_guard=_guard(monkeypatch))
