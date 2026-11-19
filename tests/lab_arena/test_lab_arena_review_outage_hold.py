"""A reviewer-provider outage holds the round; it never fails the miners.

Regression for arena-2026-11-19: a withdrawn single-provider route made every
review fail with code_review_provider_unavailable, the deadline turned that
into code_review_incomplete, and the round published with only the baseline.
"""
import argparse
import urllib.error
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import leadpoet_observability
from lab_arena import code_review, code_review_policy, driver, review_route_preflight as preflight
from lab_arena.service import ArenaService, ServiceError
from scripts import lab_arena_admin

DEADLINE = datetime(2026, 11, 19, 0, 30, tzinfo=timezone.utc)
BASELINE = {"submission_id": "baseline-2026-11-19", "miner_hotkey": "host", "is_king": True,
            "source_ref": "baseline", "source_size_bytes": 10, "status": "accepted"}
FIELDS = ("submission_id", "miner_hotkey", "source_ref", "source_size_bytes", "is_king")


def miner(index, status="error", code="code_review_provider_unavailable", attempts=6):
    row = {"submission_id": "miner-%d" % index, "miner_hotkey": "miner-%d" % index, "is_king": False,
           "code_review_status": status, "code_review_attempts": attempts,
           "source_ref": "m%d" % index, "source_size_bytes": 10, "status": "accepted"}
    if status == "error":
        row["code_review_doc"] = {"error_code": code, "retryable": False}
    return row


def held_service(miners, *, now, hold):
    updates, holds = [], []
    round_row = {"round_id": "arena-2026-11-19", "champion_funding_frozen": True,
                 "configuration_doc": {"baseline_hotkey": "host",
                                       "schedule": {"benchmark_deadline": DEADLINE.isoformat()}}}
    service = object.__new__(ArenaService)
    service._clock = lambda: now[0]
    service._round = lambda _: round_row
    service._initial_baseline = lambda _: BASELINE
    service._store = SimpleNamespace(
        freeze_champion_funding=lambda _: {"status": "existing"},
        list_submissions=lambda _, status: [BASELINE, *miners] if status == "accepted" else [],
        update_submission=lambda *args: updates.append(args) or {"status": "ok"},
        hold_round_review=lambda round_id: holds.append(round_id) or dict(hold),
    )
    return service, updates, holds


@pytest.fixture
def alerts(monkeypatch):
    captured = []
    monkeypatch.setattr(leadpoet_observability, "capture_failure",
                        lambda code, **fields: captured.append((code, fields)) or True)
    return captured


def test_full_outage_holds_the_round_and_rejects_no_miner(alerts):
    miners = [miner(index) for index in range(3)] + [miner(3, status="pending", attempts=0)]
    hold = {"status": "existing", "review_deadline": (DEADLINE + timedelta(minutes=60)).isoformat(),
            "released_at": None}
    now = [DEADLINE - timedelta(minutes=1)]
    service, updates, holds = held_service(miners, now=now, hold=hold)
    with pytest.raises(ServiceError, match="code_review_pending"):
        service.freeze_participants("arena-2026-11-19")
    assert holds == ["arena-2026-11-19"]  # Armed while the outage is visible.
    now[0] = DEADLINE + timedelta(seconds=5)
    with pytest.raises(ServiceError, match="code_review_infrastructure_hold") as raised:
        service.freeze_participants("arena-2026-11-19")
    assert raised.value.status == 409
    assert updates == []  # Every miner stays accepted and reviewable.
    assert [code for code, _ in alerts] == ["arena.review_infrastructure_hold"]
    fields = alerts[0][1]
    assert fields["backlog_count"] == 4 and fields["retryable"] is True
    assert fields["error_code"] == "mixed" and fields["decision_code"] == "retrying"
    # A closed review window asks the operator to decide.
    now[0] = DEADLINE + timedelta(minutes=61)
    with pytest.raises(ServiceError, match="code_review_infrastructure_hold"):
        service.freeze_participants("arena-2026-11-19")
    assert alerts[-1][1]["decision_code"] == "operator_required"
    assert alerts[-1][1]["retryable"] is False
    assert updates == []


def test_recovered_provider_lets_the_held_round_freeze_every_miner(alerts):
    miners = [miner(index) for index in range(2)]
    hold = {"status": "existing", "review_deadline": (DEADLINE + timedelta(minutes=60)).isoformat(),
            "released_at": None}
    now = [DEADLINE + timedelta(minutes=20)]
    service, updates, _ = held_service(miners, now=now, hold=hold)
    with pytest.raises(ServiceError, match="code_review_infrastructure_hold"):
        service.freeze_participants("arena-2026-11-19")
    for row in miners:  # The SQL-extended window let retries pass.
        row.update(code_review_status="passed", code_review_doc={"passed": True})
    alerts.clear()
    participants = service.freeze_participants("arena-2026-11-19")
    assert participants == [{key: row[key] for key in FIELDS} for row in (BASELINE, *miners)]
    assert sorted(call[1] for call in updates if call[3] == "frozen") == ["baseline-2026-11-19", "miner-0", "miner-1"]
    assert not any(call[3] == "rejected" for call in updates)
    assert alerts == []


def test_released_hold_keeps_the_historical_rejection(alerts):
    miners = [miner(0)]
    hold = {"status": "existing", "review_deadline": DEADLINE.isoformat(), "released_at": "2026-11-19T02:00:00Z"}
    service, updates, _ = held_service(miners, now=[DEADLINE + timedelta(hours=2)], hold=hold)
    assert service.freeze_participants("arena-2026-11-19") == [{key: BASELINE[key] for key in FIELDS}]
    assert ("arena-2026-11-19", "miner-0", "accepted", "rejected",
            {"rejection_rule": "code_review_incomplete"}) in updates


def test_missing_hold_record_stays_unfrozen_instead_of_rejecting(alerts):
    service, updates, _ = held_service([miner(0)], now=[DEADLINE + timedelta(minutes=1)],
                                       hold={"status": "not_open"})
    with pytest.raises(ServiceError, match="code_review_hold_unavailable"):
        service.freeze_participants("arena-2026-11-19")
    assert updates == []
    assert alerts[-1][1]["decision_code"] == "hold_unavailable"


def test_in_flight_review_arms_the_hold_before_the_deadline(alerts):
    # Without an early hold, a claim expiring at the deadline becomes terminal.
    service, _, holds = held_service([miner(0, status="reviewing")], now=[DEADLINE - timedelta(minutes=1)],
                                     hold={"status": "held", "released_at": None})
    with pytest.raises(ServiceError, match="code_review_pending"):
        service.freeze_participants("arena-2026-11-19")
    assert holds == ["arena-2026-11-19"]


@pytest.mark.parametrize("row,blocked", [
    (miner(0, status="pending", attempts=0), True),
    (miner(0, status="reviewing"), True),
    (miner(0, code="code_review_provider_timeout"), True),
    (miner(0, code="code_review_transport_error"), True),
    (miner(0, code="code_review_provider_credit"), False),  # The miner's own key.
    (miner(0, code="code_review_provider_authentication"), False),
    (miner(0, code="code_review_deadline_exceeded"), False),  # Never retryable.
    (miner(0, code="review_response_invalid"), False),
    (miner(0, code="source_contains_credentials"), False),
    (miner(0, status="passed"), False),
    (miner(0, status="rejected"), False),
    # An outage inside an open invalid-output episode is still ours ...
    ({**miner(0, attempts=2), "code_review_doc": {"error_code": "code_review_provider_unavailable",
                                                  "review_invalid_attempts": 1}}, True),
    # ... but an exhausted episode is the miner's own outcome.
    ({**miner(0, attempts=3), "code_review_doc": {"error_code": "code_review_provider_unavailable",
                                                  "review_invalid_attempts": 3}}, False),
    ({**miner(0, status="pending"), "status": "rejected"}, False),
])
def test_infrastructure_blocked_counts_only_our_failures(row, blocked):
    assert code_review_policy.infrastructure_blocked(row) is blocked


def catalog(*tags):
    return {"data": {"endpoints": [{"tag": tag} for tag in tags]}}


def test_route_match_accepts_variants_of_a_provider_slug_only():
    assert preflight.route_present(catalog("together"), "together")
    assert preflight.route_present(catalog("google-vertex/global"), "google-vertex")
    assert preflight.route_present(catalog("moonshotai/mxfp4"), "moonshotai/mxfp4")
    assert not preflight.route_present(catalog("moonshotai"), "moonshotai/mxfp4")
    assert not preflight.route_present(catalog("nebius/fp8", "togetherx"), "together")
    with pytest.raises(ValueError):
        preflight.route_present({"data": {}}, "together")


def test_pinned_routes_are_the_request_routes():
    for model in preflight.PINNED_REVIEW_MODELS:
        assert code_review.review_provider_only(model) == [code_review.REVIEW_PROVIDER_ROUTES[model]]
    # The request body itself is covered in prospective_admission_primary_test.


def test_preflight_alerts_once_per_interval_for_a_withdrawn_route():
    def fetch(model):
        if model == code_review.PROSPECTIVE_REVIEW_MODEL:
            return catalog("nebius/fp8")  # The Nov 18 withdrawal.
        if model == code_review.FALLBACK_REVIEW_MODEL:
            raise urllib.error.URLError("offline")
        return catalog(code_review.REVIEW_PROVIDER_ROUTES[model])
    alerted, clock = [], [100.0]
    check = preflight.ReviewRoutePreflight(fetch, interval_seconds=600, alert=alerted.append,
                                           monotonic=lambda: clock[0])
    results = check.run()
    status = {row["model"]: row["status"] for row in results}
    assert status[code_review.PROSPECTIVE_REVIEW_MODEL] == "missing"
    assert status[code_review.FALLBACK_REVIEW_MODEL] == "unavailable"
    assert [row["route"] for row in alerted] == ["together"]
    clock[0] += 599
    assert check.run() is None and len(alerted) == 1
    clock[0] += 1
    assert check.run() is not None and len(alerted) == 2


def test_missing_route_alert_uses_a_registered_failure_code(monkeypatch):
    captured = []
    monkeypatch.setattr(leadpoet_observability, "capture_failure",
                        lambda code, **fields: captured.append((code, fields)) or True)
    preflight.alert_missing({"model": "m", "route": "together", "status": "missing"})
    assert captured[0][0] == "arena.review_route_missing"
    assert captured[0][0] in leadpoet_observability.sentry_operations.INCIDENT_FAILURE_CODES
    assert captured[0][1]["correlation_id"] == "review-route:m:together"


def route_service(rounds, check):
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(review_route_preflight=check)
    service.active_rounds = lambda: rounds
    return service


def test_service_checks_routes_only_while_a_round_is_open():
    missing = [{"model": "m", "route": "together", "status": "missing"},
               {"model": "n", "route": "azure", "status": "unavailable"}]
    check = SimpleNamespace(run=lambda: missing)
    assert route_service([], None).check_review_routes() == {"status": "disabled"}
    assert route_service([{"status": "stage1_running"}], check).check_review_routes() == {"status": "idle"}
    assert route_service([{"status": "open"}], check).check_review_routes() == {
        "status": "missing", "missing": [{"model": "m", "route": "together"}], "unavailable": 1}
    assert route_service([{"status": "open"}], SimpleNamespace(run=lambda: None)).check_review_routes() == {
        "status": "throttled"}


def test_driver_reports_missing_routes_and_survives_a_failed_check():
    ok = SimpleNamespace(check_review_routes=lambda: {"status": "missing", "missing": [
        {"model": "deepseek/x", "route": "together"}]})
    assert driver._check_review_routes(ok) == "review routes missing deepseek/x@together"
    assert driver._check_review_routes(SimpleNamespace(check_review_routes=lambda: {"status": "ok"})) == "idle"
    assert driver._check_review_routes(SimpleNamespace()) == "idle"

    def broken():
        raise RuntimeError("down")
    assert driver._check_review_routes(SimpleNamespace(check_review_routes=broken)) == \
        "failed review_route_preflight: RuntimeError"


def test_decide_review_hold_rejects_unknown_actions():
    service = object.__new__(ArenaService)
    for action in ("reopen", "extend"):
        with pytest.raises(ServiceError, match="review_hold_action_invalid"):
            service.decide_review_hold("arena-2026-11-19", action)


def rerun_service(rows, *, status="open", hold=None):
    calls = []
    state = {"hold": hold}
    round_row = {"round_id": "arena-2026-11-19", "status": status,
                 "configuration_doc": {"baseline_hotkey": "host", "schedule": {
                     "submission_cutoff": "2026-11-19T00:00:00Z",
                     "benchmark_deadline": DEADLINE.isoformat()}}}

    def hold_round_review(round_id):
        calls.append(("hold", round_id))
        state["hold"] = state["hold"] or {"status": "held", "released_at": None,
                                          "review_deadline": "2026-11-19T01:30:01Z"}
        return dict(state["hold"])

    def decide(round_id, action):
        calls.append((action, round_id))
        return {"status": "extended", "review_deadline": "2026-11-19T02:30:01Z"}
    service = object.__new__(ArenaService)
    service._clock = lambda: DEADLINE + timedelta(hours=1)
    service._round = lambda _: round_row
    service._invalidate_hot_round = lambda: None
    service._store = SimpleNamespace(
        list_submissions=lambda _, status: [BASELINE, *rows] if status == "accepted" else [],
        hold_round_review=hold_round_review, decide_review_hold=decide)
    return service, calls


def test_rerun_requeues_reviews_that_failed_on_our_side_idempotently():
    service, calls = rerun_service([miner(0), miner(1, status="passed")])
    first = service.decide_review_hold("arena-2026-11-19", "rerun")
    assert first == {"status": "requeued", "blocked": 1, "review_deadline": "2026-11-19T02:30:01Z"}
    assert service.decide_review_hold("arena-2026-11-19", "rerun") == first
    assert calls == [("hold", "arena-2026-11-19"), ("extend", "arena-2026-11-19")] * 2


@pytest.mark.parametrize("rows,status,hold,code", [
    ([miner(0)], "stage1_running", None, "review_rerun_round_not_open"),
    ([miner(0, code="code_review_provider_credit")], "open", None, "review_rerun_nothing_blocked"),
    ([miner(0, status="rejected")], "open", None, "review_rerun_nothing_blocked"),
    ([miner(0)], "open", {"status": "existing", "released_at": "2026-11-19T01:00:00Z"}, "review_hold_released"),
])
def test_rerun_refuses_published_rounds_and_miners_own_failures(rows, status, hold, code):
    service, calls = rerun_service(rows, status=status, hold=hold)
    with pytest.raises(ServiceError, match=code):
        service.decide_review_hold("arena-2026-11-19", "rerun")
    assert ("extend", "arena-2026-11-19") not in calls


def test_admin_review_hold_command(capsys):
    decisions = []
    row = {"round_id": "arena-2026-11-19", "status": "open"}

    def decide(round_id, action):
        if action == "release" and decisions:
            raise ServiceError("review_hold_action_invalid", 400)
        decisions.append((round_id, action))
        return {"status": "requeued"}
    service = SimpleNamespace(store=SimpleNamespace(get_round=lambda _: row), decide_review_hold=decide)
    parse = lab_arena_admin.build_parser().parse_args
    assert lab_arena_admin.run(parse(["review-hold", "--round", "arena-2026-11-19", "--action", "rerun",
                                      "--dry-run"]), service) == 0
    assert decisions == []
    assert lab_arena_admin.run(parse(["review-hold", "--round", "arena-2026-11-19", "--action", "rerun"]),
                               service) == 0
    assert decisions == [("arena-2026-11-19", "rerun")]
    assert lab_arena_admin.run(parse(["review-hold", "--round", "arena-2026-11-19", "--action", "release"]),
                               service) == 4
    with pytest.raises(SystemExit):
        parse(["review-hold", "--round", "r", "--action", "reopen"])
    assert isinstance(parse(["status"]), argparse.Namespace)
