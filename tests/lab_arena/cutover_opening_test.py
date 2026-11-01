"""The daily cutoff must keep intake separate from older-round advancement."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, miner_submit
from lab_arena.api import create_app
from lab_arena.driver import drive_once, next_tick_delay
from lab_arena.service import ArenaService, ServiceError


UTC = timezone.utc


def moment(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def round_row(round_id: str, opening: datetime, cutoff: datetime) -> dict:
    schedule = {
        "submission_open": opening.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "submission_cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {"round_id": round_id, "status": "open",
            "configuration_doc": {"mode": "live", "schedule": schedule},
            "schedule": schedule}


def bare_service(now: datetime) -> ArenaService:
    service = object.__new__(ArenaService)
    service.now = lambda: now
    service._pinned_round_id = lambda: None
    service._config = SimpleNamespace(
        mode="live", defaults=SimpleNamespace(
            daily_cutoff_hour_utc=0, min_submission_hours=6)
    )
    service._require_round_ownership = lambda _round_id: None
    service.validate_request = lambda _envelope, **_kwargs: {
        "round_id": _envelope["round_id"], "hotkey": "miner"
    }
    return service


def test_expired_open_row_does_not_block_new_daily_creation():
    service = bare_service(moment(14))
    prior = round_row("arena-2026-10-14", moment(13), moment(14))
    service._chain_scope = lambda: ("finney", 71)
    service._store = SimpleNamespace(
        list_rounds=lambda **_kwargs: [prior],
        get_round=lambda _round_id: None,
    )
    service.create_round = lambda cutoff: {"round_id": "arena-2026-10-15"}

    assert service.open_round() is None
    result = service.ensure_daily_round()
    assert result["round_id"] == "arena-2026-10-15"
    assert result["cutoff"] == "2026-10-15T00:00:00Z"
    assert ArenaService._round_accepts_submissions(prior, moment(14) - timedelta(seconds=1))
    assert not ArenaService._round_accepts_submissions(prior, moment(14))


def test_future_open_row_remains_single_creation_target():
    service = bare_service(moment(14, 21))
    rows = {}
    service._chain_scope = lambda: ("finney", 71)
    service._store = SimpleNamespace(
        list_rounds=lambda **_kwargs: list(rows.values()),
        get_round=lambda round_id: rows.get(round_id),
    )

    def create(cutoff):
        round_id = "arena-" + cutoff.strftime("%Y-%m-%d")
        rows[round_id] = round_row(round_id, cutoff - timedelta(days=1), cutoff)
        return {"round_id": round_id}

    service.create_round = create
    first = service.ensure_daily_round()
    assert first["round_id"] == "arena-2026-10-16"
    assert service.open_round() is None  # Intake cannot open a day early.
    second = service.ensure_daily_round()
    assert second["status"] == "existing"
    assert second["round_id"] == first["round_id"]
    assert len(rows) == 1


def test_only_due_canonical_presign_gets_retryable_refusal():
    service = bare_service(moment(14, 0, 1))
    service._store = SimpleNamespace(get_round=lambda _round_id: None)

    with pytest.raises(ServiceError) as due:
        service._request_round({"round_id": "arena-2026-10-15"},
                               scope=contracts.SCOPE_SUBMISSION_PRESIGN)
    assert (due.value.code, due.value.status) == ("round_opening", 503)

    for round_id, scope in [
        ("arena-2026-10-16", contracts.SCOPE_SUBMISSION_PRESIGN),
        ("arena-2026-10-15", contracts.SCOPE_SUBMISSION_FINALIZE),
    ]:
        with pytest.raises(ServiceError) as other:
            service._request_round({"round_id": round_id}, scope=scope)
        assert (other.value.code, other.value.status) == ("round_unknown", 404)

    service.now = lambda: moment(14) + timedelta(seconds=180)
    with pytest.raises(ServiceError) as late:
        service._request_round({"round_id": "arena-2026-10-15"},
                               scope=contracts.SCOPE_SUBMISSION_PRESIGN)
    assert late.value.code == "round_unknown"


def test_presign_just_before_the_opening_is_retryable():
    """A miner racing the cutoff names the next round seconds early."""

    service = bare_service(moment(15) - timedelta(seconds=30))
    service._store = SimpleNamespace(get_round=lambda _round_id: None)

    with pytest.raises(ServiceError) as due:
        service._request_round({"round_id": "arena-2026-10-16"},
                               scope=contracts.SCOPE_SUBMISSION_PRESIGN)
    assert (due.value.code, due.value.status) == ("round_opening", 503)

    # Outside the bounded window, and for any other round, the refusal stays
    # terminal so a wrong round id is never retried indefinitely.
    service.now = lambda: moment(15) - timedelta(seconds=91)
    with pytest.raises(ServiceError) as early:
        service._request_round({"round_id": "arena-2026-10-16"},
                               scope=contracts.SCOPE_SUBMISSION_PRESIGN)
    assert early.value.code == "round_unknown"

    service.now = lambda: moment(15) - timedelta(seconds=30)
    with pytest.raises(ServiceError) as other:
        service._request_round({"round_id": "arena-2026-10-17"},
                               scope=contracts.SCOPE_SUBMISSION_PRESIGN)
    assert other.value.code == "round_unknown"


def test_same_signed_presign_can_retry_without_nonce_write():
    service = bare_service(moment(14, 0, 1))
    reads = []
    service.validate_request = ArenaService.validate_request.__get__(service)
    service._config.verify_signature = lambda _hotkey, _signature, _message: True
    service._store = SimpleNamespace(
        get_round=lambda round_id: reads.append(round_id) or None)
    request = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id="arena-2026-10-15", hotkey="5" * 48,
        body={}, timestamp=int(service.now().timestamp()),
        sign_message=lambda _message: "a" * 128,
    )
    with TestClient(create_app(service)) as client:
        first = client.post("/arena/v1/submissions/presign", json=request)
        second = client.post("/arena/v1/submissions/presign", json=request)
    for response in (first, second):
        assert response.status_code == 503
        assert response.json()["code"] == "round_opening"
        assert response.headers["Retry-After"] == "2"
    assert reads == ["arena-2026-10-15", "arena-2026-10-15"]
    service.now = lambda: moment(14, 0, 2)
    assert service.validate_request(request, scope=contracts.SCOPE_SUBMISSION_PRESIGN,
                                    round_id=None)["request_id"] == request["request_id"]


def test_public_current_hides_expired_open_row():
    service = bare_service(moment(14))
    prior = round_row("arena-2026-10-14", moment(13), moment(14))
    service.active_rounds = lambda **_kwargs: [prior]
    service.latest_published_round = lambda **_kwargs: None
    service._config.mode = "shadow"
    service._config.chain = SimpleNamespace(current_settlement_epoch=lambda: None)
    assert service.public_current()["open_round"] is None


def test_public_current_hides_future_open_row():
    service = bare_service(moment(14, 21))
    future = round_row("arena-2026-10-16", moment(15), moment(16))
    service.active_rounds = lambda **_kwargs: [future]
    service.latest_published_round = lambda **_kwargs: None
    service._config.mode = "shadow"
    service._config.chain = SimpleNamespace(current_settlement_epoch=lambda: None)
    assert service.public_current()["open_round"] is None
    service.now = lambda: moment(15)
    assert service.public_current()["open_round"]["round_id"] == future["round_id"]


def test_driver_creates_due_round_before_advancing_old_work():
    calls = []
    service = SimpleNamespace(
        promote_pending_baselines=lambda: {"promoted": 0},
        ensure_daily_round=lambda: calls.append("ensure") or {
            "status": "created", "round_id": "arena-2026-10-15"},
        active_rounds=lambda: [{"round_id": "arena-2026-10-14"}],
        advance_round=lambda _round_id: calls.append("advance"),
        activate_pending_rewards=lambda: {"activated": 0},
        reconcile_closed_provider_costs=lambda: {"status": "none"},
    )
    assert "created arena-2026-10-15" in drive_once(service)
    assert calls == ["ensure", "advance"]


def test_driver_wakes_at_or_immediately_after_cutoff():
    assert next_tick_delay(60, 0, started=moment(13, 23, 59),
                           now=moment(13, 23, 59) + timedelta(seconds=30)) == 30
    assert next_tick_delay(60, 0, started=moment(13, 23, 59),
                           now=moment(14) + timedelta(milliseconds=278)) == 0
    assert next_tick_delay(60, None, started=moment(13, 23, 59),
                           now=moment(14)) == 60


def test_client_retries_only_round_opening_presign(monkeypatch):
    class Response:
        def __init__(self, status, code):
            self.status_code = status
            self.code = code

        def json(self):
            return {"code": self.code}

    class Session:
        def __init__(self, replies):
            self.replies = iter(replies)
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return next(self.replies)

    monkeypatch.setattr(miner_submit.time, "sleep", lambda _delay: None)
    due = Session([Response(503, "round_opening"), Response(200, None)])
    assert miner_submit._post_submission_with_retry(due, "https://example.test", {}).status_code == 200
    assert due.calls == 2
    unrelated = Session([Response(503, "provider_unavailable")])
    assert miner_submit._post_submission_with_retry(unrelated, "https://example.test", {}).status_code == 503
    assert unrelated.calls == 1


def test_presign_does_not_retry_after_opening_budget(monkeypatch):
    class Response:
        status_code = 503

        def json(self):
            return {"code": "round_opening"}

    class Session:
        calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    observed = iter((0.0, 271.0))
    monkeypatch.setattr(miner_submit.time, "monotonic", lambda: next(observed))
    session = Session()
    response = miner_submit._post_submission_with_retry(
        session, "https://example.test", {})
    assert response.status_code == 503
    assert session.calls == 1


@pytest.mark.parametrize("round_id,expected_status,expected_code", [
    ("arena-2026-10-17", 404, "round_unknown"),
    ("not-a-round", 404, "round_unknown"),
    ("", 400, "request_invalid:request round_id required"),
])
def test_opening_presign_unknown_and_malformed_signed_requests_stay_terminal(
    round_id, expected_status, expected_code,
):
    service = bare_service(moment(15) - timedelta(seconds=30))
    service.validate_request = ArenaService.validate_request.__get__(service)
    service._config.verify_signature = lambda _hotkey, _signature, _message: True
    service._store = SimpleNamespace(get_round=lambda _round_id: None)
    request = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id=round_id, hotkey="5" * 48, body={},
        timestamp=int(service.now().timestamp()), sign_message=lambda _message: "a" * 128,
    )
    with TestClient(create_app(service)) as client:
        response = client.post("/arena/v1/submissions/presign", json=request)
    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert "Retry-After" not in response.headers


def test_client_waits_for_current_round_only_near_cutoff(monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, document):
            self.document = document

        def json(self):
            return self.document

    class Session:
        def __init__(self, replies):
            self.replies = iter(replies)
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return next(self.replies)

    cutoff = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    prior = {"round": {"schedule": {"submission_cutoff": cutoff}},
             "open_round": None}
    new = {"open_round": {"round_id": "arena-2026-10-15", "status": "open"}}
    monkeypatch.setattr(miner_submit.time, "sleep", lambda _delay: None)
    opening = Session([Response(prior), Response(new)])
    assert miner_submit.find_open_round("https://example.test", session=opening) == "arena-2026-10-15"
    assert opening.calls == 2
    old = Session([Response({"round": {"schedule": {
        "submission_cutoff": (datetime.now(UTC) - timedelta(minutes=5)).isoformat()}},
        "open_round": None})])
    with pytest.raises(miner_submit.MinerSubmissionError) as closed:
        miner_submit.find_open_round("https://example.test", session=old)
    assert closed.value.code == "submission_window_closed"
    assert old.calls == 1


@pytest.mark.parametrize("seconds_before,created_after", [(60, 44), (90, 89)])
def test_presign_retry_crosses_opening_and_waits_for_creation(
    monkeypatch, seconds_before, created_after,
):
    opening = moment(15)
    elapsed = [0.0]
    service = bare_service(opening - timedelta(seconds=seconds_before))
    service.now = lambda: opening + timedelta(seconds=elapsed[0] - seconds_before)
    service.validate_request = ArenaService.validate_request.__get__(service)
    service._config.verify_signature = lambda _hotkey, _signature, _message: True
    service._require_round_mode = lambda row: row
    row = round_row("arena-2026-10-16", opening, moment(16))
    # No store mutation or nonce API exists: refused requests must not use one.
    service._store = SimpleNamespace(get_round=lambda _round_id: (
        row if service.now() >= opening + timedelta(seconds=created_after) else None
    ))

    def presign(envelope):
        validated, resolved = service._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        )
        service._require_submission_window(resolved)
        return {"status": "accepted", "request_id": validated["request_id"]}

    service.handle_submission_presign = presign
    request = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id=row["round_id"], hotkey="5" * 48, body={},
        timestamp=int(service.now().timestamp()), sign_message=lambda _message: "a" * 128,
    )
    requests = []
    replies = []

    def sleep(delay):
        elapsed[0] += delay

    monkeypatch.setattr(miner_submit, "time", SimpleNamespace(
        monotonic=lambda: elapsed[0], sleep=sleep,
    ))
    with TestClient(create_app(service)) as client:
        class Session:
            def post(self, _url, *, json, **_kwargs):
                requests.append(json)
                response = client.post("/arena/v1/submissions/presign", json=json)
                replies.append(response)
                return response

        response = miner_submit._post_submission_with_retry(
            Session(), "https://example.test", request,
        )
    assert response.status_code == 200
    assert elapsed[0] >= seconds_before + created_after
    assert all(reply.status_code == 503 for reply in replies[:-1])
    assert all(reply.json()["code"] == "round_opening" for reply in replies[:-1])
    assert all(reply.headers["Retry-After"] == "2" for reply in replies[:-1])
    assert all(envelope is request for envelope in requests)
    assert (service.now().timestamp() - request["timestamp"]
            <= contracts.REQUEST_TIMESTAMP_WINDOW_SECONDS)
    assert service.validate_request(
        request, scope=contracts.SCOPE_SUBMISSION_PRESIGN, round_id=None,
    )["request_id"] == request["request_id"]


@pytest.mark.parametrize("offset,expected", [
    (-90.001, False), (-90, True), (-0.001, True),
    (0, True), (89.999, True), (90, True),
    (179.999, True), (180, False),
])
def test_opening_retry_window_exact_edges(offset, expected):
    service = bare_service(moment(15) + timedelta(seconds=offset))
    assert service._due_daily_round_opening("arena-2026-10-16") is expected


def test_slow_round_creation_still_reports_opening_not_unknown():
    """A failed creation attempt plus its retry must not become a 404.

    Production 2026-11-01: creation failed at the 00:00:00 opening and the
    round was first usable at 00:02:24. Under the old symmetric 90s window
    every presign after 00:01:30 was a terminal round_unknown.
    """

    service = bare_service(moment(15) + timedelta(seconds=144))
    service._store = SimpleNamespace(get_round=lambda _round_id: None)
    envelope = {"round_id": "arena-2026-10-16"}
    with pytest.raises(ServiceError) as refusal:
        service._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        )
    assert (refusal.value.code, refusal.value.status) == ("round_opening", 503)


def test_a_non_canonical_round_is_still_unknown_inside_the_window():
    service = bare_service(moment(15) + timedelta(seconds=144))
    service._store = SimpleNamespace(get_round=lambda _round_id: None)
    with pytest.raises(ServiceError) as refusal:
        service._request_round(
            {"round_id": "arena-2026-09-01"},
            scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        )
    assert (refusal.value.code, refusal.value.status) == ("round_unknown", 404)


def test_current_round_keeps_exact_cutoff_and_future_round_cannot_open_early():
    cutoff = moment(15)
    service = bare_service(cutoff - timedelta(microseconds=1))
    current = round_row("arena-2026-10-15", moment(14), cutoff)
    future = round_row("arena-2026-10-16", cutoff, moment(16))
    service._require_submission_window(current)
    assert ArenaService._round_accepts_submissions(current, service.now())
    with pytest.raises(ServiceError) as early:
        service._require_submission_window(future)
    assert (early.value.code, early.value.status) == ("submission_window_closed", 409)
    service.now = lambda: cutoff
    with pytest.raises(ServiceError) as closed:
        service._require_submission_window(current)
    assert (closed.value.code, closed.value.status) == ("submission_window_closed", 409)
    assert not ArenaService._round_accepts_submissions(current, service.now())
    service._require_submission_window(future)


@pytest.mark.parametrize("status,code", [
    (404, "round_unknown"), (409, "submission_window_closed"),
    (503, "provider_unavailable"), (200, None),
])
def test_opening_client_keeps_terminal_responses_terminal(monkeypatch, status, code):
    class Response:
        status_code = status

        def json(self):
            return {"code": code}

    class Session:
        calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    monkeypatch.setattr(miner_submit, "time", SimpleNamespace(
        monotonic=lambda: 0.0,
        sleep=lambda _delay: pytest.fail("terminal response must not retry"),
    ))
    session = Session()
    assert miner_submit._post_submission_with_retry(
        session, "https://example.test", {},
    ).status_code == status
    assert session.calls == 1
