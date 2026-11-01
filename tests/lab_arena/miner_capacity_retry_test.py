"""Only explicit server retry requests can repeat the same intake operation."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from lab_arena import contracts, miner_submit
from lab_arena.api import create_app
from lab_arena.service import ArenaService
from tests.lab_arena.cutover_opening_test import bare_service, round_row
from tests.lab_arena.test_miner_submit import (
    CREDENTIALS, NOW, _Keypair, _Response, _Session, _agent_source, _valid_signature,
)


BUDGET = miner_submit.SUBMISSION_RETRY_BUDGET_SECONDS
OVER_BUDGET = str(BUDGET + 1)


class Response(_Response):
    def __init__(self, status=503, *, retry_after=None, document=None):
        super().__init__(status, document or {"detail": "Gateway at capacity - retry shortly"})
        self.headers = {} if retry_after is None else {"Retry-After": retry_after}


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(elapsed=0.0, sleeps=[])

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.elapsed += seconds

    instant = datetime(2026, 10, 17, tzinfo=timezone.utc)

    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return (instant + timedelta(seconds=state.elapsed)).astimezone(tz)

    monkeypatch.setattr(miner_submit, "time", SimpleNamespace(
        monotonic=lambda: state.elapsed, sleep=sleep,
    ))
    monkeypatch.setattr(miner_submit, "datetime", WallClock)
    state.instant = instant
    return state


@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("header,wait", [("15", 15), (" 15 ", 15), ("0", 1), ("date", 15), ("past-date", 1)])
def test_server_retry_after_repeats_exact_signed_operation(clock, finalize, header, wait):
    if header == "date":
        header = format_datetime(clock.instant + timedelta(seconds=15), usegmt=True)
    elif header == "past-date":
        header = format_datetime(clock.instant - timedelta(seconds=15), usegmt=True)
    envelope = {"round_id": "arena-2026-10-18", "request_id": "same-signed-request"}
    calls = []
    responses = iter([Response(retry_after=header), Response(200)])

    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return next(responses)

    response = miner_submit._post_submission_with_retry(
        Session(), "https://arena.example", envelope,
        submission_id="sub-same" if finalize else None,
    )
    assert response.status_code == 200
    assert clock.sleeps == [wait]
    assert calls[0] == calls[1]
    assert all(call[1]["json"] is envelope for call in calls)
    assert calls[0][0].endswith("/sub-same/finalize" if finalize else "/presign")
    assert calls[0][1]["timeout"] == 60 and calls[0][1]["allow_redirects"] is False


@pytest.mark.parametrize("header", [None, "", "-1", "1.5", "nan", OVER_BUDGET, "9" * 129, True,
                                   "Sat, 17 Oct 2026 00:00:15"])
@pytest.mark.parametrize("finalize", [False, True])
def test_missing_invalid_or_over_budget_retry_after_stays_terminal(clock, header, finalize):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Response(retry_after=header)

    response = miner_submit._post_submission_with_retry(
        Session(), "https://arena.example", {}, submission_id="sub-same" if finalize else None,
    )
    assert response.status_code == 503
    assert len(calls) == 1 and clock.sleeps == []


@pytest.mark.parametrize("status", [401, 402, 403, 404, 409, 429, 502, 504])
@pytest.mark.parametrize("finalize", [False, True])
def test_real_denials_and_other_http_failures_are_not_retried(clock, status, finalize):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Response(status, retry_after="15")

    response = miner_submit._post_submission_with_retry(
        Session(), "https://arena.example", {}, submission_id="sub-same" if finalize else None,
    )
    assert response.status_code == status
    assert len(calls) == 1 and clock.sleeps == []


def test_capacity_retry_cannot_extend_existing_elapsed_budget(clock):
    attempts = []

    class Session:
        def post(self, *args, **kwargs):
            attempts.append(clock.elapsed)
            return Response(retry_after="15")

    assert miner_submit._post_submission_with_retry(Session(), "https://arena.example", {}).status_code == 503
    assert attempts == list(range(0, BUDGET + 1, 15))
    assert clock.elapsed == BUDGET


def test_retry_after_is_never_shortened_to_fit_remaining_budget(clock):
    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            clock.elapsed += BUDGET - 10
            return Response(retry_after="15")

    session = Session()
    assert miner_submit._post_submission_with_retry(session, "https://arena.example", {}).status_code == 503
    assert session.calls == 1 and clock.sleeps == []


def test_finalize_does_not_inherit_presign_opening_retry(clock):
    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return Response(document={"code": "round_opening"})

    session = Session()
    assert miner_submit._post_submission_with_retry(session, "https://arena.example", {}, submission_id="sub-same").status_code == 503
    assert session.calls == 1 and clock.sleeps == []


def test_transport_exception_is_not_blindly_replayed(clock):
    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise requests.ConnectionError("untrusted transport detail")

    session = Session()
    with pytest.raises(miner_submit.MinerSubmissionError) as caught:
        miner_submit._post_submission_with_retry(session, "https://arena.example", {})
    assert caught.value.code == "arena_unreachable"
    assert "untrusted" not in str(caught.value)
    assert session.calls == 1 and clock.sleeps == []


def test_presign_and_finalize_recover_without_an_extra_upload_or_new_identity(tmp_path, clock):
    class Session(_Session):
        def __init__(self):
            super().__init__({"open_round": {"round_id": "arena-2026-09-04", "status": "open"}})
            self.attempts = []

        def post(self, url, **kwargs):
            self.attempts.append((url, kwargs["json"]))
            if sum(previous == url for previous, _ in self.attempts) == 1:
                return Response(retry_after="15")
            return super().post(url, **kwargs)

    session = Session()
    result = miner_submit.submit_agent_source(
        source_dir=_agent_source(tmp_path), api_base_url="https://arena.example",
        keypair=_Keypair(), credentials=CREDENTIALS, session=session,
        now=lambda: NOW + clock.elapsed,
    )
    assert result["status"] == "accepted" and result["round_id"] == "arena-2026-09-04"
    assert len(session.attempts) == 4 and len(session.uploads) == 1
    assert session.attempts[0][1] is session.attempts[1][1]
    assert session.attempts[2][1] is session.attempts[3][1]
    assert clock.sleeps == [15, 15]
    _valid_signature(session.attempts[0][1], contracts.SCOPE_SUBMISSION_PRESIGN)
    _valid_signature(session.attempts[2][1], contracts.SCOPE_SUBMISSION_FINALIZE)
    assert session.attempts[2][1]["body"]["credentials"] == CREDENTIALS


@pytest.mark.parametrize("finalize", [False, True])
def test_retry_crossing_original_cutoff_stops_on_real_409(clock, finalize):
    cutoff = clock.instant + timedelta(seconds=5)
    row = round_row("arena-2026-10-18", clock.instant - timedelta(days=1), cutoff)
    service = bare_service(clock.instant)
    service.now = lambda: clock.instant + timedelta(seconds=clock.elapsed)
    service._store = SimpleNamespace(get_round=lambda _rid: row)
    service._require_round_mode = lambda resolved: resolved
    service._request_round = lambda envelope, **kwargs: ({}, row)
    handler_calls = []

    def presign(envelope):
        handler_calls.append(envelope)
        service._require_submission_window(row)
        pytest.fail("closed original round must not register a submission")

    service.handle_submission_presign = presign
    service.handle_submission_finalize = lambda _id, envelope: presign(envelope)
    envelope = {"round_id": row["round_id"], "request_id": "same-request",
                "body": {"submission_id": "sub-same"}}
    with TestClient(create_app(service)) as api:
        class Session:
            calls = 0

            def post(self, url, **kwargs):
                self.calls += 1
                assert kwargs["json"] is envelope
                if self.calls == 1:
                    return Response(retry_after="15")
                path = "/arena/v1/submissions/sub-same/finalize" if finalize else "/arena/v1/submissions/presign"
                return api.post(path, json=kwargs["json"])

        session = Session()
        response = miner_submit._post_submission_with_retry(
            session, "https://arena.example", envelope, submission_id="sub-same" if finalize else None,
        )
    assert response.status_code == 409 and response.json()["code"] == "submission_window_closed"
    assert session.calls == 2 and len(handler_calls) == 1
    assert envelope["round_id"] == row["round_id"] and clock.sleeps == [15]


def test_accepted_finalize_replay_uses_existing_owner_bound_result():
    row = {"submission_id": "sub-same", "round_id": "arena-2026-10-18",
           "miner_hotkey": "5" * 48, "status": "accepted",
           "source_ref": "arena/arena-2026-10-18/sources/sub-same.tar.gz", "source_size_bytes": 100}
    body = {key: row[key] for key in ("submission_id", "source_ref", "source_size_bytes")}
    body["credentials"] = CREDENTIALS
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 10, 17, tzinfo=timezone.utc)
    service._config = SimpleNamespace(credential_manager=SimpleNamespace(
        validate_and_encrypt=lambda *args, **kwargs: pytest.fail("accepted replay cannot mutate credentials"),
    ))
    service._request_round = lambda *args, **kwargs: (
        {"hotkey": row["miner_hotkey"], "body": body},
        round_row(row["round_id"], service.now(), service.now() + timedelta(days=1)),
    )
    reads = []

    def credential(submission_id, owner, provider):
        assert submission_id == row["submission_id"] and owner == row["miner_hotkey"]
        reads.append(provider)
        return {"provider": provider}

    service._store = SimpleNamespace(
        get_submission=lambda _id: row,
        get_submission_credential=credential,
        accept_submission_source_with_credentials=lambda *args, **kwargs: pytest.fail("accepted replay cannot reaccept source"),
    )
    expected = {"status": "accepted", "submission_id": row["submission_id"]}
    assert service.handle_submission_finalize(row["submission_id"], {}) == expected
    assert service.handle_submission_finalize(row["submission_id"], {}) == expected
    assert row["status"] == "accepted" and body["credentials"] == CREDENTIALS
    assert reads == ["openrouter", "deepline", "scrapingdog"] * 2
