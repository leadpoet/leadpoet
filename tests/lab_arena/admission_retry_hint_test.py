"""Temporary intake failures use the client's existing bounded signed retry."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, credentials, miner_submit, submission_rate_limit, submission_similarity
from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.cutover_opening_test import bare_service, round_row
from tests.lab_arena.miner_capacity_retry_test import clock
from tests.lab_arena.test_miner_submit import CREDENTIALS, NOW, _Keypair, _valid_signature


class Session:
    def __init__(self, api):
        self.api = api
        self.calls = []
        self.responses = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.api.post(url, json=kwargs["json"])
        self.responses.append(response)
        return response


@pytest.mark.parametrize("finalize,code", [
    (False, "source_upload_unavailable"),
    (False, "owner_resolution_unavailable"),
    (True, "credential_validation_unavailable"),
    (True, "credential_kms_unavailable"),
    (True, "future_admission_dependency_unavailable"),
])
def test_temporary_admission_retries_the_same_signed_operation(clock, finalize, code):
    scope = (contracts.SCOPE_SUBMISSION_FINALIZE if finalize
             else contracts.SCOPE_SUBMISSION_PRESIGN)
    envelope = contracts.build_signed_request(
        scope=scope, round_id="arena-2026-09-04", hotkey=_Keypair.ss58_address,
        body={"submission_id": "sub-same"} if finalize else {}, timestamp=NOW,
        sign_message=lambda message: _Keypair().sign(message.encode()).hex(),
    )
    calls = []

    def handle(document):
        calls.append(document)
        if len(calls) == 1:
            raise ServiceError(code, 503)
        return {"status": "accepted" if finalize else "upload_ready"}

    service = SimpleNamespace(
        handle_submission_presign=handle,
        handle_submission_finalize=lambda _id, document: handle(document),
    )
    with TestClient(create_app(service)) as api:
        session = Session(api)
        response = miner_submit._post_submission_with_retry(
            session, "http://testserver", envelope,
            submission_id="sub-same" if finalize else None,
        )
    assert response.status_code == 200
    assert session.responses[0].json() == {"status": "rejected", "code": code}
    assert session.responses[0].headers["Retry-After"] == "15"
    assert "Retry-After" not in response.headers
    assert clock.sleeps == [15]
    assert session.calls[0] == session.calls[1]
    assert all(kwargs["json"] is envelope for _, kwargs in session.calls)
    assert calls == [envelope, envelope]
    _valid_signature(envelope, scope)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 429])
@pytest.mark.parametrize("finalize", [False, True])
def test_terminal_admission_response_has_no_retry_hint(clock, status, finalize):
    def deny(*args):
        raise ServiceError("submission_denied", status)

    service = SimpleNamespace(handle_submission_presign=deny, handle_submission_finalize=deny)
    with TestClient(create_app(service)) as api:
        session = Session(api)
        response = miner_submit._post_submission_with_retry(
            session, "http://testserver", {"body": {"submission_id": "sub-same"}},
            submission_id="sub-same" if finalize else None,
        )
    assert response.status_code == status
    assert "Retry-After" not in response.headers
    assert len(session.calls) == 1 and clock.sleeps == []


@pytest.mark.parametrize("path,method,body", [
    ("/arena/v1/runs/claim", "handle_claim", {}),
    ("/arena/v1/runs/run-same/complete", "handle_complete", {"run_id": "run-same"}),
    ("/arena/v1/submissions/sub-same/retry-credit-failures",
     "handle_submission_credit_retry", {"submission_id": "sub-same"}),
])
def test_unrelated_503_does_not_gain_an_admission_retry_hint(path, method, body):
    def unavailable(*args):
        raise ServiceError("dependency_unavailable", 503)

    with TestClient(create_app(SimpleNamespace(**{method: unavailable}))) as api:
        response = api.post(path, json={"body": body})
    assert response.status_code == 503
    assert response.json()["code"] == "dependency_unavailable"
    assert "Retry-After" not in response.headers


def test_temporary_admission_still_stops_at_existing_retry_deadline(clock):
    def unavailable(*args):
        clock.elapsed += 82.5
        raise ServiceError("credential_validation_unavailable", 503)

    service = SimpleNamespace(handle_submission_finalize=unavailable)
    with TestClient(create_app(service)) as api:
        session = Session(api)
        response = miner_submit._post_submission_with_retry(
            session, "http://testserver", {"body": {"submission_id": "sub-same"}},
            submission_id="sub-same",
        )
    assert response.status_code == 503
    assert len(session.calls) == 2 and clock.sleeps == [15]
    assert clock.elapsed == miner_submit.ADMISSION_RETRY_SECONDS


@pytest.fixture
def finalize_stack(monkeypatch, clock):
    row = {"submission_id": "sub-same", "round_id": "arena-2026-10-18",
           "miner_hotkey": _Keypair.ss58_address, "status": "uploading",
           "source_ref": "arena/arena-2026-10-18/sources/sub-same.tar.gz", "source_size_bytes": 100}
    body = {key: row[key] for key in ("submission_id", "source_ref", "source_size_bytes")}
    body["credentials"] = CREDENTIALS
    envelope = {"body": body}
    service = bare_service(clock.instant)
    service.now = lambda: clock.instant + timedelta(seconds=clock.elapsed)
    current_round = round_row(row["round_id"], clock.instant - timedelta(days=1),
                              clock.instant + timedelta(days=1))
    service._request_round = lambda document, **kwargs: (
        {"hotkey": row["miner_hotkey"], "body": document["body"]}, current_round,
    )
    source_reads, validations, accepts, rejections, failures = [], [], [], [], []
    service._validate_uploaded_source = lambda *args, **kwargs: source_reads.append(row["status"]) or b"source"
    monkeypatch.setattr(submission_similarity, "inspect_archive", lambda payload: SimpleNamespace(
        archive_sha256="sha256:" + "1" * 64, normalized_sha256="sha256:" + "2" * 64,
    ))

    def validate(document, **kwargs):
        validations.append((document, kwargs))
        if failures:
            assert row["status"] == "uploading"
            raise failures.pop(0)
        return {"openrouter": "ciphertext", "deepline": "ciphertext"}

    def accept(*args):
        accepts.append(args)
        row["status"] = "accepted"
        return {"status": "ok"}

    service._config = SimpleNamespace(credential_manager=SimpleNamespace(validate_and_encrypt=validate))
    service._submission_request_limiter = submission_rate_limit.SubmissionRequestLimiter(
        clock=lambda: clock.elapsed,
    )
    service._store = SimpleNamespace(
        get_submission=lambda _id: row,
        get_submission_credential=lambda *args: {"ciphertext": "stored"},
        accept_submission_source_with_credentials=accept,
        update_submission=lambda *args: rejections.append(args),
    )
    return SimpleNamespace(service=service, row=row, envelope=envelope, failures=failures,
                           source_reads=source_reads, validations=validations,
                           accepts=accepts, rejections=rejections)


@pytest.mark.parametrize("code", [None, "credential_validation_unavailable", "credential_kms_unavailable"])
def test_finalize_temporary_failure_preserves_source_then_accepts_once(finalize_stack, clock, code):
    stack = finalize_stack
    if code:
        stack.failures.append(credentials.CredentialError(code, retryable=True))
    with TestClient(create_app(stack.service)) as api:
        session = Session(api)
        response = miner_submit._post_submission_with_retry(
            session, "http://testserver", stack.envelope, submission_id=stack.row["submission_id"],
        )
        replay = api.post("/arena/v1/submissions/sub-same/finalize", json=stack.envelope)
    assert response.status_code == replay.status_code == 200, (response.json(), replay.json())
    assert response.json() == replay.json() == {"status": "accepted", "submission_id": "sub-same"}
    assert len(stack.accepts) == 1 and stack.rejections == []
    assert len(stack.validations) == len(stack.source_reads) == (2 if code else 1)
    assert all(status == "uploading" for status in stack.source_reads)
    assert stack.row["status"] == "accepted" and stack.envelope["body"]["credentials"] == CREDENTIALS
    assert clock.sleeps == ([15] if code else [])


@pytest.mark.parametrize("presign_attempts", [1, 5])
def test_sustained_temporary_failure_does_not_exhaust_admission_rate_limit(finalize_stack, clock, presign_attempts):
    stack = finalize_stack
    # Preserve request history from normal presign or paced presign retries.
    for attempt in range(presign_attempts):
        clock.elapsed = attempt * 15
        stack.service._enforce_submission_request_limit(_Keypair.ss58_address)
    started = clock.elapsed
    stack.failures.extend(credentials.CredentialError(
        "credential_validation_unavailable", retryable=True,
    ) for _ in range(13))
    with TestClient(create_app(stack.service)) as api:
        session = Session(api)
        response = miner_submit._post_submission_with_retry(
            session, "http://testserver", stack.envelope, submission_id=stack.row["submission_id"],
        )
    assert response.status_code == 503
    assert response.json()["code"] == "credential_validation_unavailable"
    assert all(reply.status_code == 503 for reply in session.responses)
    assert len(session.calls) == len(stack.validations) == 13
    assert clock.elapsed - started == miner_submit.ADMISSION_RETRY_SECONDS
    assert clock.sleeps == [15] * 12
    assert stack.row["status"] == "uploading" and stack.accepts == stack.rejections == []
