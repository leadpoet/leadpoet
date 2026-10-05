"""Owner authentication and bounded public receipts for credit recovery."""
from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import requests

from lab_arena import contracts, miner_cli, miner_submit
from lab_arena.api import create_app
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore

NOW = 1791172800
HOTKEY = "5" * 48
ROUND = "arena-2026-10-05"
SUBMISSION = "sub-credit-retry"


class Keypair:
    ss58_address = HOTKEY

    def sign(self, message):
        return hashlib.sha512(message).digest()


class Store:
    def __init__(self):
        self.calls = []
        self.receipt = {"status": "queued", "requeued_count": 1}
        self.owner = HOTKEY
        self.round_id = ROUND

    def get_round(self, round_id):
        return {"round_id": round_id, "configuration_doc": {"mode": "shadow"}}

    def get_submission(self, submission_id):
        return {"round_id": self.round_id, "miner_hotkey": self.owner}

    def retry_credit_failures(self, *args):
        self.calls.append(args)
        return self.receipt


def envelope(**changes):
    values = dict(
        scope=contracts.SCOPE_SUBMISSION_CREDIT_RETRY, round_id=ROUND,
        hotkey=HOTKEY, body={"submission_id": SUBMISSION}, timestamp=NOW,
        sign_message=lambda value: Keypair().sign(value.encode()).hex(),
    )
    values.update(changes)
    return contracts.build_signed_request(**values)


@pytest.fixture
def client():
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime.fromtimestamp(NOW, timezone.utc)
    service._config = SimpleNamespace(
        mode="shadow", verify_signature=lambda hotkey, signature, message:
        hotkey == HOTKEY and signature == "0x" + Keypair().sign(message.encode()).hex(),
    )
    service._store = Store()
    with TestClient(create_app(service)) as http:
        yield http, service._store


def post(http, value):
    return http.post(
        "/arena/v1/submissions/" + SUBMISSION + "/retry-credit-failures", json=value,
    )


def test_signed_owner_receives_only_safe_receipt(client):
    http, store = client
    request = envelope()
    store.receipt["private_error"] = "must not leave the gateway"
    response = post(http, request)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"status": "queued", "requeued_count": 1}
    assert store.calls == [(ROUND, SUBMISSION, HOTKEY, contracts.request_bytes_hash(request))]


@pytest.mark.parametrize("change", ["forged", "scope", "stale", "path", "fields"])
def test_invalid_requests_never_reach_retry_rpc(client, change):
    http, store = client
    if change == "scope":
        request = envelope(scope=contracts.SCOPE_SUBMISSION_FINALIZE)
    elif change == "stale":
        request = envelope(timestamp=NOW - 3600)
    elif change == "path":
        request = envelope(body={"submission_id": "sub-another"})
    elif change == "fields":
        request = envelope(body={"submission_id": SUBMISSION, "proof": True})
    else:
        request = envelope()
        request["signature"] = "0x" + "00" * 64
    response = post(http, request)
    assert response.status_code in (400, 401)
    assert store.calls == []


@pytest.mark.parametrize("field,value", [("owner", "6" * 48), ("round_id", "arena-other")])
def test_other_owner_or_round_never_reaches_retry_rpc(client, field, value):
    http, store = client
    setattr(store, field, value)
    response = post(http, envelope())
    assert response.status_code == 403
    assert response.json()["code"] == "credit_retry_owner_required"
    assert store.calls == []


@pytest.mark.parametrize("receipt", [
    {"status": "queued", "requeued_count": True},
    {"status": "queued", "requeued_count": 21},
    {"status": "no_eligible", "requeued_count": 1, "reason": "deadline_passed"},
    {"status": "no_eligible", "requeued_count": 0, "reason": "private-secret"},
])
def test_malformed_database_receipt_fails_closed(client, receipt):
    http, store = client
    store.receipt = receipt
    response = post(http, envelope())
    assert response.status_code == 503
    assert response.json()["code"] == "credit_retry_unavailable"
    assert "private-secret" not in response.text


def test_store_passes_only_bound_rpc_parameters():
    calls = []
    def rpc(name, params):
        calls.append((name, params))
        return {"status": "queued", "requeued_count": 1}
    store = ArenaStore(SimpleNamespace(rpc=rpc))
    result = store.retry_credit_failures(ROUND, SUBMISSION, HOTKEY, "sha256:" + "a" * 64)
    assert result["status"] == "queued"
    assert calls == [("lab_arena_retry_credit_failures_v1", {
        "p_round_id": ROUND, "p_submission_id": SUBMISSION,
        "p_miner_hotkey": HOTKEY, "p_request_hash": "sha256:" + "a" * 64,
    })]


def test_client_reuses_signed_request_after_lost_response(monkeypatch):
    sent = []
    def post_request(url, **kwargs):
        sent.append((url, kwargs))
        if len(sent) == 1:
            raise requests.Timeout("private diagnostic")
        return SimpleNamespace(status_code=200, json=lambda: {
            "status": "replayed", "requeued_count": 1, "private": "discard",
        })
    monkeypatch.setattr(miner_submit.time, "sleep", lambda _: None)
    result = miner_submit.retry_credit_failures(
        round_id=ROUND, submission_id=SUBMISSION, api_base_url="https://arena.example",
        keypair=Keypair(), session=SimpleNamespace(post=post_request), now=lambda: NOW,
    )
    assert result == {"status": "replayed", "requeued_count": 1}
    assert len(sent) == 2 and sent[0] == sent[1]
    assert sent[0][1]["allow_redirects"] is False
    assert sent[0][1]["json"]["body"] == {"submission_id": SUBMISSION}
    assert sent[0][1]["json"]["scope"] == contracts.SCOPE_SUBMISSION_CREDIT_RETRY


def test_retry_cli_needs_owner_wallet_and_no_provider_secrets(monkeypatch, capsys):
    monkeypatch.setattr(miner_cli, "_keypair", lambda args: Keypair())
    monkeypatch.setattr(miner_cli, "submission_credentials_from_environment",
                        lambda: pytest.fail("retry must not load provider keys"))
    monkeypatch.setattr(miner_cli, "retry_credit_failures", lambda **kwargs: {
        "status": "queued", "requeued_count": 1,
    })
    assert miner_cli.main([
        "retry-credit-failures", "--round-id", ROUND, "--submission-id", SUBMISSION,
        "--wallet-name", "miner", "--hotkey-name", "owner",
    ]) == 0
    assert '"requeued_count": 1' in capsys.readouterr().out
