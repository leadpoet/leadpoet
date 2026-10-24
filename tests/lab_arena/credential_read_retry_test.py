"""A credential read may retry once before any provider money is reserved."""

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.credentials import CredentialError
from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport
from lab_arena.submission_runtime import SubmissionProviderKeys
from test_lab_arena_broker import CONTEXT, DL_KEY, FakeLedgerStore, FakeTransport, make_broker


class _Credentials:
    def __init__(self, *, invalid=False):
        self.invalid = invalid

    def runtime_key(self, encrypted, provider):
        assert encrypted == {
            "status": "available",
            "submission_id": CONTEXT.submission_id,
            "miner_hotkey": CONTEXT.miner_hotkey,
            "provider": "deepline",
            "ciphertext_b64": "synthetic-ciphertext",
        }
        assert provider == "deepline"
        if self.invalid:
            raise CredentialError("credential_invalid")
        return DL_KEY


def _stack(credential_replies, *, invalid_secret=False):
    credential_reads = []
    database_calls = []
    replies = list(credential_replies)

    def database_reply(request):
        database_calls.append(request.url.path)
        if request.url.path == "/rest/v1/lab_arena_submissions":
            assert request.method == "GET"
            return httpx.Response(200, json=[{
                "submission_id": CONTEXT.submission_id,
                "round_id": "arena-2026-09-02",
                "miner_hotkey": CONTEXT.miner_hotkey,
                "status": "accepted",
                "is_king": False,
            }])
        if request.url.path == "/rest/v1/rpc/lab_arena_provider_funding":
            return httpx.Response(200, json={
                "status": "available",
                "funding_source": "miner_key",
                "champion_funding": False,
                "credential_submission_id": CONTEXT.submission_id,
                "credential_miner_hotkey": CONTEXT.miner_hotkey,
                "restart_required": False,
            })
        assert request.url.path == "/rest/v1/rpc/lab_arena_get_submission_credential"
        assert request.method == "POST"
        credential_reads.append(request)
        selected = replies.pop(0)
        if selected == "timeout":
            raise httpx.ReadTimeout("private credential diagnostic", request=request)
        if selected == "pool_timeout":
            raise httpx.PoolTimeout("private credential diagnostic", request=request)
        if selected == "http_503":
            return httpx.Response(503, content=b"private credential diagnostic")
        if selected == "http_401":
            return httpx.Response(401, content=b"private credential diagnostic")
        if selected == "missing":
            return httpx.Response(200, json={"status": "missing"})
        assert selected == "available"
        return httpx.Response(200, json={
            "status": "available",
            "submission_id": CONTEXT.submission_id,
            "miner_hotkey": CONTEXT.miner_hotkey,
            "provider": "deepline",
            "ciphertext_b64": "synthetic-ciphertext",
        })

    database = httpx.Client(
        transport=httpx.MockTransport(database_reply), timeout=httpx.Timeout(8.0)
    )
    store = ArenaStore(PostgrestTransport(
        "https://project.example", service_key="sb_secret_test", http_client=database,
    ))
    keys = SubmissionProviderKeys(
        store=store, credentials=_Credentials(invalid=invalid_secret), organizer_keys={},
    )
    ledger = FakeLedgerStore()
    provider = FakeTransport([(200, {"results": []})])
    broker, _, _ = make_broker(
        store=ledger, transport=provider,
        credential_for=keys.credential_for,
        funding_source_for=keys.funding_source_for,
        provider_funding_source_for=keys.provider_funding_source_for,
    )

    class _Service:
        def handle_provider(self, run_id, lease_token, frame, cancel_requested=None):
            assert run_id == CONTEXT.run_id and lease_token == "a" * 64
            result = broker.execute(
                CONTEXT, operation_id=frame["operation_id"],
                parameters=frame["parameters"],
                action_sequence=frame["action_sequence"],
                timeout_ms=frame["timeout_ms"],
                cancel_requested=cancel_requested,
            )
            return result.to_document()

    return database, TestClient(create_app(_Service()), raise_server_exceptions=False), credential_reads, database_calls, ledger, provider


def _provider_request(client, *, sequence=201):
    return client.post(
        "/arena/v1/runs/r1/provider",
        headers={"x-lab-arena-lease": "a" * 64},
        json={
            "operation_id": "deepline.execute",
            "parameters": {"tool": "exa_search", "payload": {"query": "fintech"}},
            "action_sequence": sequence,
            "timeout_ms": 5000,
        },
    )


@pytest.mark.parametrize("first", ["timeout", "http_503"])
def test_transient_credential_read_retries_once_before_one_provider_dispatch(first):
    database, client, reads, calls, ledger, provider = _stack([first, "available"])
    with database, client:
        response = _provider_request(client)
    assert response.status_code == 200
    assert response.json()["status"] == 200
    assert len(reads) == 2
    assert reads[0].content == reads[1].content
    assert all(read.extensions["timeout"]["read"] == 8.0 for read in reads)
    assert ledger.log == ["reserve", "dispatch", "settle"]
    assert len(provider.sent) == 1 and provider.sent[0]["method"] == "POST"
    assert 0 < provider.sent[0]["timeout"] <= 5.0
    assert calls.count("/rest/v1/rpc/lab_arena_get_submission_credential") == 2


@pytest.mark.parametrize("failure", ["timeout", "http_503"])
def test_exhausted_credential_read_is_typed_503_without_ledger_or_provider(failure):
    database, client, reads, _, ledger, provider = _stack([failure, failure])
    with database, client:
        response = _provider_request(client)
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "code": "arena_store_unavailable"}
    assert response.headers["retry-after"] == "1"
    assert len(reads) == 2
    assert ledger.log == [] and provider.sent == []
    assert "private credential diagnostic" not in response.text


@pytest.mark.parametrize("reply", ["missing", "http_401", "pool_timeout"])
def test_nontransient_or_missing_credential_does_not_retry_or_dispatch(reply):
    database, client, reads, _, ledger, provider = _stack([reply])
    with database, client:
        response = _provider_request(client)
    assert len(reads) == 1
    assert ledger.log == [] and provider.sent == []
    if reply == "missing":
        assert response.status_code == 200
        assert response.json()["status"] == 402
        assert response.json()["call"]["error_code"] == "miner_credentials_unavailable"
    else:
        assert response.status_code != 200
    assert "private credential diagnostic" not in response.text


def test_invalid_decrypted_credential_does_not_retry_or_dispatch():
    database, client, reads, _, ledger, provider = _stack(
        ["available"], invalid_secret=True,
    )
    with database, client:
        response = _provider_request(client)
    assert response.status_code == 200
    assert response.json()["status"] == 402
    assert response.json()["call"]["error_code"] == "miner_credentials_unavailable"
    assert len(reads) == 1
    assert ledger.log == [] and provider.sent == []


def test_mutating_rpc_transport_timeout_does_not_retry():
    calls = []

    def fail(request):
        calls.append(request)
        raise httpx.ReadTimeout("private reservation diagnostic", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as database:
        transport = PostgrestTransport(
            "https://project.example", service_key="sb_secret_test", http_client=database,
        )
        with pytest.raises(ArenaStoreError):
            transport.rpc("lab_arena_reserve_call", {})
    assert len(calls) == 1
