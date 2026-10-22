"""Reject malformed round IDs before the real PostgREST filter boundary."""

from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport


@contextmanager
def round_api(rows=(), *, pinned=None, database_status=200):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/rest/v1/lab_arena_rounds"
        return httpx.Response(database_status, json=list(rows) if database_status == 200 else {"code": "XX000"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as database:
        service = object.__new__(ArenaService)
        service._store = ArenaStore(PostgrestTransport(
            "https://db.example", service_key="sb_secret_test", http_client=database,
        ))
        service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71, pinned_round_id=pinned)
        with TestClient(create_app(service), raise_server_exceptions=False) as client:
            yield service, client, requests


def round_row(round_id, **configuration):
    return {"round_id": round_id, "status": "open", "configuration_doc": {
        "mode": "live", "network_name": "finney", "netuid": 71, **configuration,
    }}


@pytest.mark.parametrize("round_id", [
    "arena-2026-10-22'", 'a"b', "a(b)", "a,b", "a.b", "a" * 201,
    "a\x00b", "a\tb", "a\rb", "a\x1fb", "a\x7fb",
])
def test_malformed_round_path_returns_400_without_database_request(round_id):
    with round_api() as (_service, client, requests):
        response = client.get("/arena/v1/rounds/" + quote(round_id, safe=""))
        assert response.status_code == 400
        assert response.json() == {"status": "rejected", "code": "round_id_invalid"}
        assert requests == []


@pytest.mark.parametrize("round_id", [None, 1, False, [], {}, b"arena-test", "", *map(chr, range(32)), "\x7f"])
def test_nonstring_empty_and_control_round_ids_fail_at_shared_boundary(round_id):
    with round_api() as (service, _client, requests):
        for read in (service._round, service._public_results_round,
                     service._public_submissions_round, service._hot_round):
            with pytest.raises(ServiceError, match="round_id_invalid") as error:
                read(round_id)
            assert error.value.status == 400
        assert requests == []


@pytest.mark.parametrize("round_id", [
    "arena-2026-10-22", "arena-validation-run1", "testnet-evaluation-legacy:abc_1", "a" * 200,
])
@pytest.mark.parametrize("exists", [True, False])
def test_safe_canonical_and_noncanonical_ids_preserve_lookup(round_id, exists):
    with round_api([round_row(round_id)] if exists else []) as (_service, client, requests):
        response = client.get("/arena/v1/rounds/" + quote(round_id, safe=""))
        assert response.status_code == (200 if exists else 404)
        assert len(requests) == 1
        assert requests[0].url.params["round_id"] == "eq." + round_id
        if exists:
            assert response.json()["round_id"] == round_id
        else:
            assert response.json()["code"] == "round_missing"


def test_safe_pinned_round_keeps_ownership_guard():
    pinned = "testnet-evaluation-legacy:abc_1"
    with round_api([round_row(pinned)], pinned=pinned) as (_service, client, requests):
        rejected = client.get("/arena/v1/rounds/arena-validation-other")
        assert rejected.status_code == 409 and rejected.json()["code"] == "round_scope_mismatch"
        assert requests == []
        assert client.get("/arena/v1/rounds/" + pinned).status_code == 200
        assert len(requests) == 1


@pytest.mark.parametrize("configuration,code", [
    ({"mode": "shadow"}, "round_mode_mismatch"),
    ({"network_name": "test", "netuid": 401}, "round_network_mismatch"),
])
def test_safe_id_does_not_bypass_mode_or_network_guard(configuration, code):
    with round_api([round_row("arena-validation-run1", **configuration)]) as (_service, client, requests):
        response = client.get("/arena/v1/rounds/arena-validation-run1")
        assert response.status_code == 409 and response.json()["code"] == code
        assert len(requests) == 1


def test_real_database_failure_is_not_reclassified_as_invalid_round_id():
    with round_api(database_status=500) as (service, client, requests):
        with pytest.raises(ArenaStoreError):
            service._round("arena-validation-run1")
        assert client.get("/arena/v1/rounds/arena-validation-run1").status_code == 500
        assert len(requests) == 2
