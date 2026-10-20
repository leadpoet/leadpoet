"""Retry transient read responses once without replaying mutating RPCs."""

import httpx
import pytest

from lab_arena.store import ArenaStoreError, ArenaStoreUnavailable, PostgrestTransport
from tests.lab_arena.test_source_disclosure import source_route


def client_and_calls(outcomes):
    calls = []

    def handle(request):
        calls.append(request)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome("synthetic transport failure", request=request)
        status, body = outcome
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, request=request)
        return httpx.Response(status, json=body, request=request)

    return httpx.Client(transport=httpx.MockTransport(handle)), calls


def transport(client):
    return PostgrestTransport(
        "https://db.example", service_key="sb_secret_test", http_client=client,
    )


def read(store):
    return store.select(
        "lab_arena_runs", filters={"round_id": "arena-2026-10-20"},
        columns="run_id,status", order="run_id", limit=7, offset=2,
    )


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_transient_http_read_retries_same_query_then_succeeds(status):
    expected = [{"run_id": "r1", "status": "leased"}]
    client, calls = client_and_calls([(status, {}), (200, expected)])
    with client:
        assert read(transport(client)) == expected
    assert len(calls) == 2
    assert [r.method for r in calls] == ["GET", "GET"]
    assert calls[0].url == calls[1].url
    assert calls[0].headers == calls[1].headers
    assert calls[0].extensions["timeout"] == calls[1].extensions["timeout"]


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_exhausted_http_read_preserves_typed_native_status(status):
    client, calls = client_and_calls([(status, {"code": "PGRST000"}), (status, {})])
    with client, pytest.raises(ArenaStoreUnavailable) as failure:
        read(transport(client))
    assert len(calls) == 2
    assert failure.value.store_error == {
        "operation": "select", "resource": "lab_arena_runs", "http_status": status,
    }


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 500])
def test_terminal_http_read_is_not_retried(status):
    client, calls = client_and_calls([(status, {}), (200, [])])
    with client, pytest.raises(ArenaStoreError) as failure:
        read(transport(client))
    assert not isinstance(failure.value, ArenaStoreUnavailable)
    assert len(calls) == 1


@pytest.mark.parametrize("body,error", [(b"not json", ValueError), ({}, ArenaStoreError)])
def test_successful_invalid_body_is_not_retried(body, error):
    client, calls = client_and_calls([(200, body), (200, [])])
    with client, pytest.raises(error):
        read(transport(client))
    assert len(calls) == 1


@pytest.mark.parametrize("recovers", [False, True])
def test_transient_non_json_body_is_retried_without_parsing(recovers):
    client, calls = client_and_calls([
        (502, b"<html>upstream unavailable</html>"),
        (200, []) if recovers else (502, b"not JSON"),
    ])
    with client:
        if recovers:
            assert read(transport(client)) == []
        else:
            with pytest.raises(ArenaStoreUnavailable) as failure:
                read(transport(client))
            assert failure.value.store_error["http_status"] == 502
    assert len(calls) == 2


def test_pool_timeout_still_fails_without_replay():
    client, calls = client_and_calls([httpx.PoolTimeout, (200, [])])
    with client, pytest.raises(ArenaStoreUnavailable):
        read(transport(client))
    assert len(calls) == 1


@pytest.mark.parametrize("outcomes,last_error", [
    ([httpx.ReadTimeout, (502, {})], {"http_status": 502}),
    ([(502, {}), httpx.ReadTimeout], {"transport_error": "ReadTimeout"}),
    ([(502, {}), (429, {})], {"http_status": 429}),
])
def test_http_and_transport_failures_share_one_total_retry(outcomes, last_error):
    client, calls = client_and_calls([*outcomes, (200, [])])
    with client, pytest.raises(ArenaStoreUnavailable) as failure:
        read(transport(client))
    assert len(calls) == 2
    assert failure.value.store_error == {
        "operation": "select", "resource": "lab_arena_runs", **last_error,
    }


def test_existing_transport_retry_still_succeeds():
    client, calls = client_and_calls([httpx.ReadTimeout, (200, [])])
    with client:
        assert read(transport(client)) == []
    assert len(calls) == 2


@pytest.mark.parametrize("status", [429, 502, 503, 504])
@pytest.mark.parametrize("function", ["lab_arena_claim_assignment", "lab_arena_reserve_call"])
def test_http_failure_does_not_replay_mutating_rpc(function, status):
    client, calls = client_and_calls([(status, {}), (200, {"status": "leased"})])
    with client, pytest.raises(ArenaStoreError):
        transport(client).rpc(function, {})
    assert len(calls) == 1
    assert calls[0].method == "POST"


@pytest.mark.parametrize("status", [429, 502, 503, 504])
@pytest.mark.parametrize("table", ["lab_arena_submissions", "lab_arena_rounds"])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("recovers", [False, True])
def test_actual_public_source_route_retries_read_and_preserves_disclosure(
    source_route, status, table, published, recovers,
):
    failures = 0

    def fault(request):
        nonlocal failures
        if request.url.path.endswith("/" + table) and (not recovers or not failures):
            failures += 1
            return httpx.Response(status, content=b"upstream temporarily unavailable")

    api, objects, reads = source_route(fault, published=published)
    response = api.get("/arena/v1/submissions/sub-preview/code")
    assert response.status_code == ((200 if published else 403) if recovers else 503)
    assert failures == (1 if recovers else 2)
    assert len(reads) == (3 if recovers or table == "lab_arena_rounds" else 2)
    assert all(request.method == "GET" for request in reads)
    assert objects.reads == (1 if recovers and published else 0)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    if not recovers:
        assert response.json() == {"status": "unavailable", "code": "arena_store_unavailable"}
        assert response.headers["retry-after"] == "1"
    assert "upstream temporarily unavailable" not in response.text
