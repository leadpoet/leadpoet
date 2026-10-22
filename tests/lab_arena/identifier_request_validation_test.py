"""Unchecked request IDs must not become PostgREST filter exceptions."""

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import trajectory
from lab_arena.api import create_app
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PostgrestTransport, hash_lease_token


ROUND_ID = "arena-validation-legacy:abc_1"
SUBMISSION_ID = "submission-legacy:abc_1"
RUN_ID = "run-legacy:abc_1"
RUN_ROUTES = [
    ("provider", "POST"), ("signal-judgment", "POST"), ("trajectory", "POST"),
    ("training-input", "POST"), ("training-observations", "HEAD"),
    ("training-observations", "POST"), ("source", "GET"), ("image-access", "GET"),
]


@contextmanager
def lookup_api(*, existing=False, pinned=None, mode="live", database_status=200):
    requests = []
    row = {"round_id": ROUND_ID, "status": "open", "configuration_doc": {
        "mode": mode, "network_name": "finney", "netuid": 71,
    }}
    submission = {"submission_id": SUBMISSION_ID, "round_id": ROUND_ID,
                  "status": "frozen", "code_review_status": "passed"}
    run = {"run_id": RUN_ID, "round_id": ROUND_ID, "kind": "execute", "status": "leased",
           "lease_token_hash": hash_lease_token("b" * 64)}

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        if database_status != 200:
            return httpx.Response(database_status, json={"code": "XX000"})
        table = request.url.path.removeprefix("/rest/v1/")
        record, key = {
            "lab_arena_rounds": (row, "round_id"),
            "lab_arena_submissions": (submission, "submission_id"),
            "lab_arena_runs": (run, "run_id"),
        }[table]
        return httpx.Response(200, json=[record] if existing and request.url.params[key] == "eq." + record[key] else [])

    with httpx.Client(transport=httpx.MockTransport(handle)) as database:
        service = object.__new__(ArenaService)
        service._store = ArenaStore(PostgrestTransport(
            "https://db.example", service_key="sb_secret_test", http_client=database,
        ))
        service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71, pinned_round_id=pinned)
        service._objects = None
        service._clock = lambda: datetime(2026, 10, 22, tzinfo=timezone.utc)
        with TestClient(create_app(service), raise_server_exceptions=False) as client:
            yield client, requests


def run_request(client, route, method, run_id, *, lease="a" * 64):
    document = {}
    if route == "provider":
        document = {"operation_id": "openrouter.chat", "parameters": {}, "timeout_ms": 1000, "action_sequence": 0}
    elif route == "trajectory":
        document = {"events": [trajectory.event("runtime.started", {"status": "starting"})]}
    return client.request(method, "/arena/v1/runs/" + quote(run_id, safe="") + "/" + route,
        headers={"x-lab-arena-lease": lease} if lease is not None else {},
        **({"json": document} if method == "POST" else {}))


@pytest.mark.parametrize("suffix", ["", "/code"])
@pytest.mark.parametrize("submission_id", ["bad'", "x" * 5000], ids=["quote", "oversize"])
def test_unchecked_submission_ids_return_400_before_filter(suffix, submission_id):
    with lookup_api() as (client, requests):
        response = client.get("/arena/v1/submissions/" + quote(submission_id, safe="") + suffix)
        assert response.status_code == 400
        assert response.json() == {"status": "rejected", "code": "submission_id_invalid"}
        assert requests == []


@pytest.mark.parametrize("route,method", RUN_ROUTES)
@pytest.mark.parametrize("run_id", ["bad'", "x" * 5000], ids=["quote", "oversize"])
def test_unchecked_run_ids_return_400_before_filter(route, method, run_id):
    with lookup_api() as (client, requests):
        response = run_request(client, route, method, run_id)
        assert response.status_code == 400
        if method != "HEAD":
            assert response.json() == {"status": "rejected", "code": "run_id_invalid"}
        assert requests == []


@pytest.mark.parametrize("suffix", ["", "/code"])
def test_safe_legacy_missing_submission_preserves_404(suffix):
    with lookup_api() as (client, requests):
        response = client.get("/arena/v1/submissions/" + SUBMISSION_ID + suffix)
        assert response.status_code == 404 and response.json()["code"] == "submission_missing"
        assert len(requests) == 1
        assert requests[0].url.params["submission_id"] == "eq." + SUBMISSION_ID


def test_safe_existing_submission_status_preserves_lookup():
    with lookup_api(existing=True) as (client, requests):
        response = client.get("/arena/v1/submissions/" + SUBMISSION_ID)
        assert response.status_code == 200
        assert response.json()["submission_id"] == SUBMISSION_ID
        assert len(requests) == 2


def test_safe_existing_submission_code_preserves_disclosure_guard():
    with lookup_api(existing=True) as (client, requests):
        response = client.get("/arena/v1/submissions/" + SUBMISSION_ID + "/code")
        assert response.status_code == 403 and response.json()["code"] == "source_not_public"
        assert len(requests) == 2


@pytest.mark.parametrize("route,method", RUN_ROUTES)
def test_safe_legacy_missing_run_preserves_404(route, method):
    with lookup_api() as (client, requests):
        assert run_request(client, route, method, RUN_ID).status_code == 404
        assert len(requests) == 1
        assert requests[0].url.params["run_id"] == "eq." + RUN_ID


@pytest.mark.parametrize("route,method", [("provider", "POST"), ("source", "GET")])
def test_lease_header_shape_still_precedes_lookup(route, method):
    with lookup_api() as (client, requests):
        assert run_request(client, route, method, "bad'", lease=None).status_code == 401
        assert run_request(client, route, method, "bad'", lease="invalid").status_code == 401
        assert requests == []


def test_safe_existing_run_still_requires_its_actual_lease():
    with lookup_api(existing=True) as (client, requests):
        response = run_request(client, "source", "GET", RUN_ID)
        assert response.status_code == 401 and response.json()["code"] == "lease_invalid"
        assert len(requests) == 1


@pytest.mark.parametrize("options,code,reads", [
    ({"pinned": "other-safe-round"}, "round_scope_mismatch", 1),
    ({"mode": "shadow"}, "round_mode_mismatch", 2),
])
def test_existing_submission_still_requires_its_round_scope(options, code, reads):
    with lookup_api(existing=True, **options) as (client, requests):
        response = client.get("/arena/v1/submissions/" + SUBMISSION_ID)
        assert response.status_code == 409 and response.json()["code"] == code
        assert len(requests) == reads


@pytest.mark.parametrize("kind", ["submission", "run"])
def test_valid_id_database_failure_still_returns_500(kind):
    with lookup_api(database_status=500) as (client, requests):
        response = (client.get("/arena/v1/submissions/" + SUBMISSION_ID) if kind == "submission"
                    else run_request(client, "source", "GET", RUN_ID))
        assert response.status_code == 500
        assert len(requests) == 1


def test_quota_keeps_opaque_unavailable_response_for_malformed_run():
    with lookup_api() as (client, requests):
        response = run_request(client, "quota", "GET", "bad'")
        assert response.status_code == 503 and response.json()["code"] == "quota_unavailable"
        assert requests == []
