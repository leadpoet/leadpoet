"""A lost final upload ACK recovers only an exact persisted, owned snapshot."""

import copy
import gzip
import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, runner, training_observations as obs
from lab_arena.service import ServiceError
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for
from tests.lab_arena.training_observations_test import observation
from lab_arena.api import create_app


ROUTE = "/arena/v1/runs/r1/training-observations"
WRONG_DIGEST = "sha256:" + "0" * 64


def _headers(digest, token=TOKEN):
    return {"x-lab-arena-lease": token, "If-Match": '"' + digest + '"'}


def _state(*, incomplete=False, secret=False):
    state = runner.RunState(lease={"run_id": "r1"}, lease_token=TOKEN)
    if incomplete:
        state.worker_observations.failures.add("capture_incomplete")
    if secret:
        # Exercise the gateway's cleaning contract even for a caller that did
        # not pre-clean its buffer. The actual POST bytes must stay unchanged.
        row = observation()
        row["request"]["admitted_parameters"]["messages"] = [{"role": "user", "content": TOKEN}]
        state.worker_observations.observations.append(row)
        state.worker_observations.attempted = 1
    return state


def _record(api, state):
    return runner._record_training_observations(SimpleNamespace(api=api), state.lease, TOKEN, state)


def _lost_ack_transport(gateway, requests, *, post="lost", head="verified"):
    def handle(request):
        requests.append(request)
        if request.method == "POST":
            response = gateway.post(ROUTE, content=request.content, headers=dict(request.headers))
            assert response.status_code == 200
            if post == "lost":
                raise httpx.ReadTimeout("synthetic lost response", request=request)
            if post == "nonjson":
                return httpx.Response(200, content=b"truncated ACK")
            document = response.json()
            if post == "wrong_digest":
                document["payload_sha256"] = WRONG_DIGEST
            return httpx.Response(200, json=document)
        assert request.method == "HEAD"
        assert request.content == b""
        if head == "timeout":
            raise httpx.ReadTimeout("synthetic verification timeout", request=request)
        if head == "legacy":
            return httpx.Response(204)
        if head == "wrong_digest":
            return httpx.Response(204, headers={"ETag": '"' + WRONG_DIGEST + '"'})
        if head == "wrong_status":
            return httpx.Response(200, headers={"ETag": request.headers["if-match"]})
        response = gateway.head(ROUTE, headers=dict(request.headers))
        return httpx.Response(response.status_code, headers=dict(response.headers))
    return httpx.MockTransport(handle)


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("post", ["lost", "nonjson", "wrong_digest"])
@pytest.mark.parametrize("incomplete", [False, True])
def test_lost_acks_recover_exact_persisted_digest_only_after_both_posts(tmp_path, post, incomplete, kind):
    service, objects = service_for(tmp_path, kind=kind)
    gateway = TestClient(create_app(service))
    requests = []
    state = _state(incomplete=incomplete)
    expected = contracts.document_hash(state.worker_observations.document())
    with httpx.Client(transport=_lost_ack_transport(gateway, requests, post=post)) as client:
        api = runner.HttpArenaApiClient("http://localhost", client=client)
        summary = _record(api, state)
    assert [request.method for request in requests] == ["POST", "POST", "HEAD"]
    assert requests[0].content == requests[1].content
    assert requests[2].headers["if-match"] == '"' + expected + '"'
    assert requests[2].extensions["timeout"]["read"] == 60.0
    assert summary == {"status": "stored", "observation_count": 0, "local_refusal_count": 0,
                       "complete": not incomplete, "payload_sha256": expected}
    assert obs.read_snapshot(objects, "r1", kind, 1, CLAIM_HASH)["payload_sha256"] == expected
    assert state.calls == [] and state.action_sequence == 0


@pytest.mark.parametrize("post", ["lost", "valid"])
def test_expected_digest_uses_the_gateway_privacy_cleaned_payload(tmp_path, post):
    service, objects = service_for(tmp_path)
    gateway = TestClient(create_app(service))
    requests = []
    state = _state(secret=True)
    raw = copy.deepcopy(state.worker_observations.document())
    expected = contracts.document_hash(obs._validate_payload(raw, secrets=(TOKEN,)))
    assert expected != contracts.document_hash(raw)
    with httpx.Client(transport=_lost_ack_transport(gateway, requests, post=post)) as client:
        summary = _record(runner.HttpArenaApiClient("http://localhost", client=client), state)
    assert summary["status"] == "stored" and summary["payload_sha256"] == expected
    assert json.loads(requests[0].content) == raw
    assert [request.method for request in requests] == (["POST"] if post == "valid" else ["POST", "POST", "HEAD"])
    saved = obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["payload_sha256"] == expected and saved["payload"]["observations"][0]["privacy"]["redacted"]


@pytest.mark.parametrize("post", ["lost", "wrong_digest"])
@pytest.mark.parametrize("head", ["legacy", "wrong_digest", "wrong_status", "timeout"])
def test_missing_or_wrong_head_proof_never_claims_stored(tmp_path, post, head):
    service, _ = service_for(tmp_path)
    requests = []
    with httpx.Client(transport=_lost_ack_transport(TestClient(create_app(service)), requests, post=post, head=head)) as client:
        summary = _record(runner.HttpArenaApiClient("http://localhost", client=client), _state())
    assert summary["status"] == "unavailable" and "payload_sha256" not in summary
    assert [request.method for request in requests] == ["POST", "POST", "HEAD"]


def test_valid_post_ack_needs_no_verification_request(tmp_path):
    service, _ = service_for(tmp_path)
    requests = []
    with httpx.Client(transport=_lost_ack_transport(TestClient(create_app(service)), requests, post="valid")) as client:
        summary = _record(runner.HttpArenaApiClient("http://localhost", client=client), _state())
    assert summary["status"] == "stored"
    assert [request.method for request in requests] == ["POST"]


def test_legacy_client_without_verifier_keeps_existing_two_post_failure_behavior():
    calls = []
    def fail(*args):
        calls.append(args)
        raise TimeoutError("synthetic upload failure")
    summary = _record(SimpleNamespace(training_observations=fail), _state())
    assert len(calls) == 2 and summary["status"] == "unavailable"


def test_plain_head_stays_auth_only_without_storage_reads(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    def no_read(*_):
        raise AssertionError("plain preflight must not read storage")
    monkeypatch.setattr(objects, "get_bounded", no_read)
    response = TestClient(create_app(service)).head(ROUTE, headers={"x-lab-arena-lease": TOKEN})
    assert response.status_code == 204 and response.content == b""
    assert response.headers["cache-control"] == "no-store" and "etag" not in response.headers


@pytest.mark.parametrize("value", ["*", WRONG_DIGEST, 'W/"' + WRONG_DIGEST + '"',
                                    '"' + WRONG_DIGEST + '","' + WRONG_DIGEST + '"', '"bad"'])
def test_conditional_head_refuses_nonexact_digest_syntax_before_storage(tmp_path, monkeypatch, value):
    service, objects = service_for(tmp_path)
    monkeypatch.setattr(objects, "get_bounded", lambda *_: pytest.fail("invalid digest must not read storage"))
    response = TestClient(create_app(service)).head(ROUTE, headers={"x-lab-arena-lease": TOKEN, "If-Match": value})
    assert response.status_code == 400


@pytest.mark.parametrize("changes,token,status", [
    ({}, "b" * 64, 401),
    ({"status": "accepted"}, TOKEN, 409),
    ({"lease_expires_at": "2020-01-01T00:00:00Z"}, TOKEN, 409),
    ({"lease_generation": 0}, TOKEN, 401),
    ({"claim_request_hash": "invalid"}, TOKEN, 401),
])
def test_conditional_verification_keeps_lease_guards_before_storage(tmp_path, monkeypatch, changes, token, status):
    service, objects = service_for(tmp_path, **changes)
    monkeypatch.setattr(objects, "get_bounded", lambda *_: pytest.fail("unauthorized read"))
    response = TestClient(create_app(service)).head(ROUTE, headers=_headers(WRONG_DIGEST, token))
    assert response.status_code == status


def test_conditional_verification_keeps_round_ownership_guard(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    service._require_round_ownership = lambda _: (_ for _ in ()).throw(ServiceError("round_not_owned", 403))
    monkeypatch.setattr(objects, "get_bounded", lambda *_: pytest.fail("wrong owner read"))
    assert TestClient(create_app(service)).head(ROUTE, headers=_headers(WRONG_DIGEST)).status_code == 403


@pytest.mark.parametrize("damage", ["run", "kind", "generation", "claim", "hash", "schema", "compressed", "different_body"])
def test_conditional_head_revalidates_full_object_identity_and_payload(tmp_path, monkeypatch, damage):
    service, objects = service_for(tmp_path)
    document = _state().worker_observations.document()
    receipt = service.handle_training_observations("r1", TOKEN, document)
    ref = receipt["ref"]
    snapshot = obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    if damage == "run":
        snapshot["run_id"] = "another-run"
    elif damage == "kind":
        snapshot["kind"] = "score"
    elif damage == "generation":
        snapshot["lease_generation"] = 2
    elif damage == "claim":
        snapshot["claim_request_hash"] = WRONG_DIGEST
    elif damage == "hash":
        snapshot["payload_sha256"] = WRONG_DIGEST
    elif damage == "schema":
        snapshot["schema_version"] = "invalid"
    elif damage == "different_body":
        snapshot["payload"]["complete"] = False
        snapshot["payload"]["failures"] = ["capture_incomplete"]
        snapshot["payload_sha256"] = contracts.document_hash(snapshot["payload"])
    raw = b"invalid gzip" if damage == "compressed" else gzip.compress(contracts.canonical_json(snapshot).encode(), mtime=0)
    reads = []
    def read(requested_ref, limit):
        reads.append((requested_ref, limit))
        return raw
    monkeypatch.setattr(objects, "get_bounded", read)
    response = TestClient(create_app(service)).head(ROUTE, headers=_headers(receipt["payload_sha256"]))
    assert response.status_code == (412 if damage == "different_body" else 503)
    assert reads == [(ref, obs.MAX_COMPRESSED_BYTES)]
    assert "etag" not in response.headers


def test_missing_final_object_is_not_proved_by_a_plain_preflight(tmp_path):
    service, _ = service_for(tmp_path)
    client = TestClient(create_app(service))
    assert client.head(ROUTE, headers={"x-lab-arena-lease": TOKEN}).status_code == 204
    assert client.head(ROUTE, headers=_headers(WRONG_DIGEST)).status_code == 503


@pytest.mark.parametrize("identity_change", [{"lease_generation": 2}, {"claim_request_hash": WRONG_DIGEST}])
def test_verification_uses_current_server_identity_not_the_stale_object(tmp_path, identity_change):
    service, _ = service_for(tmp_path)
    receipt = service.handle_training_observations("r1", TOKEN, _state().worker_observations.document())
    run = service._store.get_run("r1")
    run.update(identity_change)
    response = TestClient(create_app(service)).head(ROUTE, headers=_headers(receipt["payload_sha256"]))
    assert response.status_code == 503 and "etag" not in response.headers


def test_existing_finalization_grace_applies_without_extending_it(tmp_path):
    service, _ = service_for(tmp_path, status="failed", terminal_cause="lease_expired",
                             lease_expires_at=(NOW - timedelta(seconds=1)).isoformat())
    receipt = service.handle_training_observations("r1", TOKEN, _state().worker_observations.document())
    client = TestClient(create_app(service))
    assert client.head(ROUTE, headers=_headers(receipt["payload_sha256"])).status_code == 204
    service._clock = lambda: NOW + timedelta(seconds=180)
    assert client.head(ROUTE, headers=_headers(receipt["payload_sha256"])).status_code == 409


@pytest.mark.parametrize("status", [400, 409])
def test_native_post_error_status_cannot_masquerade_as_exact_success_ack(status):
    requests = []
    expected = contracts.document_hash(_state().worker_observations.document())
    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(status, json={"status": "stored", "payload_sha256": expected})
        return httpx.Response(204)  # Legacy preflight is no proof of persistence.
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        summary = _record(runner.HttpArenaApiClient("http://localhost", client=client), _state())
    assert summary["status"] == "unavailable"
    assert [request.method for request in requests] == ["POST", "POST", "HEAD"]


def test_conditional_head_keeps_unconfigured_storage_unavailable(tmp_path):
    service, _ = service_for(tmp_path)
    service._config.training_object_store = None
    client = TestClient(create_app(service))
    assert client.head(ROUTE, headers={"x-lab-arena-lease": TOKEN}).status_code == 204
    assert client.head(ROUTE, headers=_headers(WRONG_DIGEST)).status_code == 503


def test_conditional_head_never_reads_another_run(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    monkeypatch.setattr(objects, "get_bounded", lambda *_: pytest.fail("unknown run storage read"))
    response = TestClient(create_app(service)).head(ROUTE.replace("/r1/", "/r2/"), headers=_headers(WRONG_DIGEST))
    assert response.status_code == 404


@pytest.fixture
def public_capture_stack(tmp_path, monkeypatch):
    """Use real proxy/HTTPX/ASGI paths; replace only loopback network I/O."""
    from gateway.api import arena_proxy
    from fastapi import FastAPI
    from scripts.run_arena_proxy_edge import create_app as create_edge

    main, main_objects = service_for(tmp_path / "main")
    testnet, testnet_objects = service_for(tmp_path / "testnet")
    ports = {httpx.URL(arena_proxy._SIDECAR_URL).port: create_app(main),
             8793: create_app(testnet)}
    observed = []
    mode = SimpleNamespace(legacy=False)
    original_client = httpx.AsyncClient

    class LoopbackTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            observed.append(request)
            assert request.url.port in ports
            forwarded = request
            if mode.legacy and request.method == "HEAD" and "if-match" in request.headers:
                forwarded = httpx.Request("HEAD", request.url, headers={
                    key: value for key, value in request.headers.items() if key != "if-match"}, content=b"")
            return await httpx.ASGITransport(app=ports[request.url.port]).handle_async_request(forwarded)

    def loopback_client(*args, **kwargs):
        return original_client(*args, transport=LoopbackTransport(), **kwargs)

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy.httpx, "AsyncClient", loopback_client)

    def client(edge=False):
        if edge:
            return TestClient(create_edge())
        app = FastAPI()
        app.include_router(arena_proxy.router)
        app.include_router(arena_proxy.testnet_router)
        return TestClient(app)

    return SimpleNamespace(client=client, observed=observed, mode=mode,
                           main=main, main_objects=main_objects,
                           testnet=testnet, testnet_objects=testnet_objects)


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("edge", [False, True])
@pytest.mark.parametrize("head_proof", ["verified", "dropped_etag", "legacy_sidecar"])
def test_external_runner_recovers_through_proxy_sidecar_and_storage(
    public_capture_stack, prefix, edge, head_proof,
):
    from gateway.api import arena_proxy
    stack = public_capture_stack
    stack.mode.legacy = head_proof == "legacy_sidecar"
    public = stack.client(edge)
    external = []

    def send(request):
        external.append(request)
        response = public.request(request.method, prefix + "/v1/runs/r1/training-observations",
                                  content=request.content, headers=dict(request.headers))
        if request.method == "POST":
            assert response.status_code == 200
            raise httpx.ReadTimeout("ACK lost outside public gateway", request=request)
        headers = dict(response.headers)
        if head_proof == "dropped_etag":
            assert "etag" in headers  # The production proxy forwarded it.
            del headers["etag"]
        return httpx.Response(response.status_code, headers=headers)

    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        api = runner.HttpArenaApiClient("http://localhost", client=client)
        api.provider = lambda *_: pytest.fail("capture recovery must never replay a provider")
        summary = _record(api, _state())
    assert summary["status"] == ("stored" if head_proof == "verified" else "unavailable")
    assert [request.method for request in external] == ["POST", "POST", "HEAD"]
    assert [request.method for request in stack.observed] == ["HEAD", "POST", "HEAD", "POST", "HEAD"]
    assert all("if-match" not in request.headers for request in stack.observed[:-1])
    final_request = stack.observed[-1]
    assert final_request.headers["if-match"] == external[-1].headers["if-match"]
    assert final_request.content == b""
    expected_port = 8793 if prefix.startswith("/testnet") else httpx.URL(arena_proxy._SIDECAR_URL).port
    assert {request.url.port for request in stack.observed} == {expected_port}
    objects = stack.testnet_objects if prefix.startswith("/testnet") else stack.main_objects
    saved = obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["payload"] == _state().worker_observations.document()
    if head_proof == "verified":
        assert summary["payload_sha256"] == saved["payload_sha256"]


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
def test_external_plain_head_keeps_auth_only_preflight_and_private_header_filters(public_capture_stack, monkeypatch, prefix):
    stack = public_capture_stack
    objects = stack.testnet_objects if prefix.startswith("/testnet") else stack.main_objects
    monkeypatch.setattr(objects, "get_bounded", lambda *_: pytest.fail("plain HEAD must not read objects"))
    public = stack.client()
    route = prefix + "/v1/runs/r1/training-observations"
    assert public.head(route).status_code == 401
    response = public.head(route, headers={"x-lab-arena-lease": TOKEN, "authorization": "never-forward", "x-private": "never-forward"})
    assert response.status_code == 204 and response.content == b"" and "etag" not in response.headers
    assert "authorization" not in stack.observed[-1].headers and "x-private" not in stack.observed[-1].headers


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("path", ["/v1/current", "/v1/runs/r1/source", "/v1/runs/r1/training-input",
                                  "/v1/runs/r1/training-observations/extra", "/v1/submissions/s1/code",
                                  "/v1/rounds/r1/benchmark", "/v1/rounds/r1/results/s1"])
def test_other_external_head_routes_remain_405_without_sidecar_access(public_capture_stack, prefix, path):
    stack = public_capture_stack
    response = stack.client().head(prefix + path, headers=_headers(WRONG_DIGEST))
    assert response.status_code == 405 and stack.observed == []


def test_external_verification_cannot_read_mainnet_from_testnet(public_capture_stack):
    stack = public_capture_stack
    receipt = stack.main.handle_training_observations("r1", TOKEN, _state().worker_observations.document())
    public = stack.client()
    assert public.head(ROUTE, headers=_headers(receipt["payload_sha256"])).status_code == 204
    assert public.head("/testnet" + ROUTE, headers=_headers(receipt["payload_sha256"])).status_code == 503
    assert [request.url.port for request in stack.observed][-1] == 8793


@pytest.mark.parametrize("prefix,environment", [("/arena", "LAB_ARENA_MODE"),
                                                ("/testnet/arena", "LAB_ARENA_TESTNET_ENABLED")])
def test_exact_external_head_keeps_network_enablement_guards(public_capture_stack, monkeypatch, prefix, environment):
    stack = public_capture_stack
    monkeypatch.setenv(environment, "off")
    assert stack.client().head(prefix + "/v1/runs/r1/training-observations", headers=_headers(WRONG_DIGEST)).status_code == 404
    assert stack.observed == []


def test_testnet_external_head_keeps_residual_percent_guard(public_capture_stack):
    stack = public_capture_stack
    response = stack.client().head("/testnet/arena/v1/runs/r%2525/training-observations", headers=_headers(WRONG_DIGEST))
    assert response.status_code == 404 and stack.observed == []


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
def test_post_upload_preflight_never_receives_injected_conditional_headers(public_capture_stack, prefix):
    stack = public_capture_stack
    response = stack.client().post(prefix + "/v1/runs/r1/training-observations",
                                   json=_state().worker_observations.document(), headers=_headers(WRONG_DIGEST))
    assert response.status_code == 200 and "etag" not in response.headers
    assert [request.method for request in stack.observed] == ["HEAD", "POST"]
    assert all("if-match" not in request.headers for request in stack.observed)


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("token", [None, "b" * 64])
def test_external_conditional_head_keeps_auth_before_storage(public_capture_stack, monkeypatch, prefix, token):
    stack = public_capture_stack
    objects = stack.testnet_objects if prefix.startswith("/testnet") else stack.main_objects
    monkeypatch.setattr(objects, "get_bounded", lambda *_: pytest.fail("unauthorized conditional storage read"))
    headers = _headers(WRONG_DIGEST)
    if token is None:
        del headers["x-lab-arena-lease"]
    else:
        headers["x-lab-arena-lease"] = token
    response = stack.client().head(prefix + "/v1/runs/r1/training-observations", headers=headers)
    assert response.status_code == 401 and "etag" not in response.headers


@pytest.mark.parametrize("method,path", [("GET", "/arena/v1/current"),
                                          ("POST", "/arena/v1/submissions")])
def test_conditional_headers_and_etags_stay_filtered_on_other_methods(monkeypatch, method, path):
    from gateway.api import arena_proxy
    from fastapi import FastAPI
    forwarded = []
    async def respond(_method, _path, *, query, body, headers):
        forwarded.append(headers)
        return httpx.Response(200, json={}, headers={"ETag": '"' + WRONG_DIGEST + '"', "x-private": "hidden"})
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", respond)
    app = FastAPI()
    app.include_router(arena_proxy.router)
    response = TestClient(app).request(method, path, headers=_headers(WRONG_DIGEST))
    assert response.status_code == 200 and "etag" not in response.headers and "x-private" not in response.headers
    assert "if-match" not in forwarded[0]
