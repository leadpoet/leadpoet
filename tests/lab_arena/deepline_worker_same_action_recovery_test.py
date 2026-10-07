"""Bounded recovery of one dispatched Deepline action at the worker boundary."""

from __future__ import annotations

import base64
import copy
import json

import httpx
import pytest

from lab_arena import contracts, runner, scoring_provider_compat, shim
from tests.lab_arena.test_lab_arena_runner import lease, temporary_hold_document


IDENTITY = "sha256:" + "a" * 64
PARAMETERS = {"tool": "exa_search", "payload": {"query": "x"}}


def document(status=502, *, operation="deepline.execute", recovered=False, pending=False):
    error = "provider_unavailable" if status == 502 else "call_uncertain"
    body = ({"results": [{"url": "https://example.com"}],
             "billing": {"pricing_status": "pending", "estimated_cost_usd": 0.002}}
            if status == 200 else {"error": {"code": error}})
    call = {
        "operation_id": operation, "provider": "deepline", "action_sequence": 0,
        "call_identity": IDENTITY, "outcome": "uncertain", "error_code": error,
    }
    if status == 200:
        call.update(error_code=None)
        if not pending:
            call.update(outcome="settled", actual_microusd=2000)
    if recovered:
        call["transport_recovery"] = "deepline_execution_lookup"
    return {"status": status, "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(contracts.canonical_json(body).encode()).decode(),
            "call": call}


class Api:
    def __init__(self, *documents):
        self.documents = list(documents)
        self.frames = []

    def provider(self, _run_id, _lease_token, frame):
        self.frames.append(copy.deepcopy(frame))
        result = self.documents.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


def worker(tmp_path, api, **kwargs):
    return runner.WorkerSocketServer(tmp_path / "worker.sock", api,
        runner.RunState(lease=lease("same-action"), lease_token="token"), **kwargs)


@pytest.mark.parametrize("status", [502, 409])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("lookup_marker", [False, True])
@pytest.mark.parametrize("operation", ["deepline.execute", "exa.search", "exa.contents"])
def test_recovery_reuses_action_and_only_records_final_result(
    tmp_path, status, pending, lookup_marker, operation
):
    api = Api(document(status, operation=operation),
              document(200, operation=operation, recovered=lookup_marker, pending=pending))
    host = worker(tmp_path, api)
    error, result = host._dispatch_once(operation, PARAMETERS, 30_000)

    assert error is None and result["status"] == 200
    assert [frame["action_sequence"] for frame in api.frames] == [0, 0]
    assert api.frames[0]["operation_id"] == api.frames[1]["operation_id"]
    assert api.frames[0]["parameters"] == api.frames[1]["parameters"]
    assert host._state.action_sequence == 1
    assert host._state.calls == [result["call"]]
    assert host._state.refusals == 0
    if operation == "deepline.execute":
        assert result["headers"].get(runner.SETTLED_MICROUSD_HEADER) == (
            None if pending else "2000"
        )


@pytest.mark.parametrize("change", [
    {"provider": "openrouter"}, {"operation_id": "exa.contents"},
    {"action_sequence": 1}, {"action_sequence": True},
    {"outcome": "not_dispatched"}, {"outcome": "settled"},
    {"call_identity": "bad"}, {"error_code": "broker_unavailable"},
])
def test_only_exact_dispatched_deepline_uncertainty_is_retried(tmp_path, change):
    initial = document()
    initial["call"].update(change)
    api = Api(initial)
    host = worker(tmp_path, api)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000)[0] is None
    assert len(api.frames) == 1


@pytest.mark.parametrize("amount", [0, 3000])
def test_confirmed_settled_lost_response_recovers_same_action(tmp_path, amount):
    first = document(operation="scrapingdog.scrape")
    first["call"].update(
        outcome="settled", actual_microusd=amount,
        transport_error_class="ReadTimeout", provider_status=502,
        deepline_response_missing=True,
    )
    recovered = document(200, operation="scrapingdog.scrape", recovered=True)
    recovered["call"]["actual_microusd"] = amount
    api = Api(first, recovered)
    host = worker(tmp_path, api)

    error, result = host._dispatch_once(
        "scrapingdog.scrape", {"url": "https://example.com"}, 60_000
    )

    assert error is None and result["status"] == 200
    assert result["body_b64"] == recovered["body_b64"]
    assert result["call"] == recovered["call"]
    assert [frame["action_sequence"] for frame in api.frames] == [0, 0]
    assert api.frames[0] == api.frames[1]
    assert host._state.calls == [recovered["call"]]


@pytest.mark.parametrize("amount", [0, 3000])
@pytest.mark.parametrize("change", ["amount", "uncertain"])
def test_recovery_cannot_change_confirmed_charge(tmp_path, amount, change):
    first = document()
    first["call"].update(
        outcome="settled", actual_microusd=amount,
        transport_error_class="ReadTimeout", provider_status=502,
        deepline_response_missing=True,
    )
    recovered = document(200)
    recovered["call"]["actual_microusd"] = amount + 1 if change == "amount" else amount
    if change == "uncertain":
        recovered["call"]["outcome"] = "uncertain"
    api = Api(first, recovered)
    host = worker(tmp_path, api)

    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000)[1]["status"] == 502
    assert len(api.frames) == 2
    assert host._state.calls == [first["call"]]


@pytest.mark.parametrize("change", [
    {"deepline_response_missing": None},
    {"transport_error_class": None},
    {"transport_recovery": "deepline_execution_lookup"},
    {"actual_microusd": None},
    {"actual_microusd": -1},
    {"actual_microusd": True},
    {"actual_microusd": False},
    {"actual_microusd": 0.0},
    {"actual_microusd": "0"},
    {"provider_status": 403},
    {"error_code": "miner_credentials_unavailable"},
])
def test_settled_provider_failure_without_confirmed_timeout_proof_is_terminal(tmp_path, change):
    first = document()
    first["call"].update(
        outcome="settled", actual_microusd=3000,
        transport_error_class="ReadTimeout", provider_status=502,
        deepline_response_missing=True,
    )
    first["call"].update(change)
    api = Api(first)
    host = worker(tmp_path, api)

    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000)[1]["status"] == 502
    assert len(api.frames) == 1


@pytest.mark.parametrize("second", [
    document(409), runner.RunnerError("unknown transport outcome"),
    {**document(200, recovered=True), "call": {**document(200, recovered=True)["call"],
        "call_identity": "sha256:" + "b" * 64}},
    {**document(200), "body_b64": "invalid/base64!"},
    {**document(200), "body_b64": None},
])
def test_failed_or_unbound_recovery_keeps_initial_failure_and_stops(tmp_path, second):
    initial = document()
    api = Api(initial, second)
    host = worker(tmp_path, api)
    error, result = host._dispatch_once("deepline.execute", PARAMETERS, 30_000)
    assert error is None and result["status"] == 502
    assert host._state.calls == [initial["call"]]
    assert len(api.frames) == 2 and host._state.action_sequence == 1


def test_unknown_gateway_transport_outcome_is_never_retried(tmp_path):
    api = Api(runner.RunnerError("unknown transport outcome"))
    host = worker(tmp_path, api)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000) == (
        "worker_unavailable", None
    )
    assert len(api.frames) == 1 and host._state.calls[0]["outcome"] == "unknown"


def test_successful_pending_reply_is_not_retried(tmp_path):
    api = Api(document(200, pending=True))
    host = worker(tmp_path, api)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000)[1]["status"] == 200
    assert len(api.frames) == 1


def test_cancellation_preserves_initial_result_without_recovery(tmp_path):
    cancelled = [False]

    class CancellingApi(Api):
        def provider(self, *args):
            result = super().provider(*args)
            cancelled[0] = True
            return result

    api = CancellingApi(document())
    host = worker(tmp_path, api)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000,
        cancel_requested=lambda: cancelled[0])[1]["status"] == 502
    assert len(api.frames) == 1


@pytest.mark.parametrize("elapsed,expected_calls,expected_status", [
    (240.0, 2, 200), (305.0, 1, 502), (304.5, 2, 502),
])
def test_full_provider_timeout_uses_only_remaining_http_api_grace(
    tmp_path, elapsed, expected_calls, expected_status
):
    clock = [0.0]

    class Client:
        def __init__(self):
            self.requests = []

        def post(self, url, *, content, headers, timeout):
            self.requests.append((json.loads(content), timeout.read))
            if len(self.requests) == 1:
                clock[0] += elapsed
                result = document()
            else:
                clock[0] += 5.0
                result = document(200, recovered=True)
            # The API HTTP status stays 200; broker status is inside the envelope.
            return httpx.Response(200, json=result)

    client = Client()
    api = runner.HttpArenaApiClient("https://gateway.example.com", client=client)
    host = worker(tmp_path, api, monotonic=lambda: clock[0])
    error, result = host._dispatch_once("deepline.execute", PARAMETERS, 240_000)
    assert error is None and len(client.requests) == expected_calls
    assert result["status"] == expected_status
    assert client.requests[0][1] == 305.0
    if expected_calls == 2:
        initial_frame, _ = client.requests[0]
        recovery_frame, timeout = client.requests[1]
        remaining = 305.0 - elapsed
        assert recovery_frame == {**initial_frame, "timeout_ms": int(remaining * 1000)}
        assert timeout == remaining
        assert host._state.calls == [result["call"]]
        if expected_status == 200:
            assert clock[0] <= 305.0
    else:
        assert result["status"] == 502


def test_temporary_hold_time_is_included_in_recovery_deadline(monkeypatch, tmp_path):
    clock = [0.0]

    class SlowApi(Api):
        def provider(self, *args):
            clock[0] += 50.0
            return super().provider(*args)

    api = SlowApi(temporary_hold_document(), document())
    host = worker(tmp_path, api, monotonic=lambda: clock[0])
    monkeypatch.setattr(runner, "TEMPORARY_HOLD_RETRY_SECONDS", 0.0)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000)[1]["status"] == 502
    assert len(api.frames) == 2 and host._state.action_sequence == 1


@pytest.mark.parametrize("operation,parameters,adapter", [
    ("scrapingdog.scrape", {"url": "https://example.com"}, "firecrawl_raw_html"),
    ("scrapingdog.scrape", {"url": "https://boards-api.greenhouse.io/v1/boards/acme/jobs"},
     "generic_ats_json:greenhouse_board"),
    ("scrapingdog.linkedinjobs", {"job_id": "1234567890"}, "harvest_linkedin_job"),
])
def test_routed_recovery_preserves_original_provider_body_and_http_cap(
    tmp_path, operation, parameters, adapter
):
    clock = [0.0]

    class Client:
        def __init__(self):
            self.requests = []

        def post(self, url, *, content, headers, timeout):
            self.requests.append((json.loads(content), timeout.read))
            if len(self.requests) == 1:
                clock[0] += 80.0
                result = document(operation=operation)
            else:
                result = document(200, operation=operation)
            return httpx.Response(200, json=result)

    client = Client()
    host = worker(tmp_path, runner.HttpArenaApiClient("https://gateway.example.com", client=client),
                  monotonic=lambda: clock[0])
    assert host._dispatch_once(operation, parameters, 60_000)[1]["status"] == 200
    original, _ = client.requests[0]
    replay, timeout = client.requests[1]
    assert original == replay and replay["timeout_ms"] == 60_000
    assert timeout == 45.0
    routes = [scoring_provider_compat.route_for(
        kind="score", funding_source="miner_key", round_id="arena-2026-10-06",
        operation_id=frame["operation_id"], parameters=frame["parameters"],
        timeout_ms=frame["timeout_ms"],
    ) for frame in (original, replay)]
    assert all(route is not None and route.adapter == adapter for route in routes)
    assert routes[0].effective_parameters == routes[1].effective_parameters


def test_native_frame_and_plain_http_clients_receive_recovered_data(tmp_path):
    for mode in ("frame", "http"):
        api = Api(document(), document(200, recovered=True))
        host = worker(tmp_path, api)
        if mode == "frame":
            encoded = host.handle_frame(shim.build_operation_frame("deepline.execute", PARAMETERS, 5000))
            status, headers, body = shim.parse_worker_response(json.loads(encoded))
        else:
            status, headers, body = host.handle_http("POST",
                "https://code.deepline.com/api/v2/integrations/exa_search/execute",
                contracts.canonical_json({"payload": PARAMETERS["payload"]}).encode(),
                {"content-type": "application/json"})
        assert status == 200 and headers[runner.SETTLED_MICROUSD_HEADER] == "2000"
        assert json.loads(body)["results"][0]["url"] == "https://example.com"
        assert len(api.frames) == 2 and host._state.action_sequence == 1


@pytest.mark.parametrize("amount", [0, 3000])
@pytest.mark.parametrize("status", [402, 403, 404, 422, 429, 500, 502])
def test_confirmed_provider_error_without_transport_loss_is_terminal(tmp_path, amount, status):
    initial = document(status)
    initial["call"].update(outcome="settled", actual_microusd=amount,
                           provider_status=status)
    api = Api(initial)
    host = worker(tmp_path, api)

    assert host._dispatch_once("deepline.execute", PARAMETERS, 30_000)[1]["status"] == status
    assert len(api.frames) == 1
    assert host._state.calls == [initial["call"]]
