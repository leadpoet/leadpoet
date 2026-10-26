"""Use the native generic HTTP receipt without changing other tool contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, deepline_catalog, operations, shim
from tests.lab_arena.deepline_late_response_recovery_test import ResponseStore
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, DL_KEY, FakeTransport, HOST_KEYS, make_broker,
)

FIXTURE = Path(__file__).parent / "fixtures/deepline/generic_http_response_contract.json"


def saved():
    return json.loads(FIXTURE.read_text())


def parameters(response_contract="raw-v2"):
    value = saved()["parameters"]
    if response_contract is not None:
        value["response_contract"] = response_contract
    return value


def execute(broker, context=CONTEXT, *, response_contract="raw-v2"):
    return broker.execute(context, operation_id="deepline.execute",
                          parameters=parameters(response_contract), action_sequence=0,
                          timeout_ms=1000)


class ContractTransport(FakeTransport):
    """Return the two observed wire shapes based on the host's contract header."""
    def send(self, **request):
        name = ("native_raw_v2" if request["headers"].get(
            "x-deepline-execute-response-contract") == "raw-v2" else "arena_raw")
        self.responses = [(200, saved()[name])]
        return super().send(**request)


def test_native_receipt_reaches_direct_caller_and_replays_with_original_identity():
    broker, store, transport = make_broker(transport=ContractTransport())
    result = execute(broker)
    assert result.status == 200
    document = json.loads(result.body)
    # Regression: intent=raw alone produces a string with none of this proof.
    assert document == saved()["native_raw_v2"]
    raw = document["toolResponse"]["rawV2"]
    assert raw["requested_url"] == saved()["parameters"]["payload"]["url"]
    assert raw["final_url"] == raw["requested_url"]
    assert raw["status_code"] == 200 and raw["ok"] is True
    assert raw["headers"]["content-type"].startswith("text/plain")
    assert result.call["actual_microusd"] == 0
    assert result.call["request_hash"] == contracts.document_hash(parameters())
    assert result.call["request_hash"] != contracts.document_hash(parameters(None))
    request = transport.sent[0]
    assert request["headers"]["x-deepline-tool-error-schema"] == "1"
    assert json.loads(request["body"]) == {
        "provider": "generic_http", "operation": "generic_http_request",
        "payload": saved()["parameters"]["payload"],
    }
    assert "set-cookie" not in result.headers
    replay = execute(broker)
    assert replay.body == result.body and replay.call["idempotent"] is True
    assert replay.call["call_identity"] == result.call["call_identity"]
    assert len(transport.sent) == store.log.count("settle") == 1


@pytest.mark.parametrize("shape", ["arena_raw", "native_raw_v2"])
def test_historical_settled_envelope_is_replayed_without_reinterpretation(shape):
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, saved()[shape])]))
    prior = execute(broker, response_contract=None)
    historical = deepcopy(store.calls)
    replay = execute(broker, response_contract=None)
    assert json.loads(replay.body) == saved()[shape]
    assert replay.body == prior.body
    assert replay.call["call_identity"] == prior.call["call_identity"]
    assert replay.call["actual_microusd"] == 0 and replay.call["idempotent"] is True
    assert store.calls == historical and len(transport.sent) == 1


@pytest.mark.parametrize("tool", [t for t in operations.DEEPLINE_TOOLS
                                  if t != "generic_http_request"])
def test_other_tools_keep_the_existing_raw_contract(tool):
    payload = {
        "harvestapi_get_company": {"url": "https://www.linkedin.com/company/example"},
        "harvestapi_get_profile": {"url": "https://www.linkedin.com/in/example"},
        "zerobounce_validate": {"email": "a@example.com"},
        "bounceban_verify_single": {"email": "a@example.com"},
        "bounceban_get_single_status": {"id": "verification-id"},
    }.get(tool, {})
    outbound = operations.build_outbound_request("deepline.execute", {"tool": tool, "payload": payload})
    assert dict(outbound.headers) == {"x-deepline-execute-response-intent": "raw"}


def test_header_policy_is_host_owned_and_visible_in_operation_table():
    outbound = operations.build_outbound_request("deepline.execute", parameters())
    assert dict(outbound.headers) == {
        "x-deepline-execute-response-intent": "raw",
        "x-deepline-execute-response-contract": "raw-v2",
        "x-deepline-tool-error-schema": "1",
    }
    assert operations.operation_table_document()["deepline_tool_response_contracts"] == {
        "generic_http_request": {"raw-v2": {
            "x-deepline-execute-response-contract": "raw-v2",
            "x-deepline-tool-error-schema": "1",
        }},
    }
    matched = operations.match_request("POST", outbound.url,
        json.dumps({"payload": saved()["parameters"]["payload"]}).encode(),
        {"x-deepline-execute-response-contract": "raw-v2"})
    assert matched == ("deepline.execute", parameters())


@pytest.mark.parametrize("credits", [0, 0.02])
def test_native_receipt_keeps_authoritative_explicit_billing(credits):
    document = saved()["native_raw_v2"]
    document["billing"].update(credits_charged=credits, cost_usd=credits / 10)
    broker, store, transport = make_broker(transport=FakeTransport([(200, document)]))
    result = execute(broker)
    assert result.status == 200 and result.call["actual_microusd"] == int(credits * 100_000)
    assert next(iter(store.calls.values()))["actual"] == int(credits * 100_000)
    assert len(transport.sent) == 1


@pytest.mark.parametrize("location", ["body", "header"])
def test_native_receipt_cannot_expose_host_credential(location):
    document = saved()["native_raw_v2"]
    raw = document["toolResponse"]["rawV2"]
    secret = HOST_KEYS["deepline"]
    if location == "body":
        raw["data"] = secret
    else:
        raw["headers"]["x-provider-debug"] = secret
    broker, store, _ = make_broker(transport=FakeTransport([(200, document)]))
    result = execute(broker)
    assert result.status == 502
    assert secret not in json.dumps(result.to_document())
    assert secret not in json.dumps(store.calls)


def test_running_same_key_recovery_uses_native_contract_on_both_dispatches(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    fixture = saved()
    native = fixture["native_raw_v2"]["job_id"]

    class Transport:
        def __init__(self):
            self.sent = []
            self.key = None

        def send(self, **request):
            self.sent.append(request)
            if request["method"] == "POST":
                if self.key is None:
                    self.key = request["headers"]["idempotency-key"]
                    raise br.ProviderTransportError("ReadTimeout")
                assert request["headers"]["idempotency-key"] == self.key
                return br.ProviderResponse(200, {}, json.dumps(fixture["native_raw_v2"]).encode())
            if "/executions/by-key/" in request["url"]:
                document = {"requestId": native, "toolId": "generic_http_request",
                            "executionRecovery": {"idempotencyKey": self.key, "state": "running"}}
                return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"},
                                           json.dumps(document).encode())
            assert "/billing/usage?request_id=" in request["url"]
            return br.ProviderResponse(200, {}, json.dumps({"recent": {"request_id": native, "entries": []}}).encode())

    catalog = deepline_catalog.freeze_catalog({"tools": [row("generic_http_request",
        provider="generic_http", inputSchema={"type": "object", "properties": {
            "url": {"type": "string"}, "method": {"type": "string"}}, "required": ["url"]},
        billingSource="free", pricing={"unit": "call", "usdPerUnit": 0,
        "creditsPerUnit": 0, "currency": "USD"})]})
    transport = Transport()
    broker, store, _ = make_broker(store=ResponseStore(), transport=transport,
                                   credential_for=lambda *_: DL_KEY)
    context = replace(CONTEXT, deepline_catalog=catalog)
    first = execute(broker, context)
    assert first.status == 502
    recovered = execute(broker, context)
    assert recovered.status == 200 and json.loads(recovered.body) == fixture["native_raw_v2"]
    assert recovered.call["call_identity"] == first.call["call_identity"]
    assert recovered.call["actual_microusd"] == 0
    posts = [r for r in transport.sent if r["method"] == "POST"]
    assert len(posts) == 2 and store.log.count("dispatch") == 1
    for post in posts:
        assert post["headers"]["x-deepline-execute-response-contract"] == "raw-v2"
        assert post["headers"]["x-deepline-tool-error-schema"] == "1"
    assert posts[0]["body"] == posts[1]["body"]
    assert posts[0]["headers"]["idempotency-key"] == posts[1]["headers"]["idempotency-key"]


@pytest.mark.parametrize("shape", ["arena_raw", "native_raw_v2"])
@pytest.mark.parametrize("catalog_source", ["legacy", "free", "paid", None])
def test_missing_billing_preserves_only_the_same_verified_free_basis(shape, catalog_source):
    from lab_arena import provider_costs

    document = saved()[shape]
    del document["billing"]
    entry = None if catalog_source == "legacy" else {
        "tool_id": "generic_http_request", "pricing": {
            "unit": "call", "currency": "USD", "billing_source": catalog_source,
            "usd_per_unit": 0.01 if catalog_source == "paid" else 0,
            "credits_per_unit": 0.1 if catalog_source == "paid" else 0,
        },
    }
    cost = provider_costs.deepline_free_completed_cost(
        saved()["parameters"], 200, document, catalog_entry=entry)
    if catalog_source in {"legacy", "free"}:
        assert cost.microusd == 0
    else:
        assert cost is None


@pytest.mark.parametrize("shape", ["arena_raw", "native_raw_v2"])
@pytest.mark.parametrize("billing", [None, {}, {"credits_charged": -1},
    {"credits_charged": True}, {"credits_charged": "bad"}, {"credits_charged": 0.02}])
def test_present_billing_never_uses_the_fixed_zero_fallback(shape, billing):
    from lab_arena import provider_costs

    document = saved()[shape]
    document["billing"] = billing
    assert provider_costs.deepline_free_completed_cost(saved()["parameters"], 200, document) is None


@pytest.mark.parametrize("change", ["tool", "pending", "job_id", "view", "raw", "provider", "operation"])
def test_native_free_proof_rejects_unknown_tool_nonterminal_or_invalid_shape(change):
    from lab_arena import provider_costs

    document = saved()["native_raw_v2"]
    parameters = saved()["parameters"]
    del document["billing"]
    raw = document["toolResponse"]["rawV2"]
    if change == "tool":
        parameters["tool"] = "unpriced_tool"
    elif change == "pending":
        document["status"] = "running"
    elif change == "job_id":
        document["job_id"] = ""
    elif change == "view":
        document["toolResponse"]["view"] = "normalized"
    elif change == "raw":
        document["toolResponse"]["rawV2"] = "bare string"
    else:
        raw[change] = "another_tool"
    assert provider_costs.deepline_free_completed_cost(parameters, 200, document) is None


def test_native_missing_billing_still_settles_known_free_without_ledger_lookup():
    document = saved()["native_raw_v2"]
    del document["billing"]

    class Transport:
        def __init__(self):
            self.sent = []

        def send(self, **request):
            self.sent.append(request)
            assert request["method"] == "POST"
            return br.ProviderResponse(200, {}, json.dumps(document).encode())

    transport = Transport()
    broker, store, _ = make_broker(transport=transport)
    result = execute(broker)
    assert result.status == 200 and result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_generic_http_request_completed_zero"
    assert len(transport.sent) == store.log.count("settle") == 1


def test_new_legacy_call_keeps_exact_reply_headers_body_and_normalized_identity():
    broker, store, transport = make_broker(transport=ContractTransport())
    legacy = parameters(None)
    assert operations.validate_operation_request("deepline.execute", legacy) == legacy
    result = execute(broker, response_contract=None)
    assert json.loads(result.body) == saved()["arena_raw"]
    assert result.call["request_hash"] == contracts.document_hash(legacy)
    assert "x-deepline-execute-response-contract" not in transport.sent[0]["headers"]
    assert "x-deepline-tool-error-schema" not in transport.sent[0]["headers"]
    assert json.loads(transport.sent[0]["body"]) == {
        "provider": "generic_http", "operation": "generic_http_request",
        "payload": legacy["payload"],
    }
    assert result.call["actual_microusd"] == 0 and store.log.count("settle") == 1


@pytest.mark.parametrize("contract", [None, "raw-v2"])
def test_contract_choice_survives_http_frame_worker_and_broker_round_trip(monkeypatch, contract):
    broker, _, transport = make_broker(transport=ContractTransport())
    headers = {"Content-Type": "application/json"}
    if contract:
        headers.update({"X-Deepline-Execute-Response-Intent": "raw",
                        "X-Deepline-Execute-Response-Contract": contract,
                        "X-Deepline-Tool-Error-Schema": "1"})
    dispatched = []

    def dispatch(operation_id, normalized, timeout_ms):
        frame = shim.build_operation_frame(operation_id, normalized, timeout_ms)
        operation_id, normalized, timeout_ms = shim.decode_operation_frame(frame)
        dispatched.append(normalized)
        result = broker.execute(CONTEXT, operation_id=operation_id, parameters=normalized,
                                action_sequence=0, timeout_ms=timeout_ms)
        return result.status, result.headers, result.body, ""

    monkeypatch.setattr(shim, "_input_deepline_catalog", lambda: None)
    monkeypatch.setattr(shim, "trusted_scorer_mode", lambda: False)
    monkeypatch.setattr(shim, "_dispatch_with_response_url", dispatch)
    status, _, body, _ = shim._execute_with_response_url(
        method="POST", url="https://code.deepline.com/api/v2/integrations/generic_http_request/execute",
        body=json.dumps({"payload": parameters(None)["payload"]}).encode(),
        headers=headers, timeout_ms=1000)
    assert status == 200 and dispatched == [parameters(contract)]
    assert json.loads(body) == saved()["native_raw_v2" if contract else "arena_raw"]
    assert len(transport.sent) == 1


@pytest.mark.parametrize("contract", ["raw", "normalized", "RAW-V2", "", True, 1, {}, None])
def test_direct_contract_choice_is_closed(contract):
    candidate = parameters(None)
    candidate["response_contract"] = contract
    with pytest.raises(operations.OperationRequestError, match="invalid_field"):
        operations.validate_operation_request("deepline.execute", candidate)


@pytest.mark.parametrize("tool", ["exa_contents", "contextdev_post_web_search", "free_simple_company_search"])
def test_contract_choice_is_not_available_on_other_tools(tool):
    candidate = parameters()
    candidate["tool"] = tool
    with pytest.raises(operations.OperationRequestError, match="invalid_field"):
        operations.validate_operation_request("deepline.execute", candidate)


@pytest.mark.parametrize("headers", [
    {"x-deepline-execute-response-contract": "normalized"},
    {"x-deepline-execute-response-contract": "RAW-V2"},
    {"x-deepline-execute-response-contract": True},
    {"x-deepline-execute-response-intent": "normalized"},
    {"x-deepline-execute-response-contract": "raw-v2", "x-deepline-tool-error-schema": "2"},
    {"x-deepline-tool-error-schema": "1"},
    {"x-deepline-execute-response-contract": "raw-v2", "X-Deepline-Execute-Response-Contract": "raw-v2"},
])
def test_native_http_contract_headers_refuse_unsupported_or_ambiguous_values(headers):
    with pytest.raises(operations.OperationRequestError, match="invalid_request"):
        operations.match_request("POST", "https://code.deepline.com/api/v2/integrations/generic_http_request/execute",
            json.dumps({"payload": parameters(None)["payload"]}).encode(), headers)


def test_native_contract_headers_are_not_generally_forwardable():
    with pytest.raises(operations.OperationRequestError, match="unknown_header"):
        operations.match_request("POST", "https://code.deepline.com/api/v2/integrations/exa_search/execute",
            b'{"payload":{"query":"example"}}', {"x-deepline-execute-response-contract": "raw-v2"})
    with pytest.raises(operations.OperationRequestError, match="forbidden_header"):
        operations.match_request("POST", "https://code.deepline.com/api/v2/integrations/generic_http_request/execute",
            json.dumps({"payload": parameters(None)["payload"]}).encode(),
            {"x-deepline-execute-response-contract": "raw-v2", "authorization": "Bearer untrusted"})


def test_native_intent_only_and_direct_body_choice_normalize_without_defaults():
    url = "https://code.deepline.com/api/v2/integrations/generic_http_request/execute"
    body = {"payload": parameters(None)["payload"]}
    assert operations.match_request("POST", url, json.dumps(body).encode(),
        {"x-deepline-execute-response-intent": "raw"}) == ("deepline.execute", parameters(None))
    body["response_contract"] = "raw-v2"
    assert operations.match_request("POST", url, json.dumps(body).encode(),
        {"x-deepline-tool-error-schema": "1"}) == ("deepline.execute", parameters())


@pytest.mark.parametrize("protocol", ["http", "frame"])
@pytest.mark.parametrize("contract", [None, "raw-v2"])
def test_actual_worker_preserves_closed_contract_through_gateway_dispatch(tmp_path, protocol, contract):
    import base64
    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import FakeApi, lease

    catalog = deepline_catalog.freeze_catalog({"tools": [row("generic_http_request",
        provider="generic_http", inputSchema={"type": "object", "properties": {
            "url": {"type": "string"}, "method": {"type": "string"}}, "required": ["url"]})]})
    context = replace(CONTEXT, deepline_catalog=catalog)
    broker, _, transport = make_broker(transport=ContractTransport())

    class Api(FakeApi):
        def provider(self, run_id, lease_token, frame):
            self.provider_frames.append(frame)
            return broker.execute(context, operation_id=frame["operation_id"],
                parameters=frame["parameters"], action_sequence=frame["action_sequence"],
                timeout_ms=frame["timeout_ms"]).to_document()

    api = Api([])
    state = runner.RunState(lease=dict(lease(), deepline_catalog=catalog), lease_token="test-lease")
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", api, state)
    if protocol == "frame":
        frame = shim.build_operation_frame("deepline.execute", parameters(contract), 1000)
        result = json.loads(server.handle_frame(frame))
        status, body = result["status"], base64.b64decode(result["body_b64"])
    else:
        headers = {} if contract is None else {"x-deepline-execute-response-contract": contract,
                                               "x-deepline-tool-error-schema": "1"}
        status, _, body = server.handle_http("POST",
            "https://code.deepline.com/api/v2/integrations/generic_http_request/execute",
            json.dumps({"payload": parameters(None)["payload"]}).encode(), headers)
    assert status == 200
    assert api.provider_frames[0]["parameters"] == parameters(contract)
    assert json.loads(body) == saved()["native_raw_v2" if contract else "arena_raw"]
    assert state.action_sequence == len(transport.sent) == 1


def test_capture_hash_and_saved_request_retain_the_explicit_choice():
    captures = []

    def capture(document, **_kwargs):
        captures.append(document)
        return {"status": "unavailable"}

    broker, _, _ = make_broker(transport=ContractTransport(), capture_request=capture)
    result = execute(broker)
    assert result.status == 200 and len(captures) == 1
    captured = captures[0]
    for field in ("admitted_parameters", "normalized_parameters", "effective_parameters", "hashed_parameters"):
        assert captured[field] == parameters()
    assert captured["request_hash"] == contracts.document_hash(parameters())
    assert "response_contract" not in captured["outbound"]["body"]
