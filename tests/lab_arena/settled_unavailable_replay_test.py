"""A saved broker-unavailable response remains a failure without redispatch."""
import copy
from dataclasses import replace
import json
import os

import pytest

from lab_arena import contracts, operations, runner, runtime, shim
from tests.lab_arena import test_lab_arena_runner as runner_tests
from tests.lab_arena import deepline_execute_response_size_test as sizes
from tests.lab_arena import openrouter_data_policy_route_test as routes


def case(name, claimed):
    if name == "oversize":
        broker, ledger, transport, context, request = sizes._execute(sizes._body_of_size(2_437_777))
        claimed["deepline_catalog"] = sizes.CATALOG
    else:
        payload = copy.deepcopy(routes.SUCCESS)
        operation, parameters, status = "openrouter.chat", routes.CHAT, 200
        if name == "quoted_content":
            payload["choices"][0]["message"]["content"] = operations.GENERIC_UNAVAILABLE_BODY.decode()
        elif name == "caller_refusal":
            status, payload = 400, {"error": {"code": 400, "message": "Invalid request parameter"}}
        elif name == "source404":
            operation, parameters = "scrapingdog.scrape", {"url": "https://example.com/missing"}
            status, payload = 404, {"error": "not found"}
        elif name in ("auth", "credit"):
            status = 401 if name == "auth" else 402
            payload = {"error": {"code": status, "message": "Invalid API key" if name == "auth" else "Insufficient credits"}}
        elif name == "policy_route404":
            status, payload = 404, routes.ERROR
        else:
            assert name == "success"
        broker, ledger, transport = routes.make_broker([(status, payload)])
        context = routes.broker_tests.CONTEXT
        request = dict(operation_id=operation, parameters=parameters, action_sequence=0, timeout_ms=5000)
    context = replace(context, run_id=claimed["run_id"], round_id=claimed["round_id"],
                      assignment_id=claimed["assignment_id"],
                      lease_token_hash=contracts.document_hash(claimed["lease_token"]))
    return broker, ledger, transport, context, request


@pytest.mark.parametrize("preexisting", [False, True], ids=["live", "settled-replay"])
@pytest.mark.parametrize("name,expected", [
    ("oversize", "provider_error"), ("success", "accepted"),
    ("quoted_content", "accepted"), ("caller_refusal", "accepted"),
    ("source404", "accepted"), ("auth", "credential_error"),
    ("credit", "credential_error"), ("policy_route404", "provider_error"),
])
def test_real_broker_worker_runner_live_and_replay_parity(tmp_path, preexisting, name, expected):
    claimed = runner_tests.lease()
    broker, ledger, transport, context, request = case(name, claimed)
    original = broker.execute(context, **request) if preexisting else None
    preserved = copy.deepcopy(ledger.calls) if preexisting else None
    received, calls, served = [], [], []

    class Api(runner_tests.FakeApi):
        def provider(self, run_id, token, frame):
            result = broker.execute(context, operation_id=frame["operation_id"],
                parameters=frame["parameters"], action_sequence=frame["action_sequence"],
                timeout_ms=frame["timeout_ms"])
            calls.append(result.call)
            served.append((result.status, dict(result.headers), result.body))
            return result.to_document()

    class Sandbox(runner_tests.BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            self.specs.append(spec)
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                received.append(shim.dispatch(request["operation_id"], request["parameters"], request["timeout_ms"]))
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            spec.output_path.write_bytes(b'{"companies":[]}')
            return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))

    api = Api([claimed])
    (tmp_path / "work").mkdir()
    instance = runner.Runner(runner_tests.make_config(tmp_path, api, Sandbox(calls=0)))
    assert instance.run_once() == 1
    completed = api.completions[0]["body"]
    result = contracts.validate_run_result(completed["result"])
    assert result["terminal_status"] == expected
    assert result["resource_summary"]["provider_unresolved_failure_count"] == (1 if expected != "accepted" else 0)
    assert runner._unresolved_operation_failures(calls, set()) == (
        (1, 0) if expected == "credential_error" else (0, 1) if expected == "provider_error" else (0, 0))
    assert (completed["output"] is None) == (expected != "accepted")
    assert len(received) == len(calls) == 1
    native_requests = transport.requests if name == "oversize" else transport.sent
    assert [sent["method"] for sent in native_requests] == ["POST" if name != "source404" else "GET"]
    assert ledger.log.count("dispatch") == ledger.log.count("settle") == 1
    if preexisting:
        assert ledger.calls == preserved
        assert calls[0]["call_identity"] == original.call["call_identity"]
        assert calls[0]["actual_microusd"] == original.call["actual_microusd"]
        assert served[0] == (original.status, original.headers, original.body)
        assert received[0][0] == original.status and received[0][2] == original.body
        assert all(received[0][1][key] == value for key, value in original.headers.items())
    if name == "oversize":
        assert operations.get_operation("deepline.execute").max_response_bytes == sizes.LIMIT
        assert received[0][0] == 502 and received[0][2] == operations.GENERIC_UNAVAILABLE_BODY
        assert calls[0]["error_code"] == "provider_unavailable"
        assert calls[0]["actual_microusd"] == 8900
        assert calls[0].get("provider_status") is None
        if original:
            assert original.call["observed_provider_http_status"] == 200
    if name == "policy_route404":
        assert calls[0]["provider_status"] == 404


@pytest.mark.parametrize("status,body", [
    (200, operations.GENERIC_UNAVAILABLE_BODY),
    (404, operations.GENERIC_UNAVAILABLE_BODY),
    (502, b'{"error":{"code":"different_error"}}'),
    (502, b'{ "error": { "code": "provider_unavailable" } }'),
    (502, b'{"content":"provider_unavailable"}'),
])
def test_only_exact_authenticated_generic_terminal_gets_annotation(status, body):
    claimed = runner_tests.lease()
    broker, ledger, transport, context, request = case("oversize", claimed)
    initial = broker.execute(context, **request)
    # Negative stored-terminal controls cannot acquire the trusted marker
    # merely by quoting it or by using a different status/body encoding.
    terminal = ledger.calls[initial.call["call_identity"]]["terminal"]
    import base64
    terminal.update(status=status, body_b64=base64.b64encode(body).decode(),
                    headers={"content-type": "application/json", "content-length": str(len(body))})
    before = copy.deepcopy(ledger.calls)
    result = broker.execute(context, **request)
    assert result.status == status and result.body == body and result.headers == terminal["headers"]
    assert result.call.get("error_code") is None and result.call.get("provider_status") is None
    assert result.call["actual_microusd"] == 8900 and ledger.calls == before
    assert len(transport.requests) == 1 and ledger.log.count("settle") == ledger.log.count("dispatch") == 1
