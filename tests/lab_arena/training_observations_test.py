"""Worker fallback evidence is complete, bounded, and never accounting authority."""

import base64
import copy
import gzip
import json
import socket
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, operations, runner, shim, training_capture, training_observations as obs
from lab_arena.api import create_app
from lab_arena.service import ServiceError
from lab_arena.training_export import assemble_example
from tests.lab_arena.training_export_test import Objects, add_call, assert_legacy_agent_gaps, fixture
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease
from tests.lab_arena.test_deepline_catalog import frozen


IDENTITY = "sha256:" + "1" * 64


def response(status=502, body=None, *, outcome="uncertain", identity=IDENTITY, operation="openrouter.chat"):
    return {"status": status, "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(json.dumps(body or {"error": {"code": "provider_unavailable", "detail": "failure " * 300}}).encode()).decode(),
            "call": {"operation_id": operation, "provider": "openrouter", "action_sequence": 0,
                     "outcome": outcome, "call_identity": identity}}


def observation(*, kind="transport_failure", action=0, ordinal=0, identity=None, operation="openrouter.chat", contributes=True):
    document = response()
    return {"ordinal": ordinal, "action_sequence": action, "operation_id": operation,
            "kind": kind, "call_identity": identity,
            "contributes_to_provider_call_count": contributes,
            "request": {"admitted_parameters": {"messages": [{"role": "user", "content": "full model request " * 400}]}, "timeout_ms": 1000},
            "outcome": {"protocol": "frame", "response": {key: document[key] for key in ("status", "headers", "body_b64")}},
            "privacy": {"redacted": False, "redactions": []}}


def test_optional_broker_disposition_requires_nonstandard_host_error():
    row = observation(kind="nonstandard_response", identity=IDENTITY, contributes=False)
    row["outcome"]["response"] = {
        "status": 502, "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(b'{"error":{"code":"provider_unavailable"}}').decode(),
    }
    row["outcome"]["broker_disposition"] = {
        "outcome": "not_dispatched", "reason": "provider_admission_deadline"}
    assert obs.clean_observation(row)["outcome"]["broker_disposition"] == row["outcome"]["broker_disposition"]
    for changed in (
        {"kind": "transport_failure"},
        {"call_identity": None},
        {"outcome": {**row["outcome"], "broker_disposition": {
            "outcome": "not_dispatched", "reason": "unknown"}}},
        {"outcome": {**row["outcome"], "broker_disposition": {
            "outcome": "uncertain", "reason": "provider_admission_deadline"}}},
        {"outcome": {**row["outcome"], "response": {
            **row["outcome"]["response"], "status": 503}}},
    ):
        with pytest.raises(obs.ObservationError, match="broker_disposition_invalid"):
            obs.clean_observation({**row, **changed})
    # Old observations without the optional field remain valid.
    old = copy.deepcopy(row)
    del old["outcome"]["broker_disposition"]
    assert "broker_disposition" not in obs.clean_observation(old)["outcome"]


def payload(*rows):
    return {"observations": list(rows), "observation_count": len(rows),
            "local_refusal_count": sum(row["kind"] == "local_refusal" for row in rows),
            "complete": True, "failures": []}


def save(values, document, *, generation=1, claim_hash=CLAIM_HASH):
    snapshot, compressed = obs.build_snapshot("execute:1", "execute", document,
                                              lease_generation=generation, claim_request_hash=claim_hash)
    values["objects"].put(obs.object_ref("execute:1", generation, claim_hash), compressed)
    values["runs"][0]["result_doc"]["resource_summary"]["worker_observations"] = {
        "status": "stored", "payload_sha256": snapshot["payload_sha256"],
        **{key: document[key] for key in ("observation_count", "local_refusal_count", "complete")}}
    return snapshot


@pytest.mark.parametrize("operation", ["openrouter.responses", "openrouter.chat"])
@pytest.mark.parametrize("protocol", ["frame", "http"])
def test_rejected_model_controls_survive_runner_snapshot_and_export(tmp_path, operation, protocol):
    parameters = {"model": "deepseek/deepseek-v4.1-flash-1017", "reasoning": {"effort": "high"}}
    history = "synthetic history " * 2100
    parameters.update({"input": history} if operation == "openrouter.responses" else {
        "messages": [{"role": "user", "content": history}]})
    # Exercise the actual legacy admission rejection, without changing its cap.
    frame = shim.build_operation_frame(operation, parameters, 1000)
    with pytest.raises(operations.OperationRequestError):
        shim.decode_operation_frame(frame)
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    api = FakeApi([])
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", api, state)
    if protocol == "frame":
        returned = json.loads(server.handle_frame(frame))
        expected_request = {"raw_frame": json.loads(frame)}
    else:
        path = "/api/v1/responses" if operation == "openrouter.responses" else "/api/v1/chat/completions"
        url = "https://openrouter.ai" + path
        headers = {"content-type": "application/json"}
        body = contracts.canonical_json(parameters).encode()
        with pytest.raises(operations.OperationRequestError):
            operations.match_request("POST", url, body, headers)
        status, returned_headers, returned_body = server.handle_http("POST", url, body, headers)
        assert status == 400
        returned = {"status": status, "headers": returned_headers,
                    "body_b64": base64.b64encode(returned_body).decode()}
        expected_request = {"method": "POST", "url": url, "headers": headers, "body_text": parameters}
    document = state.worker_observations.document()
    row = document["observations"][0]
    assert row["kind"] == "local_parse_error" and row["outcome"]["response"] == returned
    assert row["outcome"]["protocol"] == protocol
    assert row["request"]["admitted_parameters"] == expected_request
    assert row["privacy"] == {"redacted": False, "redactions": []}
    assert document["complete"] and state.action_sequence == len(state.calls) == len(api.provider_frames) == 0

    values = fixture()
    snapshot = save(values, document)
    readback = obs.read_snapshot(values["objects"], "execute:1", "execute", 1, CLAIM_HASH)
    assert readback == snapshot
    result = assemble_example(**values)
    exported = result["worker_observation_artifacts"]["execute:1"]
    assert exported == snapshot
    assert result["completeness"]["complete"]
    assert not result["privacy"]["privacy_filtered"]
    assert not result["privacy"]["hidden_reasoning_removed"]
    for value in (row["request"], row, document, snapshot, result):
        assert training_capture.sanitize_training_value(value) == (value, [])
        assert json.loads(contracts.canonical_json(value)) == value
    assert contracts.document_hash(exported["payload"]) == snapshot["payload_sha256"]
    restored_request = exported["payload"]["observations"][0]["request"]["admitted_parameters"]
    restored_parameters = (restored_request["raw_frame"]["parameters"] if protocol == "frame"
                           else restored_request["body_text"])
    assert contracts.document_hash(restored_parameters["reasoning"]) == contracts.document_hash(parameters["reasoning"])


@pytest.mark.parametrize("protocol", ["frame", "http"])
def test_rejected_model_fix_preserves_historical_privacy_flags(protocol):
    row = observation(kind="local_parse_error", operation="unknown", action=None, contributes=False)
    row["request"] = {"admitted_parameters": {"raw_frame": {
        "operation_id": "openrouter.responses", "parameters": {
            "model": "synthetic/model", "reasoning": "[REDACTED]", "input": "synthetic"}}},
        "timeout_ms": 0}
    row["outcome"] = {"protocol": "frame", "response": {"error": "invalid_request"}}
    if protocol == "http":
        row["request"] = {"admitted_parameters": {"method": "POST",
            "url": "https://openrouter.ai/api/v1/responses", "headers": {},
            "body_text": {"model": "synthetic/model", "reasoning": "[REDACTED]", "input": "synthetic"}},
            "timeout_ms": 0}
        row["outcome"] = {"protocol": "http", "response": {"status": 400, "headers": {},
            "body_b64": base64.b64encode(b'{"error":"invalid_request"}').decode()}}
    row["privacy"] = {"redacted": True, "redactions": ["hidden_reasoning"]}
    values = fixture()
    snapshot = save(values, payload(row))
    assert obs.read_snapshot(values["objects"], "execute:1", "execute", 1, CLAIM_HASH) == snapshot
    result = assemble_example(**values)
    assert result["worker_observation_artifacts"]["execute:1"] == snapshot
    assert result["privacy"]["hidden_reasoning_removed"] and result["privacy"]["privacy_filtered"]
    assert "worker_observation_privacy_nonexact" in result["completeness"]["incomplete_reasons"]


def test_owned_endpoint_roundtrip_immutable_and_private(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    route = "/arena/v1/runs/r1/training-observations"
    assert client.post(route, json=payload()).status_code == 401
    document = payload(observation())
    result = client.post(route, json=document, headers={"x-lab-arena-lease": TOKEN})
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    assert result.json() == service.handle_training_observations("r1", TOKEN, document)
    saved = obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["payload"] == document
    assert len(saved["payload"]["observations"][0]["request"]["admitted_parameters"]["messages"][0]["content"]) > 512
    with pytest.raises(ServiceError) as error:
        service.handle_training_observations("r1", TOKEN, payload())
    assert error.value.code == "training_observations_conflict"


@pytest.mark.parametrize("changes,token,code", [
    ({}, "b" * 64, "lease_invalid"), ({"status": "accepted"}, TOKEN, "lease_inactive"),
    ({"lease_expires_at": "2020-01-01T00:00:00Z"}, TOKEN, "lease_expired"),
])
def test_endpoint_rejects_wrong_or_inactive_lease_before_storage(tmp_path, changes, token, code):
    service, objects = service_for(tmp_path, **changes)
    with pytest.raises(ServiceError) as error:
        service.handle_training_observations("r1", token, payload())
    assert error.value.code == code
    assert not list(tmp_path.rglob("*.gz"))


def test_endpoint_requires_round_ownership_before_storage(tmp_path):
    service, _ = service_for(tmp_path)
    service._require_round_ownership = lambda _: (_ for _ in ()).throw(ServiceError("round_not_owned", 403))
    with pytest.raises(ServiceError, match="round_not_owned"):
        service.handle_training_observations("r1", TOKEN, payload())
    assert not list(tmp_path.rglob("*.gz"))


@pytest.mark.parametrize("mutation", ["hash", "run", "generation", "claim", "schema", "inflate", "duplicate_ordinal", "duplicate_action"])
def test_reader_rejects_corrupt_links_hashes_and_unbounded_objects(mutation):
    objects = Objects()
    snapshot, _ = obs.build_snapshot("r1", "execute", payload(observation()), lease_generation=1, claim_request_hash=CLAIM_HASH)
    if mutation == "hash":
        snapshot["payload"]["observations"][0]["request"]["timeout_ms"] = 2
    elif mutation == "run":
        snapshot["run_id"] = "other"
    elif mutation == "generation":
        snapshot["lease_generation"] = 2
    elif mutation == "claim":
        snapshot["claim_request_hash"] = "sha256:" + "d" * 64
    elif mutation == "schema":
        snapshot["schema_version"] = "unknown"
    elif mutation.startswith("duplicate"):
        second = copy.deepcopy(snapshot["payload"]["observations"][0])
        second["action_sequence" if mutation == "duplicate_ordinal" else "ordinal"] = 1
        snapshot["payload"]["observations"].append(second)
        snapshot["payload"]["observation_count"] = 2
        snapshot["payload_sha256"] = contracts.document_hash(snapshot["payload"])
    raw = b"x" * (obs.MAX_OBJECT_BYTES + 1) if mutation == "inflate" else contracts.canonical_json(snapshot).encode()
    objects.put(obs.object_ref("r1", 1, CLAIM_HASH), gzip.compress(raw, mtime=0))
    with pytest.raises(obs.ObservationError):
        obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)


def test_encoded_response_secrets_and_hidden_reasoning_are_filtered_losslessly():
    row = observation()
    row["request"]["admitted_parameters"]["reasoning"] = {"effort": "high"}
    raw = json.dumps({"message": TOKEN, "reasoning_content": "private reasoning", "null": "\x00", "api_key": None}).encode()
    row["outcome"]["response"]["body_b64"] = base64.b64encode(raw).decode()
    buffer = obs.Buffer()
    buffer.append(row, secrets=(TOKEN,))
    saved = buffer.observations[0]
    assert saved["request"]["admitted_parameters"]["reasoning"] == {"effort": "high"}
    returned = base64.b64decode(saved["outcome"]["response"]["body_b64"])
    assert TOKEN.encode() not in returned and b"private reasoning" not in returned
    assert json.loads(returned)["null"] == "\x00" and json.loads(returned)["api_key"] is None
    assert saved["privacy"] == {"redacted": True, "redactions": ["credential", "hidden_reasoning"]}


def test_worker_observation_keeps_plaintext_response_alias_but_redacts_request_alias():
    row = observation()
    row["request"]["admitted_parameters"]["messages"].append(
        {"role": "assistant", "reasoning_content": "prior request state"})
    response = {"choices": [{"message": {"role": "assistant",
                                        "reasoning_content": "provider returned text"}}]}
    row["outcome"]["response"]["body_b64"] = base64.b64encode(json.dumps(response).encode()).decode()
    buffer = obs.Buffer()
    buffer.append(row)
    saved = buffer.observations[0]
    assert saved["request"]["admitted_parameters"]["messages"][1]["reasoning_content"] == "[REDACTED]"
    body = json.loads(base64.b64decode(saved["outcome"]["response"]["body_b64"]))
    assert body["choices"][0]["message"]["reasoning_content"] == "provider returned text"
    assert saved["privacy"] == {"redacted": True, "redactions": ["hidden_reasoning"]}


def test_worker_observation_keeps_public_article_url_in_request_and_response():
    url = ("https://www.prnewswire.com/news-releases/biohaven-and-sk-"
           "biopharmaceuticals-enter-into-strategic-global-licensing-agreement-"
           "for-novel-kv7-ion-channel-platform-and-opakalim-lead-candidate-for-"
           "epilepsy-302860299.html")
    row = observation()
    row["request"]["admitted_parameters"]["messages"][0]["content"] = url
    body = {"url": url}
    row["outcome"]["response"]["body_b64"] = base64.b64encode(json.dumps(body).encode()).decode()
    buffer = obs.Buffer()
    buffer.append(row)
    saved = buffer.observations[0]
    assert saved["privacy"] == {"redacted": False, "redactions": []}
    assert saved["request"]["admitted_parameters"]["messages"][0]["content"] == url
    assert json.loads(base64.b64decode(saved["outcome"]["response"]["body_b64"])) == body


def test_worker_web_observation_keeps_public_article_url():
    url = ("https://www.prnewswire.com/news-releases/biohaven-and-sk-"
           "biopharmaceuticals-enter-into-strategic-global-licensing-agreement-"
           "for-novel-kv7-ion-channel-platform-and-opakalim-lead-candidate-for-"
           "epilepsy-302860299.html")
    body = "Public source: " + url
    data = {
        "url": url, "final_url": url, "retrieved_at": "2026-10-28T00:00:00Z",
        "status": 200, "content_type": "text/html", "body": body, "body_ref": None,
        "body_sha256": contracts.hash_bytes(body.encode()), "body_bytes": len(body.encode()),
        "stored_body_sha256": None, "stored_body_bytes": None,
    }
    row = {"id": 0, "kind": "web_fetch", "observed_at": "2026-10-28T00:00:00Z",
           "after_action_sequence": None, "depends_on": [], "data": data,
           "privacy": {"redacted": False, "redactions": []}}
    buffer = obs.Buffer()
    assert buffer.append_agent_observation(row) == 0
    saved = buffer.agent_observations[0]
    assert saved["privacy"] == {"redacted": False, "redactions": []}
    assert saved["data"]["url"] == saved["data"]["final_url"] == url
    assert saved["data"]["body"] == body


def test_buffer_overflow_is_explicit_incomplete_and_never_clipped():
    buffer = obs.Buffer()
    buffer.append(observation())
    row = observation(action=1, ordinal=1)
    row["request"]["admitted_parameters"]["messages"][0]["content"] = "x" * obs.MAX_OBJECT_BYTES
    buffer.append(row)
    assert buffer.document()["observation_count"] == 2
    assert not buffer.document()["complete"] and len(buffer.observations) == 1
    assert len(buffer.observations[0]["request"]["admitted_parameters"]["messages"][0]["content"]) > 512


def test_worker_gateway_failure_retains_full_request_and_exact_worker_error(tmp_path):
    class Api(FakeApi):
        def provider(self, *args, **kwargs):
            raise runner.RunnerError("do not retain exception prose")
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", Api([]), state)
    parameters = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "original question " * 300}], "max_tokens": 20}
    returned = server.handle_frame(shim.build_operation_frame("openrouter.chat", parameters, 1000))
    row = state.worker_observations.observations[0]
    assert row["request"]["admitted_parameters"] == parameters
    assert row["outcome"]["response"] == json.loads(returned) == {"error": "worker_unavailable"}
    assert row["outcome"]["exception_class"] == "RunnerError"
    assert row["contributes_to_provider_call_count"] is True
    assert state.action_sequence == len(state.calls) == 1
    assert "do not retain" not in json.dumps(row)


def test_failed_tool_response_is_full_and_normal_paid_success_is_not_duplicated(tmp_path):
    failed = response(body={"error": {"code": "provider_unavailable", "detail": "tool failure " * 300}})
    success = response(200, {"answer": "paid success " * 300}, outcome="settled")
    success["call"]["action_sequence"] = 1
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", FakeApi([], broker_documents=[failed, success]), state)
    frame = shim.build_operation_frame("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "request " * 400}], "max_tokens": 20}, 1000)
    server.handle_frame(frame)
    server.handle_frame(frame)
    assert len(state.worker_observations.observations) == 1
    body = base64.b64decode(state.worker_observations.observations[0]["outcome"]["response"]["body_b64"])
    assert len(json.loads(body)["error"]["detail"]) > 512
    assert "paid success" not in json.dumps(state.worker_observations.document())
    assert state.action_sequence == len(state.calls) == 2


def test_local_catalog_schema_is_full_and_refusal_keeps_existing_counters(tmp_path):
    catalog = response(200, {"tools": [{"input_schema": {"description": "complete schema " * 300, "maximum": 2 ** 64 - 1}}]}, outcome="local_catalog", identity=None, operation="deepline.tools.list")
    active = lease()
    active["deepline_catalog"] = frozen()
    state = runner.RunState(lease=active, lease_token=TOKEN)
    api = FakeApi([], broker_documents=[catalog])
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", api, state)
    server.handle_frame(shim.build_operation_frame("deepline.tools.list", {}, 1000))
    first = state.worker_observations.observations[0]
    assert first["kind"] == "local_catalog"
    assert json.loads(base64.b64decode(first["outcome"]["response"]["body_b64"]))["tools"][0]["input_schema"]["maximum"] == 2 ** 64 - 1
    state.operation_refusals[("openrouter.chat", "")] = (runner.MAX_REFUSED_FRAMES, "budget_exhausted")
    state.operation_refusal_until[("openrouter.chat", "")] = float("inf")
    frame = shim.build_operation_frame("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "refused input " * 100}], "max_tokens": 20}, 1000)
    assert json.loads(server.handle_frame(frame)) == {"error": "budget_exhausted"}
    second = state.worker_observations.observations[1]
    assert (first["ordinal"], second["ordinal"]) == (0, 1)
    assert second["kind"] == "local_refusal" and second["action_sequence"] == 1
    assert second["contributes_to_provider_call_count"] is False
    assert state.action_sequence == 2 and len(state.calls) == len(api.provider_frames) == 1
    assert state.local_refusals == state.worker_observations.local_refusals == 1


def test_freeze_marks_pending_daemon_handlers_incomplete_without_waiting():
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    state.worker_handlers_inflight = 1
    uploads = []
    api = SimpleNamespace(training_observations=lambda run, token, document: uploads.append(copy.deepcopy(document)) or {"status": "stored", "payload_sha256": contracts.document_hash(document)})
    summary = runner._record_training_observations(SimpleNamespace(api=api), state.lease, TOKEN, state)
    assert summary["status"] == "stored" and not summary["complete"]
    assert not uploads[0]["complete"] and state.worker_observations_frozen


def test_export_local_catalog_fills_count_gap_without_paid_ledger_entries():
    values = fixture()
    values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] = 1
    row = observation(kind="local_catalog", operation="deepline.tools.list")
    row["request"]["admitted_parameters"] = {}
    save(values, payload(row))
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert result["provider_calls"] == [] and result["costs"]["confirmed_actual_microusd"] == 0
    assert result["worker_observation_artifacts"]["execute:1"]["payload"]["observations"][0]["kind"] == "local_catalog"


def test_export_uses_worker_error_without_forging_provider_receipt_or_cost():
    values = fixture()
    add_call(values, status="uncertain")
    values["ledger_rows"][-1]["terminal_response"] = None
    row = observation(identity=IDENTITY)
    row["request"]["admitted_parameters"] = {"messages": [{"role": "user", "content": "x" * 12000}]}
    save(values, payload(row))
    result = assemble_example(**values)
    call = result["provider_calls"][0]
    assert "worker_observation_ref" in call and "response" not in call
    assert call["actual_microusd"] is None
    assert "full_response_unavailable" not in result["completeness"]["incomplete_reasons"]
    assert "provider_cost_unresolved" in result["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("has_worker_response", [False, True])
def test_only_returned_worker_response_fills_missing_ledger_response(has_worker_response):
    values = fixture()
    add_call(values)
    values["ledger_rows"][-1]["terminal_response"] = None
    row = observation(identity=None)
    row["request"]["admitted_parameters"] = {"messages": [{"role": "user", "content": "x" * 12000}]}
    if not has_worker_response:
        row["outcome"] = {"protocol": "frame", "exception_class": "RunnerError"}
    save(values, payload(row))

    result = assemble_example(**values)
    call = result["provider_calls"][0]
    assert "worker_observation_ref" in call and "response" not in call
    assert call["actual_microusd"] == 42
    assert ("full_response_unavailable" in call["evidence_gaps"]) is not has_worker_response
    assert not result["completeness"]["complete"]
    assert not result["privacy"]["exact_replay"]
    assert "agent_task_state_unverified" in result["completeness"]["incomplete_reasons"]


def test_export_transport_before_admission_retains_unknown_accounting():
    values = fixture()
    values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] = 1
    save(values, payload(observation()))
    result = assemble_example(**values)
    assert "provider_capture_count_unverified" not in result["completeness"]["incomplete_reasons"]
    assert "worker_call_accounting_unverified" in result["completeness"]["incomplete_reasons"]
    assert not result["completeness"]["complete"] and result["provider_calls"] == []


@pytest.mark.parametrize("mutation", ["missing", "foreign_claim", "foreign_run", "hash", "phantom_action", "identity"])
def test_export_cannot_fill_gap_with_absent_foreign_or_mislinked_artifact(mutation):
    values = fixture()
    add_call(values)
    values["ledger_rows"][-1]["terminal_response"] = None
    row = observation(identity=IDENTITY)
    row["request"]["admitted_parameters"] = {"messages": [{"role": "user", "content": "x" * 12000}]}
    if mutation == "phantom_action":
        row["action_sequence"] = 1
    elif mutation == "identity":
        row["call_identity"] = "sha256:" + "f" * 64
    snapshot = save(values, payload(row), claim_hash="sha256:" + "d" * 64 if mutation == "foreign_claim" else CLAIM_HASH)
    ref = obs.object_ref("execute:1", 1, "sha256:" + "d" * 64 if mutation == "foreign_claim" else CLAIM_HASH)
    if mutation == "missing":
        del values["objects"].values[ref]
    elif mutation in {"foreign_run", "hash"}:
        if mutation == "foreign_run":
            snapshot["run_id"] = "other"
        else:
            snapshot["payload_sha256"] = "sha256:" + "0" * 64
        values["objects"].values[ref] = gzip.compress(contracts.canonical_json(snapshot).encode(), mtime=0)
    result = assemble_example(**values)
    assert not result["completeness"]["complete"]
    assert "full_response_unavailable" in result["completeness"]["incomplete_reasons"]


def test_export_old_paid_only_fixture_keeps_labels_and_reports_missing_agent_receipts():
    values = fixture()
    add_call(values)
    assert_legacy_agent_gaps(assemble_example(**values))
    values["runs"][0]["result_doc"]["resource_summary"]["provider_local_refusal_count"] = 1
    assert "worker_observations_unavailable" in assemble_example(**values)["completeness"]["incomplete_reasons"]


def test_export_local_refusal_does_not_add_a_phantom_provider_call():
    values = fixture()
    resource = values["runs"][0]["result_doc"]["resource_summary"]
    resource.update(provider_call_count=0, provider_local_refusal_count=1)
    save(values, payload(observation(kind="local_refusal", contributes=False)))
    assert assemble_example(**values)["completeness"]["complete"]
    resource["provider_call_count"] = 1
    assert "provider_capture_count_unverified" in assemble_example(**values)["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("ending", ["success", "transport"])
def test_hidden_responses_retry_keeps_distinct_actions_and_only_final_fallback(tmp_path, monkeypatch, ending):
    from tests.lab_arena.codex_rate_limit_retry_test import identified_document, install_clock, PARAMETERS
    throttle = identified_document(0, idempotent=False)
    success = identified_document(1, status=200, provider_status=200, error_code=None, actual=7, idempotent=False)
    class Api:
        def provider(self, _run, _token, frame):
            if frame["action_sequence"] == 0:
                return throttle
            if ending == "transport":
                raise runner.RunnerError("no response")
            return success
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", Api(), state)
    install_clock(monkeypatch, server)
    result = json.loads(server.handle_frame(shim.build_operation_frame("openrouter.responses", PARAMETERS, 120_000)))
    assert state.action_sequence == len(state.calls) == 2
    assert state.worker_observation_ordinal == 1 and not state.worker_observations_pending
    if ending == "success":
        assert result["status"] == 200 and state.worker_observations.observations == []
    else:
        row = state.worker_observations.observations[0]
        assert (row["ordinal"], row["action_sequence"]) == (0, 1)
        assert row["outcome"]["response"] == result == {"error": "worker_unavailable"}
        assert state.calls[0]["call_identity"] == throttle["call"]["call_identity"]


@pytest.mark.parametrize("phase", ["begin", "finish"])
def test_capture_helper_failure_preserves_exact_returned_bytes_and_runtime_counts(tmp_path, monkeypatch, phase):
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    document = response()
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", FakeApi([], broker_documents=[document]), state)
    def fail(*args, **kwargs):
        raise RuntimeError("private capture failure")
    monkeypatch.setattr(server, "_begin_observation_unchecked" if phase == "begin" else "_finish_observation_unchecked", fail)
    frame = shim.build_operation_frame("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "research"}]}, 1000)
    result = server.handle_frame(frame)
    returned = json.loads(result)
    assert returned["status"] == document["status"]
    assert returned["body_b64"] == document["body_b64"]
    assert returned["headers"] == {**document["headers"],
                                    shim.CALL_IDENTITY_HEADER: document["call"]["call_identity"]}
    assert state.action_sequence == len(state.calls) == 1 and state.local_refusals == 0
    assert not state.worker_observations.document()["complete"]


def test_finalize_and_log_failure_cannot_override_execution_or_cleanup_error(monkeypatch):
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    def fail():
        raise RuntimeError("private artifact construction failed")
    class BrokenLog:
        def write(self, *_):
            raise OSError("log unavailable")
    monkeypatch.setattr(state.worker_observations, "document", fail)
    monkeypatch.setattr(runner.sys, "stderr", BrokenLog())
    result = runner._record_training_observations(SimpleNamespace(api=FakeApi([])), state.lease, TOKEN, state)
    assert result == {"status": "unavailable", "observation_count": 0, "local_refusal_count": 0, "complete": False}
    assert state.worker_observations_frozen


def test_invalid_utf8_parse_is_explicit_incomplete_without_changing_error(tmp_path):
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", FakeApi([]), state)
    assert json.loads(server.handle_frame(b"\xffinvalid")) == {"error": "invalid_frame"}
    assert state.action_sequence == len(state.calls) == 0
    assert not state.worker_observations.document()["complete"]


def test_accepted_handler_before_dispatch_makes_snapshot_incomplete(tmp_path):
    from pathlib import Path
    import tempfile
    import shutil
    directory = Path(tempfile.mkdtemp(prefix="arena-obs-", dir="/tmp"))
    accepted = threading.Event()
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(directory / "w.sock", FakeApi([]), state, read_timeout_seconds=2)
    # An accepted connection can still be blocked before a provider invocation.
    server.start()
    original = server._server.process_request
    def track_accepted(*args):
        original(*args)
        accepted.set()
    server._server.process_request = track_accepted
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(str(directory / "w.sock"))
        connection.sendall(b"\x00")
        assert accepted.wait(1)
        server.stop()
        result = runner._record_training_observations(SimpleNamespace(api=FakeApi([])), state.lease, TOKEN, state)
        assert not result["complete"] and state.action_sequence == 0
    finally:
        connection.close()
        server.stop()
        shutil.rmtree(directory)


def test_daemon_late_completion_cannot_mutate_frozen_artifact(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Api:
        def provider(self, *_args):
            entered.set()
            assert release.wait(2)
            return response()
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(tmp_path / "w.sock", Api(), state)
    frame = shim.build_operation_frame("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "research"}]}, 1000)
    returned = []
    thread = threading.Thread(target=lambda: returned.append(server.handle_frame(frame)))
    thread.start()
    assert entered.wait(1)
    uploads = []
    api = SimpleNamespace(training_observations=lambda run, token, doc: uploads.append(copy.deepcopy(doc)) or {"status": "stored", "payload_sha256": contracts.document_hash(doc)})
    result = runner._record_training_observations(SimpleNamespace(api=api), state.lease, TOKEN, state)
    assert not result["complete"]
    frozen = copy.deepcopy(uploads[0])
    release.set()
    thread.join(2)
    assert not thread.is_alive() and returned
    assert uploads[0] == frozen and state.worker_observations.observations == []


@pytest.mark.parametrize("has_identity", [True, False])
def test_same_action_multiple_provider_attempts_require_exact_call_identity(has_identity):
    values = fixture()
    add_call(values)
    first = copy.deepcopy(values["ledger_rows"])
    for row in values["ledger_rows"]:
        row["entry_id"] += 3
        row["call_identity"] = "sha256:" + "2" * 64
    values["ledger_rows"][-1]["terminal_response"] = None
    values["ledger_rows"] = first + values["ledger_rows"]
    row = observation(identity="sha256:" + "2" * 64 if has_identity else None)
    row["request"]["admitted_parameters"] = {"messages": [{"role": "user", "content": "x" * 12000}]}
    save(values, payload(row))
    result = assemble_example(**values)
    calls = result["provider_calls"]
    assert len(calls) == 2
    if has_identity:
        assert "worker_observation_ref" not in calls[0] and "worker_observation_ref" in calls[1]
        assert "full_response_unavailable" not in result["completeness"]["incomplete_reasons"]
    else:
        assert "worker_observation_link_invalid" in result["completeness"]["incomplete_reasons"]
        assert "full_response_unavailable" in result["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("field,value,reason", [
    ("provider_local_refusal_count", None, "worker_local_refusals_unverified"),
    ("provider_local_refusal_count", False, "worker_local_refusals_unverified"),
    ("web_egress", None, "direct_web_usage_unverified"),
    ("web_egress", {"connection_count": False}, "direct_web_usage_unverified"),
    ("web_egress", {"connection_count": -1}, "direct_web_usage_unverified"),
])
def test_unknown_or_invalid_runtime_counts_are_not_zero(field, value, reason):
    values = fixture()
    resource = values["runs"][0]["result_doc"]["resource_summary"]
    if value is None:
        del resource[field]
    else:
        resource[field] = value
    result = assemble_example(**values)
    assert reason in result["completeness"]["incomplete_reasons"]


def test_new_optional_summary_survives_signed_completion_and_service_validation():
    from tests.lab_arena.test_lab_arena_service_rules import _completion_service
    from tests.lab_arena.test_lab_arena_runner import RUNNER, sign, verify
    service = _completion_service(run_runner=RUNNER.ss58_address)
    validated, round_row = service._request_round({})
    body = validated["body"]
    body["result"]["resource_summary"]["worker_observations"] = {
        "status": "stored", "complete": True, "observation_count": 0,
        "local_refusal_count": 0, "payload_sha256": contracts.document_hash(payload())}
    envelope = contracts.build_signed_request(scope=contracts.SCOPE_COMPLETE, round_id=round_row["round_id"],
        hotkey=RUNNER.ss58_address, body=body, timestamp=1792195200, sign_message=sign)
    service._request_round = lambda document, **kwargs: (
        contracts.validate_signed_request(json.loads(json.dumps(document)), expected_scope=contracts.SCOPE_COMPLETE,
            now=1792195200, verify_signature=verify), round_row)
    saved = []
    service._store.complete_attempt = lambda **kwargs: saved.append(kwargs) or {"status": "failed"}
    assert service.handle_complete(envelope) == {"status": "failed"}
    assert saved[0]["result"]["resource_summary"]["worker_observations"] == body["result"]["resource_summary"]["worker_observations"]


def test_nonstandard_settled_frame_headers_are_not_dropped_as_paid_success(tmp_path):
    document = response(200, {"answer": "full result " * 100}, outcome="settled")
    document["headers"] = {"content-type": 17}
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", FakeApi([], broker_documents=[document]), state)
    frame = shim.build_operation_frame("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "research"}]}, 1000)
    returned = json.loads(server.handle_frame(frame))
    with pytest.raises(shim.ShimTransportError):
        shim.parse_worker_response(returned)
    row = state.worker_observations.observations[0]
    assert row["kind"] == "invalid_response" and row["outcome"]["response"] == returned
    assert row["outcome"]["exception_class"] == "ShimTransportError"
    assert state.action_sequence == len(state.calls) == 1


@pytest.mark.parametrize("protocol", ["http", "frame"])
def test_early_protocol_rejections_retain_exact_error_and_mark_unread_body_incomplete(protocol):
    from pathlib import Path
    import shutil
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix="arena-obs-", dir="/tmp"))
    state = runner.RunState(lease=lease(), lease_token=TOKEN)
    api = FakeApi([])
    server = runner.WorkerSocketServer(directory / "w.sock", api, state)
    server.start()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(2)
    try:
        connection.connect(str(directory / "w.sock"))
        if protocol == "http":
            connection.sendall(b"POST /api/v1/chat/completions HTTP/1.1\r\nHost: openrouter.ai\r\nContent-Length: invalid\r\n\r\n")
            returned = b""
            while True:
                part = connection.recv(8192)
                if not part:
                    break
                returned += part
            assert b"400 Bad Request" in returned
            expected = json.loads(returned.split(b"\r\n\r\n", 1)[1])
        else:
            connection.sendall((shim.MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
            size = int.from_bytes(runner._recv_exact(connection, 4), "big")
            expected = json.loads(runner._recv_exact(connection, size))
            assert expected == {"error": "frame_too_large"}
        assert state.action_sequence == len(state.calls) == len(api.provider_frames) == 0
        row = state.worker_observations.observations[0]
        actual = row["outcome"]["response"]
        assert (json.loads(base64.b64decode(actual["body_b64"])) if protocol == "http" else actual) == expected
        assert row["kind"] == "local_parse_error" and row["contributes_to_provider_call_count"] is False
        assert not state.worker_observations.document()["complete"]
    finally:
        connection.close()
        server.stop()
        shutil.rmtree(directory)


def test_verified_scorer_missing_web_metadata_means_no_bridge_but_untrusted_source_is_unknown():
    values = fixture()
    score_resource = values["runs"][1]["result_doc"]["resource_summary"]
    del score_resource["web_egress"]
    assert assemble_example(**values)["completeness"]["complete"]
    score_resource["web_egress"] = None
    assert "direct_web_usage_unverified" in assemble_example(**values)["completeness"]["incomplete_reasons"]
    del score_resource["web_egress"]
    score_resource["validator_source_commit"] = "unknown"
    assert "direct_web_usage_unverified" in assemble_example(**values)["completeness"]["incomplete_reasons"]


def test_missing_execute_proxy_web_count_is_unknown():
    values = fixture()
    values["round_row"]["configuration_doc"]["parallel_twenty_icp_execution"] = True
    del values["runs"][0]["result_doc"]["resource_summary"]["web_egress"]
    assert "direct_web_usage_unverified" in assemble_example(**values)["completeness"]["incomplete_reasons"]


def test_typed_response_reasoning_keeps_text_and_hash_stays_valid():
    row = observation()
    body = {"output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "visible summary"}],
                        "content": [{"type": "reasoning_text", "text": "visible text"}],
                        "encrypted_content": "sealed state"}]}
    row["outcome"]["response"]["body_b64"] = base64.b64encode(json.dumps(body).encode()).decode()
    values = fixture()
    values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] = 1
    snapshot = save(values, payload(row))
    artifact = obs.read_snapshot(values["objects"], "execute:1", "execute", 1, CLAIM_HASH)
    result = assemble_example(**values)
    assert artifact["payload_sha256"] == snapshot["payload_sha256"]
    exported = result["worker_observation_artifacts"]["execute:1"]
    assert contracts.document_hash(exported["payload"]) == exported["payload_sha256"]
    body = base64.b64decode(exported["payload"]["observations"][0]["outcome"]["response"]["body_b64"])
    assert b"visible summary" in body and b"visible text" in body and b"sealed state" not in body
    assert result["privacy"]["hidden_reasoning_removed"] and result["privacy"]["privacy_filtered"]


def test_old_observation_redaction_is_still_a_fixed_point():
    row = observation()
    body = {"output": [{"type": "reasoning", "summary": "[REDACTED]", "text": "[REDACTED]",
                        "content": "[REDACTED]", "encrypted_content": "[REDACTED]"}],
            "choices": [{"message": {"reasoning": "[REDACTED]", "reasoning_details": "[REDACTED]"}}]}
    row["outcome"]["response"]["body_b64"] = base64.b64encode(
        contracts.canonical_json(body).encode("utf-8")).decode()
    row["privacy"] = {"redacted": True, "redactions": ["hidden_reasoning"]}
    assert obs.clean_observation(row) == row


def test_api_and_client_observation_limits_and_timeout(tmp_path):
    import httpx
    service, _ = service_for(tmp_path)
    route = "/arena/v1/runs/r1/training-observations"
    response = TestClient(create_app(service)).post(route, content=b"{}", headers={"x-lab-arena-lease": TOKEN, "content-length": str(obs.MAX_OBJECT_BYTES + 1)})
    assert response.status_code == 413
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "stored", "payload_sha256": contracts.document_hash(payload())})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        api = runner.HttpArenaApiClient("http://localhost", client=client)
        assert api.training_observations("r1", TOKEN, payload())["status"] == "stored"
    assert requests[0].url.path == route and requests[0].headers["x-lab-arena-lease"] == TOKEN
    assert requests[0].extensions["timeout"]["read"] == runner.TRAINING_OBSERVATIONS_TIMEOUT_SECONDS


def test_planned_identity_without_admission_is_not_a_provider_receipt():
    values = fixture()
    values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] = 1
    save(values, payload(observation(identity=IDENTITY, kind="nonstandard_response")))
    result = assemble_example(**values)
    assert "worker_observation_link_invalid" not in result["completeness"]["incomplete_reasons"]
    assert "worker_call_accounting_unverified" in result["completeness"]["incomplete_reasons"]
    assert result["provider_calls"] == []


def test_non_catalog_operation_cannot_claim_zero_cost_catalog_fallback():
    with pytest.raises(obs.ObservationError, match="catalog_identity_invalid"):
        obs.build_snapshot("r1", "execute", payload(observation(kind="local_catalog")),
                           lease_generation=1, claim_request_hash=CLAIM_HASH)


@pytest.mark.parametrize("operation_id", ["not-a-registered-operation", TOKEN])
def test_endpoint_rejects_untrusted_operation_metadata_before_storage(tmp_path, operation_id):
    service, _ = service_for(tmp_path)
    with pytest.raises(ServiceError) as error:
        service.handle_training_observations("r1", TOKEN, payload(observation(operation=operation_id)))
    assert error.value.code == "training_observations_invalid"
    assert not list(tmp_path.rglob("*.gz"))
