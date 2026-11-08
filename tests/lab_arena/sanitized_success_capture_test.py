"""Retain private source evidence removed by the existing response sanitizer."""
import base64
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker


def execute(body, *, operation="scrapingdog.scrape", parameters=None, thinking=False, headers=None):
    broker, store, transport = make_broker(transport=FakeTransport([(200, body, headers or {})]))
    context = replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY if thinking else None)
    arguments = dict(operation_id=operation, parameters=parameters or {"url": "https://example.com/about"},
                     action_sequence=0, timeout_ms=5000)
    result = broker.execute(context, **arguments)
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    replay = broker.execute(context, **arguments)
    assert result.status == replay.status == 200
    assert (result.headers, result.body) == (replay.headers, replay.body)
    assert len(transport.sent) == 1
    return result, terminal


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("projection", ["truncate", "scripts"])
def test_successful_text_sanitization_retains_source_or_explicit_capacity_gap(thinking, projection):
    limit = operations.OPERATIONS["scrapingdog.scrape"].max_response_bytes
    raw = (b"<html><script>" + b"x" * (limit + 1) + b"</script><p>Source facts.</p></html>"
           if projection == "scripts" else b"<html>" + b"x" * limit + b"Tail source facts.</html>")
    result, terminal = execute(raw, thinking=thinking, headers={"content-type": "text/html"})
    assert result.body != raw and len(result.body) <= limit
    receipt = terminal["upstream_response_receipt"]
    if projection == "scripts":
        assert base64.b64decode(receipt["body_b64"]) == raw
        assert receipt["privacy"] == {"redacted": False, "redactions": []}
    else:
        # Two large bodies exceed the unchanged terminal capacity. Make that
        # loss explicit without changing the delivered page or ledger bounds.
        assert receipt["status"] == "unavailable" and receipt["reason"] == "response_size_limit"


def envelope(*, people=False, secret=False):
    data = {"results": [{"url": "https://example.com/source", "text": "Source facts."}]}
    if people:
        data["results"][0]["entities"] = [
            {"type": "person", "name": "Private fixture person"},
            {"type": "organization", "name": "Public company"}]
    metadata = {"source_date": "2026-11-07"}
    if people:
        metadata["nested"] = [{"details": {"entities": [
            {"type": "PERSON", "name": "Private nested fixture person"}]}}]
    if secret:
        metadata["api_key"] = "synthetic-private-source-secret"
    return {"job_id": "fixture-job", "status": "completed", "billing": {"credits_charged": 0.02},
            "result": {"data": data}, "provider_metadata": metadata}


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("people", [False, True])
def test_deepline_unwrap_preserves_metadata_and_existing_person_privacy(thinking, people):
    source = envelope(people=people, secret=True)
    result, terminal = execute(source, operation="exa.search", parameters={"query": "company facts"}, thinking=thinking)
    receipt = terminal["upstream_response_receipt"]
    saved = json.loads(base64.b64decode(receipt["body_b64"]))
    assert saved["provider_metadata"]["source_date"] == "2026-11-07"
    assert saved["provider_metadata"]["api_key"] == "[REDACTED]"
    assert saved["result"]["data"] == json.loads(result.body)
    assert receipt["privacy"]["redactions"] == (["credential", "deepline_person_entities"] if people else ["credential"])
    assert "synthetic-private-source-secret" not in json.dumps(terminal)
    assert "Private fixture person" not in json.dumps(saved) + result.body.decode()
    assert "Private nested fixture person" not in json.dumps(saved) + result.body.decode()
    assert receipt["wire_body_sha256"] == contracts.hash_bytes(json.dumps(source).encode())
    assert receipt["privacy"]["redacted"] is True
    assert result.call["actual_microusd"] == 2000


def test_person_only_scrub_does_not_duplicate_the_filtered_response():
    source = envelope(people=True)
    result, terminal = execute(source, operation="deepline.execute", parameters={
        "tool": "exa_search", "payload": {"query": "company facts"}})
    assert "Private fixture person" not in result.body.decode()
    assert "upstream_response_receipt" not in terminal


def test_unchanged_source_response_is_not_duplicated():
    raw = b"<html>Source facts.</html>"
    result, terminal = execute(raw, headers={"content-type": "text/html"})
    assert result.body == raw and "upstream_response_receipt" not in terminal


def test_oversized_source_capture_keeps_existing_explicit_limit():
    raw = b"<html><script>" + b"x" * (3 * 1024 * 1024 + 1) + b"</script>Source facts.</html>"
    _, terminal = execute(raw, headers={"content-type": "text/html"})
    receipt = terminal["upstream_response_receipt"]
    assert receipt["status"] == "unavailable" and receipt["reason"] == "response_size_limit"


def test_semantically_equal_json_does_not_duplicate_source_and_type_changes_are_distinct():
    assert br._upstream_body_matches_response(b'{ "facts": [1, true] }', b'{"facts":[1,true]}',
        operation_id="deepline.execute", parameters={"tool": "exa_search"})
    assert not br._upstream_body_matches_response(b'{"facts": [1]}', b'{"facts":[true]}',
        operation_id="deepline.execute", parameters={"tool": "exa_search"})


def test_explicit_people_search_keeps_its_existing_person_entity_permission():
    raw = json.dumps(envelope(people=True)).encode()
    captured = br._capture_upstream_response(br.ProviderResponse(200, {}, raw),
        operation_id="deepline.execute", parameters={"tool": "exa_people_search"},
        request_hash="request", call_identity="call")
    assert base64.b64decode(captured["body_b64"]) == raw
    assert captured["privacy"] == {"redacted": False, "redactions": []}


def test_export_retains_filtered_envelope_with_truthful_upstream_only_privacy():
    from lab_arena.training_export import assemble_example
    from tests.lab_arena.training_export_test import add_call, fixture
    _, terminal = execute(envelope(people=True), operation="exa.search", parameters={"query": "company facts"})
    values = fixture()
    add_call(values, provider="deepline", operation="exa.search")
    before = assemble_example(**values)
    reserve = values["ledger_rows"][0]
    receipt = dict(terminal["upstream_response_receipt"], call_identity=reserve["call_identity"],
                   request_hash=reserve["entry_doc"]["request_hash"])
    values["ledger_rows"][-1]["terminal_response"]["upstream_response_receipt"] = receipt
    example = assemble_example(**values)
    call = example["provider_calls"][0]
    assert call["request"] == before["provider_calls"][0]["request"]
    assert call["response"] == before["provider_calls"][0]["response"]
    assert example["worker_observation_artifacts"] == before["worker_observation_artifacts"]
    retained = call["upstream_response_receipts"][0]
    assert retained["response"]["provider_metadata"]["source_date"] == "2026-11-07"
    assert retained["privacy_redactions"] == ["deepline_person_entities"]
    assert retained["delivery_to_worker"] == "not_proved"
    assert retained["wire_body_sha256"] == receipt["wire_body_sha256"]
    assert "upstream_response_privacy_nonexact" in call["evidence_gaps"]
    assert "upstream_response_capture_unavailable" not in call["evidence_gaps"]
    assert example["privacy"]["exact_replay"] is False
    assert "Private fixture person" not in json.dumps(example)
    assert "Private nested fixture person" not in json.dumps(example)

    # Auxiliary upstream evidence must not become a consumed observation,
    # prompt, or target on an independently proved full-thinking model turn.
    from lab_arena import training_thinking
    from tests.lab_arena.training_steps_test import seal
    from tests.lab_arena.training_thinking_test import full_example
    def consumed_context(tool):
        source = full_example()
        source["provider_calls"].append(dict(deepcopy(tool), run_id=source["run_id"],
                                              call_identity="unused-tool-evidence"))
        rows = training_thinking.build([seal(source)])[0]
        return [{key: row.get(key) for key in ("input", "target", "prior_observations")}
                for split in ("train", "eval") for row in rows[split]]
    old_context = consumed_context(before["provider_calls"][0])
    assert old_context and consumed_context(call) == old_context
    assert "provider_metadata" not in json.dumps(old_context)
