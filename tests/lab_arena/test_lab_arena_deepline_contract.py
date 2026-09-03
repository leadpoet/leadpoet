"""Deepline execute contract, pinned from live calls on 2026-09-02.

The fixtures under ``fixtures/deepline`` were captured through the official
Deepline CLI (its Python core sends the request body and header asserted
here) and scrubbed of the person entities Exa attaches to page contents.
They prove the request shape the broker sends and the envelope it must read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import broker as br
from lab_arena import operations as ops

FIXTURES = Path(__file__).parent / "fixtures" / "deepline"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_outbound_request_matches_the_official_client():
    envelope = load("execute_envelope_exa_contents.json")
    outbound = ops.build_outbound_request("deepline.execute", {"tool": "exa_contents", "payload": envelope["request"]["body"]["payload"]})
    assert outbound.url == "https://code.deepline.com" + envelope["request"]["path"]
    assert json.loads(outbound.body) == {"provider": "exa", "operation": "exa_contents", "payload": {"urls": ["https://example.com/"], "text": {"maxCharacters": 50}}}
    assert dict(outbound.headers) == {"x-deepline-execute-response-intent": "raw"}
    url, headers = br.inject_credential(outbound, "dl_test_key_" + "x" * 20)
    assert url == outbound.url and headers["authorization"] == "Bearer dl_test_key_" + "x" * 20
    assert headers["x-deepline-execute-response-intent"] == "raw" and headers["content-type"] == "application/json"
    # The free company search names its own provider.
    body = json.loads(ops.build_outbound_request("deepline.execute", {"tool": "free_simple_company_search", "payload": {"sql": "SELECT 1"}}).body)
    assert body == {"provider": "deepline_native", "operation": "free_simple_company_search", "payload": {"sql": "SELECT 1"}}
    assert set(ops.DEEPLINE_TOOL_PROVIDERS) == set(ops.DEEPLINE_TOOLS)


def test_envelope_carries_the_raw_exa_response_under_result_data():
    envelope = load("execute_envelope_exa_contents.json")["response"]
    assert envelope["status"] == "completed" and set(envelope) == {"job_id", "status", "result", "billing"}
    data = envelope["result"]["data"]
    assert set(data) == {"requestId", "results", "statuses", "searchTime"}
    result = data["results"][0]
    assert result["url"] == "https://example.com/" and result["id"] == result["url"]
    assert len(result["text"]) == 50  # text.maxCharacters is honored by the upstream provider
    assert data["statuses"][0] == {"id": "https://example.com/", "status": "success", "source": "cached"}
    assert result["entities"] == []  # scrubbed: Exa attaches person entities to page contents
    body = json.dumps(envelope).encode("utf-8")
    status, headers, sanitized = ops.sanitize_response("deepline.execute", 200, {"content-type": "application/json"}, body)
    assert status == 200 and json.loads(sanitized) == envelope
    assert br.deepline_cost_microusd(sanitized) == 2000  # 0.002 USD


@pytest.mark.parametrize("name, expected_results, expected_cost", [("exa_search.json", 2, 10_000), ("exa_contents.json", 1, 2_000), ("exa_contents_text_true.json", 1, 2_000)])
def test_tool_responses_keep_exa_shape_and_pricing(name, expected_results, expected_cost):
    fixture = load(name)
    raw = fixture["tool_response_raw"]
    assert len(raw["results"]) == expected_results and all("url" in item and "id" in item for item in raw["results"])
    assert raw["requestId"] and "searchTime" in raw
    assert int(fixture["billing"]["cost_usd"] * 1_000_000) == expected_cost
    if name == "exa_contents.json":
        assert len(raw["results"][0]["text"]) == 300  # maxCharacters 300 honored
    if name == "exa_contents_text_true.json":
        assert len(raw["results"][0]["text"]) > 300  # uncapped text is longer
    text = json.dumps(fixture)
    assert "workHistory" not in text and "firstName" not in text


def test_cost_extraction_fails_closed_on_missing_or_malformed_billing():
    assert br.deepline_cost_microusd(b"not json") == 0
    assert br.deepline_cost_microusd(b'{"status": "completed"}') == 0
    assert br.deepline_cost_microusd(b'{"billing": {"cost_usd": "0.5"}}') == 0
    assert br.deepline_cost_microusd(b'{"billing": {"cost_usd": -1}}') == 0
    assert br.deepline_cost_microusd(b'{"billing": {"cost_usd": true}}') == 0
    assert br.deepline_cost_microusd(b'{"billing": {"cost_usd": 0.0123456}}') == 12345


def test_person_entities_are_scrubbed_except_for_the_people_search_tool():
    person = {"id": "https://exa.ai/library/person/x", "type": "person", "properties": {"name": "Jane Roe", "workHistory": [{"title": "CTO"}]}}
    company = {"id": "https://exa.ai/library/organization/y", "type": "company", "properties": {"name": "Acme"}}
    envelope = {"job_id": "j", "status": "completed", "result": {"data": {"requestId": "r", "results": [{"id": "u", "url": "u", "text": "t", "entities": [person, company]}]}}, "billing": {"cost_usd": 0.002}}
    body = json.dumps(envelope).encode("utf-8")
    for tool in ("exa_search", "exa_contents", "exa_company_search", "exa_answer"):
        _status, headers, sanitized = ops.sanitize_response("deepline.execute", 200, {}, body, parameters={"tool": tool, "payload": {}})
        entities = json.loads(sanitized)["result"]["data"]["results"][0]["entities"]
        assert entities == [company] and b"Jane Roe" not in sanitized and headers["content-length"] == str(len(sanitized))
    # No parameters at all is treated as an unknown tool: scrubbed.
    assert b"Jane Roe" not in ops.sanitize_response("deepline.execute", 200, {}, body)[2]
    # The people search tool returns people by design and is exempt.
    kept = json.loads(ops.sanitize_response("deepline.execute", 200, {}, body, parameters={"tool": "exa_people_search", "payload": {}})[2])
    assert kept["result"]["data"]["results"][0]["entities"] == [person, company]
    # The table binds the rule and its exemption into the round identity.
    document = next(item for item in ops.operation_table_document()["operations"] if item["operation_id"] == "deepline.execute")
    assert document["response_scrub"] == "deepline_person_entities" and document["response_scrub_exempt_tools"] == ["exa_people_search"]
    assert ops.scrub_person_entities({"entities": [person], "nested": [{"entities": [company, person]}]}) == {"entities": [], "nested": [{"entities": [company]}]}
