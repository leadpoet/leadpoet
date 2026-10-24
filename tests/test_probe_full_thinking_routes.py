"""Probe tests are offline; provider calls are always explicit operator actions."""
import copy
import json
import pytest

from scripts import probe_full_thinking_routes as probe


def test_probe_preserves_thinking_and_two_tool_rounds_before_strict_json():
    requests = []
    expected = []

    def respond(body, key):
        assert key == "test-only"
        requests.append(copy.deepcopy(body))
        i = len(requests) - 1
        assert body["provider"] == {"only": ["test-provider"], "allow_fallbacks": False,
                                    "zdr": True, "data_collection": "deny", "require_parameters": True}
        assert body["reasoning"] == {"effort": "high", "exclude": False}
        for previous in expected:
            assert previous in body["messages"]
        message = {"role": "assistant", "content": None,
                   "reasoning_details": [{"type": "reasoning.text", "text": "generated test text", "index": 0}],
                   "reasoning": "generated test text"}
        if i < 2:
            message["tool_calls"] = [{"id": "call-%d" % i, "index": 0, "type": "function",
                                      "function": {"name": "lookup_evidence", "arguments": json.dumps({"evidence_id": ("profile", "headcount")[i]})}}]
            expected.append(copy.deepcopy(message))
        else:
            assert body["response_format"]["json_schema"]["strict"] is True
            assert body["response_format"]["json_schema"]["schema"] == probe.SCHEMA
            message["content"] = json.dumps({"company": "Acme Test", "qualified": True,
                                              "evidence_ids": ["profile", "headcount"]})
        return {"status": 200, "request": copy.deepcopy(body), "response": {
            "choices": [{"message": message, "finish_reason": "tool_calls" if i < 2 else "stop"}],
            "usage": {"cost": 0.001}}}

    result = probe.probe(model="test/model", provider="test-provider", key="test-only", request_fn=respond)
    assert all(result["checks"].values())
    assert result["reported_cost_usd"] == "0.003"
    assert result["approval_granted"] is False
    assert "test-only" not in json.dumps(result)


def test_probe_does_not_retry_or_switch_route_after_provider_error():
    seen = []

    def respond(body, key):
        seen.append(body)
        return {"status": 429, "request": body, "error": "rate limited"}

    result = probe.probe(model="test/model", provider="test-provider", key="unused", request_fn=respond)
    assert len(seen) == 1
    assert not any(result["checks"].values())
    assert result["approval_granted"] is False


def test_probe_rejects_wrong_final_json_even_when_thinking_present():
    def respond(body, key):
        return {"status": 200, "request": body, "response": {"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "reasoning": "some text", "content": "{}"}}]}}

    result = probe.probe(model="test/model", provider="test-provider", key="unused", request_fn=respond)
    assert result["checks"]["returned_text"]
    assert not result["checks"]["two_tool_continuations"]
    assert not result["checks"]["final_json"]
    assert result["attempts"][0]["failure"] == "expected_one_tool_call"


def test_probe_price_bound_is_sent_without_weakening_privacy_or_route():
    price = {"prompt": 0.3, "completion": 1.2, "request": 0}

    def respond(body, key):
        assert body["provider"]["max_price"] == price
        assert body["provider"]["only"] == ["tested-provider"]
        assert body["provider"]["require_parameters"] is True
        assert body["provider"]["zdr"] is True
        assert body["provider"]["allow_fallbacks"] is False
        return {"status": 429, "request": body}

    probe.probe(model="test/model", provider="tested-provider", key="unused",
                max_price=price, request_fn=respond)


@pytest.mark.parametrize("price", [{}, {"prompt": -1, "completion": 1, "request": 0},
                                  {"prompt": float("nan"), "completion": 1, "request": 0}])
def test_probe_rejects_invalid_price_bound_before_network(price):
    def forbidden(*args):
        pytest.fail("invalid price must not reach provider")

    with pytest.raises(ValueError, match="invalid_probe_price_limit"):
        probe.probe(model="test/model", provider="test", key="unused",
                    max_price=price, request_fn=forbidden)
