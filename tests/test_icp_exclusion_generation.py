"""Generator-side lab exclusion lists: real matching companies per ICP,
fail-open everywhere, hash-covered in the set."""

import os
from unittest import mock
import httpx

import gateway.tasks.icp_generator as gen


def _responses_reply(payload):
    return httpx.Response(200, json={"status": "completed", "output": [{
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": payload}],
    }]})


def test_entries_prefer_domain_and_validate():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "k"), \
         mock.patch.object(gen.httpx, "post", return_value=_responses_reply(
             '[{"name": "Acme Inc", "domain": "https://www.acme.com/x"},'
             ' {"name": "NoDomain Co", "domain": "not a domain"}]')):
        entries = gen.generate_exclusions_for_icp({"industry": "Software"}, count=2)
    assert entries == ["acme.com", "NoDomain Co"]


def test_fail_open_on_provider_error_and_bad_json():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "k"), \
         mock.patch.object(gen.httpx, "post", side_effect=RuntimeError("boom")):
        assert gen.generate_exclusions_for_icp({}, count=1) == []
    bad = _responses_reply("no json here")
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "k"), \
         mock.patch.object(gen.httpx, "post", return_value=bad):
        assert gen.generate_exclusions_for_icp({}, count=1) == []


def test_no_key_skips_without_call():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", ""):
        assert gen.generate_exclusions_for_icp({}, count=1) == []


def test_default_exclusions_use_only_approved_deepseek_responses_route():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "test-only"), \
         mock.patch.object(gen.httpx, "post", return_value=_responses_reply("[]")) as post:
        assert gen.generate_exclusions_for_icp({"industry": "Software"}, count=1) == []
    request = post.call_args.kwargs
    assert post.call_count == 1
    assert post.call_args.args[0] == "https://openrouter.ai/api/v1/responses"
    assert request["headers"]["X-OpenRouter-Metadata"] == "enabled"
    body = request["json"]
    assert body["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert body["provider"] == {
        "only": ["nebius/fp8"], "allow_fallbacks": False, "require_parameters": True,
        "data_collection": "deny", "zdr": True,
        "max_price": {"prompt": 0.30, "completion": 1.20},
    }
    assert body["reasoning"] == {"effort": "high", "context": "all_turns"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["max_output_tokens"] == 4096
    assert body["tools"] == [{"type": "openrouter:web_search", "parameters": {
        "engine": "exa", "max_uses": 3, "max_results": 5,
        "max_total_results": 10, "max_characters": 2000,
    }}]
    assert body["max_tool_calls"] == 3
    assert body["transforms"] == []
    assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert body["store"] is body["stream"] is False
    assert "temperature" not in body and "messages" not in body
    assert request["timeout"] == 45.0


def test_default_exclusion_failure_has_no_sonar_or_chat_fallback():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "test-only"), \
         mock.patch.object(gen.httpx, "post", return_value=httpx.Response(503, json={"error": "unavailable"})) as post:
        assert gen.generate_exclusions_for_icp({"industry": "Software"}) == []
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/responses")
    assert post.call_args.kwargs["json"]["model"] == "deepseek/deepseek-v4.1-flash-1017"


def test_icp_set_carries_exclusions_and_hash_covers_them():
    with mock.patch.object(gen, "generate_exclusions_for_icp",
                           side_effect=lambda icp, count=None, timeout_s=45.0: ["excluded.example"]):
        icps, _d, h1 = gen.generate_icp_set(20260716, base_seed=9)
    assert all(icp["excluded_companies"] == ["excluded.example"] for icp in icps)
    with mock.patch.object(gen, "generate_exclusions_for_icp",
                           side_effect=lambda icp, count=None, timeout_s=45.0: []):
        _i2, _d2, h2 = gen.generate_icp_set(20260716, base_seed=9)
    assert h1 != h2  # exclusions are part of the audited set hash


def test_disabled_env_yields_empty_lists():
    with mock.patch.dict(os.environ, {"LAB_ARENA_ICP_EXCLUSIONS_ENABLED": "0"}):
        icps, _d, _h = gen.generate_icp_set(20260717, base_seed=3)
    assert all(icp["excluded_companies"] == [] for icp in icps)
