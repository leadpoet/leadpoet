"""Round-frozen readable reasoning for agent model calls (agent_readable_reasoning_v1)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc, service as service_module
from lab_arena.service import ArenaService
from tests.lab_arena import test_lab_arena_broker as broker_tests
from tests.lab_arena.deepline_round_runtime_test import _service

# Shapes copied from OpenRouter's public /api/v1/models catalog (2026-10-19).
CATALOG = {"data": [
    {"id": "openai/gpt-6-luna", "supported_parameters": ["reasoning", "tools"],
     "reasoning": {"mandatory": False, "default_enabled": True, "default_effort": "medium",
                   "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"]}},
    {"id": "openai/gpt-5.4-mini", "supported_parameters": ["reasoning", "reasoning_effort"],
     "reasoning": {"mandatory": False, "default_enabled": False, "default_effort": "medium",
                   "supported_efforts": ["xhigh", "high", "medium", "low", "none"]}},
    {"id": "openai/gpt-5-mini", "supported_parameters": ["reasoning"],
     "reasoning": {"mandatory": True, "default_effort": "medium",
                   "supported_efforts": ["high", "medium", "low", "minimal"]}},
    {"id": "google/gemini-3.5-flash-lite", "supported_parameters": ["reasoning"],
     "reasoning": {"mandatory": True, "default_enabled": True, "default_effort": "minimal",
                   "supported_efforts": ["high", "medium", "low", "minimal"]}},
    {"id": "google/gemini-2.5-flash", "supported_parameters": ["reasoning", "temperature", "tools"],
     "reasoning": {"mandatory": False}},
    {"id": "anthropic/claude-sonnet-4.5", "supported_parameters": ["reasoning", "temperature"],
     "reasoning": {"mandatory": False}},
    {"id": "anthropic/claude-sonnet-5.5", "supported_parameters": ["reasoning", "tools"],
     "reasoning": {"mandatory": True, "default_effort": "high",
                   "supported_efforts": ["max", "xhigh", "high", "medium", "low"]}},
    {"id": "perplexity/sonar", "supported_parameters": ["temperature"]},
    {"id": "~alias/latest", "supported_parameters": ["reasoning"]},
]}
PROFILES = rc.profiles_from_models_response(CATALOG)
MESSAGES = [{"role": "user", "content": "Find fintech companies hiring revenue leaders."}]


TOOL_LOOP = [MESSAGES[0],
             {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                 "name": "lookup", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "call_1", "content": "{}"}]


def chat(model, **extra):
    return {"model": model, "messages": MESSAGES, "max_tokens": 1500, **extra}


def forced(operation_id, parameters, profiles=PROFILES):
    return rc.apply(operation_id, parameters, policy=rc.POLICY, profiles=profiles)


def test_catalog_profiles_keep_reasoning_models_and_validate():
    assert set(PROFILES) == {
        "anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-5.5", "google/gemini-2.5-flash",
        "google/gemini-3.5-flash-lite", "openai/gpt-5-mini", "openai/gpt-5.4-mini", "openai/gpt-6-luna",
    }
    assert PROFILES["google/gemini-2.5-flash"] == {"mandatory": False}
    assert PROFILES["openai/gpt-5.4-mini"]["default_enabled"] is False
    assert rc.validate_profiles(json.loads(json.dumps(PROFILES))) == PROFILES
    many = {"data": [{"id": "vendor/m%05d" % index, "supported_parameters": ["reasoning"]}
                     for index in range(rc.MAX_PROFILES + 5)]}
    assert len(rc.profiles_from_models_response(many)) == rc.MAX_PROFILES


@pytest.mark.parametrize("document", [
    {"bad id": {"mandatory": False}},
    {"vendor/model": {"mandatory": "no"}},
    {"vendor/model": {"mandatory": False, "extra": 1}},
    {"vendor/model": {"mandatory": False, "default_effort": "turbo"}},
    {"vendor/model": {"mandatory": False, "supported_efforts": ["high", "low"]}},
    {"vendor/model": {"mandatory": False, "supported_efforts": []}},
    [],
])
def test_invalid_profiles_are_refused(document):
    with pytest.raises(ValueError):
        rc.validate_profiles(document)


@pytest.mark.parametrize("model,extra,reasoning,max_tokens,changes", [
    # Reasons by default: its catalog default becomes explicit, plus a summary.
    ("openai/gpt-6-luna", {}, {"effort": "medium", "summary": "auto"}, 1500, ["effort", "summary"]),
    # Off by default: turned on at the floor with an output allowance.
    ("openai/gpt-5.4-mini", {}, {"effort": "low", "summary": "auto"}, 2524, ["effort", "headroom", "summary"]),
    # Mandatory with a minimal default: raised to the floor.
    ("google/gemini-3.5-flash-lite", {}, {"effort": "medium"}, 2524, ["effort", "headroom"]),
    # Budget-style thinking (no listed efforts, or optional Anthropic) gets a
    # hard 1,024-token budget, never an effort.
    ("google/gemini-2.5-flash", {"temperature": 0}, {"max_tokens": 1024}, 2524, ["budget", "headroom"]),
    ("anthropic/claude-sonnet-4.5", {}, {"max_tokens": 1024}, 2524, ["budget", "headroom"]),
    # Mandatory adaptive thinking: an explicit floor-level effort is kept.
    ("anthropic/claude-sonnet-5.5", {"reasoning": {"effort": "low"}}, {"effort": "low"}, 1500, []),
    ("openai/gpt-6-luna", {"reasoning": {"effort": "high"}}, {"effort": "high", "summary": "auto"}, 1500, ["summary"]),
    # Above the floor the agent's choice stands, even one the catalog omits.
    ("openai/gpt-5-mini", {"reasoning": {"effort": "xhigh"}}, {"effort": "xhigh", "summary": "auto"}, 1500, ["summary"]),
    ("openai/gpt-6-luna", {"reasoning": {"effort": "none"}}, {"effort": "low", "summary": "auto"}, 2524,
     ["agent_disabled_overridden", "effort", "headroom", "summary"]),
    ("openai/gpt-5-mini", {"reasoning_effort": "minimal", "include_reasoning": False},
     {"effort": "low", "exclude": False, "summary": "auto"}, 2524,
     ["effort", "exclude_cleared", "headroom", "legacy_effort", "legacy_include", "summary"]),
    ("openai/gpt-6-luna", {"reasoning": {"exclude": True, "effort": "medium"}},
     {"effort": "medium", "exclude": False, "summary": "auto"}, 1500, ["exclude_cleared", "summary"]),
    ("google/gemini-2.5-flash", {"reasoning": {"max_tokens": 2048}}, {"max_tokens": 2048}, 1500, []),
    # Reasoning the agent already turned on keeps the provider's default level.
    ("openai/gpt-5.4-mini", {"reasoning": {"enabled": True}}, {"enabled": True, "summary": "auto"}, 1500, ["summary"]),
    ("openai/gpt-5.4-mini", {"reasoning": {}}, {"effort": "low", "summary": "auto"}, 2524, ["effort", "headroom", "summary"]),
    ("openai/gpt-5.4-mini", {"include_reasoning": True}, {"enabled": True, "exclude": False, "summary": "auto"}, 1500,
     ["legacy_include", "summary"]),
    ("anthropic/claude-sonnet-4.5", {"reasoning": {"enabled": True}}, {"enabled": True}, 1500, []),
    ("openai/gpt-6-luna", {"reasoning": {"effort": "low", "max_tokens": 0}}, {"effort": "low", "summary": "auto"}, 2524,
     ["agent_disabled_overridden", "effort", "headroom", "summary"]),
])
def test_chat_policy_matrix(model, extra, reasoning, max_tokens, changes):
    parameters = chat(model, **extra)
    original = json.loads(json.dumps(parameters))
    updated, annotation = forced("openrouter.chat", parameters)
    assert parameters == original
    assert updated["reasoning"] == dict(reasoning, exclude=False)
    changes = sorted(changes + (["exclude"] if "exclude" not in reasoning else []))
    assert updated["max_tokens"] == max_tokens
    assert "reasoning_effort" not in updated and "include_reasoning" not in updated
    assert annotation["policy"] == rc.POLICY and annotation["requested"] is True
    assert annotation["changes"] == changes
    operations.validate_operation_request("openrouter.chat", updated)
    assert forced("openrouter.chat", parameters) == (updated, annotation)


@pytest.mark.parametrize("model,max_tokens,reasoning,changes", [
    # 4,000 is the most common agent limit; 4,096 is the cap and the default.
    ("google/gemini-2.5-flash", 4000, {"max_tokens": 1024}, ["budget", "headroom_capped"]),
    ("anthropic/claude-sonnet-4.5", 4096, {"max_tokens": 1024}, ["budget", "headroom_capped"]),
    ("google/gemini-3.5-flash-lite", 4000, {"effort": "medium"}, ["effort", "headroom_capped"]),
    ("openai/gpt-5.4-mini", 4096, {"effort": "low", "summary": "auto"}, ["effort", "headroom_capped", "summary"]),
])
def test_near_the_cap_reasoning_shares_the_limit_up_to_the_cap(model, max_tokens, reasoning, changes):
    updated, annotation = forced("openrouter.chat", chat(model, max_tokens=max_tokens))
    assert updated["max_tokens"] == operations.OPENROUTER_MAX_OUTPUT_TOKENS
    assert updated["reasoning"] == dict(reasoning, exclude=False)
    changes = sorted(changes + ["exclude"])
    assert annotation["requested"] is True and annotation["changes"] == changes and "skipped" not in annotation
    operations.validate_operation_request("openrouter.chat", updated)
    if "max_tokens" in reasoning:
        # A hard budget leaves the visible answer at least 3,072 tokens.
        assert updated["max_tokens"] - reasoning["max_tokens"] == 3072


def test_toggle_only_reasoning_is_added_only_with_its_full_allowance():
    # No listed efforts and no applied budget (for example DeepSeek V3.2):
    # reasoning runs at the model's own length, so it needs all 1,024 tokens.
    profiles = {"deepseek/deepseek-v3.2": {"mandatory": False}}
    updated, annotation = forced("openrouter.chat", chat("deepseek/deepseek-v3.2"), profiles)
    assert updated["reasoning"] == {"enabled": True, "exclude": False} and updated["max_tokens"] == 2524
    assert annotation["changes"] == ["enabled", "exclude", "headroom"]
    for max_tokens in (4000, 4096):
        parameters = chat("deepseek/deepseek-v3.2", max_tokens=max_tokens)
        assert forced("openrouter.chat", parameters, profiles) == (parameters, {
            "policy": rc.POLICY, "treatment": "chat", "requested": False, "changes": [],
            "skipped": "headroom_unavailable"})
    near_cap = {"model": "deepseek/deepseek-v3.2", "input": "Research.",
                "max_output_tokens": operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS - 100}
    assert forced("openrouter.responses", near_cap, profiles) == (near_cap, {
        "policy": rc.POLICY, "treatment": "responses", "requested": False, "changes": [],
        "skipped": "operation_reasoning_control_unavailable"})


def test_budget_models_take_the_budget_for_a_raised_effort_and_stay_below_their_limit():
    for model in ("google/gemini-2.5-flash", "anthropic/claude-sonnet-4.5"):
        updated, annotation = forced("openrouter.chat", chat(model, reasoning={"effort": "minimal"}))
        assert updated["reasoning"] == {"max_tokens": rc.BUDGET_TOKENS, "exclude": False}
        assert annotation["changes"] == ["budget", "exclude", "headroom"]
    # Anthropic requires the output limit to exceed the thinking budget.
    updated, _ = forced("openrouter.chat", chat("anthropic/claude-sonnet-4.5", max_tokens=1))
    assert updated["max_tokens"] == 1025 > updated["reasoning"]["max_tokens"]
    assert rc.HEADROOM_TOKENS >= rc.BUDGET_TOKENS


def test_near_the_responses_cap_reasoning_shares_the_limit_up_to_the_cap():
    parameters = {"model": "openai/gpt-5.4-mini", "input": "Research the company.",
                  "max_output_tokens": operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS - 100}
    updated, annotation = forced("openrouter.responses", parameters)
    assert updated["max_output_tokens"] == operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS
    assert annotation["changes"] == ["effort", "headroom_capped", "summary"]
    operations.validate_operation_request("openrouter.responses", updated)


def test_explicit_opt_outs_stay_off_where_thinking_state_must_round_trip():
    for model, reasoning in (("google/gemini-3.5-flash-lite", {"effort": "none"}),
                             ("google/gemini-2.5-flash", {"enabled": False}),
                             ("anthropic/claude-sonnet-4.5", {"enabled": False})):
        parameters = chat(model, messages=TOOL_LOOP, reasoning=reasoning)
        assert forced("openrouter.chat", parameters) == (parameters, {
            "policy": rc.POLICY, "treatment": "chat", "requested": False, "changes": [],
            "skipped": "thinking_tool_state_unavailable"})


def test_models_without_a_low_effort_are_not_forced_to_a_higher_one():
    profiles = {"mistralai/mistral-small": {"mandatory": False, "default_enabled": False,
                                            "supported_efforts": ["medium", "high"]}}
    parameters = chat("mistralai/mistral-small")
    assert forced("openrouter.chat", parameters, profiles) == (parameters, {
        "policy": rc.POLICY, "treatment": "chat", "requested": False, "changes": [], "skipped": "no_low_effort"})


@pytest.mark.parametrize("extra", [
    {"temperature": 0}, {"temperature": 0.2}, {"top_k": 5}, {"top_p": 0.5},
    {"tool_choice": "required"}, {"tool_choice": {"type": "function", "function": {"name": "lookup"}}},
    {"messages": TOOL_LOOP}, {"messages": MESSAGES + [{"role": "assistant", "content": "{"}]},
])
def test_anthropic_thinking_conflicts_send_the_agent_request(extra):
    parameters = chat("anthropic/claude-sonnet-4.5", **extra)
    updated, annotation = forced("openrouter.chat", parameters)
    assert updated == parameters
    assert annotation["requested"] is False and annotation["changes"] == []
    assert annotation["skipped"] in {"thinking_tool_state_unavailable", "anthropic_thinking_conflict"}
    # Mandatory thinking already runs; its effort is set as usual.
    mandatory, note = forced("openrouter.chat", chat("anthropic/claude-sonnet-5.5", **extra))
    assert note["treatment"] == "chat" and mandatory["reasoning"] == {"effort": "high", "exclude": False}


def test_models_without_reasoning_support_are_unchanged():
    for model in ("perplexity/sonar", "openai/gpt-4o-mini"):
        parameters = chat(model, temperature=0)
        assert forced("openrouter.chat", parameters) == (
            parameters, {"policy": rc.POLICY, "treatment": "no_reasoning_support", "requested": False, "changes": []})


def test_policy_absence_and_unknown_versions():
    parameters = chat("openai/gpt-6-luna")
    assert rc.apply("openrouter.chat", parameters, policy=None, profiles=PROFILES) == (parameters, None)
    assert rc.apply("deepline.execute", {"tool": "x"}, policy=rc.POLICY, profiles=PROFILES) == ({"tool": "x"}, None)
    with pytest.raises(ValueError):
        rc.apply("openrouter.chat", parameters, policy="agent_readable_reasoning_v2", profiles=PROFILES)


@pytest.mark.parametrize("reasoning,effort,max_output_tokens,changes", [
    ({"effort": "medium"}, "medium", 4096, ["summary"]),
    ({"effort": "minimal"}, "low", 5120, ["effort", "headroom", "summary"]),
    ({"effort": "high", "summary": "concise"}, "high", 4096, []),
    (None, "medium", 4096, ["effort", "summary"]),
])
def test_responses_policy_requests_summaries(reasoning, effort, max_output_tokens, changes):
    parameters = {"model": "openai/gpt-6-luna", "input": "Research the company.", "max_output_tokens": 4096,
                  **({"reasoning": reasoning} if reasoning else {})}
    updated, annotation = forced("openrouter.responses", parameters)
    assert updated["reasoning"]["effort"] == effort
    assert updated["reasoning"]["summary"] == (reasoning or {}).get("summary", "auto")
    assert updated["max_output_tokens"] == max_output_tokens
    assert annotation["changes"] == changes and annotation["requested"] is True
    operations.validate_operation_request("openrouter.responses", updated)


def test_optional_anthropic_thinking_is_never_switched_on_through_responses():
    for extra in ({}, {"tool_choice": "required"},
                  {"input": [{"type": "function_call_output", "call_id": "c1", "output": "{}"}]}):
        parameters = {"model": "anthropic/claude-sonnet-4.5", "input": "Research.", "max_output_tokens": 2000, **extra}
        assert forced("openrouter.responses", parameters) == (parameters, {
            "policy": rc.POLICY, "treatment": "responses", "requested": False, "changes": [],
            "skipped": "thinking_tool_state_unavailable" if extra.get("tool_choice") or isinstance(extra.get("input"), list) else "anthropic_thinking_conflict"})
    chosen = {"model": "anthropic/claude-sonnet-4.5", "input": "Research.", "max_output_tokens": 2000,
              "reasoning": {"effort": "low"}}
    updated, annotation = forced("openrouter.responses", chosen)
    assert updated["reasoning"] == {"effort": "low", "summary": "auto"} and annotation["changes"] == ["summary"]


# --- Gateway broker -------------------------------------------------------

AGENT_PROFILES = {"openai/gpt-4o-mini": {"mandatory": False, "default_enabled": False,
                                         "default_effort": "medium", "supported_efforts": ["low", "medium", "high"]}}
POLICY_CONTEXT = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY,
                         reasoning_model_profiles=AGENT_PROFILES)
OK = (200, {"id": "gen", "model": "openai/gpt-4o-mini", "choices": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 50, "cost": "0.000033"}})


def _execute(context, broker=None, action_sequence=0):
    broker, store, transport = broker or broker_tests.make_broker(
        transport=broker_tests.FakeTransport([OK, OK]), judge_models=(broker_tests.CHAT["model"],))
    result = broker.execute(context, operation_id="openrouter.chat", parameters=broker_tests.CHAT,
                            action_sequence=action_sequence, timeout_ms=30000)
    return result, store, transport, (broker, store, transport)


def test_agent_call_sends_and_binds_the_policy_request():
    result, store, transport, _ = _execute(POLICY_CONTEXT)
    assert result.status == 200
    body = json.loads(transport.sent[0]["body"])
    assert body["reasoning"] == {"effort": "low", "exclude": False, "summary": "auto"}
    assert body["max_tokens"] == broker_tests.CHAT["max_tokens"] + rc.HEADROOM_TOKENS
    expected = br.normalized_request("openrouter.chat", broker_tests.CHAT, reasoning_capture_policy=rc.POLICY,
                                     reasoning_model_profiles=AGENT_PROFILES)
    assert result.call["request_hash"] == contracts.document_hash(expected)
    call_doc = store.calls[result.call["call_identity"]]["call_doc"]
    assert call_doc["reasoning_capture"] == {"policy": rc.POLICY, "treatment": "chat", "requested": True,
                                             "changes": ["effort", "exclude", "headroom", "summary"], "effort": "low"}
    assert call_doc["max_output_tokens"] == expected["max_tokens"]


def test_retried_agent_call_keeps_one_identity_and_one_dispatch():
    first, _store, transport, broker = _execute(POLICY_CONTEXT)
    second, _, _, _ = _execute(POLICY_CONTEXT, broker=broker)
    assert second.call["call_identity"] == first.call["call_identity"]
    assert len(transport.sent) == 1


@pytest.mark.parametrize("context", [
    replace(POLICY_CONTEXT, kind="score"),
    broker_tests.CONTEXT,
], ids=["judge", "no-policy"])
def test_judge_runs_and_rounds_without_policy_are_unchanged(context):
    result, store, transport, _ = _execute(context)
    body = json.loads(transport.sent[0]["body"])
    assert "reasoning" not in body and body["max_tokens"] == broker_tests.CHAT["max_tokens"]
    assert "reasoning_capture" not in store.calls[result.call["call_identity"]]["call_doc"]
    assert result.call["request_hash"] == contracts.document_hash(
        br.normalized_request("openrouter.chat", broker_tests.CHAT))


def _near_size_cap(margin):
    parameters = {"model": "openai/gpt-4o-mini", "max_tokens": 200, "messages": []}

    def size():
        return len(contracts.canonical_json(parameters).encode("utf-8"))

    # Each full message adds 31,000 characters plus a little JSON; the final
    # filler then stays under the per-message content limit.
    while size() + 31_200 <= 1_000_000 - margin:
        parameters["messages"].append({"role": "user", "content": "x" * 31_000})
    parameters["messages"].append({"role": "user", "content": ""})
    parameters["messages"][-1]["content"] = "x" * (1_000_000 - margin - size())
    assert size() == 1_000_000 - margin
    return parameters


def test_policy_form_refused_by_the_schema_sends_the_agent_request():
    # A valid request just under the canonical size cap: the added reasoning
    # settings would push it over, so the agent's own form is sent instead.
    normalized = br.normalized_request("openrouter.chat", _near_size_cap(20))
    updated, annotation = br.agent_reasoning_parameters(
        "openrouter.chat", normalized, policy=rc.POLICY, profiles=AGENT_PROFILES)
    assert updated == normalized
    assert annotation["requested"] is False and annotation["unapplied"] == "operation_schema"


def test_price_table_carries_catalog_profiles_for_priced_models():
    catalog = {"data": [dict(item, pricing={"prompt": "0.000001", "completion": "0.000002"})
                        for item in CATALOG["data"] if item["id"] != "openai/gpt-5-mini"]}
    table = br.price_table_from_models_response(catalog, fetched_at="2026-10-19T00:00:00Z")
    assert "openai/gpt-5-mini" not in table["reasoning_profiles"]
    assert table["reasoning_profiles"]["openai/gpt-6-luna"] == PROFILES["openai/gpt-6-luna"]
    assert br.validate_price_table(json.loads(json.dumps(table))) == table
    with pytest.raises(contracts.ArenaContractError):
        br.validate_price_table(dict(table, reasoning_profiles={"bad": {}}))


# --- Round configuration --------------------------------------------------

def _activated(activation="2026-10-01T00:00:00+00:00", profiles=PROFILES):
    service = _service()
    service._config.reasoning_profile_source = (lambda: profiles) if profiles is not None else None
    service._config.defaults.reasoning_capture_from = activation
    return service


def test_new_round_freezes_the_policy_and_catalog_profiles():
    config = _activated().create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    assert config["reasoning_capture_policy"] == rc.POLICY
    assert config["reasoning_model_profiles"] == PROFILES
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    assert ArenaService._public_round_output_policy(config)["reasoning_capture_policy"] == rc.POLICY


def test_rounds_opened_before_activation_keep_their_requests():
    config = _activated("2026-11-01T00:00:00Z").create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    assert "reasoning_capture_policy" not in config and "reasoning_model_profiles" not in config
    assert "reasoning_capture_policy" not in ArenaService._public_round_output_policy(config)


@pytest.mark.parametrize("profiles", [None, {}, {"bad": {}}])
def test_activated_round_refuses_missing_or_invalid_profiles(profiles):
    service = _activated(profiles=profiles)
    created = []
    service._store.create_round = lambda *args: created.append(args)
    with pytest.raises(service_module.ServiceError, match="reasoning_capture_profiles_unavailable") as error:
        service.create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    assert error.value.status == 503
    assert created == []


@pytest.mark.parametrize("change,message", [
    ({"reasoning_model_profiles": None}, "invalid reasoning capture policy"),
    ({"reasoning_model_profiles": {}}, "invalid reasoning capture policy"),
    ({"reasoning_capture_policy": None}, "invalid reasoning capture policy"),
    ({"reasoning_capture_policy": "agent_readable_reasoning_v2"}, "agent_readable_reasoning_v1|choices|invalid"),
])
def test_contract_refuses_partial_or_unknown_policy(change, message):
    config = _activated().create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    with pytest.raises(contracts.ArenaContractError, match=message):
        contracts.validate_round_configuration(dict(config, **change))
    without_profiles = {key: value for key, value in config.items() if key != "reasoning_model_profiles"}
    with pytest.raises(contracts.ArenaContractError, match="frozen together"):
        contracts.validate_round_configuration(without_profiles)


def test_run_context_carries_the_frozen_policy_to_the_broker():
    config = _activated().create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    service = object.__new__(ArenaService)
    run = {"run_id": "r1", "round_id": config["round_id"], "assignment_id": "a1", "attempt": 1,
           "icp_position": 0, "miner_hotkey": "5" * 48, "submission_id": "s1", "stage": 1, "kind": "execute"}
    service._store = SimpleNamespace(get_run=lambda _run_id: run)
    service._require_round_ownership = lambda _round_id: None
    service._hot_round = lambda _round_id: {"round_id": config["round_id"], "configuration_doc": config}
    _run, context = service._run_context("r1", "lease-token")
    assert context.reasoning_capture_policy == rc.POLICY
    assert context.reasoning_model_profiles == PROFILES


def test_catalog_keeps_null_omitted_and_token_budget_capabilities_separate():
    catalog = {"data": [
        {"id": "vendor/all", "reasoning": {"supported_efforts": None}},
        {"id": "vendor/toggle", "reasoning": {}},
        {"id": "vendor/budget", "reasoning": {"supports_max_tokens": True}},
        {"id": "vendor/no-budget", "reasoning": {"supports_max_tokens": False}},
    ]}
    profiles = rc.profiles_from_models_response(catalog)
    assert profiles["vendor/all"]["supported_efforts"] is None
    assert "supported_efforts" not in profiles["vendor/toggle"]
    assert profiles["vendor/budget"]["supports_max_tokens"] is True
    assert profiles["vendor/no-budget"]["supports_max_tokens"] is False
    assert rc.validate_profiles(json.loads(json.dumps(profiles))) == profiles
    assert forced("openrouter.chat", chat("vendor/all"), profiles)[0]["reasoning"] == {"effort": "low", "exclude": False}
    assert forced("openrouter.chat", chat("vendor/budget"), profiles)[0]["reasoning"] == {"max_tokens": 1024, "exclude": False}
    source = {"vendor/all": {"mandatory": False, "supported_efforts": ["low", "high"]}}
    frozen = rc.validate_profiles(source)
    source["vendor/all"]["supported_efforts"].clear()
    assert frozen["vendor/all"]["supported_efforts"] == ["low", "high"]
    for invalid in ("true", 1, None):
        with pytest.raises(ValueError):
            rc.validate_profiles({"vendor/m": {"mandatory": False, "supports_max_tokens": invalid}})


@pytest.mark.parametrize("model", ["google/gemini-2.5-flash", "google/gemini-3.5-flash-lite", "anthropic/claude-sonnet-4.5"])
@pytest.mark.parametrize("reasoning", [{"enabled": False}, {"effort": "none"}, {"max_tokens": 0}])
def test_explicit_disabled_reasoning_is_enabled_only_without_tool_state(model, reasoning):
    updated, annotation = forced("openrouter.chat", chat(model, reasoning=reasoning))
    assert "agent_disabled_overridden" in annotation["changes"] and annotation["requested"] is True
    expected = {"effort": "medium"} if model == "google/gemini-3.5-flash-lite" else {"max_tokens": 1024}
    assert updated["reasoning"] == dict(expected, exclude=False)


@pytest.mark.parametrize("model", ["google/gemini-2.5-flash", "google/gemini-3.5-flash-lite", "anthropic/claude-sonnet-4.5"])
@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]},
    {"messages": TOOL_LOOP},
])
def test_first_tool_call_and_continuation_do_not_gain_thinking(model, extra):
    parameters = chat(model, **extra)
    updated, annotation = forced("openrouter.chat", parameters)
    if model == "google/gemini-3.5-flash-lite":
        assert updated == dict(parameters, reasoning={"exclude": False})
    else:
        assert updated == parameters
    assert annotation["skipped"] == "thinking_tool_state_unavailable"
    if model == "google/gemini-3.5-flash-lite":
        assert annotation["requested"] is True  # Existing mandatory minimal thinking remains.
    else:
        assert annotation["requested"] is False
    # Compatible thinking already requested by the agent stays on; make only
    # returned text visible, without reconstructing or changing signed state.
    parameters["reasoning"] = {"max_tokens": 2048, "exclude": True}
    updated, annotation = forced("openrouter.chat", parameters)
    assert updated["reasoning"] == {"max_tokens": 2048, "exclude": False}
    assert updated["max_tokens"] == parameters["max_tokens"]
    assert annotation["changes"] == ["exclude_cleared"]


def test_responses_continuation_protection_sees_real_function_item_shapes():
    for item in ({"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"},
                 {"type": "function_call_output", "call_id": "c", "output": "{}"}):
        parameters = {"model": "google/gemini-2.5-flash", "input": [item], "max_output_tokens": 2000}
        updated, annotation = forced("openrouter.responses", parameters)
        assert updated == parameters
        assert annotation["skipped"] == "thinking_tool_state_unavailable" and not annotation["requested"]


def test_judge_and_historical_outbound_bytes_match_the_original_operation():
    expected = operations.build_outbound_request("openrouter.chat", br.normalized_request("openrouter.chat", broker_tests.CHAT)).body
    for context in (broker_tests.CONTEXT, replace(POLICY_CONTEXT, kind="score")):
        result, store, transport, _ = _execute(context)
        assert result.status == 200 and transport.sent[0]["body"] == expected
        assert "reasoning_capture" not in store.calls[result.call["call_identity"]]["call_doc"]


def test_capture_admission_and_dispatch_bind_the_same_policy_request():
    captures = []
    instance, store, transport = broker_tests.make_broker(
        transport=broker_tests.FakeTransport([OK]),
        capture_request=lambda document, **_kwargs: captures.append(document) or {"status": "unavailable"},
    )
    result, _, _, _ = _execute(POLICY_CONTEXT, (instance, store, transport))
    sent = json.loads(transport.sent[0]["body"])
    expected = br.normalized_request("openrouter.chat", broker_tests.CHAT,
        reasoning_capture_policy=rc.POLICY, reasoning_model_profiles=AGENT_PROFILES)
    entry = store.calls[result.call["call_identity"]]
    assert captures[0]["hashed_parameters"] == captures[0]["effective_parameters"] == expected
    assert captures[0]["outbound"]["body"] == sent
    assert captures[0]["request_hash"] == result.call["request_hash"] == contracts.document_hash(expected)
    assert entry["amount"] == br.max_openrouter_cost_microusd(
        broker_tests.price_table(), expected["model"], expected, max_output_tokens=expected["max_tokens"])


def test_retry_uses_frozen_context_after_runtime_catalog_changes():
    first, _, transport, instance = _execute(POLICY_CONTEXT)
    instance[0]._price_table["reasoning_profiles"] = {broker_tests.CHAT["model"]: {"mandatory": False}}
    second, _, _, _ = _execute(POLICY_CONTEXT, instance)
    assert second.call["call_identity"] == first.call["call_identity"]
    assert second.call["request_hash"] == first.call["request_hash"]
    assert len(transport.sent) == 1


def test_freeze_uses_submission_open_and_does_not_share_runtime_metadata():
    service = _activated()
    plain = _service().create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    submission_open = service_module._parse_iso(plain["schedule"]["submission_open"])
    service._config.defaults.reasoning_capture_from = submission_open.isoformat()
    source = json.loads(json.dumps(PROFILES))
    service._config.reasoning_profile_source = lambda: source
    config = service.create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    source["openai/gpt-6-luna"]["supported_efforts"].clear()
    assert config["reasoning_model_profiles"] == PROFILES
    later = _activated((submission_open + service_module.timedelta(seconds=1)).isoformat())
    config = later.create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    assert "reasoning_capture_policy" not in config


@pytest.mark.parametrize("reasoning", [{}, {"exclude": True}, {"summary": "auto"}, {"context": "current_turn"}])
@pytest.mark.parametrize("model", ["openai/gpt-5.4-mini", "google/gemini-3.5-flash-lite"])
def test_visibility_only_reasoning_options_do_not_claim_thinking_is_on(model, reasoning):
    updated, annotation = forced("openrouter.chat", chat(model, reasoning=reasoning))
    assert updated["reasoning"]["effort"] == ("medium" if model.startswith("google/") else "low")
    assert updated["reasoning"]["exclude"] is False
    assert annotation["requested"] is True
    assert "effort" in annotation["changes"] and "headroom" in annotation["changes"]


def test_annotation_does_not_copy_opaque_agent_effort_values():
    for effort in ("arbitrary-private-text", {"text": "private"}, ["private"]):
        _, annotation = forced("openrouter.chat", chat("openai/gpt-6-luna", reasoning={"effort": effort}))
        assert "effort" not in annotation


def test_added_reasoning_output_room_is_included_before_cost_admission():
    original = br.normalized_request("openrouter.chat", broker_tests.CHAT)
    original_bound = br.max_openrouter_cost_microusd(broker_tests.price_table(), original["model"],
        original, max_output_tokens=original["max_tokens"])
    store = broker_tests.FakeLedgerStore(openrouter_capacity=original_bound)
    instance, _, transport = broker_tests.make_broker(store=store, transport=broker_tests.FakeTransport([OK]))
    result, _, _, _ = _execute(POLICY_CONTEXT, (instance, store, transport))
    assert result.status == 402 and result.call["error_code"] == "budget_refused"
    assert transport.sent == []


@pytest.mark.parametrize("effort,expected", [
    ("none", "medium"), ("minimal", "medium"), ("low", "medium"),
    ("medium", "medium"), ("high", "high"),
])
@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_google_effort_floor_uses_the_measured_readable_level(operation, effort, expected):
    model = "google/gemini-3.5-flash-lite"
    if operation == "openrouter.chat":
        parameters = chat(model, reasoning={"effort": effort})
    else:
        parameters = {"model": model, "input": "Compare current expansion evidence.",
                      "max_output_tokens": 1500, "reasoning": {"effort": effort}}
    updated, annotation = br.agent_reasoning_parameters(operation, parameters,
        policy=rc.POLICY, profiles=PROFILES)
    assert updated["reasoning"]["effort"] == expected
    assert annotation["effort"] == expected and annotation["requested"] is True
    assert ("headroom" in annotation["changes"]) == (effort in {"none", "minimal", "low"})
    if operation == "openrouter.chat":
        assert updated["reasoning"]["exclude"] is False
    else:
        assert updated["reasoning"]["summary"] == "auto"
    operations.validate_operation_request(operation, updated)


@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]},
    {"messages": TOOL_LOOP},
])
def test_google_low_effort_is_not_raised_inside_tool_workflows(extra):
    parameters = chat("google/gemini-3.5-flash-lite", reasoning={"effort": "low"}, **extra)
    updated, annotation = br.agent_reasoning_parameters("openrouter.chat", parameters,
        policy=rc.POLICY, profiles=PROFILES)
    assert updated["reasoning"] == {"effort": "low", "exclude": False}
    assert updated["max_tokens"] == parameters["max_tokens"]
    assert annotation["skipped"] == "thinking_tool_state_unavailable"
    assert annotation["changes"] == ["exclude"]
