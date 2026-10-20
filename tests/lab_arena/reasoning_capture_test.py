"""Round-frozen readable reasoning for agent model calls (agent_reasoning_v1)."""

from __future__ import annotations

import base64
import json
import logging
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
    ("google/gemini-3.5-flash-lite", {}, {"effort": "low"}, 2524, ["effort", "headroom"]),
    ("google/gemini-2.5-flash", {"temperature": 0}, {"effort": "low"}, 2524, ["effort", "headroom"]),
    # Optional Anthropic thinking uses its minimum budget, never an effort.
    ("anthropic/claude-sonnet-4.5", {}, {"max_tokens": 1024}, 2524, ["budget", "headroom"]),
    # Mandatory adaptive thinking: an explicit floor-level effort is kept.
    ("anthropic/claude-sonnet-5.5", {"reasoning": {"effort": "low"}}, {"effort": "low"}, 1500, []),
    ("openai/gpt-6-luna", {"reasoning": {"effort": "high"}}, {"effort": "high", "summary": "auto"}, 1500, ["summary"]),
    # Above the floor the agent's choice stands, even one the catalog omits.
    ("openai/gpt-5-mini", {"reasoning": {"effort": "xhigh"}}, {"effort": "xhigh", "summary": "auto"}, 1500, ["summary"]),
    ("openai/gpt-6-luna", {"reasoning": {"effort": "none"}}, {"effort": "low", "summary": "auto"}, 2524,
     ["effort", "headroom", "summary"]),
    ("openai/gpt-5-mini", {"reasoning_effort": "minimal", "include_reasoning": False},
     {"effort": "low", "summary": "auto"}, 2524, ["effort", "headroom", "legacy_effort", "legacy_include", "summary"]),
    ("openai/gpt-6-luna", {"reasoning": {"exclude": True, "effort": "medium"}},
     {"effort": "medium", "exclude": False, "summary": "auto"}, 1500, ["exclude_cleared", "summary"]),
    ("google/gemini-2.5-flash", {"reasoning": {"enabled": False}}, {"effort": "low"}, 2524,
     ["effort", "enabled", "headroom"]),
    ("google/gemini-2.5-flash", {"reasoning": {"max_tokens": 2048}}, {"max_tokens": 2048}, 1500, []),
])
def test_chat_policy_matrix(model, extra, reasoning, max_tokens, changes):
    parameters = chat(model, **extra)
    original = json.loads(json.dumps(parameters))
    updated, annotation = forced("openrouter.chat", parameters)
    assert parameters == original
    assert updated["reasoning"] == reasoning
    assert updated["max_tokens"] == max_tokens
    assert "reasoning_effort" not in updated and "include_reasoning" not in updated
    assert annotation["policy"] == rc.POLICY and annotation["requested"] is True
    assert annotation["changes"] == changes
    operations.validate_operation_request("openrouter.chat", updated)
    assert forced("openrouter.chat", parameters) == (updated, annotation)


def test_headroom_never_exceeds_the_chat_output_cap():
    updated, annotation = forced("openrouter.chat", chat("google/gemini-3.5-flash-lite", max_tokens=4096))
    assert updated["max_tokens"] == operations.OPENROUTER_MAX_OUTPUT_TOKENS
    assert annotation["changes"] == ["effort"]


@pytest.mark.parametrize("extra", [
    {"temperature": 0}, {"temperature": 0.2}, {"top_k": 5}, {"top_p": 0.5},
    {"tool_choice": "required"}, {"tool_choice": {"type": "function", "function": {"name": "lookup"}}},
])
def test_anthropic_thinking_conflicts_send_the_agent_request(extra):
    parameters = chat("anthropic/claude-sonnet-4.5", **extra)
    updated, annotation = forced("openrouter.chat", parameters)
    assert updated == parameters
    assert annotation == {"policy": rc.POLICY, "treatment": "anthropic_sampling_conflict",
                          "requested": False, "changes": []}


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
        rc.apply("openrouter.chat", parameters, policy="agent_reasoning_v2", profiles=PROFILES)


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


def test_readable_reasoning_is_text_the_provider_returned():
    chat_body = {"choices": [{"message": {
        "content": "answer", "reasoning": "Checked the careers page first.",
        "reasoning_details": [
            {"type": "reasoning.summary", "summary": "Checked the careers page first.", "format": "x", "index": 0},
            {"type": "reasoning.text", "text": "Then the press release.", "signature": "sig"},
            {"type": "reasoning.text", "signature": "gemini-only", "format": "google", "index": 2},
            {"type": "reasoning.encrypted", "data": "opaque", "format": "x"},
        ],
    }}]}
    assert rc.readable_reasoning(chat_body) == ["Checked the careers page first.", "Then the press release."]
    responses_body = {"output": [
        {"type": "reasoning", "encrypted_content": "opaque", "summary": [{"type": "summary_text", "text": "Plan."}],
         "content": [{"type": "reasoning_text", "text": "Detail."}]},
        {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "not reasoning"}]},
    ]}
    assert rc.readable_reasoning(responses_body) == ["Plan.", "Detail."]
    assert rc.readable_reasoning({"choices": [{"message": {"reasoning": None, "content": "x"}}]}) == []


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
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert body["max_tokens"] == broker_tests.CHAT["max_tokens"] + rc.HEADROOM_TOKENS
    expected = br.normalized_request("openrouter.chat", broker_tests.CHAT, reasoning_capture_policy=rc.POLICY,
                                     reasoning_model_profiles=AGENT_PROFILES)
    assert result.call["request_hash"] == contracts.document_hash(expected)
    call_doc = store.calls[result.call["call_identity"]]["call_doc"]
    assert call_doc["reasoning_capture"] == {"policy": rc.POLICY, "treatment": "chat", "requested": True,
                                             "changes": ["effort", "headroom", "summary"], "effort": "low"}
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


# --- Coverage check --------------------------------------------------------

def _ledger_rows(run_id, calls):
    rows = []
    for index, (model, annotation, body, succeeded) in enumerate(calls):
        identity = "%s:%d" % (run_id, index)
        entry_doc = {"model": model, **({"reasoning_capture": annotation} if annotation else {})}
        rows.append({"entry_id": 2 * index, "entry_kind": "reservation", "call_identity": identity,
                     "entry_doc": entry_doc})
        raw = json.dumps(body).encode("utf-8")
        rows.append({"entry_id": 2 * index + 1, "entry_kind": "settlement", "call_identity": identity,
                     "terminal_response": {"call_succeeded": succeeded,
                                           "body_b64": base64.b64encode(raw).decode("ascii")}})
    return rows


def test_coverage_counts_requested_calls_that_returned_text():
    from scripts import arena_reasoning_coverage as coverage

    asked = {"policy": rc.POLICY, "treatment": "chat", "requested": True, "changes": ["effort"]}
    skipped = {"policy": rc.POLICY, "treatment": "no_reasoning_support", "requested": False, "changes": []}
    with_text = {"choices": [{"message": {"content": "a", "reasoning": "Checked hiring pages."}}]}
    without = {"choices": [{"message": {"content": "a", "reasoning": None}}]}
    ledgers = {
        "run-1": _ledger_rows("run-1", [("openai/gpt-6-luna", asked, with_text, True),
                                        ("openai/gpt-6-luna", asked, without, True),
                                        ("perplexity/sonar", skipped, without, True),
                                        ("openai/gpt-6-luna", asked, with_text, False)]),
        "run-2": _ledger_rows("run-2", [("google/gemini-3.5-flash-lite", asked, with_text, True)]),
    }

    class Store:
        def get_round(self, round_id):
            return {"round_id": round_id, "configuration_doc": {"reasoning_capture_policy": rc.POLICY}}

        def list_runs(self, round_id, kind):
            assert kind == "execute"
            return [{"run_id": run_id} for run_id in ledgers]

        def list_ledger(self, run_id, provider, after_entry_id, limit):
            assert provider == "openrouter"
            return [row for row in ledgers[run_id] if after_entry_id is None or row["entry_id"] > after_entry_id]

    result = coverage.measure(Store(), "arena-2026-10-23", min_call_coverage=0.6)
    assert (result["calls"], result["requested_calls"], result["readable_requested_calls"]) == (4, 3, 2)
    assert result["call_coverage"] == round(2 / 3, 4) and result["passed"] is True
    assert result["runs_with_readable_reasoning"] == 2
    assert result["treatments"] == {"chat": 3, "no_reasoning_support": 1}
    assert coverage.measure(Store(), "arena-2026-10-23", min_call_coverage=0.9)["passed"] is False


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


def test_missing_catalog_profiles_never_block_round_creation(caplog):
    with caplog.at_level(logging.ERROR, logger=service_module.__name__):
        config = _activated(profiles=None).create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    assert "reasoning_capture_policy" not in config
    assert "reasoning capture profiles unavailable" in caplog.text


@pytest.mark.parametrize("change,message", [
    ({"reasoning_model_profiles": None}, "invalid reasoning capture policy"),
    ({"reasoning_model_profiles": {}}, "invalid reasoning capture policy"),
    ({"reasoning_capture_policy": None}, "invalid reasoning capture policy"),
    ({"reasoning_capture_policy": "agent_reasoning_v2"}, "agent_reasoning_v1|choices|invalid"),
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
