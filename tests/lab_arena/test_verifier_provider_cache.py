"""Exact scorer provider replay through the existing Arena call ledger."""

from __future__ import annotations

import json
import base64
import copy
import threading
from dataclasses import replace

import pytest

from lab_arena import broker, contracts, scoring
from tests.lab_arena.test_lab_arena_broker import (
    FakeLedgerStore,
    FakeTransport,
    make_broker,
)


_MODEL = "openai/gpt-4o-mini"
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict"],
    "properties": {
        "verdict": {"type": "string", "enum": ["verified", "contradicted"]},
    },
}
_REQUEST = {
    "model": _MODEL,
    "messages": [
        {"role": "system", "content": "Verify this one claim."},
        {"role": "user", "content": "Storika launched the named product."},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "strict": True, "schema": _SCHEMA},
    },
    "max_tokens": 200,
}
_POLICY = scoring.build_scorer_policy(judge_models={"intent_signal_judge": _MODEL})
_FROZEN = {
    "round_id": "arena-2026-10-02",
    "evaluation_date": "2026-10-02",
    "scorer_image_digest": "sha256:" + "a" * 64,
    "scorer_image_reference": "example.invalid/judge@sha256:" + "a" * 64,
    "scorer_policy": _POLICY,
}


def _context(label: str, *, scope=None, kind="score"):
    return broker.RunContext(
        run_id=f"score-{label}",
        assignment_id=f"arena-2026-10-02:{label}:1:0",
        icp_position=0,
        lease_token_hash=contracts.document_hash(f"lease-{label}"),
        miner_hotkey=f"hotkey-{label}",
        submission_id=f"submission-{label}",
        stage=1,
        kind=kind,
        round_id=_FROZEN["round_id"],
        judgment_cache_scope=_FROZEN if scope is None else scope,
    )


def _answer(verdict="verified", *, cost="0.000021"):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({"verdict": verdict})},
        }],
        **({"usage": {"cost": cost}} if cost is not None else {}),
    }


class JudgmentLedger(FakeLedgerStore):
    """Exercise broker behavior against the agreed atomic RPC response shape."""

    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.judgment_reserves = []
        self.busy_once = False

    def _view(self, call):
        state = super()._view(call)
        if call["kind"] == "uncertain":
            state["terminal_response"] = call.get("uncertain_doc", {}).get(
                "judgment_cache_response"
            )
        return state

    def reserve_judgment_call(self, **arguments):
        with self.lock:
            self.judgment_reserves.append(dict(arguments))
            identity = arguments["call_identity"]
            key = arguments["call_doc"]["judgment_cache_key"]
            if identity in self.calls:
                return self._view(self.calls[identity])
            if arguments["call_doc"].get("judgment_cache_own_call_only") is True:
                return super().reserve_call(**arguments)
            if self.busy_once:
                self.busy_once = False
                return {"status": "cache_busy", "judgment_cache_key": key}
            for source in self.calls.values():
                if source["call_doc"].get("judgment_cache_own_call_only") is True:
                    continue
                if source["call_doc"].get("judgment_cache_key") != key:
                    continue
                terminal = (
                    source.get("terminal")
                    if source["kind"] == "settlement" else
                    source.get("uncertain_doc", {}).get("judgment_cache_response")
                )
                if isinstance(terminal, dict) and terminal.get("judgment_cache_eligible") is True:
                    if arguments["call_doc"].get("judgment_cache_single_tool_required") is True:
                        body = json.loads(base64.b64decode(terminal["body_b64"]))
                        if any(len((choice.get("message") or {}).get("tool_calls") or []) > 1
                               for choice in body.get("choices") or []):
                            continue
                    copied = {
                        **terminal,
                        "judgment_cache_key": key,
                        "judgment_cache_source_call_identity": source["identity"],
                        "judgment_cache_source_run_id": source["run_id"],
                    }
                    self.calls[identity] = {
                        "identity": identity, "kind": "settlement",
                        "amount": 0, "actual": 0,
                        "run_id": arguments["run_id"],
                        "operation_id": arguments["operation_id"],
                        "provider": arguments["provider"],
                        "funding_source": arguments["funding_source"],
                        "call_doc": {
                            **arguments["call_doc"],
                            "judgment_cache_source_call_identity": source["identity"],
                            "judgment_cache_source_run_id": source["run_id"],
                        },
                        "terminal": copied,
                    }
                    return {
                        "status": "cache_hit", "call_identity": identity,
                        "judgment_cache_key": key, "amount_microusd": 0,
                        "actual_microusd": 0,
                        "source_call_identity": source["identity"],
                        "source_run_id": source["run_id"],
                        "terminal_response": copied,
                    }
                if source["kind"] in {"reservation", "dispatch"}:
                    return {"status": "cache_busy", "judgment_cache_key": key}
            return super().reserve_call(**arguments)


def _broker(store, responses, *, funding_source_for=None, credential_for=None):
    return make_broker(
        store=store,
        transport=FakeTransport(responses),
        judge_models=(_MODEL,),
        **({"provider_funding_source_for": funding_source_for} if funding_source_for else {}),
        **({"credential_for": credential_for} if credential_for else {}),
    )


def _call(instance, context, request=_REQUEST, *, action_sequence=1):
    return instance.execute(
        context, operation_id="openrouter.chat", parameters=request,
        action_sequence=action_sequence, timeout_ms=5000,
    )


def test_exact_verdict_reuses_across_host_and_miner_credentials():
    store = JudgmentLedger()
    owner = {"score-baseline": "host", "score-miner": "miner_key"}
    creds = {"score-baseline": "host-key-a", "score-miner": "miner-key-b"}
    instance, _, transport = _broker(
        store, [(200, _answer())],
        funding_source_for=lambda context, _provider: owner[context.run_id],
        credential_for=lambda context, _provider: creds[context.run_id],
    )
    first = _call(instance, _context("baseline"))
    second = _call(instance, _context("miner"))
    again = _call(instance, _context("miner"))

    assert first.status == second.status == again.status == 200
    assert len(transport.sent) == 1
    assert first.call["actual_microusd"] == 21
    assert first.call.get("cached") is None
    assert second.call["cached"] is True
    assert second.call["actual_microusd"] == 0
    assert second.call["source_call_identity"] == first.call["call_identity"]
    assert again.call["idempotent"] is True
    assert again.call["cached"] is True
    assert again.call["actual_microusd"] == 0
    assert store.judgment_reserves[0]["call_doc"]["judgment_cache_key"] == (
        store.judgment_reserves[1]["call_doc"]["judgment_cache_key"]
    )
    assert all(
        contracts.document_hash(json.loads(item["call_doc"]["judgment_cache_canonical"]))
        == item["call_doc"]["judgment_cache_key"]
        for item in store.judgment_reserves
    )


def test_miner_keys_share_only_exact_frozen_request():
    store = JudgmentLedger()
    instance, _, transport = _broker(
        store, [(200, _answer()), (200, _answer("contradicted"))],
        funding_source_for=lambda _context, _provider: "miner_key",
        credential_for=lambda context, _provider: "key-for-" + context.run_id,
    )
    source = _call(instance, _context("miner-a"))
    hit = _call(instance, _context("miner-b"))
    changed = _call(
        instance, _context("miner-c"),
        {**_REQUEST, "messages": [*_REQUEST["messages"][:-1],
            {"role": "user", "content": "Storika did not launch the product."}]},
    )
    assert source.call["actual_microusd"] == 21
    assert hit.call["cached"] is True
    assert changed.call.get("cached") is None
    assert len(transport.sent) == 2
    assert store.judgment_reserves[0]["call_doc"]["judgment_cache_key"] != (
        store.judgment_reserves[-1]["call_doc"]["judgment_cache_key"]
    )


def test_pending_bill_replays_complete_response_and_keeps_source_uncertain():
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, _answer(cost=None))])
    source = _call(instance, _context("pending"))
    hit = _call(instance, _context("other"))
    assert source.status == hit.status == 200
    assert source.call["outcome"] == "uncertain"
    assert "actual_microusd" not in source.call
    assert hit.call["cached"] is True and hit.call["actual_microusd"] == 0
    assert len(transport.sent) == 1
    original = store.calls[source.call["call_identity"]]
    assert original["kind"] == "uncertain"
    assert original["uncertain_doc"]["judgment_cache_response"]["judgment_cache_eligible"] is True


def test_pending_bill_own_identity_replays_without_faking_zero_cost():
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, _answer(cost=None))])
    source = _call(instance, _context("pending-own"))
    retry = _call(instance, _context("pending-own"))
    assert source.status == retry.status == 200
    assert retry.body == source.body
    assert retry.call["outcome"] == "uncertain"
    assert retry.call["idempotent"] is True
    assert "actual_microusd" not in retry.call
    assert len(transport.sent) == 1
    source_call = store.calls[source.call["call_identity"]]
    source_call.update({
        "kind": "settlement", "actual": 21,
        "terminal": source_call["uncertain_doc"]["judgment_cache_response"],
    })
    reconciled = _call(instance, _context("pending-own"))
    assert reconciled.status == 200 and reconciled.body == source.body
    assert reconciled.call["outcome"] == "settled"
    assert reconciled.call["actual_microusd"] == 21
    assert reconciled.call.get("cached") is None
    assert len(transport.sent) == 1


@pytest.mark.parametrize("body", [
    {"choices": [], "usage": {"cost": "0.000021"}},
    {"choices": [{"finish_reason": "length", "message": {"content": '{"verdict":"verified"}'}}], "usage": {"cost": "0.000021"}},
    {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"cost": "0.000021"}},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"verdict":"verified"}', "refusal": "blocked"}}], "usage": {"cost": "0.000021"}},
])
def test_incomplete_answers_do_not_freeze_request(body):
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, body), (200, _answer())])
    first = _call(instance, _context("first"))
    second = _call(instance, _context("second"))
    assert first.call.get("cached") is None
    assert second.call.get("cached") is None
    assert len(transport.sent) == 2


def test_complete_no_format_json_reuses_exact_company_criteria():
    request = {
        key: value for key, value in _REQUEST.items() if key != "response_format"
    }
    request["messages"] = [
        {"role": "system", "content": "Return one JSON company-fit judgment."},
        {"role": "user", "content": "Storika: employee range 11-50."},
    ]
    answer = _answer()
    answer["choices"][0]["message"]["content"] = json.dumps({
        "employee_size_matches": True, "reason": "The stated range matches."
    })
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, answer), (200, answer)])
    source = _call(instance, _context("fit-source"), request)
    hit = _call(instance, _context("fit-hit"), request)
    changed = {**request, "messages": [
        request["messages"][0],
        {"role": "user", "content": "Storika: employee range 51-200."},
    ]}
    miss = _call(instance, _context("fit-changed"), changed)
    assert source.status == hit.status == miss.status == 200
    assert hit.call["cached"] is True
    assert hit.body == source.body
    assert miss.call.get("cached") is None
    assert len(transport.sent) == 2


@pytest.mark.parametrize("content", [
    "{}", "", "Company matches the range.", "```json\n{\"fits\":true}\n```",
    '{"fits":', '{"error":"provider failed"}',
])
def test_no_format_incomplete_content_is_not_replayed(content):
    request = {key: value for key, value in _REQUEST.items() if key != "response_format"}
    incomplete = _answer()
    incomplete["choices"][0]["message"]["content"] = content
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, incomplete), (200, _answer())])
    first = _call(instance, _context("incomplete-first"), request)
    second = _call(instance, _context("incomplete-second"), request)
    assert first.call.get("cached") is None
    assert second.call.get("cached") is None
    assert len(transport.sent) == 2


def test_frozen_scope_version_and_execution_bypass():
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, _answer())] * 5)
    _call(instance, _context("one"))
    altered = {**_FROZEN, "scorer_image_digest": "sha256:" + "b" * 64}
    _call(instance, _context("two", scope=altered))
    older_date = {**_FROZEN, "evaluation_date": "2026-10-01"}
    _call(instance, _context("three", scope=older_date))
    _call(instance, _context("execute", kind="execute"))
    revised_policy = scoring.build_scorer_policy(
        judge_models={"intent_signal_judge": _MODEL, "company_judge": _MODEL},
    )
    _call(instance, _context("four", scope={**_FROZEN, "scorer_policy": revised_policy}))
    assert len(transport.sent) == 5
    assert len(store.judgment_reserves) == 4
    assert len({
        item["call_doc"]["judgment_cache_key"]
        for item in store.judgment_reserves
    }) == 4


def test_busy_claim_retries_without_second_dispatch(monkeypatch):
    store = JudgmentLedger()
    store.busy_once = True
    instance, _, transport = _broker(store, [(200, _answer())])
    monkeypatch.setattr(broker.time, "sleep", lambda _seconds: None)
    result = _call(instance, _context("busy"))
    assert result.status == 200
    assert len(store.judgment_reserves) == 2
    assert len(transport.sent) == 1


def _sample_for_schema(schema):
    kind = schema["type"]
    if isinstance(kind, list):
        return _sample_for_schema({**schema, "type": kind[0]})
    if kind == "object":
        return {
            key: _sample_for_schema(schema["properties"][key])
            for key in schema.get("required", [])
        }
    if kind == "array":
        return [
            _sample_for_schema(schema["items"])
            for _ in range(schema.get("minItems", 0))
        ]
    if kind == "string":
        return (schema["enum"][0] if "enum" in schema else
                "x" * max(1, schema.get("minLength", 0)))
    if kind == "integer":
        return max(0, schema.get("minimum", 0))
    if kind == "boolean":
        return False
    raise AssertionError(kind)


def test_actual_intent_verifier_schema_admits_complete_verdict_only():
    from qualification.scoring import intent_verification_three_stage as verifier

    verdict = _sample_for_schema(verifier._SCHEMA)
    verdict["signal_evaluations"] = [
        _sample_for_schema(verifier._SCHEMA["properties"]["signal_evaluations"]["items"])
    ]
    request = {
        **_REQUEST,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verification", "strict": True,
                "schema": verifier._SCHEMA,
            },
        },
    }
    complete = _answer()
    complete["choices"][0]["message"]["content"] = json.dumps(verdict)
    body = json.dumps(complete).encode()
    assert broker._complete_judgment_response(
        "openrouter.chat", request, status=200, body=body, call_succeeded=True
    )
    verdict["signal_evaluations"][0].pop("same_entity_check")
    complete["choices"][0]["message"]["content"] = json.dumps(verdict)
    assert not broker._complete_judgment_response(
        "openrouter.chat", request, status=200,
        body=json.dumps(complete).encode(), call_succeeded=True,
    )


def test_actual_intent_details_schema_checks_bounds():
    from qualification.scoring import intent_details

    response_format = intent_details._RESPONSE_FORMAT
    verdict = _sample_for_schema(response_format["json_schema"]["schema"])
    # The schema's generic sample has a false aggregate and no factual units.
    # Use a consistent negative before testing the independent array bounds.
    verdict["unit_grounding"][0].update(contains_factual_claim=True, status="UNPROVEN")
    complete = _answer()
    complete["choices"][0]["message"]["content"] = json.dumps(verdict)
    request = {**_REQUEST, "response_format": response_format}
    assert broker._complete_judgment_response(
        "openrouter.chat", request, status=200,
        body=json.dumps(complete).encode(), call_succeeded=True,
    )
    verdict["unit_grounding"] = []
    complete["choices"][0]["message"]["content"] = json.dumps(verdict)
    assert not broker._complete_judgment_response(
        "openrouter.chat", request, status=200,
        body=json.dumps(complete).encode(), call_succeeded=True,
    )


def test_rejected_investigator_tool_reply_does_not_poison_another_call():
    from qualification.scoring import company_evidence_investigator as investigator

    raw_tools = investigator._tools(["revenue"])
    tools = [{"type": "function", "function": {
        key: value for key, value in tool.items() if key != "type"
    }} for tool in raw_tools]
    parameters = tools[-1]["function"]["parameters"]
    arguments = _sample_for_schema(parameters)
    arguments["findings"][0].update({
        "status": "VERIFIED", "observed_value": "$20 million",
        "evidence_url": "https://acme.example/about",
        "evidence_quote": "Unrelated Company reported $20 million in revenue.",
    })
    # A complete tool packet can still fail the scorer's evidence binding.
    validated = investigator._validated_findings(
        arguments, targets=("revenue",), fetched_pages={},
        first_party_domains={"acme.example"}, identity_names={"Acme"},
    )
    assert validated["revenue"]["status"] == "UNPROVEN"
    tool_call = {
        "id": "call_submit_1", "type": "function",
        "function": {"name": "submit_findings", "arguments": json.dumps(arguments)},
    }
    answer = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [tool_call],
    }}], "usage": {"cost": "0.000021"}}
    request = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": "Research one target."},
            {"role": "user", "content": "Company evidence"},
            {"role": "tool", "tool_call_id": "call_fetch_1", "content": "Fetched quote A"},
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "submit_findings"}},
        "parallel_tool_calls": False,
        "max_tokens": 200,
    }
    assert broker._complete_judgment_response(
        "openrouter.chat", request, status=200,
        body=json.dumps(answer).encode(), call_succeeded=True,
    )
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, answer)] * 3)
    source = _call(instance, _context("tool-source"), request)
    hit = _call(instance, _context("tool-hit"), request)
    changed = {**request, "messages": [
        *request["messages"][:-1],
        {"role": "tool", "tool_call_id": "call_fetch_1", "content": "Fetched quote B"},
    ]}
    miss = _call(instance, _context("tool-changed"), changed)
    assert source.status == hit.status == miss.status == 200
    assert hit.call.get("cached") is None
    assert miss.call.get("cached") is None
    assert len(transport.sent) == 3


def test_tool_call_malformed_or_undeclared_does_not_freeze_response():
    from qualification.scoring import company_evidence_investigator as investigator

    tool = investigator._tools(["revenue"])[0]
    request = {**_REQUEST, "tools": [{"type": "function", "function": {
        key: value for key, value in tool.items() if key != "type"
    }}], "tool_choice": "required"}
    def response(name, arguments, finish_reason="tool_calls"):
        return json.dumps({"choices": [{"finish_reason": finish_reason, "message": {
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": name, "arguments": arguments,
            }}],
        }}]}).encode()
    for body in (
        response("search_web", "{"),
        response("search_web", '{}'),
        response("undeclared", '{"query":"hello"}'),
        response("search_web", '{"query":"hello"}', "length"),
    ):
        assert not broker._complete_judgment_response(
            "openrouter.chat", request, status=200, body=body, call_succeeded=True,
        )


def _research_tool_request(parallel=False):
    return {
        "model": _MODEL, "messages": [{"role": "user", "content": "Research Acme."}],
        "tools": [{"type": "function", "function": {
            "name": name, "strict": True, "parameters": {
                "type": "object", "additionalProperties": False,
                "required": [key], "properties": {key: {"type": "string"}},
            },
        }} for name, key in (("fetch_page", "url"), ("search_web", "query"))],
        "tool_choice": "required", "max_tokens": 200,
        **({"parallel_tool_calls": parallel} if parallel is not None else {}),
    }


def _research_tool_answer(names):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{"id": "call-" + str(i), "type": "function",
            "function": {"name": name, "arguments": json.dumps(
                {"url": "https://acme.example"} if name == "fetch_page" else {"query": "Acme"}
            )}} for i, name in enumerate(names)],
    }}], "usage": {"cost": "0.000021"}}


@pytest.mark.parametrize("names", [
    ("fetch_page", "fetch_page", "search_web"),  # retained 57 / cached 67
    ("fetch_page", "fetch_page", "fetch_page"),  # retained 63
])
def test_single_tool_request_skips_historical_paid_multi_source(names, monkeypatch):
    store = JudgmentLedger()
    multi = _research_tool_answer(names)
    single = _research_tool_answer(("search_web",))
    instance, _, transport = _broker(store, [(200, multi), (200, single), (200, single)])
    request = _research_tool_request(False)
    with monkeypatch.context() as legacy:
        legacy.setattr(broker, "_judgment_requires_own_call", lambda _: False)
        source = _call(instance, _context("legacy-multi"), request)
    source_call = store.calls[source.call["call_identity"]]
    assert source_call["terminal"].get("judgment_cache_eligible") is not True
    # A response retained by the old gateway remains paid and unmodified.
    source_call["terminal"]["judgment_cache_eligible"] = True
    before = copy.deepcopy(source_call)
    fresh = _call(instance, _context("fresh-single"), request)
    hit = _call(instance, _context("single-hit"), request)
    same_paid = _call(instance, _context("legacy-multi"), request)
    assert fresh.status == hit.status == same_paid.status == 200
    assert fresh.call.get("cached") is None and hit.call.get("cached") is None
    assert len(transport.sent) == 3
    assert store.calls[source.call["call_identity"]] == before
    assert all(x["call_doc"]["judgment_cache_single_tool_required"] is True
               for x in store.judgment_reserves)


@pytest.mark.parametrize("parallel", [True, None])
def test_allowed_multi_tool_request_keeps_each_call_independent(parallel):
    store = JudgmentLedger()
    answer = _research_tool_answer(("fetch_page", "search_web"))
    instance, _, transport = _broker(store, [(200, answer)] * 2)
    request = _research_tool_request(parallel)
    source = _call(instance, _context("allowed-source"), request)
    hit = _call(instance, _context("allowed-hit"), request)
    assert source.status == hit.status == 200 and hit.call.get("cached") is None
    assert len(transport.sent) == 2
    assert all("judgment_cache_single_tool_required" not in x["call_doc"]
               for x in store.judgment_reserves)


@pytest.mark.parametrize("count,expected", [(0, False), (1, True), (2, False), (3, False)])
def test_explicit_single_tool_cache_eligibility_preserves_zero_and_single(count, expected):
    assert broker._complete_judgment_response(
        "openrouter.chat", _research_tool_request(False), status=200,
        body=json.dumps(_research_tool_answer(("search_web",) * count)).encode(),
        call_succeeded=True,
    ) is expected


@pytest.mark.parametrize("parallel,count,expected", [
    (False, 0, True), (False, 1, True), (False, 2, False), (False, 3, False),
    (True, 2, True), (None, 2, True),
])
def test_stop_json_response_also_respects_single_tool_request(parallel, count, expected):
    answer = _research_tool_answer(("search_web",) * count)
    answer["choices"][0]["finish_reason"] = "stop"
    answer["choices"][0]["message"]["content"] = '{"result":"complete"}'
    assert broker._complete_judgment_response(
        "openrouter.chat", _research_tool_request(parallel), status=200,
        body=json.dumps(answer).encode(), call_succeeded=True,
    ) is expected


@pytest.mark.parametrize("choice", ["auto", "none", "required", {
    "type": "function", "function": {"name": "search_web"},
}])
def test_declared_tools_never_share_even_when_choice_disables_tools(choice):
    request = {**_REQUEST, "tools": _research_tool_request()["tools"],
               "tool_choice": choice}
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, _answer())] * 2)
    first = _call(instance, _context("choice-source"), request)
    second = _call(instance, _context("choice-other"), request)
    retry = _call(instance, _context("choice-source"), request)
    assert first.status == second.status == retry.status == 200
    assert first.body == retry.body and retry.call["idempotent"] is True
    assert second.call.get("cached") is None
    assert len(transport.sent) == 2
    assert all(item["call_doc"]["judgment_cache_own_call_only"] is True
               for item in store.judgment_reserves)


@pytest.mark.parametrize("marker", [
    {"messages": [{"role": "tool", "tool_call_id": "c", "content": "quote"}]},
    {"messages": [{"role": "assistant", "tool_calls": []}]},
    *[{"input": [{"type": kind}]} for kind in (
        "function_call", "function_call_output", "custom_tool_call",
        "custom_tool_call_output", "web_search_call", "openrouter:web_search", "additional_tools",
    )],
])
def test_tool_history_requires_own_call_when_tools_are_omitted(marker):
    assert broker._judgment_requires_own_call(marker) is True


@pytest.mark.parametrize("operation,parameters", [
    ("openrouter.chat", {**_REQUEST, "tools": []}),
    ("openrouter.responses", {"model": _MODEL, "input": "Verify claim", "tools": []}),
])
def test_plain_text_requests_remain_shareable(operation, parameters):
    assert broker._judgment_requires_own_call(parameters) is False
    assert broker._judgment_cache_material(
        _context("plain"), requested_operation_id=operation,
        effective_operation_id=operation, normalized_request=parameters,
        outbound_body=json.dumps(parameters).encode(),
    ) is not None


@pytest.mark.parametrize("state", ["reservation", "dispatch", "settlement", "uncertain", "cached"])
def test_historic_tool_identity_survives_new_sharing_policy(monkeypatch, state):
    request = _research_tool_request()
    answer = _research_tool_answer(("search_web",))
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, answer)] * 2)
    with monkeypatch.context() as legacy:
        legacy.setattr(broker, "_judgment_requires_own_call", lambda _: False)
        first = _call(instance, _context("historic-source"), request)
        if state == "cached":
            first = _call(instance, _context("historic-destination"), request)
    original = store.calls[first.call["call_identity"]]
    original_doc = copy.deepcopy(original["call_doc"])
    assert "judgment_cache_own_call_only" not in original_doc
    if state in {"reservation", "dispatch"}:
        original["kind"] = state
        original.pop("terminal")
        original.pop("actual")
    elif state == "uncertain":
        original.update(kind="uncertain", uncertain_doc={
            "judgment_cache_response": original.pop("terminal"),
        })
        original.pop("actual")
    transport.sent.clear()
    context = _context("historic-destination" if state == "cached" else "historic-source")
    replay = _call(instance, context, request)
    assert original["call_doc"] == original_doc
    assert len(transport.sent) == (1 if state == "reservation" else 0)
    if state == "dispatch":
        assert replay.call["outcome"] == "uncertain"
    else:
        assert replay.status == 200
        if state != "reservation":
            assert replay.body == first.body
            assert replay.call["idempotent"] is True
        if state == "cached":
            assert replay.call["cached"] is True
            assert replay.call["actual_microusd"] == 0


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_invalid_own_call_policy_cannot_relax_reservation_readback(invalid):
    store = JudgmentLedger()
    instance, _, _ = _broker(store, [(200, _answer())])
    result = _call(instance, _context("invalid-marker"))
    original = store.calls[result.call["call_identity"]]
    original.update(kind="reservation")
    original["call_doc"]["judgment_cache_own_call_only"] = invalid
    arguments = store.judgment_reserves[0]
    assert broker._reservation_readback_matches(
        store, arguments, store._view(original),
    ) is False


@pytest.mark.parametrize("history", [None, {
    "type": "function_call_output", "call_id": "call_search", "output": "quote",
}])
def test_responses_tools_and_history_keep_own_reply_and_billing(history):
    parameters = {
        "model": _MODEL, "input": [history] if isinstance(history, dict) else "Verify claim",
        "max_output_tokens": 200,
        "text": {"format": {"type": "json_schema", "name": "verdict",
                            "strict": True, "schema": _SCHEMA}},
    }
    if history is None:
        parameters.update(tools=[{"type": "function", "name": "search_web",
                                  "parameters": {"type": "object"}}], tool_choice="none")
    answer = {"status": "completed", "output": [{
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": '{"verdict":"verified"}'}],
    }], "usage": {"cost": "0.000021"}}
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, answer)] * 2)
    def call(label):
        return instance.execute(_context(label), operation_id="openrouter.responses",
                                parameters=parameters, action_sequence=1, timeout_ms=5000)
    first, other, own = call("responses-source"), call("responses-other"), call("responses-source")
    assert first.status == other.status == own.status == 200
    assert other.call.get("cached") is None and other.call["actual_microusd"] == 21
    assert own.call["idempotent"] is True and own.body == first.body
    assert len(transport.sent) == 2
    assert all(item["call_doc"]["judgment_cache_own_call_only"] is True
               for item in store.judgment_reserves)
