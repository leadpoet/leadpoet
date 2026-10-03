"""Exact scorer provider replay through the existing Arena call ledger."""

from __future__ import annotations

import json
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
            if self.busy_once:
                self.busy_once = False
                return {"status": "cache_busy", "judgment_cache_key": key}
            for source in self.calls.values():
                if source["call_doc"].get("judgment_cache_key") != key:
                    continue
                terminal = (
                    source.get("terminal")
                    if source["kind"] == "settlement" else
                    source.get("uncertain_doc", {}).get("judgment_cache_response")
                )
                if isinstance(terminal, dict) and terminal.get("judgment_cache_eligible") is True:
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


def test_actual_investigator_submit_findings_tool_call_reuses_exact_history():
    from qualification.scoring import company_evidence_investigator as investigator

    raw_tools = investigator._tools(["revenue"])
    tools = [{"type": "function", "function": {
        key: value for key, value in tool.items() if key != "type"
    }} for tool in raw_tools]
    parameters = tools[-1]["function"]["parameters"]
    arguments = _sample_for_schema(parameters)
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
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [(200, answer), (200, answer)])
    source = _call(instance, _context("tool-source"), request)
    hit = _call(instance, _context("tool-hit"), request)
    changed = {**request, "messages": [
        *request["messages"][:-1],
        {"role": "tool", "tool_call_id": "call_fetch_1", "content": "Fetched quote B"},
    ]}
    miss = _call(instance, _context("tool-changed"), changed)
    assert source.status == hit.status == miss.status == 200
    assert hit.call["cached"] is True
    assert miss.call.get("cached") is None
    assert len(transport.sent) == 2


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
