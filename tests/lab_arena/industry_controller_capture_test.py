"""Frozen industry workflow and host tool capture; no provider calls."""
import asyncio
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, scoring
from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_investigator import _finding

SOL = "openai/gpt-6.1-sol"
ROLE = "industry_research_controller"
ENV = "ARENA_INDUSTRY_RESEARCH_CONTROLLER_MODEL"


def historical_policy_without_controller():
    from tests.lab_arena.judge_default_activation_test import PREVIOUS_MODELS, PREVIOUS_OPTIONS, ROUTES, EXCEPTIONS
    policy = scoring.build_scorer_policy(judge_models=PREVIOUS_MODELS,
        judge_provider_routes=ROUTES, judge_request_options=PREVIOUS_OPTIONS,
        judge_reasoning_exceptions=EXCEPTIONS)
    del policy["judge_models"][ROLE]
    for binding in (contracts.JUDGE_PROVIDER_ROUTES_BINDING, contracts.JUDGE_REQUEST_OPTIONS_BINDING):
        values = json.loads(policy["env_bindings"][binding])
        del values["z-ai/glm-5.3-flash"]
        policy["env_bindings"][binding] = json.dumps(values)
    return policy


def test_historical_sol_exception_covers_only_final_industry_review():
    from tests.lab_arena.judge_default_activation_test import PREVIOUS_MODELS, PREVIOUS_OPTIONS, ROUTES, EXCEPTIONS
    policy = scoring.build_scorer_policy(judge_models=PREVIOUS_MODELS,
        judge_provider_routes=ROUTES, judge_request_options=PREVIOUS_OPTIONS,
        judge_reasoning_exceptions=EXCEPTIONS)
    assert policy["judge_models"][ROLE] == "z-ai/glm-5.3-flash"
    exception = contracts.judge_reasoning_exceptions(policy)[SOL]
    assert exception["roles"] == ["industry_evidence_review"]
    assert contracts.validate_scorer_policy(json.loads(json.dumps(policy))) == policy
    old = historical_policy_without_controller()
    exceptions = {SOL: {**exception, "roles": ["industry_evidence_review"]}}
    old["env_bindings"][contracts.JUDGE_REASONING_EXCEPTIONS_BINDING] = json.dumps(exceptions)
    assert contracts.validate_scorer_policy(old) == old
    for role in ("role_batch_check", "intent_details_review", ROLE):
        bad = deepcopy(policy)
        bad["judge_models"][role] = SOL
        with pytest.raises(contracts.ArenaContractError):
            contracts.validate_scorer_policy(bad)


@pytest.mark.parametrize("capture", [False, True])
def test_historical_absent_role_binds_empty_and_rejects_ambient(capture):
    policy = historical_policy_without_controller()
    if capture:
        entry = {SOL: {"roles": ["industry_evidence_review"], "reason": "quality_gate_not_met"}}
        policy["env_bindings"][contracts.JUDGE_REASONING_EXCEPTIONS_BINDING] = json.dumps(entry)
    else:
        policy["env_bindings"] = {}
    before = contracts.canonical_json(policy)
    credentials = {name: "offline-placeholder" for name in scoring.CREDENTIAL_ENV_NAMES}
    env = {}
    scoring.apply_policy_to_environment(policy, environ=env, credentials=credentials)
    assert env[ENV] == "" and contracts.canonical_json(policy) == before
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ={ENV: SOL}, credentials=credentials)


def run_attribute(monkeypatch, *, recorder=None, replies=None, capture=True):
    url = "https://acme.example/"
    text = "Acme supplies reporting software for enterprise customers."
    finding = _finding("required_attribute", status="VERIFIED", evidence_url=url, evidence_quote=text)
    default = {"choices": [{"message": {"tool_calls": [{"id": "submit-1", "type": "function",
        "function": {"name": "submit_findings", "arguments": json.dumps({"findings": [finding]})}}]}}]}
    calls = []
    documents = iter(replies or [default])
    async def post(_session, _url, *, headers, payload):
        calls.append(deepcopy(payload))
        return 200, deepcopy(next(documents))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setenv("ARENA_JUDGE_REASONING_CAPTURE_POLICY", "full_generated_thinking_v1" if capture else "")
    monkeypatch.setenv("ARENA_JUDGE_PROVIDER_ROUTES", json.dumps({"test/controller": "example/verified"}))
    monkeypatch.delenv("ARENA_JUDGE_REQUEST_OPTIONS", raising=False)
    monkeypatch.delenv("ARENA_JUDGE_REASONING_EXCEPTIONS", raising=False)
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    monkeypatch.setattr(investigator, "INVESTIGATOR_MODEL", "test/controller")
    monkeypatch.setattr(investigator, "_post_json", post)
    if recorder is not None:
        monkeypatch.setattr(investigator, "record_judge_tool_event", recorder)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": url}, targets=("required_attribute",),
        requested_attribute="Supplies reporting software", prior_observations={"submitted_source_urls": [url]},
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example"},
        prefetched_pages={url: {"final_url": url, "text": text}}))
    return result, calls


def test_terminal_host_validation_is_captured_before_return(monkeypatch):
    events = []
    result, calls = run_attribute(monkeypatch, recorder=lambda event, **_: events.append(deepcopy(event)))
    assert result["_completed_submit"] is True
    assert len(events) == 1 and events[0]["disposition"] == "accepted_terminal"
    assert events[0]["result"] == result
    assert json.loads(events[0]["arguments"])["findings"][0]["target"] == "required_attribute"
    assert events[0]["sequence"] == 0 and events[0]["host_validation"]["gold_label"] is False


def test_capture_failure_or_off_does_not_change_native_result(monkeypatch):
    def broken(*_, **__):
        raise RuntimeError("offline capture failure")
    result, _ = run_attribute(monkeypatch, recorder=broken)
    assert result["_completed_submit"] is True
    events = []
    result, _ = run_attribute(monkeypatch, recorder=lambda event, **_: events.append(event), capture=False)
    assert result["_completed_submit"] is True and events == []


def tool_reply(name, arguments, *, call_id="tool", finish="tool_calls"):
    return {"choices": [{"finish_reason": finish, "message": {"tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": name, "arguments": arguments}}]}}]}


def final_attribute():
    return tool_reply("submit_findings", json.dumps({"findings": [_finding(
        "required_attribute", status="VERIFIED", evidence_url="https://acme.example/",
        evidence_quote="Acme supplies reporting software for enterprise customers.")]}), call_id="final")


def test_empty_exhausted_reply_repair_has_no_raw_tool_call(monkeypatch):
    empty = {"choices": [{"finish_reason": "length", "message": {"content": "", "tool_calls": []}}]}
    events = []
    result, calls = run_attribute(monkeypatch, recorder=lambda event, **_: events.append(deepcopy(event)),
                                  replies=[empty, final_attribute()])
    assert result["_completed_submit"] is True and len(calls) == 2
    assert [event["disposition"] for event in events] == ["repair", "accepted_terminal"]
    assert events[0]["host_validation"]["raw_tool_call"] is None
    assert events[0]["tool_call_id"] is None and events[0]["arguments"] is None


@pytest.mark.parametrize("first", ["fetch", "malformed_submit", "ambiguous_batch"])
def test_ordered_host_results_and_repairs_preserve_raw_arguments(monkeypatch, first):
    raw = ' { "url" : "https://acme.example/" } '
    initial = tool_reply("fetch_page", raw)
    if first == "malformed_submit":
        raw = '{"findings": ['
        initial = tool_reply("submit_findings", raw, finish="length")
    if first == "ambiguous_batch":
        second = deepcopy(initial["choices"][0]["message"]["tool_calls"][0])
        second["id"] = "other"
        initial["choices"][0]["message"]["tool_calls"].append(second)
    events = []
    result, calls = run_attribute(monkeypatch, recorder=lambda event, **_: events.append(deepcopy(event)),
                                  replies=[initial, final_attribute()])
    assert result["_completed_submit"] is True and len(calls) == 2
    assert [e["sequence"] for e in events] == list(range(len(events)))
    assert len({e["investigation_id"] for e in events}) == 1
    assert events[0]["arguments"] == raw
    assert all(e["request_sha256"].startswith("sha256:") and e["response_sha256"].startswith("sha256:") for e in events)
    if first == "ambiguous_batch":
        assert [e["disposition"] for e in events] == ["rejected_not_executed", "rejected_not_executed", "accepted_terminal"]
    else:
        assert events[0]["disposition"] == ("executed" if first == "fetch" else "repair")
    assert events[-1]["result"] == result


def test_actual_terminal_callback_reaches_worker_without_another_model_turn(monkeypatch):
    from pathlib import Path
    import tempfile
    from lab_arena import lab_arena_checkpoint as checkpoint
    from tests.lab_arena.judge_local_tool_test import worker
    with tempfile.TemporaryDirectory(prefix="controller-tool-", dir="/tmp") as short:
        server, state, api = worker(Path(short))
        server.start()
        try:
            monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, str(Path(short) / "worker.sock"))
            result, calls = run_attribute(monkeypatch)
            assert result["_completed_submit"] is True and len(calls) == 1
            assert server.wait_for_handlers()
            records = [r for r in state.worker_observations.agent_observations if r["kind"] == "judge_local_tool"]
            assert len(records) == 1 and records[0]["data"]["result"] == result
            assert records[0]["data"]["disposition"] == "accepted_terminal"
            assert api.provider_frames == []  # Native model stub is separate; capture adds no provider call.
        finally:
            server.stop()


@pytest.mark.parametrize("name,args", [("search_web", '{"query": 1}'),
                                      ("fetch_page", '{"url": "not a URL"}'),
                                      ("unknown_tool", '{}')])
def test_rejected_source_proposals_never_claim_execution(monkeypatch, name, args):
    events = []
    replies = [tool_reply(name, args), final_attribute()]
    result, _ = run_attribute(monkeypatch, recorder=lambda event, **_: events.append(deepcopy(event)), replies=replies)
    assert events[0]["disposition"] == "rejected_not_executed"
    assert events[0]["arguments"] == args
    if name != "unknown_tool":
        assert result["_completed_submit"] is True
    else:
        assert result["failure_reason"] and events[0]["host_validation"]["terminal"] is True


def test_new_sol_controller_keeps_legacy_wire_and_repair_bounds(monkeypatch):
    from tests.lab_arena.judge_default_activation_test import PREVIOUS_OPTIONS, ROUTES
    url = "https://acme.example/"
    async def review(*_, **__):
        return {"target": "industry", "status": "UNPROVEN", "evidence_url": "", "evidence_quote": "",
            "_industry_review": {"decision": "match", "missing_requirements": [], "reason": "Fixture only",
                "source_urls": [url], "research_query": ""}}, "", 1
    monkeypatch.setattr(investigator, "_review_industry_sources", review)
    monkeypatch.setattr(investigator, "record_judge_tool_event", lambda *_, **__: None)
    def wire(capture, controller):
        calls = []
        async def post(_session, endpoint, *, headers, payload):
            calls.append((endpoint, deepcopy(payload)))
            if len(calls) == 1:
                reply = tool_reply("submit_findings", '{"findings": [', finish="length")
                reply["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "opaque state"}]
                return 200, reply
            return 400, {}
        monkeypatch.setattr(investigator, "_post_json", post)
        monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
        monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
        monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", SOL)
        monkeypatch.setenv(ENV, controller)
        monkeypatch.setenv("ARENA_COMPANY_REVERIFY_MODEL", "deepseek/deepseek-v4.1-flash-1017")
        monkeypatch.setenv("ARENA_JUDGE_REASONING_CAPTURE_POLICY", "full_generated_thinking_v1" if capture else "")
        monkeypatch.setenv("ARENA_JUDGE_REASONING_EXCEPTIONS", json.dumps({SOL: {
            "roles": ["industry_evidence_review"], "reason": "quality_gate_not_met"}}))
        monkeypatch.setenv("ARENA_JUDGE_PROVIDER_ROUTES", json.dumps(ROUTES))
        monkeypatch.setenv("ARENA_JUDGE_REQUEST_OPTIONS", json.dumps(PREVIOUS_OPTIONS))
        monkeypatch.setattr(investigator, "REASONING_MAX_TOKENS", 16384 if capture else 3000)
        asyncio.run(investigator.investigate_company_evidence(company_locator={"name": "Acme", "website": url},
            targets=("industry",), requested_industry="Software", prior_observations={"submitted_source_urls": [url]},
            verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example"},
            prefetched_pages={url: {"final_url": url, "text": "Acme supplies software to companies."}}))
        return calls
    old = wire(False, "")
    new = wire(True, SOL)
    assert len(old) == len(new) == 2 and old == new
    assert all(p["model"] == SOL and p["max_tokens"] == 3000 and p["reasoning"] == {"effort": "low"}
               and p["parallel_tool_calls"] is False for _, p in new)
    assert all("reasoning_details" not in m for m in new[1][1]["messages"])
    historical = wire(True, "")
    assert len(historical) == 2
    assert all(p["model"] == "deepseek/deepseek-v4.1-flash-1017" and p["max_tokens"] == 8192
               and p["provider"] == {"only": ["nebius/fp8"]} for _, p in historical)
    current = wire(True, "z-ai/glm-5.3-flash")
    assert len(current) == 2
    assert all(p["model"] == "z-ai/glm-5.3-flash" and p["max_tokens"] == 8192
               and p["provider"] == {"only": ["near-ai/fp8"]}
               and p["reasoning"] == {"effort": "low", "exclude": False}
               and p.get("parallel_tool_calls", False) is False for _, p in current)
    # Keep the existing GLM one-tool adapter and the complete research contract.
    assert current[0][1]["messages"][0] == {
        "role": "system", "content": "Return exactly one tool call per assistant turn."}
    assert current[0][1]["messages"][1:] == new[0][1]["messages"]
    assert current[0][1]["tools"] == new[0][1]["tools"]
