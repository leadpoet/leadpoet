"""Causal controller tests under the manifest's explicit injected-runtime contract.

Only trusted, checked-in synthetic bundles run here. Real action records belong
to injected tools; a candidate's self-reported trace is never used as evidence.
The test calls configure before run_icp; production Arena does not provide that
step. These are conditional controller checks, not admitted-submission accuracy,
arbitrary-code execution safety, better agent quality, or LLM-classifier proof.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import tarfile
from pathlib import Path

import pytest

from lab_arena import source_bundle


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "research_control"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
IDS = [case["id"] for case in MANIFEST["fixtures"]]
ICP = {"industry": "software"}
COMPANIES = [{"company_name": "Example", "company_website": "https://example.com"}]


def load_bundle(name):
    spec = importlib.util.spec_from_file_location("fixture_" + name, FIXTURES / name / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    def __init__(self, policy):
        self.policy = policy
        self.requests = []

    def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        return copy.deepcopy(self.policy(request))


class Tools:
    def __init__(self, observation="continue"):
        self.observation = observation
        self.calls = []

    def __call__(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        return {"url": "https://example.com/evidence", "finding": self.observation}


def action(name, **arguments):
    return {"action": name, "arguments": arguments}


def finish():
    return {"action": "finish", "companies": COMPANIES}


def native_call(name, call_id="call_1", **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def assistant(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def native_finish():
    return {"role": "assistant", "content": json.dumps({"companies": COMPANIES})}


@pytest.mark.parametrize("name", IDS)
def test_fixture_is_valid_deterministic_source_bundle(name, tmp_path):
    bundle = FIXTURES / name
    source_bundle.validate_source_directory(bundle, require_license=True)
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    source_bundle.write_source_archive(bundle, first)
    source_bundle.write_source_archive(bundle, second)
    assert first.read_bytes() == second.read_bytes()
    source_bundle.validate_source_archive(first.read_bytes(), require_license=True)
    with tarfile.open(first) as archive:
        assert "manifest.json" not in archive.getnames()
        assert set(archive.getnames()) == {"harness.py", "LICENSE"}
    assert not inspect.iscoroutinefunction(load_bundle(name).run_icp)


@pytest.mark.parametrize("name", ["adaptive_json", "bounded_hybrid"])
def test_json_model_action_and_finish_interventions_change_real_actions(name):
    runs = []
    for decision in (action("fetch", url="https://example.com/a"), action("search", query="alternative"), finish()):
        tools = Tools()

        def policy(request):
            observations = request["observations"]
            if not observations:
                return action("search", query="software")
            return decision if len(observations) == 1 else finish()

        module, model = load_bundle(name), Model(policy)
        module.configure(model, tools)
        assert module.run_icp(ICP) == COMPANIES
        runs.append(tools.calls)
    assert runs[0][0] == runs[1][0] == runs[2][0]
    assert runs[0][1] == ("fetch", {"url": "https://example.com/a"})
    assert runs[1][1] == ("search", {"query": "alternative"})
    assert runs[2] == [("search", {"query": "software"})]


@pytest.mark.parametrize("name", ["adaptive_json", "bounded_hybrid"])
def test_json_tool_observation_intervention_changes_next_action_and_stop(name):
    def policy(request):
        observations = request["observations"]
        if not observations:
            return action("search", query="software")
        if len(observations) > 1 or observations[-1]["result"]["finding"] == "sufficient":
            return finish()
        return action("fetch", url=observations[-1]["result"]["url"])

    runs = []
    for finding in ("continue", "sufficient"):
        module, model, tools = load_bundle(name), Model(policy), Tools(finding)
        module.configure(model, tools)
        assert module.run_icp(ICP) == COMPANIES
        assert model.requests[-1]["observations"][-1]["result"]["finding"] == finding
        runs.append(tools.calls)
    assert [name for name, _ in runs[0]] == ["search", "fetch"]
    assert [name for name, _ in runs[1]] == ["search"]


def test_native_model_action_and_finish_interventions_change_real_actions():
    runs = []
    for next_reply in (assistant(native_call("fetch", url="https://example.com/a")), assistant(native_call("search", query="alternative")), native_finish()):
        replies = iter((assistant(native_call("search", query="software")), next_reply, native_finish()))
        module, model, tools = load_bundle("native_tools"), Model(lambda _: next(replies)), Tools()
        module.configure(model, tools)
        assert module.run_icp(ICP) == COMPANIES
        runs.append(tools.calls)
    assert runs[0][1][0] == "fetch"
    assert runs[1][1][0] == "search"
    assert runs[2] == [("search", {"query": "software"})]


def test_native_tool_observation_controls_continuation_and_stop():
    def policy(request):
        results = [json.loads(message["content"]) for message in request["messages"] if message["role"] == "tool"]
        if not results:
            return assistant(native_call("search", query="software"))
        if len(results) > 1 or results[-1]["finding"] == "sufficient":
            return native_finish()
        return assistant(native_call("fetch", call_id="call_2", url=results[-1]["url"]))

    runs = []
    for finding in ("continue", "sufficient"):
        module, model, tools = load_bundle("native_tools"), Model(policy), Tools(finding)
        module.configure(model, tools)
        assert module.run_icp(ICP) == COMPANIES
        runs.append(tools.calls)
    assert [name for name, _ in runs[0]] == ["search", "fetch"]
    assert [name for name, _ in runs[1]] == ["search"]


def test_native_transcript_preserves_assistant_calls_tool_ids_order_and_final_json():
    batch = assistant(native_call("search", "b", query="first"), native_call("fetch", "a", url="https://example.com/second"))
    replies = iter((batch, native_finish()))
    module, model, tools = load_bundle("native_tools"), Model(lambda _: next(replies)), Tools()
    module.configure(model, tools)
    output = module.run_icp(ICP)
    assert json.loads(json.dumps({"companies": output})) == {"companies": COMPANIES}
    messages = model.requests[1]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool"]
    assert messages[1] == batch
    assert [message["tool_call_id"] for message in messages[2:]] == ["b", "a"]
    assert tools.calls == [("search", {"query": "first"}), ("fetch", {"url": "https://example.com/second"})]


@pytest.mark.parametrize("name", ["ignored_model_decoy", "forced_tool_decoy"])
def test_decoy_logs_do_not_replace_actual_tool_records(name):
    runs, logs = [], []
    for reply, finding in (({**finish(), **native_finish()}, "sufficient"), ({**action("search", query="new"), **assistant(native_call("search", query="new"))}, "continue")):
        module, model, tools, trace = load_bundle(name), Model(lambda _: reply), Tools(finding), []
        module.configure(model, tools, trace)
        assert module.run_icp(ICP) == []
        runs.append(tools.calls)
        logs.append(trace)
    assert logs[0] != logs[1]
    assert runs[0] == runs[1] == [("search", {"query": "software"}), ("fetch", {"url": "https://example.com/fixed"})]


def test_fixed_extractor_has_no_model_decision_until_research_is_complete():
    module, tools = load_bundle("fixed_extractor"), Tools()

    def extract(request):
        assert [name for name, _ in tools.calls] == ["search", "fetch"]
        assert "evidence" in request
        return {"companies": COMPANIES}

    module.configure(Model(extract), tools)
    assert module.run_icp(ICP) == COMPANIES


def test_one_shot_query_plan_cannot_stop_or_change_fixed_stages_after_observation():
    runs = []
    for finding in ("sufficient", "continue"):
        module, tools = load_bundle("fixed_query_plan"), Tools(finding)

        def plan_or_extract(request):
            if request["stage"] == "plan":
                assert tools.calls == []
                return {"queries": ["first", "second"]}
            assert [name for name, _ in tools.calls] == ["search", "fetch", "search", "fetch"]
            return {"companies": COMPANIES}

        module.configure(Model(plan_or_extract), tools)
        assert module.run_icp(ICP) == COMPANIES
        runs.append(tools.calls)
    assert runs[0] == runs[1]


def test_missing_dynamic_helper_is_unverified_even_when_injected_helpers_diverge():
    runs = []
    for helper in (lambda icp, model, tools: COMPANIES, lambda icp, model, tools: [tools("search", {"query": "external"})]):
        module, tools = load_bundle("dynamic_helper"), Tools()
        module.configure(Model(lambda _: finish()), tools, helper=helper)
        module.run_icp(ICP)
        runs.append(tools.calls)
    assert runs[0] == []
    assert runs[1] == [("search", {"query": "external"})]
    assert next(case for case in MANIFEST["fixtures"] if case["id"] == "dynamic_helper")["expected"] == "unverified"


def test_unused_adaptive_controller_does_not_change_fixed_actual_entrypoint():
    runs = []
    for decision in (finish(), action("search", query="different")):
        module, tools = load_bundle("unreachable_adaptive"), Tools()
        model = Model(lambda _: decision)
        module.configure(model, tools)
        assert module.run_icp(ICP) == []
        assert model.requests == []
        runs.append(tools.calls)
    assert runs[0] == runs[1] == [("search", {"query": "software"}), ("fetch", {"url": "https://example.com/fixed"})]


def test_unreachable_controller_itself_is_adaptive_but_is_not_the_entrypoint():
    runs = []
    for decision in (finish(), action("fetch", url="https://example.com/alternative")):
        module, tools = load_bundle("unreachable_adaptive"), Tools()
        replies = iter((decision, finish()))
        module.configure(Model(lambda _: next(replies)), tools)
        assert module.adaptive_controller(ICP) == COMPANIES
        runs.append(tools.calls)
    assert runs[0] == []
    assert runs[1] == [("fetch", {"url": "https://example.com/alternative"})]


def test_guarded_dispatch_leaves_external_controller_behavior_unverified():
    def stop(icp, model, tools):
        return COMPANIES

    def research(icp, model, tools):
        tools("fetch", {"url": "https://example.com/external"})
        return COMPANIES

    runs = []
    for controller in (stop, research):
        module, tools = load_bundle("guarded_dispatch"), Tools()
        module.configure(Model(lambda _: finish()), tools, controllers={"external": controller})
        assert module.run_icp({**ICP, "use_external_controller": True, "controller": "external"}) == COMPANIES
        runs.append(tools.calls)
    assert runs[0] == []
    assert runs[1] == [("fetch", {"url": "https://example.com/external"})]

    module, tools = load_bundle("guarded_dispatch"), Tools()
    module.configure(Model(lambda _: finish()), tools, controllers={"external": research})
    assert module.run_icp({**ICP, "use_external_controller": False, "controller": "external"}) == []
    assert tools.calls == [("search", {"query": "software"})]
    assert next(case for case in MANIFEST["fixtures"] if case["id"] == "guarded_dispatch")["expected"] == "unverified"


@pytest.mark.parametrize("name", ["adaptive_json", "bounded_hybrid"])
@pytest.mark.parametrize("bad", [None, [], {"action": []}, action("delete", path="/tmp/x"), {"action": "fetch", "arguments": []}, {"action": "finish", "companies": {}}])
def test_json_invalid_or_unknown_action_stops_before_dispatch(name, bad):
    module, tools = load_bundle(name), Tools()
    module.configure(Model(lambda _: bad), tools)
    with pytest.raises(ValueError):
        module.run_icp(ICP)
    assert tools.calls == ([] if name == "adaptive_json" else [("search", {"query": "software"})])


@pytest.mark.parametrize("bad", [assistant(native_call("delete", path="/tmp/x")), assistant(native_call([], query="x")), assistant(native_call("search", "same"), native_call("fetch", "same")), assistant({"id": "x", "function": {"name": "search", "arguments": "not JSON"}}), {"role": "assistant", "content": '{"companies":{}}'}])
def test_native_invalid_or_unknown_tool_rejects_entire_batch_before_dispatch(bad):
    module, tools = load_bundle("native_tools"), Tools()
    module.configure(Model(lambda _: bad), tools)
    with pytest.raises(ValueError):
        module.run_icp(ICP)
    assert tools.calls == []


@pytest.mark.parametrize("name", ["adaptive_json", "bounded_hybrid", "native_tools"])
def test_repeated_continuation_has_a_small_finite_bound(name):
    module, tools = load_bundle(name), Tools()
    reply = assistant(native_call("search", query="again")) if name == "native_tools" else action("search", query="again")
    model = Model(lambda _: reply)
    module.configure(model, tools)
    with pytest.raises(RuntimeError, match="research step limit"):
        module.run_icp(ICP)
    assert len(tools.calls) == 4
    assert len(model.requests) == (3 if name == "bounded_hybrid" else 4)
