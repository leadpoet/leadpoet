"""No network or credentials: test the benchmark against real review parsing."""
import copy
from decimal import Decimal
import json

import pytest

from lab_arena import broker, code_review
from scripts import benchmark_arena_internal_models as harness
from scripts.verify_arena_live_capture import SharedBudget

MODEL = "deepseek/deepseek-v4.1-flash-1017"
PROVIDER = "nebius/fp8"
KEY = "sk-or-v1-" + "q" * 40


def metadata(model=MODEL, provider=PROVIDER):
    return {"data": {"id": model, "endpoints": [{
        "tag": provider, "name": "Nebius | deepseek/deepseek-v4.1-flash-20261017",
        "provider_name": "Nebius", "context_length": 1048576,
        "max_completion_tokens": 943718, "status": 0, "uptime_last_30m": 100,
        "supported_parameters": ["reasoning", "response_format", "max_tokens"],
        "pricing": {"prompt": "0.0000003", "completion": "0.0000012"},
    }]}}


def prepared(name="benign_agent"):
    case = harness.load_cases([name])[0]
    route = harness.validate_route(metadata(), MODEL, PROVIDER)
    request = harness.prepare_case(case, model=MODEL, provider=PROVIDER, mode="low", route=route)
    return case, request, route


def reply(case, request, route):
    files = {item.path: item.content for item in request.files}
    classification = case["expected"]["research_classification"]
    claims = {"model_directed": ["observation_to_action", "finish_continue"],
              "workflow": ["fixed_control"], "unverified": ["uncertainty"]}[classification]
    binding = json.loads(request.parameters["messages"][1]["content"])["research_control"]
    findings = [{"category": category, "file": "harness.py", "evidence": files["harness.py"].strip(),
                 "explanation": "The reachable entrypoint demonstrates this finding."}
                for category in case["expected"]["categories"]]
    document = {"verdict": case["expected"]["verdict"], "reviewed_files": list(files), "findings": findings,
                "research_control": {**binding, "classification": classification,
                    "evidence": [{"claim": claim, "file": "harness.py", "excerpt": files["harness.py"].strip(),
                                  "explanation": "Source data flow supports this claim."} for claim in claims],
                    "limitations": ["Source-only review does not attest runtime behavior."]}}
    return {"model": route["canonical_model"], "provider": route["provider_name"],
            "usage": {"cost": "0.00001", "prompt_tokens": 1000, "completion_tokens": 200},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(document),
                                                                  "reasoning": "Inspect the reachable source data flow."}}],
            "openrouter_metadata": {"requested": MODEL, "strategy": "direct", "attempt": 1, "pipeline": [],
                "endpoints": {"total": 1, "available": [{"selected": True, "model": route["canonical_model"],
                                                          "provider": route["provider_name"]}]}}}


class Transport:
    def __init__(self, response=None, fail=False):
        self.response, self.fail, self.sent = response, fail, []

    def send(self, **kwargs):
        self.sent.append(kwargs)
        if self.fail:
            raise TimeoutError("no network was used")
        return broker.ProviderResponse(200, {}, json.dumps(self.response).encode())


def run(tmp_path, mutate=None, name="benign_agent", fail=False):
    case, request, route = prepared(name)
    response = reply(case, request, route)
    if mutate:
        mutate(response)
    wire = Transport(response, fail=fail)
    budget = SharedBudget(tmp_path / "budget.json", Decimal("20"))
    result = harness.run_live_case(case, request, route, key=KEY, budget=budget,
                                   private_dir=tmp_path, transport=wire)
    return result, json.loads((tmp_path / "budget.json").read_text()), wire


def test_offline_is_network_free_and_never_imports_malicious_source(monkeypatch, capsys):
    monkeypatch.setattr(harness, "fetch_metadata", lambda *_: pytest.fail("network"))
    monkeypatch.setattr(harness, "run_live_case", lambda *_a, **_k: pytest.fail("paid dispatch"))
    assert harness.main(["--model", MODEL, "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["cases"]) == 18
    assert report["metadata_verified"] is False
    assert {"dynamic_malicious", "credential_exfiltration", "defensive_patterns"}.issubset({c["case"] for c in report["cases"]})
    assert all(c["max_reservation_usd"] is None for c in report["cases"])


@pytest.mark.parametrize("model,provider", harness.MODELS.items())
def test_original_strict_schema_full_source_and_security_policy_preserved(model, provider):
    case = harness.load_cases(["prompt_injection"])[0]
    request = harness.prepare_case(case, model=model, provider=provider, mode="high")
    baseline = harness.prepare_case(case, model=code_review.DEFAULT_REVIEW_MODEL, provider="google-vertex", mode="high")
    assert request.parameters["response_format"] == baseline.parameters["response_format"]
    assert request.parameters["response_format"]["json_schema"]["strict"] is True
    assert request.parameters["messages"] == baseline.parameters["messages"]
    assert request.parameters["provider"] == {"only": [provider], "allow_fallbacks": False,
        "require_parameters": True, "data_collection": "deny", "zdr": True}
    assert request.parameters["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert request.parameters["transforms"] == []
    if model == code_review.DEFAULT_REVIEW_MODEL:
        assert "reasoning" not in request.parameters
    else:
        assert request.parameters["reasoning"] == {"effort": "high", "exclude": False}
    assert request.parameters["max_tokens"] == 32768
    assert request.context_window_tokens == 1000000
    submission = json.loads(request.parameters["messages"][1]["content"])
    assert {f["path"]: f["content"] for f in submission["submission_files"]} == case["files"]
    assert request.source_archive_sha256 == baseline.source_archive_sha256
    assert all(category in request.parameters["messages"][0]["content"] for category in code_review.REVIEW_CATEGORIES)
    assert "expected" not in submission and "rationale" not in submission


@pytest.mark.parametrize("mode", ["low", "high"])
def test_sonnet_baseline_is_exact_production_request_except_budget_price_cap(monkeypatch, mode):
    original_prepare = code_review.prepare_request
    originals = []

    def observe(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        originals.append(copy.deepcopy(result.parameters))
        return result

    monkeypatch.setattr(code_review, "prepare_request", observe)
    case = harness.load_cases(["benign_agent"])[0]
    route = {"pricing": {"prompt": "0.0000022", "completion": "0.000011", "request": "0"}}
    request = harness.prepare_case(case, model=code_review.DEFAULT_REVIEW_MODEL,
                                   provider="google-vertex", mode=mode, route=route)
    actual = copy.deepcopy(request.parameters)
    assert actual["provider"].pop("max_price") == {"prompt": 2.2, "completion": 11.0, "request": 0.0}
    assert actual == originals[0]


@pytest.mark.parametrize("field,value,error", [("status", -2, "route_unhealthy"),
    ("context_length", 200000, "route_context_below_policy"),
    ("max_completion_tokens", 8192, "route_output_below_policy"),
    ("supported_parameters", ["max_tokens"], "route_parameters_unsupported"),
    ("tag", "other", "route_unavailable")])
def test_metadata_fails_closed(field, value, error):
    doc = metadata()
    doc["data"]["endpoints"][0][field] = value
    with pytest.raises(ValueError, match=error):
        harness.validate_route(doc, MODEL, PROVIDER)


def test_google_aggregate_alias_bounds_every_region():
    doc = metadata(model=code_review.DEFAULT_REVIEW_MODEL, provider="google-vertex/global")
    second = copy.deepcopy(doc["data"]["endpoints"][0])
    second.update(tag="google-vertex/us", pricing={"prompt": "0.0000022", "completion": "0.000011"})
    doc["data"]["endpoints"].append(second)
    route = harness.validate_route(doc, code_review.DEFAULT_REVIEW_MODEL, "google-vertex")
    assert route["pricing"]["prompt"] == "0.0000022"


def test_reservation_covers_complete_wire_input_output_and_request_fee():
    _, request, route = prepared()
    assert request.parameters["reasoning"] == {"effort": "low", "exclude": False}
    route["pricing"]["request"] = "0.02"
    expected = (Decimal(request.input_tokens_upper_bound) * Decimal("0.0000003")
                + Decimal(32768) * Decimal("0.0000012") + Decimal("0.02"))
    assert harness.reservation_usd(request, route) == expected
    assert Decimal(str(request.parameters["provider"]["max_price"]["completion"])) == Decimal("1.2")


def test_success_uses_actual_parser_and_settles_provider_bill(tmp_path):
    result, state, wire = run(tmp_path)
    assert result["schema_valid"] and result["route_verified"] and result["accuracy_match"]
    assert result["cost_status"] == "settled"
    assert state["spent_usd"] == "0.00001" and Decimal(state["reserved_usd"]) == 0
    assert result["reasoning"]["full_text_returned"]
    assert len(wire.sent) == 1
    assert (tmp_path / "benign_agent.request.json").stat().st_mode & 0o077 == 0
    assert KEY not in (tmp_path / "benign_agent.response.json").read_text()


def mutate_content(response, change):
    document = json.loads(response["choices"][0]["message"]["content"])
    change(document)
    response["choices"][0]["message"]["content"] = json.dumps(document)


@pytest.mark.parametrize("change,reason", [
    (lambda d: d["reviewed_files"].pop(), "coverage"),
    (lambda d: d["research_control"]["evidence"][0].update(excerpt="fabricated source excerpt absent in the source"), "research_evidence_mismatch"),
    (lambda d: d.update(verdict="reject", findings=[{"category": "malicious_behavior", "file": "harness.py", "evidence": "not exact source evidence", "explanation": "bad"}]), "finding_evidence_mismatch"),
])
def test_incomplete_coverage_and_invented_evidence_are_not_accuracy_results(tmp_path, change, reason):
    result, state, _ = run(tmp_path, lambda response: mutate_content(response, change))
    assert result["schema_valid"] is False
    assert result["schema_error"] == reason
    assert result["accuracy_match"] is None
    assert state["spent_usd"] == "0.00001"


def test_transport_timeout_retains_full_reservation(tmp_path):
    result, state, wire = run(tmp_path, fail=True)
    assert result["cost_status"] == "held" and result["transport_ok"] is False
    assert Decimal(state["reserved_usd"]) == Decimal(result["reserved_usd"])
    assert len(wire.sent) == 1


def test_missing_or_excess_cost_retains_reservation(tmp_path):
    result, state, _ = run(tmp_path, lambda r: r["usage"].update(cost="100"))
    assert result["cost_status"] == "held"
    assert Decimal(state["spent_usd"]) == 0
    assert Decimal(state["reserved_usd"]) > 0


def test_no_reported_cost_does_not_invent_a_bill(tmp_path):
    result, state, _ = run(tmp_path, lambda r: r["usage"].pop("cost"))
    assert result["schema_valid"] and result["cost_status"] == "held"
    assert Decimal(state["reserved_usd"]) > 0


def test_no_dispatch_over_shared_budget(tmp_path):
    case, request, route = prepared()
    budget = SharedBudget(tmp_path / "budget.json", Decimal("20"))
    budget.reserve(Decimal("20"))
    wire = Transport()
    with pytest.raises(ValueError, match="cumulative_budget_exceeded"):
        harness.run_live_case(case, request, route, key=KEY, budget=budget, private_dir=tmp_path, transport=wire)
    assert wire.sent == []


@pytest.mark.parametrize("amount", ["20.01", "NaN", "Infinity", "0", "-1"])
def test_cap_cannot_be_raised_even_offline(amount):
    with pytest.raises(SystemExit):
        harness.main(["--model", MODEL, "--budget-usd", amount])


def test_route_failure_separate_from_schema_and_accuracy(tmp_path):
    result, _, _ = run(tmp_path, lambda r: r["openrouter_metadata"].update(pipeline=[{"id": "context-compression"}]))
    assert result["schema_valid"] and result["accuracy_match"]
    assert result["route_verified"] is False


def test_private_production_source_does_not_infer_historical_gold(tmp_path):
    case, request, _ = prepared()
    descriptor = tmp_path / "production.json"
    descriptor.write_text(json.dumps({"id": "saved_production", "files": case["files"], "expected": {
        "verdict": None, "categories": None, "research_classification": None}}))
    loaded = harness.load_cases([str(descriptor)])[0]
    response = reply(case, request, harness.validate_route(metadata(), MODEL, PROVIDER))
    response["model"] = MODEL
    parsed = code_review.parse_response(response, request)
    assert harness.score_review(loaded, parsed)["accuracy_match"] is None


def test_existing_ten_research_labels_retain_their_causal_controls():
    # Only existing benign research fixtures run under injected model/tools.
    # New security fixture source remains text and is never executed.
    cases = [case for case in harness.load_cases() if "research_case" in case]
    assert len(cases) == 10
    for case in cases:
        checked = harness.research.verify_causality(case["research_case"])
        assert checked["causal_check"] == "passed"
        assert checked["expected"] == case["expected"]["research_classification"]
