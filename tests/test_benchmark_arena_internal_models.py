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
    def change(r):
        r["model"] = MODEL
        r["openrouter_metadata"].update(pipeline=[{"id": "context-compression"}])
    result, _, _ = run(tmp_path, change)
    assert result["schema_valid"] and result["accuracy_match"]
    assert result["route_verified"] is False


def test_unverified_route_cannot_normalize_a_canonical_alias(tmp_path):
    result, _, _ = run(tmp_path, lambda r: r["openrouter_metadata"].update(attempt=2))
    assert result["canonical_envelope_changed"] is False
    assert result["raw_parser"]["schema_error"] == "model_mismatch"
    assert result["canonical_envelope_parser"]["schema_valid"] is False


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


def saved_run(tmp_path, *, with_manifest=True):
    case, request, route = prepared()
    result, _, _ = run(tmp_path)
    result.update(harness.case_identity(case))
    (tmp_path / 'benign_agent.summary.json').write_text(json.dumps(result))
    (tmp_path / 'fixed-labels.json').write_text(json.dumps([{k:v for k,v in case.items() if k != '_base'}]))
    (tmp_path / 'endpoint-metadata.json').write_text(json.dumps(metadata()))
    if with_manifest:
        (tmp_path / 'manifest.json').write_text(json.dumps(harness.frozen_manifest([case], [request], route, 'original')))
    return case, request, route


def test_prompt_variant_is_generic_bounded_and_preserves_full_source_schema():
    case, original, route = prepared()
    audit = harness.prepare_case(case, model=MODEL, provider=PROVIDER, mode='medium', route=route,
                                 prompt_variant='backward-evidence-audit')
    assert audit.parameters['messages'][1:] == original.parameters['messages'][1:]
    assert audit.parameters['response_format'] == original.parameters['response_format']
    assert audit.parameters['reasoning'] == {'effort': 'medium', 'exclude': False}
    appendix = harness.PROMPTS['backward-evidence-audit']
    assert len(appendix) < 2000
    assert all(c['id'] not in appendix for c in harness.load_cases())
    assert audit.parameters['messages'][0]['content'] == original.parameters['messages'][0]['content'] + appendix
    assert audit.input_tokens_upper_bound == code_review.REQUEST_TOKEN_OVERHEAD + len(harness.contracts.canonical_json(audit.parameters).encode())
    assert harness.reservation_usd(audit, route) > harness.reservation_usd(original, route)


def test_registry_model_outside_old_four_is_selectable_offline(monkeypatch, capsys):
    monkeypatch.setattr(harness, 'fetch_metadata', lambda *_: pytest.fail('network'))
    assert harness.main(['--model', 'xiaomi/mimo-v2.6-flash', '--provider', 'deepinfra/fp8',
                         '--mode', 'medium', '--cases', 'benign_agent', '--dry-run']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['model'] == 'xiaomi/mimo-v2.6-flash'
    assert report['reasoning_mode'] == 'medium'
    assert report['manifest']['cases'][0]['request_sha256'].startswith('sha256:')


def test_unapproved_route_fails_before_metadata_or_dispatch(monkeypatch):
    monkeypatch.setattr(harness, 'fetch_metadata', lambda *_: pytest.fail('network'))
    with pytest.raises(ValueError, match='route_not_registry_authorized'):
        harness.main(['--model', MODEL, '--provider', 'unapproved/fp8', '--live'])


@pytest.mark.parametrize('change,error', [
    (lambda e: e.update(name='Nebius | deepseek/wrong-model'), 'registry_endpoint_identity_mismatch'),
    (lambda e: e.update(provider_name='Another Provider'), 'registry_endpoint_identity_mismatch'),
    (lambda e: e['pricing'].update(prompt='0.01'), 'route_price_above_registry_ceiling'),
    (lambda e: e.update(supports_zdr=False), 'route_privacy_unsupported'),
])
def test_fresh_endpoint_identity_privacy_and_price_gates(change, error):
    document = metadata()
    change(document['data']['endpoints'][0])
    with pytest.raises(ValueError, match=error):
        harness.validate_route(document, MODEL, PROVIDER)


def test_raw_and_canonical_parser_results_are_separate(tmp_path):
    result, _, _ = run(tmp_path)
    assert not result['raw_parser']['schema_valid']
    assert result['raw_parser']['schema_error'] == 'model_mismatch'
    assert result['canonical_envelope_parser']['schema_valid']
    assert result['canonical_envelope_changed']
    assert result['schema_parser'] == 'canonical_envelope_parser'


def test_private_or_stale_request_bound_cannot_dispatch(tmp_path):
    from dataclasses import replace
    case, request, route = prepared()
    wire = Transport()
    budget = SharedBudget(tmp_path / 'budget.json', Decimal('20'))
    parameters = copy.deepcopy(request.parameters)
    parameters['provider']['zdr'] = False
    with pytest.raises(ValueError, match='request_privacy_or_route_controls_invalid'):
        harness.run_live_case(case, replace(request, parameters=parameters), route,
                              key=KEY, budget=budget, private_dir=tmp_path, transport=wire)
    parameters = copy.deepcopy(request.parameters)
    parameters['messages'][0]['content'] += '\nExtra input must be reserved.'
    with pytest.raises(ValueError, match='request_bound_mismatch'):
        harness.run_live_case(case, replace(request, parameters=parameters), route,
                              key=KEY, budget=budget, private_dir=tmp_path, transport=wire)
    assert wire.sent == []
    assert Decimal(json.loads((tmp_path / 'budget.json').read_text())['reserved_usd']) == 0


def test_replay_is_network_key_and_budget_free_and_preserves_saved_bytes(tmp_path, monkeypatch, capsys):
    saved_run(tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(harness, 'fetch_metadata', lambda *_: pytest.fail('network'))
    monkeypatch.setattr(harness, 'SharedBudget', lambda *_: pytest.fail('budget access'))
    monkeypatch.setattr(harness, 'run_live_case', lambda *_a, **_k: pytest.fail('paid dispatch'))
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    assert harness.main(['--replay-dir', str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['metrics']['raw_parser_valid'] == 0
    assert report['metrics']['canonical_envelope_parser_valid'] == 1
    assert report['results'][0]['saved_manifest_verified'] is True
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_legacy_replay_uses_saved_source_without_archive_or_source_execution(tmp_path, monkeypatch):
    saved_run(tmp_path, with_manifest=False)
    labels = json.loads((tmp_path / 'fixed-labels.json').read_text())
    labels[0].pop('files')
    labels[0]['source_archive'] = '/not-present/do-not-read.tar.gz'
    (tmp_path / 'fixed-labels.json').write_text(json.dumps(labels))
    monkeypatch.setattr(harness.research, 'prepared_case', lambda *_: pytest.fail('source execution'))
    results, manifest = harness.replay_directory(tmp_path)
    assert results[0]['schema_valid'] and results[0]['accuracy_match']
    assert manifest['input_manifest_verified'] is False


def test_replay_rejects_request_or_gold_tampering(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / 'fixed-labels.json'
    labels = json.loads(path.read_text())
    labels[0]['expected']['verdict'] = 'reject'
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match='replay_request_or_labels_hash_invalid'):
        harness.replay_directory(tmp_path)


def test_replay_never_accepts_invented_source_excerpts(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / 'benign_agent.response.json'
    document = json.loads(path.read_text())
    mutate_content(document, lambda d: d['research_control']['evidence'][0].update(excerpt='An invented excerpt absent from source.'))
    path.write_text(json.dumps(document))
    # Legacy recordings lack a saved response hash; the real source-aware
    # parser must still reject invented evidence after envelope normalization.
    summary_path = tmp_path / 'benign_agent.summary.json'
    summary = json.loads(summary_path.read_text())
    summary.pop('response_sha256')
    summary_path.write_text(json.dumps(summary))
    results, _ = harness.replay_directory(tmp_path)
    assert not results[0]['schema_valid']
    assert results[0]['canonical_envelope_parser']['schema_error'] == 'research_evidence_mismatch'


def test_replay_rejects_changed_hashed_response(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / 'benign_agent.response.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='replay_response_hash_invalid'):
        harness.replay_directory(tmp_path)


def test_manifest_freezes_request_gold_split_family_and_prompt(tmp_path):
    case, request, route = prepared()
    case.update(split='holdout', source_family='independent-source-family')
    manifest = harness.frozen_manifest([case], [request], route, 'original')
    identity = manifest['cases'][0]
    assert identity['split'] == 'holdout' and identity['source_family'] == 'independent-source-family'
    assert identity['labels_sha256'] == harness.contracts.document_hash(case['expected'])
    assert identity['request_sha256'] == harness.contracts.hash_bytes(harness.contracts.canonical_json(request.parameters).encode())
    assert manifest['manifest_sha256'] == harness.contracts.document_hash({k:v for k,v in manifest.items() if k != 'manifest_sha256'})


def test_metrics_do_not_turn_schema_failures_into_correct_rejections():
    result = {'case':'negative', 'expected_verdict':'reject', 'schema_valid':False, 'accuracy_match':None,
              'transport_ok':True, 'route_verified':True, 'cost_status':'held', 'reserved_usd':'0.1', 'latency_seconds':2}
    metrics = harness.summarize([result])
    assert metrics['known_negative_cases'] == 1
    assert metrics['unsafe_accepts'] == 0
    assert metrics['labelled_parsed_cases'] == metrics['accuracy_matched'] == 0
    assert metrics['unknown_charge_reserved_usd'] == '0.1'


def test_source_family_cannot_leak_across_development_and_holdout(tmp_path):
    case, _, _ = prepared()
    first = {k:v for k,v in case.items() if k != '_base'}
    first.update(source_family='same-family', split='development')
    second = {**first, 'id':'renamed_copy', 'split':'holdout'}
    path = tmp_path / 'leaking-split.json'
    path.write_text(json.dumps({'cases':[first, second]}))
    with pytest.raises(ValueError, match='source_family_crosses_splits'):
        harness.load_cases([str(path)])


def test_replay_verifies_frozen_metadata_snapshot(tmp_path):
    case, request, route = saved_run(tmp_path)
    manifest = harness.frozen_manifest([case], [request], route, 'original', metadata())
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    changed = metadata()
    changed['data']['endpoints'][0]['uptime_last_30m'] = 99
    (tmp_path / 'endpoint-metadata.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='replay_metadata_hash_invalid'):
        harness.replay_directory(tmp_path)
