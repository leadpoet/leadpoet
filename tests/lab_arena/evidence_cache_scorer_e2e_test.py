"""Native Arena scorer parity with held network replies and authority transport.

The real scorer entrypoint, company-fit verifier, signal verifier, claim
comparison and score aggregation run. Only provider replies and the durable
host transport are local substitutes. This does not test production SQL.
The held provider replies preserve the historical protocol. Current provider
and thinking-capture flows use their own explicit frozen policy.
"""
from copy import deepcopy
import json as json_module
import os
from collections import Counter

from lab_arena import contracts, evidence_judgments as ej, scorer_entrypoint, scoring, signal_cache_control, signal_judgments
from qualification.scoring import lead_scorer
from tests.lab_arena.legacy_company_judgments_test import _lease
from tests.lab_arena.scorer_environment_fixture import isolated_scorer_environment
from tests.lab_arena.judge_default_activation_test import LEGACY_MODELS
from tests.test_arena_evidence_signal_bridge import _setup

ABOUT = "https://epam.com/about"
ABOUT_TEXT = ("EPAM Systems supplies Information Technology services. "
              "EPAM Systems has 10,001+ employees. "
              "EPAM Systems is headquartered in the United States.")
HTML = ('<html><head><title>EPAM Systems</title><script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"Organization","name":"EPAM Systems",'
        '"url":"https://epam.com/","sameAs":["https://www.linkedin.com/company/epam-systems"]}'
        '</script></head><body>' + ABOUT_TEXT +
        '<a href="https://www.linkedin.com/company/epam-systems">LinkedIn</a></body></html>')


class FitAndSignalTransport:
    """Memory transport only; real host identities/admission decide reuse."""
    def __init__(self):
        self.rows, self.requests, self.uses, self.native_receipts = {}, [], [], []
        self.run, self.current_input = "baseline", None
        self.fit_source_suffix = ""

    @property
    def fit_rows(self):
        return {key: row for key, row in self.rows.items() if row["scope"].get("checkpoint_kind") == ej.FIT_KIND}

    @property
    def signal_rows(self):
        return {key: row for key, row in self.rows.items() if row["scope"].get("judgment_kind") == ej.SIGNAL_KIND}

    async def __call__(self, request):
        self.requests.append(deepcopy(request))
        scope = ej.build_scope(scoring_input=self.current_input,
            company_index=request["company_index"], signal_index=request["signal_index"],
            round_id="arena-evidence-e2e", network_name="finney", netuid=71,
            scorer_image_digest="sha256:"+"a"*64, scorer_image_reference="registry/scorer@sha256:"+"a"*64,
            runtime_image_digest="sha256:"+"a"*64, runtime_image_reference="registry/scorer@sha256:"+"a"*64,
            **{key: request[key] for key in ("evidence_input", "evidence_hash", "verified_company_identity",
                "verified_funding_stage_context", "verified_source_pages") if key in request})
        operation = request["operation"]
        if operation == "candidates":
            return {"candidates": [{"reference": row["reference"], "judgment": deepcopy(row["judgment"])}
                for row in self.rows.values() if row["scope"]["family_hash"] == scope["family_hash"]
                and ("evidence_hash" not in scope or row["scope"]["evidence_hash"] == scope["evidence_hash"])]}
        reference = request.get("candidate_reference") or signal_judgments.reference(scope, 0)
        row = self.rows.get(reference["cache_key"])
        if operation == "claim":
            if row is not None:
                assert row["scope"]["family_hash"] == scope["family_hash"]
                assert row["scope"]["evidence_hash"] == scope["evidence_hash"]
                ej.validate_equivalence(request.get("equivalence_receipt"), row["scope"]["claim"],
                                        scope["claim"], self.current_input["scorer_policy"])
                ej.validate_judgment(row["judgment"], row["scope"])
                self.uses.append((self.run, row["source_run"], deepcopy(reference)))
                return {"status": "hit", "reference": reference, "judgment": deepcopy(row["judgment"])}
            return {"status": "miss", "reference": reference}
        assert operation == "complete" and request["reference"] == reference
        if request["judgment"] is None:
            return {"status": "released", "reference": reference}
        judgment = ej.validate_judgment(request["judgment"], scope)
        self.rows.setdefault(reference["cache_key"], {"reference": reference, "scope": scope,
            "judgment": judgment, "source_run": self.run})
        self.uses.append((self.run, self.run, deepcopy(reference)))
        return {"status": "stored", "reference": reference}


class Body:
    def __init__(self, raw): self.raw = raw
    async def read(self, size=-1):
        if size < 0: size = len(self.raw)
        value, self.raw = self.raw[:size], self.raw[size:]
        return value


class Response:
    status = 200
    def __init__(self, *, url, body=None, text=""):
        self.url, self.body, self.raw_text = url, body, text
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self.content = Body(text.encode())
    async def __aenter__(self): return self
    async def __aexit__(self, *_args): return None
    async def json(self): return deepcopy(self.body)
    async def text(self): return self.raw_text


def setup_native(monkeypatch, *, paragraph=False, judge_models=LEGACY_MODELS,
                 judge_provider_routes=None):
    if judge_provider_routes is None and dict(judge_models) == LEGACY_MODELS:
        judge_provider_routes = {}
    fixture, fetch, judge, compare, _ = _setup(monkeypatch)
    host, calls = FitAndSignalTransport(), Counter()
    host.paragraph_inputs = []
    from qualification.scoring import intent_details, verification_helpers
    async def language_provider(prompt, **kwargs):
        schema = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
        if schema == "arena_intent_details_review_v3":
            from tests.test_arena_intent_details_grounding import _review_response
            calls["paragraph_judge"] += 1
            document = json_module.loads(prompt)
            host.paragraph_inputs.append(deepcopy(document))
            return json_module.dumps(_review_response(
                {name: True for name in intent_details._CHECKS},
                [{"matched_icp_signal": 0, "covered": True}], document))
        return await compare(prompt, **kwargs)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", language_provider)
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    fit_reply = {
        "observed_company_name": "EPAM Systems", "observed_company_website": "https://epam.com/",
        "observed_company_linkedin": fixture["company"]["company_linkedin"],
        "observed_employee_count": "10,001+", "employee_size_matches": True,
        "employee_size_evidence_url": ABOUT, "employee_size_evidence_quote": "EPAM Systems has 10,001+ employees.",
        "observed_industry": "Information Technology", "observed_subindustry": "IT services",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": ABOUT, "industry_evidence_quote": "EPAM Systems supplies Information Technology services.",
        "observed_hq_country": "United States", "observed_hq_state": "",
        "geography_matches": True, "geography_evidence_url": ABOUT,
        "geography_evidence_quote": "EPAM Systems is headquartered in the United States.",
        "reason": "The supplied independent source facts match the requested fit criteria.",
    }
    class Session:
        def __init__(self, *_args, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None
        def get(self, url, **kwargs):
            calls["fit_source_fetch"] += 1
            return Response(url=url, text=HTML + host.fit_source_suffix)
        def post(self, url, *, json=None, **kwargs):
            if url == "https://openrouter.ai/api/v1/chat/completions" and "response_format" in json:
                calls["fit_industry_judge"] += 1
                pages = [page for message in json["messages"][2:]
                         for page in json_module.loads(message["content"])["untrusted_source_pages"]]
                spans = [span["span_id"] for page in pages for span in page["spans"]]
                answer = {"observed_activity": "Information Technology services", "reason": "The supplied company page states its IT services.",
                          "missing_requirements": [], "role": "supplier_operator", "evidence_span_ids": spans[:8], "decision": "match"}
                if "research_query" in json["response_format"]["json_schema"]["schema"]["properties"]:
                    answer["research_query"] = ""
                return Response(url=url, body={"choices": [{"finish_reason": "stop", "message": {"content": json_module.dumps(answer)}}]})
            if url == "https://openrouter.ai/api/v1/chat/completions" and "tools" in json:
                calls["fit_research"] += 1
                finding = {"target": "industry", "status": "VERIFIED", "observed_industry": "Information Technology services",
                           "activity_role": "supplier_operator", "evidence_url": "https://epam.com/",
                           "evidence_quote": "EPAM Systems supplies Information Technology services.", "reason": "The exact first-party quote states this offering."}
                return Response(url=url, body={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{
                    "id": "submit-fit", "type": "function", "function": {"name": "submit_findings", "arguments": json_module.dumps({"findings": [finding]})}}]}}]})
            if url == "https://openrouter.ai/api/v1/chat/completions":
                calls["fit_judge"] += 1
                return Response(url=url, body={"choices": [{"message": {"content": json_module.dumps(fit_reply)}}]})
            if url == "https://code.deepline.com/api/v2/integrations/harvestapi_get_company/execute":
                calls["structured_profile_fetch"] += 1
                return Response(url=url, body={"status": 200, "element": {
                    "name": "EPAM Systems", "website": "https://epam.com/",
                    "linkedinUrl": "https://www.linkedin.com/company/epam-systems/",
                    "employeeCountRange": {"start": 10001, "end": None}, "companyType": "Public Company",
                    "description": ABOUT_TEXT,
                    "locations": [{"headquarter": True, "country": "US", "city": "Newtown", "geographicArea": "Pennsylvania",
                        "parsed": {"text": "Newtown, Pennsylvania, United States", "city": "Newtown", "state": "Pennsylvania", "country": "United States", "countryCode": "US"}}],
                }})
            if url == "https://api.exa.ai/contents":
                calls["profile_fetch"] += 1
                profile = json["ids"][0]
                return Response(url=url, body={"results": [{"url": profile, "text":
                    "# EPAM Systems\n## About\nWebsite\nhttps://epam.com/\nCompany size\n10,001+ employees\nHeadquarters\nNewtown, Pennsylvania, United States"}],
                    "statuses": [{"id": profile, "status": "success"}]})
            raise AssertionError("Unexpected provider route: " + url)
    monkeypatch.setattr(lead_scorer.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(signal_cache_control, "exchange", host)
    native = lead_scorer._verify_company_fit
    async def observe(*args, **kwargs):
        result = await native(*args, **kwargs)
        calls["native_fit"] += 1
        host.native_receipts.append(result.receipt("company_fit"))
        return result
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", observe)
    policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2",
        judge_models=judge_models, judge_provider_routes=judge_provider_routes,
        company_judgment_cache=True, evidence_judgment_cache=True, intent_details=paragraph)
    icp = deepcopy(fixture["icp"])
    icp.update(max_companies=1, company_stage="Any", industry="Information Technology",
        sub_industry="IT services", product_service="Information Technology services", required_attribute="",
        intent_signals=[fixture["target_signal_text"]], bonus_intents=[])
    company = {**fixture["company"], "industry": "Information Technology", "employee_count": "10,001+",
        "company_stage": "", "country": "United States", "state": "",
        "fit_summary": "The company supplies IT services.", "fit_evidence_urls": [ABOUT],
        "intent_signals": [{"matched_icp_signal": 0, "description": fixture["miner_claim"],
            "date": fixture["miner_date"], "url": fixture["source_url"], "snippet": fixture["miner_claim"],
            "why_now": "The announcement is current."}]}
    if paragraph:
        company.pop("fit_summary"); company.pop("fit_evidence_urls")
        company["intent_details"] = (
            "EPAM announced a partnership with Wiz on August 25, 2026 to help enterprises reduce cloud risk. "
            "The partnership connects Wiz's cloud security platform with EPAM's engineering and cloud modernization services, "
            "which is relevant to the requested IT services companies announcing technology alliances."
        )
        company["intent_signals"][0].pop("why_now")
        company["intent_signals"][0].pop("snippet")
    def run(*, row=None, run_id="baseline", claim=None, evidence_cache=True):
        current = deepcopy(company if row is None else row)
        if claim is not None:
            current["intent_signals"][0].update(description=claim, snippet=claim)
        host.run = run_id
        current_policy = policy if evidence_cache else scoring.build_scorer_policy(
            judge_models=judge_models, judge_provider_routes=judge_provider_routes,
            scoring_adapter_version="qualification_integrity_v2", company_judgment_cache=True, intent_details=paragraph)
        document = scoring.build_scoring_input(scored_run_id=run_id, icp=icp,
            companies=[current], policy=current_policy, evaluation_date="2026-10-24")
        host.current_input = deepcopy(document)
        document["company_judgment_cache"] = _lease(document)
        # Production uses a fresh scorer process. Keep its strict environment
        # checks enabled when comparing distinct frozen policies in one test.
        environment = dict(os.environ)
        for name in (*scoring.CREDENTIAL_ENV_NAMES, *policy["env_bindings"],
            "ARENA_INTENT_EVIDENCE_MODEL", "ARENA_INTENT_DETAILS_REVIEW_MODEL", "ARENA_ROLE_BATCH_MODEL",
            "ARENA_EVIDENCE_JUDGMENT_POLICY", "ARENA_INDUSTRY_EVIDENCE_MODEL"):
            environment.pop(name, None)
        environment.update(scorer_entrypoint.PLACEHOLDER_CREDENTIALS)
        with monkeypatch.context() as process:
            process.setattr(os, "environ", environment)
            output = scorer_entrypoint.score_input(document)
        return output, current
    return fixture, fetch, judge, compare, host, calls, policy, icp, run


def test_full_native_scorer_baseline(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    output, _ = run()
    assert "failure" not in output, (output, calls, host.requests)
    assert output["breakdowns"][0]["final_score"] > 0


def metrics(output, company, policy, icp):
    assert "failure" not in output, output
    run_id = output["scored_run_id"]
    plan = {"schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION, "round_id": "arena-evidence-e2e", "stage": 1,
        "work_items": [{"scored_run_id": run_id, "icp_position": 0, "submission_id": "participant",
                        "output_ref": "arena/e2e/output.json"}], "zero_rows": []}
    stage = scoring.build_stage_scores(plan=plan, policy=policy, icps_by_position={0: icp},
        outputs_by_run={run_id: [company]}, breakdowns_by_item={run_id: output["breakdowns"]},
        configuration={"stage_1_icp_count": 1, "stage_2_icp_count": 1})
    row = stage["rows"][0]
    breakdown = output["breakdowns"][0]
    return {"score": breakdown["final_score"], "qualified": breakdown["company_qualified"],
            "fp_gate": row["fp_gate_count"], "fp_primary": row["fp_unverified_primary_count"],
            "per_icp": row["per_icp_score"], "aggregate": stage["submission_scores"]["participant"]}


def test_full_native_baseline_and_rewritten_miner_keep_score_fp_qualification_and_aggregate(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    baseline, baseline_company = run()
    baseline_paid_judges = calls["fit_judge"] + calls["fit_industry_judge"] + calls["fit_research"] + judge.await_count + compare.await_count
    miner, miner_company = run(run_id="miner-rewrite",
        claim="EPAM partners with Wiz to help enterprises reduce cloud risk.")
    assert metrics(baseline, baseline_company, policy, icp) == metrics(miner, miner_company, policy, icp)
    assert baseline["breakdowns"] == miner["breakdowns"]
    observed = metrics(miner, miner_company, policy, icp)
    assert observed["score"] > 0 and observed["qualified"] is True
    assert observed["fp_gate"] == observed["fp_primary"] == 0
    # This known-positive source remains qualified: no new false negative.
    assert fetch.await_count == 2 and judge.await_count == compare.await_count == 1
    assert len(host.signal_rows) == 1
    signal_uses = [(run, source) for run, source, reference in host.uses
                   if host.rows[reference["cache_key"]]["scope"].get("judgment_kind") == ej.SIGNAL_KIND]
    assert signal_uses == [("baseline", "baseline"), ("miner-rewrite", "baseline")]
    # Native headcount/HQ projections and the completed industry reviewer own
    # every final fact. Fresh equal source content must avoid all fit judges.
    assert calls["native_fit"] == calls["fit_judge"] == calls["fit_industry_judge"] == 1
    assert len(host.fit_rows) == 1
    total_paid_judges = calls["fit_judge"] + calls["fit_industry_judge"] + calls["fit_research"] + judge.await_count + compare.await_count
    assert total_paid_judges - baseline_paid_judges == 1 < baseline_paid_judges


def test_full_native_source_change_requires_another_judge_and_keeps_original_authority(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    first, company = run()
    original = deepcopy(host.rows)
    fetch.return_value["results"][0]["text"] += " Additional current source content."
    second, second_company = run(run_id="miner-new-source")
    assert metrics(first, company, policy, icp) == metrics(second, second_company, policy, icp)
    assert judge.await_count == fetch.await_count == 2 and compare.await_count == 0
    assert len(host.signal_rows) == 2
    assert all(host.rows[key] == row for key, row in original.items())


def test_full_native_grounded_negative_reuses_original_zero_and_fp_penalty(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    first, company = run(claim=fixture["wrong_acquisition_claim"])
    second, second_company = run(run_id="miner-same-false-claim", claim=fixture["wrong_acquisition_claim"])
    assert metrics(first, company, policy, icp) == metrics(second, second_company, policy, icp)
    observed = metrics(first, company, policy, icp)
    assert observed["score"] == 0 and observed["qualified"] is False
    assert observed["fp_gate"] == 0 and observed["fp_primary"] == 1
    assert first["breakdowns"] == second["breakdowns"]
    assert judge.await_count == 1 and fetch.await_count == 2
    assert compare.await_count == 0 and len(host.signal_rows) == 1


def test_full_native_false_acquisition_cannot_inherit_partnership_approval(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    first, company = run()
    second, second_company = run(run_id="miner-different-event", claim=fixture["wrong_acquisition_claim"])
    positive, negative = metrics(first, company, policy, icp), metrics(second, second_company, policy, icp)
    assert positive["qualified"] is True and negative["qualified"] is False
    assert negative["score"] == 0 and negative["fp_primary"] == 1
    assert fetch.await_count == judge.await_count == 2 and len(host.signal_rows) == 2


def test_full_native_deterministic_fit_mismatch_reuses_receipt_without_paid_judge(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    _, company = run()
    host.rows.clear(); host.uses.clear(); calls.clear(); judge.reset_mock(); fetch.reset_mock(); compare.reset_mock()
    company["country"] = "Canada"
    first, first_company = run(row=company, run_id="baseline-fit-mismatch")
    second, second_company = run(row=company, run_id="miner-fit-mismatch", claim="Different signal wording.")
    assert metrics(first, first_company, policy, icp) == metrics(second, second_company, policy, icp)
    observed = metrics(first, first_company, policy, icp)
    assert observed["score"] == 0 and observed["fp_gate"] == 1 and observed["qualified"] is False
    assert first["breakdowns"][0]["verifier_gate_receipts"] == second["breakdowns"][0]["verifier_gate_receipts"]
    assert calls["native_fit"] == 1 and calls["fit_judge"] == judge.await_count == fetch.await_count == 0
    assert len(host.fit_rows) == 1 and host.signal_rows == {}


def test_full_native_provider_timeout_remains_retryable_and_does_not_poison_authority(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    native_reply = judge.side_effect
    judge.side_effect = None
    judge.return_value = {"_error": "timeout"}
    failed, company = run(run_id="baseline-provider-timeout")
    assert failed["failure"] == "judge_error"
    assert host.signal_rows == {} and judge.await_count >= 1
    assert not failed.get("breakdowns")
    judge.side_effect = native_reply
    recovered, recovered_company = run(row=company, run_id="baseline-provider-recovered")
    observed = metrics(recovered, recovered_company, policy, icp)
    assert observed["score"] > 0 and observed["qualified"] is True and observed["fp_primary"] == 0
    assert len(host.signal_rows) == 1
    assert host.signal_rows[next(iter(host.signal_rows))]["source_run"] == "baseline-provider-recovered"


def test_full_native_claim_comparison_timeout_uses_native_judge_without_false_zero(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    baseline, company = run()
    compare.side_effect = TimeoutError("optional comparison provider timed out")
    miner, miner_company = run(run_id="miner-comparison-timeout",
        claim="EPAM partners with Wiz to help enterprises reduce cloud risk.")
    assert metrics(baseline, company, policy, icp) == metrics(miner, miner_company, policy, icp)
    assert judge.await_count == fetch.await_count == 2 and compare.await_count == 1
    assert len(host.signal_rows) == 2
    assert metrics(miner, miner_company, policy, icp)["qualified"] is True



def test_full_native_changed_company_source_requires_fit_judgment_and_appends_variant(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    baseline, company = run()
    assert len(host.fit_rows) == 1
    original = deepcopy(host.fit_rows)
    host.fit_source_suffix = " The company published another current source detail."
    miner, miner_company = run(run_id="miner-new-company-evidence")
    assert metrics(baseline, company, policy, icp) == metrics(miner, miner_company, policy, icp)
    assert calls["native_fit"] == calls["fit_judge"] == calls["fit_industry_judge"] == 2
    assert len(host.fit_rows) == 2
    assert all(host.fit_rows[key] == row for key, row in original.items())



def test_historical_native_cold_baseline_matches_prospective_cold_and_reused_scores(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch)
    historical_policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2",
        company_judgment_cache=True)
    historical, historical_company = run(run_id="historical-native", evidence_cache=False)
    assert host.requests == []
    cold, cold_company = run(run_id="baseline-v2-cold")
    reused, reused_company = run(run_id="miner-v2-hit", claim="EPAM partners with Wiz to help enterprises reduce cloud risk.")
    expected = metrics(historical, historical_company, historical_policy, icp)
    assert expected == metrics(cold, cold_company, policy, icp) == metrics(reused, reused_company, policy, icp)
    def native_breakdowns(output):
        rows = deepcopy(output["breakdowns"])
        for row in rows:
            for signal in row.get("intent_signals_detail") or []:
                signal.pop("signal_judgment_reference", None)
        return rows
    assert native_breakdowns(historical) == native_breakdowns(cold) == native_breakdowns(reused)
    assert expected["qualified"] is True and expected["score"] > 0
    assert calls["native_fit"] == calls["fit_judge"] == calls["fit_industry_judge"] == 2
    assert judge.await_count == 2 and compare.await_count == 1 and fetch.await_count == 3



def test_v5_original_paragraph_keeps_historical_cold_and_prospective_reuse_parity(monkeypatch):
    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch, paragraph=True)
    historical_policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2",
        company_judgment_cache=True, intent_details=True)
    historical, first_company = run(run_id="historical-v5-native", evidence_cache=False)
    cold, second_company = run(run_id="baseline-v5-cold")
    reused, third_company = run(run_id="miner-v5-hit")
    expected = metrics(historical, first_company, historical_policy, icp)
    assert expected == metrics(cold, second_company, policy, icp) == metrics(reused, third_company, policy, icp)
    def native_breakdowns(output):
        rows = deepcopy(output["breakdowns"])
        for row in rows:
            for signal in row.get("intent_signals_detail") or []:
                signal.pop("signal_judgment_reference", None)
        return rows
    assert native_breakdowns(historical) == native_breakdowns(cold) == native_breakdowns(reused)
    assert expected["score"] > 0 and expected["qualified"] is True
    assert calls["paragraph_judge"] == 3
    assert host.paragraph_inputs[0] == host.paragraph_inputs[1] == host.paragraph_inputs[2]
    assert calls["native_fit"] == calls["fit_judge"] == 2 and judge.await_count == 2
    assert len(host.fit_rows) == len(host.signal_rows) == 1
