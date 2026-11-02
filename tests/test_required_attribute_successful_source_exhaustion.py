"""A local source limit must not turn exhausted evidence into a judge outage."""
from __future__ import annotations

import asyncio
import copy
import hashlib
from types import SimpleNamespace

import pytest

from gateway.qualification.models import CompanyOutput
from lab_arena import scorer_entrypoint, scoring, verify
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match, company_fit_unavailable
from qualification.scoring.competition import scorer_breakdown_has_retryable_infrastructure_failure

REAL_REQUEST = lead_scorer._request_company_reverify_json

ROOT = "https://plantservices.com/"
ARTICLE = ROOT + "industry-news/voortman-expansion"
REPAIR = ROOT + "about-us"


def _company():
    # Reduced retained-input shape: the name and submitted domain do not bind.
    return CompanyOutput.model_construct(
        company_name="Voortman Steel Machinery", company_website=ROOT,
        company_linkedin="", industry="Industrial Machinery Manufacturing",
        sub_industry="", employee_count="201-500", company_stage="Series B",
        country="United States", state="", description="", intent_details=None,
        fit_evidence_urls=[], company_stage_evidence=[], intent_signals=[],
        required_attribute=None,
    )


def _icp():
    # Synthetic criteria; no private benchmark content is embedded in this test.
    return SimpleNamespace(
        industry="Industrial Machinery Manufacturing", sub_industry="",
        product_service="Industrial machinery", employee_count="201-500",
        company_stage="Series B", country="United States", geography="United States",
        required_attribute="Announced an office opening.",
    )


def _verdict(url=ROOT):
    return {
        "observed_company_name": "Plant Services", "observed_company_website": ROOT,
        "observed_company_linkedin": "https://www.linkedin.com/company/plant-services",
        "observed_employee_count": None, "employee_size_matches": None,
        "employee_size_evidence_url": "", "employee_size_evidence_quote": "",
        "observed_industry": "Industrial Machinery Manufacturing", "observed_subindustry": "",
        "industry_matches": False, "industry_activity_role": "customer_user",
        "industry_evidence_url": ARTICLE, "industry_evidence_quote": "Industrial machinery.",
        "observed_hq_country": "United States", "observed_hq_state": "Illinois",
        "geography_matches": True, "geography_evidence_url": ROOT,
        "geography_evidence_quote": "Headquartered in Illinois, United States.",
        "observed_company_stage": "", "stage_matches": None,
        "stage_evidence_url": "", "stage_evidence_quote": "",
        "attribute_satisfied": False, "required_attribute_evidence_url": url,
        "required_attribute_evidence_quote": "A provider quote absent from the fetched page.",
        "reason": "No verified company attribute.",
    }


def _install(monkeypatch, *, source_fault=None, hydrate=True, independent_failure=None):
    calls = {"provider": 0, "fetch": [], "investigator": 0}
    for key in ("DEEPLINE_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-only")

    async def provider(**kwargs):
        calls["provider"] += 1
        return _verdict(REPAIR if kwargs["telemetry_purpose"].endswith("schema_repair") else ROOT), ""

    async def fetch(_session, url, **_kwargs):
        calls["fetch"].append(url)
        assert url in {ROOT, REPAIR}, "the source allowance must remain bounded"
        return (502 if source_fault == "failed" else 200), url, (
            "" if source_fault == "empty" else "Plant Services publishes industrial news."
        )

    async def investigate(**kwargs):
        calls["investigator"] += 1
        failed_attribute_review = source_fault in {"failed", "empty"}
        assert set(kwargs["targets"]) == {
            "stage", "headcount", "industry",
            *({"required_attribute"} if failed_attribute_review else set()),
        }
        # A validated industry claim can be retained with the disputed facts.
        claim = {
            "target": "industry", "status": "VERIFIED",
            "observed_value": "Industrial Machinery Manufacturing",
            "observed_industry": "Industrial Machinery Manufacturing",
            "observed_subindustry": "", "activity_role": "supplier_operator",
            "evidence_url": ARTICLE, "evidence_quote": "Voortman manufactures industrial machinery.",
        }
        if independent_failure == "investigation":
            kwargs["diagnostic"]["failure_reason"] = "provider_error"
        return {
            "_completed_submit": independent_failure != "investigation",
            "failure_reason": "provider_error" if independent_failure == "investigation" else "",
            "claims": {
                "industry": claim,
                **{target: {"target": target, "status": "UNPROVEN", "observed_value": "",
                            "evidence_url": "", "evidence_quote": ""}
                   for target in ("stage", "headcount")},
                **({"required_attribute": {
                    "target": "required_attribute", "status": "UNPROVEN",
                    "observed_value": "", "evidence_url": "", "evidence_quote": "",
                    "reason": "The exact source still failed during the shared review.",
                }} if failed_attribute_review else {}),
            },
            "usage": {"fetch_outcomes": [{
                "url_sha256": hashlib.sha256(ARTICLE.encode()).hexdigest(),
                "ok": True, "loaded_text_length": 41,
            }, *([{
                "url_sha256": hashlib.sha256(ROOT.encode()).hexdigest(),
                "ok": False, "loaded_text_length": 0,
                "error_class": "http_error" if source_fault == "failed" else "empty_page",
            }] if failed_attribute_review else [])]},
            investigator.PRIVATE_FETCHED_PAGES_KEY: ({ARTICLE: {
                "final_url": ARTICLE, "text": claim["evidence_quote"],
            }} if hydrate else {}),
        }

    original_ground = lead_scorer._ground_required_attribute_evidence

    async def ground(*args, **kwargs):
        result = await original_ground(*args, **kwargs)
        if calls["provider"] == 1 and source_fault in {"missing", "blank", "failure_code", "missing_final_url"}:
            entry = kwargs["source_cache"][ROOT]
            if source_fault == "missing":
                entry.pop("text")
            elif source_fault == "blank":
                entry["text"] = ""
            elif source_fault == "missing_final_url":
                entry.pop("final_url")
            else:
                entry[lead_scorer.VERIFIER_FAILURE_DETAIL_KEY] = "provider_error"
        return result

    async def refresh(value, *_args, invocation_cache, **_kwargs):
        if independent_failure == "profile":
            invocation_cache.update(refresh_outcome="retryable_failure", failure_reason_code="provider_error")
        return value

    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(lead_scorer, "_ground_required_attribute_evidence", ground)
    monkeypatch.setattr(lead_scorer, "_refresh_linkedin_employee_size_observation", refresh)
    return calls


def _run(**kwargs):
    kwargs.setdefault("verified_homepage_identity", company_fit_unavailable(
        "homepage company name unproved", details={
            "verified_homepage_transport_domain": "plantservices.com",
        },
    ))
    return asyncio.run(lead_scorer._llm_reverify_company(
        _company(), _icp(), require_company_fit_dimensions=True,
        evidence_investigator=True, **kwargs,
    ))


def _retryable(result):
    return scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    }, integrity_policy=True)


def test_successful_source_exhaustion_is_unqualified_not_infrastructure_failure(monkeypatch):
    calls = _install(monkeypatch)
    result = _run()
    assert result.decision == "unavailable"
    assert result.details["identity_receipt"]["reason_code"] == "identity_not_proven"
    assert result.details["required_attribute_grounding"]["status"] == "url_limit"
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert "failure_reason_code" not in result.details
    assert not _retryable(result)
    assert calls == {"provider": 2, "fetch": [ROOT], "investigator": 1}


@pytest.mark.parametrize("source_fault", ["failed", "empty", "missing", "blank", "failure_code", "missing_final_url"])
def test_source_failure_or_missing_evidence_is_not_exhausted_success(monkeypatch, source_fault):
    _install(monkeypatch, source_fault=source_fault)
    result = _run()
    assert result.details["required_attribute_grounding"]["status"] == (
        "unproven" if source_fault in {"failed", "empty"} else "url_limit"
    )
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert _retryable(result)


@pytest.mark.parametrize("independent_failure", ["profile", "homepage", "identity_schema", "investigation"])
def test_source_limit_cannot_hide_independent_failure(monkeypatch, independent_failure):
    _install(monkeypatch, independent_failure=independent_failure)
    kwargs = {}
    if independent_failure == "homepage":
        kwargs["verified_homepage_identity"] = company_fit_unavailable(
            "website returned HTTP 502", details={
                "source_fetch_failed": True, "verified_homepage_transport_domain": "plantservices.com",
            },
        )
    if independent_failure == "identity_schema":
        original = lead_scorer._request_company_reverify_json

        async def malformed_identity(**kwargs):
            verdict, error = await original(**kwargs)
            verdict["observed_company_name"] = None
            return verdict, error

        monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", malformed_identity)
    result = _run(**kwargs)
    if independent_failure != "investigation":
        assert result.details["required_attribute_grounding"]["status"] == "url_limit"
    assert _retryable(result)
    assert result.details.get("failure_class") != "insufficient_fit_evidence"


def test_unexhausted_source_cap_fetches_repair_and_preserves_quote_absent_behavior(monkeypatch):
    calls = _install(monkeypatch, hydrate=False)
    result = _run()
    assert calls["fetch"] == [ROOT, REPAIR]
    assert result.details["required_attribute_grounding"]["status"] == "quote_absent"
    assert result.details["failure_class"] == "required_attribute_quote_absent"
    # The existing outer retry/exhaustion behavior for an actual absent quote is unchanged.
    assert _retryable(result)


def test_grounded_positive_verdict_still_matches(monkeypatch):
    _install(monkeypatch)
    company = _company().model_copy(update={"company_name": "Plant Services"})
    verdict = _verdict()
    verdict.update(attribute_satisfied=True,
                   required_attribute_evidence_quote="Plant Services publishes industrial news.")

    async def provider(**_kwargs):
        return copy.deepcopy(verdict), ""

    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, SimpleNamespace(**{**vars(_icp()), "company_stage": ""}),
        require_company_fit_dimensions=False,
    ))
    assert result.decision == "match"
    assert result.details["required_attribute_grounding"]["status"] == "grounded"


def test_different_domain_remains_genuine_mismatch(monkeypatch):
    _install(monkeypatch)
    wrong = _company().model_copy(update={"company_website": "https://voortman.net/"})
    result = lead_scorer._reverify_decision(_verdict(), "", "", company=wrong)
    assert result.decision == "mismatch"
    assert result.details["identity_decision"] == "mismatch"
    assert not _retryable(result)


def test_real_json_parser_failure_stays_retryable(monkeypatch):
    _install(monkeypatch)
    # Restore the real parser, with an invalid native model response and no network.
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", REAL_REQUEST)

    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def json(self): return {"choices": [{"message": {"content": "{"}}]}

    class Session:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        def post(self, *_args, **_kwargs): return Response()

    monkeypatch.setattr(lead_scorer.aiohttp, "ClientSession", Session)
    assert _retryable(_run())


@pytest.mark.parametrize("with_accepted_checkpoint", [False, True])
def test_scoring_completes_with_unavailable_company_and_resumes_checkpoint(
    monkeypatch, with_accepted_checkpoint,
):
    calls = _install(monkeypatch)

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable("homepage company name unproved", details={
            "verified_homepage_transport_domain": "plantservices.com",
        })

    async def no_profile_proof(*_args, **_kwargs):
        return None

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "fetch_structured_linkedin_company_size", no_profile_proof)
    icp = {
        **vars(_icp()), "icp_id": "source-limit-regression", "prompt": "Find machinery companies.",
        "employee_count": ["201-500"], "intent_signals": ["Announced an office opening."],
        "max_companies": 1,
    }
    company = {
        "company_name": "Voortman Steel Machinery", "company_website": ROOT,
        "company_linkedin": "", "industry": "Industrial Machinery Manufacturing",
        "employee_count": "201-500", "company_stage": "Series B",
        "country": "United States", "state": "", "fit_summary": "Industrial machinery supplier.",
        "fit_evidence_urls": [ARTICLE],
        "intent_signals": [{"matched_icp_signal": 0, "description": "Announced an office opening.",
                            "date": "2026-10-02", "url": ARTICLE, "why_now": "A recent expansion.",
                            "snippet": "Voortman announced an office opening."}],
    }
    # This October14 fixture models the frozen policy before industry review.
    # Record the environment binding before score_input writes policy fields,
    # so its process-wide application cannot leak into the next test.
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "")
    models = {k: v for k, v in scoring.DEFAULT_JUDGE_MODELS.items()
              if k != "industry_evidence_review"}
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2", judge_models=models,
    )
    companies = [company]
    prior = []
    if with_accepted_checkpoint:
        from tests.test_arena_score_integrity import _positive_breakdown
        accepted = {**company, "company_name": "Accepted", "company_website": "https://accepted.example/"}
        companies.insert(0, accepted)
        icp["max_companies"] = 2
        prior = [{**_positive_breakdown("Accepted", "accepted.example"),
                  "company_index": 0, "company_qualified": True, "duplicate_company": False,
                  "company_identity_key": "domain:accepted.example",
                  "company_identity_alias_keys": ["domain:accepted.example"]}]
    document = scoring.build_scoring_input(
        scored_run_id="source-limit-execution", icp=icp, companies=companies,
        policy=policy, evaluation_date="2026-10-14",
        **({"prior_terminal_breakdowns": prior} if prior else {}),
    )
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    original = copy.deepcopy(document)
    checkpoints = []
    result = scorer_entrypoint.score_input(document, on_terminal_breakdowns=checkpoints.append)
    assert "failure" not in result
    row = result["breakdowns"][-1]
    if prior:
        assert result["breakdowns"][0] == prior[0]
    assert row["final_score"] == 0.0 and row["company_qualified"] is False
    assert row["verifier_gate_receipts"][0]["decision"] == "unavailable"
    assert row["verifier_gate_receipts"][0]["failure_class"] == "insufficient_fit_evidence"
    assert calls == {"provider": 2, "fetch": [ROOT], "investigator": 1}
    assert checkpoints[-1] == result["breakdowns"]
    assert document == original
    scored = verify.scored_row(
        "source-limit-submission", 0, document["scored_run_id"], icp, companies,
        result["breakdowns"], policy,
    )
    assert scored["cause"] == "accepted"
    assert scored["scored_run_id"] == document["scored_run_id"]
    if not prior:
        assert scored["per_icp_score"] == 0.0

    resumed_input = scoring.build_scoring_input(
        scored_run_id="source-limit-execution", icp=icp, companies=companies,
        policy=policy, evaluation_date="2026-10-14", prior_terminal_breakdowns=checkpoints[-1],
    )
    resumed = scorer_entrypoint.score_input(resumed_input)
    assert resumed == result
    # Durable terminal work performs no new provider work on resume.
    assert calls == {"provider": 2, "fetch": [ROOT], "investigator": 1}
