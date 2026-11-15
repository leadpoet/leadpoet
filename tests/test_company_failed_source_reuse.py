"""Reuse an attempted source read without suppressing bounded evidence recovery.

The Emerson fields are a minimal public-field projection of the saved November
15 model reply. Private requests, benchmark criteria, and ledger data stay out
of this fixture. All transport and model replies below are offline controls.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_verification import _ArenaProviderFetchError
from qualification.scoring.competition import scorer_breakdown_has_retryable_infrastructure_failure
from tests.test_company_evidence_investigator import _company, _complete_verdict, _finding, _icp
from tests.test_company_evidence_refusal_fallback import _install_loop


EMERSON_PDF = (
    "https://ir.emerson.com/sec-filings/all-sec-filings/content/"
    "0000032604-25-000087/0000032604-25-000087.pdf"
)
EMERSON_OBSERVATION = {
    "observed_employee_count": 71000,
    "employee_size_matches": True,
    "employee_size_evidence_url": EMERSON_PDF,
    "employee_size_evidence_quote": (
        "The Company and its subsidiaries had approximately 71,000 employees "
        "at September 30, 2025."
    ),
    "attribute_satisfied": True,
    "required_attribute_evidence_url": EMERSON_PDF,
    "required_attribute_evidence_quote": (
        "At September 30, 2025, the Company had approximately 120 manufacturing "
        "locations worldwide, of which approximately 35 were located in the "
        "United States and 85 were located outside the United States, primarily "
        "in Europe and Asia, and to a lesser extent in Canada and Latin America."
    ),
}


@pytest.mark.parametrize("failure,reason", [
    ("broker", scorer.PROVIDER_ERROR_FAILURE_REASON),
    ("timeout", scorer.PROVIDER_ERROR_FAILURE_REASON),
    ("http", scorer.SOURCE_BLOCKED_FAILURE_REASON),
])
def test_native_attribute_failure_is_an_attempted_headcount_read(monkeypatch, failure, reason):
    calls = []

    async def fetch(_session, url):
        calls.append(url)
        if failure == "broker":
            raise _ArenaProviderFetchError("source unavailable", broker_status=502)
        if failure == "timeout":
            raise TimeoutError("controlled source timeout")
        return 503, url, ""

    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch)

    async def run():
        pages, cache = {}, {}
        verdict, _ = await scorer._ground_required_attribute_evidence(
            EMERSON_OBSERVATION, active_attribute=True, source_cache=pages,
        )
        original_pages = deepcopy(pages)
        for _ in range(2):
            result = await scorer._ground_employee_size_observation(
                verdict, invocation_cache=cache, source_pages=pages,
            )
            assert result["observed_employee_count"] is None
        assert calls == [EMERSON_PDF]
        assert pages == original_pages
        assert cache["refresh_outcome"] == "retryable_failure"
        assert cache[scorer.VERIFIER_FAILURE_DETAIL_KEY] == reason

    asyncio.run(run())


@pytest.mark.parametrize("same_url", [False, True])
def test_failed_page_does_not_block_a_different_or_later_recovered_source(monkeypatch, same_url):
    url = EMERSON_PDF if same_url else "https://ir.emerson.com/other"
    quote = "Founded 202376 employees"
    text = "Founded 2023\n76 employees"
    verdict = {**EMERSON_OBSERVATION, "observed_employee_count": 76,
               "employee_size_evidence_url": url, "employee_size_evidence_quote": quote}
    calls = []

    async def fetch(_session, requested):
        calls.append(requested)
        return 200, requested, text

    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch)
    pages = {EMERSON_PDF: {"status": "source_unavailable", "text": "",
                          scorer.VERIFIER_FAILURE_DETAIL_KEY: "provider_error"}}
    cache = {}
    if same_url:
        first = asyncio.run(scorer._ground_employee_size_observation(
            verdict, invocation_cache=cache, source_pages=pages,
        ))
        assert first["observed_employee_count"] is None
        # Existing independently fetched text can replace the failed record.
        pages[url] = {"status": "fetched", "final_url": url, "text": text}
    result = asyncio.run(scorer._ground_employee_size_observation(
        verdict, invocation_cache=cache, source_pages=pages,
    ))
    assert result["observed_employee_count"] == 76
    assert result["employee_size_evidence_quote"].split() == text.split()
    assert calls == ([] if same_url else [url])


def test_recovered_native_source_still_rejects_approximate_count(monkeypatch):
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Retained source text must not add a fetch.")

    monkeypatch.setattr(scorer, "_fetch_bounded_html", forbidden)
    result = asyncio.run(scorer._ground_employee_size_observation(
        EMERSON_OBSERVATION, invocation_cache={}, source_pages={EMERSON_PDF: {
            "status": "fetched", "final_url": EMERSON_PDF,
            "text": EMERSON_OBSERVATION["employee_size_evidence_quote"],
        }},
    ))
    assert result["observed_employee_count"] is None


def test_schema_repair_keeps_successful_headcount_cache_after_attribute_fetch_failure(monkeypatch):
    headcount_url = "https://acme.example/headcount"
    attribute_url = "https://acme.example/products"
    activity = "Acme supplies SaaS software to customers."
    observation = _complete_verdict(
        observed_company_linkedin="", observed_employee_count=76,
        employee_size_evidence_url=headcount_url,
        employee_size_evidence_quote="Founded 202376 employees",
        attribute_satisfied=True, required_attribute_evidence_url=attribute_url,
        required_attribute_evidence_quote=activity,
    )
    requests, reads, grounding_inputs = [], [], []
    original_ground = scorer._ground_employee_size_observation

    async def provider(**kwargs):
        requests.append(kwargs["telemetry_purpose"])
        if len(requests) == 1:
            return {**observation, "observed_hq_country": "", "observed_hq_state": "",
                    "geography_matches": None, "geography_evidence_quote": ""}, ""
        return {**observation, "required_attribute_evidence_url": headcount_url}, ""

    async def fetch(_session, url):
        reads.append(url)
        if url == headcount_url:
            if reads.count(url) == 1:
                return 200, url, "<div>Founded 2023</div><div>76 employees</div>"
            return 503, url, ""
        assert url == attribute_url
        return 200, url, activity

    async def ground(*args, **kwargs):
        grounding_inputs.append(deepcopy(kwargs))
        return await original_ground(*args, **kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(scorer, "_ground_employee_size_observation", ground)
    result = asyncio.run(scorer._llm_reverify_company(
        _company(linkedin=""),
        _icp(employee_count="51-200", required_attribute="Supplies SaaS software to customers."),
        require_company_fit_dimensions=True,
    ))
    assert requests == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair"]
    assert reads == [attribute_url, headcount_url, headcount_url]
    assert len(grounding_inputs) == 2
    repair = grounding_inputs[1]
    assert repair["source_pages"][headcount_url]["status"] == "source_unavailable"
    assert repair["invocation_cache"]["employee_size_sources"][headcount_url]["text"]
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert result.decision == "unavailable"
    assert result.details["required_attribute_decision"] == "unavailable"
    assert scorer_breakdown_has_retryable_infrastructure_failure(
        {"verifier_gate_receipts": [result.receipt("company_fit")]}, integrity_policy=True,
    )


@pytest.mark.parametrize("control", ["valid", "wrong_domain", "approximate"])
def test_cached_headcount_success_keeps_transport_and_exact_quote_checks(monkeypatch, control):
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Both existing source records must prevent a new read.")

    monkeypatch.setattr(scorer, "_fetch_bounded_html", forbidden)
    observation = {**EMERSON_OBSERVATION, "observed_employee_count": 76,
                   "employee_size_evidence_quote": "Founded 202376 employees"}
    text = "Founded 2023\n76 employees"
    if control == "approximate":
        observation = EMERSON_OBSERVATION
        text = observation["employee_size_evidence_quote"]
    cache = {"employee_size_sources": {EMERSON_PDF: {
        "final_url": "https://other.example/headcount" if control == "wrong_domain" else EMERSON_PDF,
        "text": text,
    }}}
    before = deepcopy(cache)
    result = asyncio.run(scorer._ground_employee_size_observation(
        observation, invocation_cache=cache, source_pages={EMERSON_PDF: {
            "status": "source_unavailable", "text": "", "failure_reason_code": "provider_error",
        }},
    ))
    assert result["observed_employee_count"] == (76 if control == "valid" else None)
    assert cache == before


@pytest.mark.parametrize("recovered", [False, True])
def test_reverify_still_runs_native_investigation_and_exa(monkeypatch, recovered):
    url = "https://acme.example/brochure.pdf"
    activity = "Acme supplies SaaS software to customers."
    headcount = "Acme has 76 employees."
    text = activity + " " + headcount
    observation = _complete_verdict(
        observed_company_linkedin="", observed_employee_count=76,
        employee_size_evidence_url=url,
        employee_size_evidence_quote="Acme has approximately 76 employees.",
        industry_evidence_url=url, industry_evidence_quote=activity,
        attribute_satisfied=True, required_attribute_evidence_url=url,
        required_attribute_evidence_quote=activity,
    )
    raw_reads, investigations = [], []
    real_investigate = investigator.investigate_company_evidence

    async def provider(**_kwargs):
        return deepcopy(observation), ""

    async def grounding(_session, requested):
        raw_reads.append(requested)
        raise _ArenaProviderFetchError("source unavailable", broker_status=502)

    async def investigate(**kwargs):
        assert "headcount" in kwargs["targets"]
        findings = [
            _finding(target, status="VERIFIED" if recovered else "UNPROVEN",
                     observed_value=76 if target == "headcount" else None,
                     evidence_url=url if recovered else "",
                     evidence_quote=(headcount if target == "headcount" else activity) if recovered else "",
                     observed_industry="Software", activity_role="supplier_operator")
            for target in kwargs["targets"]
        ]
        contents = ((200, {"results": [{"url": url, "text": text}]})
                    if recovered else (503, {}))
        _, calls, _ = _install_loop(
            monkeypatch, status=502, body="", contents=contents,
            finding=findings, urls=(url,), company_name="Acme", website="https://acme.example",
        )
        result = await real_investigate(**kwargs)
        investigations.append((result, calls))
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", grounding)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigate)
    retained = {}
    result = asyncio.run(scorer._llm_reverify_company(
        _company(linkedin=""),
        _icp(employee_count="51-200", required_attribute="Supplies SaaS software to customers."),
        require_company_fit_dimensions=True, evidence_investigator=True,
        required_attribute_retry_source_cache=retained,
    ))
    assert raw_reads == [url]
    assert len(investigations) == 1
    investigation, calls = investigations[0]
    # The existing PDF path tries Exa first; failed HTML recovery can spend
    # another existing fetch slot on Exa. This change must not add a retry.
    assert len(calls["contents"]) == (1 if recovered else 2)
    assert investigation["usage"]["fetch_calls"] <= investigator.MAX_FETCH_CALLS
    if recovered:
        assert investigation["claims"]["headcount"]["status"] == "VERIFIED"
        assert result.details["dimension_decisions"]["employee_size"] == "match"
        assert retained[url]["text"] == text
        reread, _ = asyncio.run(scorer._ground_required_attribute_evidence(
            observation, active_attribute=True, source_cache={}, successful_source_sink=retained,
        ))
        assert reread[scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
        assert raw_reads == [url]
    else:
        assert result.decision == "unavailable"
        assert scorer_breakdown_has_retryable_infrastructure_failure(
            {"verifier_gate_receipts": [result.receipt("company_fit")]}, integrity_policy=True,
        )
        assert retained == {}
