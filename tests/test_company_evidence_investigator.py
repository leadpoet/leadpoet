from __future__ import annotations

import asyncio
from datetime import date
import hashlib
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring as arena_scoring
from lab_arena import operations as arena_operations
from qualification.scoring.competition import (
    CompetitionCompanyScorer,
    _normalized_company,
)
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_evidence_investigator import (
    _validated_findings,
)
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
)
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.linkedin_company_size import (
    MALFORMED_RESPONSE_FAILURE_REASON,
    PROVIDER_ERROR_FAILURE_REASON,
    SOURCE_BLOCKED_FAILURE_REASON,
    VERIFIER_FAILURE_REASON_KEY,
)
from qualification.scoring.lead_scorer import (
    _employee_size_sources_conflict,
    _project_investigator_geography,
    _project_investigator_headcount,
    _project_investigator_industry,
    _refresh_linkedin_employee_size_observation,
    _reverify_decision,
    _stage_quote_supports_observation,
    _targeted_company_investigation_dimensions,
)


def _company(
    *,
    name: str = "Acme",
    website: str = "https://acme.example",
    linkedin: str = "https://www.linkedin.com/company/acme",
) -> CompanyOutput:
    return CompanyOutput(
        company_name=name,
        company_website=website,
        company_linkedin=linkedin,
        industry="Software",
        employee_count="11-50",
        country="United States",
        state="California",
        intent_signals=[{
            "description": "raised",
            "source": "news",
            "url": "https://news.example/acme",
            "date": "2026-09-01",
            "snippet": "Acme raised a round.",
        }],
    )


def test_retry_source_scope_key_binds_company_icp_date_and_policy():
    company = _company()
    other_company = _company(name="Other Acme")
    icp = _icp()
    other_icp = icp.model_copy(
        update={"required_attribute": "Raised a Series A."}
    )
    scorer = CompetitionCompanyScorer(
        integrity_policy=True,
        company_quality=True,
        evidence_investigator=True,
        scorer_policy={"scoring_adapter_version": "one"},
    )
    other_policy_scorer = CompetitionCompanyScorer(
        integrity_policy=True,
        company_quality=True,
        evidence_investigator=True,
        scorer_policy={"scoring_adapter_version": "two"},
    )

    with use_evaluation_date("2026-09-24"):
        base = scorer._retry_source_scope_key(company, icp)
        assert scorer._retry_source_scope_key(company, icp) == base
        assert scorer._retry_source_scope_key(other_company, icp) != base
        assert scorer._retry_source_scope_key(company, other_icp) != base
        assert other_policy_scorer._retry_source_scope_key(company, icp) != base
    with use_evaluation_date("2026-09-25"):
        assert scorer._retry_source_scope_key(company, icp) != base


def _icp(**overrides) -> ICPPrompt:
    values = {
        "icp_id": "target",
        "prompt": "target",
        "industry": "Software",
        "sub_industry": "SaaS",
        "employee_count": "11-50",
        "company_stage": "",
        "geography": "United States",
        "product_service": "software",
    }
    values.update(overrides)
    return ICPPrompt(**values)


def _competition_company() -> dict:
    return {
        "company_name": "Acme",
        "company_website": "https://acme.example",
        "company_linkedin": "https://www.linkedin.com/company/acme",
        "industry": "Software",
        "employee_count": "11-50",
        "company_stage": "",
        "country": "United States",
        "state": "California",
        "fit_summary": "Acme supplies SaaS software.",
        "fit_evidence_urls": ["https://acme.example/about"],
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "Acme announced a completed funding event.",
            "date": "2026-09-01",
            "why_now": "The completed event is a current buying signal.",
            "url": "https://news.example/acme",
            "snippet": "Acme announced a completed funding event.",
        }],
    }


def _competition_company_v5() -> dict:
    row = _competition_company()
    row.pop("fit_summary")
    row.pop("fit_evidence_urls")
    row["intent_details"] = (
        "Acme announced a completed funding event that matches the requested signal."
    )
    row["contact"] = None
    row["intent_signals"][0].pop("why_now")
    row["intent_signals"][0].pop("snippet")
    return row


def test_investigator_prompt_preserves_equity_stage_across_later_debt():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())

    assert (
        "A later loan, debt facility, or grant does not by itself supersede "
        "that equity stage."
    ) in prompt
    assert (
        "A later completed priced-equity round, controlling acquisition, or "
        "IPO/listing event involving ownership of the investigated company "
        "can supersede it"
    ) in prompt
    assert (
        "the investigated company buying another business does not make the "
        "buyer Acquired and does not supersede the buyer's funding stage."
    ) in prompt
    assert (
        "a later Series C, controlling acquisition, or IPO can contradict "
        "an earlier Series B"
    ) in prompt
    assert "later debt alone cannot" in prompt
    assert "review every one before preserving the older matching stage" in prompt
    assert "if any disputed source remains" in prompt
    assert "return stage UNPROVEN" in prompt
    assert "validated different completed stage may still be" in prompt


def test_v5_stage_evidence_reaches_only_untrusted_investigator_observations(
    monkeypatch,
):
    public = _competition_company_v5()
    public.update({
        "company_stage": "Series B",
        "company_stage_evidence": [
            {
                "url": "https://acme.example/news/series-b",
                "quote": "Acme completed a Series B financing round.",
            },
            {
                "url": "https://news.example/acme-financing",
                "quote": "The financing supports Acme's expansion.",
            },
        ],
    })
    mapped = _normalized_company(public, integrity_policy=True)
    company = CompanyOutput.model_validate(mapped)
    company = CompanyOutput.model_validate_json(company.model_dump_json())
    captured = {}

    async def no_unfetched_proof(**kwargs):
        captured.update(kwargs)
        return {"claims": {}, "failure_reason": ""}

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", no_unfetched_proof
    )
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=company,
        icp=_icp(company_stage="Series B"),
        verdict=_complete_verdict(
            observed_company_stage="",
            stage_matches=None,
            stage_evidence_url="",
            stage_evidence_quote="",
        ),
        investigation_targets=("stage",),
        icp_attribute="",
        icp_stage="Series B",
        verified_identity={},
        verified_transport_domain="acme.example",
        structured_employee_size_evidence=None,
        structured_public_company_evidence=None,
        employee_size_conflict=False,
        company_quality=True,
    ))

    assert mapped["fit_evidence_urls"] == [
        "https://acme.example/news/series-b",
        "https://news.example/acme-financing",
    ]
    assert [item.model_dump(mode="json") for item in company.company_stage_evidence] == (
        public["company_stage_evidence"]
    )
    assert captured["prior_observations"]["untrusted_company_stage_evidence"] == (
        public["company_stage_evidence"]
    )
    assert captured["prior_observations"]["submitted_source_urls"] == [
        "https://acme.example/news/series-b",
        "https://news.example/acme-financing",
        "https://news.example/acme",
        "https://acme.example/",
    ]
    assert "company_stage_evidence" not in captured["company_locator"]
    assert "Fetch a relevant saved URL before using it" in investigator._SYSTEM_PROMPT


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/stage",
        "https://127.0.0.1/stage",
        "https://stage.internal/evidence",
        "evidence.example/stage",
    ],
)
def test_internal_stage_evidence_requires_absolute_public_url(url):
    mapped = _normalized_company(_competition_company_v5(), integrity_policy=True)
    mapped["company_stage_evidence"] = [{
        "url": url,
        "quote": "Acme completed a Series B financing round.",
    }]

    with pytest.raises(ValueError):
        CompanyOutput.model_validate(mapped)


def test_stage_gap_gets_one_targeted_investigation(monkeypatch):
    initial = _complete_verdict(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://acme.example/about",
        stage_evidence_quote="Acme launched its public product.",
    )
    repaired = _complete_verdict(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://acme.example/about",
        stage_evidence_quote="Acme launched its public product.",
    )
    calls = {"broad": 0, "investigator": 0}

    async def provider(**_kwargs):
        calls["broad"] += 1
        return (initial if calls["broad"] == 1 else repaired), ""

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    async def bounded_investigation(*, targets, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("stage",)
        return {
            "claims": {"stage": _finding("stage")},
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigation
    )
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company().model_copy(update={"company_stage": "Public"}),
            _icp(company_stage="Public"),
            require_company_fit_dimensions=True,
            evidence_investigator=True,
        )
    )

    assert calls == {"broad": 1, "investigator": 1}
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH
    receipt = result.details["investigation_receipt"]
    assert receipt["gate"] == "company_evidence_investigation"
    assert receipt["targets"] == ["stage"]
    assert receipt["prior_decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["prior_dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert receipt["claims"]["stage"] == _finding("stage")


def test_submitted_later_round_hint_reopens_matching_stage(monkeypatch):
    initial = _complete_verdict(
        observed_company_name="Doctronic",
        observed_company_website="https://doctronic.ai",
        observed_company_linkedin="https://www.linkedin.com/company/doctronic-ai",
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url="https://news.example/doctronic-series-a",
        stage_evidence_quote="Doctronic today announced a $20 million Series A round.",
    )
    calls = {"broad": 0, "investigator": 0}

    async def provider(**_kwargs):
        calls["broad"] += 1
        return initial, ""

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    async def bounded_investigation(*, targets, prior_observations, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("stage",)
        assert any(
            "Series-B" in url
            for url in prior_observations["submitted_source_urls"]
        )
        assert prior_observations["stage_dispute_urls"] == [
            next(
                url for url in prior_observations["submitted_source_urls"]
                if "Series-B" in url
            )
        ]
        return {
            "claims": {"stage": _finding(
                "stage",
                status="CONTRADICTED",
                observed_value="Series B",
                evidence_url="https://www.businesswire.com/doctronic-series-b",
                evidence_quote=(
                    "Doctronic today announced a completed $40 million Series B round."
                ),
            )},
            "_validated_stage_finding": _finding(
                "stage",
                status="CONTRADICTED",
                observed_value="Series B",
                evidence_url="https://www.businesswire.com/doctronic-series-b",
                evidence_quote=(
                    "Doctronic today announced a completed $40 million Series B round."
                ),
            ),
            "failure_reason": "",
        }

    base_company = _company(
        name="Doctronic",
        website="https://doctronic.ai",
        linkedin="https://www.linkedin.com/company/doctronic-ai",
    )
    company = CompanyOutput.model_validate({**base_company.model_dump(),
        "company_stage": "Series A",
        "intent_signals": [{
            "description": "Doctronic launched prescription renewal services.",
            "source": "news",
            "url": (
                "https://www.businesswire.com/news/home/20260324/"
                "Doctronic-Raises-$40M-Series-B"
            ),
            "date": "2026-03-24",
            "snippet": "Doctronic launched prescription renewal services.",
        }],
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigation
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        company,
        _icp(company_stage="Series A"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert calls == {"broad": 1, "investigator": 1}
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MISMATCH


def test_failed_later_round_fetch_is_not_resolved_by_unrelated_older_round(
    monkeypatch,
):
    trigger_url = (
        "https://www.businesswire.com/news/home/20260324/"
        "Doctronic-Raises-$40M-Series-B"
    )
    old_url = "https://vc.example/portfolio/doctronic-series-a"
    initial = _complete_verdict(
        observed_company_name="Doctronic",
        observed_company_website="https://doctronic.ai",
        observed_company_linkedin="https://www.linkedin.com/company/doctronic-ai",
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url=old_url,
        stage_evidence_quote=(
            "Doctronic announced a completed $20 million Series A round."
        ),
    )

    async def provider(**_kwargs):
        return initial, ""

    async def keep_employee(candidate, *_args, **_kwargs):
        return candidate

    stage_finding = _finding(
        "stage",
        observed_value="Series A",
        evidence_url=old_url,
        evidence_quote="Doctronic announced a completed $20 million Series A round.",
    )

    async def investigate(**_kwargs):
        return {
            "claims": {"stage": stage_finding},
            "_validated_stage_finding": stage_finding,
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                old_url: {
                    "final_url": old_url,
                    "text": stage_finding["evidence_quote"],
                },
            },
            "failure_reason": "",
        }

    base = _company(
        name="Doctronic",
        website="https://doctronic.ai",
        linkedin="https://www.linkedin.com/company/doctronic-ai",
    )
    company = CompanyOutput.model_validate({
        **base.model_dump(),
        "company_stage": "Series A",
        "intent_signals": [{
            "description": "Doctronic launched prescription renewal services.",
            "source": "news",
            "url": trigger_url,
            "date": "2026-03-24",
            "snippet": "Doctronic launched prescription renewal services.",
        }],
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee,
    )
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        company,
        _icp(company_stage="Series A"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["investigation_receipt"]["claims"]["stage"] == (
        stage_finding
    )


def _mirantis_with_submitted_signal(url: str) -> CompanyOutput:
    base = _company(
        name="Mirantis",
        website="https://www.mirantis.com",
        linkedin="https://www.linkedin.com/company/mirantis",
    )
    return CompanyOutput.model_validate({
        **base.model_dump(),
        "company_stage": "Series B",
        "intent_signals": [{
            "description": "IREN announced an acquisition involving Mirantis.",
            "source": "news",
            "url": url,
            "date": "2026-08-04",
            "snippet": "IREN announced an acquisition involving Mirantis.",
        }],
    })


def _mirantis_series_b_verdict() -> dict:
    return _complete_verdict(
        observed_company_name="Mirantis",
        observed_company_website="https://www.mirantis.com",
        observed_company_linkedin="https://www.linkedin.com/company/mirantis",
        observed_company_stage="Series B",
        stage_matches=True,
        stage_evidence_url=(
            "https://www.mirantis.com/blog/mirantis-raises-100-million-series-b-"
            "challenging-incumbents-pure-play-openstack-leader/"
        ),
        stage_evidence_quote=(
            "Mirantis, the pure-play OpenStack company, today announced "
            "$100 million in Series B funding led by Insight Venture Partners."
        ),
    )


def test_saved_mirantis_acquisition_hint_reopens_full_match(monkeypatch):
    submitted_url = (
        "https://markets.businessinsider.com/news/stocks/"
        "iren-completes-acquisition-of-mirantis-1036405471"
    )
    issuer_url = (
        "https://irisenergy.gcs-web.com/news-releases/news-release-details/"
        "iren-completes-acquisition-mirantis"
    )
    acquired_quote = (
        "IREN Limited (NASDAQ: IREN) (“IREN”) today announced it has completed "
        "the acquisition of Mirantis, Inc. (“Mirantis”), a leading provider "
        "of cloud software and services"
    )
    requests = []
    search_queries = []

    async def provider(**_kwargs):
        return _mirantis_series_b_verdict(), ""

    async def keep_employee(candidate, *_args, **_kwargs):
        return candidate

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            name, arguments = "fetch_page", {"url": issuer_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                status="CONTRADICTED",
                observed_value="Acquired",
                evidence_url=issuer_url,
                evidence_quote=acquired_quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": issuer_url}]}

    async def fake_fetch(_session, url):
        assert url == issuer_url
        return {"ok": True, "url": url, "final_url": url, "text": acquired_quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation", keep_employee
    )
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _mirantis_with_submitted_signal(submitted_url),
        _icp(company_stage="Series B"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MISMATCH
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["industry"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["geography"] == COMPANY_FIT_MATCH
    assert search_queries == [
        "Mirantis mirantis.com latest funding round acquisition IPO"
    ]
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["prior_observations"]["stage_dispute_urls"] == [
        submitted_url
    ]


def test_failed_submitted_acquisition_fetch_cannot_retain_old_stage(monkeypatch):
    submitted_url = (
        "https://markets.businessinsider.com/news/stocks/"
        "iren-completes-acquisition-of-mirantis-1036405471"
    )
    old_url = (
        "https://www.mirantis.com/blog/mirantis-raises-100-million-series-b-"
        "challenging-incumbents-pure-play-openstack-leader/"
    )
    old_quote = (
        "Mirantis, the pure-play OpenStack company, today announced "
        "$100 million in Series B funding led by Insight Venture Partners."
    )
    requests = []

    async def provider(**_kwargs):
        return _mirantis_series_b_verdict(), ""

    async def keep_employee(candidate, *_args, **_kwargs):
        return candidate

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            name, arguments = "fetch_page", {"url": submitted_url}
        elif len(requests) == 2:
            name, arguments = "fetch_page", {"url": old_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                observed_value="Series B",
                evidence_url=old_url,
                evidence_quote=old_quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del query, key
        return {"results": [{"url": submitted_url}]}

    async def fake_fetch(_session, url):
        if url == submitted_url:
            return {"ok": False, "url": url, "text": ""}
        assert url == old_url
        return {"ok": True, "url": url, "final_url": url, "text": old_quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation", keep_employee
    )
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _mirantis_with_submitted_signal(submitted_url),
        _icp(company_stage="Series B"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["industry"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["geography"] == COMPANY_FIT_MATCH


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://news.example/iren-completes-acquisition-of-mirantis-123",
            (("https://news.example/iren-completes-acquisition-of-mirantis-123", "acquired"),),
        ),
        (
            "https://news.example/mirantis-was-acquired-by-iren",
            (("https://news.example/mirantis-was-acquired-by-iren", "acquired"),),
        ),
        ("https://news.example/mirantis-acquires-otherco", ()),
        ("https://news.example/iren-acquires-otherco-with-mirantis", ()),
        ("https://news.example/iren-acquisition-of-mirantis-cloud", ()),
        ("https://news.example/mirantis-completes-acquisition-by-iren", ()),
    ],
)
def test_submitted_acquisition_hint_binds_target_position(url, expected):
    assert lead_scorer._submitted_intent_stage_conflict_hints(
        _mirantis_with_submitted_signal(url), "Series B"
    ) == expected


def test_fetched_planned_acquisition_preserves_validated_series_b(monkeypatch):
    submitted_url = (
        "https://news.example/iren-announces-planned-acquisition-of-mirantis"
    )
    old_url = (
        "https://www.mirantis.com/blog/mirantis-raises-100-million-series-b-"
        "challenging-incumbents-pure-play-openstack-leader/"
    )
    old_quote = (
        "Mirantis, the pure-play OpenStack company, today announced "
        "$100 million in Series B funding led by Insight Venture Partners."
    )
    calls = {"investigator": 0}

    async def provider(**_kwargs):
        return _mirantis_series_b_verdict(), ""

    async def keep_employee(candidate, *_args, **_kwargs):
        return candidate

    async def investigate(**kwargs):
        calls["investigator"] += 1
        assert kwargs["targets"] == ("stage",)
        finding = _finding(
            "stage",
            observed_value="Series B",
            evidence_url=old_url,
            evidence_quote=old_quote,
        )
        return {
            "claims": {"stage": finding},
            "_validated_stage_finding": finding,
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                submitted_url: {
                    "final_url": submitted_url,
                    "text": (
                        "IREN announced a planned acquisition of Mirantis, "
                        "subject to regulatory approval."
                    ),
                },
                old_url: {"final_url": old_url, "text": old_quote},
            },
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation", keep_employee
    )
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _mirantis_with_submitted_signal(submitted_url),
        _icp(company_stage="Series B"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert calls["investigator"] == 1
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH


@pytest.mark.parametrize(
    ("trigger_keys", "fetched_keys", "observed_stage", "expected"),
    [
        (("b",), ("b",), "Series A", True),
        (("b",), (), "Series A", False),
        (("b",), (), "Series B", True),
        (("b",), (), "Public", True),
        (("b",), (), "Acquired", True),
        (("b",), (), "Private Equity", True),
        (("b",), (), "Seed", True),
        (("b", "c"), ("b", "c"), "Series A", True),
        (("b", "c"), ("b",), "Series A", False),
    ],
)
def test_reopened_stage_resolution_requires_every_trigger_fetch(
    trigger_keys, fetched_keys, observed_stage, expected
):
    urls = {
        "b": "https://news.example/doctronic-raises-series-b",
        "c": "https://news.example/doctronic-raises-series-c",
    }
    base = _company(name="Doctronic", website="https://doctronic.ai")
    company = CompanyOutput.model_validate({
        **base.model_dump(),
        "intent_signals": [{
            "description": "Doctronic announced an update.",
            "source": "news",
            "url": urls[key],
            "date": "2026-03-24",
            "snippet": "Doctronic announced an update.",
        } for key in trigger_keys],
    })
    finding = _finding(
        "stage",
        observed_value=observed_stage,
        evidence_url="https://vc.example/doctronic-stage",
        evidence_quote=f"Doctronic completed {observed_stage}.",
    )
    investigation = {
        investigator.PRIVATE_FETCHED_PAGES_KEY: {
            urls[key]: {
                "final_url": urls[key],
                "text": f"Doctronic page for {key}.",
            }
            for key in fetched_keys
        }
    }

    assert lead_scorer._reopened_stage_dispute_resolved(
        company, "Series A", finding, investigation
    ) is expected


def test_reopened_stage_resolution_does_not_alias_encoded_trigger_key():
    trigger_url = "https://news.example/doctronic-raises-$40m-series-b"
    fetched_variant = "https://news.example/doctronic-raises-%2440m-series-b"
    base = _company(name="Doctronic", website="https://doctronic.ai")
    company = CompanyOutput.model_validate({
        **base.model_dump(),
        "intent_signals": [{
            "description": "Doctronic announced an update.",
            "source": "news",
            "url": trigger_url,
            "date": "2026-03-24",
            "snippet": "Doctronic announced an update.",
        }],
    })
    finding = _finding(
        "stage",
        observed_value="Series A",
        evidence_url="https://vc.example/doctronic-series-a",
        evidence_quote="Doctronic completed Series A.",
    )

    assert not lead_scorer._reopened_stage_dispute_resolved(
        company,
        "Series A",
        finding,
        {
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                fetched_variant: {
                    "final_url": fetched_variant,
                    "text": "Doctronic completed Series A.",
                }
            }
        },
    )


def test_reopened_stage_resolution_uses_validated_request_url_key():
    raw_trigger_url = "HTTPS://News.Example:443/doctronic-raises-series-b"
    base = _company(name="Doctronic", website="https://doctronic.ai")
    company = CompanyOutput.model_validate({
        **base.model_dump(),
        "intent_signals": [{
            "description": "Doctronic announced an update.",
            "source": "news",
            "url": raw_trigger_url,
            "date": "2026-03-24",
            "snippet": "Doctronic announced an update.",
        }],
    })
    trigger_url = str(company.intent_signals[0].url)
    assert trigger_url == "https://news.example:443/doctronic-raises-series-b"
    finding = _finding(
        "stage",
        observed_value="Series A",
        evidence_url="https://vc.example/doctronic-series-a",
        evidence_quote="Doctronic completed Series A.",
    )

    assert lead_scorer._reopened_stage_dispute_resolved(
        company,
        "Series A",
        finding,
        {
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                trigger_url: {
                    "final_url": "https://news.example/doctronic-raises-series-b",
                    "text": "Doctronic completed Series A.",
                }
            }
        },
    )


@pytest.mark.parametrize(
    "page_text, observed_stage, expected_decision",
    [
        (
            "Doctronic today announced a $40 million Series B round.",
            "Series B",
            COMPANY_FIT_MISMATCH,
        ),
        ("", "Series B", COMPANY_FIT_UNAVAILABLE),
        (
            "OtherCo today announced a $40 million Series B round.",
            "Series B",
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "Doctronic today announced a $20 million Series A round.",
            "Series A",
            COMPANY_FIT_MATCH,
        ),
    ],
)
def test_doctronic_submitted_later_round_uses_validated_fetch(
    monkeypatch, page_text, observed_stage, expected_decision
):
    source_url = (
        "https://www.businesswire.com/news/home/20260324814372/en/"
        "Doctronic-Raises-%2440M-Series-B-Following-Breakthrough"
    )
    initial = _complete_verdict(
        observed_company_name="Doctronic",
        observed_company_website="https://doctronic.ai",
        observed_company_linkedin="https://www.linkedin.com/company/doctronic-ai",
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url="https://news.example/doctronic-series-a",
        stage_evidence_quote="Doctronic today announced a $20 million Series A round.",
    )
    turns = []

    async def broad_provider(**_kwargs):
        return initial, ""

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        turns.append(payload)
        if len(turns) == 1:
            name, arguments = "fetch_page", {"url": source_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                status="VERIFIED",
                observed_value=observed_stage,
                evidence_url=source_url,
                evidence_quote=(page_text or "Doctronic announced a Series B round."),
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(turns)}",
            "index": 0,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        assert requested_url == source_url
        if not page_text:
            return {"ok": False, "url": requested_url, "text": ""}
        return {"ok": True, "url": requested_url, "text": page_text}

    async def fake_search(_session, query, *, key):
        del query, key
        return {"results": []}

    base_company = _company(
        name="Doctronic",
        website="https://doctronic.ai",
        linkedin="https://www.linkedin.com/company/doctronic-ai",
    )
    company = CompanyOutput.model_validate({**base_company.model_dump(),
        "company_stage": "Series A",
        "intent_signals": [{
            "description": "Doctronic launched prescription renewal services.",
            "source": "news",
            "url": source_url,
            "date": "2026-03-24",
            "snippet": "Doctronic launched prescription renewal services.",
        }],
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad_provider)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        company,
        _icp(company_stage="Series A"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert result.decision == expected_decision
    assert result.details["dimension_decisions"]["stage"] == expected_decision
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["industry"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["geography"] == COMPANY_FIT_MATCH
    if expected_decision == COMPANY_FIT_UNAVAILABLE:
        assert result.details["investigation_receipt"]["claims"]["stage"][
            "status"
        ] == "UNPROVEN"


@pytest.mark.parametrize("signal", [
    ("Doctronic launched the Series B product edition.",
     "https://doctronic.ai/products/series-b"),
    ("OtherCo announced a completed Series B round.",
     "https://news.example/otherco-raises-series-b"),
    (
        "Doctronic announced a partnership with OtherCo, which raised a "
        "$40 million Series B round.",
        "https://news.example/doctronic-otherco-partnership",
    ),
    (
        "Doctronic partners with OtherCo after OtherCo raised Series B.",
        "https://news.example/doctronic-partners-otherco-raises-40m-series-b",
    ),
    ("Doctronic announced a completed Series A round.",
     "https://news.example/doctronic-raises-series-a"),
    ("Doctronic's historical Series B estimate was incorrect.",
     "https://news.example/doctronic-history"),
])
def test_submitted_intent_without_bound_conflicting_round_does_not_reopen(signal):
    description, url = signal
    base_company = _company(name="Doctronic", website="https://doctronic.ai")
    company = CompanyOutput.model_validate(
        {**base_company.model_dump(), "company_stage": "Series A", "intent_signals": [{
            "description": description,
            "source": "news",
            "url": url,
            "date": "2026-03-24",
            "snippet": description,
        }]}
    )
    result = _reverify_decision(
        _complete_verdict(
            observed_company_name="Doctronic",
            observed_company_website="https://doctronic.ai",
            observed_company_stage="Series A",
            stage_matches=True,
            stage_evidence_url="https://news.example/doctronic-series-a",
            stage_evidence_quote="Doctronic today announced a $20 million Series A round.",
        ),
        "",
        "series a",
        icp=_icp(company_stage="Series A"),
        company=company,
        company_quality=True,
    )

    assert _targeted_company_investigation_dimensions(
        result,
        icp_stage="series a",
        employee_size_conflict=False,
        company=company,
    ) == ()


def test_matching_stage_without_later_round_hint_keeps_current_path(monkeypatch):
    initial = _complete_verdict(
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url="https://news.example/acme-series-a",
        stage_evidence_quote="Acme today announced a $20 million Series A round.",
    )

    async def provider(**_kwargs):
        return initial, ""

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    async def must_not_investigate(**_kwargs):
        raise AssertionError("matching stage without a later-round hint was reopened")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", must_not_investigate
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company().model_copy(update={"company_stage": "Series A"}),
        _icp(company_stage="Series A"),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert result.decision == COMPANY_FIT_MATCH
    assert "investigation_receipt" not in result.details


def test_investigator_hydrates_only_matching_failed_attribute_source():
    source_url = (
        "https://www.businesswire.com/news/home/20260915525333/en/"
        "TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding-With-New-"
        "Model-for-Composable-AI"
    )
    source_text = (
        "TypeSafe AI emerged from stealth with $40 million in seed funding. "
        "TypeSafe is building intelligence that developers integrate directly "
        "into software systems."
    )
    cache = {
        source_url: {
            "status": "source_unavailable",
            "final_url": "",
            "text": "",
        },
        "https://example.com/already-fetched": {
            "status": "fetched",
            "final_url": "https://example.com/already-fetched",
            "text": "keep this text",
        },
    }
    lead_scorer._hydrate_required_attribute_source_cache(
        cache,
        {
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                source_url: {"final_url": source_url, "text": source_text},
                "https://different.example/news": {
                    "final_url": "https://different.example/news",
                    "text": "Different source text.",
                },
            }
        },
    )

    assert cache[source_url] == {
        "status": "fetched",
        "final_url": source_url,
        "text": source_text,
        lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True,
    }
    assert cache["https://example.com/already-fetched"]["text"] == (
        "keep this text"
    )
    assert "https://different.example/news" not in cache


@pytest.mark.parametrize(
    "mode",
    [
        "unverified",
        "wrong_role",
        "wrong_company",
        "invalid_url",
        "unfetched",
        "quote_absent",
        "redirected_off_domain",
        "budget_exhausted",
    ],
)
def test_alternate_attribute_recovery_source_stays_fail_closed(mode):
    alternate_url = "https://www.happyrobot.ai/blog/series-c"
    quote = "HappyRobot deploys its AI agents across enterprise operations."
    claim = _finding(
        "industry",
        status="UNPROVEN" if mode == "unverified" else "VERIFIED",
        activity_role="customer_user" if mode == "wrong_role" else "supplier_operator",
        evidence_url=(
            "https://other.example/happyrobot"
            if mode == "wrong_company"
            else "http://www.happyrobot.ai/blog/series-c"
            if mode == "invalid_url"
            else alternate_url
        ),
        evidence_quote=quote,
    )
    fetched_url = claim["evidence_url"]
    investigation = {
        investigator.PRIVATE_FETCHED_PAGES_KEY: {
            fetched_url: {
                "final_url": (
                    "https://other.example/redirect"
                    if mode == "redirected_off_domain"
                    else fetched_url
                ),
                "text": (
                    "HappyRobot announced a financing round."
                    if mode == "quote_absent"
                    else quote
                ),
            }
        }
    }
    if mode == "unfetched":
        investigation[investigator.PRIVATE_FETCHED_PAGES_KEY] = {}
    cache = {
        f"https://blocked{index}.example/news": {
            "status": "source_unavailable",
            "final_url": "",
            "text": "",
        }
        for index in range(
            lead_scorer._MAX_REQUIRED_ATTRIBUTE_SOURCE_URLS
            if mode == "budget_exhausted"
            else 1
        )
    }

    lead_scorer._hydrate_verified_required_attribute_recovery_source(
        cache,
        investigation,
        claim,
        verified_transport_domain="happyrobot.ai",
    )

    assert not any(
        entry.get(lead_scorer._INVESTIGATOR_HYDRATED_SOURCE) is True
        for entry in cache.values()
    )


def test_hydrated_attribute_source_matches_only_exact_safe_final_url():
    requested_url = (
        "https://www.businesswire.com/news/home/20260915525333/en/"
        "TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding"
    )
    final_url = requested_url.replace("$40M", "%2440M")
    source_text = "TypeSafe AI emerged from stealth with $40 million."

    def cache_entry(*, observed_final_url=final_url, trusted=True):
        return {
            "status": "fetched",
            "final_url": observed_final_url,
            "text": source_text,
            **(
                {lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True}
                if trusted
                else {}
            ),
        }

    trusted = cache_entry()
    assert lead_scorer._hydrated_required_attribute_source_for_final_url(
        {requested_url: trusted},
        final_url,
    ) is trusted
    assert lead_scorer._hydrated_required_attribute_source_for_final_url(
        {f"{requested_url}?view=one": cache_entry(
            observed_final_url=f"{final_url}?view=two"
        )},
        f"{final_url}?view=two",
    ) is None
    assert lead_scorer._hydrated_required_attribute_source_for_final_url(
        {requested_url: trusted},
        final_url.replace("/news/", "/news%2F"),
    ) is None
    assert lead_scorer._hydrated_required_attribute_source_for_final_url(
        {requested_url: trusted},
        f"{final_url}%23details",
    ) is None
    other_host = final_url.replace("www.businesswire.com", "news.example.com")
    assert lead_scorer._hydrated_required_attribute_source_for_final_url(
        {requested_url: cache_entry(observed_final_url=other_host)},
        other_host,
    ) is None
    assert lead_scorer._hydrated_required_attribute_source_for_final_url(
        {requested_url: cache_entry(trusted=False)},
        final_url,
    ) is None


@pytest.mark.parametrize("exact_status", ["missing", "source_unavailable"])
def test_grounding_reuses_trusted_server_observed_final_url(
    monkeypatch,
    exact_status,
):
    requested_url = (
        "https://www.businesswire.com/news/home/20260915525333/en/"
        "TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding"
    )
    cited_url = requested_url.replace("$40M", "%2440M")
    quote = (
        "TypeSafe AI, a frontier AI lab, today emerged from stealth with "
        "$40 million in seed funding led by DCVC."
    )
    cache = {
        requested_url: {
            "status": "fetched",
            "final_url": cited_url,
            "text": quote,
            lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True,
        }
    }
    if exact_status == "source_unavailable":
        cache[cited_url] = {
            "status": "source_unavailable",
            "final_url": "",
            "text": "",
        }
    verdict = _complete_verdict(
        attribute_satisfied=True,
        required_attribute_evidence_url=cited_url,
        required_attribute_evidence_quote=quote,
    )
    source_fetch = AsyncMock(side_effect=AssertionError("must not refetch"))
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", source_fetch)

    grounded, repair = asyncio.run(
        lead_scorer._ground_required_attribute_evidence(
            verdict,
            active_attribute=True,
            source_cache=cache,
        )
    )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert grounded["attribute_satisfied"] is True
    assert receipt["status"] == "grounded"
    assert receipt["cache_hit"] is True
    assert receipt["source_url_sha256"] == hashlib.sha256(
        cited_url.encode("utf-8")
    ).hexdigest()
    assert receipt["final_url_sha256"] == hashlib.sha256(
        cited_url.encode("utf-8")
    ).hexdigest()
    assert source_fetch.await_count == 0
    assert set(cache) == (
        {requested_url, cited_url}
        if exact_status == "source_unavailable"
        else {requested_url}
    )
    assert quote not in str(receipt)
    assert repair == {}


def test_grounding_does_not_override_exact_fetched_quote_absent_entry():
    requested_url = "https://www.businesswire.com/news/typesafe-$40m"
    cited_url = requested_url.replace("$40m", "%2440m")
    quote = "TypeSafe AI raised $40 million in seed funding."
    exact_text = "This exact fetched body does not contain the submitted quote."
    cache = {
        requested_url: {
            "status": "fetched",
            "final_url": cited_url,
            "text": quote,
            lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True,
        },
        cited_url: {
            "status": "fetched",
            "final_url": cited_url,
            "text": exact_text,
        },
    }
    verdict = _complete_verdict(
        attribute_satisfied=True,
        required_attribute_evidence_url=cited_url,
        required_attribute_evidence_quote=quote,
    )

    grounded, repair = asyncio.run(
        lead_scorer._ground_required_attribute_evidence(
            verdict,
            active_attribute=True,
            source_cache=cache,
        )
    )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert grounded["attribute_satisfied"] is None
    assert receipt["status"] == "quote_absent"
    assert receipt["cache_hit"] is True
    assert repair == {"url": cited_url, "text": exact_text}


def test_model_cannot_supply_private_fetched_pages():
    finding = _finding("stage")
    injected = {"https://evil.example": {"text": "invented source"}}
    assert _validated_findings(
        {
            "findings": [finding],
            investigator.PRIVATE_FETCHED_PAGES_KEY: injected,
        },
        targets=("stage",),
        fetched_pages={finding["evidence_url"]: finding["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    ) is None


@pytest.mark.parametrize(
    "private_pages",
    [
        None,
        {"https://example.com/news": "model-shaped text"},
        {
            "http://example.com/news": {
                "final_url": "http://example.com/news",
                "text": "unsafe transport",
            }
        },
        {
            "https://[malformed/news": {
                "final_url": "https://[malformed/news",
                "text": "malformed URL text",
            }
        },
        {
            "https://example.com/news": {
                "final_url": "https://example.com/news",
                "text": "",
            }
        },
    ],
)
def test_investigator_does_not_hydrate_malformed_or_unsuccessful_pages(
    private_pages,
):
    source_url = "https://example.com/news"
    original = {
        "status": "source_unavailable",
        "final_url": "",
        "text": "",
    }
    cache = {source_url: dict(original)}
    investigation = (
        {}
        if private_pages is None
        else {investigator.PRIVATE_FETCHED_PAGES_KEY: private_pages}
    )

    lead_scorer._hydrate_required_attribute_source_cache(
        cache,
        investigation,
    )

    assert cache[source_url] == original


def test_attribute_cache_projects_exact_successful_source_for_investigator():
    url = "https://www.grab.com/sg/press/atome/"
    text = (
        "Grab Holdings Limited (NASDAQ: GRAB) (“Grab”) announced an agreement "
        "to acquire a controlling 60% interest in Atome Financial."
    )

    assert lead_scorer._investigator_prefetched_pages_from_attribute_cache(
        {
            url: {
                "status": "fetched",
                "final_url": url,
                "text": text,
            }
        },
        [url],
    ) == {url: {"final_url": url, "text": text}}


def test_grab_prefetch_repairs_only_stage_and_stays_out_of_receipt(monkeypatch):
    url = "https://www.grab.com/sg/press/others/atome-financial/"
    stage_quote = "Grab Holdings Limited (NASDAQ: GRAB) (“Grab”)"
    acquisition_quote = (
        "Grab entered an agreement to acquire a controlling 60% interest in "
        "Atome Financial."
    )
    source_text = f"{stage_quote} {acquisition_quote}"
    company_values = _company(
        name="Grab", website="https://grab.com/", linkedin=""
    ).model_dump(mode="json")
    company_values["intent_signals"] = [{
        "description": "Grab announced the proposed Atome acquisition.",
        "source": "news",
        "url": url,
        "date": "2026-09-15",
        "snippet": acquisition_quote,
    }]
    company = CompanyOutput.model_validate(company_values)
    stage_finding = _finding(
        "stage",
        observed_value="Public",
        evidence_url=url,
        evidence_quote=stage_quote,
    )

    async def bounded_investigation(**kwargs):
        assert kwargs["prefetched_pages"] == {
            url: {"final_url": url, "text": source_text}
        }
        return {
            "claims": {"stage": stage_finding},
            "_validated_stage_finding": stage_finding,
            "failure_reason": "",
            "usage": {
                "reasoning_turns": 1,
                "search_calls": 0,
                "fetch_calls": 0,
            },
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                url: {"final_url": url, "text": source_text}
            },
        }

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigation
    )
    _, result, _, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=company,
            icp=_icp(
                company_stage="Public",
                required_attribute=(
                    "Announced a strategic partnership in the last 12 months."
                ),
            ),
            verdict=_complete_verdict(
                observed_company_name="Grab",
                observed_company_website="https://grab.com/",
                observed_company_linkedin="",
                observed_company_stage="",
                stage_matches=None,
                stage_evidence_url="",
                stage_evidence_quote="",
                attribute_satisfied=False,
                required_attribute_evidence_url=url,
                required_attribute_evidence_quote=acquisition_quote,
            ),
            investigation_targets=("stage",),
            icp_attribute=(
                "Announced a strategic partnership in the last 12 months."
            ),
            icp_stage="public",
            verified_identity={},
            verified_transport_domain="grab.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            required_attribute_source_cache={
                url: {
                    "status": "fetched",
                    "final_url": url,
                    "text": source_text,
                }
            },
        )
    )

    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH
    assert result.details["required_attribute_decision"] == COMPANY_FIT_MISMATCH
    assert result.decision == COMPANY_FIT_MISMATCH
    receipt = result.details["investigation_receipt"]
    assert investigator.PRIVATE_FETCHED_PAGES_KEY not in receipt
    assert source_text not in str(receipt)
    assert source_text not in str(result.details)


@pytest.mark.parametrize(
    ("source_url", "submitted_urls", "entry"),
    [
        (
            "https://example.com/unavailable",
            ["https://example.com/unavailable"],
            {"status": "source_unavailable", "final_url": "", "text": ""},
        ),
        (
            "https://example.com/different",
            ["https://example.com/submitted"],
            {
                "status": "fetched",
                "final_url": "https://example.com/different",
                "text": "Different URL evidence.",
            },
        ),
        (
            "https://example.com/oversized",
            ["https://example.com/oversized"],
            {
                "status": "fetched",
                "final_url": "https://example.com/oversized",
                "text": "x" * (investigator.MAX_PAGE_CHARACTERS + 1),
            },
        ),
        (
            "http://example.com/unsafe",
            ["http://example.com/unsafe"],
            {
                "status": "fetched",
                "final_url": "http://example.com/unsafe",
                "text": "Unsafe transport evidence.",
            },
        ),
        (
            "https://[malformed/source",
            ["https://[malformed/source"],
            {
                "status": "fetched",
                "final_url": "https://[malformed/source",
                "text": "Malformed URL evidence.",
            },
        ),
        (
            "https://example.com/script-only",
            ["https://example.com/script-only"],
            {
                "status": "fetched",
                "final_url": "https://example.com/script-only",
                "text": investigator._plain_text(
                    "<script>Grab Holdings Limited (NASDAQ: GRAB)</script>"
                ),
            },
        ),
    ],
)
def test_attribute_cache_does_not_project_unusable_source(
    source_url, submitted_urls, entry
):
    assert lead_scorer._investigator_prefetched_pages_from_attribute_cache(
        {source_url: entry},
        submitted_urls,
    ) == {}


def test_investigator_fetch_uses_bounded_transport_and_validates_final_url(
    monkeypatch,
):
    source_url = "https://example.com/news"
    bounded_fetch = AsyncMock(return_value=(
        200,
        source_url,
        "<main>Example Company published verified evidence.</main>",
    ))
    monkeypatch.setattr(investigator, "_fetch_bounded_html", bounded_fetch)
    result = asyncio.run(investigator._fetch_page(object(), source_url))
    assert result == {
        "ok": True,
        "url": source_url,
        "final_url": source_url,
        "text": "Example Company published verified evidence.",
    }

    bounded_fetch.return_value = (
        200,
        "http://private.example/news",
        "<main>Unsafe redirect body.</main>",
    )
    rejected = asyncio.run(investigator._fetch_page(object(), source_url))
    assert rejected == {"ok": False, "error": "invalid_url"}

    assert bounded_fetch.await_count == 2


def test_schema_repair_does_not_get_a_second_targeted_investigation(monkeypatch):
    initial = _complete_verdict(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://acme.example/about",
        stage_evidence_quote="Acme launched its public product.",
    )
    repaired = _complete_verdict(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://acme.example/about",
        stage_evidence_quote="Acme remains a public company.",
    )
    calls = {"broad": 0, "investigator": 0}

    async def provider(**_kwargs):
        calls["broad"] += 1
        return (initial if calls["broad"] == 1 else repaired), ""

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    async def bounded_investigation(*, targets, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("stage",)
        return {
            "claims": {
                "stage": _finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value="",
                    evidence_url="",
                    evidence_quote="",
                    reason="Current listing proof was not established.",
                )
            },
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigation
    )
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company().model_copy(update={"company_stage": "Public"}),
            _icp(company_stage="Public"),
            require_company_fit_dimensions=True,
            evidence_investigator=True,
        )
    )

    assert calls == {"broad": 1, "investigator": 1}
    assert result.decision == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize(
    ("repaired_quote", "expected_decision"),
    [
        (
            "TypeSafe AI, a frontier AI lab building machine-native, "
            "composable AI, today emerged from stealth with $40 million in "
            "seed funding led by DCVC. Founded by former OpenAI researcher "
            "and co-inventor of RLHF/ChatGPT, Diogo Almeida, with Erik Gafni "
            "and Sasha Sheng, TypeSafe is building a new class of intelligence "
            "designed to give developers reliable, efficient intelligence "
            "they can integrate directly into software systems.",
            COMPANY_FIT_MATCH,
        ),
        (
            "TypeSafe AI, a frontier AI lab building machine-native, "
            "composable AI, today emerged from stealth with $40 million in "
            "seed funding led by DCVC. ... TypeSafe is building a new class "
            "of intelligence designed to give developers reliable, efficient "
            "intelligence they can integrate directly into software systems.",
            COMPANY_FIT_UNAVAILABLE,
        ),
    ],
)
def test_typesafe_investigator_fetch_repairs_same_attribute_source_only(
    monkeypatch,
    repaired_quote,
    expected_decision,
):
    requested_source_url = (
        "https://www.businesswire.com/news/home/20260915525333/en/"
        "TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding-With-New-"
        "Model-for-Composable-AI"
    )
    cited_source_url = requested_source_url.replace("$40M", "%2440M")
    source_text = (
        "TypeSafe AI, a frontier AI lab building machine-native, composable "
        "AI, today emerged from stealth with $40 million in seed funding led "
        "by DCVC. Founded by former OpenAI researcher and co-inventor of "
        "RLHF/ChatGPT, Diogo Almeida, with Erik Gafni and Sasha Sheng, TypeSafe "
        "is building a new class of intelligence designed to give developers "
        "reliable, efficient intelligence they can integrate directly into "
        "software systems. Jev is currently available in early access for "
        "select developers."
    )
    stage_finding = _finding(
        "stage",
        observed_value="Seed",
        evidence_url=requested_source_url,
        evidence_quote=(
            "TypeSafe AI, a frontier AI lab building machine-native, "
            "composable AI, today emerged from stealth with $40 million in "
            "seed funding led by DCVC."
        ),
    )

    def verdict(attribute_quote, attribute_url):
        return _complete_verdict(
            observed_company_name="TypeSafe AI",
            observed_company_website="https://typesafe.ai",
            observed_company_linkedin=(
                "https://www.linkedin.com/company/typesafe-ai"
            ),
            observed_company_stage="Seed",
            stage_matches=True,
            stage_evidence_url=requested_source_url,
            stage_evidence_quote="TypeSafe launched a public product.",
            attribute_satisfied=True,
            required_attribute_evidence_url=attribute_url,
            required_attribute_evidence_quote=attribute_quote,
        )

    responses = [
        verdict(stage_finding["evidence_quote"], requested_source_url),
        verdict(repaired_quote, cited_source_url),
    ]
    prompts = []

    async def provider(**kwargs):
        prompts.append(kwargs["prompt"])
        return responses.pop(0), ""

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    async def bounded_investigation(**_kwargs):
        return {
            "claims": {"stage": stage_finding},
            "_validated_stage_finding": stage_finding,
            "failure_reason": "",
            "usage": {
                "reasoning_turns": 2,
                "search_calls": 0,
                "fetch_calls": 1,
            },
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                requested_source_url: {
                    "final_url": cited_source_url,
                    "text": source_text,
                }
            },
        }

    source_fetch = AsyncMock(side_effect=lead_scorer.aiohttp.ClientError)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", source_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "investigate_company_evidence",
        bounded_investigation,
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(
            name="TypeSafe AI",
            website="https://typesafe.ai",
            linkedin="https://www.linkedin.com/company/typesafe-ai",
        ).model_copy(update={"company_stage": "Seed"}),
        _icp(
            company_stage="Seed",
            required_attribute=(
                "Sells an AI platform for developer software workflows with "
                "recent external funding evidence."
            ),
        ),
        require_company_fit_dimensions=True,
        evidence_investigator=True,
    ))

    assert result.decision == expected_decision
    assert source_fetch.await_count == 1
    assert len(prompts) == 2
    assert "<untrusted_required_attribute_source>" in prompts[1]
    assert "they can integrate directly into software systems" in prompts[1]
    receipt = result.details["investigation_receipt"]
    assert investigator.PRIVATE_FETCHED_PAGES_KEY not in receipt
    assert source_text not in str(receipt)
    assert source_text not in str(result.details)
    grounding = result.details["required_attribute_grounding"]
    assert grounding["cache_hit"] is True
    assert grounding["status"] == (
        "grounded"
        if expected_decision == COMPANY_FIT_MATCH
        else "quote_absent"
    )


def test_non_fit_reverification_never_starts_targeted_investigation(monkeypatch):
    weak_stage = _complete_verdict(
        observed_company_stage="",
        stage_matches=None,
        stage_evidence_url="",
        stage_evidence_quote="",
    )
    calls = {"broad": 0, "investigator": 0}

    async def provider(**_kwargs):
        calls["broad"] += 1
        return weak_stage, ""

    async def must_not_investigate(**_kwargs):
        calls["investigator"] += 1
        raise AssertionError("non-fit re-verification cannot use the investigator")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", must_not_investigate
    )
    monkeypatch.setattr(
        lead_scorer,
        "_incomplete_company_reverify_dimensions",
        lambda *_args, **_kwargs: ("stage",),
    )
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company().model_copy(update={"company_stage": "Public"}),
            _icp(company_stage="Public"),
            require_company_fit_dimensions=False,
            evidence_investigator=True,
        )
    )

    assert calls == {"broad": 2, "investigator": 0}
    assert result.decision == COMPANY_FIT_UNAVAILABLE


def _complete_verdict(**overrides):
    verdict = {
        "observed_company_name": "Acme",
        "observed_company_website": "https://acme.example",
        "observed_company_linkedin": "https://www.linkedin.com/company/acme",
        "observed_employee_count": "11-50",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://evidence.example/headcount",
        "employee_size_evidence_quote": "Acme has 11-50 employees.",
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://evidence.example/industry",
        "industry_evidence_quote": "Acme supplies SaaS software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": "https://evidence.example/hq",
        "geography_evidence_quote": "Acme is headquartered in California, United States.",
        "reason": "verified",
    }
    verdict.update(overrides)
    return verdict


def _finding(target: str, **overrides):
    finding = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "Public",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": "https://acme.example/investors",
        "evidence_quote": "Acme common stock is listed on NASDAQ under ticker ACME.",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "verified",
    }
    finding.update(overrides)
    return finding


@pytest.mark.parametrize(
    ("investigator_mode", "expected_decision"),
    [
        ("supported", COMPANY_FIT_MATCH),
        ("alternate_supported", COMPANY_FIT_MATCH),
        ("alternate_wrong_company", COMPANY_FIT_UNAVAILABLE),
        ("alternate_unsupported_attribute", COMPANY_FIT_UNAVAILABLE),
        ("alternate_unfetched", COMPANY_FIT_UNAVAILABLE),
        ("alternate_invalid_url", COMPANY_FIT_UNAVAILABLE),
        ("wrong_company", COMPANY_FIT_MISMATCH),
        ("quote_absent", COMPANY_FIT_UNAVAILABLE),
        ("still_blocked", COMPANY_FIT_UNAVAILABLE),
        ("unfetched_positive", COMPANY_FIT_UNAVAILABLE),
    ],
)
def test_happyrobot_blocked_attribute_source_recovers_through_full_fit_path(
    monkeypatch, investigator_mode, expected_decision
):
    source_url = (
        "https://www.businesswire.com/news/home/20260804192350/en/"
        "HappyRobot-Raises-$150-Million-Series-C-to-Build-Enterprise-"
        "Superintelligence"
    )
    attribute_quote = (
        "Its platform enables organizations to build, deploy, and manage AI "
        "agents that automate complex operational workflows across voice, "
        "email, documents, and the web."
    )
    alternate_url = (
        "https://www.happyrobot.ai/blog/"
        "happyrobot-seriesc-fundraising-announcement"
    )
    alternate_quote = (
        "This round values us at $1.2 billion post-money, bringing our total "
        "funding to around $200 million as we enter a new stage of growth, "
        "deploying our AI agents across more of the mission-critical work "
        "enterprises can't afford to get wrong."
    )
    wrong_company_quote = (
        "OtherCo lets customers deploy its own AI agents in production."
    )
    company = CompanyOutput.model_validate({
        **_company(
            name="HappyRobot",
            website="https://www.happyrobot.ai/",
            linkedin="https://www.linkedin.com/company/happyrobot",
        ).model_dump(mode="json"),
        "industry": "Artificial Intelligence",
        "employee_count": "201-500",
        "company_stage": "Series C+",
    })
    icp = _icp(
        industry="Artificial Intelligence",
        sub_industry="Generative AI platforms",
        employee_count="201-500",
        company_stage="Series C+",
        product_service="Commercial AI platform for model-driven workflows.",
        required_attribute=(
            "Builds and sells AI software used to develop, deploy, or operate "
            "machine-learning applications."
        ),
    )

    def verdict(*, attribute=True):
        return _complete_verdict(
            observed_company_name="HappyRobot",
            observed_company_website="https://www.happyrobot.ai/",
            observed_company_linkedin=(
                "https://www.linkedin.com/company/happyrobot"
            ),
            observed_employee_count="201-500",
            employee_size_evidence_url=(
                "https://www.linkedin.com/company/happyrobot"
            ),
            employee_size_evidence_quote="Company size 201-500 employees",
            observed_industry="Artificial Intelligence",
            observed_subindustry="AI agent platform for enterprise operations",
            industry_evidence_url="https://www.happyrobot.ai/home",
            industry_evidence_quote=(
                "HappyRobot helps enterprises put agents to work in complex "
                "environments"
            ),
            observed_company_stage="Series C+",
            stage_matches=True,
            stage_evidence_url=source_url,
            stage_evidence_quote=(
                "HappyRobot announced it has raised $150 million in Series C "
                "funding."
            ),
            attribute_satisfied=attribute,
            required_attribute_evidence_url=source_url,
            required_attribute_evidence_quote=attribute_quote,
        )

    calls = {"provider": 0, "investigator": 0, "direct_fetch": 0}

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match(
            "homepage identity verified",
            details={
                "identity": {
                    "decision": COMPANY_FIT_MATCH,
                    "evidence_source": "company_homepage",
                    "observed_name": "happyrobot",
                    "observed_domain": "happyrobot.ai",
                    "observed_linkedin_slug": "happyrobot",
                },
                "verified_homepage_transport_domain": "happyrobot.ai",
            },
        )

    async def provider(**_kwargs):
        calls["provider"] += 1
        if calls["provider"] == 1 or investigator_mode != "wrong_company":
            candidate = verdict()
            if calls["provider"] == 2 and investigator_mode.startswith(
                "alternate_"
            ):
                if investigator_mode == "alternate_supported":
                    candidate.update(
                        required_attribute_evidence_url=alternate_url,
                        required_attribute_evidence_quote=alternate_quote,
                    )
                else:
                    candidate = verdict(attribute=None)
                    lead_scorer._clear_required_attribute_evidence(candidate)
            return candidate, ""
        return verdict(attribute=False) | {
            "required_attribute_evidence_quote": wrong_company_quote,
        }, ""

    async def direct_fetch(_session, url):
        calls["direct_fetch"] += 1
        assert url == source_url
        return 403, url, ""

    async def investigate(*, targets, prior_observations, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("industry",)
        assert prior_observations["submitted_source_urls"][0] == source_url
        page_text = {
            "supported": (
                "HappyRobot announced its enterprise AI agent platform. "
                + attribute_quote
            ),
            "alternate_supported": (
                "We've Raised $150 Million in Series C Funding | HappyRobot. "
                + alternate_quote
            ),
            "alternate_wrong_company": attribute_quote,
            "alternate_unsupported_attribute": (
                "HappyRobot announced a financing round."
            ),
            "alternate_unfetched": "",
            "alternate_invalid_url": attribute_quote,
            "wrong_company": wrong_company_quote,
            "quote_absent": "HappyRobot announced a financing round.",
            "still_blocked": "",
            "unfetched_positive": "",
        }[investigator_mode]
        claim_url = (
            "https://other.example/happyrobot"
            if investigator_mode == "alternate_wrong_company"
            else "http://www.happyrobot.ai/blog/unsafe"
            if investigator_mode == "alternate_invalid_url"
            else alternate_url
            if investigator_mode.startswith("alternate_")
            else source_url
        )
        result = {
            "claims": {"industry": _finding(
                "industry",
                status=(
                    "UNPROVEN" if investigator_mode == "still_blocked"
                    else "VERIFIED" if (
                        investigator_mode == "unfetched_positive"
                        or investigator_mode.startswith("alternate_")
                    )
                    else "CONTRADICTED"
                ),
                observed_industry=(
                    "" if investigator_mode == "still_blocked"
                    else "Artificial Intelligence" if (
                        investigator_mode == "unfetched_positive"
                        or investigator_mode.startswith("alternate_")
                    )
                    else "Other business"
                ),
                observed_subindustry=(
                    "" if investigator_mode == "still_blocked"
                    else "AI agent platform" if (
                        investigator_mode == "unfetched_positive"
                        or investigator_mode.startswith("alternate_")
                    )
                    else "Customer use"
                ),
                activity_role=(
                    "unresolved" if investigator_mode == "still_blocked"
                    else "supplier_operator" if (
                        investigator_mode == "unfetched_positive"
                        or investigator_mode.startswith("alternate_")
                    )
                    else "customer_user"
                ),
                evidence_url=(
                    "" if investigator_mode == "still_blocked" else claim_url
                ),
                evidence_quote=(
                    attribute_quote
                    if investigator_mode == "unfetched_positive"
                    else "" if investigator_mode == "still_blocked"
                    else page_text
                ),
            )},
            "usage": {"reasoning_turns": 2, "search_calls": 0, "fetch_calls": 1},
            "failure_reason": "",
        }
        if page_text and investigator_mode != "alternate_unfetched":
            result[investigator.PRIVATE_FETCHED_PAGES_KEY] = {
                claim_url: {"final_url": claim_url, "text": page_text}
            }
        return result

    async def keep_employee(candidate, *_args, **_kwargs):
        return candidate

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", direct_fetch)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation", keep_employee
    )

    result = asyncio.run(lead_scorer._verify_company_fit(
        company,
        icp,
        0.0,
        1.0,
        set(),
        require_https_transport=True,
        company_quality=True,
        evidence_investigator=True,
    ))

    assert calls == {"provider": 2, "investigator": 1, "direct_fetch": 1}
    assert result.decision == expected_decision
    assert result.details["company_fit_dimensions"]["industry"] == (
        COMPANY_FIT_MATCH
    )
    assert result.details["required_attribute_decision"] == expected_decision
    attribute_receipt = next(
        receipt for receipt in result.details["supporting_receipts"]
        if receipt["gate"] == "required_attribute_source"
    )
    assert attribute_receipt["status"] == (
        "grounded"
        if expected_decision in {COMPANY_FIT_MATCH, COMPANY_FIT_MISMATCH}
        else (
            "quote_absent"
            if investigator_mode == "quote_absent"
            else "invalid_evidence"
            if investigator_mode.startswith("alternate_")
            else "source_unavailable"
        )
    )
    assert attribute_receipt["cache_hit"] is (
        not investigator_mode.startswith("alternate_")
        or investigator_mode == "alternate_supported"
    )
    assert investigator.PRIVATE_FETCHED_PAGES_KEY not in str(
        result.details["supporting_receipts"]
    )


@pytest.mark.parametrize(
    ("attribute_satisfied", "expected_decision"),
    [(True, COMPANY_FIT_MATCH), (False, COMPANY_FIT_MISMATCH)],
)
def test_grounded_attribute_outcome_skips_investigator_full_path(
    monkeypatch, attribute_satisfied, expected_decision
):
    source_url = "https://acme.example/platform"
    quote = "Acme builds and sells software for production AI workflows."
    calls = {"provider": 0, "fetch": 0, "investigator": 0}

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match(
            "homepage identity verified",
            details={
                "identity": {
                    "decision": COMPANY_FIT_MATCH,
                    "evidence_source": "company_homepage",
                    "observed_name": "acme",
                    "observed_domain": "acme.example",
                    "observed_linkedin_slug": "acme",
                },
                "verified_homepage_transport_domain": "acme.example",
            },
        )

    async def provider(**_kwargs):
        calls["provider"] += 1
        return _complete_verdict(
            attribute_satisfied=attribute_satisfied,
            required_attribute_evidence_url=source_url,
            required_attribute_evidence_quote=quote,
        ), ""

    async def source_fetch(_session, url):
        calls["fetch"] += 1
        assert url == source_url
        return 200, url, quote

    async def must_not_investigate(**_kwargs):
        calls["investigator"] += 1
        raise AssertionError("grounded attribute reopened investigation")

    async def keep_employee(candidate, *_args, **_kwargs):
        return candidate

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", source_fetch)
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", must_not_investigate
    )
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation", keep_employee
    )

    result = asyncio.run(lead_scorer._verify_company_fit(
        _company(),
        _icp(required_attribute="Sells software for production AI workflows."),
        0.0,
        1.0,
        set(),
        require_https_transport=True,
        company_quality=True,
        evidence_investigator=True,
    ))

    assert result.decision == expected_decision
    assert result.details["required_attribute_decision"] == expected_decision
    assert calls == {"provider": 1, "fetch": 1, "investigator": 0}


@pytest.mark.parametrize(
    ("failure_reason", "receipt_status", "expect_investigation"),
    [
        (SOURCE_BLOCKED_FAILURE_REASON, "source_unavailable", True),
        (PROVIDER_ERROR_FAILURE_REASON, "source_unavailable", True),
        (MALFORMED_RESPONSE_FAILURE_REASON, "quote_absent", False),
        ("", "grounded", False),
    ],
)
def test_attribute_recovery_trigger_is_only_blocked_or_provider_unavailable(
    failure_reason, receipt_status, expect_investigation
):
    verdict = _complete_verdict()
    verdict[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING] = {
        "status": receipt_status,
        **(
            {lead_scorer.VERIFIER_FAILURE_DETAIL_KEY: failure_reason}
            if failure_reason
            else {}
        ),
    }
    result = lead_scorer.company_fit_unavailable(
        "unproven dimensions: required_attribute",
        details={
            "identity_decision": COMPANY_FIT_MATCH,
            "required_attribute_decision": COMPANY_FIT_UNAVAILABLE,
            "dimension_decisions": {
                dimension: COMPANY_FIT_MATCH
                for dimension in ("employee_size", "industry", "geography", "stage")
            },
        },
    )

    assert lead_scorer._required_attribute_source_recovery_needed(
        result, verdict, active_attribute=True
    ) is expect_investigation


def test_selector_reopens_only_unsupported_industry_semantics():
    cloudforce = _complete_verdict(
        observed_industry="Cloud consulting",
        observed_subindustry="University AI platform",
        industry_matches=False,
        industry_activity_role="unresolved",
        industry_evidence_url="https://gocloudforce.com/education-platform",
        industry_evidence_quote=(
            "Cloudforce provides its secure AI platform to universities."
        ),
    )
    disputed = _reverify_decision(
        cloudforce,
        "",
        "",
        icp=_icp(industry="Education Technology"),
        company=_company(),
        company_quality=True,
    )
    assert disputed.details["dimension_decisions"]["industry"] == (
        COMPANY_FIT_MISMATCH
    )
    assert _targeted_company_investigation_dimensions(
        disputed,
        icp_stage="",
        employee_size_conflict=False,
    ) == ("industry",)

    proven_customer = dict(
        cloudforce,
        industry_activity_role="customer_user",
        industry_evidence_quote=(
            "Acme uses education software supplied by another company."
        ),
    )
    terminal = _reverify_decision(
        proven_customer,
        "",
        "",
        icp=_icp(industry="Education Technology"),
        company=_company(),
        company_quality=True,
    )
    assert terminal.details["dimension_decisions"]["industry"] == (
        COMPANY_FIT_MISMATCH
    )
    assert _targeted_company_investigation_dimensions(
        terminal,
        icp_stage="",
        employee_size_conflict=False,
    ) == ()


def test_selector_researches_missing_headcount_and_hq_but_not_proven_region_mismatch():
    missing_headcount = _reverify_decision(
        _complete_verdict(
            observed_employee_count=None,
            employee_size_matches=None,
            employee_size_evidence_url="",
            employee_size_evidence_quote="",
        ),
        "",
        "",
        icp=_icp(),
        company=_company(),
        company_quality=True,
    )
    assert _targeted_company_investigation_dimensions(
        missing_headcount,
        icp_stage="",
        employee_size_conflict=False,
    ) == ("headcount",)

    missing_hq = _reverify_decision(
        _complete_verdict(
            observed_hq_country="",
            observed_hq_state="",
            geography_matches=None,
            geography_evidence_url="",
            geography_evidence_quote="",
        ),
        "",
        "",
        icp=_icp(),
        company=_company(),
        company_quality=True,
    )
    assert _targeted_company_investigation_dimensions(
        missing_hq,
        icp_stage="",
        employee_size_conflict=False,
    ) == ("geography",)

    outside_region = _reverify_decision(
        _complete_verdict(),
        "",
        "",
        icp=_icp(geography="United States, South"),
        company=_company(),
        company_quality=True,
    )
    assert outside_region.details["dimension_decisions"]["geography"] == (
        COMPANY_FIT_MISMATCH
    )
    assert _targeted_company_investigation_dimensions(
        outside_region,
        icp_stage="",
        employee_size_conflict=False,
    ) == ()


def test_investigator_activity_and_hq_facts_return_to_deterministic_gates():
    industry_finding = _finding(
        "industry",
        observed_value="Education Technology",
        observed_industry="Education Technology",
        observed_subindustry="University AI platform",
        activity_role="supplier_operator",
        evidence_url="https://acme.example/platform",
        evidence_quote="Acme provides an AI education platform to universities.",
    )
    repaired_industry = _project_investigator_industry(
        _complete_verdict(
            observed_industry="Cloud consulting",
            observed_subindustry="",
            industry_matches=False,
            industry_activity_role="unresolved",
        ),
        industry_finding,
    )
    industry_result = _reverify_decision(
        repaired_industry,
        "",
        "",
        icp=_icp(industry="Education Technology"),
        company=_company(),
        company_quality=True,
    )
    assert industry_result.details["dimension_decisions"]["industry"] == (
        COMPANY_FIT_MATCH
    )

    hq_finding = _finding(
        "geography",
        observed_value="Maryland, United States",
        observed_country="United States",
        observed_state="Maryland",
        evidence_url="https://acme.example/about",
        evidence_quote="Acme is headquartered in Maryland, United States.",
    )
    northeast_company = _company().model_copy(update={"state": "Maryland"})
    repaired_hq = _project_investigator_geography(
        _complete_verdict(
            observed_hq_country="",
            observed_hq_state="",
            geography_matches=None,
            dimension_evidence={
                "geography": {
                    "url": "https://old.example/hq",
                    "quote": "Old headquarters evidence.",
                }
            },
        ),
        hq_finding,
        icp=_icp(geography="United States Northeast"),
        company=northeast_company,
        company_quality=True,
    )
    assert repaired_hq["geography_matches"] is True
    assert repaired_hq["dimension_evidence"]["geography"] == {
        "url": hq_finding["evidence_url"],
        "quote": hq_finding["evidence_quote"],
    }
    assert _reverify_decision(
        repaired_hq,
        "",
        "",
        icp=_icp(geography="United States Northeast"),
        company=northeast_company,
        company_quality=True,
    ).details["dimension_decisions"]["geography"] == COMPANY_FIT_MATCH

    # An independently proved HQ that contradicts the submitted company state
    # remains a deterministic mismatch; the investigator cannot erase it.
    conflicting_submission = _project_investigator_geography(
        _complete_verdict(
            observed_hq_country="",
            observed_hq_state="",
            geography_matches=None,
        ),
        hq_finding,
        icp=_icp(geography="United States Northeast"),
        company=_company(),
        company_quality=True,
    )
    assert conflicting_submission["geography_matches"] is True
    assert _reverify_decision(
        conflicting_submission,
        "",
        "",
        icp=_icp(geography="United States Northeast"),
        company=_company(),
        company_quality=True,
    ).details["dimension_decisions"]["geography"] == COMPANY_FIT_MISMATCH


def test_activity_and_headquarters_findings_require_bound_direct_evidence():
    industry_url = "https://acme.example/platform"
    industry_quote = "Acme provides an AI education platform to universities."
    industry = _finding(
        "industry",
        observed_value="Education Technology",
        observed_industry="Education Technology",
        observed_subindustry="University AI platform",
        activity_role="supplier_operator",
        evidence_url=industry_url,
        evidence_quote=industry_quote,
    )
    accepted = _validated_findings(
        {"findings": [industry]},
        targets=("industry",),
        fetched_pages={industry_url: industry_quote},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor={
            "submitted_domain": "acme.example",
            "observed_domain": "acme.example",
        },
    )
    assert accepted["industry"]["status"] == "VERIFIED"

    customer_claimed_as_supplier = dict(industry, activity_role="customer_user")
    rejected_customer = _validated_findings(
        {"findings": [customer_claimed_as_supplier]},
        targets=("industry",),
        fetched_pages={industry_url: industry_quote},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor={
            "submitted_domain": "acme.example",
            "observed_domain": "acme.example",
        },
    )
    assert rejected_customer["industry"]["status"] == "UNPROVEN"

    hq_url = "https://acme.example/about"
    hq = _finding(
        "geography",
        observed_value="Maryland, United States",
        observed_country="United States",
        observed_state="Maryland",
        evidence_url=hq_url,
        evidence_quote="Acme is headquartered in Maryland, United States.",
    )
    accepted_hq = _validated_findings(
        {"findings": [hq]},
        targets=("geography",),
        fetched_pages={hq_url: hq["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor={
            "submitted_domain": "acme.example",
            "observed_domain": "acme.example",
        },
    )
    assert accepted_hq["geography"]["status"] == "VERIFIED"

    office = dict(
        hq,
        evidence_quote="Acme opened an office in Maryland, United States.",
    )
    rejected_office = _validated_findings(
        {"findings": [office]},
        targets=("geography",),
        fetched_pages={hq_url: office["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor={
            "submitted_domain": "acme.example",
            "observed_domain": "acme.example",
        },
    )
    assert rejected_office["geography"]["status"] == "UNPROVEN"

    assert investigator._quote_supports_headquarters(
        "Acme headquarters: San Francisco, CA 94105.",
        observed_country="United States",
        observed_state="California",
    )
    assert not investigator._quote_supports_headquarters(
        "Acme is headquartered in Portland, or it may relocate.",
        observed_country="United States",
        observed_state="Oregon",
    )
    assert not investigator._quote_supports_headquarters(
        "Acme is headquartered in Georgia.",
        observed_country="United States",
        observed_state="Georgia",
    )
    assert investigator._quote_supports_headquarters(
        "Acme is headquartered in Georgia, United States.",
        observed_country="United States",
        observed_state="Georgia",
    )


def test_activity_and_hq_reject_same_name_wrong_domain_and_accept_bound_company():
    wrong_domain = "https://abec.co.uk/about"
    wrong_industry = _finding(
        "industry",
        observed_value="Manufacturing",
        observed_industry="Manufacturing",
        observed_subindustry="Building controls",
        activity_role="supplier_operator",
        evidence_url=wrong_domain,
        evidence_quote="ABEC manufactures building control hardware.",
    )
    wrong_hq = _finding(
        "geography",
        observed_value="United Kingdom",
        observed_country="United Kingdom",
        observed_state="",
        evidence_url=wrong_domain,
        evidence_quote="ABEC is headquartered in the United Kingdom.",
    )
    abec_anchor = {
        "submitted_domain": "abec.com",
        "observed_domain": "abec.co.uk",
        "verified_domain": "abec.com",
    }
    rejected = _validated_findings(
        {"findings": [wrong_industry, wrong_hq]},
        targets=("industry", "geography"),
        fetched_pages={
            wrong_domain: (
                f"{wrong_industry['evidence_quote']} {wrong_hq['evidence_quote']}"
            )
        },
        first_party_domains={"abec.com", "abec.co.uk"},
        identity_names={"abec"},
        identity_anchor=abec_anchor,
    )
    assert rejected["industry"]["status"] == "UNPROVEN"
    assert rejected["geography"]["status"] == "UNPROVEN"

    cloudforce_url = "https://gocloudforce.com/about"
    cloudforce_industry = _finding(
        "industry",
        observed_value="Education Technology",
        observed_industry="Education Technology",
        observed_subindustry="University AI platform",
        activity_role="supplier_operator",
        evidence_url=cloudforce_url,
        evidence_quote=(
            "Cloudforce provides a secure AI education platform to universities."
        ),
    )
    cloudforce_hq = _finding(
        "geography",
        observed_value="Maryland, United States",
        observed_country="United States",
        observed_state="Maryland",
        evidence_url=cloudforce_url,
        evidence_quote=(
            "Cloudforce is headquartered in Maryland, United States."
        ),
    )
    accepted = _validated_findings(
        {"findings": [cloudforce_industry, cloudforce_hq]},
        targets=("industry", "geography"),
        fetched_pages={
            cloudforce_url: (
                f"{cloudforce_industry['evidence_quote']} "
                f"{cloudforce_hq['evidence_quote']}"
            )
        },
        first_party_domains={"gocloudforce.com"},
        identity_names={"cloudforce"},
        identity_anchor={
            "submitted_domain": "gocloudforce.com",
            "observed_domain": "gocloudforce.com",
        },
    )
    assert accepted["industry"]["status"] == "VERIFIED"
    assert accepted["geography"]["status"] == "VERIFIED"


def test_submitted_country_contradiction_precedes_missing_state_evidence():
    company = _company().model_copy(update={
        "country": "Canada",
        "state": "",
    })
    verdict = _complete_verdict(
        observed_hq_country="United States",
        observed_hq_state="",
        geography_matches=None,
        geography_evidence_quote=(
            "Acme is headquartered in the United States."
        ),
    )
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(geography="United States"),
        company=company,
        company_quality=True,
    )
    assert result.details["dimension_decisions"]["geography"] == (
        COMPANY_FIT_MISMATCH
    )
    assert _targeted_company_investigation_dimensions(
        result,
        icp_stage="",
        employee_size_conflict=False,
    ) == ()


def test_plain_text_removes_nonvisible_blocks_before_clipping():
    visible_quote = (
        "2025 Following our Series B, we launched two new products that are "
        "available today."
    )
    raw = (
        "<html><head>"
        f"<script>{'hidden-script-content ' * 2000}</script>"
        f"<style>{'hidden-style-content ' * 2000}</style>"
        "</head><body>"
        f"<p>{visible_quote}</p>"
        "<a href='https://acme.example/news'>News</a>"
        "</body></html>"
    )

    text = investigator._plain_text(raw)

    assert visible_quote in text
    assert "hidden-script-content" not in text
    assert "hidden-style-content" not in text
    assert "https://acme.example/news" in text
    assert len(text) <= investigator.MAX_PAGE_CHARACTERS


def test_plain_text_extracts_realpage_article_before_navigation_cap():
    exact_quote = (
        "RealPage, Inc. (NASDAQ: RP), a leading global provider of software "
        "and data analytics to the real estate industry, today announced the "
        "completion of its acquisition by Thoma Bravo, a leading private "
        "equity investment firm focused on the software sector, in an all-cash "
        "transaction that valued RealPage at approximately $10.2 billion, "
        "including net debt."
    )
    hidden_quote = (
        "RealPage remains publicly listed on Nasdaq under ticker RP."
    )
    url = (
        "https://www.realpage.com/news/"
        "thoma-bravo-completes-acquisition-of-realpage/"
    )
    raw = f"""<html><head><title>Thoma Bravo: RealPage Acquisition</title>
    <script>{hidden_quote}<a href="https://script-parent.example/">Parent</a></script>
    </head><body>
    <nav>{'Solutions Back products markets platforms ' * 2000}
      <a href="https://nav-parent.example/">Parent company</a>
    </nav>
    <main><article>
      <h1>Thoma Bravo Completes Acquisition of RealPage</h1>
      <p>{exact_quote}</p>
      <p hidden>{hidden_quote}
        <a href="https://hidden-parent.example/">Hidden parent</a>
      </p>
      <a href="https://www.realpage.com/company/">About RealPage</a>
    </article></main>
    <template>{hidden_quote}</template>
    </body></html>"""

    text = investigator._plain_text(raw)

    assert exact_quote in text
    assert "Solutions Back products markets platforms" not in text
    assert hidden_quote not in text
    assert "https://script-parent.example/" not in text
    assert "https://nav-parent.example/" not in text
    assert "https://hidden-parent.example/" not in text
    assert "https://www.realpage.com/company/" in text
    assert len(text) <= investigator.MAX_PAGE_CHARACTERS

    hidden = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Public",
            evidence_url=url,
            evidence_quote=hidden_quote,
        )]},
        targets=("stage",),
        fetched_pages={url: text},
        first_party_domains={"realpage.com"},
        identity_names={"realpageinc"},
    )

    exact = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Private Equity",
            evidence_url=url,
            evidence_quote=exact_quote,
        )]},
        targets=("stage",),
        fetched_pages={url: text},
        first_party_domains={"realpage.com"},
        identity_names={"realpageinc"},
    )

    assert investigator._quote_occurs(exact_quote, text)
    assert not investigator._quote_occurs(hidden_quote, text)
    assert _stage_quote_supports_observation("private equity", exact_quote)
    assert not _stage_quote_supports_observation("public", exact_quote)
    assert exact["stage"]["status"] == "VERIFIED"
    assert hidden["stage"]["status"] == "UNPROVEN"


def test_plain_text_exposes_realpage_link_labels_as_exact_quote_surface(
    monkeypatch,
):
    quote = (
        "RealPage, Inc., a leading provider of AI-enabled software and data "
        "analytics to the real estate industry, today announced it has "
        "completed its acquisition of Cherre, a real estate data intelligence "
        "company trusted by institutional owners, investment managers, and "
        "operators worldwide."
    )
    retained_article = (
        "Realpage Newsroom Combination connects property-level operations with "
        "institutional portfolio intelligence. RICHARDSON, TX and NEW YORK, "
        "NY - [RealPage, Inc](https://www.realpage.com/)., a leading provider "
        "of AI-enabled software and data analytics to the real estate industry, "
        "today announced it has completed its acquisition of "
        "[Cherre](https://cherre.com/), a real estate data intelligence company "
        "trusted by institutional owners, investment managers, and operators "
        "worldwide."
    )
    raw = """<html><body><article>
      <a href="https://www.realpage.com/">RealPage, Inc</a>., a leading provider
      acquired <a href="https://cherre.com/">Cherre</a>.
    </article></body></html>"""
    monkeypatch.setattr(
        investigator,
        "extract_article_body",
        lambda _value: retained_article,
    )

    text = investigator._plain_text(raw)

    assert investigator._quote_occurs(quote, text)
    assert quote in investigator._visible_quote_surface(text)
    assert "https://www.realpage.com/" in text
    assert "https://cherre.com/" in text
    assert not investigator._quote_occurs("https://www.realpage.com/", text)
    assert not investigator._quote_occurs(
        "operators worldwide. https://www.realpage.com/",
        text,
    )
    assert len(text) <= investigator.MAX_PAGE_CHARACTERS


def test_visible_markdown_projection_does_not_join_invalid_link_prose():
    ordinary = "Ordinary nonlink prose remains unchanged."
    unsafe = (
        "Company [did not](javascript:alert(1)) acquire Target. "
        "Company [might not](https://broken.example/path acquire Target."
    )

    assert investigator._visible_markdown_link_label_surface(ordinary) == ordinary
    projected = investigator._visible_markdown_link_label_surface(unsafe)
    assert projected == unsafe
    assert not investigator._quote_occurs("Company acquire Target.", projected)


def test_literal_identity_link_marker_fails_closed_for_later_quote():
    fetched = (
        "Visible evidence before marker. "
        f"{investigator._IDENTITY_LINK_CONTEXT_MARKER} "
        "Claim appearing after literal marker."
    )

    assert investigator._quote_occurs("Visible evidence before marker.", fetched)
    assert not investigator._quote_occurs(
        "Claim appearing after literal marker.",
        fetched,
    )


def test_realpage_semantic_rejection_exposes_earlier_private_equity_proof():
    """The exact retained c137 article order needs 1,000 chars of lookback."""

    url = (
        "https://www.realpage.com/news/"
        "thoma-bravo-completes-acquisition-of-realpage/"
    )
    completed_acquisition_quote = (
        "RealPage, Inc. (NASDAQ: RP), a leading global provider of software "
        "and data analytics to the real estate industry, today announced the "
        "completion of its acquisition by Thoma Bravo, a leading private "
        "equity investment firm focused on the software sector, in an all-cash "
        "transaction that valued RealPage at approximately $10.2 billion, "
        "including net debt."
    )
    delisting_quote = (
        "With the completion of the acquisition, RealPage becomes a "
        "privately-held company, and its common stock ceased trading and will "
        "no longer be listed on the Nasdaq stock exchange."
    )
    retained_main_body = " ".join((
        "RICHARDSON, Texas & San Francisco, CA –",
        completed_acquisition_quote,
        (
            "The acquisition was previously announced on December 21, 2020, "
            "and RealPage’s stockholders voted their shares in favor of the "
            "transaction on March 8, 2021."
        ),
        (
            "Upon the completion of the acquisition, RealPage shareholders "
            "were entitled to receive $88.75 in cash per share."
        ),
        (
            "The price per share represents a 30.8 percent premium to the "
            "company’s closing share price of $67.83 on December 18, 2020, "
            "and a premium of 36.5 percent over RealPage’s 30-day "
            "volume-weighted average share price through December 18, 2020."
        ),
        delisting_quote,
        (
            "The closing of this transaction represents a new chapter in the "
            "RealPage journey."
        ),
    ))
    submitted = _finding(
        "stage",
        observed_value="Private Equity",
        evidence_url=url,
        evidence_quote=delisting_quote,
    )
    finding = _validated_findings(
        {"findings": [submitted]},
        targets=("stage",),
        fetched_pages={url: retained_main_body},
        first_party_domains={"realpage.com"},
        identity_names={"realpageinc"},
        identity_anchor={
            "submitted_domain": "realpage.com",
            "observed_domain": "realpage.com",
        },
    )["stage"]

    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == (
        "source quote did not prove current private-equity ownership"
    )
    context = investigator._source_context_for_quote(
        delisting_quote,
        retained_main_body,
    )
    assert completed_acquisition_quote in context
    assert len(context) <= (
        investigator.REJECTED_QUOTE_CONTEXT_BEFORE_CHARACTERS
        + len(delisting_quote)
        + investigator.REJECTED_QUOTE_CONTEXT_AFTER_CHARACTERS
    )
    assert _stage_quote_supports_observation(
        "private equity",
        completed_acquisition_quote,
    )
    assert not _stage_quote_supports_observation(
        "private equity",
        delisting_quote,
    )


@pytest.mark.parametrize(
    "quote",
    [
        (
            "RealPage expects the completion of its acquisition by Thoma Bravo, "
            "a leading private equity investment firm, next month."
        ),
        (
            "RealPage announced the proposed completion of its acquisition by "
            "Thoma Bravo, a leading private equity investment firm."
        ),
        (
            "RealPage announced the completion of its acquisition by Thoma "
            "Bravo, a leading private equity investment firm, of a minority stake."
        ),
        (
            "ParentCo announced the completion of its acquisition by Thoma Bravo, "
            "a leading private equity investment firm."
        ),
    ],
)
def test_completed_private_equity_acquisition_keeps_negative_controls(quote):
    if quote.startswith("ParentCo"):
        url = "https://parent.example/acquisition"
        finding = _validated_findings(
            {"findings": [_finding(
                "stage",
                observed_value="Private Equity",
                evidence_url=url,
                evidence_quote=quote,
            )]},
            targets=("stage",),
            fetched_pages={url: quote},
            first_party_domains={"realpage.com"},
            identity_names={"realpageinc"},
        )
        assert finding["stage"]["status"] == "UNPROVEN"
    else:
        assert not _stage_quote_supports_observation("private equity", quote)


def test_current_private_equity_control_remains_stage_proof():
    assert _stage_quote_supports_observation(
        "private equity",
        "RealPage is controlled by Thoma Bravo, a leading private equity firm.",
    )


def test_stage_quote_can_use_independently_bound_first_party_domain():
    url = "https://acme.example/about"
    quote = (
        "2025 Following our Series B, we launched two new products that are "
        "available today."
    )
    finding = _finding(
        "stage",
        observed_value="Series B",
        evidence_url=url,
        evidence_quote=quote,
    )
    bound_identity = {
        "submitted_domain": "acme.example",
        "observed_domain": "acme.example",
        "verified_domain": "acme.example",
    }

    accepted = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor=bound_identity,
    )

    assert accepted["stage"]["status"] == "VERIFIED"
    assert accepted["stage"]["evidence_quote"] == quote

    for rejected_url, rejected_domains, rejected_identity in (
        (
            "https://news.example/article",
            {"acme.example"},
            bound_identity,
        ),
        (
            url,
            {"acme.example"},
            {
                "submitted_domain": "acme.example",
                "observed_domain": "other.example",
                "verified_domain": "",
            },
        ),
    ):
        rejected_finding = {
            **finding,
            "evidence_url": rejected_url,
        }
        rejected = _validated_findings(
            {"findings": [rejected_finding]},
            targets=("stage",),
            fetched_pages={rejected_url: quote},
            first_party_domains=rejected_domains,
            identity_names={"acme"},
            identity_anchor=rejected_identity,
        )
        assert rejected["stage"]["status"] == "UNPROVEN"

    headcount_quote = "We employ 42 people across our offices."
    rejected_headcount = _validated_findings(
        {"findings": [_finding(
            "headcount",
            observed_value=42,
            evidence_url=url,
            evidence_quote=headcount_quote,
        )]},
        targets=("headcount",),
        fetched_pages={url: headcount_quote},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor=bound_identity,
    )
    assert rejected_headcount["headcount"]["status"] == "UNPROVEN"


def test_decisive_quote_must_occur_in_fetched_page():
    url = "https://acme.example/investors"
    finding = _finding("stage", evidence_url=url)
    accepted = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: "Acme common stock is listed on NASDAQ under ticker ACME."},
        first_party_domains={"acme.example"},
    )
    assert accepted["stage"]["status"] == "VERIFIED"

    rejected = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: "Acme is a privately held software company."},
        first_party_domains={"acme.example"},
    )
    assert rejected["stage"]["status"] == "UNPROVEN"
    assert rejected["stage"]["evidence_url"] == ""

    wrong_company = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: "Acme common stock is listed on NASDAQ under ticker ACME."},
        first_party_domains={"acme.example"},
        identity_names={"different company"},
    )
    assert wrong_company["stage"]["status"] == "UNPROVEN"

    locator_snippet_only = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert locator_snippet_only["stage"]["status"] == "UNPROVEN"


@pytest.mark.parametrize(
    ("company_name", "url", "page", "first_quote", "corrected_quote", "stage"),
    [
        (
            "Latent",
            "https://www.latenthealth.com/blog/latent-raises-80m",
            "Latent has raised $80M in Series A funding to expand its platform.",
            "The business secured eighty million dollars during its initial institutional financing.",
            "Latent has raised $80M in Series A funding to expand its platform.",
            "Series A",
        ),
        (
            "TypeSafe AI",
            (
                "https://dealroom.co/news/151032-typesafe-exits-stealth-with-"
                "40m-seed-to-build-ai-for-software-not-people/"
            ),
            (
                "TypeSafe AI Dealroom has a profile for this one. Try Dealroom → , "
                "a San Francisco frontier AI lab, has emerged from stealth with "
                "US$25.9M seed round led by DCVC ."
            ),
            (
                "TypeSafe AI... has emerged from stealth with US$25.9M seed round "
                "led by DCVC ."
            ),
            (
                "TypeSafe AI Dealroom has a profile for this one. Try Dealroom → , "
                "a San Francisco frontier AI lab, has emerged from stealth with "
                "US$25.9M seed round led by DCVC ."
            ),
            "Seed",
        ),
        (
            "RealPage",
            "https://www.realpage.com/news/thoma-bravo-completes-acquisition-of-realpage/",
            (
                "RealPage was acquired by Thoma Bravo, a leading private equity "
                "firm focused on software. RealPage became privately held."
            ),
            (
                "RealPage was acquired by Thoma Bravo ... RealPage became "
                "privately held."
            ),
            (
                "RealPage was acquired by Thoma Bravo, a leading private equity "
                "firm focused on software. RealPage became privately held."
            ),
            "Private Equity",
        ),
    ],
)
def test_audited_stage_quotes_get_source_context_then_exact_correction(
    monkeypatch, company_name, url, page, first_quote, corrected_quote, stage
):
    """Controlled audit-shaped excerpts test repair, not a live source verdict."""
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
        elif turn == 2:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    observed_value=stage,
                    evidence_url=url,
                    evidence_quote=first_quote,
                )]
            }
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    observed_value=stage,
                    evidence_url=url,
                    evidence_quote=corrected_quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": page}

    async def fake_search(_session, query, *, key):
        del query, key
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": company_name, "website": f"https://{company_name.casefold().replace(' ', '')}.example"},
        targets=("stage",),
        requested_stage=stage,
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_quote"] == corrected_quote
    correction = json.loads(requests[2]["messages"][-1]["content"])
    rejected = correction["rejected_findings"][0]
    assert rejected["source_url"] == url
    assert company_name in rejected["source_context"]
    assert corrected_quote in rejected["source_context"]
    if "..." in first_quote or "…" in first_quote:
        assert rejected["reason"] == (
            "submitted quote contains a prohibited ellipsis instead of one "
            "continuous fetched-page span"
        )
    assert "already fetched page" in correction["instruction"]
    assert requests[2]["tool_choice"] == "required"


def test_multiverse_spliced_quote_gets_bounded_exact_sentence_context():
    opening = (
        "Multiverse, the upskilling platform for AI and tech adoption, today "
        "announced it has raised $70 million in primary funding to drive growth "
        "across Europe."
    )
    intervening = (
        "The funding was led by Schroders Capital, with participation from "
        "existing investors. The investment will accelerate Multiverse's "
        "expansion across Europe. Multiverse completed the acquisition of "
        "Berlin-based data and AI training company StackFuel in January 2026."
    )
    later = (
        "The $2.1bn valuation, a $400m increase on the last funding round, "
        "reflects a company in its strongest position yet: revenue grew 50% yoy, "
        "and increased at an accelerating rate for the third consecutive year."
    )
    page = " ".join(("LONDON – 15th May, 2026 –", opening, intervening, later))
    spliced_quote = f"{opening} {later}"

    assert len(spliced_quote) == 361
    assert hashlib.sha256(spliced_quote.encode()).hexdigest() == (
        "269b8adf03b73ee776de21b6b7ef54f43bf721e672cf41b7446b08064287acd4"
    )
    assert not investigator._quote_occurs(spliced_quote, page)
    context = investigator._source_context_for_quote(spliced_quote, page)

    assert opening in context
    assert intervening in context
    assert later in context
    longest_exact_sentence = max((opening, later), key=len)
    assert len(context) <= (
        investigator.REJECTED_QUOTE_CONTEXT_BEFORE_CHARACTERS
        + len(longest_exact_sentence)
        + investigator.REJECTED_QUOTE_CONTEXT_AFTER_CHARACTERS
    )


def test_multiverse_amount_only_stage_fails_but_named_series_d_passes():
    url = (
        "https://www.multiverse.io/blog/"
        "multiverse-raises-70-million-europes-ai-adoption-platform"
    )
    amount_only = (
        "Multiverse, the upskilling platform for AI and tech adoption, today "
        "announced it has raised $70 million in primary funding to drive growth "
        "across Europe."
    )
    named_round = (
        "In 2022, we raised our $220m Series D funding - one of the largest "
        "venture rounds in EdTech history."
    )
    identity_anchor = {
        "submitted_domain": "multiverse.io",
        "observed_domain": "multiverse.io",
        "verified_domain": "multiverse.io",
    }

    amount_finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series D",
            evidence_url=url,
            evidence_quote=amount_only,
        )]},
        targets=("stage",),
        fetched_pages={url: amount_only},
        first_party_domains={"multiverse.io"},
        identity_names={"multiverse"},
        identity_anchor=identity_anchor,
    )["stage"]
    assert amount_finding["status"] == "UNPROVEN"
    assert amount_finding["reason"] == (
        "source quote did not name the submitted venture stage"
    )

    named_finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series D",
            evidence_url=url,
            evidence_quote=named_round,
        )]},
        targets=("stage",),
        fetched_pages={url: named_round},
        first_party_domains={"multiverse.io"},
        identity_names={"multiverse"},
        identity_anchor=identity_anchor,
    )["stage"]
    assert named_finding["status"] == "VERIFIED"


def test_common_wealth_amount_quote_uses_only_bounded_series_a_context():
    url = (
        "https://www.newswire.ca/news-releases/common-wealth-raises-12-million-"
        "series-a-to-expand-retirement-security-for-canadians-836585010.html"
    )
    quote = (
        "Common Wealth, Canada's fastest-growing group retirement provider, "
        "today announced $12 million CAD in new equity financing"
    )
    retained_opening = (
        "Common Wealth Raises $12 Million Series A to Expand Retirement "
        "Security for Canadians Accessibility Statement Skip Navigation "
        "News provided by Common Wealth Pension Services Inc. Apr 13, 2026, "
        "07:00 ET Canadian-owned retirement fintech accelerates growth among "
        "small and mid-sized employers. TORONTO, April 13, 2026 /CNW/ - "
    )
    local_page = f"{retained_opening}{quote}. The round includes new investors."

    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series A",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: local_page},
        first_party_domains={"commonwealthretirement.com"},
        identity_names={"commonwealth"},
    )["stage"]

    assert not investigator._quote_names_compatible_venture_stage(
        "series a", quote
    )
    assert "Series A" in investigator._source_context_for_quote(quote, local_page)
    assert finding["status"] == "VERIFIED"


@pytest.mark.parametrize(
    "page",
    [
        (
            "Common Wealth Raises $12 Million Series B. "
            "Common Wealth, Canada's fastest-growing group retirement "
            "provider, today announced $12 million CAD in new equity financing."
        ),
        (
            "Common Wealth completed a Series A. "
            + ("distant filler " * 100)
            + "Common Wealth, Canada's fastest-growing group retirement "
            "provider, today announced $12 million CAD in new equity financing."
        ),
    ],
)
def test_venture_stage_context_rejects_different_or_distant_round(page):
    url = "https://news.example/common-wealth-financing"
    quote = (
        "Common Wealth, Canada's fastest-growing group retirement provider, "
        "today announced $12 million CAD in new equity financing"
    )
    context = investigator._source_context_for_quote(quote, page)
    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series A",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: page},
        first_party_domains={"commonwealthretirement.com"},
        identity_names={"commonwealth"},
    )["stage"]

    assert "series a" not in context.casefold()
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == (
        "source quote did not name the submitted venture stage"
    )


@pytest.mark.parametrize(
    ("normalized_stage", "quote", "expected"),
    [
        ("seed", "TypeSafe emerged with a seed round.", True),
        ("seed", "Acme closed a pre-seed round.", False),
        ("seed", "Acme closed a pre–seed round.", False),
        (
            "seed",
            "After a pre-seed investment, Acme closed its Seed round.",
            True,
        ),
        ("series a", "Acme completed its Series A.", True),
        ("series a", "Acme completed its Series B.", False),
        ("series c+", "Acme completed its Series D.", True),
        ("series c+", "Acme completed its Series B.", False),
    ],
)
def test_venture_stage_token_must_match_normalized_category(
    normalized_stage, quote, expected
):
    assert (
        investigator._quote_names_compatible_venture_stage(
            normalized_stage,
            quote,
        )
        is expected
    )


def _domain_styled_stage_finding(
    quote: str,
    *,
    identity_anchor: dict,
):
    url = "https://investor.example/investing-in-genhealth"
    return _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series A",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"genhealth.ai"},
        identity_names={"genhealthai"},
        identity_anchor=identity_anchor,
    )["stage"]


def _genhealth_identity_anchor(**updates):
    anchor = {
        "submitted_name": "GenHealth.ai",
        "submitted_domain": "genhealth.ai",
        "observed_name": "GenHealth.ai",
        "observed_domain": "genhealth.ai",
        "verified_name": "GenHealth.ai",
        "verified_domain": "genhealth.ai",
    }
    anchor.update(updates)
    return anchor


def test_domain_styled_brand_binds_saved_genhealth_stage_quote():
    finding = _domain_styled_stage_finding(
        "Flare Capital Partners has led the $16.5 million Series A in GenHealth",
        identity_anchor=_genhealth_identity_anchor(),
    )

    assert finding["status"] == "VERIFIED"


@pytest.mark.parametrize(
    "identity_anchor",
    [
        _genhealth_identity_anchor(verified_domain="other.ai"),
        _genhealth_identity_anchor(verified_domain=""),
        _genhealth_identity_anchor(verified_name="Other.ai"),
        _genhealth_identity_anchor(observed_name="Gen Health"),
        {
            "submitted_name": "AI.ai",
            "submitted_domain": "ai.ai",
            "observed_name": "AI.ai",
            "observed_domain": "ai.ai",
            "verified_name": "AI.ai",
            "verified_domain": "ai.ai",
        },
    ],
    ids=("domain_mismatch", "missing_domain", "name_mismatch", "unbound_name", "short_stem"),
)
def test_domain_styled_stage_alias_requires_complete_matching_identity(identity_anchor):
    finding = _domain_styled_stage_finding(
        "Flare Capital Partners has led the $16.5 million Series A in GenHealth",
        identity_anchor=identity_anchor,
    )

    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "source quote did not identify the investigated company"


@pytest.mark.parametrize(
    "quote",
    [
        "Flare led the $16.5 million Series A in OtherHealth.",
        "Flare led the $16.5 million Series A in GenHealthcare.",
        "Flare led the $16.5 million Series A in MyGenHealth.",
    ],
    ids=("wrong_company", "superstring", "prefix"),
)
def test_domain_styled_stage_alias_requires_exact_brand_word(quote):
    finding = _domain_styled_stage_finding(
        quote,
        identity_anchor=_genhealth_identity_anchor(),
    )

    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "source quote did not identify the investigated company"


def test_investigator_stage_keeps_other_recipient_control():
    url = "https://news.example/quantumco-funding"
    quote = "QuantumCo completed a Series D financing round led by Growth Partners."
    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series D",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"multiverse.io"},
        identity_names={"multiverse"},
    )["stage"]

    assert finding["status"] == "UNPROVEN"


@pytest.mark.parametrize(
    ("quote", "identity_names", "expected"),
    [
        (
            "Acme has been acquired by Parent Corp and is now part of its platform.",
            {"acme"},
            "VERIFIED",
        ),
        (
            "Acme announced its acquisition of BetterCloud, a SaaS platform.",
            {"acme"},
            "UNPROVEN",
        ),
        (
            "Acme announced that OtherCo has been acquired by Parent Corp.",
            {"acme"},
            "UNPROVEN",
        ),
        (
            "Acme will be acquired by Parent Corp if regulators approve the deal.",
            {"acme"},
            "UNPROVEN",
        ),
        (
            "Parent Corp acquired a minority interest in Acme.",
            {"acme"},
            "UNPROVEN",
        ),
    ],
)
def test_investigator_acquired_stage_requires_company_as_completed_target(
    quote, identity_names, expected
):
    url = "https://acme.example/acquisition"
    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Acquired",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"acme.example"},
        identity_names=identity_names,
    )["stage"]

    assert finding["status"] == expected
    assert lead_scorer._acquired_stage_quote_supports_names(
        tuple(identity_names), quote
    ) is lead_scorer._acquired_stage_quote_supports_company(
        _company(), "Acme", quote
    )
    if expected == "UNPROVEN":
        assert finding["reason"] == (
            "source quote did not prove the investigated company was the "
            "completed acquisition target"
        )


def test_corestack_buyer_quote_is_rejected_before_submitted_series_b_fetch(
    monkeypatch,
):
    acquisition_url = "https://corestack.io/news/corestack-acquires-bettercloud"
    series_b_url = "https://corestack.io/news/corestack-series-b"
    buyer_quote = (
        "CoreStack, the global authority in cloud governance, today announced "
        "its acquisition of BetterCloud, the leading SaaS management platform."
    )
    series_b_quote = (
        "CoreStack Closes $30 Million Series B Financing Round Led by Avatar "
        "Growth Capital."
    )
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    status="CONTRADICTED",
                    observed_value="Acquired",
                    evidence_url=acquisition_url,
                    evidence_quote=buyer_quote,
                )]
            }
        elif turn == 2:
            name, arguments = "fetch_page", {"url": series_b_url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    observed_value="Series B",
                    evidence_url=series_b_url,
                    evidence_quote=series_b_quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        assert requested_url == series_b_url
        return {
            "ok": True,
            "url": requested_url,
            "final_url": requested_url,
            "text": series_b_quote,
        }

    async def fake_search(_session, query, *, key):
        del query, key
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "CoreStack",
            "website": "https://corestack.io/",
            "linkedin": "https://www.linkedin.com/company/corestack/",
        },
        targets=("stage",),
        requested_stage="Series B",
        prior_observations={
            "submitted_source_urls": [acquisition_url, series_b_url]
        },
        verified_homepage_identity={
            "normalized_name": "CoreStack",
            "registrable_dns_domain": "corestack.io",
            "linkedin_company_slug": "corestack",
        },
        prefetched_pages={
            acquisition_url: {
                "final_url": acquisition_url,
                "text": buyer_quote,
            }
        },
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["observed_value"] == "Series B"
    assert result["claims"]["stage"]["evidence_quote"] == series_b_quote
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 1,
        "fetch_calls": 1,
    }
    rejection = json.loads(requests[1]["messages"][-1]["content"])
    assert rejection["rejected_findings"] == [{
        "target": "stage",
        "reason": (
            "source quote did not prove the investigated company was the "
            "completed acquisition target"
        ),
        "source_url": acquisition_url,
        "source_context": buyer_quote,
    }]


def test_investigator_can_accept_semantic_acculon_retrospective_receipt():
    url = "https://www.acculonenergy.com/resources/facility-opening"
    quote = (
        "The facility’s opening follows a period of rapid growth for Acculon, "
        "including a Series A investment led by Terex Corporation and the "
        "expansion of its Acculon Labs testing division."
    )
    assert not _stage_quote_supports_observation("series a", quote)

    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series A",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"acculonenergy.com"},
        identity_names={"acculonenergy", "acculon"},
    )["stage"]

    assert finding["status"] == "VERIFIED"


def test_submit_tool_requires_one_exact_continuous_quote_span():
    submit = next(
        tool
        for tool in investigator._tools(("stage",))
        if tool["name"] == "submit_findings"
    )
    description = submit["parameters"]["properties"]["findings"]["items"][
        "properties"
    ]["evidence_quote"]["description"]

    assert "continuous substring exactly from fetch_page text" in description
    assert "Never insert ... or …" in description


def test_dbs_same_domain_legal_alias_does_not_bind_parent_public_stage():
    alias_url = "https://www.dbs.com/default.page"
    alias_quote = 'DBS Bank Ltd ("DBS Bank") is headquartered in Singapore.'
    parent_url = "https://links.sgx.com/dbs-group"
    parent_quote = (
        "DBS Group Holdings Ltd shares are listed and traded on the Singapore "
        "Exchange under stock code D05."
    )
    anchor = {
        "submitted_name": "DBS",
        "submitted_domain": "dbs.com",
        "submitted_linkedin_slug": "",
        "observed_name": "DBS Bank",
        "observed_domain": "dbs.com",
        "observed_linkedin_slug": "dbs-bank",
        "verified_name": "DBS",
        "verified_domain": "dbs.com",
        "verified_linkedin_slug": "dbs-bank",
    }
    alias = _finding(
        "rebrand",
        observed_value="DBS",
        evidence_url=alias_url,
        evidence_quote=alias_quote,
        old_name="DBS Bank",
        new_name="DBS",
        old_domain="dbs.com",
        new_domain="dbs.com",
        shared_linkedin_slug="dbs-bank",
    )
    stage = _finding(
        "stage",
        observed_value="Public",
        evidence_url=parent_url,
        evidence_quote=parent_quote,
    )

    result = _validated_findings(
        {"findings": [stage, alias]},
        targets=("stage", "rebrand"),
        fetched_pages={alias_url: alias_quote, parent_url: parent_quote},
        first_party_domains={"dbs.com"},
        identity_names={"dbs", "dbsbank"},
        identity_anchor=anchor,
    )

    assert result["rebrand"]["status"] == "VERIFIED"
    assert result["stage"]["status"] == "UNPROVEN"
    assert result["stage"]["reason"] == (
        "source quote did not identify the investigated company"
    )

    company = _company(
        name="DBS",
        website="https://dbs.com/",
        linkedin="",
    )
    verdict = _complete_verdict(
        observed_company_name="DBS Bank",
        observed_company_website="https://dbs.com",
        observed_company_linkedin="https://www.linkedin.com/company/dbs-bank",
    )
    projected = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(company_stage=""),
        company=company,
        verified_homepage_identity={
            "normalized_name": "dbs",
            "registrable_dns_domain": "dbs.com",
            "linkedin_company_slug": "dbs-bank",
        },
        verified_homepage_transport_domain="dbs.com",
        verified_rebrand_identity=result["rebrand"],
        company_quality=True,
    )
    assert projected.decision == COMPANY_FIT_MATCH
    assert projected.details["identity_receipt"]["reason_code"] == (
        "verified_same_domain_alias"
    )


def test_acculon_submitted_opening_page_is_first_stage_source(monkeypatch):
    opening_url = (
        "https://www.acculonenergy.com/resources/"
        "acculon-energy-announces-opening-of-new-2gwh-battery-manufacturing-"
        "facility-in-mason-ohio"
    )
    company_values = _company(
        name="Acculon Energy",
        website="https://acculonenergy.com/",
        linkedin="",
    ).model_dump(mode="json")
    company_values.update({
        "company_stage": "Series A",
        "intent_signals": [{
            "description": "Acculon Energy opened its Mason battery facility.",
            "source": "news",
            "url": opening_url,
            "date": "2026-04-21",
            "snippet": "The opening followed a Series A investment.",
        }],
    })
    company = CompanyOutput.model_validate(company_values)
    captured = {}

    async def capture_investigation(**kwargs):
        captured.update(kwargs)
        return {"claims": {}, "failure_reason": ""}

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", capture_investigation
    )
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=company,
        icp=_icp(company_stage="Series A"),
        verdict=_complete_verdict(
            observed_company_stage="",
            stage_matches=None,
            stage_evidence_url="",
            stage_evidence_quote="",
        ),
        investigation_targets=("stage",),
        icp_attribute="",
        icp_stage="Series A",
        verified_identity={},
        verified_transport_domain="acculonenergy.com",
        structured_employee_size_evidence=None,
        structured_public_company_evidence=None,
        employee_size_conflict=False,
        company_quality=True,
    ))

    assert captured["prior_observations"]["submitted_source_urls"][0] == opening_url


def test_fiuu_parent_listing_stays_an_explicit_unproven_stage_control():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())
    result = _validated_findings(
        {"findings": [_finding(
            "stage",
            status="UNPROVEN",
            observed_value=None,
            evidence_url="",
            evidence_quote="",
            reason="Razer's listing does not establish that subsidiary Fiuu is public.",
        )]},
        targets=("stage",),
        fetched_pages={},
        first_party_domains={"fiuu.com"},
        identity_names={"fiuu"},
    )

    assert result["stage"]["status"] == "UNPROVEN"
    assert "Never inherit Public or another stage" in prompt


def test_rebel_headquarters_conflict_requires_dates_or_move_evidence():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())

    assert "source dates or explicit move evidence" in prompt
    assert "return UNPROVEN" in prompt
    assert "A factory opening cannot prove an HQ move" in prompt


@pytest.mark.parametrize(
    ("name", "domain", "quote", "country", "state", "expected"),
    [
        (
            "Foundation Alloy", "foundationalloy.com",
            "Foundation Alloy is headquartered in Woburn, Massachusetts, United States.",
            "United States", "Massachusetts", "CONTRADICTED",
        ),
        (
            "Visa", "visa.com",
            "Visa Inc. is headquartered in Foster City, California, United States.",
            "United States", "California", "CONTRADICTED",
        ),
        (
            "JCB", "global.jcb",
            "JCB Co., Ltd. is headquartered in Tokyo, Japan.",
            "Japan", "", "CONTRADICTED",
        ),
        (
            "Nucleus RadioPharma", "nucleusrad.com",
            "Nucleus RadioPharma opened a manufacturing facility in Rochester, Minnesota.",
            "United States", "Minnesota", "UNPROVEN",
        ),
    ],
)
def test_audited_headquarters_controls_require_actual_hq_language(
    name, domain, quote, country, state, expected
):
    url = f"https://{domain}/company"
    key = re.sub(r"[^a-z0-9]+", "", name.casefold())
    finding = _finding(
        "geography",
        status="CONTRADICTED",
        observed_value=country,
        observed_country=country,
        observed_state=state,
        evidence_url=url,
        evidence_quote=quote,
    )
    result = _validated_findings(
        {"findings": [finding]},
        targets=("geography",),
        fetched_pages={url: quote},
        first_party_domains={domain},
        identity_names={key},
        identity_anchor={
            "submitted_domain": domain,
            "observed_domain": domain,
            "submitted_name": name,
            "observed_name": name,
        },
    )

    assert result["geography"]["status"] == expected



@pytest.mark.parametrize("quote,expected", [
    (
        "TPG takes a hands-on approach to private equity investing, leveraging "
        "deep expertise and operational engagement to transform companies "
        "through strategic improvements.",
        "UNPROVEN",
    ),
    ("Acme is majority-owned by a private equity firm.", "VERIFIED"),
    ("Acme received a minority investment from a private equity firm.", "UNPROVEN"),
])
def test_stage_investigator_retains_existing_private_equity_ownership_gate(quote, expected):
    # The first quote is the exact false-positive intermediate finding from
    # live shadow round verifierf2626, ICP 16. Investing is not ownership.
    url = "https://tpg.com/our-approach-to-private-equity-investing"
    proof = _validated_findings(
        {"findings": [_finding(
            "stage", observed_value="Private Equity", evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",), fetched_pages={url: quote},
        first_party_domains={"tpg.com"}, identity_names={"tpg", "acme"},
    )["stage"]
    assert proof["status"] == expected
    prior = {**_complete_verdict(), "observed_company_stage": None,
             "stage_matches": None, "stage_evidence_quote": "", "stage_evidence_url": ""}
    projected = lead_scorer._project_investigator_stage(prior, proof, icp_stage="private equity")
    assert lead_scorer._decision_from_observed_stage(
        projected, "private equity", validated_stage_finding=proof,
    ) == (COMPANY_FIT_MATCH if expected == "VERIFIED" else COMPANY_FIT_UNAVAILABLE)


def test_semantic_stage_proof_is_bound_to_the_validated_investigator_finding():
    url = "https://acme.example/investors"
    quote = (
        "Acme is listed on the main board of the Example Exchange under "
        "the stock code 1828."
    )
    assert not _stage_quote_supports_observation("public", quote)
    proof = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Public",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )["stage"]
    assert proof["status"] == "VERIFIED"

    projected = lead_scorer._project_investigator_stage(
        _complete_verdict(),
        proof,
        icp_stage="public",
    )
    plain = _reverify_decision(
        projected,
        "",
        "public",
        icp=_icp(company_stage="Public"),
        company=_company(),
        company_quality=True,
    )
    assert plain.details["dimension_decisions"]["stage"] == COMPANY_FIT_UNAVAILABLE

    verified = _reverify_decision(
        projected,
        "",
        "public",
        icp=_icp(company_stage="Public"),
        company=_company(),
        validated_stage_finding=proof,
        company_quality=True,
    )
    assert verified.decision == COMPANY_FIT_MATCH

    null_flag = _reverify_decision(
        {**projected, "stage_matches": None},
        "",
        "public",
        icp=_icp(company_stage="Public"),
        company=_company(),
        validated_stage_finding=proof,
        company_quality=True,
    )
    assert null_flag.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )

    contradicted_proof = {**proof, "status": "CONTRADICTED"}
    contradicted = _reverify_decision(
        lead_scorer._project_investigator_stage(
            _complete_verdict(),
            contradicted_proof,
            icp_stage="series b",
        ),
        "",
        "series b",
        icp=_icp(company_stage="Series B"),
        company=_company(),
        validated_stage_finding=contradicted_proof,
        company_quality=True,
    )
    assert contradicted.decision == COMPANY_FIT_MISMATCH


@pytest.mark.parametrize(
    ("observed", "company_name", "quote"),
    [
        (
            "Public",
            "Example Company 2",
            "Example Company 2 is listed on the main board of the Example "
            "Exchange under the stock code 1828.",
        ),
        (
            "Public",
            "Example Company 3",
            "Example Company 3 is listed on the Example Stock Exchange of "
            "Example Jurisdiction under the stock code 1828.",
        ),
    ],
)
def test_investigator_accepts_semantic_stage_wording_outside_fixed_regex(
    observed,
    company_name,
    quote,
):
    normalized = lead_scorer._normalize_company_stage(observed)
    assert not _stage_quote_supports_observation(normalized, quote)
    url = "https://example.test/investors"
    result = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value=observed,
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"example.test"},
        identity_names={"".join(company_name.casefold().split())},
    )
    assert result["stage"]["status"] == "VERIFIED"


@pytest.mark.parametrize(
    "observed",
    ["Series C", "Series D", "Series E", "Series F", "Series G", "Series H"],
)
def test_investigator_canonicalizes_supported_late_series_stages(observed):
    url = "https://dexory.com/news/funding"
    quote = f"Dexory completed a {observed} financing round."
    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value=observed,
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"dexory.com"},
        identity_names={"dexory"},
    )["stage"]

    assert finding["status"] == "VERIFIED"
    matching = lead_scorer._project_investigator_stage(
        _complete_verdict(),
        finding,
        icp_stage="series c+",
    )
    assert matching["observed_company_stage"] == "series c+"
    assert matching["stage_matches"] is True

    earlier_stage = lead_scorer._project_investigator_stage(
        _complete_verdict(),
        finding,
        icp_stage="series b",
    )
    assert earlier_stage["stage_matches"] is False


def test_investigator_still_rejects_nonstage_late_series_wording():
    url = "https://dexory.com/news/funding"
    quote = "Dexory describes itself as a late-stage company."
    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Late stage",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"dexory.com"},
        identity_names={"dexory"},
    )["stage"]

    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "observed company stage was not canonical"


@pytest.mark.parametrize(
    "invalid_receipt",
    [
        None,
        {},
        {"target": "stage", "status": None},
        {"target": "industry", "status": "VERIFIED"},
        _finding("stage", evidence_quote="different quote"),
    ],
)
def test_malformed_stage_receipts_and_verdict_flags_cannot_bypass_regex(
    invalid_receipt,
):
    quote = "Acme common shares are quoted on Nasdaq under the symbol ACME."
    verdict = _complete_verdict(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://acme.example/investors",
        stage_evidence_quote=quote,
        _validated_stage_finding=_finding("stage"),
    )
    result = _reverify_decision(
        verdict,
        "",
        "public",
        icp=_icp(company_stage="Public"),
        company=_company(),
        validated_stage_finding=invalid_receipt,
        company_quality=True,
    )
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_UNAVAILABLE


def test_unknown_and_unproven_investigator_stages_cannot_project():
    url = "https://acme.example/investors"
    quote = "Acme describes itself as a growth-stage company."
    unknown = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="growth stage",
            evidence_url=url,
            evidence_quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )["stage"]
    assert unknown["status"] == "UNPROVEN"

    unchanged = lead_scorer._project_investigator_stage(
        _complete_verdict(observed_company_stage=""),
        {**_finding("stage"), "status": "UNPROVEN"},
        icp_stage="Public",
    )
    assert unchanged["observed_company_stage"] == ""


def test_submit_schema_advertises_only_requested_targets_and_count():
    tools = investigator._tools(("headcount",))
    submit = next(tool for tool in tools if tool["name"] == "submit_findings")
    findings = submit["parameters"]["properties"]["findings"]
    assert findings["minItems"] == 1
    assert findings["maxItems"] == 1
    assert findings["items"]["properties"]["target"]["enum"] == ["headcount"]

    extra_target = {
        "findings": [
            _finding(
                "headcount",
                status="UNPROVEN",
                observed_value=None,
                evidence_url="",
                evidence_quote="",
            ),
            _finding(
                "stage",
                status="UNPROVEN",
                observed_value=None,
                evidence_url="",
                evidence_quote="",
            ),
        ]
    }
    assert _validated_findings(
        extra_target,
        targets=("headcount",),
        fetched_pages={},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    ) is None

    two_target_tools = investigator._tools(("rebrand", "stage"))
    two_target_submit = next(
        tool for tool in two_target_tools if tool["name"] == "submit_findings"
    )
    two_target_findings = two_target_submit["parameters"]["properties"]["findings"]
    assert two_target_findings["minItems"] == 2
    assert two_target_findings["maxItems"] == 2
    assert two_target_findings["items"]["properties"]["target"]["enum"] == [
        "rebrand",
        "stage",
    ]


def test_rebrand_needs_first_party_explicit_old_and_new_name_continuity():
    url = "https://help.wayground.com/rebrand"
    explicit = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url=url,
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    accepted = _validated_findings(
        {"findings": [explicit]},
        targets=("rebrand",),
        fetched_pages={
            url: (
                "Quizizz is now Wayground following our rebrand. "
                "The domain changed from quizizz.com to wayground.com."
            )
        },
        first_party_domains={"quizizz.com", "wayground.com"},
    )
    assert accepted["rebrand"]["status"] == "VERIFIED"

    redirect_only = dict(explicit)
    redirect_only["evidence_quote"] = "Visit our new website at Wayground."
    rejected = _validated_findings(
        {"findings": [redirect_only]},
        targets=("rebrand",),
        fetched_pages={url: "Visit our new website at Wayground."},
        first_party_domains={"quizizz.com", "wayground.com"},
    )
    assert rejected["rebrand"]["status"] == "UNPROVEN"

    prospective = dict(explicit)
    prospective["evidence_quote"] = "Quizizz plans to rebrand as Wayground."
    prospective_result = _validated_findings(
        {"findings": [prospective]},
        targets=("rebrand",),
        fetched_pages={
            url: (
                "Quizizz plans to rebrand as Wayground. "
                "quizizz.com wayground.com"
            )
        },
        first_party_domains={"quizizz.com", "wayground.com"},
    )
    assert prospective_result["rebrand"]["status"] == "UNPROVEN"


def test_verified_rebrand_binds_completed_stage_under_old_name_only():
    rebrand_url = "https://wayground.com/home/from-quizizz-to-wayground"
    stage_url = "https://news.example/quizizz-series-b"
    rebrand = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url=rebrand_url,
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    stage = _finding(
        "stage",
        observed_value="Series B",
        evidence_url=stage_url,
        evidence_quote="Quizizz completed its Series B funding round in 2022.",
    )
    fetched_pages = {
        rebrand_url: (
            "Quizizz is now Wayground following our rebrand. "
            "The domain changed from quizizz.com to wayground.com."
        ),
        stage_url: stage["evidence_quote"],
    }

    accepted = _validated_findings(
        # Stage is intentionally first to prove validation is order-independent.
        {"findings": [stage, rebrand]},
        targets=("stage", "rebrand"),
        fetched_pages=fetched_pages,
        first_party_domains={"quizizz.com", "wayground.com"},
        identity_names={"waygroundformerlyquizizz"},
    )
    assert accepted["rebrand"]["status"] == "VERIFIED"
    assert accepted["stage"]["status"] == "VERIFIED"

    wrong_entity_stage = dict(
        stage,
        evidence_quote="Otherco completed its Series B funding round in 2022.",
    )
    rejected = _validated_findings(
        {"findings": [wrong_entity_stage, rebrand]},
        targets=("stage", "rebrand"),
        fetched_pages={
            **fetched_pages,
            stage_url: wrong_entity_stage["evidence_quote"],
        },
        first_party_domains={"quizizz.com", "wayground.com"},
        identity_names={"waygroundformerlyquizizz"},
    )
    assert rejected["rebrand"]["status"] == "VERIFIED"
    assert rejected["stage"]["status"] == "UNPROVEN"

    prospective_rebrand = dict(
        rebrand,
        evidence_quote="Quizizz plans to rebrand as Wayground.",
    )
    unproven_alias = _validated_findings(
        {"findings": [stage, prospective_rebrand]},
        targets=("stage", "rebrand"),
        fetched_pages={
            **fetched_pages,
            rebrand_url: (
                "Quizizz plans to rebrand as Wayground. "
                "quizizz.com wayground.com"
            ),
        },
        first_party_domains={"quizizz.com", "wayground.com"},
        identity_names={"waygroundformerlyquizizz"},
    )
    assert unproven_alias["rebrand"]["status"] == "UNPROVEN"
    assert unproven_alias["stage"]["status"] == "UNPROVEN"


def test_verified_rebrand_is_a_separate_identity_proof_not_a_domain_rewrite():
    company = _company(
        name="Wayground formerly Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/quizizz",
    )
    assert _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        company_quality=True,
    ).decision == COMPANY_FIT_MISMATCH

    proof = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url="https://help.wayground.com/rebrand",
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    )
    assert result.decision == COMPANY_FIT_MATCH
    receipt = result.details["identity_receipt"]
    assert receipt["submitted_domain"] == "quizizz.com"
    assert receipt["observed_domain"] == "wayground.com"
    assert receipt["reason_code"] == "verified_rebrand_continuity"


def test_same_domain_alias_flows_from_selector_through_evidence_to_identity():
    company = _company(name="Acme", website="https://acme.example")
    verdict = _complete_verdict(observed_company_name="Northstar Systems LLC")
    initial = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        company_quality=True,
    )

    assert initial.decision == COMPANY_FIT_UNAVAILABLE
    assert _targeted_company_investigation_dimensions(
        initial,
        icp_stage="",
        employee_size_conflict=False,
    ) == ("rebrand",)

    url = "https://acme.example/legal-name"
    quote = "Northstar Systems LLC trades as Acme."
    anchor = {
        "submitted_name": company.company_name,
        "submitted_domain": "acme.example",
        "submitted_linkedin_slug": "acme",
        "observed_name": verdict["observed_company_name"],
        "observed_domain": "acme.example",
        "observed_linkedin_slug": "acme",
    }
    proof = _validated_findings(
        {"findings": [_finding(
            "rebrand",
            observed_value="Acme",
            evidence_url=url,
            evidence_quote=quote,
            old_name="Northstar Systems LLC",
            new_name="Acme",
            old_domain="acme.example",
            new_domain="acme.example",
            shared_linkedin_slug="acme",
        )]},
        targets=("rebrand",),
        fetched_pages={url: quote},
        first_party_domains={"acme.example"},
        identity_anchor=anchor,
    )["rebrand"]

    assert proof["status"] == "VERIFIED"
    projected = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    )
    assert projected.decision == COMPANY_FIT_MATCH
    assert projected.details["identity_receipt"]["reason_code"] == (
        "verified_same_domain_alias"
    )


@pytest.mark.parametrize(
    ("evidence_url", "fetched_text", "quote"),
    [
        (
            "https://acme.example/legal-name",
            "Acme and Northstar Systems LLC use the same website.",
            "Acme and Northstar Systems LLC use the same website.",
        ),
        (
            "https://acme.example/legal-name",
            'Northstar Systems LLC ("Northstar") uses the Acme website.',
            'Northstar Systems LLC ("Northstar") uses the Acme website.',
        ),
        (
            "https://acme.example/legal-name",
            "Acme is the parent of Northstar Systems LLC, which operates as a subsidiary.",
            "Acme is the parent of Northstar Systems LLC, which operates as a subsidiary.",
        ),
        (
            "https://acme.example/legal-name",
            "This fetched page has different text.",
            "Northstar Systems LLC trades as Acme.",
        ),
        (
            "https://unrelated.example/legal-name",
            "Northstar Systems LLC trades as Acme.",
            "Northstar Systems LLC trades as Acme.",
        ),
    ],
)
def test_same_domain_alias_rejects_weak_or_unfetched_first_party_evidence(
    evidence_url, fetched_text, quote
):
    anchor = {
        "submitted_name": "Acme",
        "submitted_domain": "acme.example",
        "submitted_linkedin_slug": "acme",
        "observed_name": "Northstar Systems LLC",
        "observed_domain": "acme.example",
        "observed_linkedin_slug": "acme",
    }
    finding = _finding(
        "rebrand",
        evidence_url=evidence_url,
        evidence_quote=quote,
        old_name="Northstar Systems LLC",
        new_name="Acme",
        old_domain="acme.example",
        new_domain="acme.example",
        shared_linkedin_slug="acme",
    )
    result = _validated_findings(
        {"findings": [finding]},
        targets=("rebrand",),
        fetched_pages={evidence_url: fetched_text},
        first_party_domains={"acme.example"},
        identity_anchor=anchor,
    )

    assert result["rebrand"]["status"] == "UNPROVEN"


def test_same_domain_alias_requires_matching_linkedin_slug_and_unproven_is_retryable():
    company = _company(name="Acme", website="https://acme.example")
    conflicting = _complete_verdict(
        observed_company_name="Northstar Systems LLC",
        observed_company_linkedin="https://www.linkedin.com/company/other",
    )
    conflict_result = _reverify_decision(
        conflicting,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=_finding(
            "rebrand",
            old_name="Northstar Systems LLC",
            new_name="Acme",
            old_domain="acme.example",
            new_domain="acme.example",
            shared_linkedin_slug="acme",
        ),
        company_quality=True,
    )
    assert conflict_result.decision == COMPANY_FIT_MISMATCH
    assert _targeted_company_investigation_dimensions(
        conflict_result,
        icp_stage="",
        employee_size_conflict=False,
    ) == ()

    matching = _complete_verdict(observed_company_name="Northstar Systems LLC")
    unproven = _reverify_decision(
        matching,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity={"status": "UNPROVEN"},
        company_quality=True,
    )
    assert unproven.decision == COMPANY_FIT_UNAVAILABLE
    assert unproven.details["identity_receipt"]["reason_code"] == (
        "rebrand_continuity_unproven"
    )


def _completed_same_domain_unproven_rebrand_receipt(**updates):
    receipt = {
        "decision": COMPANY_FIT_UNAVAILABLE,
        "reason_code": "rebrand_continuity_unproven",
        "evidence_source": "company_web_reverification",
        "submitted_name": "dbs",
        "submitted_domain": "dbs.com",
        "submitted_linkedin_slug": "",
        "observed_name": "dbsbankltd",
        "observed_domain": "dbs.com",
        "observed_linkedin_slug": "dbs-bank",
    }
    receipt.update(updates)
    return receipt


def test_completed_same_domain_unproven_rebrand_is_insufficient_evidence():
    receipt = _completed_same_domain_unproven_rebrand_receipt()

    assert lead_scorer._is_same_domain_unproven_web_identity(receipt)
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity",),
        identity_receipt=receipt,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"observed_domain": "dbs.com.sg"},
        {"observed_linkedin_slug": ""},
        {"submitted_linkedin_slug": "dbs", "observed_linkedin_slug": "dbs-bank"},
        {"observed_name": "dbs"},
        {"evidence_source": "company_homepage"},
        {"reason_code": "identity_provider_error"},
        {"observed_name": ""},
    ],
)
def test_incomplete_or_unbound_unproven_rebrand_stays_retryable(updates):
    receipt = _completed_same_domain_unproven_rebrand_receipt(**updates)

    assert not lead_scorer._is_same_domain_unproven_web_identity(receipt)
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity",),
        identity_receipt=receipt,
    )


@pytest.mark.parametrize("receipt", [None, {}, {"decision": "unavailable"}])
def test_empty_or_malformed_identity_receipt_stays_retryable(receipt):
    assert not lead_scorer._is_same_domain_unproven_web_identity(receipt)
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity",),
        identity_receipt=receipt,
    )


def test_rebrand_proof_cannot_bind_an_unrelated_linkedin_company():
    company = _company(
        name="Wayground formerly Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/unrelated",
    )
    proof = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url="https://help.wayground.com/rebrand",
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    assert _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    ).decision == COMPANY_FIT_MISMATCH


def test_rebrand_proof_binds_composite_observed_name_symmetrically():
    company = _company(
        name="Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground (formerly Quizizz)",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/quizizz",
    )
    proof = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url="https://help.wayground.com/rebrand",
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )

    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity_receipt"]["reason_code"] == (
        "verified_rebrand_continuity"
    )


def test_conflicting_current_headcount_is_unproven():
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_quote="PitchBook lists 7 employees.",
    )
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example",
    }
    assert _employee_size_sources_conflict(verdict, structured) is True
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=_company(),
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_UNAVAILABLE
    assert result.details["employee_size_conflict_receipt"] == {
        "status": "UNPROVEN",
        "reason_code": "conflicting_current_headcount",
        "resolution": "unresolved",
        "web_evidence": {
            "url": "https://evidence.example/headcount",
            "quote": "PitchBook lists 7 employees.",
        },
        "structured_evidence": structured,
    }


def test_exact_entity_linkedin_range_is_primary_over_third_party_exact_estimate():
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example/",
    }
    anchor = {
        "normalized_name": "Acme",
        "registrable_dns_domain": "acme.example",
        "linkedin_company_slug": "acme",
    }

    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=_company(),
        verified_homepage_identity=anchor,
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_MATCH
    receipt = result.details["employee_size_conflict_receipt"]
    assert receipt["status"] == "VERIFIED"
    assert receipt["resolution"] == (
        "structured_linkedin_employee_count_range_primary"
    )
    assert receipt["primary_method"] == (
        "harvestapi_exact_company_employeeCountRange"
    )
    assert receipt["evaluation_date"]
    assert receipt["primary_source_url"] == (
        "https://www.linkedin.com/company/acme"
    )
    assert _targeted_company_investigation_dimensions(
        result,
        icp_stage="",
        employee_size_conflict=True,
    ) == ()

    disjoint = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(employee_count="51-200"),
        company=_company(),
        verified_homepage_identity=anchor,
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )
    assert disjoint.decision == COMPANY_FIT_MISMATCH
    assert disjoint.details["employee_size_conflict_receipt"]["status"] == (
        "CONTRADICTED"
    )
    assert _targeted_company_investigation_dimensions(
        disjoint,
        icp_stage="",
        employee_size_conflict=True,
    ) == ()


def test_linkedin_primary_rejects_wrong_identity_member_count_and_linkedin_conflict():
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example/",
    }
    anchor = {
        "normalized_name": "Acme",
        "registrable_dns_domain": "acme.example",
        "linkedin_company_slug": "acme",
    }

    for rejected_evidence, rejected_verdict in (
        (
            dict(structured, url="https://www.linkedin.com/company/unrelated"),
            verdict,
        ),
        (
            dict(structured, source_field="employeeCount"),
            verdict,
        ),
        (
            structured,
            dict(
                verdict,
                employee_size_evidence_url=(
                    "https://www.linkedin.com/company/acme"
                ),
            ),
        ),
    ):
        result = _reverify_decision(
            rejected_verdict,
            "",
            "",
            icp=_icp(),
            company=_company(),
            verified_homepage_identity=anchor,
            structured_employee_size_evidence=rejected_evidence,
            employee_size_conflict=True,
            company_quality=True,
        )
        assert result.decision == COMPANY_FIT_UNAVAILABLE
        assert result.details["employee_size_conflict_receipt"]["resolution"] == (
            "unresolved"
        )


def test_arena_conflict_check_collects_structured_size_without_replacing_direct_proof(
    monkeypatch,
):
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example",
    }
    calls = []

    async def fake_fetch(
        domain, profile_url, *, diagnostic, public_company_evidence, **_kwargs
    ):
        del diagnostic, public_company_evidence
        calls.append((domain, profile_url))
        return structured

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        fake_fetch,
    )
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    cache = {}
    refreshed = asyncio.run(_refresh_linkedin_employee_size_observation(
        verdict,
        _company(),
        _icp(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        invocation_cache=cache,
        collect_structured_conflict=True,
    ))

    assert refreshed == verdict
    assert cache["structured_evidence"] == structured
    assert calls == [
        ("acme.example", "https://www.linkedin.com/company/acme")
    ]
    assert _employee_size_sources_conflict(refreshed, structured) is True


def test_oxpay_schema_repair_recomputes_stale_employee_conflict(monkeypatch):
    """A same-entity repair may omit size without erasing bound profile proof."""

    company = _company(
        name="OxPay",
        website="https://oxpayfinancial.com/",
        linkedin="https://www.linkedin.com/company/oxpayfinancial",
    ).model_copy(update={
        "employee_count": "51-200",
        "company_stage": "Public",
        "country": "Singapore",
        "state": "",
    })
    icp = _icp(
        employee_count="51-200",
        company_stage="Public",
        industry="Payments",
        sub_industry="Payments infrastructure and merchant acquiring",
        product_service="A merchant payments platform.",
        geography="Singapore",
        required_attribute=(
            "Operates a payments business with evidence of a recent strategic "
            "partnership."
        ),
    )
    initial = _complete_verdict(
        observed_company_name="OxPay",
        observed_company_website="https://oxpayfinancial.com",
        observed_company_linkedin=(
            "https://www.linkedin.com/company/oxpayfinancial"
        ),
        observed_employee_count=37,
        employee_size_matches=False,
        employee_size_evidence_url=(
            "https://www.oxpayfinancial.com/reports/sustainability-2024.pdf"
        ),
        employee_size_evidence_quote=(
            "we had 36 full-time employees in FY2023 and 37 full-time "
            "employees in FY2024"
        ),
        observed_industry="Payments",
        observed_subindustry="merchant payment services",
        industry_matches=True,
        industry_activity_role="supplier_operator",
        industry_evidence_url="https://oxpayfinancial.com/about-us/",
        industry_evidence_quote=(
            "OxPay provides merchant payment services through an integrated "
            "platform."
        ),
        observed_hq_country="Singapore",
        observed_hq_state="",
        geography_matches=False,
        geography_evidence_url=(
            "https://www.linkedin.com/company/oxpayfinancial"
        ),
        geography_evidence_quote=(
            "OxPay Financial Limited is headquartered in Singapore."
        ),
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://oxpayfinancial.com/investor-relations/",
        stage_evidence_quote=(
            "OxPay ordinary shares are listed on SGX under ticker TVV."
        ),
        attribute_satisfied=True,
        required_attribute_evidence_url=(
            "https://www.linkedin.com/company/oxpayfinancial"
        ),
        required_attribute_evidence_quote=(
            "OxPay has offices in Singapore, Malaysia, Indonesia and Thailand."
        ),
    )
    repaired = dict(
        initial,
        observed_employee_count=None,
        employee_size_matches=None,
        employee_size_evidence_url="",
        employee_size_evidence_quote="",
        geography_matches=True,
        required_attribute_evidence_url="https://oxpayfinancial.com/about-us/",
        required_attribute_evidence_quote=(
            "OxPay helps merchants process payments through its integrated "
            "platform and announced a strategic partnership."
        ),
    )
    structured = {
        "employee_count": "51-200",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/oxpayfinancial",
        "website": "https://oxpayfinancial.com/",
    }
    calls = {"provider": 0, "investigator": 0, "structured": 0, "current": 0}

    async def provider(**_kwargs):
        calls["provider"] += 1
        return (initial if calls["provider"] == 1 else repaired), ""

    async def structured_profile(
        domain,
        profile_url,
        *,
        diagnostic,
        public_company_evidence,
        **_kwargs,
    ):
        del diagnostic, public_company_evidence
        calls["structured"] += 1
        assert domain == "oxpayfinancial.com"
        assert profile_url == (
            "https://www.linkedin.com/company/oxpayfinancial"
        )
        return structured

    async def current_profile(profile_url, **_kwargs):
        calls["current"] += 1
        return {
            "outcome": lead_scorer.CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE,
            "url": profile_url,
        }

    async def bounded_investigation(*, targets, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("geography",)
        return {
            "claims": {
                "geography": _finding(
                    "geography",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                    reason="No independent headquarters label was found.",
                )
            },
            "failure_reason": "",
            "usage": {"reasoning_turns": 1, "search_calls": 0, "fetch_calls": 0},
        }

    grounding_calls = 0

    async def ground_attribute(candidate, **_kwargs):
        nonlocal grounding_calls
        grounding_calls += 1
        grounded = dict(candidate)
        if grounding_calls == 1:
            grounded.update(
                attribute_satisfied=None,
                required_attribute_evidence_url="",
                required_attribute_evidence_quote="",
            )
        return grounded, {}

    homepage_identity = lead_scorer.company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "oxpay",
                "observed_domain": "oxpayfinancial.com",
                "observed_linkedin_slug": "oxpayfinancial",
            },
            "verified_homepage_transport_domain": "oxpayfinancial.com",
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_profile,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        current_profile,
    )
    monkeypatch.setattr(
        lead_scorer,
        "investigate_company_evidence",
        bounded_investigation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_ground_required_attribute_evidence",
        ground_attribute,
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        company,
        icp,
        require_company_fit_dimensions=True,
        verified_homepage_identity=homepage_identity,
        company_quality=True,
        evidence_investigator=True,
    ))

    assert calls == {
        "provider": 2,
        "investigator": 1,
        "structured": 1,
        "current": 2,
    }
    assert grounding_calls == 2
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_MATCH
    )
    assert result.details["dimension_evidence"]["employee_size"] == structured
    assert result.details["employee_size_conflict"] is False


@pytest.mark.parametrize(
    (
        "investigator_status",
        "activity_role",
        "repair_identity",
        "repair_industry_match",
        "repair_attribute",
        "expected_decision",
        "expected_industry_decision",
        "expected_provider_calls",
    ),
    [
        (
            "VERIFIED",
            "supplier_operator",
            "same",
            False,
            True,
            COMPANY_FIT_MATCH,
            COMPANY_FIT_MATCH,
            2,
        ),
        (
            "CONTRADICTED",
            "internal_function",
            "same",
            True,
            True,
            COMPANY_FIT_MISMATCH,
            COMPANY_FIT_MISMATCH,
            1,
        ),
        (
            "UNPROVEN",
            "unresolved",
            "same",
            None,
            True,
            COMPANY_FIT_UNAVAILABLE,
            COMPANY_FIT_UNAVAILABLE,
            2,
        ),
        (
            "UNPROVEN",
            "unresolved",
            "same",
            True,
            True,
            COMPANY_FIT_MATCH,
            COMPANY_FIT_MATCH,
            2,
        ),
        (
            "VERIFIED",
            "supplier_operator",
            "wrong",
            False,
            True,
            COMPANY_FIT_UNAVAILABLE,
            COMPANY_FIT_MATCH,
            2,
        ),
        (
            "VERIFIED",
            "supplier_operator",
            "same",
            False,
            False,
            COMPANY_FIT_MISMATCH,
            COMPANY_FIT_MATCH,
            2,
        ),
    ],
)
def test_schema_repair_reconciles_bounded_industry_decision(
    monkeypatch,
    investigator_status,
    activity_role,
    repair_identity,
    repair_industry_match,
    repair_attribute,
    expected_decision,
    expected_industry_decision,
    expected_provider_calls,
):
    company = _company()
    icp = _icp(
        industry="Web security",
        sub_industry="Web application firewall",
        product_service="Customer-operated web application firewall controls",
        required_attribute="Offers configurable web application firewall rules.",
    )
    initial = _complete_verdict(
        observed_industry="",
        observed_subindustry="",
        industry_matches=None,
        industry_activity_role="unresolved",
        industry_evidence_url="",
        industry_evidence_quote="",
        attribute_satisfied=None,
        required_attribute_evidence_url="",
        required_attribute_evidence_quote="",
    )
    repaired = _complete_verdict(
        observed_industry=(
            "Cloud platform" if repair_industry_match is not None else ""
        ),
        observed_subindustry=(
            "Web application firewall"
            if repair_industry_match is not None
            else ""
        ),
        industry_matches=repair_industry_match,
        industry_activity_role=(
            "supplier_operator"
            if repair_industry_match is not None
            else "unresolved"
        ),
        industry_evidence_url=(
            "https://acme.example/platform"
            if repair_industry_match is not None
            else ""
        ),
        industry_evidence_quote=(
            "Acme's platform includes configurable web application firewall rules."
            if repair_industry_match is not None
            else ""
        ),
        attribute_satisfied=repair_attribute,
        required_attribute_evidence_url="https://acme.example/platform",
        required_attribute_evidence_quote=(
            "Acme's platform includes configurable web application firewall rules."
        ),
    )
    if repair_identity == "wrong":
        repaired.update(
            observed_company_name="Other Acme",
            observed_company_website="https://other.example",
            observed_company_linkedin=(
                "https://www.linkedin.com/company/other-acme"
            ),
        )
    calls = {"provider": 0, "investigator": 0}

    async def provider(**_kwargs):
        calls["provider"] += 1
        return (initial if calls["provider"] == 1 else repaired), ""

    async def bounded_investigation(*, targets, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("industry",)
        finding = _finding(
            "industry",
            status=investigator_status,
            observed_value=(
                "Web application firewall"
                if investigator_status != "UNPROVEN"
                else None
            ),
            observed_industry=(
                "Web security" if investigator_status != "UNPROVEN" else ""
            ),
            observed_subindustry=(
                "Web application firewall"
                if investigator_status != "UNPROVEN"
                else ""
            ),
            activity_role=activity_role,
            evidence_url=(
                "https://acme.example/firewall"
                if investigator_status != "UNPROVEN"
                else ""
            ),
            evidence_quote=(
                "Acme sells customer-operated web application firewall controls."
                if investigator_status == "VERIFIED"
                else (
                    "Acme uses firewall controls only for its internal compliance."
                    if investigator_status == "CONTRADICTED"
                    else ""
                )
            ),
            reason="bounded industry decision",
        )
        return {
            "claims": {"industry": finding},
            "failure_reason": "",
            "usage": {"reasoning_turns": 1, "search_calls": 0, "fetch_calls": 1},
        }

    async def keep_observation(verdict, *_args, **_kwargs):
        return verdict

    async def keep_attribute(verdict, **_kwargs):
        return verdict, {}

    homepage_identity = lead_scorer.company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "acme",
                "observed_domain": "acme.example",
                "observed_linkedin_slug": "acme",
            },
            "verified_homepage_transport_domain": "acme.example",
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "investigate_company_evidence",
        bounded_investigation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_ground_required_attribute_evidence",
        keep_attribute,
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        company,
        icp,
        require_company_fit_dimensions=True,
        verified_homepage_identity=homepage_identity,
        company_quality=True,
        evidence_investigator=True,
    ))

    assert calls == {
        "provider": expected_provider_calls,
        "investigator": 1,
    }
    assert result.decision == expected_decision
    assert result.details["dimension_decisions"]["industry"] == (
        expected_industry_decision
    )
    if repair_attribute is False:
        assert result.details["required_attribute_decision"] == (
            COMPANY_FIT_MISMATCH
        )


def test_repaired_real_employee_range_conflict_stays_unproven():
    repaired = _complete_verdict(
        observed_company_name="OxPay",
        observed_company_website="https://oxpayfinancial.com/",
        observed_company_linkedin=(
            "https://www.linkedin.com/company/oxpayfinancial"
        ),
        observed_employee_count="11-50",
        employee_size_matches=False,
        employee_size_evidence_url="https://directory.example/oxpay",
        employee_size_evidence_quote="OxPay has 11-50 employees.",
    )
    structured = {
        "employee_count": "51-200",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/oxpayfinancial",
        "website": "https://oxpayfinancial.com/",
    }

    conflict = _employee_size_sources_conflict(repaired, structured)
    result = _reverify_decision(
        repaired,
        "",
        "",
        icp=_icp(employee_count="51-200"),
        company=_company(
            name="OxPay",
            website="https://oxpayfinancial.com/",
            linkedin="https://www.linkedin.com/company/oxpayfinancial",
        ),
        verified_homepage_identity={
            "normalized_name": "OxPay",
            "registrable_dns_domain": "oxpayfinancial.com",
            "linkedin_company_slug": "oxpayfinancial",
        },
        structured_employee_size_evidence=structured,
        employee_size_conflict=conflict,
        company_quality=True,
    )

    assert conflict is True
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )


def test_structured_conflict_check_does_not_expand_calls_for_matching_headcount(
    monkeypatch,
):
    async def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("matching direct evidence must not add a provider call")

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        unexpected_fetch,
    )
    verdict = _complete_verdict()
    cache = {}

    refreshed = asyncio.run(_refresh_linkedin_employee_size_observation(
        verdict,
        _company(),
        _icp(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        invocation_cache=cache,
        collect_structured_conflict=True,
    ))

    assert refreshed == verdict
    assert "structured_attempted" not in cache


def test_non_arena_mismatch_does_not_add_structured_fetch(monkeypatch):
    async def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("ordinary scoring must not add a structured provider call")

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        unexpected_fetch,
    )
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    cache = {}
    refreshed = asyncio.run(_refresh_linkedin_employee_size_observation(
        verdict,
        _company(),
        _icp(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        invocation_cache=cache,
        collect_structured_conflict=False,
    ))

    assert refreshed == verdict
    assert "structured_attempted" not in cache


def test_fetched_current_headcount_repairs_a_nonconflicting_false_negative():
    verdict = _complete_verdict(
        observed_employee_count="2-10",
        employee_size_matches=False,
        employee_size_evidence_quote="Acme has 2-10 employees.",
    )
    finding = _finding(
        "headcount",
        observed_value=27,
        evidence_url="https://acme.example/about",
        evidence_quote="Acme has 27 employees company-wide.",
    )
    projected = _project_investigator_headcount(
        verdict,
        finding,
        icp=_icp(),
        existing_conflict=False,
    )
    assert projected["observed_employee_count"] == 27
    assert projected["employee_size_matches"] is True

    unchanged = _project_investigator_headcount(
        verdict,
        finding,
        icp=_icp(),
        existing_conflict=True,
    )
    assert unchanged["observed_employee_count"] == "2-10"

    canonical_band = dict(finding, observed_value="11-50")
    projected_band = _project_investigator_headcount(
        verdict,
        canonical_band,
        icp=_icp(),
        existing_conflict=False,
    )
    assert projected_band["observed_employee_count"] == "11-50"
    assert projected_band["employee_size_matches"] is True

    for invalid_value in ("11ish-50ish", "100000-200000"):
        rejected = _project_investigator_headcount(
            verdict,
            dict(finding, observed_value=invalid_value),
            icp=_icp(),
            existing_conflict=False,
        )
        assert rejected["observed_employee_count"] == "2-10"


def test_conflicting_third_party_ranges_remain_unproven():
    verdict = _complete_verdict(
        observed_employee_count="11-50",
        employee_size_matches=True,
        employee_size_evidence_url="https://directory.example/acme",
        employee_size_evidence_quote="Acme Company size 11-50 employees.",
    )
    structured = {
        "employee_count": "51-200",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example/",
    }
    assert _employee_size_sources_conflict(verdict, structured) is True

    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=_company(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    receipt = result.details["employee_size_conflict_receipt"]
    assert receipt["status"] == "UNPROVEN"
    assert receipt["resolution"] == "unresolved"


def test_headcount_finding_binds_value_and_rejects_scoped_counts():
    url = "https://acme.example/about"
    finding = _finding(
        "headcount",
        observed_value=27,
        evidence_url=url,
        evidence_quote="Acme has 27 employees company-wide.",
    )
    accepted = _validated_findings(
        {"findings": [finding]},
        targets=("headcount",),
        fetched_pages={url: finding["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert accepted["headcount"]["status"] == "VERIFIED"

    linkedin_band = dict(
        finding,
        observed_value="11-50",
        evidence_quote="Acme Company size 11-50 employees.",
    )
    accepted_band = _validated_findings(
        {"findings": [linkedin_band]},
        targets=("headcount",),
        fetched_pages={url: linkedin_band["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert accepted_band["headcount"]["status"] == "VERIFIED"

    mismatched_value = dict(finding, observed_value=99)
    rejected_value = _validated_findings(
        {"findings": [mismatched_value]},
        targets=("headcount",),
        fetched_pages={url: finding["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert rejected_value["headcount"]["status"] == "UNPROVEN"

    lower_bound = dict(
        finding,
        observed_value=4100,
        evidence_quote=(
            "Samsara has more than 4,100 full-time employees company-wide."
        ),
    )
    rejected_lower_bound = _validated_findings(
        {"findings": [lower_bound]},
        targets=("headcount",),
        fetched_pages={url: lower_bound["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"samsara"},
    )
    assert rejected_lower_bound["headcount"]["status"] == "UNPROVEN"

    upper_bound = dict(
        finding,
        observed_value=4100,
        evidence_quote="Samsara has up to 4,100 employees company-wide.",
    )
    rejected_upper_bound = _validated_findings(
        {"findings": [upper_bound]},
        targets=("headcount",),
        fetched_pages={url: upper_bound["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"samsara"},
    )
    assert rejected_upper_bound["headcount"]["status"] == "UNPROVEN"

    for quote in (
        "Samsara has >4,100 employees company-wide.",
        "Samsara has 3,000-4,100 employees company-wide.",
        "Samsara has approximately 4,100 employees company-wide.",
    ):
        bounded = dict(finding, observed_value=4100, evidence_quote=quote)
        rejected = _validated_findings(
            {"findings": [bounded]},
            targets=("headcount",),
            fetched_pages={url: quote},
            first_party_domains={"acme.example"},
            identity_names={"samsara"},
        )
        assert rejected["headcount"]["status"] == "UNPROVEN"

    office_count = dict(
        finding,
        evidence_quote="Acme's London office has 27 employees.",
    )
    rejected_scope = _validated_findings(
        {"findings": [office_count]},
        targets=("headcount",),
        fetched_pages={url: office_count["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert rejected_scope["headcount"]["status"] == "UNPROVEN"

    associated_members = dict(
        finding,
        evidence_quote="Acme has 27 associated members on LinkedIn.",
    )
    rejected_members = _validated_findings(
        {"findings": [associated_members]},
        targets=("headcount",),
        fetched_pages={url: associated_members["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert rejected_members["headcount"]["status"] == "UNPROVEN"

    for non_headcount_quote in (
        "Acme was founded in 2020.",
        "Acme reported 27 million dollars in annual revenue.",
    ):
        unrelated_number = dict(
            finding,
            observed_value=2020 if "2020" in non_headcount_quote else 27,
            evidence_quote=non_headcount_quote,
        )
        rejected_number = _validated_findings(
            {"findings": [unrelated_number]},
            targets=("headcount",),
            fetched_pages={url: non_headcount_quote},
            first_party_domains={"acme.example"},
            identity_names={"acme"},
        )
        assert rejected_number["headcount"]["status"] == "UNPROVEN"


def test_unproven_rebrand_conflict_is_not_turned_into_a_false_mismatch():
    company = _company(
        name="Wayground formerly Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/quizizz",
    )
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity={"status": "UNPROVEN"},
        company_quality=True,
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert (
        result.details["identity_receipt"]["reason_code"]
        == "rebrand_continuity_unproven"
    )


def test_public_stage_needs_listing_proof_not_labels_or_plans():
    assert _stage_quote_supports_observation(
        "public",
        "CoStar Group common stock is listed on NASDAQ under ticker CSGP.",
    )
    assert not _stage_quote_supports_observation("public", "Company type: Public Company")
    assert not _stage_quote_supports_observation(
        "public", "The company plans an initial public offering next year."
    )


def test_acculon_retrospective_series_a_statement_requires_recipient_binding():
    quote = (
        "The facility’s opening follows a period of rapid growth for Acculon, "
        "including a Series A investment led by Terex Corporation and the "
        "expansion of its Acculon Labs testing division."
    )

    assert not _stage_quote_supports_observation("series a", quote)
    assert not _stage_quote_supports_observation("series b", quote)
    assert _stage_quote_supports_observation(
        "series a",
        "Acculon completed a Series A financing round led by Terex Corporation.",
    )


@pytest.mark.parametrize(
    "quote",
    [
        (
            "Terex is making a Series A investment in Acculon Energy to "
            "accelerate electrification."
        ),
        (
            "The facility opening is expected to follow a period of growth, "
            "including a Series A investment led by Terex Corporation."
        ),
        (
            "The facility opening follows planned growth, including a "
            "proposed Series A investment led by Terex Corporation."
        ),
        (
            "If completed, the facility opening follows a period of growth, "
            "including a Series A investment led by Terex Corporation."
        ),
        (
            "Subject to approval, the facility opening follows a period of "
            "growth, including a Series A investment led by Terex Corporation."
        ),
        (
            "Previously, the facility opening followed a period of growth, "
            "including a Series A investment led by Terex Corporation."
        ),
        (
            "The facility opening follows a period of growth, including a "
            "Series A investment that might close next year."
        ),
        (
            "Acculon's investment policy follows a diversified portfolio "
            "strategy, including a Series A investment in Beta Corporation "
            "led by Terex."
        ),
        (
            "Acculon reported that the facility opening follows a period of "
            "rapid growth for Beta Corporation, including a Series A "
            "investment led by Terex."
        ),
        (
            "The facility opening follows a period of growth for Acculon, "
            "including a Series A investment for Beta Corporation led by Terex."
        ),
        (
            "Terex's investor portfolio follows a period of growth for Acculon, "
            "including a Series A investment led by Terex."
        ),
    ],
)
def test_retrospective_round_pattern_rejects_uncompleted_or_old_events(quote):
    assert not _stage_quote_supports_observation("series a", quote)


def test_retrospective_round_stays_unproven_and_keeps_company_binding():
    series_a = (
        "The facility opening follows rapid growth for Acculon, including a "
        "Series A investment led by Terex Corporation."
    )
    later_series_b = "Acculon later completed a Series B financing round."
    combined = f"{series_a} {later_series_b}"

    assert not _stage_quote_supports_observation("series a", series_a)
    assert not _stage_quote_supports_observation("series a", combined)
    assert _stage_quote_supports_observation("series b", combined)

    url = "https://news.example/facility"
    quote_without_company = (
        "The facility opening follows rapid growth, including a Series A "
        "investment led by Terex Corporation."
    )
    finding = _validated_findings(
        {"findings": [_finding(
            "stage",
            observed_value="Series A",
            evidence_url=url,
            evidence_quote=quote_without_company,
        )]},
        targets=("stage",),
        fetched_pages={url: quote_without_company},
        first_party_domains={"acculonenergy.com"},
        identity_names={"acculonenergy"},
    )["stage"]
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "source quote did not identify the investigated company"


def test_investigation_request_uses_frozen_evaluation_date(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        arguments = {
            "findings": [
                _finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                )
            ]
        }
        return 200, {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "submit_findings",
                            "arguments": json.dumps(arguments),
                        },
                    }]
                }
            }]
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "evaluation_date", lambda: date(2026, 9, 18))
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        requested_industry="Technology",
        requested_subindustry="Education Technology",
        requested_product_service="Platforms that enable online learning",
        requested_attribute="Sells a university learning platform",
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["evaluation_date"] == "2026-09-18"
    assert input_document["requested_industry"] == "Technology"
    assert input_document["requested_subindustry"] == "Education Technology"
    assert input_document["requested_product_service"] == (
        "Platforms that enable online learning"
    )
    assert input_document["requested_attribute"] == (
        "Sells a university learning platform"
    )
    assert input_document["investigation_limits"] == {
        "reasoning_turns": 8,
        "search_calls": 2,
        "fetch_calls": 3,
        "admission_deadline_seconds": 110.0,
    }
    assert "official company investor-relations pages" in (
        requests[0]["messages"][0]["content"]
    )
    assert "never combine a quote from one page" in (
        requests[0]["messages"][0]["content"]
    )


def test_full_harness_loop_searches_fetches_and_submits_fetched_quote(monkeypatch):
    url = "https://acme.example/investors"
    quote = "Acme common stock is listed on NASDAQ under ticker ACME."
    reasoning_requests = []
    search_queries = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        reasoning_requests.append(payload)
        turn = len(reasoning_requests)
        if turn == 1:
            name, arguments = "search_web", {"query": "Acme current stock listing"}
        elif turn == 2:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding("stage", evidence_url=url, evidence_quote=quote)]
            }
        return 200, {
            "choices": [{"message": {"tool_calls": [{
                "id": f"call-{turn}",
                "index": 0,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}}]
        }

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": url}], "notice": "discovery_only_not_evidence"}

    async def fake_fetch(_session, requested_url):
        fetched_urls.append(requested_url)
        return {"ok": True, "url": requested_url, "text": quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_quote"] == quote
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 1,
        "fetch_calls": 1,
    }
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        url: {"final_url": url, "text": quote}
    }
    assert investigator.PRIVATE_FETCHED_PAGES_KEY not in result["claims"]
    assert search_queries == ["Acme current stock listing"]
    assert fetched_urls == [url]
    assert all(
        request["model"] == investigator.INVESTIGATOR_MODEL
        for request in reasoning_requests
    )
    replayed_call = reasoning_requests[1]["messages"][-2]["tool_calls"][0]
    assert set(replayed_call) == {"id", "type", "function"}
    assert set(replayed_call["function"]) == {"name", "arguments"}
    assert all(
        request["tool_choice"] == "required"
        for request in reasoning_requests
    )


def test_saved_doctronic_case_script_discovers_and_fetches_current_series_b(
    monkeypatch,
):
    old_url = (
        "https://www.vcaonline.com/news/2025091504/"
        "doctronic-raises-20-million-series-a-to-bring-private-and-"
        "personalized-ai-doctor-to-the-masses/"
    )
    current_url = (
        "https://www.businesswire.com/news/home/20260324814372/en/"
        "Doctronic-Raises-%2440M-Series-B-Following-Breakthrough-as-First-"
        "AI-to-Legally-Renew-Prescriptions-in-the-U.S."
    )
    old_quote = (
        "Doctronic, the AI-native platform delivering fast, private, and "
        "personalized healthcare at scale, today announced a $20 million "
        "Series A round led by Lightspeed Venture Partners"
    )
    current_quote = (
        "Doctronic, the first AI system legally authorized to practice medicine "
        "in the United States, today announced a $40 million Series B round "
        "co-led by Abstract and Lightspeed Venture Partners"
    )
    requests = []
    search_queries = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": old_url}
        elif turn == 2:
            name, arguments = "fetch_page", {"url": current_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                status="CONTRADICTED",
                observed_value="Series B",
                evidence_url=current_url,
                evidence_quote=current_quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": current_url}]}

    async def fake_fetch(_session, url):
        fetched_urls.append(url)
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "text": old_quote if url == old_url else current_quote,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Doctronic", "website": "https://doctronic.ai"},
        targets=("stage",),
        requested_stage="Series A",
        prior_observations={"submitted_source_urls": [old_url]},
    ))

    assert result["claims"]["stage"]["status"] == "CONTRADICTED"
    assert result["claims"]["stage"]["observed_value"] == "Series B"
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 1,
        "fetch_calls": 2,
    }
    assert search_queries == [
        "Doctronic doctronic.ai latest funding round acquisition IPO"
    ]
    assert fetched_urls == [old_url, current_url]
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    stage_discovery = input_document["server_current_stage_discovery"]
    assert stage_discovery["ok"] is True
    assert stage_discovery["query"] == search_queries[0]
    assert stage_discovery["discovery"]["results"] == [{"url": current_url}]
    assert stage_discovery["notice"] == (
        "server_search_results_are_discovery_only_not_evidence"
    )
    assert input_document["investigation_limits"]["remaining_search_calls"] == 1


@pytest.mark.parametrize(
    ("later_url", "later_quote"),
    [
        ("", ""),
        (
            "https://news.example/acme-debt",
            "Acme secured a new debt facility to support continued growth.",
        ),
        (
            "https://news.example/acme-grant",
            "Acme received a government grant for product development.",
        ),
        (
            "https://news.example/acme-buys-otherco",
            "Acme completed its acquisition of OtherCo.",
        ),
    ],
)
def test_current_stage_search_preserves_valid_series_a_without_superseding_event(
    monkeypatch,
    later_url,
    later_quote,
):
    old_url = "https://acme.example/news/series-a"
    old_quote = "Acme today announced a completed $20 million Series A round."
    requests = []
    search_queries = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if not later_url and turn == 1:
            name, arguments = "search_web", {"query": "Acme financing history"}
        elif turn == (2 if not later_url else 1):
            name, arguments = "fetch_page", {"url": old_url}
        elif later_url and turn == 2:
            name, arguments = "fetch_page", {"url": later_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                observed_value="Series A",
                evidence_url=old_url,
                evidence_quote=old_quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": ([{"url": later_url}] if later_url else [])}

    async def fake_fetch(_session, url):
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "text": old_quote if url == old_url else later_quote,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Series A",
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["observed_value"] == "Series A"
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 1 if later_url else 2,
        "fetch_calls": 2 if later_url else 1,
    }
    assert search_queries[0] == (
        "Acme acme.example latest funding round acquisition IPO"
    )


@pytest.mark.parametrize(
    "search_mode", ["provider_failure", "malformed_response", "zero_budget"]
)
@pytest.mark.parametrize("submitted_status", ["VERIFIED", "CONTRADICTED"])
def test_required_current_stage_search_failure_returns_unproven(
    monkeypatch,
    search_mode,
    submitted_status,
):
    old_url = "https://acme.example/news/series-a"
    old_quote = "Acme today announced a completed $20 million Series A round."
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": old_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                status=submitted_status,
                observed_value="Series A",
                evidence_url=old_url,
                evidence_quote=old_quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del query, key
        if search_mode == "zero_budget":
            raise AssertionError("zero search budget must not call the provider")
        if search_mode == "malformed_response":
            raise ValueError("malformed search response")
        raise RuntimeError("search unavailable")

    async def fake_fetch(_session, url):
        return {"ok": True, "url": url, "final_url": url, "text": old_quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    if search_mode == "zero_budget":
        monkeypatch.setattr(investigator, "MAX_SEARCH_CALLS", 0)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Series A",
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["stage"]["evidence_url"] == ""
    assert result["claims"]["stage"]["evidence_quote"] == ""
    assert result["usage"]["search_calls"] == (
        0 if search_mode == "zero_budget" else 1
    )
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["investigation_limits"]["remaining_search_calls"] == (
        0 if search_mode == "zero_budget" else 1
    )


@pytest.mark.parametrize("deadline_mode", ["before_admission", "after_admission"])
def test_current_stage_search_deadline_returns_unproven(monkeypatch, deadline_mode):
    old_url = "https://acme.example/news/series-a"
    old_quote = "Acme today announced a completed $20 million Series A round."
    requests = []
    searches = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            name, arguments = "fetch_page", {"url": old_url}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage",
                observed_value="Series A",
                evidence_url=old_url,
                evidence_quote=old_quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, url):
        return {"ok": True, "url": url, "final_url": url, "text": old_quote}

    async def fake_search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": []}

    monotonic_values = iter(
        (0.0, 111.0, 111.0)
        if deadline_mode == "before_admission"
        else (0.0, 0.0, 111.0)
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Series A",
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"] == {
        "reasoning_turns": 0,
        "search_calls": 0 if deadline_mode == "before_admission" else 1,
        "fetch_calls": 0,
    }
    assert len(searches) == result["usage"]["search_calls"]


@pytest.mark.parametrize("search_mode", ["provider_failure", "malformed_response"])
def test_failed_current_stage_search_preserves_valid_other_findings(
    monkeypatch,
    search_mode,
):
    url = "https://acme.example/about-and-series-a"
    stage_quote = "Acme today announced a completed $20 million Series A round."
    headcount_quote = "Acme has 11-50 employees."
    page = f"{stage_quote} {headcount_quote}"
    requests = []

    def findings():
        return [
            _finding(
                "stage",
                observed_value="Series A",
                evidence_url=url,
                evidence_quote=stage_quote,
            ),
            _finding(
                "headcount",
                observed_value="11-50",
                evidence_url=url,
                evidence_quote=headcount_quote,
            ),
        ]

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {"findings": findings()}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {
            "ok": True,
            "url": requested_url,
            "final_url": requested_url,
            "text": page,
        }

    async def fake_search(_session, query, *, key):
        del query, key
        if search_mode == "malformed_response":
            raise ValueError("malformed search response")
        raise RuntimeError("search unavailable")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage", "headcount"),
        requested_stage="Series A",
        requested_employee_buckets=("11-50",),
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["headcount"]["status"] == "VERIFIED"
    assert result["claims"]["headcount"]["evidence_quote"] == headcount_quote


def test_curated_armada_relationship_rejection_forces_targeted_research(
    monkeypatch,
):
    """A curated Armada-shaped fixture tests control flow, not a live verdict."""

    infrastructure_url = "https://www.armada.ai/news/edge-ai"
    infrastructure_quote = (
        "Armada is the hyperscaler for the edge, delivering modular AI "
        "infrastructure from first deployment to AI factory with speed, "
        "scale and sovereignty."
    )
    software_url = "https://www.armada.ai/platform"
    software_quote = (
        "Armada provides cloud software for IT teams to manage edge "
        "infrastructure."
    )
    requests = []
    search_queries = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": infrastructure_url}
        elif turn == 2:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status="CONTRADICTED",
                    observed_industry="AI infrastructure",
                    observed_subindustry="Modular edge infrastructure",
                    activity_role="supplier_operator",
                    evidence_url=infrastructure_url,
                    evidence_quote=infrastructure_quote,
                )]
            }
        elif turn == 3:
            name, arguments = "search_web", {
                "query": "Armada cloud software IT teams"
            }
        elif turn == 4:
            name, arguments = "fetch_page", {"url": software_url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    observed_industry="Software",
                    observed_subindustry="Cloud software",
                    activity_role="supplier_operator",
                    evidence_url=software_url,
                    evidence_quote=software_quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": software_url}]}

    async def fake_fetch(_session, url):
        fetched_urls.append(url)
        text = (
            infrastructure_quote if url == infrastructure_url else software_quote
        )
        return {"ok": True, "url": url, "text": text}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Armada",
            "website": "https://www.armada.ai",
        },
        targets=("industry",),
        requested_industry="Software",
        requested_subindustry="Cloud software",
        requested_product_service="Cloud platforms for IT teams",
        requested_attribute="Provides cloud software to IT teams",
        verified_homepage_identity={
            "normalized_name": "Armada",
            "registrable_dns_domain": "armada.ai",
        },
    ))

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_url"] == software_url
    assert result["usage"] == {
        "reasoning_turns": 5,
        "search_calls": 1,
        "fetch_calls": 2,
    }
    assert search_queries == ["Armada cloud software IT teams"]
    assert fetched_urls == [infrastructure_url, software_url]
    assert requests[2]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_web"},
    }
    feedback = json.loads(requests[2]["messages"][-1]["content"])
    assert "requested industry, product/service, and attribute" in (
        feedback["instruction"]
    )
    assert "Search results are discovery only" in feedback["instruction"]


def test_verified_supplier_industry_finding_does_not_force_research(monkeypatch):
    status = "VERIFIED"
    activity_role = "supplier_operator"
    quote = "Armada provides cloud software for IT teams."
    url = "https://www.armada.ai/platform"
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status=status,
                    observed_industry="Software",
                    observed_subindustry="Cloud software",
                    activity_role=activity_role,
                    evidence_url=url,
                    evidence_quote=quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": quote}

    search = AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Armada",
            "website": "https://www.armada.ai",
        },
        targets=("industry",),
        requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "Armada",
            "registrable_dns_domain": "armada.ai",
        },
    ))

    assert result["claims"]["industry"]["status"] == status
    assert len(requests) == 2
    search.assert_not_awaited()


@pytest.mark.parametrize(
    (
        "company_name",
        "company_domain",
        "requested_industry",
        "initial_quote",
        "followup_path",
        "followup_quote",
        "final_status",
        "final_role",
    ),
    [
        pytest.param(
            "IDP",
            "idp.com",
            "Education",
            "IDP helps students apply to universities around the world.",
            "english-schools",
            "IDP operates English-language schools for international students.",
            "VERIFIED",
            "supplier_operator",
            id="other-activity-page-recovers-requested-role",
        ),
        pytest.param(
            "Dimer Health",
            "dimerhealth.com",
            "Software",
            "Dimer Health is a medical practice that treats patients online.",
            "technology",
            "Dimer Health uses scheduling software supplied by Example Systems.",
            "CONTRADICTED",
            "customer_user",
            id="medical-practice-remains-not-software-vendor",
        ),
        pytest.param(
            "RetailCo",
            "retail.example",
            "Payments",
            "RetailCo sells clothing through its online store.",
            "checkout",
            "RetailCo's partner ExamplePay supplies payment processing at checkout.",
            "CONTRADICTED",
            "third_party",
            id="synthetic-wrong-role-remains-rejected",
        ),
    ],
)
def test_no_search_customer_role_gets_one_bounded_industry_followup(
    monkeypatch,
    company_name,
    company_domain,
    requested_industry,
    initial_quote,
    followup_path,
    followup_quote,
    final_status,
    final_role,
):
    """Other activity is not absence; repeated customer evidence remains negative."""

    homepage_url = f"https://{company_domain}/about"
    followup_url = f"https://{company_domain}/{followup_path}"
    requests = []
    search_queries = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": homepage_url}
        elif turn == 2:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status="CONTRADICTED",
                    observed_industry="Other activity",
                    observed_subindustry="Other business role",
                    activity_role="customer_user",
                    evidence_url=homepage_url,
                    evidence_quote=initial_quote,
                )]
            }
        elif turn == 3:
            name, arguments = "search_web", {
                "query": f"{company_name} {requested_industry} company activity"
            }
        elif turn == 4:
            name, arguments = "fetch_page", {"url": followup_url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status=final_status,
                    observed_industry=requested_industry,
                    observed_subindustry=(
                        "English-language schools"
                        if final_role == "supplier_operator"
                        else "Customer use"
                    ),
                    activity_role=final_role,
                    evidence_url=followup_url,
                    evidence_quote=followup_quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": followup_url}]}

    async def fake_fetch(_session, url):
        fetched_urls.append(url)
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "text": initial_quote if url == homepage_url else followup_quote,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": company_name,
            "website": f"https://{company_domain}",
        },
        targets=("industry",),
        requested_industry=requested_industry,
        verified_homepage_identity={
            "normalized_name": company_name,
            "registrable_dns_domain": company_domain,
        },
    ))

    assert result["claims"]["industry"]["status"] == final_status
    assert result["claims"]["industry"]["activity_role"] == final_role
    assert result["usage"] == {
        "reasoning_turns": 5,
        "search_calls": 1,
        "fetch_calls": 2,
    }
    assert search_queries == [
        f"{company_name} {requested_industry} company activity"
    ]
    assert fetched_urls == [homepage_url, followup_url]
    assert requests[2]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_web"},
    }
    assert requests[3]["tool_choice"] == {
        "type": "function",
        "function": {"name": "fetch_page"},
    }


def test_successful_targeted_search_does_not_add_industry_followup_calls(
    monkeypatch,
):
    url = "https://dimerhealth.com/technology"
    quote = "Dimer Health uses scheduling software supplied by Example Systems."
    requests = []
    search_queries = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "search_web", {"query": "Dimer Health software"}
        elif turn == 2:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status="CONTRADICTED",
                    observed_industry="Software",
                    observed_subindustry="Customer use",
                    activity_role="customer_user",
                    evidence_url=url,
                    evidence_quote=quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": url}]}

    async def fake_fetch(_session, requested_url):
        fetched_urls.append(requested_url)
        return {
            "ok": True,
            "url": requested_url,
            "final_url": requested_url,
            "text": quote,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Dimer Health",
            "website": "https://dimerhealth.com",
        },
        targets=("industry",),
        requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "Dimer Health",
            "registrable_dns_domain": "dimerhealth.com",
        },
    ))

    assert result["claims"]["industry"]["status"] == "CONTRADICTED"
    assert result["claims"]["industry"]["activity_role"] == "customer_user"
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 1,
        "fetch_calls": 1,
    }
    assert search_queries == ["Dimer Health software"]
    assert fetched_urls == [url]
    assert len(requests) == 3


@pytest.mark.parametrize(
    "forced_response",
    ["wrong_tool", "malformed_arguments", "overlong_query"],
)
def test_forced_industry_search_must_return_a_valid_search_call(
    monkeypatch, forced_response,
):
    url = "https://www.armada.ai/news/edge-ai"
    quote = "Armada supplies modular AI infrastructure at the edge."
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
            raw_arguments = json.dumps(arguments)
        elif turn == 2:
            name = "submit_findings"
            raw_arguments = json.dumps({"findings": [_finding(
                "industry",
                status="CONTRADICTED",
                observed_industry="AI infrastructure",
                activity_role="supplier_operator",
                evidence_url=url,
                evidence_quote=quote,
            )]})
        elif forced_response == "wrong_tool":
            name = "submit_findings"
            raw_arguments = json.dumps({"findings": [_finding(
                "industry", status="UNPROVEN"
            )]})
        elif forced_response == "malformed_arguments":
            name = "search_web"
            raw_arguments = json.dumps({})
        else:
            name = "search_web"
            raw_arguments = json.dumps({"query": "x" * 501})
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": raw_arguments},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": quote}

    search = AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Armada",
            "website": "https://www.armada.ai",
        },
        targets=("industry",),
        requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "Armada",
            "registrable_dns_domain": "armada.ai",
        },
    ))

    assert requests[2]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_web"},
    }
    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    search.assert_not_awaited()


def test_forced_industry_search_does_not_accept_an_unfetched_result(monkeypatch):
    infrastructure_url = "https://www.armada.ai/news/edge-ai"
    infrastructure_quote = "Armada supplies modular AI infrastructure at the edge."
    software_url = "https://www.armada.ai/platform"
    software_quote = "Armada provides cloud software for IT teams."
    requests = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": infrastructure_url}
        elif turn == 2:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status="CONTRADICTED",
                    observed_industry="AI infrastructure",
                    activity_role="supplier_operator",
                    evidence_url=infrastructure_url,
                    evidence_quote=infrastructure_quote,
                )]
            }
        elif turn == 3:
            name, arguments = "search_web", {"query": "Armada cloud software"}
        elif turn == 4:
            name, arguments = "fetch_page", {"url": software_url}
        elif turn == 5:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    observed_industry="Software",
                    activity_role="supplier_operator",
                    evidence_url=software_url,
                    evidence_quote=software_quote,
                )]
            }
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding("industry", status="UNPROVEN")]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, url):
        fetched_urls.append(url)
        return {"ok": True, "url": url, "text": infrastructure_quote}

    async def fake_search(_session, query, *, key):
        del query, key
        return {"results": [{"url": software_url}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Armada",
            "website": "https://www.armada.ai",
        },
        targets=("industry",),
        requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "Armada",
            "registrable_dns_domain": "armada.ai",
        },
    ))

    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["evidence_url"] == ""
    assert fetched_urls == [infrastructure_url, software_url]


@pytest.mark.parametrize("exhausted_budget", ["search", "fetch"])
def test_exhausted_budget_does_not_force_a_useless_industry_search(
    monkeypatch, exhausted_budget,
):
    url = "https://www.armada.ai/news/edge-ai"
    quote = "Armada supplies modular AI infrastructure at the edge."
    requests = []
    search_queries = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if exhausted_budget == "search" and turn == 1:
            name, arguments = "search_web", {"query": "Armada company profile"}
        elif turn == (2 if exhausted_budget == "search" else 1):
            name, arguments = "fetch_page", {"url": url}
        elif turn == (3 if exhausted_budget == "search" else 2):
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "industry",
                    status="CONTRADICTED",
                    observed_industry="AI infrastructure",
                    activity_role="supplier_operator",
                    evidence_url=url,
                    evidence_quote=quote,
                )]
            }
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding("industry", status="UNPROVEN")]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": []}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    if exhausted_budget == "search":
        monkeypatch.setattr(investigator, "MAX_SEARCH_CALLS", 1)
    else:
        monkeypatch.setattr(investigator, "MAX_FETCH_CALLS", 1)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Armada",
            "website": "https://www.armada.ai",
        },
        targets=("industry",),
        requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "Armada",
            "registrable_dns_domain": "armada.ai",
        },
    ))

    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert search_queries == (
        ["Armada company profile"] if exhausted_budget == "search" else []
    )
    submission_after_rejection = 3 if exhausted_budget == "search" else 2
    assert requests[submission_after_rejection]["tool_choice"] == "required"


def test_prefetched_grab_source_requires_independent_exact_stage_submission(
    monkeypatch,
):
    url = "https://www.grab.com/sg/press/others/atome-financial/"
    quote = "Grab Holdings Limited (NASDAQ: GRAB) (“Grab”)"
    text = (
        "Grab to acquire majority stake in Atome Financial. SINGAPORE, "
        f"September 15, 2026 — {quote} and Atome Financial announced that "
        "Grab entered an agreement to acquire a controlling 60% interest."
    )
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        arguments = {
            "findings": [_finding(
                "stage",
                observed_value="Public",
                evidence_url=url,
                evidence_quote=quote,
            )]
        }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit-grab-stage",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps(arguments),
            },
        }]}}]}

    network_fetch = AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", network_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Grab", "website": "https://grab.com/"},
        targets=("stage",),
        requested_stage="Public",
        requested_attribute=(
            "Announced a strategic partnership in the last 12 months."
        ),
        prior_observations={"submitted_source_urls": [url]},
        prefetched_pages={url: {"final_url": url, "text": text}},
    ))

    assert result["claims"] == {
        "stage": _finding(
            "stage",
            observed_value="Public",
            evidence_url=url,
            evidence_quote=quote,
        )
    }
    assert result["_validated_stage_finding"] == result["claims"]["stage"]
    assert _stage_quote_supports_observation("public", quote)
    assert result["usage"] == {
        "reasoning_turns": 1,
        "search_calls": 0,
        "fetch_calls": 0,
    }
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        url: {"final_url": url, "text": text}
    }
    assert network_fetch.await_count == 0
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["prefetched_sources"] == [{"url": url, "text": text}]
    assert input_document["investigation_limits"]["remaining_fetch_calls"] == 2
    assert "strategic partnership" in input_document["requested_attribute"]
    assert "required_attribute" not in result["claims"]


def test_provider_injected_prefetched_body_is_not_reused(monkeypatch):
    url = "https://evil.example/grab"
    quote = "Grab Holdings Limited (NASDAQ: GRAB)"
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        arguments = {"findings": [_finding(
            "stage",
            observed_value="Public",
            evidence_url=url,
            evidence_quote=quote,
        )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"submit-{len(requests)}",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps(arguments),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 1)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Grab", "website": "https://grab.com/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [url],
            "prefetched_sources": [{"url": url, "text": quote}],
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                url: {"final_url": url, "text": quote}
            },
        },
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert "prefetched_sources" not in requests[0]["messages"][1]["content"]
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {}


def test_prefetched_pages_reduce_remaining_network_fetch_budget(monkeypatch):
    first_url = "https://acme.example/investors"
    second_url = "https://acme.example/about"
    third_url = "https://exchange.example/acme"
    fourth_url = "https://news.example/acme"
    quote = "Acme common stock is listed on NASDAQ under ticker ACME."
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            name, arguments = "fetch_page", {"url": third_url}
        elif len(requests) == 2:
            name, arguments = "fetch_page", {"url": fourth_url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage", evidence_url=first_url, evidence_quote=quote
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    network_fetch = AsyncMock(return_value={
        "ok": True,
        "url": third_url,
        "final_url": third_url,
        "text": "Exchange page with no additional company proof.",
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", network_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [first_url, second_url]
        },
        prefetched_pages={
            first_url: {"final_url": first_url, "text": quote},
            second_url: {
                "final_url": second_url,
                "text": "Acme builds enterprise software.",
            },
        },
    ))

    assert network_fetch.await_count == 1
    assert result["usage"]["fetch_calls"] == 1
    assert len(result[investigator.PRIVATE_FETCHED_PAGES_KEY]) == 3
    exhausted = json.loads(requests[2]["messages"][-1]["content"])
    assert exhausted == {"ok": False, "error": "fetch_budget_exhausted"}


def test_harness_accepts_semantic_stage_finding_after_source_checks(monkeypatch):
    url = "https://costar.example/investors"
    quote = (
        "CoStar Group is listed on the Example Stock Exchange of Example "
        "Jurisdiction under the stock code 1828."
    )
    reasoning_requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        reasoning_requests.append(payload)
        turn = len(reasoning_requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    evidence_url=url,
                    evidence_quote=quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "index": 0,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        assert requested_url == url
        return {
            "ok": True,
            "url": requested_url,
            "text": quote,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "CoStar Group",
            "website": "https://costar.example",
        },
        targets=("stage",),
        requested_stage="Public",
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_quote"] == quote
    assert result["_validated_stage_finding"] == result["claims"]["stage"]
    assert result["usage"] == {
        "reasoning_turns": 2,
        "search_calls": 0,
        "fetch_calls": 1,
    }
    assert len(reasoning_requests) == 2
    assert all(
        request["tool_choice"] == "required"
        for request in reasoning_requests
    )


def test_agent_unproven_historical_listing_remains_unproven(monkeypatch):
    url = "https://costar.example/investors"
    weak_quote = "CoStar Group was listed on 20 May 1970 on the Example Exchange."
    reasoning_requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        reasoning_requests.append(payload)
        turn = len(reasoning_requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                    reason="A historic listing event does not prove current listing.",
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": weak_quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 2)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "CoStar Group",
            "website": "https://costar.example",
        },
        targets=("stage",),
        requested_stage="Public",
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["_validated_stage_finding"] == {}
    assert result["usage"]["reasoning_turns"] == 2
    assert len(reasoning_requests) == 2
    assert reasoning_requests[-1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }


def test_final_unproven_with_evidence_gets_one_submit_only_correction(monkeypatch):
    url = "https://news.example/acme-seed"
    quote = "Acme Raises $40 Million Seed Funding At $200 Million Valuation"
    reasoning_requests = []
    provider_searches = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        reasoning_requests.append(payload)
        turn = len(reasoning_requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
        elif turn < investigator.MAX_REASONING_TURNS:
            name, arguments = "search_web", {"query": f"Acme source {turn}"}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    status=(
                        "UNPROVEN"
                        if turn == investigator.MAX_REASONING_TURNS
                        else "VERIFIED"
                    ),
                    observed_value="Seed",
                    evidence_url=url,
                    evidence_quote=quote,
                    reason="The exact quote names Acme and Seed Funding.",
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": quote}

    async def fake_search(_session, query, *, key):
        del key
        provider_searches.append(query)
        return {"results": [], "notice": "discovery_only_not_evidence"}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Seed",
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_quote"] == quote
    assert result["usage"] == {
        "reasoning_turns": investigator.MAX_REASONING_TURNS + 1,
        "search_calls": investigator.MAX_SEARCH_CALLS,
        "fetch_calls": 1,
    }
    assert len(provider_searches) == investigator.MAX_SEARCH_CALLS
    assert len(reasoning_requests) == investigator.MAX_REASONING_TURNS + 1
    correction_feedback = json.loads(
        reasoning_requests[-1]["messages"][-1]["content"]
    )
    assert correction_feedback["rejected_findings"] == [{
        "target": "stage",
        "reason": "UNPROVEN must have empty evidence_url and evidence_quote fields",
        "source_url": url,
        "source_context": quote,
    }]
    assert "single final submit-only correction" in correction_feedback["instruction"]
    assert "exact continuous company-bound span" in correction_feedback["instruction"]
    assert {
        tool["function"]["name"] for tool in reasoning_requests[-1]["tools"]
    } == {"submit_findings"}


def test_final_noncontiguous_quote_correction_stays_unproven(monkeypatch):
    url = "https://acme.example/acquisition"
    page = (
        "Acme completed its acquisition by Example Capital. "
        "Acme is now a privately held company."
    )
    joined_quote = (
        "Acme completed its acquisition by Example Capital. ... "
        "Acme is now a privately held company."
    )
    reasoning_requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        reasoning_requests.append(payload)
        turn = len(reasoning_requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    observed_value="Private Equity",
                    evidence_url=url,
                    evidence_quote=joined_quote,
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, requested_url):
        return {"ok": True, "url": requested_url, "text": page}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 2)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Private Equity",
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["stage"]["reason"] == (
        "submitted quote contains a prohibited ellipsis instead of one "
        "continuous fetched-page span"
    )
    assert result["claims"]["stage"]["evidence_url"] == ""
    assert result["_validated_stage_finding"] == {}
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 0,
        "fetch_calls": 1,
    }
    assert len(reasoning_requests) == 3
    assert "do not paraphrase, join passages, or insert ellipses" in json.loads(
        reasoning_requests[-1]["messages"][-1]["content"]
    )["instruction"]
    assert {
        tool["function"]["name"] for tool in reasoning_requests[-1]["tools"]
    } == {"submit_findings"}


def test_required_tool_turn_does_not_interpret_provider_prose(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        return 200, {
            "choices": [{"message": {
                "content": (
                    "Acme common stock is listed on NASDAQ under ticker ACME."
                )
            }}]
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    diagnostic = {}

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        diagnostic=diagnostic,
    ))

    assert requests[0]["tool_choice"] == "required"
    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == MALFORMED_RESPONSE_FAILURE_REASON


def test_harness_rejects_multiple_tool_calls_as_malformed(monkeypatch):
    async def fake_post_json(_session, _url, *, headers, payload):
        del headers, payload
        call = {
            "id": "call",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "Acme"}),
            },
        }
        return 200, {"choices": [{"message": {"tool_calls": [call, call]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    diagnostic = {}
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        diagnostic=diagnostic,
    ))

    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == MALFORMED_RESPONSE_FAILURE_REASON


def test_harness_closes_search_budget_and_admission_boundary(monkeypatch):
    reasoning_turns = []
    provider_searches = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        reasoning_turns.append(payload)
        turn = len(reasoning_turns)
        if turn < investigator.MAX_REASONING_TURNS:
            name, arguments = "search_web", {"query": f"Acme query {turn}"}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        provider_searches.append(query)
        return {"results": [], "notice": "discovery_only_not_evidence"}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
    ))
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["reasoning_turns"] == investigator.MAX_REASONING_TURNS
    assert result["usage"]["search_calls"] == investigator.MAX_SEARCH_CALLS
    assert len(provider_searches) == investigator.MAX_SEARCH_CALLS
    assert reasoning_turns[-1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }

    monotonic_values = iter((0.0, investigator.ADMISSION_DEADLINE_SECONDS + 1.0))
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    admission_result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
    ))
    assert admission_result["claims"]["stage"]["status"] == "UNPROVEN"
    assert admission_result["usage"]["reasoning_turns"] == 0


def test_embedded_reasoning_provider_error_is_typed_infrastructure(monkeypatch):
    async def fake_post_json(_session, _url, *, headers, payload):
        del headers, payload
        return 200, {"error": {"code": 429, "message": "rate limited"}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    diagnostic = {}

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        diagnostic=diagnostic,
    ))

    assert result["claims"] == {}
    assert result["failure_reason"] == PROVIDER_ERROR_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == PROVIDER_ERROR_FAILURE_REASON


def test_only_the_lab_scorer_activates_the_investigator_by_default():
    assert CompetitionCompanyScorer().evidence_investigator is False
    scorer = arena_scoring.lab_scorer(
        arena_scoring.build_scorer_policy(
            scoring_adapter_version="qualification_contacts_v3"
        )
    )
    assert scorer.evidence_investigator is True
    assert scorer.company_quality is False


@pytest.mark.parametrize(
    "investigation_failure",
    [None, PROVIDER_ERROR_FAILURE_REASON, MALFORMED_RESPONSE_FAILURE_REASON],
)
@pytest.mark.parametrize("current_output_contract", [False, True])
def test_targeted_stage_classification_through_lab_scorer(
    monkeypatch, investigation_failure, current_output_contract
):
    verdict = _complete_verdict(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://evidence.example/stage",
        stage_evidence_quote="Acme launched its public product.",
    )
    calls = {"broad": 0, "investigator": 0}

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match("homepage identity verified")

    async def broad_provider(**_kwargs):
        calls["broad"] += 1
        return verdict, ""

    async def bounded_investigation(*, diagnostic, **kwargs):
        calls["investigator"] += 1
        assert kwargs["targets"] == ("stage",)
        if investigation_failure:
            diagnostic[VERIFIER_FAILURE_REASON_KEY] = investigation_failure
            return {
                "claims": {},
                "failure_reason": investigation_failure,
            }
        return {
            "claims": {
                "stage": _finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value="",
                    evidence_url="",
                    evidence_quote="",
                    reason="No current listing proof was found.",
                )
            },
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer, "_request_company_reverify_json", broad_provider
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigation
    )
    policy = arena_scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
        intent_details=current_output_contract,
    )
    scorer = arena_scoring.lab_scorer(policy)
    company = (
        _competition_company_v5()
        if current_output_contract
        else _competition_company()
    )
    if current_output_contract:
        assert policy["intent_details_policy"] == "intent_details_v1"
        assert scorer.contacts_required is True
        assert "fit_summary" not in company
        assert "intent_details" in company
    icp = _icp(
        company_stage="Public",
        intent_signals=["Announced a completed funding event"],
    ).model_dump(mode="json")

    if investigation_failure:
        with pytest.raises(arena_scoring.ScoringError):
            arena_scoring.score_work_item(
                {"scored_run_id": "targeted-investigator-failure"},
                icp=icp,
                companies=[company],
                scorer=scorer,
                max_retries=3,
            )
        assert calls == {"broad": 3, "investigator": 3}
        return

    accepted = arena_scoring.score_work_item(
        {"scored_run_id": "targeted-unproven-stage"},
        icp=icp,
        companies=[company],
        scorer=scorer,
        max_retries=3,
    )
    receipt = accepted[0]["verifier_gate_receipts"][0]

    assert calls == {"broad": 1, "investigator": 1}
    assert accepted[0]["final_score"] == 0.0
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["failure_class"] == "insufficient_fit_evidence"
    assert receipt["company_fit_dimensions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )


def test_completed_same_domain_alias_is_terminal_zero_through_lab_scorer(
    monkeypatch,
):
    verdict = _complete_verdict(
        observed_company_name="DBS Bank Ltd",
        observed_company_website="https://www.dbs.com/",
        observed_company_linkedin="https://www.linkedin.com/company/dbs-bank",
        observed_industry="Financial Services",
        observed_subindustry="Banking",
        industry_evidence_url="https://www.dbs.com/about-us/default.page",
        industry_evidence_quote="DBS is a leading financial services group in Asia.",
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://www.dbs.com/investors/fixed-income/overview",
        stage_evidence_quote="DBS Group Holdings Ltd is listed in Singapore.",
    )
    calls = {"broad": 0, "investigator": 0, "good": 0}

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match(
            "homepage identity verified",
            details={
                "identity": {
                    "decision": COMPANY_FIT_MATCH,
                    "evidence_source": "company_homepage",
                    "observed_name": "dbs",
                    "observed_domain": "dbs.com",
                    "observed_linkedin_slug": "dbs-bank",
                },
                "verified_homepage_transport_domain": "dbs.com",
            },
        )

    async def broad_provider(**_kwargs):
        calls["broad"] += 1
        return dict(verdict), ""

    async def no_profile_fetch(*_args, **_kwargs):
        return None

    async def preserve_broad_employee_observation(value, *_args, **_kwargs):
        return value

    async def bounded_investigation(*, targets, **_kwargs):
        calls["investigator"] += 1
        assert targets == ("stage", "rebrand")
        stage_finding = _finding(
            "stage",
            status="VERIFIED",
            observed_value="Public",
            evidence_url="https://www.dbs.com/investors/fixed-income/overview",
            evidence_quote="DBS Group Holdings Ltd is listed in Singapore.",
            reason="The cited parent is publicly listed.",
        )
        return {
            "claims": {
                "stage": stage_finding,
                "rebrand": _finding(
                    "rebrand",
                    status="UNPROVEN",
                    observed_value="",
                    evidence_url="",
                    evidence_quote="",
                    reason=(
                        "The same-domain names could not be bound to one legal "
                        "issuer."
                    ),
                )
            },
            "failure_reason": "",
            "_validated_stage_finding": stage_finding,
        }

    original_score_company = lead_scorer.score_company_competition_intent

    async def score_company(**kwargs):
        if kwargs["company"].company_name == "DBS":
            return await original_score_company(**kwargs)
        calls["good"] += 1
        return lead_scorer.LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=77,
            time_decay_multiplier=1,
            intent_signal_final=77,
            cost_penalty=0,
            time_penalty=0,
            final_score=77,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer, "_request_company_reverify_json", broad_provider
    )
    monkeypatch.setattr(
        lead_scorer,
        "_fetch_structured_linkedin_profile_once",
        no_profile_fetch,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        preserve_broad_employee_observation,
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigation
    )
    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent", score_company
    )

    scorer = arena_scoring.lab_scorer(
        arena_scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2"
        )
    )
    dbs = {
        **_competition_company(),
        "company_name": "DBS",
        "company_website": "https://www.dbs.com/",
        "company_linkedin": "",
        "industry": "Financial Services",
        "company_stage": "Public",
    }
    accepted = arena_scoring.score_work_item(
        {"scored_run_id": "same-domain-unproven-alias"},
        icp=_icp(
            industry="Financial Services",
            sub_industry="Banking",
            product_service="banking",
            company_stage="Public",
            intent_signals=["Announced a completed funding event"],
        ).model_dump(mode="json"),
        companies=[dbs, _competition_company()],
        scorer=scorer,
        max_retries=3,
    )

    assert calls == {"broad": 1, "investigator": 1, "good": 1}
    assert [row["final_score"] for row in accepted] == [0.0, 77.0]
    dbs_result = accepted[0]
    receipt = dbs_result["verifier_gate_receipts"][0]
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["failure_class"] == "insufficient_fit_evidence"
    assert receipt["company_fit_dimensions"]["identity"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert receipt["company_fit_dimensions"]["stage"] == COMPANY_FIT_MATCH
    assert receipt["dimension_evidence"]["identity"][
        "web_identity_receipt"
    ]["reason_code"] == "rebrand_continuity_unproven"
    assert dbs_result["company_qualified"] is False


@pytest.mark.parametrize("reject_submission", [False, True])
def test_extra_turn_is_only_a_rejected_submission_correction(monkeypatch, reject_submission):
    calls = []
    searches = []
    async def fake_post(_session, _url, *, headers, payload):
        calls.append(payload)
        if reject_submission and len(calls) == 1:
            name, args = "submit_findings", {"findings": [_finding("stage")]}
        else:
            name, args = "search_web", {"query": "Acme stage"}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(calls)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}
    async def fake_search(*args, **kwargs):
        searches.append(1)
        return {"results": []}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 1)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
    ))
    assert len(calls) == (2 if reject_submission else 1)
    assert len(searches) == (0 if reject_submission else 1)
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert result["claims"] == {}


def test_final_invalid_target_does_not_erase_independent_valid_stage(monkeypatch):
    quote = "Acme common stock is listed on NASDAQ under ticker ACME."
    calls = []
    async def fake_post(_session, _url, *, headers, payload):
        calls.append(payload)
        if len(calls) == 1:
            name, args = "fetch_page", {"url": "https://acme.example/investors"}
        else:
            name, args = "submit_findings", {"findings": [
                _finding("stage"),
                _finding("industry", evidence_quote="Acme invented a claim absent from the page."),
            ]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(calls)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}
    async def fake_fetch(_session, url):
        return {"ok": True, "url": url, "text": quote}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 2)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage", "industry"), requested_stage="Public",
    ))
    assert len(calls) == 3
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["_validated_stage_finding"] == result["claims"]["stage"]
