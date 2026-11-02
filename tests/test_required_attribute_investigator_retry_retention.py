"""Keep an exact, checked attribute page available to a later company retry."""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from lab_arena import scoring as arena_scoring
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.competition import CompetitionCompanyScorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_company_evidence_investigator import _competition_company
from tests.test_matched_company_retry_source_continuity import (
    _company as _platform_company, _icp as _platform_icp,
)


SOURCE_URL = (
    "https://origisenergy.com/insights/"
    "origis-energy-and-meta-partner-on-landmark-texas-project/"
)
SOURCE_QUOTE = (
    "Origis Energy, one of America’s leading renewable energy and "
    "decarbonization platforms, and Meta have signed a long-term power "
    "purchase agreement for the Greyhound A Solar project in West Odessa, Texas."
)
SOURCE_TEXT = "Origis Energy announcement. " + SOURCE_QUOTE + " Additional company news."
CAPTURED_PAGE = Path(__file__).with_name("fixtures").joinpath(
    "origis_meta_ppa_20261028.txt"
)


def _verdict(quote=SOURCE_QUOTE):
    return {
        "attribute_satisfied": True,
        "required_attribute_evidence_url": SOURCE_URL,
        "required_attribute_evidence_quote": quote,
    }


def _investigator_cache(quote=SOURCE_QUOTE):
    return lead_scorer._reviewed_required_attribute_source_cache(
        {lead_scorer.PRIVATE_FETCHED_PAGES_KEY: {
            SOURCE_URL: {"final_url": SOURCE_URL, "text": SOURCE_TEXT},
        }},
        {"evidence_url": SOURCE_URL, "evidence_quote": quote},
        {},
    )


def test_grounded_investigator_attribute_page_survives_next_retry():
    """Origis's first verified PPA page must not vanish after another gate retries."""

    current = _investigator_cache()
    assert list(current) == [SOURCE_URL]
    retained = {}
    no_network = AsyncMock(side_effect=AssertionError("a loaded page must not refetch"))
    with patch.object(lead_scorer, "_fetch_bounded_html", no_network):
        first, _ = asyncio.run(lead_scorer._ground_required_attribute_evidence(
            _verdict(), active_attribute=True, source_cache=current,
            successful_source_sink=retained,
        ))
        assert list(retained) == [SOURCE_URL]
        second, _ = asyncio.run(lead_scorer._ground_required_attribute_evidence(
            _verdict(), active_attribute=True, source_cache={},
            successful_source_sink=retained,
        ))

    assert first[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
    assert first[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["cache_hit"] is True
    assert list(retained) == [SOURCE_URL]
    assert retained[SOURCE_URL]["text"] == SOURCE_TEXT
    assert second[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
    assert second[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["cache_hit"] is True
    assert no_network.await_count == 0


def test_untrusted_quote_cannot_enter_retry_cache():
    retained = {}
    wrong_quote = "Origis Energy signed no power purchase agreement."
    # The investigator's own exact-quote gate excludes an invented citation.
    assert _investigator_cache(wrong_quote) == {}
    current = {
        SOURCE_URL: {"status": "fetched", "final_url": SOURCE_URL, "text": SOURCE_TEXT},
    }
    no_network = AsyncMock(side_effect=AssertionError("cached page must not refetch"))
    with patch.object(lead_scorer, "_fetch_bounded_html", no_network):
        rejected, _ = asyncio.run(lead_scorer._ground_required_attribute_evidence(
            _verdict(wrong_quote), active_attribute=True, source_cache=current,
            successful_source_sink=retained,
        ))
    assert rejected[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "quote_absent"
    assert rejected["attribute_satisfied"] is None
    assert retained == {}
    assert no_network.await_count == 0


def test_missing_source_stays_unavailable_and_unretained():
    retained = {}
    no_network = AsyncMock(return_value=(404, SOURCE_URL, ""))
    with patch.object(lead_scorer, "_fetch_bounded_html", no_network):
        rejected, _ = asyncio.run(lead_scorer._ground_required_attribute_evidence(
            _verdict(), active_attribute=True, source_cache={},
            successful_source_sink=retained,
        ))
    assert rejected[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "source_unavailable"
    assert rejected["attribute_satisfied"] is None
    assert retained == {}
    assert no_network.await_count == 1


def test_newly_fetched_body_overrides_conflicting_retained_quote():
    """A source changed during retry cannot ground an old quote from the sink."""

    retained = {}
    lead_scorer._retain_successful_required_attribute_source(
        retained, SOURCE_URL,
        {"status": "fetched", "final_url": SOURCE_URL, "text": SOURCE_TEXT},
    )
    current_text = "Origis Energy updated the page without the cited agreement."
    current = {
        SOURCE_URL: {
            "status": "fetched", "final_url": SOURCE_URL, "text": current_text,
        },
    }
    no_network = AsyncMock(side_effect=AssertionError("current body must not refetch"))
    with patch.object(lead_scorer, "_fetch_bounded_html", no_network):
        projected, _ = asyncio.run(lead_scorer._ground_required_attribute_evidence(
            _verdict(), active_attribute=True, source_cache=current,
            successful_source_sink=retained,
        ))

    assert projected[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "quote_absent"
    assert projected["attribute_satisfied"] is None
    assert projected["required_attribute_evidence_url"] == ""
    assert projected["required_attribute_evidence_quote"] == ""
    assert current[SOURCE_URL]["text"] == current_text
    assert retained[SOURCE_URL]["text"] == SOURCE_TEXT
    assert no_network.await_count == 0


def test_required_attribute_retry_cache_scope_isolates_company_icp_and_date(
    monkeypatch,
):
    """The attribute sink cannot cross a changed scoring input or date."""

    observed = []

    async def score_company(**kwargs):
        observed.append(kwargs["required_attribute_retry_source_cache"])
        return {"final_score": 0.0, "failure_reason": "test-only"}

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    scorer = CompetitionCompanyScorer(
        integrity_policy=True, company_quality=True, evidence_investigator=True,
        scorer_policy={"scoring_adapter_version": "qualification_integrity_v2"},
    )
    company = {
        "company_name": "Unibuddy",
        "company_website": "https://unibuddy.com/",
        "company_linkedin": "https://www.linkedin.com/company/unibuddy",
        "industry": "Education",
        "employee_count": "201-500",
        "company_stage": "",
        "country": "United Kingdom",
        "state": "",
        "fit_summary": "Unibuddy supplies higher-education software.",
        "fit_evidence_urls": ["https://unibuddy.com/"],
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "Unibuddy announced a new enrollment workflow.",
            "date": "2026-09-01",
            "why_now": "The workflow is a current product signal.",
            "url": "https://unibuddy.com/news/enrollment-workflow",
            "snippet": "The workflow supports university enrollment teams.",
        }],
    }
    icp = _platform_icp().model_dump(mode="json")
    changed_company = {**company, "company_name": "Different Company"}
    changed_icp = {**icp, "required_attribute": "Different criterion"}
    scope = {}

    with use_evaluation_date("2026-10-28"):
        for submitted_company, submitted_icp in (
            (company, icp), (company, icp), (changed_company, icp),
            (company, changed_icp),
        ):
            asyncio.run(scorer.score_with_breakdowns(
                [submitted_company], submitted_icp, False,
                retry_evidence_scope=scope,
            ))
    with use_evaluation_date("2026-10-29"):
        asyncio.run(scorer.score_with_breakdowns(
            [company], icp, False, retry_evidence_scope=scope,
        ))

    assert len(observed) == 5
    assert all(isinstance(cache, dict) for cache in observed)
    assert observed[0] is observed[1]
    assert all(cache is not observed[0] for cache in observed[2:])
    assert len({id(cache) for cache in observed}) == 4


def test_captured_origis_fit_and_page_survive_intent_retry(monkeypatch):
    """Intent retries retain the checked fit and its exact captured source."""

    page = CAPTURED_PAGE.read_text()
    assert hashlib.sha256(page.encode()).hexdigest() == (
        "64ed843cb7cecef4f5be50614a6edd4c77e8b4ec231db055c6b053a57d6ccdf9"
    )
    assert SOURCE_QUOTE in page
    icp = _platform_icp().model_copy(update={
        "icp_id": "origis-ppa-retry",
        "industry": "Energy",
        "sub_industry": "Renewable energy",
        "product_service": "Utility-scale solar projects",
        "geography": "United States",
        "required_attribute": (
            "Develops or operates utility-scale renewable generation or energy "
            "storage assets in the U.S. that sell power under long-term offtake agreements."
        ),
    })
    submitted = _competition_company()
    submitted.update(
        company_name="Origis Energy",
        company_website="https://origisenergy.com/",
        company_linkedin="https://www.linkedin.com/company/origis-energy",
        industry="Energy",
        employee_count="201-500",
        country="United States",
        state="Florida",
        fit_evidence_urls=[],
    )
    identity = {
        "normalized_name": "Origis Energy",
        "registrable_dns_domain": "origisenergy.com",
        "linkedin_company_slug": "origis-energy",
    }
    investigations = []
    intent_calls = []
    retained_attribute_sources = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("independent prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_match("independent homepage identity verified", details={
            "identity": {
                "decision": "match", "evidence_source": "company_homepage",
                "observed_name": "Origis Energy",
                "observed_domain": "origisenergy.com",
                "observed_linkedin_slug": "origis-energy",
            },
            "verified_homepage_transport_domain": "origisenergy.com",
        })

    async def investigate(**kwargs):
        investigations.append(kwargs)
        return {
            "claims": {"required_attribute": {
                "target": "required_attribute", "status": "VERIFIED",
                "evidence_url": SOURCE_URL, "evidence_quote": SOURCE_QUOTE,
                "reason": "The captured announcement states the long-term PPA.",
            }},
            "_completed_submit": True, "failure_reason": "",
            lead_scorer.PRIVATE_FETCHED_PAGES_KEY: {
                SOURCE_URL: {"final_url": SOURCE_URL, "text": page},
            },
        }

    async def web_boundary(company, actual_icp, **kwargs):
        retained_attribute_sources.append(kwargs["required_attribute_retry_source_cache"])
        projected, _result, claims, *_ = await (
            lead_scorer._run_targeted_company_evidence_investigation(
                company=company, icp=actual_icp,
                verdict={
                    "observed_company_name": "Origis Energy",
                    "observed_company_website": "https://origisenergy.com/",
                    "observed_company_linkedin": (
                        "https://www.linkedin.com/company/origis-energy"
                    ),
                },
                investigation_targets=("required_attribute",),
                icp_attribute=actual_icp.required_attribute, icp_stage="",
                verified_identity=identity,
                verified_transport_domain="origisenergy.com",
                structured_employee_size_evidence=None,
                structured_public_company_evidence=None,
                employee_size_conflict=False,
                company_quality=kwargs["company_quality"],
                required_attribute_source_cache={},
                successful_required_attribute_source_sink=(
                    kwargs["required_attribute_retry_source_cache"]
                ),
                preserve_matched_industry=True,
            )
        )
        assert claims["required_attribute"]["status"] == "VERIFIED"
        assert projected[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
        # The other fit gates are fixture stubs. Only the PPA attribute is
        # asserted against the captured Origis page in this test.
        fixture_evidence = {
            "url": "https://fixture.example/other-gates",
            "quote": "Fixture proof for unrelated fit gates.",
        }
        return company_fit_match("other fit gates stubbed", details={
            "identity_decision": "match",
            "identity_receipt": {
                "decision": "match",
                "evidence_source": "company_web_reverification",
                "submitted_name": "Origis Energy",
                "submitted_domain": "origisenergy.com",
                "submitted_linkedin_slug": "origis-energy",
                "observed_name": "Origis Energy",
                "observed_domain": "origisenergy.com",
                "observed_linkedin_slug": "origis-energy",
            },
            "dimension_decisions": {
                "identity": "match", "employee_size": "match",
                "industry": "match", "geography": "match", "stage": "match",
            },
            "dimension_evidence": {
                "employee_size": fixture_evidence,
                "industry": fixture_evidence,
                "geography": fixture_evidence,
                "required_attribute": {"url": SOURCE_URL, "quote": SOURCE_QUOTE},
            },
            "required_attribute_decision": "match",
            "required_attribute_grounding": projected[
                lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING
            ],
        })

    async def intent_boundary(*_args, **_kwargs):
        assert retained_attribute_sources[0][SOURCE_URL]["text"] == page
        intent_calls.append(True)
        raise TimeoutError("fixture intent provider timeout")

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web_boundary)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", intent_boundary)
    scorer = arena_scoring.lab_scorer(arena_scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
    ))
    with pytest.raises(arena_scoring.ScoringError):
        arena_scoring.score_work_item(
            {"scored_run_id": "origis-source-retention-retry"},
            icp=icp.model_dump(mode="json"), companies=[submitted],
            scorer=scorer, max_retries=2,
        )
    assert len(investigations) == len(retained_attribute_sources) == 1
    assert len(intent_calls) == 2
    assert SOURCE_URL not in investigations[0]["prefetched_pages"]
    assert retained_attribute_sources[0][SOURCE_URL]["final_url"] == SOURCE_URL


@pytest.mark.parametrize("current_fetched", [False, True])
def test_retained_origis_page_does_not_force_a_later_verified_finding(
    monkeypatch, current_fetched,
):
    page = CAPTURED_PAGE.read_text()
    retained = {}
    lead_scorer._retain_successful_required_attribute_source(
        retained, SOURCE_URL,
        {"status": "fetched", "final_url": SOURCE_URL, "text": page},
    )
    assert list(retained) == [SOURCE_URL]
    observed_prefetch = []

    async def investigate(**kwargs):
        observed_prefetch.append(kwargs["prefetched_pages"])
        return {
            "claims": {"required_attribute": {
                "target": "required_attribute", "status": "UNPROVEN",
                "evidence_url": "", "evidence_quote": "",
                "reason": "The new review did not verify the complete criterion.",
            }},
            "_completed_submit": True, "failure_reason": "",
        }

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    fresh_text = "Origis Energy updated announcement without the cited PPA span."
    current_source = (
        {"status": "fetched", "final_url": SOURCE_URL, "text": fresh_text}
        if current_fetched else {"status": "source_unavailable"}
    )
    company = _platform_company().model_copy(update={
        "company_name": "Origis Energy",
        "company_website": "https://origisenergy.com/",
        "company_linkedin": "https://www.linkedin.com/company/origis-energy",
    })
    icp = _platform_icp().model_copy(update={
        "required_attribute": (
            "Develops or operates utility-scale renewable generation or energy "
            "storage assets in the U.S. that sell power under long-term offtake agreements."
        ),
    })
    projected, _result, *_ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=icp,
            verdict={
                "observed_company_name": "Origis Energy",
                "observed_company_website": "https://origisenergy.com/",
                "observed_company_linkedin": (
                    "https://www.linkedin.com/company/origis-energy"
                ),
            },
            investigation_targets=("required_attribute",),
            icp_attribute=icp.required_attribute, icp_stage="",
            verified_identity={
                "normalized_name": "Origis Energy",
                "registrable_dns_domain": "origisenergy.com",
                "linkedin_company_slug": "origis-energy",
            },
            verified_transport_domain="origisenergy.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            required_attribute_source_cache={SOURCE_URL: current_source},
            successful_required_attribute_source_sink=retained,
        )
    )
    assert observed_prefetch == [{
        SOURCE_URL: {
            "final_url": SOURCE_URL,
            "text": fresh_text if current_fetched else page,
        },
    }]
    assert projected["attribute_satisfied"] is None
    assert projected[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "unproven"


def test_multitarget_prefetch_keeps_homepage_and_stage_priority():
    homepage = "https://origisenergy.com/"
    stage_a = "https://origisenergy.com/investors/series-a/"
    stage_b = "https://origisenergy.com/investors/series-b/"
    second_attribute = "https://origisenergy.com/insights/another-ppa/"
    retained = {}
    for url in (SOURCE_URL, second_attribute):
        lead_scorer._retain_successful_required_attribute_source(
            retained, url,
            {"status": "fetched", "final_url": url,
             "text": "Origis Energy power agreement source body."},
        )
    home_page = {"final_url": homepage, "text": "Origis Energy homepage body."}
    stage_pages = {
        url: {"final_url": url, "text": "Origis Energy financing stage body."}
        for url in (stage_a, stage_b)
    }
    prefetch = lead_scorer._investigator_prefetched_pages(
        retained, [homepage, stage_a, stage_b, SOURCE_URL, second_attribute],
        verified_homepage_pages={homepage: home_page},
        matched_company_retry_pages=stage_pages,
    )
    assert list(prefetch) == [homepage, stage_a, stage_b]
    assert len(prefetch) == lead_scorer.MAX_PREFETCHED_PAGES


def _prefetch_for_non_attribute_review(targets, *, preserve_matched_industry):
    retained = {}
    lead_scorer._retain_successful_required_attribute_source(
        retained, SOURCE_URL,
        {"status": "fetched", "final_url": SOURCE_URL, "text": SOURCE_TEXT},
    )
    company = _platform_company().model_copy(update={
        "company_name": "Origis Energy",
        "company_website": "https://origisenergy.com/",
        "company_linkedin": "https://www.linkedin.com/company/origis-energy",
        "fit_evidence_urls": [SOURCE_URL],
    })
    icp = _platform_icp()
    observed = []

    async def investigate(**kwargs):
        observed.append(kwargs)
        return {"claims": {}, "failure_reason": "offline probe"}

    with patch.object(lead_scorer, "investigate_company_evidence", investigate):
        asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=icp,
            verdict={
                "observed_company_name": "Origis Energy",
                "observed_company_website": "https://origisenergy.com/",
                "observed_company_linkedin": (
                    "https://www.linkedin.com/company/origis-energy"
                ),
            },
            investigation_targets=targets,
            icp_attribute=icp.required_attribute, icp_stage="",
            verified_identity={
                "normalized_name": "Origis Energy",
                "registrable_dns_domain": "origisenergy.com",
                "linkedin_company_slug": "origis-energy",
            },
            verified_transport_domain="origisenergy.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            required_attribute_source_cache={
                SOURCE_URL: {"status": "source_unavailable"},
            },
            successful_required_attribute_source_sink=retained,
            preserve_matched_industry=preserve_matched_industry,
        ))
    assert len(observed) == 1
    assert SOURCE_URL in observed[0]["prior_observations"]["submitted_source_urls"]
    return observed[0]["prefetched_pages"]


@pytest.mark.parametrize("targets", [("stage",), ("industry",)])
def test_unrelated_review_does_not_prefetch_retained_attribute_body(targets):
    pages = _prefetch_for_non_attribute_review(
        targets, preserve_matched_industry=False,
    )
    assert SOURCE_URL not in pages


def test_attribute_source_recovery_keeps_retained_body_for_industry_target():
    pages = _prefetch_for_non_attribute_review(
        ("industry",), preserve_matched_industry=True,
    )
    assert pages[SOURCE_URL] == {"final_url": SOURCE_URL, "text": SOURCE_TEXT}
