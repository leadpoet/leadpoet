import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_verification, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
)


def _company(**updates):
    values = {
        "company_name": "Example Company",
        "company_website": "https://example.com/",
        "company_linkedin": "",
        "industry": "Software",
        "sub_industry": "",
        "employee_count": "201-500",
        "company_stage": "Series B",
        "country": "United States",
        "state": "",
        "description": "",
        "intent_details": None,
        "fit_evidence_urls": [],
        "company_stage_evidence": [],
        "intent_signals": [],
        "required_attribute": None,
    }
    values.update(updates)
    return CompanyOutput.model_construct(**values)


def _icp(**updates):
    values = {
        "employee_count": ["201-500"],
        "industry": "Software",
        "country": "United States",
        "geography": "United States",
        "company_stage": "Series B",
        "required_attribute": "",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_operating_homepage_navigation_label_is_not_a_parked_domain():
    operating_page = """
        <html>
          <title>Example Company | Industrial systems</title>
          <a href="https://www.linkedin.com/company/example-company">LinkedIn</a>
          <nav><a href="/projects">Under construction</a></nav>
          <main>Example Company builds industrial systems for customers.</main>
        </html>
    """

    with patch.object(
        company_verification,
        "_fetch_bounded_html",
        AsyncMock(return_value=(200, "https://example.com/", operating_page)),
    ):
        result = asyncio.run(company_verification.verify_company_exists(
            "Example Company",
            "https://example.com/",
            company_linkedin=(
                "https://www.linkedin.com/company/example-company"
            ),
            require_https_transport=True,
        ))

    assert result.decision == COMPANY_FIT_MATCH


def test_real_parked_domain_marker_remains_a_mismatch():
    with patch.object(
        company_verification,
        "_fetch_bounded_html",
        AsyncMock(return_value=(
            200,
            "https://example.com/",
            "<html><body>This domain is for sale</body></html>",
        )),
    ):
        result = asyncio.run(company_verification.verify_company_exists(
            "Example Company",
            "https://example.com/",
            require_https_transport=True,
        ))

    assert result.decision == COMPANY_FIT_MISMATCH


def test_verified_homepage_linkedin_completes_common_legal_name_observation():
    company = _company(
        company_name="Crusoe",
        company_website="https://crusoe.ai/",
    )
    verdict = {
        "observed_company_name": "Crusoe Technologies Inc.",
        "observed_company_website": "https://www.crusoe.ai/",
        "observed_company_linkedin": "",
    }
    anchor = {
        "normalized_name": "crusoe",
        "registrable_dns_domain": "crusoe.ai",
        "linkedin_company_slug": "crusoe",
    }

    receipt = lead_scorer._web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="crusoe.ai",
    )

    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["observed_linkedin_slug"] == "crusoe"
    assert receipt["linkedin_evidence_source"] == "company_homepage"
    assert receipt["web_observed_linkedin_slug"] == ""

    strict_receipt = lead_scorer._web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="crusoe.ai",
        company_quality=True,
    )
    assert strict_receipt["decision"] == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize(
    ("observed", "quote", "supported"),
    [
        (
            "acquired",
            "Vibe.co is a Walmart company, acquired by Walmart in August 2026.",
            True,
        ),
        (
            "acquired",
            "Vibe.co announced that it will be acquired by Walmart, subject to closing.",
            False,
        ),
        (
            "acquired",
            "Walmart invested in Vibe.co during its Series B round.",
            False,
        ),
        (
            "public",
            "Acme is a subsidiary of NYSE-listed Parent Corp.",
            False,
        ),
        (
            "acquired",
            "Acme is now a wholly owned subsidiary of Parent Corp.",
            True,
        ),
        (
            "series c+",
            "AliveCor announced that it has closed a Series F financing round.",
            True,
        ),
        (
            "series c+",
            "Forus announced that it has closed its Series C financing.",
            True,
        ),
        (
            "series c+",
            "|5. Later Stage VC (Series C)|14-Apr-2016|Completed|",
            False,
        ),
        (
            "series c+",
            "Privately Held. Backed by leading investors.",
            False,
        ),
    ],
)
def test_stage_evidence_keeps_current_ownership_and_completed_round_distinctions(
    observed, quote, supported
):
    assert lead_scorer._stage_quote_supports_observation(
        observed, quote
    ) is supported


@pytest.mark.parametrize(
    "requested",
    ["seed", "series a", "series b", "series c+", "private equity", "public"],
)
def test_acquired_observation_never_auto_matches_an_allowed_icp_stage(requested):
    assert lead_scorer._company_stage_matches("acquired", requested) is False


@pytest.mark.parametrize(
    ("quote", "supported"),
    [
        ("Walmart completed the acquisition of Vibe.", True),
        ("Walmart completed the acquisition of Another Company.", False),
        ("Walmart announced that it will acquire Vibe.", False),
        ("Walmart completed an investment in Vibe.", False),
        ("Walmart completed the acquisition of a minority stake in Vibe.", False),
        ("Walmart completed the acquisition of Vibe's minority shares.", False),
        ("Walmart has not completed the acquisition of Vibe.", False),
        ("Walmart will have completed the acquisition of Vibe.", False),
        (
            "Walmart completed the acquisition of Vibe subject to final approval.",
            False,
        ),
        ("Vibe was formerly a subsidiary of Walmart.", False),
        ("Vibe was acquired by Walmart but is now independent.", False),
        (
            "Another Company was acquired by Walmart. "
            "Vibe has not been acquired by Walmart.",
            False,
        ),
        (
            "Walmart completed the acquisition of Vibe. Vibe later went public.",
            False,
        ),
        (
            "Walmart completed the acquisition of Vibe. Vibe is now listed on NYSE.",
            False,
        ),
    ],
)
def test_completed_acquisition_proof_is_bound_to_the_exact_subject(
    quote, supported
):
    assert lead_scorer._acquired_stage_quote_supports_company(
        _company(company_name="Vibe", company_website="https://vibe.co/"),
        "Vibe.co",
        quote,
    ) is supported


def test_exact_completed_acquisition_stage_contradicts_an_older_venture_round():
    company = _company(
        company_name="Vibe",
        company_website="https://vibe.co/",
        company_stage="Series B",
        industry="Advertising",
    )
    icp = _icp(
        industry="Advertising",
        company_stage="Series B",
        required_attribute=(
            "Sells advertising technology and has a recent public signal of "
            "product launch, market expansion, or partnership."
        ),
    )
    verdict = {
        "observed_company_name": "Vibe",
        "observed_company_website": "https://vibe.co/",
        "observed_company_linkedin": "https://linkedin.com/company/vibedotco",
        "observed_employee_count": "201-500",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://linkedin.com/company/vibedotco",
        "employee_size_evidence_quote": "Company size: 201-500 employees.",
        "observed_industry": "Advertising",
        "observed_subindustry": "Connected TV advertising platform",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://vibe.co/",
        "industry_evidence_quote": "Vibe operates a connected TV ad platform.",
        "observed_hq_country": "United States",
        "observed_hq_state": "New York",
        "geography_matches": True,
        "geography_evidence_url": "https://vibe.co/about",
        "geography_evidence_quote": "Vibe is headquartered in New York.",
        "observed_company_stage": "Acquired",
        "stage_matches": False,
        "stage_evidence_url": "https://corporate.example/acquisition-complete",
        "stage_evidence_quote": "Walmart completed the acquisition of Vibe.",
        "attribute_satisfied": True,
        "required_attribute_evidence_url": "https://vibe.co/company",
        "required_attribute_evidence_quote": (
            "Vibe is a Walmart company, acquired by Walmart in August 2026."
        ),
    }

    result = lead_scorer._reverify_decision(
        verdict,
        icp.required_attribute,
        "series b",
        icp=icp,
        company=company,
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MISMATCH
    assert result.details["dimension_evidence"]["stage"] == {
        "url": "https://corporate.example/acquisition-complete",
        "quote": "Walmart completed the acquisition of Vibe.",
    }


@pytest.mark.parametrize(
    ("url", "quote", "conflicts"),
    [
        (
            "https://vibe.co/llms.txt",
            "Vibe.co is a Walmart company, acquired by Walmart in August 2026.",
            True,
        ),
        (
            "https://vibe.co/llms.txt",
            "Another Company was acquired by Walmart in August 2026.",
            False,
        ),
        (
            "https://news.example/acquisition",
            "Vibe.co was acquired by Walmart in August 2026.",
            False,
        ),
        (
            "https://vibe.co/llms.txt",
            "Walmart made a strategic minority investment in Vibe.co.",
            False,
        ),
    ],
)
def test_first_party_acquisition_only_invalidates_a_conflicting_venture_stage(
    url, quote, conflicts
):
    verdict = {
        "observed_company_name": "Vibe.co",
        "required_attribute_evidence_url": url,
        "required_attribute_evidence_quote": quote,
    }

    assert lead_scorer._first_party_acquisition_conflicts_with_venture_stage(
        _company(company_name="Vibe", company_website="https://vibe.co/"),
        verdict,
        "series b",
    ) is conflicts


def test_stale_venture_stage_with_bound_current_owner_proof_requires_repair():
    verdict = {
        "observed_company_name": "Vibe.co",
        "observed_company_stage": "Series B",
        "stage_matches": True,
        "stage_evidence_url": "https://news.example/series-b",
        "stage_evidence_quote": (
            "Vibe.co announced that it has closed $50 million in Series B financing."
        ),
        "required_attribute_evidence_url": "https://vibe.co/llms.txt",
        "required_attribute_evidence_quote": (
            "Vibe.co is a Walmart company, acquired by Walmart in August 2026."
        ),
    }

    decision = lead_scorer._decision_from_observed_stage(
        verdict,
        "series b",
        company=_company(company_name="Vibe", company_website="https://vibe.co/"),
    )

    assert decision == COMPANY_FIT_UNAVAILABLE


def test_investigator_receipt_cannot_bypass_acquired_subject_binding():
    quote = "Another Company was acquired by Walmart."
    verdict = {
        "observed_company_name": "Vibe.co",
        "observed_company_stage": "Acquired",
        "stage_matches": False,
        "stage_evidence_url": "https://vibe.co/company",
        "stage_evidence_quote": quote,
    }
    finding = {
        "target": "stage",
        "status": "CONTRADICTED",
        "observed_value": "Acquired",
        "evidence_url": "https://vibe.co/company",
        "evidence_quote": quote,
    }

    decision = lead_scorer._decision_from_observed_stage(
        verdict,
        "series b",
        validated_stage_finding=finding,
        company=_company(company_name="Vibe", company_website="https://vibe.co/"),
    )

    assert decision == COMPANY_FIT_UNAVAILABLE


def test_company_research_prompt_selects_full_activity_and_current_stage(
    monkeypatch,
):
    prompts = []

    async def provider(**kwargs):
        prompts.append(kwargs["prompt"])
        return {
            "observed_company_name": "Example Company",
            "observed_company_website": "https://example.com/",
            "observed_company_linkedin": (
                "https://linkedin.com/company/example-company"
            ),
            "observed_employee_count": "201-500",
            "employee_size_matches": True,
            "employee_size_evidence_url": (
                "https://linkedin.com/company/example-company"
            ),
            "employee_size_evidence_quote": "Company size: 201-500.",
            "observed_industry": "AI infrastructure",
            "observed_subindustry": "Electrical equipment manufacturing",
            "industry_matches": True,
            "industry_activity_role": "supplier_operator",
            "industry_evidence_url": "https://example.com/manufacturing",
            "industry_evidence_quote": (
                "We manufacture switchgear and industrial controls."
            ),
            "observed_hq_country": "United States",
            "observed_hq_state": "New York",
            "geography_matches": True,
            "geography_evidence_url": "https://example.com/about",
            "geography_evidence_quote": (
                "Example Company is headquartered in New York."
            ),
            "observed_company_stage": "Acquired",
            "stage_matches": False,
            "stage_evidence_url": "https://parent.example/acquisition",
            "stage_evidence_quote": (
                "Parent completed the acquisition of Example Company."
            ),
            "attribute_satisfied": True,
            "required_attribute_evidence_url": (
                "https://example.com/manufacturing"
            ),
            "required_attribute_evidence_quote": (
                "Example Company opened a new switchgear factory."
            ),
        }, ""

    async def keep_observation(verdict, *_args, **_kwargs):
        return verdict

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        lead_scorer, "_request_company_reverify_json", provider
    )
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(industry="Hardware"),
        _icp(
            industry="Hardware",
            required_attribute=(
                "Designs physical hardware and has a recent facility opening."
            ),
        ),
        require_company_fit_dimensions=True,
    ))

    assert result.decision == COMPANY_FIT_MISMATCH
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "broad positioning label as exclusive" in prompt
    assert "Prefer that direct full-body activity quote" in prompt
    assert "Score this attribute independently from employee size" in prompt
    assert "When one page discusses multiple companies" in prompt
    assert "Do not stop when you find a venture round" in prompt
    assert "Check chronology before selecting evidence" in prompt
    assert "completed acquisition/current parent" in prompt
    assert "first-party completed-round announcement and its full body" in prompt
    assert "Public, or Acquired" in prompt
