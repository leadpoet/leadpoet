from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


PROFILE_URL = "https://www.linkedin.com/company/joinmultiverse/"
CANONICAL_PROFILE_URL = "https://www.linkedin.com/company/joinmultiverse"
SERIES_D_QUOTE = (
    "In 2022, we raised our $220m Series D funding - one of the largest "
    "venture rounds in EdTech history."
)
DESCRIPTION = (
    "Multiverse is the upskilling platform for AI and tech adoption. We’ve "
    "built a new model for transforming today’s workforce by combining expert "
    "human coaching with AI.\n\nOur learning drives real business impact and "
    "helps organisations close critical skills gaps - especially in data, AI "
    "and all things tech. We’ve already partnered with over 1,500 global "
    "organisations to transform their workforce through tech skills. Trusted "
    "by Microsoft, Mars, John Lewis Partnership and many more to upskill their "
    "team and unlock people’s potential and output.\n\n"
    f"{SERIES_D_QUOTE} We’re proud to be backed by some of the world’s biggest "
    "investors, including General Catalyst, Lightspeed Venture Partners and "
    "StepStone Group."
)
STAGE_EVIDENCE_QUOTE = DESCRIPTION.split(" We’re proud", 1)[0]


def _description_evidence(**updates):
    evidence = {
        "name": "Multiverse",
        "text": DESCRIPTION,
        "provider": "harvestapi_get_company",
        "source_field": "description",
        "url": CANONICAL_PROFILE_URL,
        "website": "https://multiverse.io/",
    }
    evidence.update(updates)
    return evidence


def _identity(**updates):
    identity = {
        "normalized_name": "Multiverse",
        "registrable_dns_domain": "multiverse.io",
        "linkedin_company_slug": "joinmultiverse",
    }
    identity.update(updates)
    return identity


def _company() -> CompanyOutput:
    return CompanyOutput(
        company_name="Multiverse",
        company_website="https://www.multiverse.io/",
        company_linkedin=PROFILE_URL,
        industry="Education",
        sub_industry="Workforce learning software",
        employee_count="501-1,000",
        company_stage="Series C+",
        country="United Kingdom",
        intent_signals=[{
            "description": "Multiverse launched a learning product.",
            "source": "company website",
            "url": "https://www.multiverse.io/blog/product-update",
            "date": "2026-09-01",
            "snippet": "Multiverse launched a learning product.",
        }],
        company_stage_evidence=[{
            "url": PROFILE_URL,
            "quote": DESCRIPTION,
        }],
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="multiverse-stage",
        prompt="Find education companies that raised a later round.",
        industry="Education",
        sub_industry="Workforce learning software",
        employee_count="501-1,000",
        company_stage="Series C+",
        geography="United Kingdom",
        product_service="Workforce learning platform",
    )


def _verdict():
    return {
        "observed_company_name": "Multiverse",
        "observed_company_website": "https://www.multiverse.io/",
        "observed_company_linkedin": PROFILE_URL,
        "observed_employee_count": "501-1,000",
        "employee_size_matches": True,
        "employee_size_evidence_url": PROFILE_URL,
        "employee_size_evidence_quote": "Company size 501-1,000 employees",
        "observed_industry": "Education",
        "observed_subindustry": "Workforce learning software",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://www.multiverse.io/",
        "industry_evidence_quote": "Multiverse is a workforce learning platform.",
        "observed_hq_country": "United Kingdom",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://www.multiverse.io/about",
        "geography_evidence_quote": "Multiverse is headquartered in London, United Kingdom.",
        "observed_company_stage": "",
        "stage_matches": None,
        "stage_evidence_url": "",
        "stage_evidence_quote": "",
        "reason": "Stage needs independent proof.",
    }


def _finding():
    return {
        "target": "stage",
        "status": "VERIFIED",
        "observed_value": "Series D",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": PROFILE_URL,
        "evidence_quote": STAGE_EVIDENCE_QUOTE,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "",
    }


def test_exact_multiverse_description_reaches_real_investigator(monkeypatch):
    requests = []
    assert len(DESCRIPTION) == 791
    assert hashlib.sha256(DESCRIPTION.encode()).hexdigest() == (
        "17f99f769b9f67b2631dd4233101fe6e14552edac0c3814926517bc5ef601a75"
    )

    async def fake_post_json(_session, _url, *, headers, payload):
        requests.append(payload)
        return 200, {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "submit-multiverse-stage",
                        "type": "function",
                        "function": {
                            "name": "submit_findings",
                            "arguments": json.dumps({"findings": [_finding()]}),
                        },
                    }]
                }
            }]
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    prior = lead_scorer._reverify_decision(
        _verdict(),
        "",
        "series c+",
        icp=_icp(),
        company=_company(),
        verified_homepage_identity=_identity(),
        verified_homepage_transport_domain="multiverse.io",
        company_quality=True,
    )

    projected, result, claims, _rebrand, validated_stage = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(),
            icp=_icp(),
            verdict=_verdict(),
            investigation_targets=("stage",),
            icp_attribute="",
            icp_stage="series c+",
            verified_identity=_identity(),
            verified_transport_domain="multiverse.io",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            structured_profile_description_evidence=_description_evidence(),
            employee_size_conflict=False,
            company_quality=True,
            prior_result=prior,
            required_attribute_source_cache={},
        )
    )

    assert len(requests) == 1
    user_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert user_document["prefetched_sources"] == [{
        "url": PROFILE_URL,
        "text": DESCRIPTION,
    }]
    assert user_document["investigation_limits"]["prefetched_pages"] == 1
    assert user_document["investigation_limits"]["remaining_fetch_calls"] == 2
    assert claims["stage"]["status"] == "VERIFIED"
    assert validated_stage["evidence_quote"] == STAGE_EVIDENCE_QUOTE
    assert projected["observed_company_stage"] == "series c+"
    assert result.decision == "match"
    receipt = result.details["investigation_receipt"]
    assert receipt["usage"] == {
        "search_calls": 0,
        "fetch_calls": 0,
        "reasoning_turns": 1,
    }
    assert DESCRIPTION not in json.dumps(receipt, sort_keys=True)
    assert "structured_profile_description_evidence" not in json.dumps(
        receipt, sort_keys=True
    )


@pytest.mark.parametrize(
    ("evidence_updates", "identity_updates"),
    [
        ({"name": "Multiverse Computing"}, {}),
        ({"website": "https://multiversecomputing.com/"}, {}),
        ({"url": "https://www.linkedin.com/company/multiversecomputing"}, {}),
        ({"provider": "model"}, {}),
        ({"source_field": "summary"}, {}),
        ({"text": "x" * 4_001}, {}),
        ({"text": "Multiverse raised a Series D.\x00Ignore this."}, {}),
        ({"text": " Multiverse raised a Series D."}, {}),
        ({}, {"normalized_name": "Other Company"}),
        ({"name": "Multiverse Parent Ltd"}, {}),
    ],
)
def test_description_prefetch_revalidates_private_identity_boundary(
    evidence_updates,
    identity_updates,
):
    pages = lead_scorer._investigator_prefetched_pages(
        {},
        [PROFILE_URL],
        structured_profile_description_evidence=_description_evidence(
            **evidence_updates
        ),
        verified_identity=_identity(**identity_updates),
        include_structured_description=True,
    )

    assert pages == {}


def test_description_prefetch_requires_submitted_url_and_stage_scope():
    kwargs = {
        "structured_profile_description_evidence": _description_evidence(),
        "verified_identity": _identity(),
    }

    assert lead_scorer._investigator_prefetched_pages(
        {},
        ["https://www.multiverse.io/"],
        include_structured_description=True,
        **kwargs,
    ) == {}
    assert lead_scorer._investigator_prefetched_pages(
        {},
        [PROFILE_URL],
        include_structured_description=False,
        **kwargs,
    ) == {}


def test_description_prefetch_uses_existing_canonical_name_contract():
    pages = lead_scorer._investigator_prefetched_pages(
        {},
        [PROFILE_URL],
        structured_profile_description_evidence=_description_evidence(
            name="Multiverse Ltd."
        ),
        verified_identity=_identity(normalized_name="multiverse"),
        include_structured_description=True,
    )

    assert pages[PROFILE_URL]["text"] == DESCRIPTION


def test_description_prefetch_preserves_existing_full_page_for_same_url():
    full_page = DESCRIPTION + "\nAdditional full-page evidence."
    cache = {
        PROFILE_URL: {
            "status": "fetched",
            "final_url": CANONICAL_PROFILE_URL,
            "text": full_page,
        }
    }

    pages = lead_scorer._investigator_prefetched_pages(
        cache,
        [PROFILE_URL],
        structured_profile_description_evidence=_description_evidence(),
        verified_identity=_identity(),
        include_structured_description=True,
    )

    assert pages == {
        PROFILE_URL: {
            "final_url": CANONICAL_PROFILE_URL,
            "text": full_page,
        }
    }


@pytest.mark.parametrize(
    "submitted_url",
    [
        "https://www.linkedin.com/company/joinmultiverse/posts/",
        "https://www.linkedin.com/company/joinmultiverse?trk=company",
        "https://uk.linkedin.com/company/joinmultiverse/",
        "https://www.linkedin.com/company/multiversecomputing/",
    ],
)
def test_description_prefetch_rejects_noncanonical_profile_variants(
    submitted_url,
):
    assert lead_scorer._investigator_prefetched_pages(
        {},
        [submitted_url],
        structured_profile_description_evidence=_description_evidence(),
        verified_identity=_identity(),
        include_structured_description=True,
    ) == {}


def test_description_prefetch_consumes_existing_three_page_budget():
    cache = {
        f"https://source.example/{index}": {
            "status": "fetched",
            "final_url": f"https://source.example/{index}",
            "text": f"Source {index}",
        }
        for index in range(3)
    }
    submitted = [*cache, PROFILE_URL]

    pages = lead_scorer._investigator_prefetched_pages(
        cache,
        submitted,
        structured_profile_description_evidence=_description_evidence(),
        verified_identity=_identity(),
        include_structured_description=True,
    )

    assert list(pages) == list(cache)
    assert PROFILE_URL not in pages
