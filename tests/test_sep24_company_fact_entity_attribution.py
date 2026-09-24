"""Regression coverage for source-bound company fact attribution."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
)
from qualification.scoring.competition import (
    scorer_breakdown_has_retryable_infrastructure_failure,
)
from qualification.scoring.lead_scorer import (
    INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS,
    _evidence_has_no_established_source_conflict,
    _ground_required_attribute_evidence,
    _llm_reverify_company,
    _reverify_decision,
)


def _company(
    name: str = "Multiverse",
    website: str = "https://multiverse.io",
    linkedin: str = "https://linkedin.com/company/joinmultiverse",
) -> CompanyOutput:
    return CompanyOutput(
        company_name=name,
        company_website=website,
        company_linkedin=linkedin,
        industry="Education",
        employee_count="201-500",
        country="United States",
        intent_signals=[{
            "description": "Multiverse expanded its learning platform.",
            "source": "company_website",
            "url": "https://multiverse.io/news/expansion",
            "date": "2026-08-01",
            "snippet": "Multiverse expanded its learning platform.",
        }],
    )


def _icp(*, stage: str = "", attribute: str = "") -> ICPPrompt:
    return ICPPrompt(
        icp_id="sep24-entity-attribution",
        prompt="Education companies with a learning platform",
        industry="Education",
        sub_industry="Learning platforms",
        employee_count="201-500",
        company_stage=stage,
        geography="United States",
        product_service="learning platform",
        required_attribute=attribute,
    )


def _anchor() -> dict[str, object]:
    return {
        "normalized_name": "Multiverse",
        "registrable_dns_domain": "multiverse.io",
        "linkedin_company_slug": "joinmultiverse",
        "verified_legal_name_aliases": ["Multiverse Learning Ltd"],
    }


def _verdict() -> dict[str, object]:
    return {
        "observed_company_name": "Multiverse",
        "observed_company_website": "https://multiverse.io/about",
        "observed_company_linkedin": "https://linkedin.com/company/joinmultiverse",
        "observed_employee_count": "201-500",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://multiverse.io/about",
        "employee_size_evidence_quote": "Multiverse has 201-500 employees.",
        "observed_industry": "Education",
        "observed_subindustry": "Learning platforms",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://multiverse.io/about",
        "industry_evidence_quote": "Multiverse provides a learning platform.",
        "observed_hq_country": "United States",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://multiverse.io/about",
        "geography_evidence_quote": "Multiverse is headquartered in the United States.",
        "observed_company_stage": "",
        "stage_matches": None,
        "stage_evidence_url": "",
        "stage_evidence_quote": "",
        "attribute_satisfied": None,
        "required_attribute_evidence_url": "",
        "required_attribute_evidence_quote": "",
        "reason": "Independent sources were checked.",
    }


def _attributed(
    url: str,
    quote: str,
    *,
    aliases: list[str] | None = None,
    rebrand: dict[str, object] | None = None,
) -> bool:
    anchor = _anchor()
    if aliases is not None:
        anchor["verified_legal_name_aliases"] = aliases
    return _evidence_has_no_established_source_conflict(
        {"url": url, "quote": quote},
        verified_homepage_identity=anchor,
        verified_rebrand_identity=rebrand,
    )


@pytest.mark.parametrize(
    ("url", "quote", "expected"),
    [
        (
            "https://multiverse.io/news/expansion",
            "The learning platform expanded into the United States.",
            True,
        ),
        (
            "https://www.prnewswire.com/news/multiverse-expansion",
            "Multiverse announces an expansion of its learning platform.",
            True,
        ),
        (
            "https://www.prnewswire.com/news/multiverse-expansion",
            "The company expanded its learning platform into a new market.",
            True,
        ),
        (
            "https://partner.example/news/multiverse",
            "The Multiverse AI platform expanded its learning service.",
            True,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "Multiverse Computing announces a Series C fundraising target.",
            False,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "Announces Series C Fundraising Targeting up to $570M to power AI.",
            False,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "Multiverse Computing announced a fundraise. Multiverse offers "
            "learning services, and the company plans further expansion.",
            False,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "The expansion belongs to the learning platform at multiverse.io.",
            True,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "The company profile is linkedin.com/company/joinmultiverse.",
            True,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "See notmultiverse.io.evil.example for the company profile.",
            False,
        ),
    ],
)
def test_company_fact_attribution_is_bound_without_brittle_verb_rules(
    url: str,
    quote: str,
    expected: bool,
) -> None:
    assert _attributed(url, quote) is expected


def test_registered_alias_and_verified_rebrand_names_are_eligible() -> None:
    conflict_url = (
        "https://www.linkedin.com/posts/multiversecomputing_"
        "multiverse-computing-series-c-activity-7487401725201453056-mugN"
    )
    assert _attributed(
        conflict_url,
        "Multiverse Learning Ltd expanded its education platform.",
    )
    assert _attributed(
        conflict_url,
        "Oldverse launched its learning platform.",
        aliases=[],
        rebrand={
            "status": "VERIFIED",
            "old_name": "Oldverse",
            "new_name": "Multiverse",
        },
    )


def test_publisher_collision_requires_complete_verified_homepage_anchor() -> None:
    url = (
        "https://www.linkedin.com/posts/multiversecomputing_"
        "multiverse-computing-series-c-activity-7487401725201453056-mugN"
    )
    assert _evidence_has_no_established_source_conflict(
        {"url": url, "quote": "The company announced a funding round."},
        verified_homepage_identity=None,
    )


def test_non_ascii_verified_name_does_not_create_empty_prefix_conflict() -> None:
    assert _evidence_has_no_established_source_conflict(
        {
            "url": "https://www.linkedin.com/posts/duoyuancomputing_update-1",
            "quote": "The company announced an expansion.",
        },
        verified_homepage_identity={
            "normalized_name": "多元",
            "registrable_dns_domain": "duoyuan.example",
            "linkedin_company_slug": "join-duoyuan",
        },
    )


def test_exact_anchor_only_resolves_publisher_not_event_subject() -> None:
    # The deterministic guard does not parse multi-company prose. The verifier
    # prompt still requires the model to bind the event to the intended entity.
    url = (
        "https://www.linkedin.com/posts/multiversecomputing_"
        "multiverse-computing-series-c-activity-7487401725201453056-mugN"
    )
    assert _attributed(
        url,
        "Multiverse Computing announced a fundraise. Footer: multiverse.io.",
    )


def test_grounded_multiverse_computing_quote_does_not_transfer_to_multiverse() -> None:
    url = (
        "https://www.linkedin.com/posts/multiversecomputing_"
        "multiverse-computing-series-c-activity-7487401725201453056-mugN"
    )
    quote = (
        "Announces Series C Fundraising Targeting up to $570M (€500M) "
        "to Power Efficient AI from Edge to Cloud"
    )
    verdict = _verdict()
    verdict.update(
        attribute_satisfied=True,
        required_attribute_evidence_url=url,
        required_attribute_evidence_quote=quote,
    )
    grounded, repair = asyncio.run(
        _ground_required_attribute_evidence(
            verdict,
            active_attribute=True,
            source_cache={
                url: {
                    "status": "fetched",
                    "final_url": url,
                    "text": (
                        "Multiverse Computing, developer of CompactifAI, "
                        f"{quote}. Multiverse Computing builds efficient AI."
                    ),
                }
            },
        )
    )

    assert repair == {}
    assert grounded["_server_verified_required_attribute_grounding"]["status"] == (
        "grounded"
    )
    result = _reverify_decision(
        grounded,
        "learning platform AND recent expansion",
        "",
        icp=_icp(attribute="learning platform AND recent expansion"),
        company=_company(),
        verified_homepage_identity=_anchor(),
        verified_homepage_transport_domain="multiverse.io",
    )
    assert result.details["required_attribute_decision"] == (
        COMPANY_FIT_UNAVAILABLE
    )


def test_persistent_entity_conflict_is_company_local_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = (
        "https://www.linkedin.com/posts/multiversecomputing_"
        "multiverse-computing-series-c-activity-7487401725201453056-mugN"
    )
    quote = (
        "Announces Series C Fundraising Targeting up to $570M (€500M) "
        "to Power Efficient AI from Edge to Cloud"
    )
    verdict = _verdict()
    verdict.update(
        attribute_satisfied=True,
        required_attribute_evidence_url=url,
        required_attribute_evidence_quote=quote,
    )
    provider = AsyncMock(side_effect=[(dict(verdict), ""), (dict(verdict), "")])
    fetch = AsyncMock(return_value=(
        200,
        url,
        f"Multiverse Computing builds CompactifAI. {quote}.",
    ))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        provider,
    )
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._fetch_bounded_html",
        fetch,
    )
    verified_homepage = company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "Multiverse",
                "observed_domain": "multiverse.io",
                "observed_linkedin_slug": "joinmultiverse",
            }
        },
    )

    result = asyncio.run(_llm_reverify_company(
        _company(),
        _icp(attribute="learning platform AND recent expansion"),
        verified_homepage_identity=verified_homepage,
    ))

    assert provider.await_count == 2
    assert fetch.await_count == 1
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == (
        INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS
    )
    assert "failure_reason_code" not in result.details
    assert result.details["entity_attribution_conflicts"] == [
        "required_attribute"
    ]
    assert not scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


@pytest.mark.parametrize(
    ("url", "quote", "expected"),
    [
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "Announces Series C Fundraising Targeting up to $570M (€500M) "
            "to Power Efficient AI from Edge to Cloud",
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN",
            "Multiverse Computing announces Series C fundraising to power AI.",
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "https://www.prnewswire.com/news/multiverse-expansion",
            "Multiverse announces an expansion of its learning platform.",
            COMPANY_FIT_MATCH,
        ),
        (
            "https://multiverse.io/news/expansion",
            "The learning platform expanded into a new market.",
            COMPANY_FIT_MATCH,
        ),
    ],
)
def test_required_attribute_rejects_wrong_entity_but_keeps_bound_sources(
    url: str,
    quote: str,
    expected: str,
) -> None:
    verdict = _verdict()
    verdict.update(
        attribute_satisfied=True,
        required_attribute_evidence_url=url,
        required_attribute_evidence_quote=quote,
    )

    result = _reverify_decision(
        verdict,
        "learning platform AND recent expansion",
        "",
        icp=_icp(attribute="learning platform AND recent expansion"),
        company=_company(),
        verified_homepage_identity=_anchor(),
        verified_homepage_transport_domain="multiverse.io",
    )

    assert result.details["required_attribute_decision"] == expected


def test_stage_evidence_from_confusable_linkedin_publisher_stays_unproven() -> None:
    verdict = _verdict()
    verdict.update(
        observed_company_stage="Series C",
        stage_matches=True,
        stage_evidence_url=(
            "https://www.linkedin.com/posts/multiversecomputing_"
            "multiverse-computing-series-c-activity-7487401725201453056-mugN"
        ),
        stage_evidence_quote=(
            "Multiverse Computing announced its $570M Series C funding round."
        ),
    )

    result = _reverify_decision(
        verdict,
        "",
        "series c+",
        icp=_icp(stage="Series C+"),
        company=_company(),
        verified_homepage_identity=_anchor(),
        verified_homepage_transport_domain="multiverse.io",
    )

    assert result.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
