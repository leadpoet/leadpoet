from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator


UNIBUDDY_URL = "https://unibuddy.com/"
UNIBUDDY_QUOTE = (
    "Our platform boosts enrollment by building trust and confidence through "
    "scalable peer-to-peer & community engagement."
)


def _finding(*, status: str, role: str, url: str = "", quote: str = "", reason: str):
    return {
        "target": "industry",
        "status": status,
        "observed_value": "Higher education enrollment platform",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Education",
        "observed_subindustry": "Higher education services",
        "activity_role": role,
        "evidence_url": url,
        "evidence_quote": quote,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": reason,
    }


def _identity_anchor(name: str, domain: str) -> dict[str, str]:
    normalized_name = "".join(
        character for character in name.casefold() if character.isalnum()
    )
    linkedin_slug = normalized_name
    return {
        "submitted_name": name,
        "submitted_domain": domain,
        "submitted_linkedin_slug": linkedin_slug,
        "observed_name": normalized_name,
        "observed_domain": domain,
        "observed_linkedin_slug": linkedin_slug,
        "verified_name": normalized_name,
        "verified_domain": domain,
        "verified_linkedin_slug": linkedin_slug,
    }


def _validate_industry(
    finding: dict,
    *,
    name: str,
    domain: str,
    page_text: str = "",
):
    url = finding["evidence_url"]
    return investigator._validated_findings(
        {"findings": [finding]},
        targets=("industry",),
        fetched_pages={url: page_text} if url else {},
        fetched_final_urls={url: url} if url else {},
        first_party_domains={domain},
        identity_names={
            "".join(character for character in name.casefold() if character.isalnum())
        },
        identity_anchor=_identity_anchor(name, domain),
    )["industry"]


def test_unibuddy_full_criterion_reaches_prompt_and_verified_final_validation(
    monkeypatch,
):
    requested_industry = "Education"
    requested_subindustry = "Higher education services"
    requested_product_service = (
        "A student enrollment, learning, or campus-operations platform used by "
        "education providers to manage admissions, programs, and learner workflows."
    )
    requested_attribute = (
        "Operates a software or services platform used by education providers to "
        "manage enrollment, learning delivery, student communication, or campus "
        "operations."
    )
    finding = _finding(
        status="VERIFIED",
        role="supplier_operator",
        url=UNIBUDDY_URL,
        quote=UNIBUDDY_QUOTE,
        reason=(
            "Unibuddy supplies a platform used for enrollment and student "
            "communication."
        ),
    )
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "submit-unibuddy",
                                "type": "function",
                                "function": {
                                    "name": "submit_findings",
                                    "arguments": json.dumps({"findings": [finding]}),
                                },
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock())
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock())

    result = asyncio.run(
        investigator.investigate_company_evidence(
            company_locator={
                "name": "Unibuddy",
                "website": UNIBUDDY_URL,
                "linkedin": "https://www.linkedin.com/company/unibuddy",
            },
            targets=("industry",),
            requested_industry=requested_industry,
            requested_subindustry=requested_subindustry,
            requested_product_service=requested_product_service,
            requested_attribute=requested_attribute,
            positive_semantic_review=True,
            prior_observations={
                "observed_company_name": "unibuddy",
                "observed_company_website": UNIBUDDY_URL,
                "observed_company_linkedin": (
                    "https://www.linkedin.com/company/unibuddy"
                ),
                "submitted_source_urls": [UNIBUDDY_URL],
            },
            verified_homepage_identity={
                "normalized_name": "unibuddy",
                "registrable_dns_domain": "unibuddy.com",
                "linkedin_company_slug": "unibuddy",
            },
            prefetched_pages={
                UNIBUDDY_URL: {"final_url": UNIBUDDY_URL, "text": UNIBUDDY_QUOTE}
            },
        )
    )

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["activity_role"] == "supplier_operator"
    assert result["claims"]["industry"]["evidence_quote"] == UNIBUDDY_QUOTE
    assert len(requests) == 1
    request_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert request_document["requested_industry"] == requested_industry
    assert request_document["requested_subindustry"] == requested_subindustry
    assert request_document["requested_product_service"] == requested_product_service
    assert request_document["requested_attribute"] == requested_attribute
    assert request_document["prefetched_sources"] == [
        {"url": UNIBUDDY_URL, "text": UNIBUDDY_QUOTE}
    ]


def test_prompt_derives_sector_platform_supplier_role_from_full_criterion():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())

    assert (
        "Derive the requested supplier/operator role from that complete context, "
        "not from the industry or sub-industry label alone"
    ) in prompt
    assert (
        "direct evidence that the investigated company supplies or operates that "
        "platform or service can qualify it as supplier_operator"
    ) in prompt
    assert "Do not require the supplier to operate the customer institution" in prompt
    assert "require the criterion to literally say that vendors qualify" in prompt


def test_generic_sector_platform_supplier_can_pass_real_final_validation():
    url = "https://campusflow.example/platform"
    quote = (
        "CampusFlow provides universities with a platform to manage admissions, "
        "student communications, and enrollment workflows."
    )
    finding = _finding(
        status="VERIFIED",
        role="supplier_operator",
        url=url,
        quote=quote,
        reason="The company supplies the requested sector platform.",
    )

    validated = _validate_industry(
        finding,
        name="CampusFlow",
        domain="campusflow.example",
        page_text=quote,
    )

    assert validated["status"] == "VERIFIED"
    assert validated["activity_role"] == "supplier_operator"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        (
            "zen-school-staffing",
            "School staffing does not prove the requested enrollment, learning, "
            "student-communication, or campus-operations platform.",
        ),
        (
            "vendor-only-to-sector",
            "The customer sector is only a target market; the requested supplied "
            "activity is not established.",
        ),
        (
            "explicit-institution-operator",
            "Supplying software to universities does not prove that the company "
            "itself operates the required university institution.",
        ),
        (
            "missing-and-capability",
            "Enrollment evidence alone does not prove the separately required "
            "learning-delivery capability.",
        ),
    ],
)
def test_semantic_negative_controls_remain_unproven_in_final_validation(
    case,
    reason,
):
    finding = _finding(
        status="UNPROVEN",
        role="unresolved",
        reason=reason,
    )

    validated = _validate_industry(
        finding,
        name=case,
        domain=f"{case}.example",
    )

    assert validated["status"] == "UNPROVEN"
    assert validated["activity_role"] == "unresolved"
    assert validated["evidence_url"] == ""
    assert validated["evidence_quote"] == ""
    assert validated["reason"] == reason


@pytest.mark.parametrize("role", ["customer_user", "internal_function"])
def test_buyer_or_internal_tool_cannot_pass_real_final_validation(role):
    url = "https://buyer.example/operations"
    quote = (
        "BuyerCo uses a third-party enrollment system for its internal training "
        "operations."
    )
    finding = _finding(
        status="VERIFIED",
        role=role,
        url=url,
        quote=quote,
        reason="The company only uses the tool.",
    )

    validated = _validate_industry(
        finding,
        name="BuyerCo",
        domain="buyer.example",
        page_text=quote,
    )

    assert validated["status"] == "UNPROVEN"
    assert validated["reason"] == investigator.INDUSTRY_RELATIONSHIP_UNPROVEN_REASON


def test_prompt_preserves_negative_controls_and_all_required_conjuncts():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())

    assert (
        "A customer sector named only as a target market remains insufficient" in prompt
    )
    assert (
        "a supplier does not satisfy a criterion that explicitly requires the "
        "investigated company itself to be the institution or downstream operator"
    ) in prompt
    assert (
        "Customer use of another vendor's platform and the investigated vendor's "
        "own internal console or workflow remain customer_user or internal_function"
    ) in prompt
    assert (
        "Still prove every separate industry, sub-industry, product/service" in prompt
    )
    assert "required-attribute, or other clause joined by AND" in prompt
    assert "An adjacent activity for the same customer group is UNPROVEN" in prompt
