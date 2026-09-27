"""Prompt regressions for source-grounded functional hiring criteria."""

from __future__ import annotations

import pytest

from qualification.scoring import intent_verification_three_stage as intent


TARGET = (
    "Actively hiring for admissions, student operations, platform, or growth "
    "roles, per current job postings or a careers page."
)
UNIBUDDY_URL = (
    "https://unibuddy-1668416154.teamtailor.com/jobs/"
    "7606923-software-engineer-ii-chat-systems-fully-remote-europe"
)
UNIBUDDY_NATIVE_DESCRIPTION = (
    "“You will gain exposure to the full stack of the Unibuddy platform across "
    "web, native, and backend to deliver engaging solutions to our users and "
    "customers.”"
)
UNIBUDDY_TEXT = (
    "We’re looking for an exceptional mid-level software engineer with a "
    "passion for building great products to join our team. You will gain "
    "exposure to the full stack of the Unibuddy platform across web, native, "
    "and backend to deliver engaging solutions to our users and customers. "
    "This role will be focussed on our Chat, Assistant, and Intelligence "
    "products. You will be working directly with products that provide a "
    "series of experiences and tools to prospective students to engage with "
    "universities, and for university staff and ambassadors to enable positive "
    "conversations, valuable data, and ultimately increase yield. Demonstrate "
    "true ownership of a product - owning its ongoing upkeep, maintenance, "
    "performance, and bug resolution."
)


def _row(
    *,
    target: str = TARGET,
    claim: str = "Acme is hiring for a platform role.",
) -> dict:
    return {
        "id": "signal-1",
        "company": "Acme",
        "website": "https://acme.example",
        "company_linkedin": "https://linkedin.com/company/acme",
        "contact_linkedin": "",
        "claim": claim,
        "signal_date": "2026-04-21",
        "signal_type": "intent",
        "claimed_source_urls": ["https://acme.example/jobs/role"],
        "_target_signal_text": target,
        "_evidence_type": "HIRING",
    }


def _stage_three_prompt(
    row: dict,
    source_text: str,
    *,
    url: str | None = None,
    verified_identity_context: dict | None = None,
) -> str:
    source_url = url or row["claimed_source_urls"][0]
    return intent._build_final_judge_prompt(
        row,
        {
            "results": [{
                "url": source_url,
                "title": "Current job posting",
                "text": source_text,
                "source_publication_date": "2026-04-21",
            }],
            "statuses": [],
        },
        verified_identity_context=verified_identity_context,
    )


def test_unibuddy_source_and_functional_role_rule_reach_stage_three() -> None:
    row = _row(
        claim=UNIBUDDY_NATIVE_DESCRIPTION,
    )
    row.update(
        company="Unibuddy",
        website="https://unibuddy.com/",
        company_linkedin="",
        claimed_source_urls=[UNIBUDDY_URL],
        _integrity_policy=True,
        _buyer_max_age_days=365,
    )

    prompt = _stage_three_prompt(
        row,
        UNIBUDDY_TEXT,
        url=UNIBUDDY_URL,
        verified_identity_context={
            "observed_name": "unibuddy",
            "observed_domain": "unibuddy.com",
            "observed_linkedin_slug": "unibuddy",
        },
    )
    normalized_prompt = " ".join(prompt.split())

    assert TARGET in prompt
    assert UNIBUDDY_NATIVE_DESCRIPTION[1:-1] in prompt
    assert UNIBUDDY_URL in prompt
    assert UNIBUDDY_TEXT in prompt
    assert "title and submitted claim quote need not repeat" in prompt
    assert (
        "direct responsibility to build, operate, own, maintain, or improve "
        "platform products or components qualifies"
    ) in normalized_prompt
    assert "including platform architecture, production readiness" in normalized_prompt
    assert "whether the work is internal or customer-facing" in normalized_prompt
    assert (
        "one sentence may establish the platform scope and another may assign "
        "the direct duties"
    ) in normalized_prompt
    assert "The duty sentence need not repeat the platform name" in normalized_prompt
    assert "an internal developer platform" in prompt
    assert (
        "When miner_claim is an exact quote grounded in the supplied source, bind "
        "it using the surrounding exact body for that same documented hiring event "
        "and role"
    ) in normalized_prompt
    assert (
        "The quote may omit function, event, or role words; that omission does not "
        "fail PART A"
    ) in normalized_prompt
    assert (
        "surrounding exact source text for the same documented hiring event and role "
        "may complete an exact submitted quote that omits function, event, or role words"
    ) in normalized_prompt
    assert "It may not replace a contradictory claim" in normalized_prompt
    assert "ARENA INTEGRITY DATE POLICY" in prompt
    assert '"linkedin_company_slug":"unibuddy"' in prompt


@pytest.mark.parametrize(
    ("source_text", "required_rule"),
    [
        pytest.param(
            (
                "Software Engineer, Core Systems. You will build and operate "
                "the shared identity and data services that power every layer "
                "of the Acme platform."
            ),
            "direct responsibility to build,\n    operate, own, maintain, or improve platform products or components",
            id="direct-platform-duties",
        ),
        pytest.param(
            (
                "Software Engineer. You will deliver customer features with "
                "our product team. About Acme: Acme sells a workflow platform."
            ),
            "generic software role at a company that sells a\n    platform",
            id="generic-software-at-platform-company",
        ),
        pytest.param(
            (
                "Business Analyst. You will use our reporting tools and gain "
                "exposure to the Acme platform while preparing weekly reports."
            ),
            "gaining incidental exposure to it",
            id="incidental-platform-exposure",
        ),
        pytest.param(
            (
                "Sales Solutions Engineer. This role belongs to the sales "
                "organization and demonstrates the platform to prospects."
            ),
            "Generic presales, customer\n    support",
            id="explicitly-excluded-function",
        ),
    ],
)
def test_functional_role_boundaries_reach_source_grounded_prompt(
    source_text: str,
    required_rule: str,
) -> None:
    target = "Actively hiring for platform roles, excluding sales roles."
    prompt = _stage_three_prompt(_row(target=target), source_text)

    assert target in prompt
    assert source_text in prompt
    assert required_rule in prompt


def test_functional_hiring_guidance_is_bounded_to_hiring_evidence() -> None:
    row = _row(target="Launched a major product capability this quarter.")
    row["_evidence_type"] = "PRODUCT_LAUNCH"

    for prompt in (
        intent._build_verification_prompt(row),
        _stage_three_prompt(row, "Acme launched a new reporting API."),
    ):
        assert "HIRING — FUNCTIONAL ROLE MATCH" not in prompt
        assert "job title to repeat the category word" not in prompt


def test_functional_role_precedence_does_not_import_unrelated_evidence() -> None:
    source_text = (
        "This role gains exposure to the Acme platform. A separate sales role "
        "operates the platform. About Acme: our platform powers global teams."
    )

    prompt = _stage_three_prompt(_row(), source_text)
    normalized_prompt = " ".join(prompt.split())

    assert source_text in prompt
    assert "Never borrow duties from another role or hiring event" in normalized_prompt
    assert "merely using a platform" in normalized_prompt
    assert "generic software role" in normalized_prompt


def test_broad_platform_role_does_not_erase_narrow_target_qualifiers() -> None:
    target = "Actively hiring for internal developer platform roles."
    prompt = _stage_three_prompt(
        _row(target=target),
        "Product Engineer. Own and maintain customer-facing product features.",
    )
    normalized_prompt = " ".join(prompt.split())

    assert target in prompt
    assert "If the target instead asks for" in normalized_prompt
    assert "an internal developer platform" in prompt
    assert "preserve that qualifier" in prompt


def test_customer_facing_platform_operations_reach_broad_platform_prompt() -> None:
    target = (
        "Is actively hiring for platform, infrastructure, or revenue "
        "operations roles, with evidence in current job postings or a careers page."
    )
    source_text = (
        "Senior Platform Architect - West. Work with customers to define "
        "production readiness criteria and operational standards. Guide "
        "reliability, failure handling, scaling, upgrades, alerting, migrations, "
        "observability, and readiness certification."
    )

    prompt = _stage_three_prompt(_row(target=target), source_text)
    normalized_prompt = " ".join(prompt.split())

    assert target in prompt
    assert source_text in prompt
    assert "whether the work is internal or customer-facing" in normalized_prompt
    assert "production readiness, reliability, scaling" in normalized_prompt
    assert "same role has no direct platform duties" in normalized_prompt
