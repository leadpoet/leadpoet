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
    )


def test_unibuddy_source_and_functional_role_rule_reach_stage_three() -> None:
    row = _row(
        claim=(
            "Unibuddy is recruiting a Software Engineer II to work across its "
            "web, native, and backend platform."
        )
    )
    row.update(
        company="Unibuddy",
        website="https://unibuddy.com/",
        company_linkedin="https://linkedin.com/company/unibuddy",
        claimed_source_urls=[UNIBUDDY_URL],
    )

    prompt = _stage_three_prompt(row, UNIBUDDY_TEXT, url=UNIBUDDY_URL)

    assert TARGET in prompt
    assert UNIBUDDY_URL in prompt
    assert UNIBUDDY_TEXT in prompt
    assert "do not require the job title to repeat the category word" in prompt
    assert "work across the\n    employer's platform stack or components" in prompt
    assert "A submitted claim quote is not the complete evidentiary window" in prompt
    assert "Direct\n    functional duties established there take precedence" in prompt
    assert (
        "Incidental exposure\n    alone is insufficient only when no direct "
        "functional duties are established"
    ) in prompt
    assert "elsewhere in that same job body" in prompt


@pytest.mark.parametrize(
    ("source_text", "required_rule"),
    [
        pytest.param(
            (
                "Software Engineer, Core Systems. You will build and operate "
                "the shared identity and data services that power every layer "
                "of the Acme platform."
            ),
            "employer's platform stack or components",
            id="direct-platform-duties",
        ),
        pytest.param(
            (
                "Software Engineer. You will deliver customer features with "
                "our product team. About Acme: Acme sells a workflow platform."
            ),
            "generic software role at a company that sells\n    a platform",
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
            "Preserve every explicit inclusion and\n    exclusion",
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

    assert source_text in prompt
    assert (
        "Do not replace missing duties with evidence\n    from another role or "
        "hiring event, or with company boilerplate"
    ) in prompt
    assert "Merely using a platform" in prompt
    assert "generic software role" in prompt
