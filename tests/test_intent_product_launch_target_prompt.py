"""Prompt contract tests for target-relative product-launch semantics."""

import pytest

from qualification.scoring import intent_verification_three_stage as intent


REBEL_TARGET = (
    "Launched a new commerce capability, storefront format, or retail channel "
    "expansion in the last 12 months, with proof in a press release, product "
    "page, or company announcement."
)
REBEL_SOURCE = (
    "Rebel, which sources excess, unused inventory from brands and retailers "
    "to sell on its marketplace, added bulk-sized snacks and pantry products "
    "to its lineup in mid-May."
)


def _row(target: str, claim: str) -> dict:
    return {
        "id": "signal-1",
        "company": "REBEL",
        "website": "https://fromrebel.com",
        "company_linkedin": "https://linkedin.com/company/fromrebel",
        "contact_linkedin": "",
        "claim": claim,
        "signal_date": "2026-06-12",
        "signal_type": "intent",
        "claimed_source_urls": ["https://publisher.example/rebel-launch"],
        "_target_signal_text": target,
        "_evidence_type": "PRODUCT_LAUNCH",
        "_integrity_policy": True,
        "_buyer_max_age_days": 365,
    }


def _prompts(target: str, claim: str, source: str) -> tuple[str, str]:
    row = _row(target, claim)
    return (
        intent._build_verification_prompt(row),
        intent._build_final_judge_prompt(
            row,
            {
                "results": [{
                    "url": row["claimed_source_urls"][0],
                    "title": "Launch report",
                    "text": source,
                }],
                "statuses": [],
            },
        ),
    )


def test_rebel_projection_keeps_narrow_target_and_assortment_boundary() -> None:
    prompts = _prompts(
        REBEL_TARGET,
        "REBEL expanded its marketplace into CPG with snacks and pantry products.",
        REBEL_SOURCE,
    )

    for prompt in prompts:
        assert REBEL_TARGET in prompt
        assert "`PRODUCT_LAUNCH` is only an evidence category" in prompt
        assert "does not add\n        an alternative to target_icp_signal" in prompt
        assert "semantically\n        satisfy one actual requested alternative" in prompt
        assert "do not require literal word overlap" in prompt
        assert "adding\n        merchandise to an existing catalog or marketplace does not by itself" in prompt
    assert REBEL_SOURCE in prompts[1]


@pytest.mark.parametrize(
    ("target", "claim", "source"),
    [
        (
            "Launched a new product category or assortment in the last year.",
            "REBEL launched a new snack category.",
            REBEL_SOURCE,
        ),
        (
            "Launched a new commerce capability in the last year.",
            "Example launched one-click checkout.",
            "Example introduced one-click checkout for all signed-in shoppers.",
        ),
        (
            "Launched a new storefront or store format in the last year.",
            "Example opened its first shop-in-shop format.",
            "Example opened its first shop-in-shop format inside 20 stores.",
        ),
        (
            "Expanded into a major retail channel in the last year.",
            "Example launched nationwide at Target.",
            "Example products launched nationwide at Target in August 2026.",
        ),
    ],
)
def test_product_launch_controls_keep_each_actual_target_visible(
    target: str, claim: str, source: str,
) -> None:
    stage_one, stage_three = _prompts(target, claim, source)

    for prompt in (stage_one, stage_three):
        assert target in prompt
        assert "when the target asks for that\n        kind of launch" in prompt
        assert "one actual requested alternative" in prompt
    assert source in stage_three


@pytest.mark.parametrize('target,rule', [
    ('Launched a major capability in the last year.',
     "When the target leaves proof sources unrestricted"),
    ('Launched a major capability; evidence may include a product page or changelog.',
     "presents source types as examples"),
    ('Launched a major capability, such as documented in a press release.',
     "reliable independent reporting of the same company event can prove it"),
    ('Launched a major capability; must be documented in a product page or changelog.',
     "'must be documented in a product page or changelog' requires one of those document types"),
    ('Launched a major capability; proof must be a company-issued announcement.',
     "'proof must be a company-issued announcement' requires that provenance"),
    ('Launched a major capability; source must be the company.',
     "'source must be the company' requires that provenance"),
    ('Launched a major capability announced by the company.',
     "Naming the company as the announcing actor alone does not restrict the source publisher"),
])
def test_source_requirement_prompt_contract_in_both_intent_stages(target, rule):
    # Preserve source requirements in the actual prompts; no model decision is simulated.
    prompts = _prompts(target, 'REBEL launched a major capability.', REBEL_SOURCE)
    for prompt in prompts:
        normalized = ' '.join(prompt.split())
        assert target in normalized
        assert rule in normalized
        assert "Preserve explicitly mandatory evidence source types and issuer requirements" in normalized
        assert "Independent reporting cannot replace a mandatory document type or issuer" in normalized
        assert "Naming the company as the announcing actor alone does not restrict the source publisher" in normalized
        assert "'announced by the company' requires that issuer" not in normalized
        assert "only when the target says the proof must come only from the company" not in normalized
