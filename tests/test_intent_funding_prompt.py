"""Funding guidance is type-scoped and preserves explicit ICP narrowing."""

from qualification.scoring import intent_verification_three_stage as intent


GUIDANCE = "TARGET-COMPANY CAPITAL PROOF"


def _row(evidence_type: str, target: str = "Announced a funding round") -> dict:
    return {
        "id": "signal-1",
        "company": "Example",
        "website": "https://example.test",
        "company_linkedin": "",
        "contact_linkedin": "",
        "claim": "Example issued corporate notes.",
        "signal_date": "2026-08-01",
        "signal_type": "intent",
        "claimed_source_urls": ["https://example.test/news/notes"],
        "_target_signal_text": target,
        "_evidence_type": evidence_type,
    }


def test_funding_guidance_routes_to_both_judge_stages() -> None:
    for evidence_type in ("FUNDING", "FINANCING"):
        row = _row(evidence_type)
        prompts = (
            intent._build_verification_prompt(row),
            intent._build_final_judge_prompt(row, {"results": [{
                "url": row["claimed_source_urls"][0],
                "title": "Corporate notes",
                "text": "Example issued $50 million of corporate notes.",
            }]}),
        )

        for prompt in prompts:
            assert GUIDANCE in prompt
            assert "corporate debt or notes as well as equity" in prompt
            assert "Do not require equity-round terminology" in prompt


def test_funding_guidance_preserves_narrow_targets_and_excludes_other_capital() -> None:
    prompt = intent._build_verification_prompt(
        _row("FUNDING", "Raised a Series A equity round")
    )

    assert "Series A equity round" in prompt
    assert "apply" in prompt
    assert "that narrower requirement as written" in prompt
    assert "Customer loans made by the company" in prompt
    assert "limited-partner fund close" in prompt
    assert "assets under management" in prompt


def test_funding_guidance_does_not_leak_to_other_evidence_types() -> None:
    assert GUIDANCE not in intent._build_verification_prompt(_row("HIRING"))
