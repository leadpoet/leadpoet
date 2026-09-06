"""Prompt regressions for exact new-market intent alignment."""

from qualification.scoring import intent_verification_three_stage as intent


TARGET = "Expanded into a new market in the last year."
FACILITY_TARGET = "Opened a new operating facility in the last year."
GUIDANCE = "Another facility or asset"


def _row(evidence_type: str) -> dict:
    return {
        "id": "signal-1",
        "company": "Example Energy",
        "website": "https://example.test",
        "company_linkedin": "",
        "contact_linkedin": "",
        "claim": "Example Energy commissioned another storage facility.",
        "signal_date": "2026-08-01",
        "signal_type": "intent",
        "claimed_source_urls": ["https://example.test/news/facility"],
        "_target_signal_text": (
            TARGET if evidence_type == "MARKET_EXPANSION" else FACILITY_TARGET
        ),
        "_evidence_type": evidence_type,
    }


def test_market_expansion_guidance_is_in_both_verifier_stages() -> None:
    row = _row("MARKET_EXPANSION")
    stage_one = intent._build_verification_prompt(row)
    stage_three = intent._build_final_judge_prompt(
        row,
        {
            "results": [{
                "url": row["claimed_source_urls"][0],
                "title": "New facility",
                "text": "The company commissioned its second storage facility.",
            }],
            "statuses": [],
        },
    )

    for prompt in (stage_one, stage_three):
        assert TARGET in prompt
        assert GUIDANCE in prompt
        assert "new geography, customer market" in prompt
        assert "added capacity" in prompt
        assert "quote the source text establishing that fact" in prompt
        assert "PART A fails: return contradicted" in prompt


def test_facility_opening_prompt_does_not_receive_market_expansion_guidance() -> None:
    row = _row("FACILITY_OPENING")

    stage_one = intent._build_verification_prompt(row)
    stage_three = intent._build_final_judge_prompt(
        row,
        {
            "results": [{
                "url": row["claimed_source_urls"][0],
                "title": "New facility",
                "text": "The company commissioned a storage facility.",
            }],
            "statuses": [],
        },
    )

    for prompt in (stage_one, stage_three):
        assert FACILITY_TARGET in prompt
        assert GUIDANCE not in prompt


def test_market_expansion_excludes_capital_raise_but_preserves_explicit_targets() -> None:
    prompt = intent._build_verification_prompt(_row("MARKET_EXPANSION"))

    assert "bond issue, debt offering, equity listing" in prompt
    assert "does not satisfy a general MARKET_EXPANSION" in prompt
    assert "target ICP text itself explicitly asks" in prompt
    assert "corporate debt and other explicit" in prompt
    assert "financing events remain valid evidence" in prompt
    assert "genuinely new customer market" in prompt
