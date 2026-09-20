"""Prompt regressions for generic AND/OR criterion semantics."""

from qualification.scoring import intent_verification_three_stage as intent


def _row() -> dict:
    return {
        "id": "signal-1",
        "company": "example.com",
        "website": "https://example.com",
        "company_linkedin": "https://linkedin.com/company/example",
        "contact_linkedin": "",
        "claim": "Example launched a new security capability this quarter.",
        "signal_date": "2026-08-03",
        "signal_type": "intent",
        "claimed_source_urls": ["https://example.com/news/security"],
        "_target_signal_text": (
            "Completed a certification OR launched a major security "
            "capability this quarter."
        ),
        "_evidence_type": "PRODUCT_LAUNCH",
    }


def test_logical_criterion_rules_reach_stage_one_and_stage_three() -> None:
    row = _row()
    stage_one = intent._build_verification_prompt(row)
    stage_three = intent._build_final_judge_prompt(
        row,
        {
            "results": [{
                "url": row["claimed_source_urls"][0],
                "title": "Security release",
                "text": "Example released a new runtime security control today.",
            }],
            "statuses": [],
        },
    )

    for prompt in (stage_one, stage_three):
        assert "`OR` / `either` separates alternatives" in prompt
        assert "One complete alternative is\n        sufficient" in prompt
        assert "`AND` / `all` joins requirements" in prompt
        assert "Every joined requirement must hold" in prompt
        assert "time\n        window, threshold, scope qualifier, entity constraint" in prompt
        assert "An alternative does not relax its own requirements" in prompt
        assert "`major` need not appear\n        verbatim" in prompt
        assert "routine update, relabeling, or unsupported marketing claim" in prompt


def test_leadership_rules_reach_both_prompt_stages() -> None:
    row = _row()
    row.update(
        claim="A verified business-unit chair retired from Example.",
        claimed_source_urls=["https://independent.example/news/retirement"],
        _target_signal_text="Announced a leadership change in the last 12 months.",
        _evidence_type="LEADERSHIP_CHANGE",
    )
    prompts = [
        intent._build_verification_prompt(row),
        intent._build_final_judge_prompt(
            row,
            {
                "results": [{
                    "url": row["claimed_source_urls"][0],
                    "title": "Board appointment",
                    "text": "The executive retired as a business-unit chair at Example.",
                }],
                "statuses": [],
            },
        ),
    ]

    for prompt in prompts:
        assert "reliable independent publisher can establish" in prompt
        assert "ordinary employee move does not automatically establish" in prompt
        assert "retirement of a senior division or function leader" in prompt
        assert "A third-party issuer's release" in prompt
        assert "`Announced` does not by itself require" in prompt
        assert "company name in an old\n        or undated biography" in prompt
        assert "different company fails the entity check" in prompt
        assert "Preserve every explicit" in prompt
        assert "company involved in the CLAIMED EVENT" in prompt
        assert "that departure belongs to Company A" in prompt
        assert "Merely mentioning a former employer without a departure" in prompt


def test_advertising_measurement_and_acquisition_completion_reach_both_stages() -> None:
    row = _row()
    for prompt in (
        intent._build_verification_prompt(row),
        intent._build_final_judge_prompt(row, {"results": [], "statuses": []}),
    ):
        assert "paid campaign" in prompt
        assert "visibility, attribution, or advertising ROI" in prompt
        assert "even if the product does not" in prompt
        assert "requires evidence that the transaction closed" in prompt
        assert "court approval" in prompt
        assert "expected future closing do not prove completion" in prompt
