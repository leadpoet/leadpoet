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


def test_leadership_rules_keep_independent_proof_and_negatives_distinct() -> None:
    row = _row()
    row.update(
        claim="A verified business-unit chair retired from Example.",
        claimed_source_urls=["https://independent.example/news/retirement"],
        _target_signal_text="Announced a leadership change in the last 12 months.",
        _evidence_type="LEADERSHIP_CHANGE",
    )
    broad = intent._build_final_judge_prompt(
        row,
        {
            "results": [{
                "url": row["claimed_source_urls"][0],
                "title": "Board appointment",
                "text": (
                    "The executive recently retired as chair of a business unit "
                    "at Example."
                ),
            }],
            "statuses": [],
        },
    )

    assert (
        "do not invent an issuer or\n        companywide-restructuring requirement"
        in broad
    )
    assert (
        "evidence from a reliable independent publisher can establish" in broad
    )
    assert (
        "A business-unit or function leader can satisfy broad leadership-change"
        in broad
    )
    assert (
        "An ordinary employee move does not automatically establish a\n"
        "        leadership change"
        in broad
    )
    assert (
        "A true departure\n        at a different company fails the entity check"
        in broad
    )
    assert (
        "role, seniority, source or issuer, and organizational-scope constraint"
        in broad
    )
    assert row["claimed_source_urls"][0] in broad

    ordinary_employee = dict(
        row,
        claim="A software engineer left Example.",
    )
    ordinary_employee_prompt = intent._build_verification_prompt(ordinary_employee)
    assert ordinary_employee["claim"] in ordinary_employee_prompt
    assert (
        "An ordinary employee move does not automatically establish a\n"
        "        leadership change"
    ) in ordinary_employee_prompt

    wrong_entity = dict(
        row,
        claim="A verified business-unit chair retired from Other Corp.",
    )
    wrong_entity_prompt = intent._build_verification_prompt(wrong_entity)
    assert wrong_entity["claim"] in wrong_entity_prompt
    assert (
        "A true departure\n        at a different company fails the entity check"
        in wrong_entity_prompt
    )

    explicit_issuer = dict(
        row,
        claim="An independent publisher reported Example's CEO succession.",
        _target_signal_text=(
            "The target company itself announced a companywide CEO succession "
            "in the last 30 days."
        ),
    )
    explicit_issuer_prompt = intent._build_verification_prompt(explicit_issuer)
    assert explicit_issuer["claim"] in explicit_issuer_prompt
    assert explicit_issuer["_target_signal_text"] in explicit_issuer_prompt
    assert (
        "Unless the target explicitly\n"
        "        requires a target-company announcement"
    ) in explicit_issuer_prompt
    assert (
        "Preserve every explicit\n        role, seniority, source or issuer, and "
        "organizational-scope constraint"
    ) in explicit_issuer_prompt
