"""Cache identities bind the submitted source hints consumed by Arena research."""

import copy
from lab_arena import judgment_cache, scoring
from qualification.scoring import competition

COMPANY = {
    "company_linkedin": "https://www.linkedin.com/company/ibm",
    "company_name": "IBM",
    "company_stage": "Public",
    "company_stage_evidence": [
        {
            "quote": "IBM common stock is listed on the New York Stock Exchange and the NYSE Texas under the symbol “IBM”.",
            "url": "https://www.ibm.com/investor/help/general-faqs",
        }
    ],
    "company_website": "https://ibm.com/",
    "country": "United States",
    "employee_count": "10,001+",
    "industry": "Data and Analytics",
    "intent_details": "On July 1, 2026, IBM released Cognos Analytics 12.1.3, which brings agentic AI into the governed BI layer to help teams move faster while staying connected to approved data, metrics, and security rules. This release may help IBM expand its data and analytics platform with AI-driven business intelligence capabilities.",
    "intent_signals": [
        {
            "date": "2026-07-01",
            "description": "IBM Cognos Analytics 12.1.3 brings agentic AI into the governed BI layer, helping teams move faster while staying connected to approved data, metrics, and security rules.",
            "matched_icp_signal": 0,
            "url": "https://www.ibm.com/new/announcements/ibm-cognos-analytics-12-1-3-governed-bi-for-the-agentic-enterprise",
        }
    ],
    "required_attribute": {
        "evidence_quote": "Get 20% off your first annual IBM Statistics Base subscription or 10% off your first quarterly subscription.",
        "evidence_url": "https://www.ibm.com/products/offers-and-discounts",
        "explanation": "The quoted sentence on ibm.com describes IBM's offering; it does not show every part of the attribute.",
        "passed": False,
        "text": "Sells a subscription analytics platform and has a recent public signal of product expansion or team growth",
    },
    "state": "New York",
}
ICP = {
    "bonus_intents": [
        {
            "intent_category": "HIRING",
            "intent_max_age_days": 90,
            "intent_signal": "Posted current openings for data engineering, analytics engineering, or integrations roles on its careers page.",
        }
    ],
    "company_stage": "Public",
    "country": "United States",
    "employee_count": [
        "201-500",
        "501-1,000",
        "1,001-5,000",
        "5,001-10,000",
        "10,001+",
    ],
    "excluded_companies": ["domo.com"],
    "geography": "United States",
    "icp_id": "icp_20261006_005",
    "industry": "Data and Analytics",
    "intent_category": "PRODUCT_LAUNCH",
    "intent_max_age_days": 365,
    "intent_signal": "Launched a new product or major capability in the last 12 months, per a press release, product page, or changelog.",
    "intent_signals": [
        "Launched a new product or major capability in the last 12 months, per a press release, product page, or changelog.",
        "Posted current openings for data engineering, analytics engineering, or integrations roles on its careers page.",
    ],
    "max_companies": 5,
    "product_service": "A data and analytics platform that helps teams collect, model, analyze, and visualize business information",
    "prompt": "I’m looking for analytics companies that just launched something big or are hiring hard around data, integrations, and platform work.",
    "required_attribute": "Sells a subscription analytics platform and has a recent public signal of product expansion or team growth",
    "sub_industry": "Business intelligence and analytics software",
    "verified_example_company": "Snowflake",
}


def document(company=None):
    return scoring.build_scoring_input(
        scored_run_id="fixture-execute",
        icp=ICP,
        companies=[copy.deepcopy(COMPANY if company is None else company)],
        policy=scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2",
            intent_details=True,
        ),
        evaluation_date="2026-10-07",
    )


def scope(doc):
    return judgment_cache.build_cache_scope(
        scoring_input=doc,
        round_id="arena-2026-10-07",
        network_name="finney",
        netuid=71,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
    )


def test_consumed_url_and_quote_require_new_keys_but_self_assessment_does_not():
    original = scope(document())
    assert scope(document())["cache_key"] == original["cache_key"]
    for field, value in (
        ("evidence_url", "https://www.ibm.com/different"),
        ("evidence_quote", "Different exact source quote"),
    ):
        c = copy.deepcopy(COMPANY)
        c["required_attribute"][field] = value
        assert scope(document(c))["cache_key"] != original["cache_key"]
    for field, value in (
        ("passed", True),
        ("text", "Different model text"),
        ("explanation", "Different model explanation"),
    ):
        c = copy.deepcopy(COMPANY)
        c["required_attribute"][field] = value
        assert scope(document(c))["cache_key"] == original["cache_key"]


def test_no_hint_inputs_keep_historical_keys_and_malformed_hints_still_fail():
    import pytest
    from tests.lab_arena.judgment_cache_test import _input, _scope

    old = _input()
    expected = _scope(old)
    blank = copy.deepcopy(old)
    blank["companies"][0]["required_attribute"] = None
    assert _scope(blank) == expected
    # This fixed hash is the pre-change projection, not a mirror of the new code.
    assert (
        expected["cache_key"]
        == "sha256:86afab1d39f512ca1a3477eefc05fb6da2c4204415f8afee4168b44dcece3014"
    )
    c = copy.deepcopy(COMPANY)
    c["required_attribute"]["evidence_quote"] = ""
    with pytest.raises(competition.CompetitionScorerInputError):
        scope(document(c))


def test_criteria_date_policy_and_image_still_invalidate():
    original = scope(document())["cache_key"]
    for change in ("criterion", "date", "policy"):
        d = document()
        if change == "criterion":
            d["icp"]["required_attribute"] += " and more"
        elif change == "date":
            d["evaluation_date"] = "2026-10-08"
        else:
            d["scorer_policy"]["judge_models"][
                "company_fit_reverification"
            ] = "other/model"
        assert scope(d)["cache_key"] != original
    d = document()
    changed = judgment_cache.build_cache_scope(
        scoring_input=d,
        round_id="arena-2026-10-07",
        network_name="finney",
        netuid=71,
        scorer_image_digest="sha256:" + "b" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
    )
    assert changed["cache_key"] != original


def test_company_quality_projection_also_binds_only_consumed_hints():
    for quality in (False, True):
        original = competition.effective_competition_input(
            [COMPANY], ICP, company_quality=quality
        )
        c = copy.deepcopy(COMPANY)
        c["required_attribute"]["passed"] = True
        assert (
            competition.effective_competition_input([c], ICP, company_quality=quality)
            == original
        )
        c["required_attribute"]["evidence_quote"] = "Different quote"
        assert (
            competition.effective_competition_input([c], ICP, company_quality=quality)
            != original
        )
