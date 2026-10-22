"""New ICPs must not award bonus breadth for a primary sentence alias."""

import json
from pathlib import Path

import pytest

from gateway.tasks.icp_generator import (
    _normalize_intent_signals,
    canonicalize_generated_icp,
)
from qualification.scoring.competition import (
    _normalized_icp,
    available_intent_score_cap,
)


CAPTURED_ICPS = json.loads(
    (Path(__file__).parent / "fixtures" / "oct22_generated_intent_aliases.json").read_text()
)["icps"]


@pytest.mark.parametrize("original", CAPTURED_ICPS, ids=lambda x: x["icp_id"])
def test_captured_oct22_aliases_are_removed_before_new_icp_is_frozen(original):
    # Read the captured shape without changing the frozen fixture or scorer.
    assert len(_normalized_icp(original)["intent_signals"]) == 3
    assert available_intent_score_cap(original) == 88.0
    out = canonicalize_generated_icp(
        original, industry=original["industry"], sub_industry=original["sub_industry"],
        employee_bucket_radius=0,
    )
    assert out["intent_signal"] == original["intent_signal"]
    assert out["intent_signals"] == [original["intent_signal"], original["intent_signals"][2]]
    assert out["bonus_intents"] == [original["bonus_intents"][1]]
    assert out["intent_category"] == original["intent_category"]
    assert out["intent_max_age_days"] == original["intent_max_age_days"]
    assert out["prompt"].count("Optional bonus intent (") == 1
    assert len(_normalized_icp(out)["intent_signals"]) == 2
    assert available_intent_score_cap(out) == 80.0
    assert canonicalize_generated_icp(
        out, industry=original["industry"], sub_industry=original["sub_industry"],
        employee_bucket_radius=0,
    ) == out


def test_normalizer_preserves_first_wording_and_order_and_deduplicates_before_limit():
    assert _normalize_intent_signals([
        "  Launched   a product. ", "Launched a product", "Launched a product.",
        "Raised funding", "Raised funding.", "Hired a CTO", "Opened a site",
        "Acquired a business", "Partnered with a supplier",
    ]) == [
        "Launched a product.", "Raised funding", "Hired a CTO", "Opened a site",
        "Acquired a business",
    ]


@pytest.mark.parametrize("signals", [
    ["Launched a product", "Did not launch a product"],
    ["Launched a product in the last 90 days", "Launched a product in the last 365 days"],
    ["Launched a product on 2026-10-21", "Launched a product on 2026-10-22"],
    ["Launched a product or service", "Launched a product and service"],
    ["Uses example.com", "Uses example.com."],
    ["Uses https://example.com/v1.2", "Uses https://example.com/v1.2."],
    ["Uses release v1.2", "Uses release v12"],
    ["Uses API.", "Uses A.P.I."],
    ["Released product.", "Released product..", "Released product..."],
    ["Launched a product!", "Launched a product?", "Launched a product"],
])
def test_distinct_requirements_and_identifier_punctuation_are_preserved(signals):
    assert _normalize_intent_signals(signals) == signals


@pytest.mark.parametrize("primary,signals,expected", [
    ("Required launch.", ["Required launch", "Bonus funding", "Required launch."],
     ["Required launch.", "Bonus funding"]),
    ("Required launch.", ["Bonus funding", "Required launch", "Later bonus", "Bonus funding."],
     ["Required launch.", "Bonus funding", "Later bonus"]),
    ("Required launch", ["Bonus funding", "Later bonus"],
     ["Required launch", "Bonus funding", "Later bonus"]),
    (None, ["First launch.", "First launch", "Bonus funding"],
     ["First launch.", "Bonus funding"]),
    ({"bad": "primary"}, ["First launch"], ["First launch"]),
])
def test_primary_is_first_and_duplicate_bonus_aliases_do_not_return(primary, signals, expected):
    out = canonicalize_generated_icp(
        {"intent_signal": primary, "intent_signals": signals, "employee_count": "51-200"},
        industry="Software", sub_industry="B2B SaaS", employee_bucket_radius=0,
    )
    assert out["intent_signal"] == expected[0]
    assert out["intent_signals"] == expected
    assert [x["intent_signal"] for x in out["bonus_intents"]] == expected[1:]
    assert canonicalize_generated_icp(
        out, industry="Software", sub_industry="B2B SaaS", employee_bucket_radius=0,
    ) == out


@pytest.mark.parametrize("malformed", [None, 42, {}, [None, 42, {}, [], "", "  "]])
def test_malformed_signals_do_not_create_textual_requirements(malformed):
    assert _normalize_intent_signals(malformed) == []


def test_missing_signals_keep_existing_fallback(monkeypatch):
    from gateway.tasks import icp_generator

    monkeypatch.setattr(icp_generator.random, "sample", lambda *_: ["Fallback launch"])
    out = canonicalize_generated_icp(
        {"intent_signals": [None, {}]}, industry="Software", sub_industry="B2B SaaS",
    )
    assert out["intent_signal"] == "Fallback launch"
    assert out["intent_signals"] == ["Fallback launch"]
    assert out["bonus_intents"] == []
