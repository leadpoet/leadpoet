from decimal import Decimal

import pytest

from lab_arena.provider_costs import (
    deepline_cost,
    deepline_reservation_cost,
    openrouter_cost,
    scrapingdog_cost,
)


@pytest.mark.parametrize(
    ("operation_id", "parameters", "credits", "microusd"),
    [
        ("scrapingdog.scrape", {}, "5", 250),
        ("scrapingdog.profile", {"type": "profile"}, "100", 5_000),
        ("scrapingdog.profile", {"type": "company"}, "10", 500),
        ("scrapingdog.profile", {"type": "COMPANY"}, "10", 500),
        ("scrapingdog.instagram_profile", {}, "15", 750),
        ("scrapingdog.google", {}, "5", 250),
        ("scrapingdog.future_approved_path", {}, "5", 250),
    ],
)
def test_scrapingdog_legacy_map_and_approved_operation_fallback(
    operation_id, parameters, credits, microusd
):
    cost = scrapingdog_cost(operation_id, parameters)
    assert cost.units == Decimal(credits)
    assert cost.unit_name == "credits"
    assert cost.microusd == microusd


@pytest.mark.parametrize("field", ["credits_charged", "totalCredits", "total_credits", "credits_used", "credits"])
def test_deepline_accepts_bounded_top_level_billing_aliases(field):
    cost = deepline_cost({"billing": {field: "1.23456789"}})
    assert cost is not None
    assert cost.units == Decimal("1.23456789")
    assert cost.microusd == 123_457
    assert field in cost.price_basis


def test_deepline_primary_field_takes_precedence_and_zero_is_valid():
    cost = deepline_cost({"billing": {"credits_charged": 0, "totalCredits": 9}})
    assert cost is not None
    assert cost.units == 0
    assert cost.microusd == 0
    assert "credits_charged" in cost.price_basis


@pytest.mark.parametrize(
    ("tool", "credits", "microusd"),
    [
        ("predictleads_company_job_openings", "0.56", 56_000),
        ("harvestapi_get_post", "0.03", 3_000),
        ("harvestapi_get_job", "0.01", 1_000),
        ("exa_answer", "0.07", 7_000),
        ("hunter_discover", "0", 0),
        ("generic_http_request", "0", 0),
    ],
)
def test_deepline_fixed_and_free_reservations(tool, credits, microusd):
    cost = deepline_reservation_cost({"tool": tool, "payload": {}})
    assert cost is not None
    assert cost.units == Decimal(credits)
    assert cost.microusd == microusd


@pytest.mark.parametrize("tool", ["exa_search", "exa_contents", "firecrawl_scrape", "twitterapi_tweets_by_ids", "unknown"])
def test_deepline_dynamic_reservations_are_not_guessed(tool):
    assert deepline_reservation_cost({"tool": tool, "payload": {}}) is None


@pytest.mark.parametrize("value", [True, -1, "-0.1", "NaN", "Infinity", "-Infinity", "", None, {}, []])
def test_deepline_rejects_invalid_credit_values(value):
    assert deepline_cost({"billing": {"credits_charged": value}}) is None


@pytest.mark.parametrize(
    "response",
    [None, {}, {"billing": None}, {"credits_charged": 1}, {"result": {"billing": {"credits_charged": 1}}}, {"billing": {"cost_usd": 1}}],
)
def test_deepline_does_not_walk_untrusted_or_usd_fields(response):
    assert deepline_cost(response) is None


@pytest.mark.parametrize(
    ("value", "microusd"),
    [(0, 0), ("0", 0), ("0.0000001", 1), (Decimal("1.23456789"), 1_234_568), (50.5, 50_500_000)],
)
def test_openrouter_exact_usage_cost_ceilings(value, microusd):
    cost = openrouter_cost({"usage": {"cost": value}})
    assert cost is not None
    assert cost.units == Decimal(str(value))
    assert cost.unit_name == "usd"
    assert cost.microusd == microusd


@pytest.mark.parametrize("value", [True, -1, "-0.1", "NaN", "Infinity", "-Infinity", "", None, {}, []])
def test_openrouter_rejects_invalid_usage_cost(value):
    assert openrouter_cost({"usage": {"cost": value}}) is None


@pytest.mark.parametrize("response", [None, {}, {"usage": None}, {"cost": 1}, {"usage": {}}])
def test_openrouter_requires_top_level_usage_cost(response):
    assert openrouter_cost(response) is None
