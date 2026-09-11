from decimal import Decimal

import pytest

from lab_arena.provider_costs import (
    deepline_billing_history_cost,
    deepline_cost,
    deepline_free_completed_cost,
    deepline_reservation_cost,
    openrouter_cost,
    openrouter_generation_cost,
    openrouter_insured_error_cost,
    scrapingdog_cost,
)


def test_deepline_history_exact_terminal_match_ignores_absent_pagination_fields():
    state, cost, has_more, next_offset = deepline_billing_history_cost(
        {"recent": {"entries": [{"request_id": "job-1", "operation": "exa_search",
                                  "provider": "exa", "charge_state": "posted", "credits": 0.14}]}},
        request_id="job-1", operation="exa_search",
    )
    assert state == "matched" and cost is not None and cost.microusd == 14_000
    assert has_more is False and next_offset is None


def test_deepline_history_accepts_strict_failed_error_zero_charge():
    state, cost, has_more, next_offset = deepline_billing_history_cost(
        {"recent": {"entries": [{
            "request_id": "job-error", "operation": "exa_search",
            "provider": "exa", "charge_state": "failed", "status": "error",
            "credits": 0, "delta": 0,
        }]}},
        request_id="job-error", operation="exa_search",
    )
    assert state == "matched" and cost is not None and cost.microusd == 0
    assert cost.price_basis == "deepline_billing_history_failed_zero"
    assert has_more is False and next_offset is None


@pytest.mark.parametrize(
    "patch",
    [
        {"credits": 0.1},
        {"delta": 0.1},
        {"delta": -0.1},
        {"status": "completed"},
        {"delta": None},
    ],
)
def test_deepline_history_rejects_nonzero_or_malformed_failed_charge(patch):
    entry = {
        "request_id": "job-error", "operation": "exa_search",
        "provider": "exa", "charge_state": "failed", "status": "error",
        "credits": 0, "delta": 0,
    }
    entry.update(patch)
    assert deepline_billing_history_cost(
        {"recent": {"entries": [entry]}},
        request_id="job-error", operation="exa_search",
    ) == ("invalid", None, False, None)


def test_deepline_history_matches_exact_charge_group_alias():
    state, cost, has_more, next_offset = deepline_billing_history_cost(
        {
            "recent": {
                "entries": [
                    {
                        "request_id": "internal-job",
                        "operation": "exa_search",
                        "provider": "exa",
                        "charge_state": "posted",
                        "credits": 0.2,
                        "metadata": {"chargeGroupIds": ["wrapper-job"]},
                    }
                ]
            }
        },
        request_id="wrapper-job",
        operation="exa_search",
    )
    assert state == "matched" and cost is not None and cost.microusd == 20_000
    assert has_more is False and next_offset is None


@pytest.mark.parametrize(
    "charge_group_ids",
    [
        "wrapper-job",
        {"wrapper-job": True},
        ["wrapper-job", "wrapper-job"],
        ["wrapper-job", ""],
        ["wrapper-job", True],
        ["x" * 513, "wrapper-job"],
        ["wrapper-job", *["group-%d" % index for index in range(128)]],
    ],
)
def test_deepline_history_rejects_malformed_charge_groups(charge_group_ids):
    assert deepline_billing_history_cost(
        {
            "recent": {
                "entries": [
                    {
                        "request_id": "internal-job",
                        "operation": "exa_search",
                        "provider": "exa",
                        "charge_state": "posted",
                        "credits": 0.2,
                        "metadata": {"chargeGroupIds": charge_group_ids},
                    }
                ],
                "has_more": False,
            }
        },
        request_id="wrapper-job",
        operation="exa_search",
    ) == ("invalid", None, False, None)


def test_deepline_history_rejects_direct_and_charge_group_ambiguity():
    entries = [
        {
            "request_id": "wrapper-job",
            "operation": "exa_search",
            "provider": "exa",
            "charge_state": "posted",
            "credits": 0.2,
        },
        {
            "request_id": "internal-job",
            "operation": "exa_search",
            "provider": "exa",
            "charge_state": "posted",
            "credits": 0.2,
            "metadata": {"chargeGroupIds": ["wrapper-job"]},
        },
    ]
    assert deepline_billing_history_cost(
        {"recent": {"entries": entries}},
        request_id="wrapper-job",
        operation="exa_search",
    ) == ("invalid", None, False, None)


def test_deepline_history_charge_group_keeps_operation_binding():
    entry = {
        "request_id": "internal-job",
        "operation": "exa_contents",
        "provider": "exa",
        "charge_state": "posted",
        "credits": 0.2,
        "metadata": {"chargeGroupIds": ["wrapper-job"]},
    }
    assert deepline_billing_history_cost(
        {"recent": {"entries": [entry]}},
        request_id="wrapper-job",
        operation="exa_search",
    ) == ("invalid", None, False, None)


def test_deepline_history_rejects_shared_charge_group_cost():
    entry = {
        "request_id": "internal-job",
        "operation": "exa_search",
        "provider": "exa",
        "charge_state": "posted",
        "credits": 0.2,
        "metadata": {"chargeGroupIds": ["internal-job", "wrapper-job"]},
    }
    assert deepline_billing_history_cost(
        {"recent": {"entries": [entry]}},
        request_id="wrapper-job",
        operation="exa_search",
    ) == ("invalid", None, False, None)


def test_deepline_history_direct_singleton_group_counts_once():
    entry = {
        "request_id": "wrapper-job",
        "operation": "exa_search",
        "provider": "exa",
        "charge_state": "posted",
        "credits": 0.2,
        "metadata": {"chargeGroupIds": ["wrapper-job"]},
    }
    state, cost, has_more, next_offset = deepline_billing_history_cost(
        {"recent": {"entries": [entry]}},
        request_id="wrapper-job",
        operation="exa_search",
    )
    assert state == "matched" and cost is not None and cost.microusd == 20_000
    assert has_more is False and next_offset is None


@pytest.mark.parametrize(
    ("tool", "basis"),
    [
        (
            "free_simple_company_search",
            "deepline_free_simple_company_search_completed_zero",
        ),
        ("hunter_discover", "deepline_hunter_discover_completed_zero"),
    ],
)
def test_deepline_verified_no_bill_completed_tools_settle_zero(tool, basis):
    cost = deepline_free_completed_cost(
        {"tool": tool},
        200,
        {"job_id": "wrapper-job", "status": "completed", "result": []},
    )
    assert cost is not None
    assert cost.microusd == 0 and cost.price_basis == basis


def test_deepline_hunter_no_bill_structured_error_settles_zero_without_job_id():
    cost = deepline_free_completed_cost(
        {"tool": "hunter_discover"},
        502,
        {"error": {"code": "upstream_error"}},
    )
    assert cost is not None
    assert cost.microusd == 0
    assert cost.price_basis == "deepline_hunter_discover_error_zero"


@pytest.mark.parametrize(
    ("tool", "response"),
    [
        ("exa_search", {"error": {"code": "upstream_error"}}),
        ("hunter_discover", {"error": {"code": "upstream_error"}, "billing": None}),
        (
            "hunter_discover",
            {"error": {"code": "upstream_error"}, "billing": {"credits": "invalid"}},
        ),
    ],
)
def test_deepline_error_zero_proof_rejects_nonfree_or_present_billing(tool, response):
    assert deepline_free_completed_cost({"tool": tool}, 502, response) is None


@pytest.mark.parametrize(
    "response_status,response",
    [
        (399, {"job_id": "wrapper-job", "status": "completed", "result": []}),
        (200, {"status": "completed", "result": []}),
        (200, {"job_id": "wrapper-job", "status": "failed", "result": []}),
        (200, {"job_id": "wrapper-job", "status": "completed"}),
        (
            200,
            {
                "job_id": "wrapper-job",
                "status": "completed",
                "result": [],
                "billing": None,
            },
        ),
    ],
)
def test_deepline_hunter_zero_proof_fails_closed(response_status, response):
    assert (
        deepline_free_completed_cost(
            {"tool": "hunter_discover"}, response_status, response
        )
        is None
    )


@pytest.mark.parametrize("next_offset", [None, True, False, 0, -1, 50.0, "50", 1_000_001])
def test_deepline_history_rejects_invalid_or_nonadvancing_offset(next_offset):
    state, cost, has_more, offset = deepline_billing_history_cost(
        {"recent": {"entries": [], "has_more": True, "next_offset": next_offset}},
        request_id="job-1", operation="exa_search", current_offset=0,
    )
    assert (state, cost, has_more, offset) == ("invalid", None, False, None)


def test_deepline_history_returns_forward_offset_for_missing_job():
    assert deepline_billing_history_cost(
        {"recent": {"entries": [], "has_more": True, "next_offset": 100}},
        request_id="job-1", operation="exa_search", current_offset=50,
    ) == ("pending", None, True, 100)


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


def test_openrouter_generation_cost_requires_one_matching_exact_charge():
    cost = openrouter_generation_cost(
        {"data": {"id": "gen-1", "total_cost": "0.0012", "usage": 0.0012}},
        generation_id="gen-1",
    )
    assert cost is not None and cost.microusd == 1200
    assert cost.price_basis == "openrouter_generation_cost"
    for response in (
        {"data": {"id": "other", "total_cost": "0.0012"}},
        {"data": {"id": "gen-1"}},
        {"data": {"id": "gen-1", "total_cost": "invalid"}},
        {"data": {"id": "gen-1", "total_cost": "0.0012", "usage": "0.0013"}},
        {"data": {"id": "gen-1", "total_cost": True}},
        {"data": {"id": "gen-1", "total_cost": "0.0012"}, "error": {"code": 502}},
    ):
        assert openrouter_generation_cost(response, generation_id="gen-1") is None


def _insured_openrouter_error(**metadata_patch):
    metadata = {
        "requested": "openai/gpt-4o-mini",
        "is_byok": False,
        "attempt": 1,
        "attempts": [{"provider": "OpenAI", "status": 502}],
    }
    metadata.update(metadata_patch)
    return {
        "error": {"code": 502, "message": "Provider returned an error"},
        "openrouter_metadata": metadata,
        "user_id": "documented-harmless-field",
    }


def _plain_openrouter_pricing(**patch):
    pricing = {"request": "0", "image": "0", "web_search": "0"}
    pricing.update(patch)
    return pricing


def test_openrouter_insured_error_proves_only_plain_non_byok_502_zero():
    parameters = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }
    cost = openrouter_insured_error_cost(
        parameters, _plain_openrouter_pricing(), 502, _insured_openrouter_error()
    )
    assert cost is not None and cost.microusd == 0
    assert cost.price_basis == "openrouter_zero_completion_insurance_error_20260911"
    reasoning_parameters = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": True},
        "include_reasoning": True,
    }
    reasoning_cost = openrouter_insured_error_cost(
        reasoning_parameters,
        _plain_openrouter_pricing(),
        502,
        _insured_openrouter_error(),
    )
    assert reasoning_cost is not None and reasoning_cost.microusd == 0


@pytest.mark.parametrize(
    ("parameters", "status", "response"),
    [
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 500, _insured_openrouter_error()),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502.0, _insured_openrouter_error()),
        ({"model": "openai/gpt-4o-mini:online", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(requested="openai/gpt-4o-mini:online")),
        ({"model": "perplexity/sonar-pro", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(requested="perplexity/sonar-pro")),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}], "plugins": []}, 502, _insured_openrouter_error()),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": ["multimodal"]}]}, 502, _insured_openrouter_error()),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(is_byok=True)),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(pipeline=[{"type": "plugin", "name": "web-search"}])),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(attempts=[{"status": 200}])),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(attempts=[{"status": -1}])),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(attempts=[{"status": 0}])),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(attempts=[{"status": 699}])),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, {**_insured_openrouter_error(), "usage": None}),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, {**_insured_openrouter_error(), "id": "gen-1"}),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, {**_insured_openrouter_error(), "choices": []}),
        ({"model": "openai/gpt-4o-mini", "messages": [{"content": "x"}]}, 502, {**_insured_openrouter_error(), "error": {"code": 503, "message": "wrong"}}),
        ({"model": "unknown/plain-model", "messages": [{"content": "x"}]}, 502, _insured_openrouter_error(requested="unknown/plain-model")),
    ],
)
def test_openrouter_insured_error_rejects_unproven_or_auxiliary_costs(
    parameters, status, response
):
    assert openrouter_insured_error_cost(
        parameters, _plain_openrouter_pricing(), status, response
    ) is None


@pytest.mark.parametrize(
    "pricing",
    [
        {"image": "0", "web_search": "0"},
        _plain_openrouter_pricing(request="0.01"),
        _plain_openrouter_pricing(request="NaN"),
        _plain_openrouter_pricing(request=True),
    ],
)
def test_openrouter_insured_error_requires_zero_valid_request_price(pricing):
    parameters = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert openrouter_insured_error_cost(
        parameters, pricing, 502, _insured_openrouter_error()
    ) is None


@pytest.mark.parametrize(
    "model", ("anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-5")
)
def test_openrouter_insured_error_allows_optional_sonnet_prices_when_unused(model):
    parameters = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }
    pricing = _plain_openrouter_pricing(image="0.000001", web_search="0.01")
    response = _insured_openrouter_error(requested=model)

    cost = openrouter_insured_error_cost(parameters, pricing, 502, response)

    assert cost is not None and cost.microusd == 0
    assert openrouter_insured_error_cost(
        parameters,
        pricing,
        502,
        _insured_openrouter_error(
            requested=model,
            pipeline=[{"type": "plugin", "name": "web-search"}],
        ),
    ) is None
