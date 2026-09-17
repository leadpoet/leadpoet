from decimal import Decimal

import pytest

from lab_arena.provider_costs import (
    deepline_billing_ledger_cost,
    deepline_billing_history_cost,
    deepline_cost,
    deepline_free_completed_cost,
    deepline_payment_refusal_cost,
    deepline_reservation_cost,
    openrouter_cost,
    openrouter_generation_cost,
    openrouter_insured_error_cost,
    scrapingdog_cost,
)


def _deepline_ledger_entry(**patch):
    entry = {
        "id": "ledger-row-1",
        "delta": -0.02,
        "reason": "charge_settle",
        "provider": "firecrawl",
        "operation": "firecrawl_scrape",
        "request_id": "ctx-tool-0123456789abcdef0123456789abcdef",
        "billing_stage": "posted",
        "charge_state": "posted",
        "billing_mode": "post_deduct",
        "pricing_model": "per_page",
        "pricing_basis": "page",
        "charge_credits": 0.02,
        "metadata": {
            "requestId": "ctx-tool-0123456789abcdef0123456789abcdef",
            "chargeGroupId": "ctx-tool-0123456789abcdef0123456789abcdef",
            "operation": "firecrawl_scrape",
            "provider": "firecrawl",
            "billingStage": "posted",
            "billingMode": "post_deduct",
            "pricingModel": "per_page",
            "postedCredits": 0.02,
        },
        "billing_audit": {
            "request_id": "ctx-tool-0123456789abcdef0123456789abcdef",
            "charge_group_id": "ctx-tool-0123456789abcdef0123456789abcdef",
            "operation": "firecrawl_scrape",
            "provider": "firecrawl",
            "billing_stage": "posted",
            "charge_state": "posted",
            "billing_mode": "post_deduct",
            "pricing_model": "per_page",
            "pricing_basis": "page",
            "charge_credits": 0.02,
        },
    }
    entry.update(patch)
    return entry


def test_deepline_billing_ledger_matches_one_exact_posted_charge():
    state, cost, has_more, cursor = deepline_billing_ledger_cost(
        {"entries": [_deepline_ledger_entry()], "has_more": False},
        request_id="ctx-tool-0123456789abcdef0123456789abcdef",
        operation="firecrawl_scrape",
    )
    assert state == "matched" and cost is not None
    assert cost.microusd == 2_000 and cost.units == Decimal("0.02")
    assert cost.price_basis == "deepline_billing_ledger_charge_credits_x_0.10_usd"
    assert has_more is False and cursor is None


def test_deepline_billing_ledger_returns_one_forward_cursor_for_absent_id():
    assert deepline_billing_ledger_cost(
        {"entries": [], "has_more": True, "next_cursor": "cursor-2"},
        request_id="ctx-tool-0123456789abcdef0123456789abcdef",
        operation="firecrawl_scrape",
        current_cursor="cursor-1",
    ) == ("pending", None, True, "cursor-2")


@pytest.mark.parametrize(
    "document,current_cursor",
    [
        ({"entries": [], "has_more": True}, None),
        ({"entries": [], "has_more": True, "next_cursor": ""}, None),
        ({"entries": [], "has_more": True, "next_cursor": "cursor-1"}, "cursor-1"),
        ({"entries": [], "has_more": True, "next_cursor": "bad\nvalue"}, None),
        ({"entries": [], "has_more": "yes", "next_cursor": "cursor-2"}, None),
    ],
)
def test_deepline_billing_ledger_rejects_invalid_pagination(document, current_cursor):
    assert deepline_billing_ledger_cost(
        document,
        request_id="ctx-tool-0123456789abcdef0123456789abcdef",
        operation="firecrawl_scrape",
        current_cursor=current_cursor,
    ) == ("invalid", None, False, None)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(operation="exa_search"),
        lambda e: e.update(delta=-0.01),
        lambda e: e.update(delta=0.02),
        lambda e: e.update(reason="other"),
        lambda e: e.update(charge_state="pending"),
        lambda e: e.update(billing_stage="pending"),
        lambda e: e.update(metadata={**e["metadata"], "requestId": "other"}),
        lambda e: e.update(metadata={**e["metadata"], "postedCredits": 0.01}),
        lambda e: e.update(billing_audit={**e["billing_audit"], "charge_group_id": "other"}),
        lambda e: e.update(billing_audit={**e["billing_audit"], "charge_credits": 0.01}),
    ],
)
def test_deepline_billing_ledger_rejects_conflicting_or_nonterminal_charge(mutate):
    entry = _deepline_ledger_entry()
    mutate(entry)
    assert deepline_billing_ledger_cost(
        {"entries": [entry], "has_more": False},
        request_id="ctx-tool-0123456789abcdef0123456789abcdef",
        operation="firecrawl_scrape",
    ) == ("invalid", None, False, None)


def test_deepline_billing_ledger_rejects_duplicate_exact_request_rows():
    entry = _deepline_ledger_entry()
    assert deepline_billing_ledger_cost(
        {"entries": [entry, dict(entry)], "has_more": False},
        request_id="ctx-tool-0123456789abcdef0123456789abcdef",
        operation="firecrawl_scrape",
    ) == ("invalid", None, False, None)


def test_deepline_billing_ledger_accepts_exact_free_zero_only():
    entry = _deepline_ledger_entry(
        delta=0,
        reason="no_bill",
        charge_state="free",
        billing_stage="free",
        billing_mode="no_bill",
        charge_credits=0,
    )
    entry["metadata"].update(
        billingStage="free", billingMode="no_bill", postedCredits=0
    )
    entry["billing_audit"].update(
        billing_stage="free",
        charge_state="free",
        billing_mode="no_bill",
        charge_credits=0,
    )
    state, cost, _, _ = deepline_billing_ledger_cost(
        {"entries": [entry], "has_more": False},
        request_id="ctx-tool-0123456789abcdef0123456789abcdef",
        operation="firecrawl_scrape",
    )
    assert state == "matched" and cost is not None and cost.microusd == 0
    for field, value in (("delta", -0.01), ("charge_credits", 0.01)):
        changed = _deepline_ledger_entry(**{field: value})
        changed["charge_state"] = "free"
        assert deepline_billing_ledger_cost(
            {"entries": [changed], "has_more": False},
            request_id="ctx-tool-0123456789abcdef0123456789abcdef",
            operation="firecrawl_scrape",
        )[0] == "invalid"


def test_deepline_insufficient_credit_refusal_is_zero():
    cost = deepline_payment_refusal_cost(
        402,
        {
            "code": "INSUFFICIENT_CREDITS",
            "error": "Insufficient credits",
            "billing": {
                "kind": "insufficient_credits",
                "required_credits": 5,
                "balance_credits": 4.14,
                "needed_credits": 0.86,
            },
        },
    )
    assert cost is not None
    assert cost.microusd == 0
    assert cost.price_basis == "deepline_payment_required_error_zero"


@pytest.mark.parametrize(
    "status,response",
    [
        (200, {"error": {"code": "payment_required"}, "billing": None}),
        (402, {"error": {"code": "payment_required"}}),
        (402, {"code": "INSUFFICIENT_CREDITS", "error": "x", "billing": None}),
        (402, {"code": "INSUFFICIENT_CREDITS", "error": "x", "billing": {"kind": "insufficient_credits", "required_credits": 5, "balance_credits": 4.14, "needed_credits": 1}}),
        (402, {"error": {}, "billing": None}),
        (402, {"error": "payment_required", "billing": None}),
        (402, {"error": {"code": "payment_required"}, "billing": {}}),
        (
            402,
            {
                "error": {"code": "payment_required"},
                "billing": {"credits_charged": "invalid"},
            },
        ),
    ],
)
def test_deepline_payment_refusal_zero_proof_fails_closed(status, response):
    assert deepline_payment_refusal_cost(status, response) is None


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


@pytest.mark.parametrize(("charge_state", "credits", "expected_microusd"), [("free", 0, 0), ("posted", 0.2, 20_000)])
@pytest.mark.parametrize("exact_first", [False, True])
def test_deepline_history_accepts_exact_charge_beside_large_unrelated_group(
    charge_state, credits, expected_microusd, exact_first
):
    unrelated = {
        "request_id": "internal-unrelated",
        "operation": "exa_search",
        "provider": "exa",
        "charge_state": "posted",
        "credits": 0.2,
        "metadata": {
            "chargeGroupIds": ["unrelated-%d" % index for index in range(147)]
        },
    }
    exact = {
        "request_id": "wrapper-job",
        "operation": "exa_search",
        "provider": "exa",
        "charge_state": charge_state,
        "credits": credits,
    }

    state, cost, has_more, next_offset = deepline_billing_history_cost(
        {"recent": {"entries": [exact, unrelated] if exact_first else [unrelated, exact]}},
        request_id="wrapper-job",
        operation="exa_search",
    )

    assert state == "matched" and cost is not None and cost.microusd == expected_microusd
    assert has_more is False and next_offset is None


def test_deepline_history_rejects_target_inside_large_aggregate_group():
    charge_group_ids = ["unrelated-%d" % index for index in range(147)]
    charge_group_ids[146] = "wrapper-job"

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
                ]
            }
        },
        request_id="wrapper-job",
        operation="exa_search",
    ) == ("invalid", None, False, None)


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
        (
            "generic_http_request",
            "deepline_generic_http_request_completed_zero",
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
        ("generic_http_request", {"error": {"code": "upstream_error"}}),
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


@pytest.mark.parametrize("billing", [None, {"credits_charged": "invalid"}])
def test_deepline_generic_http_completed_zero_rejects_present_billing(billing):
    assert (
        deepline_free_completed_cost(
            {"tool": "generic_http_request"},
            200,
            {
                "job_id": "wrapper-job",
                "status": "completed",
                "result": {},
                "billing": billing,
            },
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
        ("harvestapi_get_company", "0.03", 3_000),
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
    "model",
    (
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.5",
        "openai/gpt-5.6-sol",
    ),
)
def test_openrouter_insured_error_allows_optional_model_prices_when_unused(model):
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
