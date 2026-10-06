"""Exact billing proofs never assign a shared or provisional charge to a call."""

from copy import deepcopy
from decimal import Decimal

import pytest

from lab_arena import provider_costs as costs


REQUEST_ID = "iad1::exact-1791234567000-abcdef123456"
TOOL = "firecrawl_scrape"


def exact_document(**patch):
    row = {
        "id": "usage-1", "request_id": REQUEST_ID, "provider": "firecrawl",
        "operation": TOOL, "status": "completed", "outcome": "unknown",
        "charge_state": "posted", "charge_finality": "final",
        "billing_mode": "deduct_on_settle", "credits": "0.12345678901234567",
        "delta": "-0.12345678901234567", "batch_count": None, "metadata": {},
    }
    row.update(patch)
    return {"org_id": "organization-1", "recent": {"request_id": REQUEST_ID, "entries": [row]}}


def read(document):
    return costs.deepline_exact_request_cost(document, request_id=REQUEST_ID, operation=TOOL)


def test_exact_final_amount_uses_decimal_and_rounds_charge_up():
    document = exact_document()
    saved = deepcopy(document)
    state, price = read(document)
    assert state == "matched"
    assert price.microusd == 12_346
    assert price.units == Decimal("0.12345678901234567")
    assert document == saved


@pytest.mark.parametrize("state", ["free", "hold_release", "failed", "posted"])
def test_final_zero_and_billed_failure_use_the_final_charge(state):
    status, price = read(exact_document(charge_state=state, status="error", credits=0, delta=0))
    assert status == "matched" and price.microusd == 0
    status, price = read(exact_document(status="error", credits="0.5", delta="-0.5"))
    assert status == "matched" and price.microusd == 50_000


@pytest.mark.parametrize("patch", [
    {"request_id": "another-request"}, {"operation": "firecrawl_search"},
    {"provider": "exa"}, {"id": None}, {"id": ""},
    {"batch_count": 2}, {"batch_count": True},
    {"metadata": {"chargeGroupIds": [REQUEST_ID, "another-request"]}},
    {"metadata": {"chargeGroupIds": "group"}}, {"metadata": "group"},
    {"credits": True}, {"credits": -1}, {"credits": "NaN"},
    {"delta": None}, {"delta": True}, {"delta": "0.12345678901234567"},
    {"charge_state": "free"}, {"charge_state": "hold_release"},
    {"charge_state": "failed"}, {"charge_state": "unknown"},
    {"billing_mode": "no_bill"},
])
def test_unbound_or_inconsistent_charge_is_invalid(patch):
    assert read(exact_document(**patch)) == ("invalid", None)


@pytest.mark.parametrize("patch", [
    {"charge_finality": "pending"}, {"charge_finality": "unknown"},
    {"charge_finality": None}, {"charge_state": "temporary_hold"},
    {"charge_state": "pending"},
    {"billing_mode": "async_hold", "credits": 0, "delta": 0},
])
def test_nonfinal_and_asynchronous_zero_are_not_zero_cost(patch):
    assert read(exact_document(**patch)) == ("nonterminal", None)


def test_missing_duplicate_and_wrong_lookup_do_not_settle():
    empty = exact_document()
    empty["recent"]["entries"] = []
    assert read(empty) == ("pending", None)
    duplicate = exact_document()
    duplicate["recent"]["entries"] *= 2
    assert read(duplicate) == ("invalid", None)
    wrong_lookup = exact_document()
    wrong_lookup["recent"]["request_id"] = "another-request"
    assert read(wrong_lookup) == ("invalid", None)
    for malformed in (None, {}, {"recent": {}}, {"recent": {"request_id": REQUEST_ID, "entries": "rows"}}):
        assert read(malformed) == ("invalid", None)


def test_catalog_provider_has_to_match_billing_provider():
    assert costs.deepline_exact_request_cost(exact_document(), request_id=REQUEST_ID,
        operation=TOOL, provider="exa") == ("invalid", None)


def test_catalog_call_estimate_uses_usd_instead_of_credit_conversion():
    entry = {"tool_id": TOOL, "pricing": {"unit": "call", "currency": "USD",
        "usd_per_unit": 0.04, "credits_per_unit": 0.01, "billing_source": "provider"}}
    assert costs.deepline_reservation_cost({"tool": TOOL}, catalog_entry=entry).microusd == 40_000
    entry["pricing"]["unit"] = "usage"
    assert costs.deepline_reservation_cost({"tool": TOOL}, catalog_entry=entry) is None


def test_frozen_zero_price_is_only_free_with_explicit_source_and_completed_receipt():
    entry = {"tool_id": "custom_company_search", "pricing": {"unit": "call", "currency": "USD",
        "usd_per_unit": 0, "credits_per_unit": 0, "billing_source": "free"}}
    response = {"job_id": REQUEST_ID, "status": "completed", "result": {"data": []}}
    assert costs.deepline_free_completed_cost({"tool": entry["tool_id"]}, 200, response,
        catalog_entry=entry).microusd == 0
    entry["pricing"]["billing_source"] = None
    assert costs.deepline_free_completed_cost({"tool": entry["tool_id"]}, 200, response,
        catalog_entry=entry) is None
