"""Request-local catalog work must preserve policy and exact broker outputs."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest

from lab_arena import deepline_catalog as catalog
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker


def execute(broker, snapshot, **changes):
    return broker.execute(replace(CONTEXT, deepline_catalog=snapshot, **changes),
        operation_id="deepline.execute",
        parameters={"tool": "vendor_company_search", "payload": {"query": "solar"}},
        action_sequence=0, timeout_ms=1000)


def count_full_encodes(monkeypatch):
    original = catalog._encoded
    calls = []
    def encoded(value):
        if isinstance(value, dict) and value.get("schema_version") == catalog.SCHEMA_VERSION:
            calls.append(value)
        return original(value)
    monkeypatch.setattr(catalog, "_encoded", encoded)
    return calls


def test_broker_serializes_frozen_catalog_once_and_preserves_wire_and_billing(monkeypatch):
    snapshot = frozen()
    catalog.validate_catalog(snapshot)  # Existing exact-byte validation cache is warm.
    calls = count_full_encodes(monkeypatch)
    funding = []
    def payer(context, provider):
        funding.append((provider, catalog.tool_entry(context.deepline_catalog,
            "vendor_company_search")["pricing"]["usd_per_unit"]))
        return "host"
    broker, store, transport = make_broker(provider_funding_source_for=payer,
        transport=FakeTransport([(200, {"result": {"companies": []},
            "billing": {"credits_charged": .02}})]))
    result = execute(broker, snapshot)
    assert result.status == 200 and result.call["actual_microusd"] == 2000
    assert store.log == ["reserve", "dispatch", "settle"]
    assert funding == [("deepline", .02)]
    assert transport.sent[0]["body"] == b'{"operation":"vendor_company_search","payload":{"query":"solar"},"provider":"vendor"}'
    assert json.loads(result.body)["result"] == {"companies": []}
    assert len(calls) == 1


def test_broker_prepared_catalog_does_not_hide_later_raw_mutation(monkeypatch):
    snapshot = frozen()
    broker, store, transport = make_broker()
    assert execute(broker, snapshot).status == 200
    before = (copy.deepcopy(store.calls), len(transport.sent))
    snapshot["tools"][0]["pricing"]["usd_per_unit"] = 99
    result = execute(broker, snapshot)
    assert result.status == 400 and json.loads(result.body)["error"]["code"] == "invalid_request"
    assert (store.calls, len(transport.sent)) == before


def test_prepared_snapshot_owns_exact_bytes_and_returns_only_copies():
    snapshot = frozen()
    original = copy.deepcopy(snapshot)
    prepared = catalog.prepare_catalog(snapshot)
    assert prepared == original
    snapshot["tools"][0]["provider"] = "hubspot"
    for returned in (prepared["tools"], dict(prepared)["tools"],
                     catalog.validate_catalog(prepared)["tools"]):
        returned[0]["provider"] = "hubspot"
    entry = catalog.tool_entry(prepared, "vendor_company_search")
    entry["input_schema"]["properties"]["query"]["type"] = "integer"
    assert catalog.validate_catalog(prepared) == original
    assert catalog.validate_payload(prepared, "vendor_company_search", {"query": "solar"}) == {"query": "solar"}
    with pytest.raises((FrozenInstanceError, AttributeError)):
        prepared.encoded = "{}"
    with pytest.raises(TypeError):
        prepared["allow_people"] = True
    with pytest.raises(catalog.CatalogError):
        catalog.allowed_tool_ids(snapshot)


@pytest.mark.parametrize("change", ["hash", "policy", "schema", "provider", "people", "missing", "extra"])
def test_prepared_constructor_cannot_grant_mutated_authority(change):
    snapshot = frozen()
    if change == "hash": snapshot["catalog_hash"] = "0" * 64
    elif change == "policy": snapshot["policy_version"] = "future-policy"
    elif change == "schema": snapshot["tools"][0]["input_schema"]["properties"]["query"] = {"$ref": "https://internal.example/secret"}
    elif change == "provider": snapshot["tools"][0]["provider"] = "hubspot"
    elif change == "people": snapshot["allow_people"] = "false"
    elif change == "missing": del snapshot["tools"][0]["pricing"]
    elif change == "extra": snapshot["unchecked"] = True
    if change != "hash": snapshot["catalog_hash"] = catalog._hash(snapshot)
    # Direct construction still passes through the existing complete validator.
    with pytest.raises(catalog.CatalogError):
        catalog._PreparedCatalog(catalog._encoded(snapshot))


def test_prepared_catalogs_remain_independent_across_rounds_and_cache_eviction():
    snapshots = [frozen(row(pricing={"unit": "call", "usdPerUnit": .01 * i,
        "creditsPerUnit": i, "currency": "USD"})) for i in range(1, 7)]
    prepared = [catalog.prepare_catalog(snapshot) for snapshot in snapshots]
    def use(index):
        value = prepared[index % len(prepared)]
        assert catalog.allowed_tool_ids(value) == ("vendor_company_search",)
        entry = catalog.tool_entry(value, "vendor_company_search")
        price = entry["pricing"]["usd_per_unit"]
        entry["pricing"]["usd_per_unit"] = 999
        assert catalog.validate_catalog(value) == snapshots[index % len(prepared)]
        return price
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(use, range(48))) == [.01 * (i % 6 + 1) for i in range(48)]


def test_unrelated_provider_keeps_prior_invalid_catalog_behavior():
    broker, _, _ = make_broker()
    result = broker.execute(replace(CONTEXT, deepline_catalog={"fixture": True}),
        operation_id="scrapingdog.scrape", parameters={"url": "https://example.com"},
        action_sequence=0, timeout_ms=1000)
    assert result.status == 200


@pytest.mark.parametrize("stale,status,code", [(True, 409, "lease_stale"),
    (False, 503, "broker_unavailable")])
@pytest.mark.parametrize("operation", ["deepline.tools.list", "deepline.tools.search"])
def test_discovery_preserves_lease_and_invalid_catalog_failure_order(stale, status, code, operation):
    class Store:
        def run_quota_snapshot(self, *_args):
            return {"status": "stale" if stale else "active"}
    broker, _, transport = make_broker(store=Store())
    result = broker.execute(replace(CONTEXT, deepline_catalog={"invalid": True}),
        operation_id=operation, parameters={"query": "solar"} if operation.endswith("search") else {},
        action_sequence=0, timeout_ms=1000)
    assert result.status == status
    assert json.loads(result.body) == {"error": {"code": code}}
    assert result.call == {"operation_id": operation, "provider": "deepline",
        "action_sequence": 0, "outcome": "local_catalog", "actual_microusd": 0,
        "error_code": code}
    assert transport.sent == []


def test_prepared_constructor_rejects_mutable_or_malformed_encoded_data():
    for value in ({}, [], None, b"{}", "{", "[]", "null"):
        with pytest.raises(catalog.CatalogError):
            catalog._PreparedCatalog(value)


def test_prepared_people_policy_is_frozen_independently_of_returned_values():
    people = row("vendor_people_search", categories=["people_search"],
        inputSchema={"type": "object", "properties": {"fullName": {"type": "string"}},
                     "required": ["fullName"]})
    allowed = catalog.prepare_catalog(frozen(people, allow_people=True))
    denied = catalog.prepare_catalog(frozen())
    assert catalog.validate_payload(allowed, "vendor_people_search", {"fullName": "Test Person"}) == {"fullName": "Test Person"}
    with pytest.raises(catalog.CatalogError):
        catalog.validate_payload(denied, "vendor_people_search", {"fullName": "Test Person"})
    assert allowed["allow_people"] is True and denied["allow_people"] is False
