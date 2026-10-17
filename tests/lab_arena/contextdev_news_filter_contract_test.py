"""The frozen ContextDev news contract rejects incompatible result filters."""

import copy
import itertools
import json
from dataclasses import replace

import pytest

from lab_arena import deepline_catalog, operations
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker


EXCLUSIVE = (
    "sourceDomain", "sourceCountry", "articleLanguage", "articleType",
)
VALUES = {
    "sourceDomain": ["example.com"],
    "sourceCountry": ["us"],
    "articleLanguage": ["en"],
    "articleType": ["editorial"],
}
SEARCH = {"type": "entity", "entity": {"type": "domain", "domain": "example.com"}}


def _row(tool="contextdev_post_news_search", provider="contextdev"):
    return {
        "toolId": tool,
        "provider": provider,
        "categories": ["research"],
        "description": "Search recent news.",
        "connected": True,
        "credentialStatus": "managed",
        "billingSource": "deepline_credits",
        "pricing": {"unit": "call", "usdPerUnit": 0, "creditsPerUnit": 0, "currency": "USD"},
        "inputSchema": {
            "type": "object",
            "required": ["searchBy"],
            "additionalProperties": False,
            "properties": {
                "searchBy": {
                    "type": "object", "required": ["type", "entity"],
                    "properties": {
                        "type": {"const": "entity"},
                        "entity": {"type": "object", "required": ["type", "domain"],
                                   "properties": {"type": {"const": "domain"},
                                                  "domain": {"type": "string"}}},
                    },
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                "filterBy": {
                    "type": "object", "description": "Optional result filters.",
                    "additionalProperties": False,
                    "properties": {
                        "date": {"type": "object", "properties": {"from": {"type": "integer"}}},
                        "sourceDomain": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "sourceCountry": {"type": "array", "items": {"enum": ["us", "gb"]}, "minItems": 1},
                        "articleLanguage": {"type": "array", "items": {"enum": ["en", "fr"]}, "minItems": 1},
                        "articleType": {"type": "array", "items": {"enum": ["editorial"]}, "minItems": 1},
                    },
                },
            },
        },
    }


def _catalog_rows(*rows):
    sources = list(rows)
    return deepline_catalog.fetch_catalog(
        read_json=lambda url, headers: {"tools": sources}
        if url == deepline_catalog.CATALOG_URL else next(
            source for source in sources if source["toolId"] == url.rsplit("/", 2)[-2]
        ),
    )


def _catalog(row=None):
    return _catalog_rows(row or _row())


def _payload(filters=None):
    value = {"searchBy": copy.deepcopy(SEARCH)}
    if filters is not None:
        value["filterBy"] = copy.deepcopy(filters)
    return value


def test_freeze_advertises_complete_rule_without_mutating_provider_metadata():
    source = _row()
    before = copy.deepcopy(source)
    first = _catalog(source)
    second = _catalog(source)
    assert source == before
    assert first == second
    assert first["catalog_hash"] == deepline_catalog._hash(first)
    assert deepline_catalog.validate_catalog(first) == first
    entry = deepline_catalog.tool_entry(first, source["toolId"])
    frozen_filter = entry["input_schema"]["properties"]["filterBy"]
    assert "at most one" in frozen_filter["description"]
    assert "date may be combined" in frozen_filter["description"]
    assert len(frozen_filter["allOf"][0]["not"]["anyOf"]) == 6
    public = deepline_catalog.public_tool_definition(entry)
    assert public["inputSchema"]["jsonSchema"] == entry["input_schema"]
    assert entry["input_schema"]["properties"]["limit"] == before["inputSchema"]["properties"]["limit"]


@pytest.mark.parametrize("filters", [
    None, {}, {"date": {"from": 0}},
    *({name: value} for name, value in VALUES.items()),
    *({"date": {"from": 0}, name: value} for name, value in VALUES.items()),
])
def test_absent_empty_date_and_one_named_filter_are_valid(filters):
    snapshot = _catalog()
    payload = _payload(filters)
    assert deepline_catalog.validate_payload(snapshot, "contextdev_post_news_search", payload) == payload
    request = operations.build_outbound_request(
        "deepline.execute",
        {"tool": "contextdev_post_news_search", "payload": payload},
        deepline_catalog=snapshot,
    )
    assert json.loads(request.body) == {
        "provider": "contextdev", "operation": "contextdev_post_news_search", "payload": payload,
    }


@pytest.mark.parametrize("names", [
    *itertools.combinations(EXCLUSIVE, 2),
    *itertools.combinations(EXCLUSIVE, 3),
    EXCLUSIVE,
])
def test_every_filter_pair_and_larger_combination_fails_before_outbound(names):
    snapshot = _catalog()
    filters = {name: VALUES[name] for name in names}
    filters["date"] = {"from": 0}
    payload = _payload(filters)
    with pytest.raises(deepline_catalog.CatalogError, match="invalid_payload_schema"):
        deepline_catalog.validate_payload(snapshot, "contextdev_post_news_search", payload)
    with pytest.raises(operations.OperationRequestError, match="invalid_field"):
        operations.build_outbound_request(
            "deepline.execute",
            {"tool": "contextdev_post_news_search", "payload": payload},
            deepline_catalog=snapshot,
        )


def test_original_provider_constraints_and_other_tools_stay_unchanged():
    snapshot = _catalog()
    for payload in (
        {**_payload(), "limit": 6},
        _payload({"sourceCountry": ["xx"]}),
        _payload({"articleLanguage": []}),
    ):
        with pytest.raises(deepline_catalog.CatalogError):
            deepline_catalog.validate_payload(snapshot, "contextdev_post_news_search", payload)
    for source in (
        _row(provider="another_provider"),
        _row(tool="contextdev_post_web_search"),
    ):
        frozen = _catalog(source)
        assert deepline_catalog.tool_entry(frozen, source["toolId"])["input_schema"] == source["inputSchema"]
        assert deepline_catalog.validate_payload(
            frozen, source["toolId"],
            _payload({"sourceCountry": ["us"], "articleLanguage": ["en"]}),
        )


def test_existing_upstream_filter_constraint_is_preserved_and_correction_is_idempotent():
    source = _row()
    upstream_clause = {"properties": {"sourceDomain": {"maxItems": 1}}}
    source["inputSchema"]["properties"]["filterBy"]["allOf"] = [upstream_clause]
    first = _catalog(source)
    constraints = deepline_catalog.tool_entry(first, source["toolId"])["input_schema"]["properties"]["filterBy"]["allOf"]
    assert constraints[0] == upstream_clause
    assert len(constraints) == 2
    advertised = copy.deepcopy(source)
    advertised["inputSchema"] = deepline_catalog.tool_entry(first, source["toolId"])["input_schema"]
    assert _catalog(advertised) == first


@pytest.mark.parametrize("ineligible", [
    {"connected": False}, {"categories": ["admin"]},
])
def test_ineligible_news_metadata_does_not_change_other_tools(ineligible):
    news = _row()
    news.update(ineligible)
    news["inputSchema"]["properties"].pop("filterBy")
    other = _row(tool="contextdev_post_web_search")
    snapshot = _catalog_rows(news, other)
    assert deepline_catalog.allowed_tool_ids(snapshot) == (other["toolId"],)
    assert snapshot == deepline_catalog.freeze_catalog({"tools": [news, other]})


@pytest.mark.parametrize("broken_filter", [
    None, {"type": "string"}, {"type": "not-a-type"},
    {"type": "object", "properties": {"sourceCountry": {"type": "array"}}},
])
def test_unsupported_news_schema_drops_only_that_tool(broken_filter):
    news = _row()
    if broken_filter is None:
        news["inputSchema"]["properties"].pop("filterBy")
    else:
        news["inputSchema"]["properties"]["filterBy"] = broken_filter
    other = _row(tool="contextdev_post_web_search")
    snapshot = _catalog_rows(news, other)
    assert deepline_catalog.allowed_tool_ids(snapshot) == (other["toolId"],)
    assert deepline_catalog.tool_entry(snapshot, other["toolId"])["input_schema"] == other["inputSchema"]


def test_broker_rejects_invalid_pair_without_transport_and_dispatches_valid_payload():
    snapshot = _catalog()
    context = replace(CONTEXT, deepline_catalog=snapshot)
    broker, store, transport = make_broker(transport=FakeTransport())
    invalid = _payload({"sourceCountry": ["us"], "articleLanguage": ["en"]})
    result = broker.execute(context, operation_id="deepline.execute",
                            parameters={"tool": "contextdev_post_news_search", "payload": invalid},
                            action_sequence=1, timeout_ms=5000)
    assert result.status == 400
    assert transport.sent == []
    assert store.calls == {}
    valid = _payload({"sourceCountry": ["us"]})
    accepted = broker.execute(context, operation_id="deepline.execute",
                              parameters={"tool": "contextdev_post_news_search", "payload": valid},
                              action_sequence=2, timeout_ms=5000)
    assert accepted.status == 200
    sent = [request for request in transport.sent if request["url"].endswith(
        "/integrations/contextdev_post_news_search/execute")]
    assert len(sent) == 1
    assert json.loads(sent[0]["body"])["payload"] == valid


def test_already_frozen_catalog_is_not_rewritten_on_validation():
    old = deepline_catalog.freeze_catalog({"tools": [_row()]})
    assert deepline_catalog.validate_catalog(old) == old
    assert old["catalog_hash"] != _catalog()["catalog_hash"]
    assert deepline_catalog.validate_payload(
        old, "contextdev_post_news_search",
        _payload({"sourceCountry": ["us"], "articleLanguage": ["en"]}),
    )
