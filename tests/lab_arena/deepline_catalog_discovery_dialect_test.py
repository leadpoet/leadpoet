"""Local discovery and the provider's unconstrained schema dialect stay bounded."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lab_arena import broker, deepline_catalog as catalog, operations
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, price_table


def answer_row():
    # Captured from official metadata on 2026-10-06. Store only public schema.
    schema = json.loads((Path(__file__).parent / "fixtures/deepline/exa_answer_input_schema.json").read_text())
    return row("exa_answer", provider="exa", categories=["research"], inputSchema=schema)


@pytest.mark.parametrize("payload", [
    {"query": "Public solar companies"},
    {"query": "Public solar companies", "outputSchema": {
        "type": "object", "properties": {"company_name": {"type": "string"}},
        "required": ["company_name"], "additionalProperties": False,
    }},
])
def test_exa_answer_optional_any_schema_is_valid_and_callable(payload):
    source = answer_row()
    unchanged = deepcopy(source)
    snapshot = frozen(source)
    assert catalog.allowed_tool_ids(snapshot) == ("exa_answer",)
    entry = catalog.tool_entry(snapshot, "exa_answer")
    Draft202012Validator.check_schema(entry["input_schema"])
    assert entry["input_schema"]["properties"]["outputSchema"] == {
        "description": source["inputSchema"]["jsonSchema"]["properties"]["outputSchema"]["description"]}
    assert snapshot == catalog.validate_catalog(json.loads(json.dumps(snapshot)))
    params = {"tool": "exa_answer", "payload": payload}
    assert operations.validate_operation_request("deepline.execute", params, deepline_catalog=snapshot) == params
    assert source == unchanged


def test_provider_fields_only_schema_normalizes_the_same_optional_any():
    source = answer_row()
    del source["inputSchema"]["jsonSchema"]
    snapshot = frozen(source)
    assert catalog.validate_payload(snapshot, "exa_answer", {"query": "companies", "outputSchema": {"type": "string"}})
    assert snapshot == catalog.validate_catalog(snapshot)


@pytest.mark.parametrize("nested", [
    {"type": "object", "properties": {"value": {"type": "any"}}},
    {"type": "array", "items": {"type": "any"}},
    {"anyOf": [{"type": "any", "enum": ["bounded"]}, {"type": "integer"}]},
    {"$defs": {"value": {"type": "any", "const": "bounded"}}, "$ref": "#/$defs/value"},
    {"type": "object", "additionalProperties": {"type": "any", "enum": [1]}},
])
def test_nested_any_normalizes_only_schema_nodes(nested):
    source = answer_row()
    source["inputSchema"]["jsonSchema"]["properties"]["outputSchema"] = nested
    snapshot = frozen(source)
    schema = catalog.tool_entry(snapshot, "exa_answer")["input_schema"]
    Draft202012Validator.check_schema(schema)
    assert snapshot == catalog.validate_catalog(snapshot)


def test_any_preserves_constraints_and_literal_annotation_data():
    source = answer_row()
    spec = {"type": "any", "enum": [{"type": "any"}],
            "default": {"type": "any"}, "examples": [{"type": "any"}]}
    source["inputSchema"]["jsonSchema"]["properties"]["outputSchema"] = spec
    snapshot = frozen(source)
    normalized = catalog.tool_entry(snapshot, "exa_answer")["input_schema"]["properties"]["outputSchema"]
    assert normalized == {key: value for key, value in spec.items() if key != "type"}
    assert catalog.validate_payload(snapshot, "exa_answer", {"query": "companies", "outputSchema": {"type": "any"}})
    with pytest.raises(catalog.CatalogError):
        catalog.validate_payload(snapshot, "exa_answer", {"query": "companies", "outputSchema": 1})


@pytest.mark.parametrize("invalid", [
    {"type": "anything"}, {"type": ["string", "any"]},
    {"type": "any", "minimum": "invalid"},
    {"type": "any", "$ref": "https://internal.example/schema"},
    {"type": "any", "$dynamicRef": "#/unsafe"},
    {"type": "object", "properties": {"nested": {"type": "invalid"}}},
])
def test_other_invalid_schemas_and_remote_references_still_reject(invalid):
    source = answer_row()
    source["inputSchema"]["jsonSchema"]["properties"]["outputSchema"] = invalid
    assert catalog.allowed_tool_ids(frozen(row("safe_company_search"), source)) == ("safe_company_search",)


@pytest.mark.parametrize("changes", [
    {"toolId": "exa_people_answer"}, {"operationAliases": ["search_people"]},
    {"categories": ["research", "people_search"]}, {"provider": "hubspot"},
    {"operation": "send_company_message"},
])
def test_any_dialect_does_not_grant_unsafe_tools(changes):
    source = answer_row()
    source.update(changes)
    assert catalog.allowed_tool_ids(frozen(row("safe_company_search"), source)) == ("safe_company_search",)


@pytest.mark.parametrize("field,value", [("contacts", []), ("headers", {}), ("tools", []), ("type", "person")])
def test_any_payload_still_enforces_people_and_secret_policy(field, value):
    snapshot = frozen(answer_row())
    with pytest.raises(catalog.CatalogError):
        catalog.validate_payload(snapshot, "exa_answer", {"query": "companies", "outputSchema": {field: value}})


@pytest.mark.parametrize("query", ["q=company", "query=company"])
def test_search_matches_specific_path_and_normalizes_public_query(query):
    snapshot = frozen()
    assert "deepline.tools.search" not in operations.OPERATIONS
    assert operations.match_request("GET", "https://code.deepline.com/api/v2/tools/search?" + query,
        None, {}, deepline_catalog=snapshot) == ("deepline.tools.search", {"query": "company", "compact": True})
    assert operations.match_request("GET", "https://code.deepline.com/api/v2/tools",
        None, {}, deepline_catalog=snapshot)[0] == "deepline.tools.list"
    assert operations.match_request("GET", "https://code.deepline.com/api/v2/integrations/vendor_company_search/get",
        None, {}, deepline_catalog=snapshot)[0] == "deepline.tools.get"


@pytest.mark.parametrize("query", [
    "", "q=", "q=" + "a" * 513, "q=one&q=two", "q=one&query=two",
    "q=company&unknown=true", "q=company&compact=maybe", "q=company&categories",
])
def test_search_query_validation(query):
    with pytest.raises(operations.OperationRequestError):
        operations.match_request("GET", "https://code.deepline.com/api/v2/tools/search?" + query,
            None, {}, deepline_catalog=frozen())


def test_search_requires_round_catalog_and_rejects_credentials_and_body():
    url = "https://code.deepline.com/api/v2/tools/search?q=company"
    with pytest.raises(operations.OperationRequestError):
        operations.match_request("GET", url, None, {})
    for body, headers in ((None, {"Authorization": "private"}), (b"{}", {})):
        with pytest.raises(operations.OperationRequestError):
            operations.match_request("GET", url, body, headers, deepline_catalog=frozen())


@pytest.mark.parametrize("stale", [False, True])
def test_search_is_local_free_and_obeys_lease_and_frozen_policy(stale):
    snapshot = frozen(answer_row(), row("company_search"), row("vendor_people_lookup", categories=["research"]))
    context = replace(CONTEXT, deepline_catalog=snapshot)
    reads = []
    class Store:
        def run_quota_snapshot(self, run_id, lease_token_hash):
            reads.append((run_id, lease_token_hash))
            return {"status": "stale" if stale else "active"}
    def forbidden(*args, **kwargs):
        pytest.fail("Local discovery must not read credentials, reserve quota or call a provider")
    class Transport:
        send = forbidden
    service = broker.Broker(store=Store(), key_for=forbidden, transport=Transport(), price_table=price_table())
    operation, params = operations.match_request("GET",
        "https://code.deepline.com/api/v2/tools/search?q=COMPANY&categories=company_search&compact=false",
        None, {}, deepline_catalog=snapshot)
    result = service.execute(context, operation_id=operation, parameters=params, action_sequence=1, timeout_ms=5000)
    assert reads == [(context.run_id, context.lease_token_hash)]
    assert result.call["actual_microusd"] == 0
    if stale:
        assert result.status != 200
        assert result.call["error_code"] == "lease_stale"
    else:
        assert result.status == 200
        document = json.loads(result.body)
        assert document["total"] == 1
        assert document["tools"][0]["toolId"] == "company_search"
        assert "inputSchema" in document["tools"][0]
        assert result.call["catalog_hash"] == snapshot["catalog_hash"]
