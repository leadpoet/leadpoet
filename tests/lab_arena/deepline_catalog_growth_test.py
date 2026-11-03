"""Catalog growth preserves every eligible tool and frozen-round authority."""
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, deepline_catalog as catalog, operations, shim
from tests.lab_arena.deepline_round_runtime_test import _round, _service
from tests.lab_arena.test_deepline_catalog import frozen, row


CURRENT_CATALOG_ROWS = 4104


def growth_rows(count=CURRENT_CATALOG_ROWS):
    return [row("vendor_company_search_%05d" % index) for index in range(count)]


@pytest.mark.parametrize("reader", ["fetch", "freeze"])
def test_current_catalog_count_accepts_every_eligible_tool(reader):
    rows = growth_rows()
    document = {"tools": rows}
    if reader == "fetch":
        calls = []
        def read(url, headers):
            calls.append((url, headers))
            return document
        snapshot = catalog.fetch_catalog(read_json=read)
        assert calls == [(catalog.CATALOG_URL, {})]
    else:
        snapshot = catalog.freeze_catalog(document)
    assert len(snapshot["tools"]) == CURRENT_CATALOG_ROWS
    assert catalog.allowed_tool_ids(snapshot) == tuple(row["toolId"] for row in rows)
    assert catalog.validate_catalog(json.loads(json.dumps(snapshot))) == snapshot


def test_grown_catalog_crosses_round_creation_and_runtime_without_truncation():
    snapshot = catalog.fetch_catalog(read_json=lambda *_: {"tools": growth_rows()})
    config = _round(lambda **_: snapshot)
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    assert config["deepline_catalog"] == snapshot
    # Restarted discovery cannot replace an existing round's frozen snapshot.
    service = _service(lambda **_: pytest.fail("frozen round must not fetch metadata"))
    service._freeze_deepline_catalog(config)
    assert config["deepline_catalog"] == snapshot
    tail = snapshot["tools"][-1]["tool_id"]
    parameters = {"tool": tail, "payload": {"query": "company"}}
    frame = {"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION,
             "operation_id": "deepline.execute", "timeout_ms": 5000,
             "parameters": parameters}
    assert shim.decode_operation_frame(
        json.dumps(frame).encode(), deepline_catalog=snapshot
    ) == ("deepline.execute", parameters, 5000)
    request = operations.build_outbound_request(
        "deepline.execute", parameters, deepline_catalog=snapshot)
    assert json.loads(request.body)["operation"] == tail


@pytest.mark.parametrize("reader", ["fetch", "freeze"])
@pytest.mark.parametrize("tools", [None, {}, [None]])
def test_catalog_growth_does_not_accept_malformed_rows(reader, tools):
    with pytest.raises(catalog.CatalogError):
        if reader == "fetch":
            catalog.fetch_catalog(read_json=lambda *_: {"tools": tools})
        else:
            catalog.freeze_catalog({"tools": tools})


def test_catalog_count_bound_is_shared_and_does_not_truncate():
    assert catalog.MAX_CATALOG_TOOLS == 16_384
    document = {"tools": [row()] * (catalog.MAX_CATALOG_TOOLS + 1)}
    with pytest.raises(catalog.CatalogError, match="^invalid_catalog$"):
        catalog.fetch_catalog(read_json=lambda *_: document)
    with pytest.raises(catalog.CatalogError, match="^invalid_catalog$"):
        catalog.freeze_catalog(document)
    snapshot = frozen()
    snapshot["tools"] *= catalog.MAX_CATALOG_TOOLS + 1
    snapshot["catalog_hash"] = catalog._hash(snapshot)
    with pytest.raises(catalog.CatalogError, match="^invalid_catalog_snapshot$"):
        catalog.validate_catalog(snapshot)


def test_count_headroom_keeps_security_filters_and_historical_hash():
    # A hard-coded pre-fix hash proves that this operational bound changes no
    # historical schema, price, ordering, policy or hash input.
    snapshot = frozen()
    assert snapshot["catalog_hash"] == "8b6d41ae4f521067e959a88a1f03ff9b7dfd04ea179de993253b267831f01380"
    assert catalog.validate_catalog(deepcopy(snapshot)) == snapshot
    unsafe = [row("vendor_company_update_%05d" % index)
              for index in range(catalog.MAX_CATALOG_TOOLS - 1)]
    document = {"tools": unsafe + [row()]}
    assert catalog.freeze_catalog(document) == snapshot
    assert catalog.fetch_catalog(read_json=lambda *_: document) == snapshot
