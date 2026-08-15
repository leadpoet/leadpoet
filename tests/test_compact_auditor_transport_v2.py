from __future__ import annotations

import pytest


def test_compact_authority_store_rejects_oversized_document(monkeypatch):
    from gateway.research_lab import attested_v2_store as store

    monkeypatch.setattr(
        store,
        "COMPACT_WEIGHT_AUTHORITY_MAX_BYTES_V2",
        128,
    )

    with pytest.raises(
        store.AttestedV2StoreError,
        match="exceeds the 8 MiB transport bound",
    ):
        store._compact_weight_authority_row_from_normalized_v2(
            {"oversized": "x" * 256}
        )


def test_compact_authority_store_size_check_is_ascii_conservative(monkeypatch):
    from gateway.research_lab import attested_v2_store as store

    observed = {}
    monkeypatch.setattr(
        store,
        "validate_compact_weight_submission_shape_v2",
        lambda value: observed.setdefault("compact", value),
    )
    monkeypatch.setattr(
        store,
        "COMPACT_WEIGHT_AUTHORITY_MAX_BYTES_V2",
        2_048,
    )
    normalized = {
        "bundle_hash": "sha256:" + "a" * 64,
        "authority_stage": "published",
        "schema_version": "schema",
        "lineage_id": "sha256:" + "b" * 64,
        "authority_hash": "sha256:" + "c" * 64,
        "compact_submission": {
            "weight_result": {
                "netuid": 71,
                "epoch_id": 1,
            },
            "validator_hotkey": "validator",
            "compact_submission_hash": "sha256:" + "d" * 64,
        },
        "publication": {
            "publication_receipt_hash": "sha256:" + "e" * 64,
        },
        "finalization": None,
    }

    row = store._compact_weight_authority_row_from_normalized_v2(normalized)

    assert row["authority_doc"] == normalized
    assert observed["compact"] == normalized["compact_submission"]


@pytest.mark.asyncio
async def test_compact_identity_read_never_loads_full_bundle(monkeypatch):
    from gateway.research_lab import attested_v2_store as store

    authority = {"kind": "bounded-authority"}
    expected = {
        "bundle_hash": "sha256:" + "a" * 64,
        "netuid": 71,
        "epoch_id": 24084,
        "validator_hotkey": "validator",
        "authority_stage": "published",
        "authority_doc": authority,
    }
    calls = []

    async def fake_select(table, filters):
        calls.append((table, tuple(filters)))
        if table != store.COMPACT_WEIGHT_AUTHORITY_TABLE:
            raise AssertionError("bounded read attempted a full-bundle query")
        stage = dict(filters)["authority_stage"]
        return expected if stage == "published" else None

    monkeypatch.setattr(store, "select_one", fake_select)
    monkeypatch.setattr(
        store,
        "validate_compact_published_weight_authority_shape_v2",
        lambda value: value,
    )
    monkeypatch.setattr(
        store,
        "_compact_weight_authority_row_from_normalized_v2",
        lambda value: expected,
    )

    loaded = await store.load_compact_weight_authority_for_identity_v2(
        netuid=71,
        epoch_id=24084,
        validator_hotkey="validator",
    )

    assert loaded == authority
    assert [table for table, _filters in calls] == [
        store.COMPACT_WEIGHT_AUTHORITY_TABLE,
        store.COMPACT_WEIGHT_AUTHORITY_TABLE,
    ]


@pytest.mark.asyncio
async def test_compact_gateway_route_uses_only_bounded_identity_loader(monkeypatch):
    from gateway.api import weights as weights_api
    from gateway.research_lab import attested_v2_store as store

    authority = {
        "schema_version": "leadpoet.compact_published_weight_authority.v2",
        "authority_stage": "published",
    }
    calls = []

    async def fake_load(**kwargs):
        calls.append(kwargs)
        return authority

    monkeypatch.setattr(
        store,
        "load_compact_weight_authority_for_identity_v2",
        fake_load,
    )
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator"})
    request = weights_api.Request({"type": "http", "headers": []})

    response = await weights_api.get_compact_published_weights_v2(
        71, 24084, request
    )

    assert response.status_code == 200
    assert calls == [
        {"netuid": 71, "epoch_id": 24084, "validator_hotkey": "validator"}
    ]
