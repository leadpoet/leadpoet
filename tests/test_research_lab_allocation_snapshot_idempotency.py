from __future__ import annotations

import pytest

from gateway.research_lab import store


def _allocation() -> dict:
    return {
        "allocation_hash": "sha256:" + "a" * 64,
        "input_hash": "sha256:" + "b" * 64,
        "lab_cap_percent": 30.0,
        "source_add_alpha_percent": 1.0,
        "reimbursement_alpha_percent": 2.0,
        "champion_alpha_percent": 3.0,
        "queued_champion_alpha_percent": 4.0,
        "unallocated_percent": 20.0,
    }


def _stored_row() -> dict:
    allocation = _allocation()
    return {
        "allocation_id": "lab_allocation:" + allocation["allocation_hash"],
        "schema_version": "1.0",
        "epoch": 24121,
        "netuid": 71,
        "policy_id": "policy-v2",
        "snapshot_status": "active",
        "lab_cap_alpha_percent": "30.000000",
        "source_add_alpha_percent": "1.000000",
        "reimbursement_alpha_percent": "2.000000",
        "champion_alpha_percent": "3.000000",
        "queued_champion_alpha_percent": "4.000000",
        "unallocated_alpha_percent": "20.000000",
        "input_hash": allocation["input_hash"],
        "allocation_hash": allocation["allocation_hash"],
        "allocation_doc": allocation,
        "created_at": "2026-07-24T00:00:00+00:00",
    }


@pytest.mark.asyncio
async def test_allocation_snapshot_duplicate_race_reloads_exact_row(monkeypatch):
    reads = iter((None, _stored_row()))

    async def select_one(*_args, **_kwargs):
        return next(reads)

    async def insert_row(*_args, **_kwargs):
        raise RuntimeError("duplicate key value violates unique constraint (23505)")

    monkeypatch.setattr(store, "select_one", select_one)
    monkeypatch.setattr(store, "insert_row", insert_row)

    result = await store.create_research_lab_emission_allocation_snapshot(
        epoch=24121,
        netuid=71,
        policy_id="policy-v2",
        snapshot_status="active",
        allocation_doc=_allocation(),
    )

    assert result == _stored_row()


@pytest.mark.asyncio
async def test_allocation_snapshot_duplicate_race_rejects_mismatch(monkeypatch):
    mismatched = _stored_row()
    mismatched["epoch"] = 24120
    reads = iter((None, mismatched))

    async def select_one(*_args, **_kwargs):
        return next(reads)

    async def insert_row(*_args, **_kwargs):
        raise RuntimeError("duplicate key value violates unique constraint")

    monkeypatch.setattr(store, "select_one", select_one)
    monkeypatch.setattr(store, "insert_row", insert_row)

    with pytest.raises(RuntimeError, match="existing epoch differs"):
        await store.create_research_lab_emission_allocation_snapshot(
            epoch=24121,
            netuid=71,
            policy_id="policy-v2",
            snapshot_status="active",
            allocation_doc=_allocation(),
        )


@pytest.mark.asyncio
async def test_allocation_snapshot_existing_row_is_verified(monkeypatch):
    existing = _stored_row()

    async def select_one(*_args, **_kwargs):
        return existing

    async def unexpected_insert(*_args, **_kwargs):
        raise AssertionError("existing exact snapshot must not be inserted")

    monkeypatch.setattr(store, "select_one", select_one)
    monkeypatch.setattr(store, "insert_row", unexpected_insert)

    result = await store.create_research_lab_emission_allocation_snapshot(
        epoch=24121,
        netuid=71,
        policy_id="policy-v2",
        snapshot_status="active",
        allocation_doc=_allocation(),
    )

    assert result == existing


@pytest.mark.asyncio
async def test_allocation_snapshot_duplicate_without_readback_fails(monkeypatch):
    reads = iter((None, None))

    async def select_one(*_args, **_kwargs):
        return next(reads)

    async def insert_row(*_args, **_kwargs):
        raise RuntimeError("duplicate key value violates unique constraint")

    monkeypatch.setattr(store, "select_one", select_one)
    monkeypatch.setattr(store, "insert_row", insert_row)

    with pytest.raises(RuntimeError, match="duplicate snapshot could not be reloaded"):
        await store.create_research_lab_emission_allocation_snapshot(
            epoch=24121,
            netuid=71,
            policy_id="policy-v2",
            snapshot_status="active",
            allocation_doc=_allocation(),
        )
