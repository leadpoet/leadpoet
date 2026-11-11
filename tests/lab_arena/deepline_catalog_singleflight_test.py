"""Concurrent requests retain the full exact-byte catalog authority check."""
import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from lab_arena import deepline_catalog as catalog
from tests.lab_arena.test_deepline_catalog import frozen, row


def test_concurrent_cold_raw_and_prepared_reads_rebuild_once(monkeypatch):
    snapshot = frozen(row("singleflight_company_search"))
    prepared = catalog.prepare_catalog(snapshot)
    catalog._validated.cache_clear()
    original = catalog.freeze_catalog
    entered, duplicate, release = Event(), Event(), Event()
    started, ready, count_lock = 0, Event(), Lock()
    rebuilds = []

    def rebuild(*args, **kwargs):
        with count_lock:
            rebuilds.append(True)
            if len(rebuilds) > 1:
                duplicate.set()
        entered.set()
        assert release.wait(5), "test did not release the cold validator"
        return original(*args, **kwargs)

    monkeypatch.setattr(catalog, "freeze_catalog", rebuild)
    barrier = Barrier(10)
    readers = (
        lambda: catalog.validate_catalog(snapshot) == snapshot,
        lambda: catalog.prepare_catalog(snapshot) == snapshot,
        lambda: prepared["catalog_hash"] == snapshot["catalog_hash"],
        lambda: set(iter(prepared)) == set(snapshot),
        lambda: len(prepared) == len(snapshot),
    )

    def read(index):
        nonlocal started
        barrier.wait(timeout=5)
        with count_lock:
            started += 1
            if started == 10:
                ready.set()
        return readers[index % len(readers)]()

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(read, index) for index in range(10)]
        try:
            assert ready.wait(5) and entered.wait(5)
            # Keep the first rebuild blocked while all ten readers contend.
            # This checks duplicate work, not a machine-dependent time limit.
            assert not duplicate.wait(0.1)
        finally:
            release.set()
        assert all(future.result(timeout=5) for future in futures)
    assert len(rebuilds) == 1
    assert catalog._validated.cache_info().misses == 1
    assert catalog._validated.cache_info().currsize == 1


def test_cached_snapshot_does_not_hide_raw_mutation():
    snapshot = frozen(row("singleflight_mutation_company_search"))
    prepared = catalog.prepare_catalog(snapshot)
    original = copy.deepcopy(snapshot)
    snapshot["tools"][0]["description"] += " changed"
    with pytest.raises(catalog.CatalogError):
        catalog.validate_catalog(snapshot)
    assert catalog.validate_catalog(prepared) == original
    snapshot["catalog_hash"] = catalog._hash(snapshot)
    assert catalog.validate_catalog(snapshot) == snapshot
    assert catalog.validate_catalog(prepared) == original


def test_invalid_schema_is_rejected_again_after_failed_validation(monkeypatch):
    snapshot = frozen(row("singleflight_invalid_company_search"))
    snapshot["tools"][0]["input_schema"]["properties"]["query"]["type"] = "invalid-type"
    snapshot["catalog_hash"] = catalog._hash(snapshot)
    original = catalog.freeze_catalog
    rebuilds = []

    def rebuild(*args, **kwargs):
        rebuilds.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(catalog, "freeze_catalog", rebuild)
    for _ in range(2):
        with pytest.raises(catalog.CatalogError):
            catalog.validate_catalog(snapshot)
    assert len(rebuilds) == 2


def test_four_entry_cache_revalidates_evicted_prepared_snapshot(monkeypatch):
    snapshots = [frozen(row("singleflight_eviction_company_search", description=str(i)))
                 for i in range(5)]
    catalog._validated.cache_clear()
    original = catalog.freeze_catalog
    rebuilds = []

    def rebuild(*args, **kwargs):
        rebuilds.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(catalog, "freeze_catalog", rebuild)
    prepared = [catalog.prepare_catalog(snapshot) for snapshot in snapshots]
    assert len(rebuilds) == 5
    assert catalog._validated.cache_info().currsize == 4
    assert catalog.validate_catalog(prepared[0]) == snapshots[0]
    assert len(rebuilds) == 6
    assert catalog._validated.cache_info().currsize == 4
