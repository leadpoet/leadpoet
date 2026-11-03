"""Frozen migration fixtures keep their exact pre-deadline review contract."""
from types import SimpleNamespace
import pytest
from lab_arena.store import ArenaStore, ArenaStoreError, CODE_REVIEW_SCHEMA_VERSION
from tests.lab_arena.historical_service_harness import HistoricalArenaStore


def capability():
    return dict(schema_version=CODE_REVIEW_SCHEMA_VERSION, version=207,
                claim_ttl_seconds=600, retry_backoff_seconds=60, max_attempts=3,
                retry_policy="bounded_transient_v1", max_transient_attempts=6,
                transient_retry_backoff_seconds=[60,120,240,480,900],
                legacy_max_attempts=3,
                retry_window="replacement_freeze_to_benchmark_deadline")


def transport(document):
    def rpc(name, parameters):
        assert name == "lab_arena_code_review_schema_v1" and parameters == {}
        return dict(document)
    return SimpleNamespace(rpc=rpc)


def test_frozen_reader_checks_original_capability_while_current_reader_rejects_it():
    old = capability()
    assert HistoricalArenaStore(transport(old)).code_review_schema() == old
    with pytest.raises(ArenaStoreError, match="code review schema mismatch"):
        ArenaStore(transport(old)).code_review_schema()


@pytest.mark.parametrize("field,value", [
    ("version",208), ("claim_ttl_seconds",601), ("max_transient_attempts",None),
    ("retry_policy","deadline_transient_v2"), ("max_attempts",4),
    ("retry_backoff_seconds",0), ("transient_retry_backoff_seconds",[60]),
    ("legacy_max_attempts",4), ("retry_window","unbounded"),
])
def test_frozen_reader_rejects_changed_contract(field, value):
    old=capability();old[field]=value
    with pytest.raises(ArenaStoreError, match="code review schema mismatch"):
        HistoricalArenaStore(transport(old)).code_review_schema()
