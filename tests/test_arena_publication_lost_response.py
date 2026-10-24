"""A lost publication response is not a proved rollback.

An upstream gateway that stops waiting answers 504 without learning how the
statement ended, so ``lab_arena_transition_round`` may already have committed.
Treating that as a hard failure discards a published round's result.
"""

from __future__ import annotations

from lab_arena.service import _commit_outcome_unknown
from lab_arena.store import ArenaStoreError, ArenaStoreUnavailable

RESOURCE = "lab_arena_transition_round"


def _store_error(**diagnostics) -> ArenaStoreError:
    return ArenaStoreError("rpc failed", operation="rpc", resource=RESOURCE, **diagnostics)


def test_gateway_timeout_leaves_the_commit_unknown() -> None:
    for status in (502, 503, 504):
        assert _commit_outcome_unknown(_store_error(http_status=status), RESOURCE)


def test_transport_failure_leaves_the_commit_unknown() -> None:
    error = ArenaStoreUnavailable(
        "rpc transport failure", operation="rpc", resource=RESOURCE,
        transport_error="ReadTimeout",
    )
    assert _commit_outcome_unknown(error, RESOURCE)


def test_proved_rollback_stays_a_failure() -> None:
    error = ArenaStoreUnavailable(
        "rpc rolled back", operation="rpc", resource=RESOURCE,
        http_status=503, transaction_rolled_back=True,
    )
    assert not _commit_outcome_unknown(error, RESOURCE)


def test_server_error_stays_a_failure() -> None:
    assert not _commit_outcome_unknown(_store_error(http_status=500), RESOURCE)


def test_another_resource_is_never_reconciled_here() -> None:
    error = ArenaStoreError(
        "rpc failed", operation="rpc", resource="lab_arena_cancel_round",
        http_status=504,
    )
    assert not _commit_outcome_unknown(error, RESOURCE)


def test_an_error_without_store_metadata_stays_a_failure() -> None:
    assert not _commit_outcome_unknown(ArenaStoreError("rpc failed"), RESOURCE)
