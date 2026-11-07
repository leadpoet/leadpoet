"""A 55P03 lock timeout rolls its statement back, so reads may repeat it."""

import httpx
import pytest

from lab_arena.store import ArenaStoreUnavailable, PostgrestTransport

LOCK_TIMEOUT = {"code": "55P03", "message": "canceling statement due to lock timeout"}


def _transport(handle):
    client = httpx.Client(
        transport=httpx.MockTransport(handle),
        timeout=httpx.Timeout(connect=1.5, read=8.0, write=2.5, pool=3.5),
    )
    return PostgrestTransport(
        "https://db.example", service_key="sb_secret_test", http_client=client,
    ), client


def test_review_guard_state_rpc_retries_a_lock_timeout():
    attempts = []

    def handle(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(500, json=LOCK_TIMEOUT)
        return httpx.Response(200, json={
            "schema_version": "leadpoet.lab_arena.review_guard_state.v1",
            "guard_present": False,
        })

    store, client = _transport(handle)
    assert store.rpc("lab_arena_review_guard_state_v1", {})["guard_present"] is False
    assert len(attempts) == 2
    store.close()
    client.close()


def test_review_guard_state_rpc_still_fails_when_the_lock_never_clears():
    attempts = []

    def handle(request):
        attempts.append(request.url.path)
        return httpx.Response(500, json=LOCK_TIMEOUT)

    store, client = _transport(handle)
    with pytest.raises(ArenaStoreUnavailable):
        store.rpc("lab_arena_review_guard_state_v1", {})
    assert len(attempts) > 1
    store.close()
    client.close()


@pytest.mark.parametrize("function", [
    # lab_arena_claim_assignment is deliberately excluded here: it was moved
    # into LOCK_TIMEOUT_REPLAYABLE_RPCS by "Retry a contended Arena work claim
    # instead of refusing it" (merged to main 2026-11-06) because it is keyed
    # on an exact caller identity and safe to repeat. This test is about
    # mutating RPCs that are NOT in either replay set.
    "lab_arena_transition_round",
])
def test_a_mutating_rpc_still_never_replays_a_lock_timeout(function):
    attempts = []

    def handle(request):
        attempts.append(request.url.path)
        return httpx.Response(500, json=LOCK_TIMEOUT)

    store, client = _transport(handle)
    with pytest.raises(ArenaStoreUnavailable):
        store.rpc(function, {})
    assert len(attempts) == 1
    store.close()
    client.close()


def test_select_retries_a_lock_timeout_once():
    attempts = []

    def handle(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(500, json=LOCK_TIMEOUT)
        return httpx.Response(200, json=[{"round_id": "r1"}])

    store, client = _transport(handle)
    assert store.select("lab_arena_rounds", filters={"round_id": "r1"}) == [{"round_id": "r1"}]
    assert len(attempts) == 2
    store.close()
    client.close()


def test_select_surfaces_unavailable_when_both_attempts_lock_out():
    attempts = []

    def handle(request):
        attempts.append(request.url.path)
        return httpx.Response(500, json=LOCK_TIMEOUT)

    store, client = _transport(handle)
    with pytest.raises(ArenaStoreUnavailable):
        store.select("lab_arena_rounds", filters={"round_id": "r1"})
    assert len(attempts) == 2
    store.close()
    client.close()
