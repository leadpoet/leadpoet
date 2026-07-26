"""The attested-allocation endpoint must build each epoch's bundle at most once.

Assembling a bundle reconstructs the full ancestry receipt graph and takes tens
of seconds. The validator polls this endpoint inside a fixed on-chain window and
retries slow responses, so without coordination each retry and each concurrent
poll launched a fresh rebuild; the rebuilds contended for the database pool and
the enclave, every one slowed past the validator's fetch timeout, and the
validator's fail-closed guard then blocked the epoch's weight submission. These
tests pin the coordination: one build per epoch, cache hits skip the build,
failures are not cached, the guard runs on every request, and the cache expires.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gateway.research_lab import api
from leadpoet_canonical.constants import EPOCH_LENGTH


def _config():
    return SimpleNamespace(
        api_enabled=True,
        reports_enabled=True,
        shadow_bundles_enabled=True,
        reimbursements_enabled=False,
        weight_mutation_enabled=False,
    )


def _live_config():
    return SimpleNamespace(
        api_enabled=True,
        reports_enabled=True,
        shadow_bundles_enabled=True,
        reimbursements_enabled=True,
        weight_mutation_enabled=False,
        internal_api_key="validator-secret",
    )


@pytest.fixture(autouse=True)
def _clear_allocation_cache():
    api._ALLOCATION_HANDOFF_CACHE.clear()
    api._ALLOCATION_BUILD_TASKS.clear()
    yield
    api._ALLOCATION_HANDOFF_CACHE.clear()
    api._ALLOCATION_BUILD_TASKS.clear()


def _install(monkeypatch, *, build, guard=None, handoff=None):
    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", build)
    if guard is None:
        async def guard(config, epoch, key):
            return False
    monkeypatch.setattr(api, "_allocation_epoch_guard_and_persistence", guard)
    if handoff is None:
        def handoff(**kwargs):
            return {"handoff_for": kwargs["bundle"]["epoch"]}
    monkeypatch.setattr(
        "leadpoet_canonical.allocation_handoff_v2.build_allocation_handoff_v2",
        handoff,
    )


def _matched_build(counter, *, delay=0.0):
    receipt = {"receipt_hash": "sha256:" + "1" * 64}
    graph = {"root_receipt_hash": receipt["receipt_hash"], "receipts": [receipt]}

    async def build(**kwargs):
        counter["n"] += 1
        if delay:
            await asyncio.sleep(delay)
        kwargs["attestation_out"].update({
            "status": "matched",
            "receipt": receipt,
            "receipt_graph": graph,
            "lineage_bindings": [],
            "lineage_complete": True,
            "persistence": {"root_receipt_hash": receipt["receipt_hash"]},
        })
        return {"bundle_type": "live", "epoch": kwargs["epoch"]}

    return build


@pytest.mark.asyncio
async def test_prelaunch_current_epoch_build_persists_without_gateway_chain_client(
    monkeypatch,
):
    observed = []
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        _live_config,
    )

    async def handoff(*, config, epoch, persist_snapshot):
        observed.append((config, epoch, persist_snapshot))
        return {"handoff_for": epoch}

    async def unexpected_guard(*_args, **_kwargs):
        raise AssertionError("pre-resolved maintenance must not use gateway lifespan state")

    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_handoff",
        handoff,
    )
    monkeypatch.setattr(
        api,
        "_allocation_epoch_guard_and_persistence",
        unexpected_guard,
    )

    result = await (
        api._get_research_lab_attested_allocation_for_resolved_current_epoch(
            epoch=24177,
            current_epoch=24177,
            internal_key="validator-secret",
        )
    )

    assert result == {"handoff_for": 24177}
    assert observed == [(_live_config(), 24177, True)]


@pytest.mark.asyncio
async def test_prelaunch_resolved_epoch_keeps_historical_build_read_only(
    monkeypatch,
):
    observed = []
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        _live_config,
    )

    async def handoff(*, config, epoch, persist_snapshot):
        observed.append((config, epoch, persist_snapshot))
        return {"handoff_for": epoch}

    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_handoff",
        handoff,
    )

    result = await (
        api._get_research_lab_attested_allocation_for_resolved_current_epoch(
            epoch=24176,
            current_epoch=24177,
            internal_key="validator-secret",
        )
    )

    assert result == {"handoff_for": 24176}
    assert observed == [(_live_config(), 24176, False)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("epoch", "internal_key", "status_code"),
    (
        (24178, "validator-secret", 422),
        (24177, "wrong-secret", 401),
    ),
)
async def test_prelaunch_resolved_epoch_rejects_unsafe_persistence(
    monkeypatch,
    epoch,
    internal_key,
    status_code,
):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        _live_config,
    )

    async def unexpected_handoff(**_kwargs):
        raise AssertionError("rejected persistence must not build a handoff")

    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_handoff",
        unexpected_handoff,
    )

    with pytest.raises(HTTPException) as caught:
        await (
            api._get_research_lab_attested_allocation_for_resolved_current_epoch(
                epoch=epoch,
                current_epoch=24177,
                internal_key=internal_key,
            )
        )

    assert getattr(caught.value, "status_code", None) == status_code


@pytest.mark.asyncio
async def test_repeat_request_serves_cache_without_rebuild(monkeypatch):
    counter = {"n": 0}
    _install(monkeypatch, build=_matched_build(counter))

    first = await api.get_research_lab_attested_allocation(7)
    second = await api.get_research_lab_attested_allocation(7)

    assert first == second == {"handoff_for": 7}
    assert counter["n"] == 1  # built once, second served from cache


@pytest.mark.asyncio
async def test_concurrent_requests_build_once(monkeypatch):
    counter = {"n": 0}
    # A slow build so all pollers overlap on the same in-flight rebuild.
    _install(monkeypatch, build=_matched_build(counter, delay=0.2))

    results = await asyncio.gather(
        *[api.get_research_lab_attested_allocation(9) for _ in range(12)]
    )

    assert all(r == {"handoff_for": 9} for r in results)
    assert counter["n"] == 1  # 12 concurrent pollers, exactly one build


@pytest.mark.asyncio
async def test_client_cancellation_does_not_cancel_build_or_cache(monkeypatch):
    counter = {"n": 0}
    _install(monkeypatch, build=_matched_build(counter, delay=0.1))

    disconnected = asyncio.create_task(
        api.get_research_lab_attested_allocation(10)
    )
    await asyncio.sleep(0.02)
    disconnected.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnected

    # The server-owned task finishes independently and populates the cache.
    await asyncio.sleep(0.12)
    recovered = await api.get_research_lab_attested_allocation(10)

    assert recovered == {"handoff_for": 10}
    assert counter["n"] == 1


@pytest.mark.asyncio
async def test_distinct_epochs_build_independently(monkeypatch):
    counter = {"n": 0}
    _install(monkeypatch, build=_matched_build(counter))

    await api.get_research_lab_attested_allocation(1)
    await api.get_research_lab_attested_allocation(2)
    await api.get_research_lab_attested_allocation(1)

    assert counter["n"] == 2  # one build per distinct epoch, epoch 1 reused


@pytest.mark.asyncio
async def test_failed_build_is_not_cached(monkeypatch):
    counter = {"n": 0}

    async def failing_build(**kwargs):
        counter["n"] += 1
        raise ValueError("allocation not ready")

    _install(monkeypatch, build=failing_build)

    for _ in range(3):
        with pytest.raises(Exception):
            await api.get_research_lab_attested_allocation(5)

    assert counter["n"] == 3  # every retry rebuilds; failures never cached


@pytest.mark.asyncio
async def test_guard_runs_on_every_request_including_cache_hits(monkeypatch):
    counter = {"n": 0}
    guard_calls = {"n": 0}

    async def guard(config, epoch, key):
        guard_calls["n"] += 1
        return False

    _install(monkeypatch, build=_matched_build(counter), guard=guard)

    await api.get_research_lab_attested_allocation(3)
    await api.get_research_lab_attested_allocation(3)
    await api.get_research_lab_attested_allocation(3)

    assert counter["n"] == 1  # one build
    assert guard_calls["n"] == 3  # guard (future-epoch + persistence) every time


@pytest.mark.asyncio
async def test_read_only_cache_cannot_suppress_authenticated_persistence(monkeypatch):
    counter = {"n": 0}
    persistence_modes = []
    matched_build = _matched_build(counter)

    async def build(**kwargs):
        persistence_modes.append(bool(kwargs["persist_snapshot"]))
        return await matched_build(**kwargs)

    async def guard(config, epoch, key):
        return key == "validator-key"

    _install(monkeypatch, build=build, guard=guard)

    await api.get_research_lab_attested_allocation(6)
    await api.get_research_lab_attested_allocation(
        6,
        x_leadpoet_internal_key="validator-key",
    )
    await api.get_research_lab_attested_allocation(
        6,
        x_leadpoet_internal_key="validator-key",
    )

    assert persistence_modes == [False, True]
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_cache_entry_expires_after_ttl(monkeypatch):
    counter = {"n": 0}
    _install(monkeypatch, build=_matched_build(counter))
    monkeypatch.setattr(api, "_ALLOCATION_CACHE_TTL_SECONDS", 0.05)

    await api.get_research_lab_attested_allocation(4)
    assert counter["n"] == 1
    time.sleep(0.1)  # let the entry expire
    await api.get_research_lab_attested_allocation(4)
    assert counter["n"] == 2  # rebuilt after expiry


def test_default_cache_lifetime_spans_a_full_target_epoch():
    assert api._ALLOCATION_CACHE_TTL_SECONDS >= EPOCH_LENGTH * 12


@pytest.mark.asyncio
async def test_cache_is_bounded_by_max_epochs(monkeypatch):
    counter = {"n": 0}
    _install(monkeypatch, build=_matched_build(counter))
    monkeypatch.setattr(api, "_ALLOCATION_CACHE_MAX_EPOCHS", 4)

    for epoch in range(10):
        await api.get_research_lab_attested_allocation(epoch)

    assert len(api._ALLOCATION_HANDOFF_CACHE) <= 4
    assert not api._ALLOCATION_BUILD_TASKS


@pytest.mark.asyncio
async def test_anonymous_request_reuses_the_persisted_build(monkeypatch):
    """An anonymous poll must not rebuild an epoch the validator already built.

    The cache is keyed by (epoch, persist_snapshot), so an anonymous caller used
    to miss the persisting build's entry and launch a second full rebuild of the
    same epoch — another multi-minute pass over the ancestry tables and another
    ~9 MiB handoff, contending for the database pool and enclave the validator
    needs inside its submission window.
    """

    counter = {"n": 0}

    async def guard(config, epoch, key):
        return key == "validator-key"

    _install(monkeypatch, build=_matched_build(counter), guard=guard)

    authenticated = await api.get_research_lab_attested_allocation(
        11,
        x_leadpoet_internal_key="validator-key",
    )
    assert counter["n"] == 1

    anonymous = await api.get_research_lab_attested_allocation(11)

    assert anonymous == authenticated == {"handoff_for": 11}
    assert counter["n"] == 1  # reused, not rebuilt


@pytest.mark.asyncio
async def test_anonymous_request_joins_inflight_persisted_build(monkeypatch):
    counter = {"n": 0}
    started = asyncio.Event()
    release = asyncio.Event()
    disk_lookup_started = threading.Event()
    release_disk_lookup = threading.Event()
    disk_lookup_count = 0
    disk_lookup_lock = threading.Lock()

    async def build(**kwargs):
        assert kwargs["persist_snapshot"] is True
        counter["n"] += 1
        started.set()
        await release.wait()
        return await _matched_build({"n": 0})(**kwargs)

    async def guard(config, epoch, key):
        return key == "validator-key"

    def load_handoff(*_args, **_kwargs):
        nonlocal disk_lookup_count
        with disk_lookup_lock:
            disk_lookup_count += 1
            lookup_ordinal = disk_lookup_count
        if lookup_ordinal == 1:
            return None
        disk_lookup_started.set()
        assert release_disk_lookup.wait(timeout=2)
        return None

    _install(monkeypatch, build=build, guard=guard)
    monkeypatch.setattr(
        api.allocation_handoff_disk_cache,
        "load_handoff",
        load_handoff,
    )

    authenticated = asyncio.create_task(
        api.get_research_lab_attested_allocation(
            12,
            x_leadpoet_internal_key="validator-key",
        )
    )
    await started.wait()
    anonymous = asyncio.create_task(
        api.get_research_lab_attested_allocation(12)
    )
    assert await asyncio.to_thread(disk_lookup_started.wait, 2)
    release.set()
    await authenticated
    release_disk_lookup.set()

    authenticated_result = authenticated.result()
    anonymous_result = await anonymous

    assert authenticated_result == anonymous_result == {"handoff_for": 12}
    assert counter["n"] == 1


@pytest.mark.asyncio
async def test_inflight_read_only_build_cannot_suppress_persistence(monkeypatch):
    persistence_modes = []
    read_only_started = asyncio.Event()
    release = asyncio.Event()

    async def build(**kwargs):
        persistence_modes.append(bool(kwargs["persist_snapshot"]))
        if not kwargs["persist_snapshot"]:
            read_only_started.set()
        await release.wait()
        return await _matched_build({"n": 0})(**kwargs)

    async def guard(config, epoch, key):
        return key == "validator-key"

    _install(monkeypatch, build=build, guard=guard)

    anonymous = asyncio.create_task(
        api.get_research_lab_attested_allocation(13)
    )
    await read_only_started.wait()
    authenticated = asyncio.create_task(
        api.get_research_lab_attested_allocation(
            13,
            x_leadpoet_internal_key="validator-key",
        )
    )
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(anonymous, authenticated)

    assert persistence_modes == [False, True]


@pytest.mark.asyncio
async def test_anonymous_request_prefers_its_existing_read_only_build(monkeypatch):
    read_only_started = asyncio.Event()
    persisted_started = asyncio.Event()
    release = asyncio.Event()

    async def build(**kwargs):
        if kwargs["persist_snapshot"]:
            persisted_started.set()
            await release.wait()
            raise RuntimeError("snapshot persistence failed")
        read_only_started.set()
        await release.wait()
        return await _matched_build({"n": 0})(**kwargs)

    async def guard(config, epoch, key):
        return key == "validator-key"

    _install(monkeypatch, build=build, guard=guard)

    first_anonymous = asyncio.create_task(
        api.get_research_lab_attested_allocation(15)
    )
    await read_only_started.wait()
    authenticated = asyncio.create_task(
        api.get_research_lab_attested_allocation(
            15,
            x_leadpoet_internal_key="validator-key",
        )
    )
    await persisted_started.wait()
    second_anonymous = asyncio.create_task(
        api.get_research_lab_attested_allocation(15)
    )
    await asyncio.sleep(0)
    release.set()

    first_result, second_result = await asyncio.gather(
        first_anonymous,
        second_anonymous,
    )
    with pytest.raises(Exception):
        await authenticated

    assert first_result == second_result == {"handoff_for": 15}


@pytest.mark.asyncio
async def test_expired_persisted_entry_is_not_served_to_anonymous_callers(
    monkeypatch,
):
    counter = {"n": 0}

    async def guard(config, epoch, key):
        return key == "validator-key"

    _install(monkeypatch, build=_matched_build(counter), guard=guard)
    monkeypatch.setattr(api, "_ALLOCATION_CACHE_TTL_SECONDS", 0.05)

    await api.get_research_lab_attested_allocation(
        12,
        x_leadpoet_internal_key="validator-key",
    )
    assert counter["n"] == 1
    time.sleep(0.1)  # let the persisted entry expire

    await api.get_research_lab_attested_allocation(12)
    assert counter["n"] == 2  # expiry still applies across the reuse path


@pytest.mark.asyncio
async def test_expired_read_only_entry_does_not_hide_fresh_persisted_entry(
    monkeypatch,
):
    counter = {"n": 0}
    now = [0.0]

    async def guard(config, epoch, key):
        return key == "validator-key"

    _install(monkeypatch, build=_matched_build(counter), guard=guard)
    monkeypatch.setattr(api, "_ALLOCATION_CACHE_TTL_SECONDS", 10.0)
    monkeypatch.setattr(api.time, "monotonic", lambda: now[0])

    await api.get_research_lab_attested_allocation(14)
    now[0] = 5.0
    persisted = await api.get_research_lab_attested_allocation(
        14,
        x_leadpoet_internal_key="validator-key",
    )
    now[0] = 11.0

    anonymous = await api.get_research_lab_attested_allocation(14)

    assert anonymous == persisted == {"handoff_for": 14}
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_not_ready_attestation_is_logged_before_failing(monkeypatch, caplog):
    """The status must be logged beyond the callback's generic HTTPException."""

    async def unmatched_build(**kwargs):
        kwargs["attestation_out"].update({"status": "pending"})
        return {"bundle_type": "live", "epoch": kwargs["epoch"]}

    _install(monkeypatch, build=unmatched_build)

    with caplog.at_level("ERROR"):
        with pytest.raises(Exception):
            await api.get_research_lab_attested_allocation(13)

    assert "research_lab_attested_allocation_not_ready" in caplog.text
    assert "status=pending" in caplog.text


@pytest.mark.asyncio
async def test_live_allocation_failure_is_logged_before_failing(monkeypatch, caplog):
    async def failing_build(**kwargs):
        raise RuntimeError("coordinator enclave unavailable")

    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", failing_build)

    async def guard(config, epoch, key):
        return False

    monkeypatch.setattr(api, "_allocation_epoch_guard_and_persistence", guard)

    with caplog.at_level("ERROR"):
        with pytest.raises(Exception):
            await api.get_research_lab_live_allocation(14)

    assert "research_lab_live_allocation_build_failed" in caplog.text
    assert "RuntimeError" in caplog.text
