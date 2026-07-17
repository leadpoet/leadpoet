from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from Leadpoet.utils.subnet_epoch import (
    LEGACY_EPOCH_MODE,
    STATEFUL_EPOCH_MODE,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
)


def _state(lifecycle: str, *, first: int = 101, mapping_hash=None) -> dict:
    if lifecycle == "legacy_open":
        return {
            "lifecycle_state": lifecycle,
            "mapping_hash": None,
            "last_legacy_epoch_id": None,
            "first_settlement_epoch_id": None,
        }
    return {
        "lifecycle_state": lifecycle,
        "mapping_hash": mapping_hash,
        "last_legacy_epoch_id": first - 1,
        "first_settlement_epoch_id": first,
    }


def _cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash="0x" + "11" * 32,
        netuid=71,
        cutover_block=36_396,
        cutover_block_hash="0x" + "22" * 32,
        first_subnet_epoch_index=23_928,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )


def test_legacy_namespace_gate_allows_last_and_reserves_first(monkeypatch):
    from gateway.utils import epoch

    monkeypatch.setattr(epoch, "_cutover_state_cache", None)
    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _state("cutover_fenced"),
    )

    assert epoch.assert_legacy_epoch_namespace_open(
        100, force_refresh=True
    )["lifecycle_state"] == "cutover_fenced"
    with pytest.raises(
        epoch.LegacyEpochNamespaceFencedError,
        match="reserved settlement ordinal 101",
    ):
        epoch.assert_legacy_epoch_namespace_open(101, force_refresh=True)

    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _state("stateful_active", mapping_hash=_cutover().mapping_hash),
    )
    with pytest.raises(
        epoch.LegacyEpochNamespaceFencedError,
        match="disabled after stateful activation",
    ):
        epoch.assert_legacy_epoch_namespace_open(100, force_refresh=True)


def test_forced_namespace_read_never_uses_stale_state_on_outage(monkeypatch):
    from gateway.utils import epoch

    monkeypatch.setattr(epoch, "_cutover_state_cache", None)
    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _state("legacy_open"),
    )
    assert epoch.get_legacy_epoch_namespace_state(
        force_refresh=True
    )["lifecycle_state"] == "legacy_open"

    def unavailable():
        raise SubnetEpochError("database unavailable")

    monkeypatch.setattr(epoch, "_read_cutover_state_from_db_sync", unavailable)
    with pytest.raises(SubnetEpochError, match="database unavailable"):
        epoch.get_legacy_epoch_namespace_state(force_refresh=True)


def test_missing_service_credentials_use_fixed_public_rpc_not_synthetic_open(
    monkeypatch,
):
    from gateway import config
    from gateway.utils import epoch
    from Leadpoet.utils import cloud_db
    import supabase

    observed = {}
    expected = _state("stateful_active", mapping_hash=_cutover().mapping_hash)

    class Client:
        def rpc(self, name):
            observed["rpc"] = name
            return self

        def execute(self):
            return SimpleNamespace(data=[expected])

    def create_client(url, key):
        observed["authority"] = (url, key)
        return Client()

    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(supabase, "create_client", create_client)
    monkeypatch.setattr(epoch, "_cutover_state_table_observed", False)
    monkeypatch.setattr(epoch, "_cutover_state_cache", None)

    assert epoch._read_cutover_state_from_db_sync() == expected
    assert observed == {
        "authority": (cloud_db.SUPABASE_URL, cloud_db.SUPABASE_ANON_KEY),
        "rpc": "research_lab_stateful_subnet_epoch_cutover_public_state_v1",
    }


def test_explicit_finney_sn71_scope_uses_public_rpc_without_runtime_env(
    monkeypatch,
):
    from gateway import config
    from gateway.utils import epoch
    import supabase

    observed = []

    class Client:
        def rpc(self, name):
            observed.append(name)
            return self

        def execute(self):
            return SimpleNamespace(data=[_state("legacy_open")])

    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.delenv("BITTENSOR_NETWORK", raising=False)
    monkeypatch.delenv("BITTENSOR_NETUID", raising=False)
    monkeypatch.setattr(supabase, "create_client", lambda _url, _key: Client())

    state = epoch._read_cutover_state_from_db_sync(
        network="finney",
        netuid=71,
    )

    assert state["lifecycle_state"] == "legacy_open"
    assert observed == [
        "research_lab_stateful_subnet_epoch_cutover_public_state_v1"
    ]


def test_public_lifecycle_rpc_outage_never_synthesizes_legacy_open(monkeypatch):
    from gateway import config
    from gateway.utils import epoch
    import supabase

    class Client:
        def rpc(self, _name):
            return self

        def execute(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(supabase, "create_client", lambda _url, _key: Client())
    monkeypatch.setattr(epoch, "_cutover_state_table_observed", False)
    monkeypatch.setattr(epoch, "_cutover_state_cache", None)

    with pytest.raises(
        SubnetEpochError,
        match="durable epoch namespace state database is unavailable",
    ):
        epoch._read_cutover_state_from_db_sync()


@pytest.mark.parametrize(
    ("network", "netuid"),
    [("test", "71"), ("finney", "72")],
)
def test_fixed_public_lifecycle_authority_is_scoped_to_finney_sn71(
    monkeypatch,
    network,
    netuid,
):
    from gateway import config
    from gateway.utils import epoch
    import supabase

    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.setenv("BITTENSOR_NETWORK", network)
    monkeypatch.setenv("BITTENSOR_NETUID", netuid)
    monkeypatch.setattr(
        supabase,
        "create_client",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-production runtime must not read production DB")
        ),
    )

    assert epoch._read_cutover_state_from_db_sync() == _state("legacy_open")


def test_stateful_runtime_lifecycle_requires_active_exact_mapping(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    monkeypatch.setattr(epoch, "_cutover_state_cache", None)
    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _state("stateful_staged", mapping_hash=cutover.mapping_hash),
    )
    with pytest.raises(SubnetEpochError, match="does not match"):
        epoch.validate_epoch_runtime_lifecycle(
            mode=STATEFUL_EPOCH_MODE,
            cutover=cutover,
        )

    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _state("stateful_active", mapping_hash="sha256:" + "f" * 64),
    )
    with pytest.raises(SubnetEpochError, match="does not match"):
        epoch.validate_epoch_runtime_lifecycle(
            mode=STATEFUL_EPOCH_MODE,
            cutover=cutover,
        )

    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _state("stateful_active", mapping_hash=cutover.mapping_hash),
    )
    assert epoch.validate_epoch_runtime_lifecycle(
        mode=STATEFUL_EPOCH_MODE,
        cutover=cutover,
    )["mapping_hash"] == cutover.mapping_hash


@pytest.mark.asyncio
async def test_legacy_epoch_status_force_refreshes_fence_before_advertising(
    monkeypatch,
):
    from gateway.utils import epoch

    snapshot = SubnetEpochSnapshot(
        network_genesis_hash="0x" + "11" * 32,
        netuid=71,
        head_kind="best",
        block_hash="0x" + "33" * 32,
        current_block=36_360,
        last_epoch_block=36_036,
        pending_epoch_at=0,
        subnet_epoch_index=23_927,
        tempo=360,
        blocks_since_last_step=324,
        observed_at="2026-07-16T12:00:00Z",
    )
    observed = []

    async def lifecycle(**kwargs):
        observed.append(kwargs)
        raise epoch.LegacyEpochNamespaceFencedError("stateful_active")

    monkeypatch.setattr(epoch, "get_epoch_mode", lambda: LEGACY_EPOCH_MODE)
    monkeypatch.setattr(
        epoch,
        "get_current_subnet_epoch_snapshot_async",
        lambda **_kwargs: _async_value(snapshot),
    )
    monkeypatch.setattr(
        epoch,
        "validate_epoch_runtime_lifecycle_async",
        lifecycle,
    )

    with pytest.raises(epoch.LegacyEpochNamespaceFencedError):
        await epoch.get_epoch_authority_status_async()
    assert observed == [
        {
            "mode": LEGACY_EPOCH_MODE,
            "epoch_id": 101,
            "force_refresh": True,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status"),
    (("fenced", 409), ("outage", 503)),
)
async def test_legacy_weight_route_fails_closed_at_fence(monkeypatch, failure, status):
    from gateway.api import weights
    from gateway.utils.epoch import LegacyEpochNamespaceFencedError

    class Subtensor:
        def get_current_block(self):
            return 36_360

    async def gate(*_args, **_kwargs):
        if failure == "fenced":
            raise LegacyEpochNamespaceFencedError("fenced")
        raise SubnetEpochError("database unavailable")

    monkeypatch.setattr(weights, "get_subtensor", lambda: Subtensor())
    monkeypatch.setattr(weights, "get_epoch_mode", lambda: LEGACY_EPOCH_MODE)
    monkeypatch.setattr(weights, "assert_legacy_epoch_namespace_open_async", gate)

    with pytest.raises(HTTPException) as exc:
        await weights._verify_epoch_block_authority(
            netuid=71,
            epoch_id=101,
            submitted_block=36_360,
            require_submission_window=False,
        )
    assert exc.value.status_code == status


@pytest.mark.asyncio
async def test_research_lab_fence_bypasses_cache_and_hints(monkeypatch):
    from gateway.research_lab import chain
    from gateway.utils import epoch

    monkeypatch.setattr(chain, "get_epoch_mode", lambda: LEGACY_EPOCH_MODE)
    monkeypatch.setattr(
        epoch,
        "get_legacy_epoch_namespace_state_async",
        lambda **_kwargs: _async_value(_state("cutover_fenced")),
    )

    async def gateway_unavailable():
        raise RuntimeError("gateway chain unavailable")

    monkeypatch.setattr(epoch, "get_current_epoch_id_async", gateway_unavailable)
    monkeypatch.setattr(
        chain,
        "_fetch_current_chain_epoch_direct",
        lambda: (100, 36_359, "finney"),
    )
    monkeypatch.setattr(
        chain,
        "_read_gateway_epoch_hint",
        lambda **_kwargs: pytest.fail("fenced path used a gateway hint"),
    )

    async def allow_last(epoch_id, **_kwargs):
        assert epoch_id == 100
        return _state("cutover_fenced")

    monkeypatch.setattr(
        epoch,
        "assert_legacy_epoch_namespace_open_async",
        allow_last,
    )
    chain._EPOCH_CACHE = (99, 35_999, "stale", 10**12)

    assert await chain.resolve_research_lab_evaluation_epoch() == (
        100,
        36_359,
        "direct_subtensor_fenced:finney",
    )
    assert chain._EPOCH_CACHE is None


@pytest.mark.asyncio
async def test_research_lab_configured_override_is_disabled_after_fence(monkeypatch):
    from gateway.research_lab import chain
    from gateway.utils import epoch

    monkeypatch.setattr(chain, "get_epoch_mode", lambda: LEGACY_EPOCH_MODE)
    monkeypatch.setattr(
        epoch,
        "get_legacy_epoch_namespace_state_async",
        lambda **_kwargs: _async_value(_state("cutover_fenced")),
    )
    with pytest.raises(RuntimeError, match="overrides are forbidden"):
        await chain.resolve_research_lab_evaluation_epoch(100)


@pytest.mark.asyncio
async def test_legacy_monitor_schedules_nothing_for_reserved_epoch(monkeypatch):
    from gateway.tasks import epoch_monitor
    from gateway.utils import epoch

    monitor = epoch_monitor.EpochMonitor()

    async def fenced(*_args, **_kwargs):
        raise epoch.LegacyEpochNamespaceFencedError("fenced")

    monkeypatch.setattr(
        epoch,
        "assert_legacy_epoch_namespace_open_async",
        fenced,
    )
    await monitor._process_block(36_360)

    assert monitor.last_epoch is None
    assert monitor.startup_block_count == 0
    assert monitor.initializing_epochs == set()


async def _async_value(value):
    return value
