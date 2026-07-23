from __future__ import annotations

from types import SimpleNamespace

import pytest

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
)


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


def _active_state(cutover: SubnetEpochCutover) -> dict:
    return {
        "lifecycle_state": "stateful_active",
        "mapping_hash": cutover.mapping_hash,
        "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
        "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
    }


def test_missing_service_credentials_use_fixed_public_rpc(monkeypatch):
    from Leadpoet.utils import cloud_db
    from gateway import config
    from gateway.utils import epoch
    import supabase

    cutover = _cutover()
    observed = {}

    class Client:
        def rpc(self, name):
            observed["rpc"] = name
            return self

        def execute(self):
            return SimpleNamespace(data=[_active_state(cutover)])

    def create_client(url, key):
        observed["authority"] = (url, key)
        return Client()

    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(supabase, "create_client", create_client)

    assert epoch._read_cutover_state_from_db_sync() == _active_state(cutover)
    assert observed == {
        "authority": (cloud_db.SUPABASE_URL, cloud_db.SUPABASE_ANON_KEY),
        "rpc": "research_lab_stateful_subnet_epoch_cutover_public_state_v1",
    }


def test_public_lifecycle_rpc_outage_fails_closed(monkeypatch):
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
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(supabase, "create_client", lambda _url, _key: Client())

    with pytest.raises(
        SubnetEpochError,
        match="durable epoch namespace state database is unavailable",
    ):
        epoch._read_cutover_state_from_db_sync()


@pytest.mark.parametrize(
    ("network", "netuid"),
    [("test", "71"), ("finney", "72")],
)
def test_unconfigured_nonproduction_runtime_has_no_database_fallback(
    monkeypatch,
    network,
    netuid,
):
    from gateway import config
    from gateway.utils import epoch

    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", network)
    monkeypatch.setenv("BITTENSOR_NETUID", netuid)

    with pytest.raises(
        SubnetEpochError,
        match="durable epoch namespace state database is unavailable",
    ):
        epoch._read_cutover_state_from_db_sync()


def test_runtime_lifecycle_requires_active_exact_mapping(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    monkeypatch.setattr(epoch, "_cutover_state_cache", None)
    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: {
            **_active_state(cutover),
            "lifecycle_state": "stateful_staged",
        },
    )
    with pytest.raises(SubnetEpochError, match="does not match"):
        epoch.validate_epoch_runtime_lifecycle(cutover=cutover)

    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: {
            **_active_state(cutover),
            "mapping_hash": "sha256:" + "f" * 64,
        },
    )
    with pytest.raises(SubnetEpochError, match="does not match"):
        epoch.validate_epoch_runtime_lifecycle(cutover=cutover)

    monkeypatch.setattr(
        epoch,
        "_read_cutover_state_from_db_sync",
        lambda: _active_state(cutover),
    )
    assert epoch.validate_epoch_runtime_lifecycle(
        cutover=cutover
    )["mapping_hash"] == cutover.mapping_hash


def test_forced_refresh_never_uses_stale_active_mapping(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    monkeypatch.setattr(
        epoch,
        "_cutover_state_cache",
        (("configured_service", "test"), 10**12, _active_state(cutover)),
    )

    def unavailable():
        raise SubnetEpochError("database unavailable")

    monkeypatch.setattr(epoch, "_read_cutover_state_from_db_sync", unavailable)
    with pytest.raises(SubnetEpochError, match="database unavailable"):
        epoch.get_cutover_state(force_refresh=True)
