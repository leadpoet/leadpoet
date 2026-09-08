from __future__ import annotations

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


def _testnet401_cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash=(
            "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        ),
        netuid=401,
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
    from gateway.utils import epoch

    cutover = _cutover()
    observed = {}

    def public_rpc(name, *, timeout_seconds):
        observed["rpc"] = name
        observed["timeout_seconds"] = timeout_seconds
        return [_active_state(cutover)]

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(epoch, "call_public_rpc", public_rpc)

    assert epoch._read_cutover_state_from_db_sync() == _active_state(cutover)
    assert observed == {
        "rpc": "research_lab_stateful_subnet_epoch_cutover_public_state_v1",
        "timeout_seconds": 30.0,
    }


def test_public_lifecycle_rpc_outage_fails_closed(monkeypatch):
    from gateway.utils import epoch

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(epoch, "call_public_rpc", unavailable)

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
    from gateway.utils import epoch

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", network)
    monkeypatch.setenv("BITTENSOR_NETUID", netuid)

    with pytest.raises(
        SubnetEpochError,
        match="durable epoch namespace state database is unavailable",
    ):
        epoch._read_cutover_state_from_db_sync()


def test_configured_non_finney_runtime_reads_keyed_fresh_authority(monkeypatch):
    from gateway.db import client as db_client
    from gateway.utils import epoch

    cutover = _testnet401_cutover()
    observed = []

    class Result:
        data = [{
            "schema_version": "leadpoet.subnet_epoch_cutover_authority.v3",
            "mapping_hash": cutover.mapping_hash,
            "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
            "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
            "network_genesis_hash": cutover.network_genesis_hash,
            "netuid": cutover.netuid,
        }]

    class Query:
        def table(self, name):
            observed.append(("table", name))
            return self

        def select(self, columns):
            observed.append(("select", columns))
            return self

        def eq(self, field, value):
            observed.append(("eq", field, value))
            return self

        def limit(self, count):
            observed.append(("limit", count))
            return self

        def execute(self):
            return Result()

    monkeypatch.setenv("SUPABASE_URL", "https://test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("BITTENSOR_NETWORK", "test")
    monkeypatch.setenv("BITTENSOR_NETUID", str(cutover.netuid))
    monkeypatch.setattr(epoch, "_load_cutover", lambda: cutover)
    monkeypatch.setattr(db_client, "get_write_client", Query)

    assert epoch._read_cutover_state_from_db_sync() == _active_state(cutover)
    assert ("table", "research_lab_stateful_subnet_epoch_cutovers_v1") in observed
    assert ("eq", "network_genesis_hash", cutover.network_genesis_hash) in observed
    assert ("eq", "netuid", cutover.netuid) in observed


@pytest.mark.parametrize(
    ("network", "netuid"),
    [(None, None), ("finney", "71")],
)
def test_configured_service_keeps_singleton_path_without_manifest(
    monkeypatch,
    network,
    netuid,
):
    from gateway.db import client as db_client
    from gateway.utils import epoch

    cutover = _cutover()
    observed = []

    class Result:
        data = [_active_state(cutover)]

    class Query:
        def table(self, name):
            observed.append(("table", name))
            return self

        def select(self, _columns):
            return self

        def eq(self, field, value):
            observed.append(("eq", field, value))
            return self

        def limit(self, _count):
            return self

        def execute(self):
            return Result()

    monkeypatch.setenv("SUPABASE_URL", "https://test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    if network is None:
        monkeypatch.delenv("BITTENSOR_NETWORK", raising=False)
        monkeypatch.delenv("BITTENSOR_NETUID", raising=False)
    else:
        monkeypatch.setenv("BITTENSOR_NETWORK", network)
        monkeypatch.setenv("BITTENSOR_NETUID", netuid)

    def manifest_must_not_load():
        raise AssertionError("singleton service reads must not load a manifest")

    monkeypatch.setattr(epoch, "_load_cutover", manifest_must_not_load)
    monkeypatch.setattr(db_client, "get_write_client", Query)

    assert epoch._read_cutover_state_from_db_sync() == _active_state(cutover)
    assert observed[0] == (
        "table",
        "research_lab_stateful_subnet_epoch_cutover_state_v1",
    )
    assert ("eq", "singleton", True) in observed


def test_configured_test401_service_rejects_mismatched_manifest(monkeypatch):
    from gateway.db import client as db_client
    from gateway.utils import epoch

    monkeypatch.setenv("SUPABASE_URL", "https://test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("BITTENSOR_NETWORK", "test")
    monkeypatch.setenv("BITTENSOR_NETUID", "401")
    monkeypatch.setattr(epoch, "_load_cutover", _cutover)

    def client_must_not_query():
        raise AssertionError("mismatched manifest must fail before a DB query")

    monkeypatch.setattr(db_client, "get_write_client", client_must_not_query)

    with pytest.raises(
        SubnetEpochError,
        match="durable epoch namespace state database is unavailable",
    ) as exc_info:
        epoch._read_cutover_state_from_db_sync()
    assert isinstance(exc_info.value.__cause__, SubnetEpochError)
    assert "does not match testnet401" in str(exc_info.value.__cause__)


def test_runtime_lifecycle_requires_active_exact_mapping(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    monkeypatch.setattr(epoch, "_cutover_state_cache", None)
    monkeypatch.setattr(epoch, "_validated_terminal_cutover_state", None)
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


def test_forced_refresh_reuses_terminal_active_mapping_during_outage(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    monkeypatch.setenv("SUPABASE_URL", "https://test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setattr(
        epoch,
        "_validated_terminal_cutover_state",
        (
            (
                ("configured_service", "https://test"),
                cutover.mapping_hash,
            ),
            _active_state(cutover),
        ),
    )

    def unavailable():
        raise SubnetEpochError("database unavailable")

    monkeypatch.setattr(epoch, "_read_cutover_state_from_db_sync", unavailable)
    assert epoch.validate_epoch_runtime_lifecycle(
        cutover=cutover,
        force_refresh=True,
    ) == _active_state(cutover)


def test_forced_refresh_never_uses_stale_preactive_mapping(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    monkeypatch.setenv("SUPABASE_URL", "https://test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setattr(epoch, "_validated_terminal_cutover_state", None)
    monkeypatch.setattr(
        epoch,
        "_cutover_state_cache",
        (
            ("configured_service", "https://test"),
            10**12,
            {
                **_active_state(cutover),
                "lifecycle_state": "stateful_staged",
            },
        ),
    )

    def unavailable():
        raise SubnetEpochError("database unavailable")

    monkeypatch.setattr(epoch, "_read_cutover_state_from_db_sync", unavailable)
    with pytest.raises(SubnetEpochError, match="database unavailable"):
        epoch.get_cutover_state(force_refresh=True)
