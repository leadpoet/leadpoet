"""Terminal cutover authority must survive transient database outages.

Migration 101 permits only ``legacy_open -> cutover_fenced ->
stateful_staged -> stateful_active`` and gives service_role SELECT-only access
to the singleton. A process that already proved ``stateful_active`` therefore
must not turn a PostgREST outage into a later weight-submission outage.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    SubnetEpochCutover,
    SubnetEpochError,
)
from gateway.utils import epoch as epoch_utils

GENESIS = "0x" + "aa" * 32


def _cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash=GENESIS,
        netuid=71,
        cutover_block=8_660_000,
        cutover_block_hash="0x" + "66" * 32,
        first_subnet_epoch_index=24_000,
        first_settlement_epoch_id=24_073,
        last_legacy_epoch_id=24_072,
    )


@pytest.fixture
def authority_env(monkeypatch):
    cutover = _cutover()
    monkeypatch.setenv(CUTOVER_JSON_ENV, json.dumps(cutover.to_dict()))
    monkeypatch.setattr(
        epoch_utils, "_configured_cutover_service_authority_enabled", lambda: True
    )
    monkeypatch.setattr(
        epoch_utils,
        "_fixed_public_cutover_authority_enabled",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(epoch_utils, "_validated_cutover_authority_hash", None)
    monkeypatch.setattr(epoch_utils, "_validated_terminal_cutover_state", None)
    monkeypatch.setattr(epoch_utils, "_cutover_state_cache", None)
    # No Supabase credentials in tests: the singleton check is the authority.
    import gateway.config as gateway_config

    monkeypatch.setattr(gateway_config, "SUPABASE_URL", "", raising=False)
    monkeypatch.setattr(
        gateway_config, "SUPABASE_SERVICE_ROLE_KEY", "", raising=False
    )
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    return cutover


def test_terminal_authority_is_not_revalidated_after_success(
    monkeypatch, authority_env
):
    cutover = authority_env
    calls = {"n": 0}

    def active_then_unavailable(**_kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("PostgREST timeout")
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
        }

    monkeypatch.setattr(
        epoch_utils,
        "get_cutover_state",
        active_then_unavailable,
    )
    for _ in range(100):
        epoch_utils._validate_cutover_authority_sync(cutover)
    assert calls["n"] == 1


def test_terminal_lifecycle_force_refresh_survives_one_hundred_epochs(
    monkeypatch, authority_env
):
    cutover = authority_env
    calls = {"n": 0}

    def active_then_unavailable(**_kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise SubnetEpochError("durable authority unavailable")
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
            "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
            "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        }

    monkeypatch.setattr(
        epoch_utils,
        "_read_cutover_state_from_db_sync",
        active_then_unavailable,
    )
    for _ in range(100):
        state = epoch_utils.validate_epoch_runtime_lifecycle(
            cutover=cutover,
            force_refresh=True,
        )
        assert state["lifecycle_state"] == "stateful_active"
        assert state["mapping_hash"] == cutover.mapping_hash
    assert calls["n"] == 1


def test_preactive_lifecycle_still_revalidates_and_fails_closed(
    monkeypatch, authority_env
):
    cutover = authority_env
    calls = {"n": 0}

    def staged_then_unavailable(**_kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise SubnetEpochError("durable authority unavailable")
        return {
            "lifecycle_state": "stateful_staged",
            "mapping_hash": cutover.mapping_hash,
            "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
            "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        }

    monkeypatch.setattr(
        epoch_utils,
        "_read_cutover_state_from_db_sync",
        staged_then_unavailable,
    )
    with pytest.raises(SubnetEpochError, match="does not match"):
        epoch_utils.validate_epoch_runtime_lifecycle(
            cutover=cutover,
            force_refresh=True,
        )
    with pytest.raises(SubnetEpochError, match="unavailable"):
        epoch_utils.validate_epoch_runtime_lifecycle(
            cutover=cutover,
            force_refresh=True,
        )
    assert calls["n"] == 2


def test_concurrent_terminal_validation_reads_authority_once(
    monkeypatch, authority_env
):
    cutover = authority_env
    calls = {"n": 0}

    def slow_active_read(**_kwargs):
        calls["n"] += 1
        time.sleep(0.01)
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
            "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
            "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        }

    monkeypatch.setattr(
        epoch_utils,
        "_read_cutover_state_from_db_sync",
        slow_active_read,
    )
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(
            executor.map(
                lambda _index: epoch_utils.validate_epoch_runtime_lifecycle(
                    cutover=cutover,
                    force_refresh=True,
                ),
                range(100),
            )
        )

    assert calls["n"] == 1
    assert all(
        state["lifecycle_state"] == "stateful_active"
        and state["mapping_hash"] == cutover.mapping_hash
        for state in results
    )


def test_gateway_receipt_ledger_is_verified_once_then_survives_outage(
    monkeypatch, authority_env
):
    from gateway.db import client as db_client

    cutover = authority_env
    state_calls = {"n": 0}
    ledger_calls = {"n": 0}

    def active_then_unavailable(**_kwargs):
        state_calls["n"] += 1
        if state_calls["n"] > 1:
            raise RuntimeError("PostgREST timeout")
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
        }

    class LedgerQuery:
        def table(self, name):
            assert name == "research_lab_stateful_subnet_epoch_cutovers_v1"
            return self

        def select(self, columns):
            assert columns == "mapping_hash,manifest_doc"
            return self

        def eq(self, field, value):
            assert (field, value) == ("mapping_hash", cutover.mapping_hash)
            return self

        def limit(self, value):
            assert value == 2
            return self

        def execute(self):
            ledger_calls["n"] += 1
            return SimpleNamespace(
                data=[
                    {
                        "mapping_hash": cutover.mapping_hash,
                        "manifest_doc": cutover.to_dict(),
                    }
                ]
            )

    monkeypatch.setenv("SUPABASE_URL", "https://gateway-authority.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setattr(
        epoch_utils,
        "get_cutover_state",
        active_then_unavailable,
    )
    monkeypatch.setattr(
        db_client,
        "create_http1_sync_client",
        lambda *_args: LedgerQuery(),
    )

    for _ in range(100):
        epoch_utils._validate_cutover_authority_sync(cutover)

    assert state_calls["n"] == 1
    assert ledger_calls["n"] == 1
