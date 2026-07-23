"""The cutover-authority success cache must expire and honor regressions.

Confirmed in production: a gateway validated while the lifecycle was
active kept serving weight-input requests after the singleton moved back
to ``cutover_fenced``, because the success cache never re-checked. The
cache now re-validates on a cadence so a re-fence or rollback is honored
by long-running processes.
"""

import json

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
    monkeypatch.setattr(epoch_utils, "_validated_cutover_authority_at", 0.0)
    # No Supabase credentials in tests: the singleton check is the authority.
    import gateway.config as gateway_config

    monkeypatch.setattr(gateway_config, "SUPABASE_URL", "", raising=False)
    monkeypatch.setattr(
        gateway_config, "SUPABASE_SERVICE_ROLE_KEY", "", raising=False
    )
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    return cutover


def test_lifecycle_regression_is_honored_after_cache_expiry(
    monkeypatch, authority_env
):
    cutover = authority_env
    state = {"lifecycle_state": "stateful_active", "mapping_hash": cutover.mapping_hash}

    monkeypatch.setattr(
        epoch_utils, "get_cutover_state", lambda **_kwargs: dict(state)
    )
    epoch_utils._validate_cutover_authority_sync(cutover)  # passes, caches

    # Regression: operator re-fences. Within the revalidation window the
    # cache still answers (bounded staleness)...
    state["lifecycle_state"] = "cutover_fenced"
    epoch_utils._validate_cutover_authority_sync(cutover)

    # ...but once the cache expires the regression must be honored.
    monkeypatch.setattr(
        epoch_utils, "_CUTOVER_AUTHORITY_REVALIDATE_SECONDS", -1.0
    )
    with pytest.raises(SubnetEpochError):
        epoch_utils._validate_cutover_authority_sync(cutover)


def test_success_refreshes_cache_and_keeps_passing(monkeypatch, authority_env):
    cutover = authority_env
    calls = {"n": 0}

    def counting_state(**_kwargs):
        calls["n"] += 1
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
        }

    monkeypatch.setattr(epoch_utils, "get_cutover_state", counting_state)
    monkeypatch.setattr(
        epoch_utils, "_CUTOVER_AUTHORITY_REVALIDATE_SECONDS", -1.0
    )
    epoch_utils._validate_cutover_authority_sync(cutover)
    epoch_utils._validate_cutover_authority_sync(cutover)
    assert calls["n"] == 2  # expired cache re-checks every time
