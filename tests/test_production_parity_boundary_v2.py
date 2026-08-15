from __future__ import annotations

import pytest

from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_PARITY_MODE_ENV,
    PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST_ENV,
    PRODUCTION_PARITY_CHAIN_HOST,
    PRODUCTION_PARITY_CHAIN_HOST_ENV,
    PRODUCTION_PARITY_RUN_ID_ENV,
    PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV,
    PRODUCTION_SUPABASE_ORIGIN,
    ProductionParityBoundaryV2Error,
    configured_supabase_origin_v2,
    validate_production_parity_boundary_v2,
)
from gateway.tee.provider_broker_v2 import (
    provider_registry_hash,
    provider_routes_for_execution_config,
)
from gateway.tee.research_lab_runtime_config_v2 import (
    ResearchLabRuntimeConfigV2Error,
    build_research_lab_execution_config,
)
from gateway.tee.supabase_source_v2 import (
    SupabaseSourceReaderV2,
    SupabaseSourceV2Error,
)
from tests.v2_epoch_test_utils import epoch_test_environment


RUN_ID = "parity-20260815-abc123"
PARITY_ORIGIN = f"https://database-{RUN_ID}.parity.example.com"
HASH = "sha256:" + "a" * 64


def _environment(**overrides: str | None) -> dict[str, str | None]:
    return epoch_test_environment(
        **{
            PRODUCTION_PARITY_MODE_ENV: "enabled",
            PRODUCTION_PARITY_RUN_ID_ENV: RUN_ID,
            PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: PARITY_ORIGIN,
            PRODUCTION_PARITY_CHAIN_HOST_ENV: PRODUCTION_PARITY_CHAIN_HOST,
            PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST_ENV: PRODUCTION_PARITY_CHAIN_HOST,
            **overrides,
        }
    )


def test_production_boundary_is_unchanged_without_parity_configuration():
    assert configured_supabase_origin_v2({}) == PRODUCTION_SUPABASE_ORIGIN
    assert (
        validate_production_parity_boundary_v2({}, network="finney", netuid=71)
        == PRODUCTION_SUPABASE_ORIGIN
    )


@pytest.mark.parametrize(
    "environment",
    [
        {PRODUCTION_PARITY_MODE_ENV: "enabled"},
        {
            PRODUCTION_PARITY_MODE_ENV: "disabled",
            PRODUCTION_PARITY_RUN_ID_ENV: RUN_ID,
            PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: PARITY_ORIGIN,
        },
        {
            PRODUCTION_PARITY_MODE_ENV: "enabled",
            PRODUCTION_PARITY_RUN_ID_ENV: RUN_ID,
            PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: "http://database-"
            + RUN_ID
            + ".parity.example.com",
        },
        {
            PRODUCTION_PARITY_MODE_ENV: "enabled",
            PRODUCTION_PARITY_RUN_ID_ENV: RUN_ID,
            PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: "https://other.parity.example.com",
        },
    ],
)
def test_parity_boundary_rejects_partial_or_unscoped_configuration(environment):
    with pytest.raises(ProductionParityBoundaryV2Error):
        configured_supabase_origin_v2(environment)


@pytest.mark.parametrize(
    ("network", "netuid"),
    [("finney", 71), ("test", 71), ("local", 1)],
)
def test_execution_config_rejects_parity_boundary_outside_isolated_testnet(
    network, netuid
):
    with pytest.raises(
        ResearchLabRuntimeConfigV2Error,
        match="production-parity boundary",
    ):
        build_research_lab_execution_config(
            environment=_environment(), network=network, netuid=netuid
        )


def test_execution_config_binds_clone_origin_into_provider_registry_and_reader():
    execution = build_research_lab_execution_config(
        environment=_environment(), network="test", netuid=1
    )
    routes = provider_routes_for_execution_config(execution)
    assert routes["supabase"].hosts == (f"database-{RUN_ID}.parity.example.com",)
    assert routes["bittensor_chain"].hosts == (PRODUCTION_PARITY_CHAIN_HOST,)
    assert routes["bittensor_archive"].hosts == (PRODUCTION_PARITY_CHAIN_HOST,)
    assert provider_registry_hash(execution_config=execution) != provider_registry_hash()

    requests = []
    reader = SupabaseSourceReaderV2(
        execute_provider=lambda request: requests.append(dict(request)) or {},
        retry_policy_hash=HASH,
        origin=validate_production_parity_boundary_v2(
            execution["behavior_environment"], network="test", netuid=1
        ),
        sleep=lambda _seconds: None,
    )
    with pytest.raises(SupabaseSourceV2Error, match="terminal attempt is missing"):
        reader.read(
            policy_id="banned_hotkeys",
            parameters={},
            job_id="parity-boundary-probe",
            purpose="research_lab.ban_input.v2",
            record_transport=lambda _attempt: None,
            record_artifact=lambda _artifact: None,
        )
    assert requests[0]["url"].startswith(PARITY_ORIGIN + "/rest/v1/")
