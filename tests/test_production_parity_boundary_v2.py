from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_CHAIN_ARCHIVE_HOST,
    PRODUCTION_CHAIN_HOST,
    PRODUCTION_PARITY_BENCHMARK_DATE_ENV,
    PRODUCTION_PARITY_MODE_ENV,
    PRODUCTION_PARITY_RUN_ID_ENV,
    PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV,
    PRODUCTION_SUPABASE_ORIGIN,
    ProductionParityBoundaryV2Error,
    configured_rebenchmark_now_v2,
    configured_supabase_origin_v2,
    validate_production_parity_boundary_document_v2,
    validate_production_parity_boundary_v2,
)
from tests.v2_epoch_test_utils import epoch_test_environment


RUN_ID = "parity-20260815-abc123"
PARITY_ORIGIN = "https://d111111abcdef8.cloudfront.net"
HASH = "sha256:" + "a" * 64


def _environment(**overrides: str | None) -> dict[str, str | None]:
    return epoch_test_environment(
        **{
            PRODUCTION_PARITY_MODE_ENV: "enabled",
            PRODUCTION_PARITY_RUN_ID_ENV: RUN_ID,
            PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: PARITY_ORIGIN,
            PRODUCTION_PARITY_BENCHMARK_DATE_ENV: "2026-08-16",
            **overrides,
        }
    )


def test_production_boundary_is_unchanged_without_parity_configuration():
    assert configured_supabase_origin_v2({}) == PRODUCTION_SUPABASE_ORIGIN
    assert validate_production_parity_boundary_document_v2(
        {}, network="finney", netuid=71
    ) == {
        "mode": "production",
        "supabase_origin": PRODUCTION_SUPABASE_ORIGIN,
        "benchmark_date": None,
        "chain_host": PRODUCTION_CHAIN_HOST,
        "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
    }


@pytest.mark.parametrize(
    "environment",
    [
        {PRODUCTION_PARITY_MODE_ENV: "enabled"},
        {
            PRODUCTION_PARITY_MODE_ENV: "disabled",
            PRODUCTION_PARITY_RUN_ID_ENV: RUN_ID,
            PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: PARITY_ORIGIN,
            PRODUCTION_PARITY_BENCHMARK_DATE_ENV: "2026-08-16",
        },
        _environment(
            **{PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: "http://example.invalid"}
        ),
        _environment(
            **{
                PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV:
                    "https://qplwoislplkcegvdmbim.supabase.co"
            }
        ),
        _environment(**{PRODUCTION_PARITY_BENCHMARK_DATE_ENV: "not-a-date"}),
    ],
)
def test_parity_boundary_rejects_partial_or_unscoped_configuration(environment):
    with pytest.raises(ProductionParityBoundaryV2Error):
        configured_supabase_origin_v2(environment)


@pytest.mark.parametrize(
    ("network", "netuid"),
    [("test", 1), ("finney", 1), ("local", 71)],
)
def test_execution_config_rejects_parity_that_changes_production_chain_identity(
    network, netuid
):
    with pytest.raises(
        ResearchLabRuntimeConfigV2Error,
        match="production-parity boundary",
    ):
        build_research_lab_execution_config(
            environment=_environment(), network=network, netuid=netuid
        )


def test_parity_changes_only_database_and_benchmark_date():
    document = validate_production_parity_boundary_document_v2(
        _environment(), network="finney", netuid=71
    )
    assert document == {
        "mode": "production-parity",
        "run_id": RUN_ID,
        "supabase_origin": PARITY_ORIGIN,
        "benchmark_date": "2026-08-16",
        "chain_host": PRODUCTION_CHAIN_HOST,
        "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
    }
    observed = configured_rebenchmark_now_v2(
        environment=_environment(),
        now=datetime(2026, 8, 15, 23, 59, 12, tzinfo=timezone.utc),
    )
    assert observed.isoformat() == "2026-08-16T23:59:12+00:00"


def test_execution_config_binds_clone_origin_into_registry_and_reader():
    execution = build_research_lab_execution_config(
        environment=_environment(), network="finney", netuid=71
    )
    routes = provider_routes_for_execution_config(execution)
    assert routes["supabase"].hosts == ("d111111abcdef8.cloudfront.net",)
    assert routes["bittensor_chain"].hosts == (PRODUCTION_CHAIN_HOST,)
    assert routes["bittensor_archive"].hosts == (PRODUCTION_CHAIN_ARCHIVE_HOST,)
    assert provider_registry_hash(execution_config=execution) != provider_registry_hash()

    requests = []
    reader = SupabaseSourceReaderV2(
        execute_provider=lambda request: requests.append(dict(request)) or {},
        retry_policy_hash=HASH,
        origin=validate_production_parity_boundary_v2(
            execution["behavior_environment"], network="finney", netuid=71
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
