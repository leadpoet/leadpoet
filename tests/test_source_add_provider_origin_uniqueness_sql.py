"""Fail-closed checks for SOURCE_ADD exact-host provider uniqueness."""

from pathlib import Path

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
)
from research_lab.source_add_identity import (
    normalize_source_add_provider_origin,
    source_provider_origin_hash,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/168-research-lab-source-add-provider-origin-uniqueness.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def _function(name: str, next_name: str) -> str:
    return SQL.split(f"FUNCTION public.{name}", 1)[1].split(
        f"FUNCTION public.{next_name}", 1
    )[0]


def test_migration_is_transactional_append_only_and_fails_closed():
    assert SQL.startswith("-- Reserve one path-independent exact provider host")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    assert "extensions.digest(bytea,text)" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()
    assert "prevent_research_lab_source_add_provider_origin_mutation" in SQL
    assert "SOURCE_ADD must be paused before provider-origin migration" in SQL
    assert "SOURCE_ADD work is leased during provider-origin migration" in SQL
    assert "work_status = 'leased'" in SQL
    quiet = SQL.split("DO $quiet_pause$", 1)[1].split("$quiet_pause$;", 1)[0]
    control_lock = (
        "LOCK TABLE public.research_lab_source_add_control\n"
        "        IN ACCESS EXCLUSIVE MODE NOWAIT"
    )
    assert control_lock in quiet
    for table in (
        "research_lab_source_add_work_items",
        "research_lab_source_add_submissions",
        "research_lab_source_add_functional_probe_attempts",
        "research_lab_source_catalog",
        "research_lab_source_add_provisioning_events",
        "research_lab_source_add_reward_intents",
        "research_lab_source_add_reward_obligations",
    ):
        assert f"public.{table}" in quiet
    assert "IN SHARE ROW EXCLUSIVE MODE NOWAIT" in quiet
    assert quiet.index(control_lock) < quiet.index(
        "public.research_lab_source_add_work_items"
    )
    assert (
        "LOCK TABLE public.research_lab_source_add_provider_origin_events\n"
        "    IN SHARE ROW EXCLUSIVE MODE NOWAIT"
    ) in SQL
    assert SQL.count("LOCK TABLE") == 3
    for work_kind in (
        "provenance",
        "functional_probe",
        "provisioning_smoke",
        "leg1_reward",
    ):
        assert f"'{work_kind}'" in SQL


def test_exact_host_hash_excludes_paths_and_preserves_distinct_subdomains():
    host = _function(
        "research_lab_source_add_provider_origin_host_v1",
        "research_lab_source_add_provider_origin_hash_v1",
    )
    digest = _function(
        "research_lab_source_add_provider_origin_hash_v1",
        "prevent_research_lab_source_add_provider_origin_mutation",
    )
    assert "split_part(v_remainder, '/', 1)" in host
    assert "left(v_host, 4) = 'www.'" in host
    assert "identity_kind\":\"provider_origin" in digest
    assert "identity_version\":\"v1" in digest
    assert "provider_host" in digest
    assert "normalized.provider_host" in digest


@pytest.mark.parametrize(
    "value",
    (
        "https://api.example.com:/v1",
        "https://api.example.com:0443/v1",
        "https://api.example.com:0/v1",
        "https://[::ffff:192.0.2.1]/v1",
        "https://[::ffff:c000:201]/v1",
        "https://[fe80::1%25eth0]/v1",
    ),
)
def test_python_origin_rejects_noncanonical_authorities(value):
    assert normalize_source_add_provider_origin(value) == ""
    assert source_provider_origin_hash(value) == ""


def test_backfill_reconciles_collisions_and_aborts_unsafe_owners():
    assert "source_add_provider_origin_backfill" in SQL
    assert "permanent_adapters" in SQL
    assert "research_lab_source_catalog" in SQL
    assert "research_lab_source_add_reward_intents" in SQL
    assert "research_lab_source_add_reward_obligations" in SQL
    assert "provider-origin permanent owner is orphaned" in SQL
    assert "provider-origin backfill input is malformed" in SQL
    assert "COUNT(DISTINCT submission_id) > 1" in SQL
    assert "source_add_provider_origin_losers" in SQL
    assert "backfill.permanent_owner DESC" in SQL
    assert "backfill.admitted_at ASC" in SQL
    assert "provider-origin has multiple permanent owners" in SQL
    assert "duplicate_provider_origin_existing_owner" in SQL
    assert "provider-origin reconciliation differs" in SQL
    assert "provider-origin backfill coverage differs" in SQL
    assert "IN SHARE ROW EXCLUSIVE MODE" in SQL


def test_v2_admission_and_recheck_lock_and_reserve_origin_atomically():
    admission = _function(
        "research_lab_source_add_admit_v2",
        "research_lab_source_add_requeue_provenance_v2",
    )
    recheck = _function(
        "research_lab_source_add_requeue_provenance_v2",
        "research_lab_source_add_provider_origin_contract_v1",
    )
    for section in (admission, recheck):
        assert "source-add-provider-origin:" in section
        assert "source-add-identity:" in section
        assert "ORDER BY lock_key" in section
        assert "pg_advisory_xact_lock" in section
        assert "reservation_status = 'reserved'" in section
        assert "jsonb_build_object('status', 'duplicate')" in section
    assert "public.research_lab_source_add_admit(" in admission
    assert "admission_v2_not_admitted" in admission
    assert "public.research_lab_source_add_requeue_provenance(" in recheck
    assert "provenance_recheck_v2_not_queued" in recheck
    assert "DROP FUNCTION IF EXISTS public.research_lab_source_add_admit" not in SQL


def test_terminal_release_never_releases_a_rewarded_or_cataloged_owner():
    release = _function(
        "release_research_lab_source_add_provider_origin_terminal",
        "assert_research_lab_source_add_provider_origin_owner",
    )
    assert "'rejected', 'rejected_precheck', 'functional_probe_failed'" in release
    assert "research_lab_source_add_reward_intents" in release
    assert "research_lab_source_add_reward_obligations" in release
    assert "research_lab_source_catalog" in release
    assert "'released'" in release
    assert "'terminal_pre_reward'" in release


def test_catalog_and_provisioning_inserts_assert_exact_origin_owner():
    assert "trg_source_catalog_provider_origin" in SQL
    assert "trg_source_add_provision_provider_origin" in SQL
    assert SQL.count("assert_research_lab_source_add_provider_origin_owner(") >= 3
    assert "provider-origin owner is unavailable" in SQL
    assert "source_metadata,api_base_url" in SQL
    assert "IN ('', current.provider_origin_hash)" in SQL


def test_origin_authority_is_private_and_required_by_restart_preflight():
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert "TO service_role" in SQL
    for field in (
        "terminal_release_trigger_enabled",
        "append_only_trigger_enabled",
        "row_level_security_enabled",
        "service_role_policy_enabled",
    ):
        assert field in SQL
    assert (
        MIGRATION,
        "research_lab_source_add_provider_origin_current",
        (
            "origin_version",
            "provider_origin_hash",
            "submission_id",
            "adapter_id",
            "miner_hotkey",
            "reservation_status",
            "seq",
        ),
    ) in REQUIRED_SUPABASE_V2_SCHEMA
    required = {
        function
        for migration, function in REQUIRED_SUPABASE_V2_RPCS
        if migration == MIGRATION
    }
    assert required == {
        "research_lab_source_add_admit_v2",
        "research_lab_source_add_requeue_provenance_v2",
        "research_lab_source_add_provider_origin_contract_v1",
    }
