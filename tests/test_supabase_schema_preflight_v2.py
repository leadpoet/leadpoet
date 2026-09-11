"""Arena-only gateway schema preflight contract."""
import json
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
    SupabaseSchemaPreflightV2Error,
    verify_required_supabase_v2_schema,
)


class _Response:
    def __init__(self, body=b"[]", status=200): self.body, self.status = body, status
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def getcode(self): return self.status
    def read(self, size=-1): return self.body if size < 0 else self.body[:size]


def _opener(*, missing=None, bad_capability=None, retired_storage_absent=False):
    paths = {f"/rpc/{name}": {} for _, name in REQUIRED_SUPABASE_V2_RPCS}
    retired_tables = {
        "validator_sourcing_epoch_inputs_v2",
        "research_lab_attested_ancestry_checkpoints_v2",
        "research_lab_attested_ancestry_activations_v2",
    }
    retired_rpcs = {
        "leadpoet_production_parity_reader_contract_v1",
        "persist_research_lab_ancestry_checkpoint_v2",
        "research_lab_ancestry_checkpoint_bootstrap_contract_v2",
        "research_lab_ancestry_disclosure_lookup_contract_v1",
        "research_lab_compact_checkpoint_graph_contract_v1",
    }
    if retired_storage_absent:
        for name in retired_rpcs:
            paths.pop(f"/rpc/{name}", None)
    if missing:
        paths.pop(f"/rpc/{missing}", None)
    capabilities = {
        "lab_arena_code_review_schema_v1": {"schema_version": "leadpoet.lab_arena.code_review.v1", "version": 207},
        "lab_arena_schema_version_v1": {"schema_version": "leadpoet.lab_arena.schema_version.v1", "version": 197},
        "lab_arena_weight_state_schema_v1": {"schema_version": "leadpoet.lab_arena.weight_state_schema.v1", "version": 202},
        "lab_arena_incentive_retirement_schema_v1": {"schema_version": "leadpoet.lab_arena.incentive_retirement_schema.v1", "version": 203},
    }
    def open_(request, timeout):
        path = urlparse(request.full_url).path
        if retired_storage_absent and path.rsplit("/", 1)[-1] in retired_tables:
            raise HTTPError(request.full_url, 404, "retired storage absent", {}, None)
        if missing and missing in path:
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        if path == "/rest/v1/":
            return _Response(json.dumps({"paths": paths}).encode())
        if "/rpc/" in path:
            name = path.rsplit("/", 1)[-1]
            value = capabilities[name]
            if bad_capability == name: value = dict(value, version=0)
            return _Response(json.dumps(value).encode())
        return _Response()
    return open_


def test_preflight_proves_arena_203_and_generic_scoring_schema_only():
    result = verify_required_supabase_v2_schema(
        {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"},
        opener=_opener(),
    )
    assert result["status"] == "ready"
    assert result["schema_capabilities"]["lab_arena_incentive_retirement_schema_v1"]["version"] == 203


def test_preflight_accepts_current_schema_without_retired_host_receipt_storage():
    result = verify_required_supabase_v2_schema(
        {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"},
        opener=_opener(retired_storage_absent=True),
    )
    assert result["status"] == "ready"
    assert result["schema_capabilities"]["lab_arena_weight_state_schema_v1"]["version"] == 202


@pytest.mark.parametrize("missing", [
    "research_lab_provider_evidence_cache_v2",
    "research_lab_stateful_subnet_epoch_cutovers_v1",
    "put_research_lab_provider_evidence_cache_v2",
    "research_lab_stateful_subnet_epoch_cutover_public_state_v1",
    "lab_arena_publish_weight_state_v1",
])
def test_preflight_still_requires_active_scoring_epoch_and_arena_dependencies(missing):
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="required (schema|RPC)"):
        verify_required_supabase_v2_schema(
            {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"},
            opener=_opener(missing=missing),
        )


def test_first_transition_defers_only_retirement_203_after_arena_202_exists():
    result = verify_required_supabase_v2_schema(
        {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"},
        opener=_opener(missing="lab_arena_incentive_retirement_schema_v1"),
        defer_incentive_retirement=True,
    )
    assert result["incentive_retirement_deferred"] is True
    assert "lab_arena_weight_state_schema_v1" in result["schema_capabilities"]
    assert "lab_arena_incentive_retirement_schema_v1" not in result["schema_capabilities"]
    names = {name for _, name, _ in REQUIRED_SUPABASE_V2_SCHEMA}
    assert "lab_arena_accepted_weight_states" in names
    assert "research_lab_chain_realized_epoch_settlements_v1" not in names
    assert all("compact_weight" not in name and "allocation" not in name for _, name in REQUIRED_SUPABASE_V2_RPCS)


def test_preflight_fails_closed_for_missing_arena_table_and_wrong_retirement_capability():
    env = {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"}
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="202-arena"):
        verify_required_supabase_v2_schema(env, opener=_opener(missing="lab_arena_accepted_weight_states"))
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="capability differs"):
        verify_required_supabase_v2_schema(env, opener=_opener(bad_capability="lab_arena_incentive_retirement_schema_v1"))


def test_preflight_requires_credentials():
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="credentials"):
        verify_required_supabase_v2_schema({})


def test_arena_table_probes_execute_against_committed_migration_202():
    """JSON document fields must not become nonexistent PostgREST columns."""
    from tests.lab_arena.lab_arena_pg_harness import (
        DEFAULT_MIGRATIONS,
        LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION,
        database_with_lab_arena_migration,
    )

    end = DEFAULT_MIGRATIONS.index(LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION) + 1
    database = database_with_lab_arena_migration(DEFAULT_MIGRATIONS[:end])
    psycopg2, dsn = next(database)
    connection = psycopg2.connect(**dsn)
    try:
        from psycopg2 import sql

        with connection.cursor() as cursor:
            probes = [row for row in REQUIRED_SUPABASE_V2_SCHEMA
                      if row[0] == "scripts/" + LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION]
            assert len(probes) == 2
            for _, table, columns in probes:
                cursor.execute(sql.SQL("SELECT {} FROM public.{} LIMIT 0").format(
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.Identifier(table),
                ))
                assert cursor.fetchall() == []
    finally:
        connection.close()
        database.close()
