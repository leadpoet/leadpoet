"""Arena-only gateway schema preflight contract."""
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    PRIVATE_ARENA_MIGRATIONS,
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
    SupabaseSchemaPreflightV2Error,
    verify_required_supabase_v2_schema,
)


def test_required_migrations_use_canonical_deployment_filenames():
    root = Path(__file__).resolve().parents[1]
    paths = {row[0] for row in REQUIRED_SUPABASE_V2_SCHEMA + REQUIRED_SUPABASE_V2_RPCS}
    for relative in paths:
        path = root / relative
        assert path.is_file()
        assert re.fullmatch(r"[0-9]{2,4}-[a-z0-9][a-z0-9-]*\.sql", path.name)


def _environment():
    return {
        "SUPABASE_URL": "https://db.example",
        "SUPABASE_SERVICE_ROLE_KEY": "service-role-secret",
        "LAB_ARENA_SERVICE_KEY": "sb_secret_arena",
    }


class _Response:
    def __init__(self, body=b"[]", status=200): self.body, self.status = body, status
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def getcode(self): return self.status
    def read(self, size=-1): return self.body if size < 0 else self.body[:size]


def _opener(
    *, missing=None, bad_capability=None, retired_storage_absent=False,
    enforce_role_separation=False, enforce_url_separation=False,
):
    service_role_paths = {
        f"/rpc/{name}": {}
        for migration, name in REQUIRED_SUPABASE_V2_RPCS
        if migration not in PRIVATE_ARENA_MIGRATIONS
    }
    arena_paths = {
        f"/rpc/{name}": {}
        for migration, name in REQUIRED_SUPABASE_V2_RPCS
        if migration in PRIVATE_ARENA_MIGRATIONS
    }
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
            service_role_paths.pop(f"/rpc/{name}", None)
            arena_paths.pop(f"/rpc/{name}", None)
    if missing:
        service_role_paths.pop(f"/rpc/{missing}", None)
        arena_paths.pop(f"/rpc/{missing}", None)
    capabilities = {
        "lab_arena_participation_schema_v1": {
            "schema_version": "leadpoet.lab_arena.participation_schema.v1",
            "version": 218,
        },
        "lab_arena_code_review_schema_v1": {
            "schema_version": "leadpoet.lab_arena.code_review.v1",
            "version": 207,
            "claim_ttl_seconds": 600,
            "retry_backoff_seconds": 60,
            "max_attempts": 3,
        },
        "lab_arena_schema_version_v1": {"schema_version": "leadpoet.lab_arena.schema_version.v1", "version": 197},
        "lab_arena_weight_state_schema_v1": {"schema_version": "leadpoet.lab_arena.weight_state_schema.v1", "version": 202},
        "lab_arena_incentive_retirement_schema_v1": {"schema_version": "leadpoet.lab_arena.incentive_retirement_schema.v1", "version": 203},
    }
    def open_(request, timeout):
        path = urlparse(request.full_url).path
        request_headers = {
            name.lower(): value for name, value in request.header_items()
        }
        arena_authority = request_headers.get("apikey") == "sb_secret_arena"
        arena_url = urlparse(request.full_url).netloc == "arena.example"
        private_review = (
            path.endswith("/lab_arena_submissions")
            or path.endswith("/lab_arena_runs")
            or path.endswith("/rpc/lab_arena_code_review_schema_v1")
            or path.endswith("/rpc/lab_arena_participation_schema_v1")
        )
        if enforce_role_separation and private_review and not arena_authority:
            raise HTTPError(request.full_url, 403, "private", {}, None)
        if enforce_url_separation and arena_authority != arena_url:
            raise HTTPError(request.full_url, 404, "wrong project", {}, None)
        if retired_storage_absent and path.rsplit("/", 1)[-1] in retired_tables:
            raise HTTPError(request.full_url, 404, "retired storage absent", {}, None)
        if missing and missing in path:
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        if path == "/rest/v1/":
            paths = arena_paths if arena_authority else service_role_paths
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
        _environment(),
        opener=_opener(),
    )
    assert result["status"] == "ready"
    assert result["schema_capabilities"]["lab_arena_incentive_retirement_schema_v1"]["version"] == 203
    assert result["schema_capabilities"]["lab_arena_participation_schema_v1"]["version"] == 218


def test_code_review_preflight_uses_only_the_scoped_arena_role():
    env = _environment()
    env["LAB_ARENA_SUPABASE_URL"] = "https://arena.example"
    result = verify_required_supabase_v2_schema(
        env,
        opener=_opener(
            enforce_role_separation=True,
            enforce_url_separation=True,
        ),
    )
    assert result["schema_capabilities"]["lab_arena_code_review_schema_v1"] == {
        "schema_version": "leadpoet.lab_arena.code_review.v1",
        "version": 207,
        "claim_ttl_seconds": 600,
        "retry_backoff_seconds": 60,
        "max_attempts": 3,
    }
    assert any(
        table == "lab_arena_submissions"
        for _migration, table, _columns in REQUIRED_SUPABASE_V2_SCHEMA
    )
    required_rpc_names = {name for _migration, name in REQUIRED_SUPABASE_V2_RPCS}
    assert "lab_arena_code_review_schema_v1" in required_rpc_names
    assert "lab_arena_begin_submission_review" in required_rpc_names
    assert "lab_arena_finish_submission_review" in required_rpc_names


def test_preflight_accepts_current_schema_without_retired_host_receipt_storage():
    result = verify_required_supabase_v2_schema(
        _environment(),
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
    "lab_arena_has_recent_participation_v1",
    "lab_arena_participation_schema_v1",
    "lab_arena_runs",
])
def test_preflight_still_requires_active_scoring_epoch_and_arena_dependencies(missing):
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="required (schema|RPC)"):
        verify_required_supabase_v2_schema(
            _environment(),
            opener=_opener(missing=missing),
        )


def test_preflight_rejects_participation_schema_before_original_judgment_fix():
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="participation_schema_v1 capability differs"):
        verify_required_supabase_v2_schema(
            _environment(),
            opener=_opener(bad_capability="lab_arena_participation_schema_v1"),
        )


def test_first_transition_defers_only_retirement_203_after_arena_202_exists():
    result = verify_required_supabase_v2_schema(
        _environment(),
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
    env = _environment()
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="202-arena"):
        verify_required_supabase_v2_schema(env, opener=_opener(missing="lab_arena_accepted_weight_states"))
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="capability differs"):
        verify_required_supabase_v2_schema(env, opener=_opener(bad_capability="lab_arena_incentive_retirement_schema_v1"))


def test_preflight_requires_credentials():
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="credentials"):
        verify_required_supabase_v2_schema({})
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="Arena schema credential"):
        verify_required_supabase_v2_schema({
            "SUPABASE_URL": "https://db.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-role-secret",
        })


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
