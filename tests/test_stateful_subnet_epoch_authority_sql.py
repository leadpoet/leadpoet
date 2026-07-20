import os
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import time
import uuid

import pytest

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "101-stateful-subnet-epoch-authority.sql"
).read_text(encoding="utf-8")
INDEX_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "100-stateful-subnet-epoch-high-water-indexes.concurrent.sql"
).read_text(encoding="utf-8")
HISTORICAL_PREDECESSOR_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "105-stateful-subnet-epoch-historical-predecessor-v2.sql"
).read_text(encoding="utf-8")
CUTOVER_TIMEOUT_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "107-stateful-epoch-cutover-rpc-timeouts.sql"
).read_text(encoding="utf-8")
INDEX_VALIDATION_MATCH = re.search(
    r"(DO \$\$.*?\n\$\$;)\n\n-- The DO block above",
    INDEX_SQL,
    re.DOTALL,
)
if INDEX_VALIDATION_MATCH is None:
    raise RuntimeError("stateful epoch index validation block is missing")
INDEX_VALIDATION_SQL = INDEX_VALIDATION_MATCH.group(1)
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    REPO_ROOT / "docs" / "stateful_subnet_epoch_cutover_runbook.md"
).read_text(encoding="utf-8")
AMCHECK_SQL_MATCH = re.search(
    r"```sql\n(BEGIN READ ONLY;.*?\$amcheck\$;\n\nCOMMIT;)\n```",
    RUNBOOK,
    re.DOTALL,
)
if AMCHECK_SQL_MATCH is None:
    raise RuntimeError("stateful epoch amcheck runbook block is missing")
AMCHECK_SQL = AMCHECK_SQL_MATCH.group(1)


TABLES = (
    "research_lab_stateful_subnet_epoch_candidates_v1",
    "research_lab_stateful_subnet_epoch_cutovers_v1",
    "research_lab_stateful_subnet_epoch_boundaries_v1",
    "research_lab_stateful_subnet_epoch_snapshots_v1",
)


def test_receipt_allowlist_retains_canonical_contract_and_adds_epoch_authorities():
    for role, canonical_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        migrated_purposes = set(re.findall(r"'([^']+)'", match.group(1)))
        expected_purposes = set(canonical_purposes)
        if role == "gateway_coordinator":
            expected_purposes.add("research_lab.subnet_epoch_cutover.v2")
        if role == "validator_weights":
            expected_purposes.add("validator.subnet_epoch_snapshot.v2")
        assert migrated_purposes == expected_purposes, role

    assert ") NOT VALID;" in SQL
    assert (
        "VALIDATE CONSTRAINT "
        "research_lab_attested_execution_receipts_v2_role_purpose_check"
    ) in SQL


def test_migration_is_additive_and_does_not_activate_a_cutover():
    for table in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS public.{table}" in SQL
        if table == "research_lab_stateful_subnet_epoch_cutovers_v1":
            assert SQL.count(f"INSERT INTO public.{table}") == 1
        else:
            assert f"INSERT INTO public.{table}" not in SQL
        assert f"DELETE FROM public.{table}" not in SQL

    assert "does not insert or activate a cutover" in SQL
    assert "An empty table means no cutover is activated" in SQL


def test_high_water_prerequisite_is_nontransactional_and_covers_live_gaps():
    executable = "\n".join(
        line for line in INDEX_SQL.splitlines() if not line.lstrip().startswith("--")
    )
    assert "BEGIN;" not in executable
    assert "COMMIT;" not in executable
    assert INDEX_SQL.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 13
    for relation_and_key in (
        "public.epoch_audit_logs(epoch_id DESC)",
        "public.published_weight_bundles(epoch_id DESC)",
        "public.research_lab_attested_weight_bundles(epoch_id DESC)",
        "public.research_lab_attested_weight_bundles_v2(epoch_id DESC)",
        "public.research_lab_champion_reward_obligations(evaluation_epoch DESC)",
        "public.research_lab_legacy_finalized_allocation_migrations_v2(epoch_id DESC)",
        "public.research_lab_private_model_benchmark_bundles(evaluation_epoch DESC)",
        "public.research_lab_scoring_runs(evaluation_epoch DESC)",
        "public.transparency_log(epoch_id DESC)",
        "public.validation_evidence_private(epoch_id DESC)",
        "public.validator_attestations(epoch_id DESC)",
        "public.validator_sourcing_epoch_inputs_v2(epoch_id DESC)",
    ):
        assert relation_and_key in INDEX_SQL
    assert "idx_transparency_log_payload_epoch_identity_v1" in INDEX_SQL
    assert "((payload->>'epoch_id')::BIGINT) DESC" in INDEX_SQL
    assert "SKIP_VIEW: research_lab_epoch_payouts" in INDEX_SQL
    assert "public.research_lab_epoch_payouts(epoch DESC)" not in INDEX_SQL
    assert "idx_rl_epoch_payouts_epoch_identity_v1" not in INDEX_SQL
    for contract_fragment in (
        "pg_stat_progress_create_index",
        "progress.datname = pg_catalog.current_database()",
        "index_namespace.nspname = 'public'",
        "index_meta.indrelid = table_relation.oid",
        "access_method.amname = 'btree'",
        "index_meta.indisvalid",
        "index_meta.indisready",
        "index_meta.indislive",
        "index_meta.indpred IS NULL",
        "index_meta.indexprs IS NULL",
        "index_meta.indnatts = 1",
        "index_meta.indnkeyatts = 1",
        "index_meta.indkey[0] = column_meta.attnum",
        "operator_class.opcdefault",
        "operator_class.opcmethod = index_relation.relam",
        "wrong exact definition",
    ):
        assert contract_fragment in INDEX_SQL
    assert "scripts/101-stateful-subnet-epoch-authority.sql" in INDEX_SQL


def test_cutover_is_single_per_chain_lineage_and_collision_safe():
    assert "UNIQUE (network_genesis_hash, netuid)" in SQL
    assert "mapping_hash" in SQL and "NOT NULL UNIQUE" in SQL
    assert "first_epoch_ref" in SQL
    assert "first_settlement_epoch_id::BIGINT = last_legacy_epoch_id::BIGINT + 1" in SQL
    assert "first_subnet_epoch_index" in SQL
    assert "previous_epoch_scheme" in SQL
    assert "legacy_global_360_v1" in SQL
    assert "bittensor.subnet_epoch_index.v1" in SQL
    assert "LOCK TABLE %I.%I IN SHARE MODE" in SQL
    assert "a.atttypid IN (20, 21, 23)" in SQL
    assert "a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')" in SQL
    assert "reward_expires_epoch must not raise the high-water mark" in SQL
    assert "transparency_log.payload.epoch_id are included" in SQL
    assert "payload ? 'epoch_id'" in SQL
    assert "stateful epoch pre-boundary fence expected high-water %" in SQL
    assert "prerequisite btree index is missing for %.%" in SQL
    assert "ORDER BY %1$I DESC LIMIT 1" in SQL
    assert "research_lab_attested_execution_receipts_v2" in SQL
    assert "WHERE receipt_hash <> ALL($2)" not in SQL
    assert "LANGUAGE plpgsql\nSECURITY DEFINER\nSET search_path = ''" in SQL
    for field in (
        "validator_hotkey",
        "candidate_payload_hash",
        "validator_hotkey_signature",
        "candidate_authorization_hash",
    ):
        assert field in SQL


def test_cutover_rpc_family_has_scoped_database_timeout_budget():
    functions = (
        "research_lab_stateful_subnet_epoch_cutover_preflight_v1",
        "research_lab_stateful_subnet_epoch_cutover_fence_v1",
        "research_lab_stateful_subnet_epoch_cutover_bind_v1",
        "research_lab_stateful_subnet_epoch_cutover_bind_v2",
        "research_lab_stateful_subnet_epoch_stage_v1",
        "research_lab_stateful_subnet_epoch_stage_v2",
        "research_lab_stateful_subnet_epoch_activate_v1",
    )
    for function in functions:
        assert f"ALTER FUNCTION public.{function}(" in CUTOVER_TIMEOUT_SQL
    assert CUTOVER_TIMEOUT_SQL.count("SET statement_timeout = '120s'") == len(
        functions
    )
    assert "ALTER ROLE" not in CUTOVER_TIMEOUT_SQL
    assert "ALTER DATABASE" not in CUTOVER_TIMEOUT_SQL


def test_boundary_mapping_is_bijective_affine_and_strictly_forward():
    assert "UNIQUE (mapping_hash, subnet_epoch_index)" in SQL
    assert "UNIQUE (mapping_hash, settlement_epoch_id)" in SQL
    assert "UNIQUE (mapping_hash, boundary_block)" in SQL
    assert "UNIQUE (mapping_hash, boundary_block_hash)" in SQL
    assert (
        "NEW.subnet_epoch_index - cutover_row.first_subnet_epoch_index"
        in SQL
    )
    assert "NEW.settlement_epoch_id::BIGINT IS DISTINCT FROM expected_settlement" in SQL
    assert "NEW.subnet_epoch_index <= cutover_row.first_subnet_epoch_index" in SQL
    assert "stateful epoch boundary epoch_ref is already mapped" in SQL
    assert "ORDER BY subnet_epoch_index DESC" in SQL
    assert "NEW.subnet_epoch_index <= latest_boundary_row.subnet_epoch_index" in SQL
    assert "latest_boundary_row.settlement_epoch_id::BIGINT" in SQL
    assert "NEW.boundary_block <= latest_boundary_row.boundary_block" in SQL
    assert (
        "stateful epoch boundary does not advance the latest accepted boundary"
        in SQL
    )
    assert "subnet_epoch_index = NEW.subnet_epoch_index - 1" not in SQL
    # Insert validation, read-only preflight, pre-boundary fence, binding,
    # generic fence writes, atomic stage, and explicit activation serialize.
    assert SQL.count("pg_catalog.pg_advisory_xact_lock(") >= 8
    assert "WHERE mapping_hash = NEW.mapping_hash\n        FOR UPDATE;" in SQL


def test_snapshot_requires_declared_boundary_and_exact_derived_fields():
    assert "NEW.subnet_epoch_index = cutover_row.first_subnet_epoch_index" in SQL
    assert "stateful epoch snapshot has no matching boundary mapping" in SQL
    assert "epoch_block = current_block - last_epoch_block" in SQL
    assert "blocks_remaining = GREATEST(0, next_epoch_block - current_block)" in SQL
    assert "UNIQUE (mapping_hash, current_block)" in SQL
    assert "UNIQUE (mapping_hash, block_hash)" in SQL
    assert "head_kind IN ('finalized', 'exact')" in SQL
    assert "NEW.current_block < boundary_row.boundary_block" in SQL
    assert "boundary_row.boundary_block IS DISTINCT FROM NEW.last_epoch_block" not in SQL
    assert "NEW.last_epoch_block IS DISTINCT FROM cutover_row.cutover_block" not in SQL
    assert SQL.count("50401 - blocks_since_last_step") == 3
    assert "WHEN blocks_since_last_step > 50400 THEN current_block" in SQL
    assert "WHEN blocks_since_last_step > 50400 THEN boundary_block" in SQL


def test_cutover_and_snapshots_are_bound_to_successful_v2_receipts():
    for purpose in (
        "research_lab.subnet_epoch_cutover.v2",
        "validator.subnet_epoch_snapshot.v2",
        "validator.weights.finalized.v2",
    ):
        assert purpose in SQL

    assert "receipt_row.receipt_status IS DISTINCT FROM 'succeeded'" in SQL
    assert "receipt_row.output_root IS DISTINCT FROM NEW.cutover_authority_hash" in SQL
    assert "receipt_row.output_root IS DISTINCT FROM NEW.first_snapshot_hash" in SQL
    assert "receipt_row.output_root IS DISTINCT FROM NEW.boundary_hash" in SQL
    assert "receipt_row.output_root IS DISTINCT FROM NEW.snapshot_hash" in SQL
    assert "publication.bundle_hash = finalization.bundle_hash" in SQL
    assert "finalization_row.bundle_hash IS DISTINCT FROM NEW.last_legacy_bundle_hash" in SQL
    assert (
        "finalization_row.finalization_receipt_hash IS DISTINCT FROM"
        in SQL
    )
    assert "finalization_row.finalized_block > NEW.cutover_block" in SQL
    assert "36 blocks in the observed cutover schedule" in SQL
    assert "receipt_row.receipt_doc->'parent_receipt_hashes'" in SQL
    assert "pg_catalog.jsonb_array_length(" in SQL
    assert "pg_catalog.jsonb_build_array(" in SQL
    assert "NEW.first_snapshot_receipt_hash" in SQL
    assert "NEW.last_legacy_finalization_receipt_hash" in SQL


def test_historical_predecessor_migration_separates_proof_from_high_water():
    migration = HISTORICAL_PREDECESSOR_SQL
    assert "predecessor_epoch_id BETWEEN 0 AND last_legacy_epoch_id" in migration
    assert "legacy_finalized_chain_migration_v2" in migration
    assert "research_lab_legacy_finalized_allocation_migrations_v2" in migration
    assert "research_lab.legacy_finalized_allocation.v2" in migration
    assert "predecessor_row.epoch_id > NEW.last_legacy_epoch_id" in migration
    assert "predecessor_row.output_root IS DISTINCT FROM" in migration
    assert "NEW.predecessor_authority_hash" in migration
    assert "predecessor_row.receipt_status IS DISTINCT FROM 'succeeded'" in migration
    assert "jsonb_array_length(" in migration
    assert "IS DISTINCT FROM 2" in migration
    assert "research_lab_stateful_subnet_epoch_cutover_bind_v2" in migration
    assert "research_lab_stateful_subnet_epoch_stage_v2" in migration
    assert "UPDATE public.research_lab_legacy_finalized" not in migration
    assert "DELETE FROM public.research_lab_legacy_finalized" not in migration


def test_json_documents_are_exact_shape_and_secret_scrubbed():
    for document in (
        "manifest_doc",
        "first_snapshot_doc",
        "authority_doc",
        "boundary_doc",
        "snapshot_doc",
    ):
        assert f"jsonb_typeof({document}) = 'object'" in SQL
        assert f"{document}::TEXT !~*" in SQL

    assert "openrouter_api_key" in SQL
    assert "service_role" in SQL
    assert "proxy-authorization" in SQL
    assert "first_snapshot_doc - ARRAY[" in SQL
    assert "authority_doc - ARRAY[" in SQL
    assert "(boundary_doc->'snapshot') - ARRAY[" in SQL
    assert "snapshot_doc - ARRAY[" in SQL


def test_tables_are_append_only_and_service_role_only():
    for table in TABLES:
        assert f"ALTER TABLE public.{table}\n    ENABLE ROW LEVEL SECURITY" in SQL
        assert f"ON TABLE public.{table}\n    TO service_role" in SQL
        assert f"ON TABLE public.{table}\n    FROM PUBLIC, anon, authenticated" in SQL

    assert SQL.count("prevent_research_lab_attested_v2_mutation();") == 4
    assert "BEFORE UPDATE OR DELETE" in SQL
    assert "GRANT SELECT, INSERT" in SQL
    assert "GRANT UPDATE" not in SQL
    assert "GRANT DELETE" not in SQL


def test_public_cutover_state_rpc_is_sanitized_and_read_only():
    rpc = "research_lab_stateful_subnet_epoch_cutover_public_state_v1"
    assert f"CREATE OR REPLACE FUNCTION public.{rpc}()" in SQL
    assert "LANGUAGE SQL\nSTABLE\nSECURITY DEFINER\nSET search_path = ''" in SQL
    assert (
        f"REVOKE ALL ON FUNCTION public.{rpc}()\n"
        "    FROM PUBLIC, anon, authenticated, service_role"
    ) in SQL
    assert (
        f"ON FUNCTION public.{rpc}()\n"
        "    TO anon, authenticated, service_role"
    ) in SQL
    expected_fields = {
        "lifecycle_state",
        "mapping_hash",
        "network_genesis_hash",
        "netuid",
        "last_legacy_epoch_id",
        "first_settlement_epoch_id",
        "fenced_at",
        "staged_at",
        "activated_at",
        "updated_at",
    }
    signature = re.search(
        rf"RETURNS TABLE \((.*?)\n\)\nLANGUAGE SQL",
        SQL[SQL.index(f"CREATE OR REPLACE FUNCTION public.{rpc}()"):],
        re.DOTALL,
    )
    assert signature is not None
    returned_fields = set(re.findall(r"^\s*([a-z_]+)\s+", signature.group(1), re.MULTILINE))
    assert returned_fields == expected_fields
    for protected_field in (
        "candidate_snapshot_hash",
        "candidate_receipt_hash",
        "last_legacy_finalization_receipt_hash",
        "cutover_authority_hash",
        "cutover_receipt_hash",
        "initialization_nonce",
        "initialization_payload_hash",
    ):
        assert protected_field not in returned_fields


def test_mapping_view_is_security_invoker_and_has_collision_proof_query():
    assert "CREATE OR REPLACE VIEW public.research_lab_stateful_subnet_epoch_mapping_v1" in SQL
    assert "WITH (security_invoker = true)" in SQL
    assert "TRUE AS is_cutover_boundary" in SQL
    assert "UNION ALL" in SQL
    assert "FALSE AS is_cutover_boundary" in SQL
    assert "HAVING COUNT(DISTINCT settlement_epoch_id) <> 1" in SQL
    assert "HAVING COUNT(DISTINCT subnet_epoch_index) <> 1" in SQL
    assert "Contiguous affine mapping proof (must return zero rows)" in SQL
    assert "research_lab_stateful_subnet_epoch_cutover_preflight_v1" in SQL
    assert "p_cutover_receipt_hash TEXT DEFAULT NULL" in SQL


def test_postgres_15_happy_adversarial_rerun_and_locking_contract():
    """Exercise the migration against PostgreSQL 15 when explicitly enabled.

    The normal unit suite stays hermetic. Release verification runs this test
    with RUN_POSTGRES_15_INTEGRATION=1 on a machine with Docker.
    """

    if os.environ.get("RUN_POSTGRES_15_INTEGRATION") != "1":
        pytest.skip("set RUN_POSTGRES_15_INTEGRATION=1 for the PostgreSQL 15 test")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the PostgreSQL 15 integration test")

    container = f"leadpoet-stateful-epoch-pg15-{uuid.uuid4().hex[:10]}"
    background_processes = []

    def sha(number: int) -> str:
        return f"sha256:{number:064x}"

    def raw(number: int) -> str:
        return f"0x{number:064x}"

    def hex64(number: int) -> str:
        return f"{number:064x}"

    def psql(statement: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "psql",
                "-X",
                "-A",
                "-t",
                "-U",
                "postgres",
                "-d",
                "leadpoet",
                "-v",
                "ON_ERROR_STOP=1",
            ],
            input=statement,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if check and result.returncode != 0:
            raise AssertionError(
                f"psql failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def rejected(statement: str, message: str) -> None:
        result = psql(statement, check=False)
        assert result.returncode != 0, result.stdout
        assert message in result.stderr, result.stderr

    def background_psql(statement: str, application_name: str):
        process = subprocess.Popen(
            [
                "docker",
                "exec",
                "--env",
                f"PGAPPNAME={application_name}",
                "-i",
                container,
                "psql",
                "-X",
                "-A",
                "-t",
                "-U",
                "postgres",
                "-d",
                "leadpoet",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                statement,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        background_processes.append(process)
        return process

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container,
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--env",
                "POSTGRES_DB=leadpoet",
                "postgres:15",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        for _ in range(60):
            ready = subprocess.run(
                [
                    "docker", "exec", container, "psql", "-X", "-U",
                    "postgres", "-d", "leadpoet", "-c", "SELECT 1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL 15 did not become ready")

        psql(
            "CREATE ROLE anon; CREATE ROLE authenticated; "
            "CREATE ROLE service_role; CREATE ROLE legacy_writer;"
        )
        psql((REPO_ROOT / "scripts/86-research-lab-attested-v2-authority.sql").read_text())
        psql((REPO_ROOT / "scripts/99-research-lab-v2-champion-settlement.sql").read_text())
        psql(
            "CREATE TABLE public.transparency_log ("
            "id BIGSERIAL PRIMARY KEY, event_type TEXT NOT NULL, "
            "epoch_id BIGINT, "
            "actor_hotkey TEXT, nonce UUID UNIQUE, ts TIMESTAMPTZ, "
            "payload_hash TEXT, build_id TEXT, signature TEXT, "
            "payload JSONB NOT NULL DEFAULT '{}'::JSONB, "
            "tee_sequence BIGINT, event_hash TEXT);"
        )
        psql(
            "CREATE TABLE public.legacy_epoch_keys ("
            "epoch BIGINT, evaluation_epoch INTEGER, start_epoch INTEGER, "
            "epoch_count INTEGER, epoch_block BIGINT, reward_epochs INTEGER);"
        )
        psql(
            "CREATE TABLE IF NOT EXISTS public.epoch_audit_logs(epoch_id INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.published_weight_bundles(epoch_id INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.research_lab_attested_weight_bundles(epoch_id INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.research_lab_champion_reward_obligations(evaluation_epoch INTEGER);"
            "CREATE VIEW public.research_lab_epoch_payouts AS "
            "SELECT NULL::INTEGER AS epoch WHERE FALSE;"
            "CREATE TABLE IF NOT EXISTS public.research_lab_private_model_benchmark_bundles(evaluation_epoch INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.research_lab_scoring_runs(evaluation_epoch INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.validation_evidence_private(epoch_id INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.validator_attestations(epoch_id INTEGER);"
            "CREATE TABLE IF NOT EXISTS public.validator_sourcing_epoch_inputs_v2(epoch_id INTEGER);"
        )
        psql(
            "CREATE INDEX test_legacy_epoch_identity "
            "ON public.legacy_epoch_keys(epoch DESC);"
            "CREATE INDEX test_legacy_evaluation_epoch_identity "
            "ON public.legacy_epoch_keys(evaluation_epoch DESC);"
        )
        # Execute the real non-transactional production prerequisite through
        # psql, which dispatches each concurrent index statement separately.
        psql(INDEX_SQL)
        psql(
            "CREATE SCHEMA extensions; "
            "CREATE EXTENSION amcheck WITH SCHEMA extensions;"
        )
        amcheck_result = psql(AMCHECK_SQL)
        assert "idx_epoch_audit_logs_epoch_identity_v1" in amcheck_result.stderr
        assert "test_legacy_epoch_identity" in amcheck_result.stderr
        assert "test_legacy_evaluation_epoch_identity" in amcheck_result.stderr
        payout_view_probe = psql(
            "SELECT relation.relkind, "
            "pg_catalog.to_regclass("
            "'public.idx_rl_epoch_payouts_epoch_identity_v1') IS NULL "
            "FROM pg_catalog.pg_class relation "
            "WHERE relation.oid="
            "'public.research_lab_epoch_payouts'::pg_catalog.regclass;"
        )
        assert "v|t" in payout_view_probe.stdout

        # IF NOT EXISTS is name-only.  Keep an independent correct covering
        # index in place and prove the named contract still rejects every
        # valid-looking but noncanonical definition instead of being masked by
        # the generic physical-column coverage check.
        psql(
            "ALTER TABLE public.epoch_audit_logs ADD COLUMN other_key INTEGER;"
            "CREATE INDEX test_epoch_audit_alternate "
            "ON public.epoch_audit_logs(epoch_id);"
            "CREATE TABLE public.index_adversary_wrong_target(epoch_id INTEGER);"
            "CREATE INDEX test_wrong_target_epoch_cover "
            "ON public.index_adversary_wrong_target(epoch_id);"
        )
        plain_index_name = "idx_epoch_audit_logs_epoch_identity_v1"
        malformed_named_indexes = (
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.index_adversary_wrong_target(epoch_id DESC);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(other_key DESC);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC) "
            "WHERE epoch_id IS NOT NULL;",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(((epoch_id + 0)) DESC);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs USING hash(epoch_id);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(other_key) INCLUDE(epoch_id);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(other_key, epoch_id);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id ASC);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC, other_key);",
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC) INCLUDE(other_key);",
            "CREATE UNIQUE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC);",
        )
        for malformed_index_sql in malformed_named_indexes:
            psql(f"DROP INDEX public.{plain_index_name};")
            psql(malformed_index_sql)
            rejected(
                INDEX_VALIDATION_SQL,
                plain_index_name,
            )

        psql(f"DROP INDEX public.{plain_index_name};")
        rejected(INDEX_VALIDATION_SQL, plain_index_name)

        # A failed concurrent unique build leaves a real invalid pg_index row.
        # The exact validator must reject it even though an alternate valid
        # physical identity index still exists.
        psql(
            "INSERT INTO public.epoch_audit_logs(epoch_id) VALUES (7), (7);"
        )
        invalid_build = psql(
            "CREATE UNIQUE INDEX CONCURRENTLY "
            "idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC);",
            check=False,
        )
        assert invalid_build.returncode != 0
        invalid_flags = psql(
            "SELECT index_meta.indisvalid, index_meta.indisready, "
            "index_meta.indislive "
            "FROM pg_catalog.pg_index index_meta "
            "WHERE index_meta.indexrelid="
            "'public.idx_epoch_audit_logs_epoch_identity_v1'::pg_catalog.regclass;"
        )
        assert invalid_flags.stdout.strip() in {"f|f|t", "f|t|t"}
        rejected(INDEX_VALIDATION_SQL, plain_index_name)
        psql(
            f"DROP INDEX public.{plain_index_name};"
            "DELETE FROM public.epoch_audit_logs WHERE epoch_id=7;"
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC);"
        )

        # A client timeout can leave PostgreSQL legitimately building.  Hold a
        # writer open, start a real concurrent build, and prove validation says
        # to wait rather than misclassifying it as an invalid remnant to drop.
        psql(f"DROP INDEX public.{plain_index_name};")
        blocker = background_psql(
            "BEGIN; "
            "INSERT INTO public.epoch_audit_logs(epoch_id) VALUES (8); "
            "SELECT pg_catalog.pg_sleep(4); "
            "COMMIT;",
            "stateful-epoch-index-blocker",
        )
        for _ in range(40):
            blocker_state = psql(
                "SELECT COUNT(*) FROM pg_catalog.pg_stat_activity "
                "WHERE application_name='stateful-epoch-index-blocker' "
                "AND state='active';"
            )
            if blocker_state.stdout.strip() == "1":
                break
            time.sleep(0.1)
        else:
            raise AssertionError("index-build blocker did not become active")

        builder = background_psql(
            "CREATE INDEX CONCURRENTLY "
            "idx_epoch_audit_logs_epoch_identity_v1 "
            "ON public.epoch_audit_logs(epoch_id DESC);",
            "stateful-epoch-index-builder",
        )
        for _ in range(60):
            build_phase = psql(
                "SELECT progress.phase "
                "FROM pg_catalog.pg_stat_progress_create_index progress "
                "JOIN pg_catalog.pg_stat_activity activity "
                "ON activity.pid=progress.pid "
                "WHERE activity.application_name="
                "'stateful-epoch-index-builder';"
            )
            if build_phase.stdout.strip():
                break
            time.sleep(0.1)
        else:
            raise AssertionError("concurrent index build did not report progress")

        rejected(INDEX_VALIDATION_SQL, "indexes are still building")
        assert blocker.wait(timeout=10) == 0, blocker.stderr.read()
        assert builder.wait(timeout=10) == 0, builder.stderr.read()
        psql(INDEX_VALIDATION_SQL)
        psql("DELETE FROM public.epoch_audit_logs WHERE epoch_id=8;")

        # Same names in unrelated schemas must not poison exact public-schema
        # validation, which was a false failure in the original global lookup.
        psql(
            "CREATE SCHEMA index_adversary;"
            "CREATE TABLE index_adversary.other(epoch_id INTEGER);"
            "CREATE INDEX idx_epoch_audit_logs_epoch_identity_v1 "
            "ON index_adversary.other(epoch_id);"
        )
        psql(INDEX_VALIDATION_SQL)
        psql("DROP SCHEMA index_adversary CASCADE;")

        # The payload expression and all three predicate clauses are an exact
        # contract, not substring evidence that happens to mention the regex.
        psql("DROP INDEX public.idx_transparency_log_payload_epoch_identity_v1;")
        psql(
            "CREATE INDEX idx_transparency_log_payload_epoch_identity_v1 "
            "ON public.transparency_log "
            "(((payload->>'epoch_id')::BIGINT) DESC) "
            "WHERE payload ? 'epoch_id' "
            "AND payload->>'epoch_id' ~ '^[0-9]+$';"
        )
        rejected(
            INDEX_VALIDATION_SQL,
            "idx_transparency_log_payload_epoch_identity_v1",
        )
        psql(
            "DROP INDEX public.idx_transparency_log_payload_epoch_identity_v1;"
            "CREATE INDEX idx_transparency_log_payload_epoch_identity_v1 "
            "ON public.transparency_log "
            "(((payload->>'epoch_id')::BIGINT) DESC) "
            "WHERE pg_catalog.jsonb_typeof(payload)='object' "
            "AND payload ? 'epoch_id' "
            "AND payload->>'epoch_id' ~ '^[0-9]+$';"
            "DROP INDEX public.test_epoch_audit_alternate;"
            "DROP TABLE public.index_adversary_wrong_target;"
            "ALTER TABLE public.epoch_audit_logs DROP COLUMN other_key;"
        )
        psql(INDEX_VALIDATION_SQL)
        psql(SQL)

        # The public runtime contract is an exact, sanitized singleton RPC.
        # Anon can read it before, during, and after cutover, but cannot read
        # or mutate the protected state/authority/receipt relations directly.
        exposed_keys = psql(
            "SET ROLE anon; "
            "SELECT pg_catalog.string_agg(exposed.key, ',' ORDER BY exposed.key) "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_public_state_v1() state "
            "CROSS JOIN LATERAL pg_catalog.jsonb_object_keys("
            "pg_catalog.to_jsonb(state)) AS exposed(key); "
            "RESET ROLE;"
        )
        assert (
            "activated_at,fenced_at,first_settlement_epoch_id,"
            "last_legacy_epoch_id,lifecycle_state,mapping_hash,"
            "netuid,network_genesis_hash,staged_at,updated_at"
            in exposed_keys.stdout
        )
        public_legacy = psql(
            "SET ROLE anon; "
            "SELECT lifecycle_state, mapping_hash IS NULL, "
            "network_genesis_hash IS NULL, netuid IS NULL, "
            "last_legacy_epoch_id IS NULL, first_settlement_epoch_id IS NULL, "
            "fenced_at IS NULL, staged_at IS NULL, activated_at IS NULL, "
            "updated_at IS NOT NULL "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_public_state_v1(); "
            "RESET ROLE;"
        )
        assert "legacy_open|t|t|t|t|t|t|t|t|t" in public_legacy.stdout
        for protected_statement in (
            "SELECT * FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1",
            "INSERT INTO public.research_lab_stateful_subnet_epoch_cutover_state_v1 "
            "(singleton,lifecycle_state) VALUES (FALSE,'legacy_open')",
            "UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1 "
            "SET updated_at = updated_at",
            "DELETE FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1",
            "SELECT * FROM public.research_lab_stateful_subnet_epoch_candidates_v1",
            "SELECT * FROM public.research_lab_stateful_subnet_epoch_cutovers_v1",
            "SELECT * FROM public.research_lab_attested_execution_receipts_v2",
        ):
            rejected(
                f"SET ROLE anon; {protected_statement};",
                "permission denied",
            )

        rejected(
            "INSERT INTO public.research_lab_stateful_subnet_epoch_candidates_v1("
            f"snapshot_hash) VALUES ('{sha(899)}');",
            "stateful epoch candidate requires the durable pre-boundary fence",
        )

        payload_plan = psql(
            "SET enable_seqscan=off; "
            "EXPLAIN SELECT (payload->>'epoch_id')::BIGINT "
            "FROM public.transparency_log "
            "WHERE pg_catalog.jsonb_typeof(payload)='object' "
            "AND payload ? 'epoch_id' "
            "AND payload->>'epoch_id' ~ '^[0-9]+$' "
            "ORDER BY (payload->>'epoch_id')::BIGINT DESC LIMIT 1;"
        )
        assert "idx_transparency_log_payload_epoch_identity_v1" in payload_plan.stdout
        timed_index_probe = psql(
            "BEGIN; "
            "INSERT INTO public.transparency_log(event_type,payload) "
            "SELECT 'INDEX_PROBE', pg_catalog.jsonb_build_object('epoch_id',40) "
            "FROM pg_catalog.generate_series(1,20000); "
            "SET LOCAL statement_timeout='100ms'; "
            "SET LOCAL enable_seqscan=off; "
            "SELECT (payload->>'epoch_id')::BIGINT "
            "FROM public.transparency_log "
            "WHERE pg_catalog.jsonb_typeof(payload)='object' "
            "AND payload ? 'epoch_id' "
            "AND payload->>'epoch_id' ~ '^[0-9]+$' "
            "ORDER BY (payload->>'epoch_id')::BIGINT DESC LIMIT 1; "
            "ROLLBACK;"
        )
        assert "40" in timed_index_probe.stdout

        legacy_epoch_index = psql(
            "SELECT index_relation.oid::pg_catalog.regclass::TEXT "
            "FROM pg_catalog.pg_index index_meta "
            "JOIN pg_catalog.pg_class index_relation "
            "ON index_relation.oid=index_meta.indexrelid "
            "JOIN pg_catalog.pg_attribute column_meta "
            "ON column_meta.attrelid=index_meta.indrelid "
            "AND column_meta.attname='epoch' "
            "WHERE index_meta.indrelid='public.legacy_epoch_keys'::pg_catalog.regclass "
            "AND index_meta.indkey[0]=column_meta.attnum LIMIT 1;"
        ).stdout.strip()
        assert legacy_epoch_index
        psql(f"DROP INDEX {legacy_epoch_index};")
        rejected(
            "SELECT * FROM public.research_lab_stateful_subnet_epoch_cutover_fence_v1("
            f"'{raw(900)}', 71, 0, 1);",
            "stateful epoch fence prerequisite btree index is missing for legacy_epoch_keys.epoch",
        )
        psql(
            "CREATE INDEX test_legacy_epoch_identity_repaired "
            "ON public.legacy_epoch_keys(epoch DESC);"
        )

        validator_boot = sha(1)
        coordinator_boot = sha(2)
        legacy_root_receipt = sha(10)
        publication_receipt = sha(11)
        legacy_final_receipt = sha(12)
        first_boundary_receipt = sha(13)
        cutover_receipt = sha(14)
        boundary_11_receipt = sha(15)
        boundary_12_receipt = sha(16)
        mutable_snapshot_receipt = sha(17)
        bad_snapshot_receipt = sha(18)
        safety_snapshot_receipt = sha(19)
        safety_50399_snapshot_receipt = sha(100)
        safety_50400_snapshot_receipt = sha(101)
        boundary_13_receipt = sha(120)
        boundary_14_receipt = sha(121)
        boundary_15_receipt = sha(122)
        invalid_boundary_receipt = sha(123)
        boundary_14_snapshot_receipt = sha(135)
        first_snapshot_hash = sha(33)
        cutover_authority_hash = sha(34)
        boundary_11_hash = sha(35)
        boundary_12_hash = sha(36)
        mutable_snapshot_hash = sha(37)
        bad_snapshot_hash = sha(38)
        safety_snapshot_hash = sha(39)
        safety_50399_snapshot_hash = sha(102)
        safety_50400_snapshot_hash = sha(103)
        boundary_13_hash = sha(124)
        boundary_14_hash = sha(125)
        boundary_15_hash = sha(126)
        invalid_boundary_hash = sha(127)
        boundary_14_snapshot_hash = sha(136)
        candidate_payload_hash = sha(27)
        candidate_authorization_hash = sha(28)
        candidate_hotkey = "5" + ("A" * 47)
        candidate_signature = "0x" + ("a" * 128)
        legacy_bundle_hash = sha(40)
        publication_event_hash = sha(41)
        finalization_event_hash = sha(42)
        mapping_hash = sha(43)
        epoch_ref_10 = sha(44)
        epoch_ref_11 = sha(45)
        epoch_ref_12 = sha(46)
        wrong_epoch_ref = sha(47)
        epoch_ref_13 = sha(128)
        epoch_ref_14 = sha(129)
        epoch_ref_15 = sha(130)
        genesis_hash = raw(50)
        cutover_block_hash = raw(51)
        boundary_11_block_hash = raw(52)
        boundary_12_block_hash = raw(53)
        mutable_snapshot_block_hash = raw(54)
        bad_snapshot_block_hash = raw(55)
        safety_snapshot_block_hash = raw(56)
        safety_50399_snapshot_block_hash = raw(104)
        safety_50400_snapshot_block_hash = raw(105)
        boundary_13_block_hash = raw(131)
        boundary_14_block_hash = raw(132)
        boundary_15_block_hash = raw(133)
        invalid_boundary_block_hash = raw(134)
        boundary_14_snapshot_block_hash = raw(137)
        observed_at = "2026-07-16T12:34:56+00:00"

        fixture = f"""
        GRANT INSERT ON public.legacy_epoch_keys TO service_role;
        GRANT INSERT ON public.legacy_epoch_keys TO legacy_writer;
        GRANT INSERT ON public.transparency_log TO service_role;
        GRANT USAGE, SELECT ON SEQUENCE public.transparency_log_id_seq
            TO service_role;
        SET ROLE legacy_writer;
        INSERT INTO public.legacy_epoch_keys
            (epoch, evaluation_epoch, start_epoch, epoch_count, epoch_block, reward_epochs)
        VALUES (40, 39, 38, 999999, 999999, 999999);
        RESET ROLE;

        INSERT INTO public.research_lab_attested_boot_identities_v2 (
            boot_identity_hash, schema_version, role, physical_role, commit_sha,
            pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
            signing_pubkey, transport_pubkey, transport_certificate_hash,
            boot_nonce, attestation_user_data_hash, attestation_document_ref,
            attestation_document_hash, identity_doc, issued_at
        ) VALUES
        (
            '{validator_boot}', 'leadpoet.attested_boot_identity.v2',
            'validator_weights', 'validator_weights', repeat('a', 40),
            repeat('b', 96), '{sha(60)}', '{sha(61)}', '{sha(62)}',
            repeat('c', 64), repeat('d', 64), '{sha(63)}', repeat('e', 32),
            '{sha(64)}', 'test:validator', '{sha(65)}', '{{}}'::JSONB,
            '{observed_at}'::TIMESTAMPTZ
        ),
        (
            '{coordinator_boot}', 'leadpoet.attested_boot_identity.v2',
            'gateway_coordinator', 'gateway_coordinator', repeat('f', 40),
            repeat('1', 96), '{sha(66)}', '{sha(67)}', '{sha(68)}',
            repeat('2', 64), repeat('3', 64), '{sha(69)}', repeat('4', 32),
            '{sha(70)}', 'test:coordinator', '{sha(71)}', '{{}}'::JSONB,
            '{observed_at}'::TIMESTAMPTZ
        );

        CREATE OR REPLACE FUNCTION public.test_insert_epoch_receipt(
            p_receipt_hash TEXT,
            p_role TEXT,
            p_purpose TEXT,
            p_epoch_id INTEGER,
            p_output_root TEXT,
            p_parent_receipt_hashes JSONB DEFAULT '[]'::JSONB
        ) RETURNS VOID
        LANGUAGE plpgsql
        AS $fn$
        BEGIN
            INSERT INTO public.research_lab_attested_execution_receipts_v2 (
                receipt_hash, schema_version, role, purpose, job_id, epoch_id,
                sequence, commit_sha, pcr0, build_manifest_hash,
                dependency_lock_hash, config_hash, boot_identity_hash,
                input_root, output_root, transport_root, host_operation_root,
                artifact_root, receipt_status, failure_code, enclave_pubkey,
                enclave_signature, receipt_doc, issued_at
            ) VALUES (
                p_receipt_hash, 'leadpoet.attested_execution_receipt.v2',
                p_role, p_purpose, 'test:' || p_receipt_hash, p_epoch_id,
                0, repeat('a', 40), repeat('b', 96), '{sha(72)}',
                '{sha(73)}', '{sha(74)}',
                CASE WHEN p_role = 'gateway_coordinator'
                     THEN '{coordinator_boot}' ELSE '{validator_boot}' END,
                p_receipt_hash, p_output_root, '{sha(75)}', '{sha(76)}',
                '{sha(77)}', 'succeeded', NULL, repeat('5', 64),
                repeat('6', 128),
                pg_catalog.jsonb_build_object(
                    'receipt_hash', p_receipt_hash,
                    'role', p_role,
                    'purpose', p_purpose,
                    'epoch_id', p_epoch_id,
                    'output_root', p_output_root,
                    'status', 'succeeded',
                    'parent_receipt_hashes', p_parent_receipt_hashes
                ),
                '{observed_at}'::TIMESTAMPTZ
            );
        END;
        $fn$;

        SET statement_timeout = '1000ms';
        SELECT lifecycle_state
        FROM public.research_lab_stateful_subnet_epoch_cutover_fence_v1(
            '{genesis_hash}', 71, 40, 41
        );
        RESET statement_timeout;
        -- The early fence reserves 41 but deliberately leaves the measured
        -- last legacy ordinal writable through its final weight lifecycle.
        SELECT public.test_insert_epoch_receipt(
            '{legacy_root_receipt}', 'validator_weights',
            'validator.weight_snapshot.v2', 40, '{sha(30)}'
        );
        SELECT public.test_insert_epoch_receipt(
            '{publication_receipt}', 'gateway_coordinator',
            'gateway.weights.publication.v2', 40, '{sha(31)}'
        );
        SELECT public.test_insert_epoch_receipt(
            '{legacy_final_receipt}', 'validator_weights',
            'validator.weights.finalized.v2', 40, '{sha(32)}'
        );
        SELECT public.test_insert_epoch_receipt(
            '{first_boundary_receipt}', 'validator_weights',
            'validator.subnet_epoch_snapshot.v2', 41, '{first_snapshot_hash}'
        );

        INSERT INTO public.research_lab_stateful_subnet_epoch_candidates_v1 (
            snapshot_hash, schema_version, mapping_hash, epoch_scheme,
            network_genesis_hash, netuid, head_kind, block_hash,
            current_block, last_epoch_block, pending_epoch_at,
            subnet_epoch_index, epoch_ref, proposed_settlement_epoch_id,
            validator_hotkey, candidate_payload_hash,
            validator_hotkey_signature, candidate_authorization_hash,
            tempo, blocks_since_last_step, next_epoch_block,
            blocks_remaining, chain_state_receipt_hash, snapshot_doc,
            observed_at
        ) VALUES (
            '{first_snapshot_hash}', 'leadpoet.subnet_epoch_snapshot.v1',
            '{mapping_hash}', 'bittensor.subnet_epoch_index.v1',
            '{genesis_hash}', 71, 'finalized', '{cutover_block_hash}',
            1000, 1000, 0, 10, '{epoch_ref_10}', 41,
            '{candidate_hotkey}', '{candidate_payload_hash}',
            '{candidate_signature}', '{candidate_authorization_hash}',
            360, 0, 1360,
            360, '{first_boundary_receipt}',
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.subnet_epoch_snapshot.v1',
                'epoch_scheme', 'bittensor.subnet_epoch_index.v1',
                'network_genesis_hash', '{genesis_hash}', 'netuid', 71,
                'head_kind', 'finalized', 'block_hash', '{cutover_block_hash}',
                'current_block', 1000, 'last_epoch_block', 1000,
                'pending_epoch_at', 0, 'subnet_epoch_index', 10,
                'tempo', 360, 'blocks_since_last_step', 0,
                'observed_at', '{observed_at}', 'epoch_id', 10,
                'epoch_ref', '{epoch_ref_10}', 'epoch_block', 0,
                'next_epoch_block', 1360, 'blocks_remaining', 360,
                'settlement_epoch_id', 41,
                'cutover_mapping_hash', '{mapping_hash}'
            ),
            '{observed_at}'::TIMESTAMPTZ
        );

        INSERT INTO public.research_lab_attested_weight_bundles_v2 (
            bundle_hash, schema_version, netuid, epoch_id, block,
            validator_hotkey, root_receipt_hash, weights_hash, snapshot_hash,
            bundle_doc
        ) VALUES (
            '{legacy_bundle_hash}', 'leadpoet.published_weight_bundle.v2',
            71, 40, 990, 'validator-hotkey', '{legacy_root_receipt}',
            '{hex64(80)}', '{sha(81)}', '{{}}'::JSONB
        );
        INSERT INTO public.research_lab_attested_publication_events_v2 (
            weight_submission_event_hash, bundle_hash,
            publication_receipt_hash, transparency_event_hash,
            durable_readback_hash, publication_doc
        ) VALUES (
            '{publication_event_hash}', '{legacy_bundle_hash}',
            '{publication_receipt}', '{sha(82)}', '{sha(83)}', '{{}}'::JSONB
        );
        INSERT INTO public.research_lab_attested_weight_finalizations_v2 (
            weight_finalization_event_hash, weight_submission_event_hash,
            bundle_hash, finalization_receipt_hash,
            extrinsic_authorization_hash, extrinsic_hash, finalized_block,
            finalized_block_hash, state_transition_hash, finalization_doc
        ) VALUES (
            '{finalization_event_hash}', '{publication_event_hash}',
            '{legacy_bundle_hash}', '{legacy_final_receipt}', '{sha(84)}',
            '{raw(85)}', 964, '{hex64(86)}', '{sha(87)}', '{{}}'::JSONB
        );

        CREATE OR REPLACE FUNCTION public.test_insert_cutover(
            p_cutover_block BIGINT DEFAULT 1000
        )
        RETURNS VOID
        LANGUAGE SQL
        AS $fn$
        INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1 (
            cutover_authority_hash, schema_version, mapping_hash,
            manifest_schema_version, epoch_scheme, previous_epoch_scheme,
            network_genesis_hash, netuid, cutover_block, cutover_block_hash,
            first_subnet_epoch_index, first_epoch_ref,
            first_settlement_epoch_id, last_legacy_epoch_id, first_tempo,
            first_pending_epoch_at, first_blocks_since_last_step,
            first_next_epoch_block, first_observed_at, first_snapshot_hash,
            first_snapshot_receipt_hash, last_legacy_bundle_hash,
            last_legacy_weight_finalization_event_hash,
            last_legacy_finalization_receipt_hash, cutover_receipt_hash,
            manifest_doc, first_snapshot_doc, authority_doc
        ) VALUES (
            '{cutover_authority_hash}',
            'leadpoet.subnet_epoch_cutover_authority.v1', '{mapping_hash}',
            'leadpoet.subnet_epoch_cutover.v1',
            'bittensor.subnet_epoch_index.v1', 'legacy_global_360_v1',
            '{genesis_hash}', 71, p_cutover_block, '{cutover_block_hash}', 10,
            '{epoch_ref_10}', 41, 40, 360, 0, 0, p_cutover_block + 360,
            '{observed_at}'::TIMESTAMPTZ, '{first_snapshot_hash}',
            '{first_boundary_receipt}', '{legacy_bundle_hash}',
            '{finalization_event_hash}', '{legacy_final_receipt}',
            '{cutover_receipt}',
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.subnet_epoch_cutover.v1',
                'epoch_scheme', 'bittensor.subnet_epoch_index.v1',
                'network_genesis_hash', '{genesis_hash}', 'netuid', 71,
                'cutover_block', p_cutover_block,
                'cutover_block_hash', '{cutover_block_hash}',
                'first_subnet_epoch_index', 10,
                'first_settlement_epoch_id', 41,
                'last_legacy_epoch_id', 40, 'mapping_hash', '{mapping_hash}'
            ),
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.subnet_epoch_snapshot.v1',
                'epoch_scheme', 'bittensor.subnet_epoch_index.v1',
                'network_genesis_hash', '{genesis_hash}', 'netuid', 71,
                'head_kind', 'finalized', 'block_hash', '{cutover_block_hash}',
                'current_block', p_cutover_block,
                'last_epoch_block', p_cutover_block,
                'pending_epoch_at', 0, 'subnet_epoch_index', 10,
                'tempo', 360, 'blocks_since_last_step', 0,
                'observed_at', '{observed_at}', 'epoch_id', 10,
                'epoch_ref', '{epoch_ref_10}', 'epoch_block', 0,
                'next_epoch_block', p_cutover_block + 360,
                'blocks_remaining', 360,
                'settlement_epoch_id', 41,
                'cutover_mapping_hash', '{mapping_hash}'
            ),
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.subnet_epoch_cutover_authority.v1',
                'mapping_hash', '{mapping_hash}',
                'first_epoch_ref', '{epoch_ref_10}',
                'first_snapshot_hash', '{first_snapshot_hash}',
                'first_snapshot_receipt_hash', '{first_boundary_receipt}',
                'last_legacy_bundle_hash', '{legacy_bundle_hash}',
                'last_legacy_weight_finalization_event_hash', '{finalization_event_hash}',
                'last_legacy_finalization_receipt_hash', '{legacy_final_receipt}',
                'manifest', pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.subnet_epoch_cutover.v1',
                    'epoch_scheme', 'bittensor.subnet_epoch_index.v1',
                    'network_genesis_hash', '{genesis_hash}', 'netuid', 71,
                    'cutover_block', p_cutover_block,
                    'cutover_block_hash', '{cutover_block_hash}',
                    'first_subnet_epoch_index', 10,
                    'first_settlement_epoch_id', 41,
                    'last_legacy_epoch_id', 40,
                    'mapping_hash', '{mapping_hash}'
                )
            )
        );
        $fn$;
        """
        psql(fixture)

        preflight = psql(
            f"SET ROLE service_role; "
            f"SELECT eligible, legacy_high_water, "
            f"expected_last_legacy_epoch_id, first_settlement_epoch_id, "
            f"first_settlement_occupied, candidate_snapshot_hash, "
            f"candidate_receipt_hash "
            f"FROM public.research_lab_stateful_subnet_epoch_cutover_preflight_v1("
            f"'{mapping_hash}'); RESET ROLE;"
        )
        assert (
            f"t|40|40|41|f|{first_snapshot_hash}|{first_boundary_receipt}"
            in preflight.stdout
        )

        fence_state = psql(
            "SELECT lifecycle_state, legacy_high_water, first_settlement_occupied "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_fence_v1("
            f"'{genesis_hash}', 71, 40, 41);"
        )
        assert "cutover_fenced|40|f" in fence_state.stdout
        public_fenced = psql(
            "SET ROLE anon; "
            "SELECT lifecycle_state, network_genesis_hash, netuid, "
            "last_legacy_epoch_id, first_settlement_epoch_id, "
            "mapping_hash IS NULL, fenced_at IS NOT NULL, "
            "staged_at IS NULL, activated_at IS NULL "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_public_state_v1(); "
            "RESET ROLE;"
        )
        assert (
            f"cutover_fenced|{genesis_hash}|71|40|41|t|t|t|t"
            in public_fenced.stdout
        )

        # Every fenced write serializes on the same advisory lock. A concurrent
        # first-ordinal insert cannot slip past while another fence-sensitive
        # transaction is in flight, and is rejected on retry after the lock.
        holder = subprocess.Popen(
            [
                "docker", "exec", container, "psql", "-X", "-U", "postgres",
                "-d", "leadpoet", "-v", "ON_ERROR_STOP=1", "-c",
                "BEGIN; SELECT pg_catalog.pg_advisory_xact_lock(7100,0); "
                "SELECT pg_catalog.pg_sleep(1.5); COMMIT;",
            ],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for _ in range(30):
            held = psql(
                "SELECT COUNT(*) FROM pg_catalog.pg_locks "
                "WHERE locktype='advisory' AND granted;"
            ).stdout.strip()
            if int(held or "0") > 0:
                break
            time.sleep(0.05)
        else:
            holder.kill()
            raise AssertionError("concurrent advisory lock holder did not start")
        blocked = psql(
            "SET lock_timeout='200ms'; SET ROLE service_role; "
            "INSERT INTO public.legacy_epoch_keys(epoch) VALUES (41);",
            check=False,
        )
        assert blocked.returncode != 0
        assert "lock timeout" in blocked.stderr.lower()
        assert holder.wait(timeout=5) == 0, (holder.stderr.read() if holder.stderr else "")

        # The pre-boundary fence closes the offset gap immediately. Old IDs
        # remain usable, while first/future physical IDs and JSON payload IDs
        # are rejected even before a manifest/candidate is bound.
        psql(
            "SET ROLE legacy_writer; "
            "INSERT INTO public.legacy_epoch_keys(epoch) VALUES (40); "
            "RESET ROLE;"
        )
        rejected(
            "SET ROLE legacy_writer; "
            "INSERT INTO public.legacy_epoch_keys(epoch) VALUES (41);",
            "stateful epoch fence rejects legacy_epoch_keys.epoch identity 41",
        )
        rejected(
            "SET ROLE service_role; "
            "INSERT INTO public.legacy_epoch_keys(epoch) VALUES (41);",
            "stateful epoch fence rejects legacy_epoch_keys.epoch identity 41",
        )
        rejected(
            "SET ROLE service_role; "
            "INSERT INTO public.legacy_epoch_keys(evaluation_epoch) VALUES (42);",
            "stateful epoch fence rejects legacy_epoch_keys.evaluation_epoch identity 42",
        )
        rejected(
            "SET ROLE service_role; "
            "INSERT INTO public.transparency_log(event_type, payload) VALUES "
            "('EPOCH_END', '{\"epoch_id\": 41}'::JSONB);",
            "stateful epoch fence rejects transparency epoch identity 41",
        )
        rejected(
            "SET ROLE service_role; "
            "INSERT INTO public.transparency_log(event_type, epoch_id, payload) "
            "VALUES ('LEGACY_PHYSICAL', 41, '{}'::JSONB);",
            "stateful epoch fence rejects transparency physical identity",
        )

        binding = psql(
            "SELECT lifecycle_state, mapping_hash, legacy_high_water "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_bind_v1("
            f"'{mapping_hash}', '{cutover_authority_hash}', "
            f"'{legacy_final_receipt}', NULL);"
        )
        assert f"cutover_fenced|{mapping_hash}|40" in binding.stdout

        # A coordinator receipt is the only post-binding first-ID exception,
        # and its two exact parents are enforced by the generic fence.
        rejected(
            f"SELECT public.test_insert_epoch_receipt("
            f"'{cutover_receipt}', 'gateway_coordinator', "
            f"'research_lab.subnet_epoch_cutover.v2', 41, "
            f"'{cutover_authority_hash}', "
            f"pg_catalog.jsonb_build_array('{first_boundary_receipt}'));",
            "stateful epoch fence rejects receipt epoch identity 41",
        )
        psql(
            f"SELECT public.test_insert_epoch_receipt("
            f"'{cutover_receipt}', 'gateway_coordinator', "
            f"'research_lab.subnet_epoch_cutover.v2', 41, "
            f"'{cutover_authority_hash}', "
            f"pg_catalog.jsonb_build_array("
            f"'{first_boundary_receipt}', '{legacy_final_receipt}'));"
        )
        resumed_preflight = psql(
            f"SELECT eligible, legacy_high_water, first_settlement_occupied "
            f"FROM public.research_lab_stateful_subnet_epoch_cutover_preflight_v1("
            f"'{mapping_hash}', '{cutover_receipt}');"
        )
        assert "t|40|f" in resumed_preflight.stdout

        manifest_doc = {
            "schema_version": "leadpoet.subnet_epoch_cutover.v1",
            "epoch_scheme": "bittensor.subnet_epoch_index.v1",
            "network_genesis_hash": genesis_hash,
            "netuid": 71,
            "cutover_block": 1000,
            "cutover_block_hash": cutover_block_hash,
            "first_subnet_epoch_index": 10,
            "first_settlement_epoch_id": 41,
            "last_legacy_epoch_id": 40,
            "mapping_hash": mapping_hash,
        }
        snapshot_doc = {
            "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
            "epoch_scheme": "bittensor.subnet_epoch_index.v1",
            "network_genesis_hash": genesis_hash,
            "netuid": 71,
            "head_kind": "finalized",
            "block_hash": cutover_block_hash,
            "current_block": 1000,
            "last_epoch_block": 1000,
            "pending_epoch_at": 0,
            "subnet_epoch_index": 10,
            "tempo": 360,
            "blocks_since_last_step": 0,
            "observed_at": observed_at,
            "epoch_id": 10,
            "epoch_ref": epoch_ref_10,
            "epoch_block": 0,
            "next_epoch_block": 1360,
            "blocks_remaining": 360,
            "settlement_epoch_id": 41,
            "cutover_mapping_hash": mapping_hash,
        }
        authority_doc = {
            "schema_version": "leadpoet.subnet_epoch_cutover_authority.v1",
            "mapping_hash": mapping_hash,
            "first_epoch_ref": epoch_ref_10,
            "first_snapshot_hash": first_snapshot_hash,
            "first_snapshot_receipt_hash": first_boundary_receipt,
            "last_legacy_bundle_hash": legacy_bundle_hash,
            "last_legacy_weight_finalization_event_hash": finalization_event_hash,
            "last_legacy_finalization_receipt_hash": legacy_final_receipt,
            "manifest": manifest_doc,
        }
        cutover_row = {
            "cutover_authority_hash": cutover_authority_hash,
            "schema_version": "leadpoet.subnet_epoch_cutover_authority.v1",
            "mapping_hash": mapping_hash,
            "manifest_schema_version": "leadpoet.subnet_epoch_cutover.v1",
            "epoch_scheme": "bittensor.subnet_epoch_index.v1",
            "previous_epoch_scheme": "legacy_global_360_v1",
            "network_genesis_hash": genesis_hash,
            "netuid": 71,
            "cutover_block": 1000,
            "cutover_block_hash": cutover_block_hash,
            "first_subnet_epoch_index": 10,
            "first_epoch_ref": epoch_ref_10,
            "first_settlement_epoch_id": 41,
            "last_legacy_epoch_id": 40,
            "first_tempo": 360,
            "first_pending_epoch_at": 0,
            "first_blocks_since_last_step": 0,
            "first_next_epoch_block": 1360,
            "first_observed_at": observed_at,
            "first_snapshot_hash": first_snapshot_hash,
            "first_snapshot_receipt_hash": first_boundary_receipt,
            "last_legacy_bundle_hash": legacy_bundle_hash,
            "last_legacy_weight_finalization_event_hash": finalization_event_hash,
            "last_legacy_finalization_receipt_hash": legacy_final_receipt,
            "cutover_receipt_hash": cutover_receipt,
            "manifest_doc": manifest_doc,
            "first_snapshot_doc": snapshot_doc,
            "authority_doc": authority_doc,
        }
        initialization_payload = {
            "epoch_id": 41,
            "epoch_key_semantics": "settlement_ordinal",
            "epoch_authority": snapshot_doc,
            "epoch_boundaries": {
                "start_block": 1000,
                "end_block": 1360,
                "expected_end_block": 1360,
                "pending_epoch_at": 0,
                "tempo": 360,
                "start_timestamp": observed_at,
                "estimated_end_timestamp": observed_at,
            },
            "queue_state": {
                "queue_merkle_root": "0" * 64,
                "pending_lead_count": 0,
            },
            "assignment": {
                "assigned_lead_ids": [],
                "assigned_to_validators": [],
                "validator_count": 0,
            },
            "timestamp": observed_at,
        }
        payload_hash = hashlib.sha256(
            json.dumps(
                initialization_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        initialization_nonce = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "leadpoet:epoch-lifecycle:v1:EPOCH_INITIALIZATION:41",
            )
        )
        initialization_event = {
            "event_type": "EPOCH_INITIALIZATION",
            "actor_hotkey": "system",
            "nonce": initialization_nonce,
            "ts": observed_at,
            "payload_hash": payload_hash,
            "build_id": "pg15-test",
            "signature": "system",
            "payload": initialization_payload,
        }
        cutover_json = json.dumps(cutover_row, sort_keys=True).replace("'", "''")
        initialization_json = json.dumps(
            initialization_event, sort_keys=True
        ).replace("'", "''")

        # A conflicting legacy row with the deterministic nonce makes the
        # initialization insert fail. The cutover and plan binding must roll
        # back with it, leaving the durable pre-boundary fence closed.
        psql(
            "INSERT INTO public.transparency_log("
            "event_type,actor_hotkey,nonce,ts,payload_hash,build_id,signature,payload) "
            f"VALUES ('LEGACY_TEST','system','{initialization_nonce}',"
            f"'{observed_at}','{'0' * 64}','pg15-test','system',"
            "'{\"epoch_id\":40}'::JSONB);"
        )
        rejected(
            "SELECT * FROM public.research_lab_stateful_subnet_epoch_stage_v1("
            f"'{cutover_json}'::JSONB, '{initialization_json}'::JSONB);",
            "stateful epoch initialization exact readback failed",
        )
        rolled_back = psql(
            "SELECT lifecycle_state, cutover_receipt_hash, initialization_nonce, "
            "(SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_cutovers_v1) "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1;"
        )
        assert "cutover_fenced|||0" in rolled_back.stdout
        psql(
            f"DELETE FROM public.transparency_log WHERE nonce = '{initialization_nonce}';"
        )

        stage_sql = (
            "SELECT lifecycle_state, mapping_hash, cutover_receipt_hash, "
            "initialization_nonce, initialization_payload_hash "
            "FROM public.research_lab_stateful_subnet_epoch_stage_v1("
            f"'{cutover_json}'::JSONB, '{initialization_json}'::JSONB);"
        )
        staged = psql(stage_sql)
        expected_stage = (
            f"stateful_staged|{mapping_hash}|{cutover_receipt}|"
            f"{initialization_nonce}|{payload_hash}"
        )
        assert expected_stage in staged.stdout
        assert expected_stage in psql(stage_sql).stdout

        # Staged is still fenced: stale first/future legacy writes remain
        # rejected and only the separately confirmed activation opens it.
        rejected(
            "SET ROLE service_role; "
            "INSERT INTO public.legacy_epoch_keys(epoch) VALUES (41);",
            "stateful epoch fence rejects legacy_epoch_keys.epoch identity 41",
        )
        activated = psql(
            "SELECT lifecycle_state, mapping_hash FROM "
            "public.research_lab_stateful_subnet_epoch_activate_v1("
            f"'{mapping_hash}', TRUE);"
        )
        assert f"stateful_active|{mapping_hash}" in activated.stdout
        public_active = psql(
            "SET ROLE anon; "
            "SELECT lifecycle_state, mapping_hash, network_genesis_hash, "
            "netuid, last_legacy_epoch_id, first_settlement_epoch_id, "
            "fenced_at IS NOT NULL, staged_at IS NOT NULL, "
            "activated_at IS NOT NULL "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_public_state_v1(); "
            "RESET ROLE;"
        )
        assert (
            f"stateful_active|{mapping_hash}|{genesis_hash}|71|40|41|t|t|t"
            in public_active.stdout
        )

        # Activation does not reopen legacy authority. Durable lifecycle rows
        # must carry exact settlement semantics/mapping, and a V2 bundle must
        # include the same-epoch subnet snapshot receipt from its graph.
        canonical_end_payload = json.dumps(
            {
                "epoch_id": 41,
                "epoch_key_semantics": "settlement_ordinal",
                "epoch_authority": snapshot_doc,
                "timestamp": observed_at,
            },
            sort_keys=True,
        ).replace("'", "''")
        psql(
            "INSERT INTO public.transparency_log(event_type,payload) VALUES ("
            f"'EPOCH_END','{canonical_end_payload}'::JSONB);"
        )
        rejected(
            "INSERT INTO public.transparency_log(event_type,payload) VALUES ("
            "'EPOCH_INITIALIZATION',"
            "'{\"epoch_id\":42,\"epoch_key_semantics\":\"legacy_global_360\"}'::JSONB);",
            "stateful epoch active lifecycle identity lacks exact authority 42",
        )
        rejected(
            "INSERT INTO public.research_lab_attested_weight_bundles_v2("
            "bundle_hash,schema_version,netuid,epoch_id,block,validator_hotkey,"
            "root_receipt_hash,weights_hash,snapshot_hash,bundle_doc) VALUES ("
            f"'{sha(901)}','leadpoet.published_weight_bundle.v2',71,41,1001,"
            f"'stale-validator','{legacy_root_receipt}','{hex64(902)}',"
            f"'{sha(903)}','{{}}'::JSONB);",
            "stateful epoch active V2 bundle lacks subnet authority 41",
        )

        boundary_helpers = f"""
        CREATE OR REPLACE FUNCTION public.test_insert_boundary(
            p_boundary_hash TEXT, p_receipt_hash TEXT, p_index BIGINT,
            p_epoch_ref TEXT, p_settlement INTEGER, p_block BIGINT,
            p_block_hash TEXT, p_tempo INTEGER
        ) RETURNS VOID
        LANGUAGE SQL
        AS $fn$
        INSERT INTO public.research_lab_stateful_subnet_epoch_boundaries_v1 (
            boundary_hash, schema_version, mapping_hash, epoch_scheme,
            network_genesis_hash, netuid, subnet_epoch_index, epoch_ref,
            settlement_epoch_id, boundary_block, boundary_block_hash, tempo,
            pending_epoch_at, blocks_since_last_step, next_epoch_block,
            chain_state_receipt_hash, boundary_doc, observed_at
        ) VALUES (
            p_boundary_hash, 'leadpoet.subnet_epoch_boundary.v1',
            '{mapping_hash}', 'bittensor.subnet_epoch_index.v1',
            '{genesis_hash}', 71, p_index, p_epoch_ref, p_settlement,
            p_block, p_block_hash, p_tempo, 0, 0, p_block + p_tempo,
            p_receipt_hash,
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.subnet_epoch_boundary.v1',
                'mapping_hash', '{mapping_hash}',
                'snapshot', pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.subnet_epoch_snapshot.v1',
                    'epoch_scheme', 'bittensor.subnet_epoch_index.v1',
                    'network_genesis_hash', '{genesis_hash}', 'netuid', 71,
                    'head_kind', 'finalized', 'block_hash', p_block_hash,
                    'current_block', p_block, 'last_epoch_block', p_block,
                    'pending_epoch_at', 0, 'subnet_epoch_index', p_index,
                    'tempo', p_tempo, 'blocks_since_last_step', 0,
                    'observed_at', '{observed_at}', 'epoch_id', p_index,
                    'epoch_ref', p_epoch_ref, 'epoch_block', 0,
                    'next_epoch_block', p_block + p_tempo,
                    'blocks_remaining', p_tempo,
                    'settlement_epoch_id', p_settlement,
                    'cutover_mapping_hash', '{mapping_hash}'
                )
            ),
            '{observed_at}'::TIMESTAMPTZ
        );
        $fn$;

        CREATE OR REPLACE FUNCTION public.test_insert_snapshot(
            p_snapshot_hash TEXT, p_receipt_hash TEXT, p_index BIGINT,
            p_epoch_ref TEXT, p_settlement INTEGER, p_current BIGINT,
            p_last BIGINT, p_block_hash TEXT, p_tempo INTEGER,
            p_blocks_since BIGINT, p_next BIGINT, p_remaining BIGINT
        ) RETURNS VOID
        LANGUAGE SQL
        AS $fn$
        INSERT INTO public.research_lab_stateful_subnet_epoch_snapshots_v1 (
            snapshot_hash, schema_version, mapping_hash, epoch_scheme,
            network_genesis_hash, netuid, head_kind, block_hash,
            current_block, last_epoch_block, pending_epoch_at,
            subnet_epoch_index, epoch_ref, settlement_epoch_id, tempo,
            blocks_since_last_step, epoch_block, next_epoch_block,
            blocks_remaining, chain_state_receipt_hash, snapshot_doc,
            observed_at
        ) VALUES (
            p_snapshot_hash, 'leadpoet.subnet_epoch_snapshot.v1',
            '{mapping_hash}', 'bittensor.subnet_epoch_index.v1',
            '{genesis_hash}', 71, 'exact', p_block_hash, p_current, p_last, 0,
            p_index, p_epoch_ref, p_settlement, p_tempo, p_blocks_since,
            p_current - p_last, p_next, p_remaining, p_receipt_hash,
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.subnet_epoch_snapshot.v1',
                'epoch_scheme', 'bittensor.subnet_epoch_index.v1',
                'network_genesis_hash', '{genesis_hash}', 'netuid', 71,
                'head_kind', 'exact', 'block_hash', p_block_hash,
                'current_block', p_current, 'last_epoch_block', p_last,
                'pending_epoch_at', 0, 'subnet_epoch_index', p_index,
                'tempo', p_tempo, 'blocks_since_last_step', p_blocks_since,
                'observed_at', '{observed_at}', 'epoch_id', p_index,
                'epoch_ref', p_epoch_ref, 'epoch_block', p_current - p_last,
                'next_epoch_block', p_next, 'blocks_remaining', p_remaining,
                'settlement_epoch_id', p_settlement,
                'cutover_mapping_hash', '{mapping_hash}'
            ),
            '{observed_at}'::TIMESTAMPTZ
        );
        $fn$;
        """
        psql(boundary_helpers)

        # The normal contiguous path remains valid.
        psql(
            f"SELECT public.test_insert_epoch_receipt('{boundary_11_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 42, "
            f"'{boundary_11_hash}'); "
            f"SELECT public.test_insert_boundary('{boundary_11_hash}', "
            f"'{boundary_11_receipt}', 11, '{epoch_ref_11}', 42, 1360, "
            f"'{boundary_11_block_hash}', 360); "
            f"SELECT public.test_insert_epoch_receipt('{boundary_12_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 43, "
            f"'{boundary_12_hash}'); "
            f"SELECT public.test_insert_boundary('{boundary_12_hash}', "
            f"'{boundary_12_receipt}', 12, '{epoch_ref_12}', 43, 1720, "
            f"'{boundary_12_block_hash}', 360);"
        )

        # Missing index 13 must not brick the ledger: authenticated index 14
        # advances from the latest durable boundary. Once 14 is accepted, the
        # historical gap at 13 and any numerically-future row whose boundary
        # block moves backward are permanently rejected.
        psql(
            f"SELECT public.test_insert_epoch_receipt('{boundary_14_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 45, "
            f"'{boundary_14_hash}'); "
            f"SELECT public.test_insert_boundary('{boundary_14_hash}', "
            f"'{boundary_14_receipt}', 14, '{epoch_ref_14}', 45, 2440, "
            f"'{boundary_14_block_hash}', 360); "
            f"SELECT public.test_insert_epoch_receipt("
            f"'{boundary_14_snapshot_receipt}', 'validator_weights', "
            f"'validator.subnet_epoch_snapshot.v2', 45, "
            f"'{boundary_14_snapshot_hash}'); "
            f"SELECT public.test_insert_snapshot('{boundary_14_snapshot_hash}', "
            f"'{boundary_14_snapshot_receipt}', 14, '{epoch_ref_14}', 45, 2500, "
            f"2440, '{boundary_14_snapshot_block_hash}', 360, 60, 2800, 300); "
            f"SELECT public.test_insert_epoch_receipt('{boundary_13_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 44, "
            f"'{boundary_13_hash}'); "
            f"SELECT public.test_insert_epoch_receipt('{boundary_15_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 46, "
            f"'{boundary_15_hash}');"
        )
        rejected(
            f"SELECT public.test_insert_boundary('{boundary_13_hash}', "
            f"'{boundary_13_receipt}', 13, '{epoch_ref_13}', 44, 2080, "
            f"'{boundary_13_block_hash}', 360);",
            "stateful epoch boundary does not advance the latest accepted boundary",
        )
        rejected(
            f"SELECT public.test_insert_boundary('{boundary_15_hash}', "
            f"'{boundary_15_receipt}', 15, '{epoch_ref_15}', 46, 2439, "
            f"'{boundary_15_block_hash}', 360);",
            "stateful epoch boundary does not advance the latest accepted boundary",
        )

        # A forward row is still receipt-authenticated. The failed invalid
        # attempt does not consume index 15; its valid contiguous retry works.
        rejected(
            f"SELECT public.test_insert_boundary('{invalid_boundary_hash}', "
            f"'{invalid_boundary_receipt}', 15, '{epoch_ref_15}', 46, 2800, "
            f"'{invalid_boundary_block_hash}', 360);",
            "stateful epoch boundary chain-state receipt is invalid",
        )
        psql(
            f"SELECT public.test_insert_boundary('{boundary_15_hash}', "
            f"'{boundary_15_receipt}', 15, '{epoch_ref_15}', 46, 2800, "
            f"'{boundary_15_block_hash}', 360);"
        )
        psql(
            "INSERT INTO public.research_lab_attested_weight_bundles_v2("
            "bundle_hash,schema_version,netuid,epoch_id,block,validator_hotkey,"
            "root_receipt_hash,weights_hash,snapshot_hash,bundle_doc) VALUES ("
            f"'{sha(904)}','leadpoet.published_weight_bundle.v2',71,42,1700,"
            f"'stateful-validator','{boundary_11_receipt}','{hex64(905)}',"
            f"'{sha(906)}',pg_catalog.jsonb_build_object("
            f"'cutover_mapping_hash','{mapping_hash}',"
            "'receipt_graph',pg_catalog.jsonb_build_object("
            "'receipts',pg_catalog.jsonb_build_array("
            "pg_catalog.jsonb_build_object("
            f"'receipt_hash','{boundary_11_receipt}',"
            "'role','validator_weights',"
            "'purpose','validator.subnet_epoch_snapshot.v2',"
            "'epoch_id',42)))));"
        )

        # LastEpochBlock may move while SubnetEpochIndex and epoch_ref stay
        # stable. The post-block counter predicts the safety deadline at +2,
        # then +1, and clamps next to current once already over MAX_TEMPO.
        psql(
            f"SELECT public.test_insert_epoch_receipt('{mutable_snapshot_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 42, "
            f"'{mutable_snapshot_hash}'); "
            f"SELECT public.test_insert_snapshot('{mutable_snapshot_hash}', "
            f"'{mutable_snapshot_receipt}', 11, '{epoch_ref_11}', 42, 1400, "
            f"1380, '{mutable_snapshot_block_hash}', 400, 40, 1780, 380); "
            f"SELECT public.test_insert_epoch_receipt('{safety_50399_snapshot_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 42, "
            f"'{safety_50399_snapshot_hash}'); "
            f"SELECT public.test_insert_snapshot('{safety_50399_snapshot_hash}', "
            f"'{safety_50399_snapshot_receipt}', 11, '{epoch_ref_11}', 42, 1480, "
            f"1380, '{safety_50399_snapshot_block_hash}', 400, 50399, 1482, 2); "
            f"SELECT public.test_insert_epoch_receipt('{safety_50400_snapshot_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 42, "
            f"'{safety_50400_snapshot_hash}'); "
            f"SELECT public.test_insert_snapshot('{safety_50400_snapshot_hash}', "
            f"'{safety_50400_snapshot_receipt}', 11, '{epoch_ref_11}', 42, 1490, "
            f"1380, '{safety_50400_snapshot_block_hash}', 400, 50400, 1491, 1); "
            f"SELECT public.test_insert_epoch_receipt('{safety_snapshot_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 42, "
            f"'{safety_snapshot_hash}'); "
            f"SELECT public.test_insert_snapshot('{safety_snapshot_hash}', "
            f"'{safety_snapshot_receipt}', 11, '{epoch_ref_11}', 42, 1500, "
            f"1380, '{safety_snapshot_block_hash}', 400, 50401, 1500, 0);"
        )
        psql(
            f"SELECT public.test_insert_epoch_receipt('{bad_snapshot_receipt}', "
            f"'validator_weights', 'validator.subnet_epoch_snapshot.v2', 42, "
            f"'{bad_snapshot_hash}');"
        )
        rejected(
            f"SELECT public.test_insert_snapshot('{bad_snapshot_hash}', "
            f"'{bad_snapshot_receipt}', 11, '{wrong_epoch_ref}', 42, 1510, "
            f"1380, '{bad_snapshot_block_hash}', 400, 130, 1780, 270);",
            "stateful epoch snapshot has no matching boundary mapping",
        )

        # Rerunning after authority data exists must preserve rows and validate
        # the widened receipt purpose constraint successfully.
        psql(SQL)
        rerun_public_active = psql(
            "SET ROLE anon; SELECT lifecycle_state, mapping_hash "
            "FROM public.research_lab_stateful_subnet_epoch_cutover_public_state_v1(); "
            "RESET ROLE;"
        )
        assert f"stateful_active|{mapping_hash}" in rerun_public_active.stdout
        psql(HISTORICAL_PREDECESSOR_SQL)
        psql(HISTORICAL_PREDECESSOR_SQL)
        result = psql(
            """
            SELECT
              (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_candidates_v1),
              (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_cutovers_v1),
              (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_boundaries_v1),
              (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_snapshots_v1),
              (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_mapping_v1),
              (SELECT COUNT(*) FROM pg_catalog.pg_class
               WHERE relnamespace = 'public'::REGNAMESPACE
                 AND relname LIKE 'research_lab_stateful_subnet_epoch_%_v1'
                 AND relrowsecurity);
            """
        )
        assert "1|1|4|5|5|5" in result.stdout.replace(" ", "")
        rejected(
            "UPDATE public.research_lab_stateful_subnet_epoch_boundaries_v1 "
            "SET tempo = tempo WHERE subnet_epoch_index = 11;",
            "append-only",
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        for process in background_processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
