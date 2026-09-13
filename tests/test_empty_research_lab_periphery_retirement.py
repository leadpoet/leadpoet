"""Contract tests for the confirmed-empty Research Lab periphery retirement."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.postgres_migration_harness import _database_with_migrations


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "231-retire-empty-research-lab-periphery.sql"

RETIRED_TABLES = (
    "engine_generated_candidates",
    "engine_generated_datasets",
    "engine_generated_evaluators",
    "engine_issue_events",
    "engine_issues",
    "research_lab_autoresearch_frontier_commitments",
    "research_lab_autoresearch_operation_settlements",
    "research_lab_autoresearch_tree_events",
    "research_lab_autoresearch_tree_handoffs",
    "research_lab_autoresearch_tree_nodes",
    "research_lab_autoresearch_trees",
    "research_lab_candidate_model_unit_terminals",
    "research_lab_candidate_waterfall_metrics",
    "research_lab_candidate_waterfall_receipts",
    "research_lab_conditional_validation_events",
    "research_lab_routing_adapter_failures_v2",
    "research_lab_routing_budget_events_v2",
    "research_lab_routing_decision_receipts_v2",
    "research_lab_routing_evaluation_receipts_v2",
    "research_lab_routing_execution_request_leases_v2",
    "research_lab_routing_execution_requests_v2",
    "research_lab_routing_experiment_claim_closures_v2",
    "research_lab_routing_experiment_claim_closures_v3",
    "research_lab_routing_experiment_claim_heartbeats_v2",
    "research_lab_routing_experiment_claim_heartbeats_v3",
    "research_lab_routing_experiment_claims_v2",
    "research_lab_routing_experiment_claims_v3",
    "research_lab_routing_experiment_events_v2",
    "research_lab_routing_experiments_v2",
    "research_lab_routing_lab_references_v2",
    "research_lab_routing_provider_attempts_v2",
    "research_lab_scoring_job_candidate",
    "research_lab_scoring_job_queue",
)
PROTECTED_SOURCE_ADD_GRAPH = (
    "research_lab_attested_ancestry_activations_v2",
    "research_lab_attested_ancestry_checkpoints_v2",
    "research_lab_attested_artifact_links_v2",
    "research_lab_attested_boot_identities_v2",
    "research_lab_attested_business_artifact_links_v2",
    "research_lab_attested_execution_receipts_v2",
    "research_lab_attested_execution_results_v2",
    "research_lab_attested_host_operations_v2",
    "research_lab_attested_receipt_edges_v2",
    "research_lab_attested_receipt_transport_v2",
    "research_lab_attested_transport_attempts_v2",
)
RETIRED_VIEWS = (
    "research_lab_autoresearch_operation_current",
    "research_lab_autoresearch_run_tree_current",
    "research_lab_autoresearch_tree_current",
    "research_lab_autoresearch_tree_node_current",
)


def _without_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_retirement_sql_has_exact_fail_closed_boundaries() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable = _without_comments(sql).upper()

    assert "CASCADE" not in executable
    assert "ACCESS EXCLUSIVE MODE" in executable
    assert "SELECT EXISTS (SELECT FROM PUBLIC.%I LIMIT 1)" in executable
    assert "UNEXPECTED ROUTINE DEPENDS ON RETIREMENT CLOSURE" in executable
    assert "RETIREMENT ROUTINE IS USED BY A RETAINED TRIGGER" in executable
    assert len(RETIRED_TABLES) == 33
    for table in RETIRED_TABLES:
        assert f"('{table}')" in sql
    for table in PROTECTED_SOURCE_ADD_GRAPH:
        assert f"('{table}')" not in sql
    for retained_v1 in (
        "research_lab_attested_artifact_links",
        "research_lab_attested_execution_receipts",
    ):
        assert f"('{retained_v1}')" not in sql
    assert "SOURCE_ADD" in sql


BASE_SETUP_SQL = r"""
CREATE TABLE public.engine_generated_candidates (id TEXT PRIMARY KEY);
CREATE TABLE public.engine_generated_datasets (id TEXT PRIMARY KEY);
CREATE TABLE public.engine_generated_evaluators (id TEXT PRIMARY KEY);
CREATE TABLE public.engine_issue_events (id TEXT PRIMARY KEY);
CREATE TABLE public.engine_issues (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_autoresearch_frontier_commitments (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_autoresearch_operation_settlements (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_autoresearch_tree_events (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_autoresearch_tree_handoffs (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_autoresearch_tree_nodes (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_autoresearch_trees (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_candidate_claim (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_candidate_model_unit_terminals (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_candidate_waterfall_metrics (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_candidate_waterfall_receipts (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_conditional_validation_events (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_adapter_failures_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_budget_events_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_decision_receipts_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_evaluation_receipts_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_execution_request_leases_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_execution_requests_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_claim_closures_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_claim_closures_v3 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_claim_heartbeats_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_claim_heartbeats_v3 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_claims_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_claims_v3 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiment_events_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_experiments_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_lab_references_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_routing_provider_attempts_v2 (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_scoring_job_candidate (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_scoring_job_queue (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_attested_ancestry_activations_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_ancestry_checkpoints_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_artifact_links_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_boot_identities_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_business_artifact_links_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_execution_receipts_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_execution_results_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_host_operations_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_receipt_edges_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_receipt_transport_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_transport_attempts_v2 (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_artifact_links (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.research_lab_attested_execution_receipts (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.lab_arena_rounds (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
CREATE TABLE public.qualification_private_icp_sets (id TEXT PRIMARY KEY, payload JSONB NOT NULL DEFAULT '{}'::JSONB);
INSERT INTO public.research_lab_attested_ancestry_activations_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_ancestry_checkpoints_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_artifact_links_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_boot_identities_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_business_artifact_links_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_execution_receipts_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_execution_results_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_host_operations_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_receipt_edges_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_receipt_transport_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_transport_attempts_v2 VALUES ('source-add-history', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_artifact_links VALUES ('v1-link', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_attested_execution_receipts VALUES ('v1-receipt', '{"purpose":"SOURCE_ADD"}'::JSONB);
INSERT INTO public.research_lab_candidate_claim VALUES ('retained-claim');
INSERT INTO public.lab_arena_rounds VALUES ('active-round', '{}'::JSONB);
INSERT INTO public.qualification_private_icp_sets VALUES ('active-icp', '{}'::JSONB);
CREATE VIEW public.research_lab_autoresearch_operation_current AS SELECT * FROM public.research_lab_autoresearch_operation_settlements;
CREATE VIEW public.research_lab_autoresearch_run_tree_current AS SELECT * FROM public.research_lab_autoresearch_trees;
CREATE VIEW public.research_lab_autoresearch_tree_current AS SELECT * FROM public.research_lab_autoresearch_trees;
CREATE VIEW public.research_lab_autoresearch_tree_node_current AS SELECT * FROM public.research_lab_autoresearch_tree_nodes;
CREATE FUNCTION public.research_lab_routing_canonical_jsonb_v2(p_value JSONB)
RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$ SELECT p_value::TEXT $$;
CREATE FUNCTION public.research_lab_routing_jsonb_hash_v2(p_value JSONB)
RETURNS TEXT LANGUAGE SQL IMMUTABLE
AS $$ SELECT public.research_lab_routing_canonical_jsonb_v2(p_value) $$;
CREATE FUNCTION public.research_lab_official_baseline_hash_v1(p_value JSONB)
RETURNS TEXT LANGUAGE SQL IMMUTABLE
AS $$ SELECT public.research_lab_routing_jsonb_hash_v2(p_value) $$;
CREATE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    PERFORM 1 FROM public.research_lab_attested_execution_receipts LIMIT 1;
    RETURN NEW;
END $$;
CREATE TRIGGER retained_epoch_fence_v1
BEFORE UPDATE ON public.research_lab_attested_execution_receipts
FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();
CREATE TRIGGER retained_epoch_fence_v2
BEFORE UPDATE ON public.research_lab_attested_execution_receipts_v2
FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();
CREATE FUNCTION public.guard_research_lab_candidate_claim()
RETURNS TRIGGER LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$;
CREATE TRIGGER retained_candidate_claim_guard
BEFORE UPDATE ON public.research_lab_candidate_claim
FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_candidate_claim();
"""


def _open_database(setup_sql: str):
    database = _database_with_migrations((), setup_sql=setup_sql)
    psycopg2, dsn = next(database)
    return database, psycopg2, dsn


def test_disposable_postgres_drops_only_empty_retired_closure() -> None:
    database, psycopg2, dsn = _open_database(BASE_SETUP_SQL)
    try:
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                migration = MIGRATION.read_text(encoding="utf-8")
                cursor.execute(migration)
                cursor.execute(migration)

                for relation in RETIRED_TABLES + RETIRED_VIEWS:
                    cursor.execute("SELECT to_regclass(%s)", ("public." + relation,))
                    assert cursor.fetchone() == (None,)
                for relation in PROTECTED_SOURCE_ADD_GRAPH:
                    cursor.execute(
                        f"SELECT payload ->> 'purpose' FROM public.{relation} WHERE id = %s",
                        ("source-add-history",),
                    )
                    assert cursor.fetchone() == ("SOURCE_ADD",)

                cursor.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM public.research_lab_attested_artifact_links),
                        (SELECT count(*) FROM public.research_lab_attested_execution_receipts),
                        (SELECT count(*) FROM public.research_lab_candidate_claim),
                        (SELECT count(*) FROM public.lab_arena_rounds),
                        (SELECT count(*) FROM public.qualification_private_icp_sets),
                        to_regprocedure('public.enforce_research_lab_stateful_epoch_fence_v1()') IS NOT NULL,
                        to_regprocedure('public.guard_research_lab_candidate_claim()') IS NOT NULL,
                        to_regprocedure('public.research_lab_routing_jsonb_hash_v2(jsonb)') IS NOT NULL,
                        to_regprocedure('public.research_lab_official_baseline_hash_v1(jsonb)') IS NOT NULL,
                        to_regprocedure('public.research_lab_routing_append_event_v2(text,text,text,jsonb)') IS NULL,
                        public.research_lab_official_baseline_hash_v1('{"ok":true}'::JSONB)
                    """
                )
                assert cursor.fetchone() == (
                    1, 1, 1, 1, 1, True, True, True, True, True, '{"ok": true}'
                )
    finally:
        database.close()


@pytest.mark.parametrize(
    "extra_sql, expected_error",
    (
        (
            "INSERT INTO public.research_lab_routing_experiments_v2 VALUES ('new-row');",
            "refusing to retire nonempty table",
        ),
        (
            """
            CREATE SCHEMA private;
            CREATE VIEW private.unexpected_candidate_view AS
            SELECT * FROM public.research_lab_routing_experiments_v2;
            """,
            "unexpected view depends on retirement table",
        ),
        (
            """
            CREATE SCHEMA private;
            CREATE FUNCTION private.unexpected_candidate_reader()
            RETURNS BIGINT LANGUAGE plpgsql AS $$
            DECLARE result BIGINT;
            BEGIN
                SELECT count(*) INTO result
                FROM public.research_lab_routing_experiments_v2;
                RETURN result;
            END $$;
            """,
            "unexpected routine depends on retirement closure",
        ),
        (
            """
            CREATE FUNCTION public.research_lab_routing_append_event_v2(
                p_event_hash TEXT, p_experiment_hash TEXT,
                p_event_type TEXT, p_event_doc JSONB
            ) RETURNS VOID LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$;
            """,
            "retirement routine changed after review",
        ),
        (
            """
            CREATE SCHEMA cron;
            CREATE TABLE cron.job (jobname TEXT PRIMARY KEY, command TEXT NOT NULL);
            INSERT INTO cron.job VALUES (
                'unexpected-retired-call',
                'SELECT public.research_lab_routing_append_event_v2()'
            );
            """,
            "scheduled job depends on retirement closure",
        ),
        (
            """
            CREATE TABLE public.unexpected_candidate_reference (
                id TEXT PRIMARY KEY REFERENCES public.research_lab_routing_experiments_v2(id)
            );
            """,
            "cannot drop desired object",
        ),
    ),
)
def test_disposable_postgres_rolls_back_for_rows_or_new_dependencies(
    extra_sql: str, expected_error: str
) -> None:
    database, psycopg2, dsn = _open_database(BASE_SETUP_SQL + "\n" + extra_sql)
    try:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                with pytest.raises(psycopg2.Error) as exc_info:
                    cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                assert expected_error in str(exc_info.value).lower()
                cursor.execute("ROLLBACK")
                cursor.execute(
                    """
                    SELECT
                        to_regclass('public.research_lab_routing_experiments_v2') IS NOT NULL,
                        to_regclass('public.engine_issues') IS NOT NULL,
                        (SELECT count(*) FROM public.lab_arena_rounds)
                    """
                )
                assert cursor.fetchone() == (True, True, 1)
        finally:
            connection.close()
    finally:
        database.close()
