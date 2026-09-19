from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "307-retire-legacy-research-lab-functions.sql"

RETIRED_SIGNATURES = (
    "public.put_research_lab_provider_evidence_cache_v2(jsonb)",
    "public.research_lab_attested_execution_result_secret_free_v2(text,jsonb)",
    "public.research_lab_execution_trace_id(uuid)",
    "public.research_lab_trajectory_id(uuid)",
    "public.research_lab_deterministic_uuid(text)",
    "public.research_lab_routing_jsonb_hash_v2(jsonb)",
    "public.research_lab_routing_canonical_jsonb_v2(jsonb)",
    "public.research_lab_unpaid_ticket_expires_at(timestamp with time zone)",
)

ACTIVE_SIGNATURES = (
    "public.prevent_research_lab_attested_v2_mutation()",
    "public.enforce_research_lab_stateful_epoch_fence_v1()",
    "public.validate_research_lab_stateful_epoch_cutover_v2()",
    "public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()",
)

ACTIVE_TRIGGERS = (
    "prevent_research_lab_provider_evidence_cache_v2_mutation",
    "prevent_research_lab_stateful_epoch_cutover_v1_mutation",
    "enforce_research_lab_stateful_epoch_fence_v1",
    "validate_research_lab_stateful_epoch_cutover_v1",
)

SETUP_SQL = r"""
CREATE TABLE public.research_lab_provider_evidence_cache_v2 (
  cache_key text PRIMARY KEY,
  payload jsonb NOT NULL
);
INSERT INTO public.research_lab_provider_evidence_cache_v2(cache_key, payload)
VALUES ('historical', '{"retained": true, "row_count_at_audit": 103910}'::jsonb);

CREATE TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 (
  epoch_id bigint PRIMARY KEY,
  state jsonb NOT NULL
);
INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1(epoch_id, state)
VALUES (71, '{"active": true}'::jsonb);

CREATE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  RETURN NEW;
END
$$;

CREATE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RETURN NEW;
END
$$;

CREATE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RETURN NEW;
END
$$;

CREATE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()
RETURNS jsonb LANGUAGE sql STABLE
AS $$ SELECT '{"active": true}'::jsonb $$;

CREATE TRIGGER prevent_research_lab_provider_evidence_cache_v2_mutation
BEFORE UPDATE OR DELETE ON public.research_lab_provider_evidence_cache_v2
FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

CREATE TRIGGER prevent_research_lab_stateful_epoch_cutover_v1_mutation
BEFORE UPDATE OR DELETE ON public.research_lab_stateful_subnet_epoch_cutovers_v1
FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
BEFORE INSERT OR UPDATE ON public.research_lab_stateful_subnet_epoch_cutovers_v1
FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

CREATE TRIGGER validate_research_lab_stateful_epoch_cutover_v1
BEFORE INSERT OR UPDATE ON public.research_lab_stateful_subnet_epoch_cutovers_v1
FOR EACH ROW EXECUTE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2();

CREATE FUNCTION public.put_research_lab_provider_evidence_cache_v2(cache_row jsonb)
RETURNS jsonb LANGUAGE sql AS $$ SELECT cache_row $$;

CREATE FUNCTION public.research_lab_attested_execution_result_secret_free_v2(
  p_operation text,
  p_result_doc jsonb
)
RETURNS boolean LANGUAGE sql AS $$ SELECT p_operation <> '' AND p_result_doc IS NOT NULL $$;

CREATE FUNCTION public.research_lab_deterministic_uuid(p_canonical_json text)
RETURNS uuid LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
  RETURN '00000000-0000-0000-0000-000000000001'::uuid;
END;

CREATE FUNCTION public.research_lab_execution_trace_id(p_run_id uuid)
RETURNS uuid LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
  RETURN public.research_lab_deterministic_uuid('trace:' || p_run_id::text);
END;

CREATE FUNCTION public.research_lab_trajectory_id(p_run_id uuid)
RETURNS uuid LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
  RETURN public.research_lab_deterministic_uuid('trajectory:' || p_run_id::text);
END;

CREATE FUNCTION public.research_lab_routing_canonical_jsonb_v2(p_value jsonb)
RETURNS jsonb LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
  RETURN p_value;
END;

CREATE FUNCTION public.research_lab_routing_jsonb_hash_v2(p_value jsonb)
RETURNS text LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
  RETURN md5(public.research_lab_routing_canonical_jsonb_v2(p_value)::text);
END;

CREATE FUNCTION public.research_lab_unpaid_ticket_expires_at(
  ticket_created_at timestamp with time zone
)
RETURNS timestamp with time zone LANGUAGE sql IMMUTABLE
AS $$ SELECT ticket_created_at + interval '1 hour' $$;
"""


def _regprocedure_oids(cursor, signatures: tuple[str, ...]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for signature in signatures:
        cursor.execute("SELECT pg_catalog.to_regprocedure(%s)::oid", (signature,))
        result[signature] = cursor.fetchone()[0]
    return result


def test_migration_is_restrictive_idempotent_and_preserves_active_state():
    database = database_with_lab_arena_migration(())
    try:
        psycopg2, dsn = next(database)
        migration_sql = MIGRATION.read_text(encoding="utf-8")
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute(SETUP_SQL)
                cursor.execute(
                    "SELECT 'public.research_lab_provider_evidence_cache_v2'::regclass::oid, "
                    "'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass::oid"
                )
                table_oids = cursor.fetchone()
                active_oids = _regprocedure_oids(cursor, ACTIVE_SIGNATURES)

                cursor.execute(
                    "CREATE VIEW public.research_lab_legacy_dependency_tripwire AS "
                    "SELECT public.put_research_lab_provider_evidence_cache_v2('{}'::jsonb) AS payload"
                )
                with pytest.raises(psycopg2.Error) as error:
                    cursor.execute(migration_sql)
                assert error.value.pgcode == "2BP01"
                cursor.execute("ROLLBACK")

                assert all(_regprocedure_oids(cursor, RETIRED_SIGNATURES).values())
                cursor.execute(
                    "SELECT pg_catalog.to_regclass("
                    "'public.research_lab_legacy_dependency_tripwire') IS NOT NULL"
                )
                assert cursor.fetchone()[0] is True
                cursor.execute("DROP VIEW public.research_lab_legacy_dependency_tripwire")

                cursor.execute(migration_sql)
                cursor.execute(migration_sql)

                assert not any(_regprocedure_oids(cursor, RETIRED_SIGNATURES).values())
                assert _regprocedure_oids(cursor, ACTIVE_SIGNATURES) == active_oids
                cursor.execute(
                    "SELECT 'public.research_lab_provider_evidence_cache_v2'::regclass::oid, "
                    "'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass::oid"
                )
                assert cursor.fetchone() == table_oids
                cursor.execute(
                    "SELECT payload FROM public.research_lab_provider_evidence_cache_v2 "
                    "WHERE cache_key = 'historical'"
                )
                assert cursor.fetchone()[0] == {
                    "retained": True,
                    "row_count_at_audit": 103910,
                }
                cursor.execute(
                    "SELECT state FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 "
                    "WHERE epoch_id = 71"
                )
                assert cursor.fetchone()[0] == {"active": True}
                cursor.execute(
                    "SELECT tgname, tgenabled FROM pg_catalog.pg_trigger "
                    "WHERE NOT tgisinternal AND tgname = ANY(%s) ORDER BY tgname",
                    (list(ACTIVE_TRIGGERS),),
                )
                assert cursor.fetchall() == sorted((name, "O") for name in ACTIVE_TRIGGERS)
        finally:
            connection.close()
    finally:
        database.close()
