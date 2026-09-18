from __future__ import annotations

import json

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_COST_BACKFILL_RETIREMENT_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.postgres_migration_harness import SCRIPTS


_CURRENT_WRITE_GUARD = """(NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference') =
          (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference')"""
_UNKNOWN_LEGACY_WRITE_GUARD = """(
        (NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference') =
          (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference')
        OR (
          OLD.configuration_doc ->> 'mode' = 'live'
          AND NOT OLD.configuration_doc ? 'cost_per_company_microusd'
          AND OLD.icp_set_date IS NULL
          AND NEW.icp_set_date IS NOT NULL
          AND (NEW.configuration_doc ->> 'execution_cap_microusd')::BIGINT = 50000001
          AND (NEW.configuration_doc ->> 'cost_per_company_microusd')::BIGINT = 500001
          AND (NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference'
               - 'execution_cap_microusd' - 'cost_per_company_microusd') =
              (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference'
               - 'execution_cap_microusd')
        )
      )"""


_LEGACY_FUNCTION_FIXTURE = r"""
DO $fixture$
DECLARE
  v_definition TEXT;
  v_current_assignment TEXT := 'configuration_doc = v_round.configuration_doc || pg_catalog.jsonb_build_object(
        ''scorer_image_digest'', p_scorer_image_digest,
        ''scorer_image_reference'', p_scorer_image_reference
      )';
  v_legacy_assignment TEXT := 'configuration_doc = (
        CASE
          WHEN v_round.configuration_doc ->> ''mode'' = ''live''
               AND NOT v_round.configuration_doc ? ''cost_per_company_microusd''
          THEN v_round.configuration_doc || pg_catalog.jsonb_build_object(
            ''execution_cap_microusd'', 50000000,
            ''cost_per_company_microusd'', 500000
          )
          ELSE v_round.configuration_doc
        END
      ) || pg_catalog.jsonb_build_object(
        ''scorer_image_digest'', p_scorer_image_digest,
        ''scorer_image_reference'', p_scorer_image_reference
      )';
  v_guarded_anchor TEXT := '  IF v_round.configuration_doc ->> ''mode'' = ''live''
     AND (
       NOT v_round.configuration_doc ? ''execution_cap_microusd''
       OR NOT v_round.configuration_doc ? ''cost_per_company_microusd''
     ) THEN
    RAISE EXCEPTION ''lab_arena_round_cost_policy_missing''
      USING ERRCODE = ''22023'';
  END IF;
  SELECT COUNT(*) INTO v_pending_admissions
  FROM public.lab_arena_submissions';
  v_unguarded_anchor TEXT := '  SELECT COUNT(*) INTO v_pending_admissions
  FROM public.lab_arena_submissions';
  v_current_write_guard TEXT := '(NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'') =
          (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'')';
  v_legacy_write_guard TEXT := '(
        (NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'') =
          (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'')
        OR (
          OLD.configuration_doc ->> ''mode'' = ''live''
          AND NOT OLD.configuration_doc ? ''cost_per_company_microusd''
          AND OLD.icp_set_date IS NULL
          AND NEW.icp_set_date IS NOT NULL
          AND (NEW.configuration_doc ->> ''execution_cap_microusd'')::BIGINT = 50000000
          AND (NEW.configuration_doc ->> ''cost_per_company_microusd'')::BIGINT = 500000
          AND (NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference''
               - ''execution_cap_microusd'' - ''cost_per_company_microusd'') =
              (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference''
               - ''execution_cap_microusd'')
        )
      )';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, v_current_assignment) = 0
     OR pg_catalog.strpos(v_definition, v_guarded_anchor) = 0 THEN
    RAISE EXCEPTION 'current commit fixture differs';
  END IF;
  EXECUTE pg_catalog.replace(
    pg_catalog.replace(
      v_definition, v_current_assignment, v_legacy_assignment
    ),
    v_guarded_anchor, v_unguarded_anchor
  );

  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure('public.lab_arena_rounds_write_once_v1()')
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, v_current_write_guard) = 0 THEN
    RAISE EXCEPTION 'current write guard fixture differs';
  END IF;
  EXECUTE pg_catalog.replace(
    v_definition, v_current_write_guard, v_legacy_write_guard
  );
END;
$fixture$;
"""


def _definitions(cursor):
    cursor.execute(
        "SELECT pg_get_functiondef(to_regprocedure("
        "'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'))"
    )
    commit = cursor.fetchone()[0]
    cursor.execute(
        "SELECT pg_get_functiondef("
        "to_regprocedure('public.lab_arena_rounds_write_once_v1()'))"
    )
    return commit, cursor.fetchone()[0]


def _state(cursor):
    cursor.execute(
        """
        SELECT pg_catalog.jsonb_build_object(
          'rounds', (
            SELECT COALESCE(pg_catalog.jsonb_agg(to_jsonb(row) ORDER BY round_id), '[]'::JSONB)
            FROM public.lab_arena_rounds AS row
          ),
          'submissions', (
            SELECT COALESCE(pg_catalog.jsonb_agg(to_jsonb(row) ORDER BY submission_id), '[]'::JSONB)
            FROM public.lab_arena_submissions AS row
          ),
          'runs', (
            SELECT COALESCE(pg_catalog.jsonb_agg(to_jsonb(row) ORDER BY run_id), '[]'::JSONB)
            FROM public.lab_arena_runs AS row
          ),
          'ledger', (
            SELECT COALESCE(pg_catalog.jsonb_agg(to_jsonb(row) ORDER BY entry_id), '[]'::JSONB)
            FROM public.lab_arena_ledger AS row
          )
        )
        """
    )
    return cursor.fetchone()[0]


def test_migration_retires_deployed_backfill_without_rewriting_round_state():
    database = database_with_lab_arena_migration(DEFAULT_MIGRATIONS)
    connection = None
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        migration = (
            SCRIPTS / LAB_ARENA_COST_BACKFILL_RETIREMENT_MIGRATION
        ).read_text(encoding="utf-8")
        with connection.cursor() as cursor:
            # A fresh database already has the retired behavior in migration
            # 206. Migration 294 must therefore be safely repeatable.
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(_LEGACY_FUNCTION_FIXTURE)

            digest = "sha256:" + "a" * 64
            reference = "registry.example/lab/scorer@" + digest
            hotkey = "5" + "A" * 47
            current = {
                "round_id": "arena-2026-09-21-current",
                "mode": "live",
                "schedule": {
                    "submission_open": "2026-09-20T00:00:00Z",
                    "submission_cutoff": "2026-09-21T00:00:00Z",
                },
                "max_challengers": 1,
                "execution_cap_microusd": 80_000_000,
                "cost_per_company_microusd": 800_000,
                "scorer_image_digest": digest,
                "scorer_image_reference": reference,
                "baseline_hotkey": hotkey,
            }
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc) VALUES (%s, 'open', %s::JSONB)",
                (current["round_id"], json.dumps(current)),
            )
            historical = []
            for index in range(10):
                status = "cancelled" if index < 6 else "published"
                round_id = "arena-2026-08-%02d-h%d" % (index + 1, index)
                configuration = {"round_id": round_id, "mode": "live"}
                historical.append((round_id, status, json.dumps(configuration)))
            cursor.executemany(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc) VALUES (%s, %s, %s::JSONB)",
                historical,
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id, round_id, miner_hotkey, status, is_king) "
                "VALUES ('current-baseline', %s, %s, 'frozen', TRUE)",
                (current["round_id"], hotkey),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id, assignment_id, round_id, submission_id, miner_hotkey, "
                "stage, icp_position, attempt) VALUES "
                "('current-run', 'current-assignment', %s, 'current-baseline', %s, 1, 0, 1)",
                (current["round_id"], hotkey),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind, miner_hotkey, round_id, submission_id, run_id, "
                "stage, call_identity, provider, operation_id, funding_source, "
                "amount_microusd) VALUES "
                "('refusal', %s, %s, 'current-baseline', 'current-run', 1, %s, "
                "'deepline', 'test', 'host', 0)",
                (hotkey, current["round_id"], "sha256:" + "b" * 64),
            )
            before = _state(cursor)
            cursor.execute(
                "SELECT public.lab_arena_submission_costs('current-baseline')"
            )
            costs_before = cursor.fetchone()[0]

            cursor.execute(migration)

            assert _state(cursor) == before
            cursor.execute(
                "SELECT public.lab_arena_submission_costs('current-baseline')"
            )
            assert cursor.fetchone()[0] == costs_before
            commit_definition, write_guard_definition = _definitions(cursor)
            assert "lab_arena_round_cost_policy_missing" in commit_definition
            assert "THEN v_round.configuration_doc ||" not in commit_definition
            assert "AND NOT OLD.configuration_doc" not in write_guard_definition

            cursor.execute(
                "SELECT public.lab_arena_commit_round_v2("
                "%s, %s::JSONB, %s, %s, %s::DATE, %s, %s)",
                (
                    current["round_id"],
                    json.dumps(
                        [
                            {
                                "submission_id": "current-baseline",
                                "miner_hotkey": hotkey,
                                "is_king": True,
                            }
                        ]
                    ),
                    "arena/arena-2026-09-21-current/benchmark.json",
                    "2026-09-21",
                    "2026-09-20",
                    digest,
                    reference,
                ),
            )
            assert cursor.fetchone()[0]["status"] == "ok"
            cursor.execute(
                "SELECT configuration_doc FROM public.lab_arena_rounds "
                "WHERE round_id = %s",
                (current["round_id"],),
            )
            assert cursor.fetchone()[0] == current
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds "
                "WHERE status IN ('published', 'cancelled') "
                "AND NOT configuration_doc ? 'cost_per_company_microusd'"
            )
            assert cursor.fetchone()[0] == 10

            commit_before_unknown, write_guard = _definitions(cursor)
            assert _CURRENT_WRITE_GUARD in write_guard
            unknown_write_guard = write_guard.replace(
                _CURRENT_WRITE_GUARD, _UNKNOWN_LEGACY_WRITE_GUARD
            )
            assert unknown_write_guard != write_guard
            cursor.execute(unknown_write_guard)
            with pytest.raises(
                psycopg2.Error,
                match="lab_arena_cost_policy_write_guard_unknown",
            ):
                cursor.execute(migration)
            cursor.execute("ROLLBACK")
            assert _definitions(cursor) == (
                commit_before_unknown,
                unknown_write_guard,
            )

            missing = {
                **current,
                "round_id": "arena-2026-09-22-missing",
            }
            missing.pop("cost_per_company_microusd")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc) VALUES (%s, 'open', %s::JSONB)",
                (missing["round_id"], json.dumps(missing)),
            )
            with pytest.raises(
                psycopg2.Error, match="lab_arena_round_cost_policy_missing"
            ):
                cursor.execute(
                    "SELECT public.lab_arena_commit_round_v2("
                    "%s, '[]'::JSONB, 'benchmark', '2026-09-21', "
                    "'2026-09-20'::DATE, %s, %s)",
                    (missing["round_id"], digest, reference),
                )
            cursor.execute(
                "SELECT status, configuration_doc FROM public.lab_arena_rounds "
                "WHERE round_id = %s",
                (missing["round_id"],),
            )
            assert cursor.fetchone() == ("open", missing)
            with pytest.raises(
                psycopg2.Error, match="active live round is missing its cost policy"
            ):
                cursor.execute(migration)
    finally:
        if connection is not None:
            connection.close()
        database.close()
