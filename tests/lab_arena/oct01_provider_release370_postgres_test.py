"""Provider recovery requeues only failed October 1 baseline judgments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena import oct01_provider_hold369_postgres_test as hold
from lab_arena import contracts
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete


ROOT = Path(__file__).parents[2]
RELEASE = ROOT / "scripts/370-arena-2026-10-01-provider-score-release.sql"
database = hold.database


def _prepare(cursor):
    hold._seed_active(cursor)
    cursor.execute(hold._render_hold(cursor))
    cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                   "terminal_cause='accepted',output_ref='arena/test/score2.json' "
                   "WHERE run_id='score369:2:1'")
    cursor.execute("UPDATE public.lab_arena_runs SET status='failed',"
                   "terminal_cause='judge_error',result_doc="
                   "'{\"terminal_status\":\"judge_error\",\"failure_diagnostic\":"
                   "{\"reason\":\"provider_error\"}}'::jsonb "
                   "WHERE run_id='score369:3:1'")
    cursor.execute("INSERT INTO public.lab_arena_runs "
                   "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                   "stage,icp_position,attempt,kind,status,stage_generation,"
                   "scored_run_id,terminal_cause,result_doc) "
                   "SELECT 'score369:3:2',assignment_id,round_id,submission_id,"
                   "miner_hotkey,stage,icp_position,2,kind,'failed',stage_generation,"
                   "scored_run_id,'judge_error',result_doc "
                   "FROM public.lab_arena_runs WHERE run_id='score369:3:1'")
    return _replace_hashes(cursor, RELEASE.read_text())


def _replace_hashes(cursor, sql):
    source = hold.HOLD.read_text()
    rendered = hold._render_hold(cursor)
    import re
    original = re.findall(r"'[0-9a-f]{64}'", source)
    fixture = re.findall(r"'[0-9a-f]{64}'", rendered)
    assert len(original) == len(fixture) == 5
    for old, new in zip(original, fixture):
        assert old in sql
        sql = sql.replace(old, new)
    sql = sql.replace(
        "arena-2026-10-01:baseline-2026-10-01:1:0:score:1",
        "score369:0:1",
    ).replace(
        "sha256:0ecfb67d5dee9938a55a0362dc5a37893191ac883485f403b0d8eab06e56c09d",
        "sha256:" + "a" * 64,
    ).replace(
        "arena-2026-10-01:baseline-2026-10-01:1:3:score:2",
        "score369:3:2",
    ).replace(
        "sha256:f513b2588cd3f4ee4a02aea4beb1b71105737b79674f4f1e5aa305cc242fd1e8",
        "sha256:" + "b" * 64,
    )
    return sql


def test_release_requeues_failed_judge_and_preserves_all_existing_rows(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SELECT run_id,status,terminal_cause,scored_run_id,output_ref "
                           "FROM public.lab_arena_runs WHERE round_id=%s "
                           "ORDER BY run_id", (hold.recovery.ROUND,))
            before = cursor.fetchall()
            cursor.execute(sql)
            cursor.execute("SELECT run_id,status,terminal_cause,scored_run_id,output_ref "
                           "FROM public.lab_arena_runs WHERE round_id=%s "
                           "ORDER BY run_id", (hold.recovery.ROUND,))
            after = cursor.fetchall()
            assert all(row in after for row in before)
            added = [row for row in after if row not in before]
            assert added == [
                ("score369:0:2", "pending", None,
                 next(r[3] for r in before if r[0] == "score369:0:1"), None),
                ("score369:3:3", "pending", None,
                 next(r[3] for r in before if r[0] == "score369:3:2"), None),
            ]
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref "
                           "FROM public.lab_arena_restart_claim_control")
            assert cursor.fetchone() == (False, "", "oct01-provider-recovery370")
            cursor.execute(sql)
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE run_id='score369:0:2'")
            assert cursor.fetchone()[0] == 1
            cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                           "WHERE run_id='score369:0:2'")
            cursor.execute("SELECT status FROM public.lab_arena_runs "
                           "WHERE run_id='score369:0:2'")
            assert cursor.fetchone()[0] == "leased"


def test_appended_attempt_three_claims_completes_and_counts_at_scoring_close(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute(sql)
            # All remaining positions have an accepted score in this control;
            # only the appended retries remain to finish the stage.
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',"
                           "output_ref='arena/test/'||run_id||'.json' "
                           "WHERE round_id=%s AND kind='score' AND status='pending' "
                           "AND icp_position NOT IN (0,3)", (hold.recovery.ROUND,))
            cursor.execute("SET session_replication_role=origin")
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    runner = hold.recovery._configuration()["runner_hotkeys"][0]
    claimed = []
    for expected in ("score369:0:2", "score369:3:3"):
        run, token, _, _ = claim(
            store, hold.recovery.ROUND, runner,
            excluded=[runner], parallelism=1, ceiling=20,
        )
        assert run["status"] == "leased" and run["run_id"] == expected
        assert complete(
            store, expected, hash_lease_token(token), "accepted",
            output_ref="arena/test/" + expected + ".json",
        )["status"] == "accepted"
        claimed.append(run)
    assert store.close_scoring(hold.recovery.ROUND, 1)["status"] == "closed"
    chosen = ArenaService._select_scoring_outputs(
        store.list_runs(hold.recovery.ROUND, stage=1, kind="score")
    )
    for run in claimed:
        assert chosen[run["scored_run_id"]]["run_id"] == run["run_id"]
    assert store.get_round(hold.recovery.ROUND)["status"] == "stage1_judged"
    source_runs = store.list_runs(hold.recovery.ROUND, stage=1, kind="execute")
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": hold.recovery.ROUND,
        "stage": 1,
        "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
        "work_items": [
            {"scored_run_id": run["run_id"],
             "submission_id": run["submission_id"],
             "icp_position": run["icp_position"],
             "output_ref": run["output_ref"]}
            for run in source_runs
        ],
        "zero_rows": [],
    }
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("UPDATE public.lab_arena_rounds "
                           "SET stage1_scoring_plan_doc=%s::jsonb "
                           "WHERE round_id=%s",
                           (json.dumps(plan), hold.recovery.ROUND))
    service = object.__new__(ArenaService)
    service._store = store
    service._round = lambda round_id: store.get_round(round_id)
    service._load_scoring_plan = lambda round_row, stage: plan
    service.evaluation_icps = lambda round_id: daily_icps()[:10]
    service._outputs_by_run = lambda round_id, stage: {
        run["run_id"]: [] for run in source_runs
    }
    service._verified_breakdowns = lambda run, **kwargs: []
    scored = service.score_stage(hold.recovery.ROUND, 1)
    assert scored["status"] == "ok" and scored["judge_executions"] == 10
    assert store.get_round(hold.recovery.ROUND)["status"] == "stage1_scored"
    assert all(run["per_icp_score"] == 0.0 for run in
               store.list_runs(hold.recovery.ROUND, stage=1, kind="execute"))


@pytest.mark.parametrize("matching_event", [True, False])
def test_second_exact_credential_failure_requires_pinned_provider_receipt(
    database, matching_event,
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET "
                           "terminal_cause='credential_error',"
                           "result_doc='{\"terminal_status\":\"credential_error\"}'::jsonb "
                           "WHERE run_id='score369:3:2'")
            cursor.execute("SET session_replication_role=origin")
            if matching_event:
                cursor.execute(
                    "INSERT INTO public.lab_arena_trajectory_events "
                    "(run_id,event_id,round_id,submission_id,miner_hotkey,"
                    "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,"
                    "attempt,run_kind,model_role,event_kind,occurred_at,content) "
                    "SELECT run_id,'00000000-0000-0000-0000-000000000370',round_id,"
                    "submission_id,miner_hotkey,miner_hotkey,assignment_id,'oct01-3',"
                    "stage,icp_position,attempt,'score','baseline','provider.response',"
                    "'2026-10-01T06:23:41Z',"
                    "'{\"operation_id\":\"scrapingdog.scrape\",\"action_sequence\":11,"
                    "\"call\":{\"error_code\":\"miner_credentials_unavailable\","
                    "\"call_identity\":\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
                    "\"provider_status\":403}}'::jsonb "
                    "FROM public.lab_arena_runs WHERE run_id='score369:3:2'"
                )
                cursor.execute(sql)
                cursor.execute("SELECT status FROM public.lab_arena_runs "
                               "WHERE run_id='score369:3:3'")
                assert cursor.fetchone()[0] == "pending"
            else:
                with pytest.raises(psycopg.Error, match="Oct01 release"):
                    cursor.execute(sql)
                cursor.execute("ROLLBACK")
                cursor.execute("SELECT operator_paused FROM "
                               "public.lab_arena_restart_claim_control")
                assert cursor.fetchone()[0] is True


@pytest.mark.parametrize("tamper", ["owner", "active_score", "non_provider", "source"])
def test_release_fails_closed_without_touching_hold(database, tamper):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SET session_replication_role=replica")
            if tamper == "owner":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET actor_ref='other'")
            elif tamper == "active_score":
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE run_id='score369:4:1'")
            elif tamper == "non_provider":
                cursor.execute("UPDATE public.lab_arena_runs "
                               "SET terminal_cause='judge_timeout' "
                               "WHERE run_id='score369:0:1'")
            else:
                cursor.execute("UPDATE public.lab_arena_submissions "
                               "SET source_size_bytes=1 WHERE submission_id=%s",
                               (hold.recovery.BASELINE,))
            cursor.execute("SET session_replication_role=origin")
            with pytest.raises(psycopg.Error, match="Oct01 release"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control")
            assert cursor.fetchone()[0] is True
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE run_id='score369:0:2'")
            assert cursor.fetchone()[0] == 0
