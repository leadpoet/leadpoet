"""October 1 provider hold uses the existing global claim gate only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena import oct01_cancelled_baseline_recovery368_postgres_test as recovery


ROOT = Path(__file__).parents[2]
HOLD = ROOT / "scripts/369-arena-2026-10-01-provider-claim-hold.sql"
database = recovery.database


def _seed_active(cursor):
    recovery._seed(cursor, foreign=True)
    cursor.execute(recovery._render_sql(cursor))
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
        "status_generation=6,stage_generation=5,"
        "configuration_doc=jsonb_set(configuration_doc,'{integrity_policy}',"
        "'\"arena_integrity_v1\"'::jsonb) WHERE round_id=%s",
        (recovery.ROUND,),
    )
    cursor.execute(
        "UPDATE public.lab_arena_runs SET status='accepted',"
        "terminal_cause='accepted',"
        "output_ref='arena/arena-2026-10-01/outputs/'||run_id||'.json',"
        "result_doc='{\"terminal_status\":\"accepted\"}'::jsonb "
        "WHERE round_id=%s AND kind='execute'",
        (recovery.ROUND,),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
        "stage,icp_position,attempt,kind,status,stage_generation,scored_run_id,"
        "terminal_cause,output_ref,result_doc) "
        "SELECT 'score369:'||e.icp_position||':1',"
        "'score369:'||e.icp_position,e.round_id,e.submission_id,"
        "e.miner_hotkey,1,e.icp_position,1,'score',"
        "CASE e.icp_position WHEN 0 THEN 'failed' WHEN 1 THEN 'accepted' "
        "WHEN 2 THEN 'leased' ELSE 'pending' END,5,e.run_id,"
        "CASE e.icp_position WHEN 0 THEN 'credential_error' "
        "WHEN 1 THEN 'accepted' ELSE NULL END,"
        "CASE e.icp_position WHEN 1 THEN 'arena/test/score.json' ELSE NULL END,"
        "CASE e.icp_position WHEN 0 THEN "
        "'{\"terminal_status\":\"credential_error\"}'::jsonb "
        "WHEN 1 THEN '{\"terminal_status\":\"accepted\"}'::jsonb "
        "ELSE NULL END "
        "FROM public.lab_arena_runs e WHERE e.round_id=%s AND e.kind='execute'",
        (recovery.ROUND,),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_trajectory_events "
        "(run_id,event_id,round_id,submission_id,miner_hotkey,"
        "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,"
        "attempt,run_kind,model_role,event_kind,occurred_at,content) "
        "SELECT run_id,'00000000-0000-0000-0000-000000000369',round_id,"
        "submission_id,miner_hotkey,miner_hotkey,assignment_id,'oct01-0',"
        "stage,icp_position,attempt,'score','baseline','provider.response',"
        "'2026-10-01T06:00:26Z',"
        "'{\"operation_id\":\"scrapingdog.scrape\",\"action_sequence\":56,"
        "\"call\":{\"error_code\":\"miner_credentials_unavailable\","
        "\"call_identity\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"provider_status\":403}}'::jsonb "
        "FROM public.lab_arena_runs WHERE run_id='score369:0:1'"
    )
    cursor.execute("SET session_replication_role=origin")


def _hash(cursor, expression):
    cursor.execute("SELECT encode(extensions.digest((" + expression + ")::text,'sha256'),'hex')")
    return cursor.fetchone()[0]


def _render_hold(cursor):
    sql = HOLD.read_text()
    hashes = {
        "e7cd2d1d2fea2a1f2ae41b313889545d36ca3e9b09e5d3d4021fec36a99b0a7b":
            _hash(cursor, "(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        "872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19":
            _hash(cursor, "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        "8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61":
            _hash(cursor, "(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)"),
        "1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3":
            _hash(cursor, "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        "24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa":
            _hash(cursor, "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND kind='execute')"),
    }
    for original, fixture in hashes.items():
        assert original in sql
        sql = sql.replace(original, fixture)
    return sql


def test_hold_blocks_new_claims_preserves_active_completion_and_replays(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _seed_active(cursor)
            sql = _render_hold(cursor)
            cursor.execute(sql)
            cursor.execute(
                "SELECT operator_paused,pause_reason,actor_ref,updated_at "
                "FROM public.lab_arena_restart_claim_control WHERE singleton"
            )
            held = cursor.fetchone()
            assert held[:3] == (True, "oct01_deepline_outage", "oct01-provider-recovery369")
            with pytest.raises(psycopg.Error, match="lab_arena_claims_paused"):
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE run_id='score369:3:1'")
            cursor.execute("ROLLBACK")
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/score2.json' "
                           "WHERE run_id='score369:2:1'")
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == held
            cursor.execute("SELECT status FROM public.lab_arena_runs "
                           "WHERE run_id='score369:3:1'")
            assert cursor.fetchone()[0] == "pending"


@pytest.mark.parametrize("conflict", ["operator", "guard", "foreign_live", "foreign_lease", "source"])
def test_hold_fails_closed_for_foreign_owner_guard_or_frozen_tamper(database, conflict):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _seed_active(cursor)
            sql = _render_hold(cursor)
            cursor.execute("SET session_replication_role=replica")
            if conflict == "operator":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET operator_paused=true,pause_reason='other',actor_ref='other'")
            elif conflict == "guard":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET guard_commitment='sha256:'||repeat('a',64),"
                               "owner_commitment='sha256:'||repeat('b',64),"
                               "guard_generation=1,guard_expires_at=now()+interval '1 hour',"
                               "candidate_commit=repeat('c',40),restart_scope='all',"
                               "restart_phase='draining'")
            elif conflict == "foreign_live":
                cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1' "
                               "WHERE round_id='arena-2026-10-02'")
            elif conflict == "foreign_lease":
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE run_id='foreign368:1'")
            else:
                cursor.execute("UPDATE public.lab_arena_submissions SET source_size_bytes=1 "
                               "WHERE submission_id=%s", (recovery.BASELINE,))
            cursor.execute("SET session_replication_role=origin")
            with pytest.raises(psycopg.Error, match="Oct01 hold"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control")
            assert cursor.fetchone()[0] == (conflict == "operator")
