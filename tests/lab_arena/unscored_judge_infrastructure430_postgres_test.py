"""Exhausted judge faults remain unscored without weakening publication proof."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import copy
import json

import pytest

from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge, _enable_verified_proxy_runtime,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/430-lab-arena-unscored-judge-infrastructure.sql"
HELPER = "public.lab_arena__publication_execution_incomplete_v1(text,text)"


@pytest.fixture(scope="module")
def database():
    migrations = CURRENT_PROVIDER_SERVICE_MIGRATIONS + (
        "410-lab-arena-publication-transition-timeout.sql",
        "423-lab-arena-partial-publication.sql",
        "424-lab-arena-partial-baseline-stage-transition.sql",
    )
    yield from database_with_lab_arena_migration(migrations)


def test_migration_is_exact_repeatable_and_keeps_security_shape(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT proowner,proacl,prosecdef,provolatile,proconfig "
            "FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,)
        )
        identity = cur.fetchone()
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute(
            "SELECT proowner,proacl,prosecdef,provolatile,proconfig,"
            "encode(extensions.digest(pg_get_functiondef(oid),'sha256'),'hex') "
            "FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,)
        )
        after = cur.fetchone()
        assert after[:-1] == identity
        assert after[-1] == (
            "1d0b203f8e0b8d2ece0db3c288806e85dc310a69b9786fce27e0ca9c3d308ecf"
        )


def test_exhausted_judge_proof_preserves_null_ranking_and_blocks_false_proof(
    database, tmp_path, monkeypatch,
):
    psycopg, dsn = database
    connect = lambda: psycopg.connect(**dsn)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["JudgeIncompleteMiner"], runners=["alpha"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    assert fixtures._start_round(harness, day=29, epoch=62_429) == 2
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("stage1_scored", runners=1)
    harness.service.advance_round(harness.round_id)
    harness.advance_until("scored", runners=1)
    round_row = harness.service.store.get_round(harness.round_id)
    ids = {
        bool(item["is_king"]): item["submission_id"]
        for item in round_row["participants"]
    }
    publication_patch = {}
    original_transition = harness.service.store.transition_round

    def capture(round_id, expected, target, patch=None):
        assert (round_id, expected, target) == (
            harness.round_id, "scored", "published"
        )
        publication_patch.update(patch or {})
        return {"status": "captured"}

    harness.service.store.transition_round = capture
    try:
        assert harness.service.publish(harness.round_id)["status"] == "captured"
    finally:
        harness.service.store.transition_round = original_transition
    full_publication = publication_patch["publication_doc"]

    def probe(*, baseline=False, cause="judge_error", second=True,
              active=False, alias=False, accepted=False,
              before_deadline=True, expect=False, publish=False):
        submission_id = ids[baseline]
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL session_replication_role=replica")
                cur.execute(
                    "SELECT run_id,stage FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                    "AND status='accepted' ORDER BY icp_position LIMIT 1",
                    (harness.round_id, submission_id),
                )
                execution_id, stage = cur.fetchone()
                cur.execute(
                    "SELECT run_id,assignment_id,judgment_cache_key "
                    "FROM public.lab_arena_runs WHERE scored_run_id=%s "
                    "AND kind='score' AND status='accepted' LIMIT 1",
                    (execution_id,),
                )
                score_id, assignment_id, cache_key = cur.fetchone()
                cur.execute(
                    "UPDATE public.lab_arena_runs SET per_icp_score=NULL, "
                    "qualification_doc=NULL WHERE run_id=%s", (execution_id,)
                )
                cur.execute(
                    "UPDATE public.lab_arena_runs SET status='failed', "
                    "terminal_cause=%s,result_doc=%s::jsonb,output_ref=NULL "
                    "WHERE run_id=%s",
                    (cause, json.dumps({"terminal_status": cause}), score_id),
                )
                if cache_key is not None:
                    # The fixture first accepted this judge. A genuinely
                    # failed judge could not have written its accepted cache.
                    cur.execute(
                        "DELETE FROM public.lab_arena_judgment_cache "
                        "WHERE cache_key=%s", (cache_key,)
                    )
                if second:
                    cur.execute(
                        "INSERT INTO public.lab_arena_runs "
                        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                        "stage,icp_position,attempt,status,terminal_cause,"
                        "result_doc,kind,scored_run_id,stage_generation,"
                        "judgment_cache_key) "
                        "SELECT assignment_id||':2',assignment_id,round_id,"
                        "submission_id,miner_hotkey,stage,icp_position,2,"
                        "'failed',%s,%s::jsonb,'score',scored_run_id,"
                        "stage_generation,judgment_cache_key "
                        "FROM public.lab_arena_runs WHERE run_id=%s",
                        (cause, json.dumps({"terminal_status": cause}), score_id),
                    )
                if active or accepted:
                    cur.execute(
                        "UPDATE public.lab_arena_runs SET status=%s,"
                        "terminal_cause=%s WHERE run_id=%s",
                        ("pending" if active else "accepted",
                         None if active else "accepted",
                         assignment_id + (":2" if second else ":1")),
                    )
                if alias:
                    assert cache_key is not None
                    cur.execute(
                        "SELECT run_id FROM public.lab_arena_runs "
                        "WHERE round_id=%s AND kind='score' AND status='accepted' "
                        "AND scored_run_id<>%s AND stage=%s LIMIT 1",
                        (harness.round_id, execution_id, stage),
                    )
                    alias_id = cur.fetchone()[0]
                    cur.execute(
                        "UPDATE public.lab_arena_runs SET status='pending',"
                        "terminal_cause=NULL,judgment_cache_key=%s "
                        "WHERE run_id=%s", (cache_key, alias_id),
                    )
                close_key = (
                    "stage_1_scoring_close" if stage == 1
                    else "final_scoring_close"
                )
                schedule = dict(round_row["configuration_doc"]["schedule"])
                schedule[close_key] = (
                    datetime.now(timezone.utc) + timedelta(
                        minutes=1 if before_deadline else -1
                    )
                ).isoformat()
                configuration = dict(round_row["configuration_doc"])
                configuration["schedule"] = schedule
                cur.execute(
                    "UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb "
                    "WHERE round_id=%s",
                    (json.dumps(configuration), harness.round_id),
                )
                cur.execute("SET LOCAL session_replication_role=origin")
                cur.execute(
                    "SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)",
                    (harness.round_id, submission_id),
                )
                assert cur.fetchone()[0] is expect, (
                    baseline, cause, second, active, alias, accepted,
                    before_deadline,
                )
                if publish:
                    publication = copy.deepcopy(full_publication)
                    ranking = next(
                        item for item in publication["final_ranking"]
                        if item["submission_id"] == submission_id
                    )
                    ranking.update(
                        final_score=None, cost_summary=None, eligible=False,
                        eligibility_reason="execution_incomplete",
                    )
                    if baseline:
                        publication["king_decision"]["outcome"] = "no_king"
                    cur.execute(
                        "SELECT public.lab_arena_transition_round(%s,'scored',"
                        "'published',%s::jsonb)",
                        (harness.round_id, json.dumps({
                            "publication_doc": publication,
                            "published_at": publication["published_at"],
                        })),
                    )
                    assert cur.fetchone()[0]["status"] == "ok"
        finally:
            conn.rollback()
            conn.close()

    probe(expect=True, publish=True)
    probe(baseline=True, cause="judge_timeout", expect=True, publish=True)
    probe(second=False, expect=False)
    probe(second=False, before_deadline=False, expect=True)
    probe(active=True, expect=False)
    probe(alias=True, expect=False)
    # An older failure cannot displace a later accepted attempt.
    probe(accepted=True, expect=False)
    probe(cause="credential_error", expect=False)

    # Now let the real service score and publish an unfinished stage-two ICP.
    # Only the failed judge state is seeded; ranking and publication are built
    # by the production methods and checked by the database transition guard.
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute(
            "SELECT execution.run_id,judge.run_id,judge.assignment_id,"
            "judge.judgment_cache_key FROM public.lab_arena_runs AS execution "
            "JOIN public.lab_arena_runs AS judge "
            "ON judge.scored_run_id=execution.run_id "
            "WHERE execution.round_id=%s AND execution.submission_id=%s "
            "AND execution.kind='execute' AND execution.stage=2 "
            "AND execution.status='accepted' AND judge.kind='score' "
            "AND judge.status='accepted' ORDER BY execution.icp_position LIMIT 1",
            (harness.round_id, ids[False]),
        )
        execution_id, score_id, assignment_id, cache_key = cur.fetchone()
        cur.execute(
            "UPDATE public.lab_arena_runs SET per_icp_score=NULL,"
            "qualification_doc=NULL WHERE run_id=%s", (execution_id,)
        )
        cur.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='judge_error',"
            "result_doc='{\"terminal_status\":\"judge_error\"}'::jsonb,"
            "output_ref=NULL WHERE run_id=%s", (score_id,)
        )
        if cache_key is not None:
            cur.execute(
                "DELETE FROM public.lab_arena_judgment_cache WHERE cache_key=%s",
                (cache_key,),
            )
        cur.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
            "stage,icp_position,attempt,status,terminal_cause,result_doc,"
            "kind,scored_run_id,stage_generation,judgment_cache_key) "
            "SELECT assignment_id||':2',assignment_id,round_id,submission_id,"
            "miner_hotkey,stage,icp_position,2,'failed','judge_timeout',"
            "'{\"terminal_status\":\"judge_timeout\"}'::jsonb,'score',"
            "scored_run_id,stage_generation,judgment_cache_key "
            "FROM public.lab_arena_runs WHERE run_id=%s", (score_id,)
        )
        cur.execute(
            "UPDATE public.lab_arena_rounds SET status='stage2_judged' "
            "WHERE round_id=%s", (harness.round_id,)
        )
        cur.execute("SET LOCAL session_replication_role=origin")
    scored = harness.service.score_stage(harness.round_id, 2)
    assert scored["status"] == "ok"
    assert harness.service.store.get_run(execution_id)["per_icp_score"] is None
    assert harness.service.publish(harness.round_id)["status"] == "ok"
    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    rows = {
        row["submission_id"]: row
        for row in published["publication_doc"]["final_ranking"]
    }
    assert rows[ids[False]]["final_score"] is None
    assert rows[ids[False]]["eligible"] is False
    assert rows[ids[False]]["eligibility_reason"] == "execution_incomplete"
    assert rows[ids[True]]["final_score"] is not None
    assert published["publication_doc"]["king_decision"]["outcome"] == "no_king"
