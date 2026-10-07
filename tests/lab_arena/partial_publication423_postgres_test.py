"""Publication permits only proof-backed unfinished Arena assignments."""

from pathlib import Path
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import copy
import json

import pytest

from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge,
    _enable_verified_proxy_runtime,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/423-lab-arena-partial-publication.sql"
HELPER = "public.lab_arena__publication_execution_incomplete_v1(text,text)"
OUTER = "public.lab_arena_integrity_publication_guard_v1()"
BASELINE = "public.lab_arena_publication_baseline_guard_v1()"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_partial_publication_migration_preserves_guard_identity(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT oid::regprocedure::text, pg_get_userbyid(proowner),
                      prosecdef, provolatile, proconfig, proacl
               FROM pg_proc WHERE oid IN (%s::regprocedure, %s::regprocedure)
               ORDER BY oid::regprocedure::text""",
            (OUTER, BASELINE),
        )
        before = cursor.fetchall()
        cursor.execute(MIGRATION.read_text())
        cursor.execute(
            """SELECT oid::regprocedure::text, pg_get_userbyid(proowner),
                      prosecdef, provolatile, proconfig, proacl
               FROM pg_proc WHERE oid IN (%s::regprocedure, %s::regprocedure)
               ORDER BY oid::regprocedure::text""",
            (OUTER, BASELINE),
        )
        assert cursor.fetchall() == before
        cursor.execute(
            """SELECT pg_get_userbyid(proowner), prosecdef, provolatile, proacl
               FROM pg_proc WHERE oid=%s::regprocedure""",
            (HELPER,),
        )
        owner, security, volatility, acl = cursor.fetchone()
        assert (owner, security, volatility) == ("lab_arena_owner", True, "s")
        assert acl is not None
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (OUTER,))
        definition = cursor.fetchone()[0]
        assert definition.count(
            "public.lab_arena__publication_execution_incomplete_v1("
        ) == 3
        # Migration 289's per-ICP guard exits this loop before v_main. The
        # incomplete branch must execute first, while complete rows still use
        # that existing strict helper.
        assert (
            definition.index("A proven unfinished assignment")
            < definition.index("PERFORM public.lab_arena__per_icp_publication_valid")
            < definition.index("v_main := public.lab_arena__integrity_submission_summary")
        )


def test_partial_publication_requires_closed_proof_and_cannot_promote_unknown_baseline(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["PartialMiner"], runners=["alpha"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    assert fixtures._start_round(harness, day=27, epoch=62_427) == 2
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("stage1_scored", runners=1)
    harness.service.advance_round(harness.round_id)
    harness.advance_until("scored", runners=1)
    round_row = harness.service.store.get_round(harness.round_id)
    ids = {
        bool(item["is_king"]): item["submission_id"]
        for item in round_row["participants"]
    }
    assert set(ids) == {False, True}

    captured = {}
    transition = harness.service.store.transition_round
    def capture(round_id, expected, target, patch=None):
        assert (round_id, expected, target) == (harness.round_id, "scored", "published")
        captured.update(patch or {})
        return {"status": "captured"}
    harness.service.store.transition_round = capture
    try:
        assert harness.service.publish(harness.round_id)["status"] == "captured"
    finally:
        harness.service.store.transition_round = transition
    full = captured["publication_doc"]
    assert full["king_decision"]["outcome"] == "no_king"

    def attempt(*, incomplete_baseline, marker=True, null_score=True,
                incomplete_reason=True, promote=False, unfinished_judge=False,
                before_deadline=False, missing_deadline=False):
        submission_id = ids[incomplete_baseline]
        connection = connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
                cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
                cursor.execute(
                    """SELECT run_id,stage FROM public.lab_arena_runs
                       WHERE round_id=%s AND submission_id=%s
                         AND kind='execute' AND status='accepted'
                       ORDER BY icp_position LIMIT 1""",
                    (harness.round_id, submission_id),
                )
                execution_id, execution_stage = cursor.fetchone()
                if unfinished_judge:
                    cursor.execute(
                        """UPDATE public.lab_arena_runs
                           SET per_icp_score=NULL, qualification_doc=NULL
                           WHERE run_id=%s""",
                        (execution_id,),
                    )
                else:
                    cursor.execute(
                        """UPDATE public.lab_arena_runs
                           SET status='failed', terminal_cause='stage_closed',
                               terminal_doc=%s::jsonb, per_icp_score=NULL,
                               qualification_doc=NULL
                           WHERE run_id=%s""",
                        (json.dumps({"infrastructure_incomplete": marker}), execution_id),
                    )
                cursor.execute(
                    """UPDATE public.lab_arena_runs
                       SET status='failed', terminal_cause='stage_closed'
                       WHERE scored_run_id=%s AND kind='score'""",
                    (execution_id,),
                )
                schedule = dict(round_row["configuration_doc"]["schedule"])
                deadline_key = (
                    "stage_1_scoring_close" if execution_stage == 1 else "final_scoring_close"
                ) if unfinished_judge else f"stage_{execution_stage}_close"
                if missing_deadline:
                    schedule.pop(deadline_key)
                else:
                    schedule[deadline_key] = (
                        datetime.now(timezone.utc) + (
                            timedelta(minutes=1) if before_deadline
                            else -timedelta(minutes=1)
                        )
                    ).isoformat()
                configuration = dict(round_row["configuration_doc"])
                configuration["schedule"] = schedule
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s",
                    (json.dumps(configuration), harness.round_id),
                )
                cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
                cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
                publication = copy.deepcopy(full)
                ranking = next(
                    item for item in publication["final_ranking"]
                    if item["submission_id"] == submission_id
                )
                if null_score:
                    ranking.update(final_score=None, cost_summary=None,
                                   eligible=False, eligibility_reason=(
                                       "execution_incomplete" if incomplete_reason else "eligible"
                                   ))
                if promote:
                    publication["king_decision"] = {
                        "outcome": "crowned", "winner_submission_id": ids[False],
                        "king_submission_id": ids[False],
                        "king_hotkey": next(
                            item["miner_hotkey"] for item in round_row["participants"]
                            if item["submission_id"] == ids[False]
                        ),
                    }
                cursor.execute(
                    "SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)",
                    (harness.round_id, submission_id),
                )
                proof = cursor.fetchone()[0]
                if before_deadline or missing_deadline or not marker:
                    assert proof is False
                cursor.execute(
                    "SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",
                    (harness.round_id, json.dumps({
                        "publication_doc": publication,
                        "published_at": publication["published_at"],
                    })),
                )
                return proof
        finally:
            connection.rollback()
            connection.close()

    assert attempt(incomplete_baseline=False) is True
    assert attempt(incomplete_baseline=True) is True
    assert attempt(incomplete_baseline=False, unfinished_judge=True) is True
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=True, promote=True)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, marker=False)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, before_deadline=True)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, missing_deadline=True)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, unfinished_judge=True, before_deadline=True)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, unfinished_judge=True, missing_deadline=True)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, null_score=False)
    with pytest.raises(psycopg2.Error):
        attempt(incomplete_baseline=False, incomplete_reason=False)
