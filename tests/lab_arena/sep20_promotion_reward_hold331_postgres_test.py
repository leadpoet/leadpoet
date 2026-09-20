"""Focused PostgreSQL proof for the Sep20 rerun328 authority hold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.store import hash_lease_token, new_lease_token
from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import sep20_terminal_fixed_source_baseline_rerun330_postgres_test as rerun330
from tests.lab_arena.sep20_terminal_fixed_source_baseline_rerun330_postgres_test import (
    database,
)


MIGRATION = (
    Path(__file__).parents[2]
    / "scripts/331-arena-2026-09-20-promotion-reward-hold.sql"
)
ROUND = rerun330.ROUND
BASELINE = rerun330.BASELINE


def _body() -> str:
    _, remainder = MIGRATION.read_text().split("BEGIN;\n", 1)
    body, _ = remainder.rsplit("COMMIT;\n", 1)
    return body


def _round_snapshot(cursor):
    cursor.execute(
        "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
        (ROUND,),
    )
    return cursor.fetchone()[0]


def _make_one_stage2_score_pending(connection):
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET per_icp_score=0 WHERE round_id=%s "
            "AND submission_id=%s AND kind='execute' AND status='accepted' "
            "AND icp_position IN(1,8)",
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 2
        cursor.execute(
            "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='score' AND stage=2 AND status='accepted' "
            "ORDER BY run_id LIMIT 1",
            (ROUND,),
        )
        run_id = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='pending',runner_hotkey=NULL,"
            "lease_token_hash=NULL,lease_expires_at=NULL,claim_request_id=NULL,"
            "claim_request_hash=NULL,claim_response=NULL,result_doc=NULL,"
            "output_ref=NULL,terminal_cause=NULL,terminal_doc=NULL "
            ",stage_generation=(SELECT stage_generation FROM "
            "public.lab_arena_rounds WHERE round_id=%s),"
            "participation_accepted_at=NULL "
            "WHERE run_id=%s",
            (ROUND, run_id),
        )
        assert cursor.rowcount == 1
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage2_scoring',"
            "publication_doc=NULL,published_at=NULL,king_outcome=NULL,king_hotkey=NULL,"
            "king_start_epoch=NULL,effective_reward_epoch=NULL,reward_basis_hash=NULL,"
            "reward_basis_doc=NULL,signing_key_doc=NULL,reward_activated_at=NULL,"
            "promotion_required=TRUE,promotion_doc=NULL,baseline_promoted_at=NULL,"
            "rewards_enabled=TRUE,configuration_doc=jsonb_set(jsonb_set("
            "configuration_doc,'{rewards_enabled}','true'::jsonb,FALSE),"
            "'{scoring_cap_microusd}','50000000'::jsonb,TRUE) WHERE round_id=%s",
            (ROUND,),
        )
        assert cursor.rowcount == 1
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    return run_id


def _claim_score(store, runner_hotkey):
    token = new_lease_token()
    request_id = contracts.new_request_id()
    run = store.claim_assignment(
        round_id=ROUND,
        runner_hotkey=runner_hotkey,
        declared_parallelism=1,
        slot_ceiling=20,
        excluded_miner_hotkeys=[runner_hotkey],
        request_id=request_id,
        request_hash=contracts.document_hash({"request_id": request_id}),
        lease_token_hash=hash_lease_token(token),
        lease_ttl_seconds=6300,
    )
    assert run["status"] == "leased" and run["kind"] == "score"
    return run, token


def test_hold_preserves_active_scoring_lease_cost_and_completion(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        harness = rerun330.IsolatedHarness(
            lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
        )
        harness.round_id = ROUND
        rerun330._publish_rerun328(connection, harness, monkeypatch)
        pending_run_id = _make_one_stage2_score_pending(connection)

        # Fail closed before any change when authority is already prepared or
        # either proven-zero baseline position differs.
        with connection.cursor() as cursor:
            for mutation in (
                "UPDATE public.lab_arena_rounds SET promotion_doc='{}'::jsonb "
                "WHERE round_id='arena-2026-09-20'",
                "UPDATE public.lab_arena_runs SET per_icp_score=1 WHERE round_id="
                "'arena-2026-09-20' AND submission_id='baseline-2026-09-20' "
                "AND kind='execute' AND status='accepted' AND icp_position=1",
            ):
                clean = _round_snapshot(cursor)
                cursor.execute("SAVEPOINT invalid_hold")
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(mutation)
                with pytest.raises(psycopg2.Error):
                    cursor.execute(_body())
                cursor.execute("ROLLBACK TO SAVEPOINT invalid_hold")
                assert _round_snapshot(cursor) == clean
        connection.rollback()

        store = harness.service.store
        run, token = _claim_score(store, harness.runner_keys[0])
        assert run["run_id"] == pending_run_id
        token_hash = hash_lease_token(token)
        identity, reserved = lifecycle._reserve_cost_call(
            store, run, token, "rerun328-hold-active-score", 1234
        )
        assert reserved["status"] == "reserved"
        before_run = store.get_run(run["run_id"])
        before_configuration = dict(store.get_round(ROUND)["configuration_doc"])

        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
        connection.commit()

        held = store.get_round(ROUND)
        hold = held["configuration_doc"]["sep20_rerun328_promotion_reward_hold"]
        assert held["rewards_enabled"] is False
        assert held["promotion_required"] is False
        assert held["configuration_doc"]["rewards_enabled"] is False
        assert hold["original_configuration_rewards_enabled"] is True
        assert hold["original_row_rewards_enabled"] is True
        assert hold["original_promotion_required"] is True
        assert {
            key: value
            for key, value in held["configuration_doc"].items()
            if key not in {
                "rewards_enabled",
                "sep20_rerun328_promotion_reward_hold",
            }
        } == {
            key: value
            for key, value in before_configuration.items()
            if key != "rewards_enabled"
        }
        assert store.get_run(run["run_id"]) == before_run

        lifecycle._settle_cost_call(
            store, run, token, identity, 1234, succeeded=True
        )
        scored_run = store.get_run(run["scored_run_id"])
        executed = json.loads(harness.objects.get(scored_run["output_ref"]).decode())
        document = scoring.build_scoring_output(
            run["scored_run_id"],
            [lifecycle._proof_breakdown(executed["companies"][0], 0)],
        )
        output_ref = "arena/score/%s-hold.json" % run["run_id"]
        harness.objects.put(output_ref, json.dumps(document).encode())
        stored = store.get_run(run["run_id"])
        evidence = judgment_cache.build_evidence_snapshot(
            output=document,
            cache_scope=stored["judgment_scope_doc"],
            source_score_run_id=run["run_id"],
            source_scored_run_id=run["scored_run_id"],
            source_output_ref=output_ref,
            source_runner_hotkey=harness.runner_keys[0],
            runner_authority_exclusions=run["runner_authority_exclusions"],
        )
        completed = store.complete_attempt(
            run_id=run["run_id"],
            lease_token_hash=token_hash,
            result={"terminal_status": "accepted"},
            terminal_cause="accepted",
            output_ref=output_ref,
            judgment_evidence=evidence,
            judgment_evidence_hash=contracts.document_hash(evidence),
        )
        assert completed["status"] == "accepted"
        assert store.get_run(run["run_id"])["status"] == "accepted"
    finally:
        connection.close()


def test_hold_is_narrow_and_numbered_after_existing_329():
    body = MIGRATION.read_text()
    assert MIGRATION.name.startswith("331-")
    assert "329-lab-arena-explicit-90m-lease.sql" != MIGRATION.name
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "promotion_required=FALSE" in body
    assert "rewards_enabled=FALSE" in body
    assert "stage2_scoring','stage2_judged','scored','published" in body
    assert "baseline positions 1 and 8" in body
    assert "reward_activated_at IS NOT NULL" in body
    assert "promotion_doc IS NOT NULL" in body
