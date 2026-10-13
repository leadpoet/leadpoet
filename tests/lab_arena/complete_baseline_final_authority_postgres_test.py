"""Exercise migration 469 and its independent final-authority guard verbatim."""
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import CURRENT_REWARD_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_reward_migration_postgres import BASELINE, MINER_A
from tests.lab_arena.test_lab_arena_reward_migration_postgres import _basis
from lab_arena import rewards, signing
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from lab_arena.promotion import GitPromoter
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge, _enable_verified_proxy_runtime
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.icp_fixtures import daily_icps

MIGRATION = Path(__file__).resolve().parents[2] / "scripts/469-lab-arena-complete-baseline-final-authority.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_REWARD_SERVICE_MIGRATIONS + (MIGRATION.name,))


@pytest.fixture()
def normal_database():
    # Each normal flow owns its baseline funding history. A winner from a
    # previous parameter must not replace this fixture's host-funded baseline.
    # The current service also calls the bounded recovery RPC during stage 2.
    # Use the complete deployed schema for this normal transition proof.
    from tests.lab_arena.bounded_baseline_recovery470_postgres_test import database as current_database
    yield from current_database.__wrapped__()


@pytest.fixture()
def connection(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            # Controlled fixtures bypass unrelated input/immutable-row guards;
            # the new final authority trigger is exercised separately below.
            for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_submissions", "lab_arena_ledger"):
                cursor.execute("ALTER TABLE public.%s DISABLE TRIGGER USER" % table)
            config = {"mode": "live", "network_name": "test", "netuid": 71,
                      "rewards_enabled": True, "baseline_hotkey": BASELINE,
                      "reward_constants": rewards.reward_constants_document(),
                      "stage_1_icp_count": 1, "stage_2_icp_count": 1,
                      "execution_sequence_policy": "baseline_scored_first_v1",
                      "sourcing_cost_eligibility_policy": "successful_calls_per_icp_v1"}
            cursor.execute("INSERT INTO lab_arena_rounds(round_id,status,configuration_doc,participants) VALUES ('arena-2026-10-13-finalgate','scored',%s::jsonb,%s::jsonb)",
                           (json.dumps(config), json.dumps([{"submission_id": "final-baseline", "miner_hotkey": BASELINE, "is_king": True}])))
            cursor.execute("INSERT INTO lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES ('final-baseline','arena-2026-10-13-finalgate',%s,'frozen',true)", (BASELINE,))
            for position in range(2):
                for kind in ("execute", "score"):
                    cursor.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,scored_run_id,status,terminal_cause,per_icp_score,result_doc,output_ref) VALUES (%s,%s,'arena-2026-10-13-finalgate','final-baseline',%s,1,%s,1,%s,%s,'accepted','accepted',0,%s::jsonb,'accepted-output')",
                                   ("%s-%s" % (kind, position), "%s-assignment-%s" % (kind, position), BASELINE, position, kind,
                                    "execute-%s" % position if kind == "score" else None,
                                    json.dumps({"terminal_status": "accepted", "schema_version": "cached" if kind == "score" else "execution"})))
        yield connection
        connection.rollback()


def ready(cursor):
    cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1('arena-2026-10-13-finalgate')")
    return cursor.fetchone()[0]


def test_measured_zero_cache_alias_and_failed_previous_attempt_pass(connection):
    with connection.cursor() as c:
        assert ready(c)
        c.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,per_icp_score,result_doc,output_ref) SELECT 'failed-first','old-assignment',round_id,submission_id,miner_hotkey,stage,icp_position,2,kind,'failed','provider_error',0,result_doc,output_ref FROM lab_arena_runs WHERE run_id='execute-0'")
        assert ready(c)


@pytest.mark.parametrize("change", [
    "UPDATE lab_arena_runs SET status='failed',terminal_cause='provider_error' WHERE run_id='execute-0'",
    "UPDATE lab_arena_runs SET status='failed',terminal_cause='judge_error' WHERE run_id='score-0'",
    "UPDATE lab_arena_runs SET status='pending',terminal_cause=NULL WHERE run_id='score-0'",
    "UPDATE lab_arena_runs SET per_icp_score=NULL WHERE run_id='execute-0'",
    "UPDATE lab_arena_runs SET output_ref=NULL WHERE run_id='execute-0'",
    "UPDATE lab_arena_runs SET stage=2 WHERE run_id='score-0'",
    "UPDATE lab_arena_runs SET scored_run_id='execute-1' WHERE run_id='score-0'",
    "UPDATE lab_arena_runs SET status='failed',terminal_cause='stage_closed',per_icp_score=NULL WHERE icp_position=1",
])
def test_missing_or_unbound_baseline_cannot_create_any_final_authority(connection, change):
    with connection.cursor() as c:
        c.execute(change)
        assert not ready(c)
        c.execute("ALTER TABLE lab_arena_rounds ENABLE TRIGGER lab_arena_complete_baseline_final_guard")
        for assignment in ("status='published'", "promotion_doc='{}'::jsonb", "baseline_promoted_at=now()", "reward_activated_at=now()"):
            c.execute("SAVEPOINT attempted_final")
            with pytest.raises(Exception, match="lab_arena_final_baseline_incomplete"):
                c.execute("UPDATE lab_arena_rounds SET %s WHERE round_id='arena-2026-10-13-finalgate'" % assignment)
            c.execute("ROLLBACK TO SAVEPOINT attempted_final")


@pytest.mark.parametrize("entry_kind,succeeded,valid", [("uncertain", False, True), ("uncertain", True, False),
    ("dispatch", False, False), ("settlement", None, False), ("settlement", True, True)])
def test_existing_successful_cost_policy_preserves_known_failed_fees(connection, entry_kind, succeeded, valid):
    with connection.cursor() as c:
        c.execute("INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,amount_microusd,entry_doc,terminal_response) VALUES (%s,%s,'arena-2026-10-13-finalgate','final-baseline','execute-0',1,%s,'openrouter','test',123,%s::jsonb,%s::jsonb)",
                  (entry_kind, BASELINE, "sha256:" + "1" * 64, json.dumps({"call": {"call_succeeded": succeeded}}), json.dumps({"call_succeeded": succeeded})))
        assert ready(c) is valid



def test_same_day_ineligible_legacy_publication_does_not_block_new_activation(connection):
    signer = signing.LocalSigner.generate()
    key = signing.signing_key_document(signer.public_key_der)
    moment = "2026-10-13T00:00:00Z"
    rid = "arena-2026-10-13-finalgate"
    basis = _basis(signer, rid, moment, 123, "no_king", "")
    with connection.cursor() as c:
        # A stored legacy partial result is preserved, but creates no pending
        # authority barrier against the later complete same-day result.
        c.execute("INSERT INTO lab_arena_rounds(round_id,status,configuration_doc,rewards_enabled,evaluation_date,publication_doc,created_at) SELECT 'arena-2026-10-13-incomplete','published',configuration_doc,true,'2026-10-13',%s::jsonb,'2026-10-13T00:00:00Z' FROM lab_arena_rounds WHERE round_id=%s",
                  (json.dumps({"king_decision": {"outcome": "crowned"}}), rid))
        c.execute("UPDATE lab_arena_rounds SET status='published',rewards_enabled=true,evaluation_date='2026-10-13',published_at=%s,publication_doc=%s::jsonb,created_at='2026-10-13T00:01:00Z' WHERE round_id=%s",
                  (moment, json.dumps({"king_decision": {"outcome": "no_king"}}), rid))
        c.execute("SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id='arena-2026-10-13-incomplete'")
        legacy = c.fetchone()[0]
        c.execute("ALTER TABLE lab_arena_rounds ENABLE TRIGGER lab_arena_complete_baseline_final_guard")
        c.execute("ALTER TABLE lab_arena_rounds ENABLE TRIGGER lab_arena_reward_requires_promotion")
        c.execute("SET LOCAL ROLE lab_arena_service")
        c.execute("SELECT lab_arena_activate_reward(%s,%s::jsonb,%s::jsonb)", (rid, json.dumps(basis), json.dumps(key)))
        assert c.fetchone()[0]["status"] == "activated"
        c.execute("RESET ROLE")
        c.execute("SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id='arena-2026-10-13-incomplete'")
        assert c.fetchone()[0] == legacy


def test_direct_promotion_prepare_cannot_store_incomplete_baseline_intent(connection):
    with connection.cursor() as c:
        c.execute("UPDATE lab_arena_runs SET status='failed',terminal_cause='provider_error' WHERE run_id='execute-0'")
        c.execute("INSERT INTO lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES ('final-miner','arena-2026-10-13-finalgate',%s,'frozen',false)", (MINER_A,))
        publication = {"king_decision": {"outcome": "crowned", "winner_submission_id": "final-miner", "king_submission_id": "final-miner", "king_hotkey": MINER_A}}
        c.execute("UPDATE lab_arena_rounds SET status='published',publication_doc=%s::jsonb WHERE round_id='arena-2026-10-13-finalgate'", (json.dumps(publication),))
        c.execute("ALTER TABLE lab_arena_rounds ENABLE TRIGGER lab_arena_complete_baseline_final_guard")
        c.execute("SAVEPOINT attempted_prepare")
        plan = {"commit": "1" * 40, "main_before": "2" * 40, "lab_before": "3" * 40, "timestamp": "2026-10-13T00:00:00Z"}
        with pytest.raises(Exception, match="lab_arena_final_baseline_incomplete"):
            c.execute("SELECT lab_arena_prepare_promotion('arena-2026-10-13-finalgate',%s::jsonb)", (json.dumps(plan),))
        c.execute("ROLLBACK TO SAVEPOINT attempted_prepare")
        c.execute("SELECT promotion_doc FROM lab_arena_rounds WHERE round_id='arena-2026-10-13-finalgate'")
        assert c.fetchone()[0] is None


def test_service_deadline_failure_preserves_real_accepted_rows_and_failed_fee_ledger(connection):
    rid = "arena-2026-10-13-finalgate"
    with connection.cursor() as c:
        c.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule}','{\"publication_deadline\":\"2026-10-13T20:30:03Z\"}'::jsonb) WHERE round_id=%s", (rid,))
        c.execute("UPDATE lab_arena_runs SET status='failed',terminal_cause='provider_error' WHERE run_id='execute-0'")
        c.execute("UPDATE lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE run_id='score-0'")
        c.execute("INSERT INTO lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES ('preserved-miner',%s,%s,'frozen',false)", (rid, MINER_A))
        c.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,per_icp_score,result_doc,output_ref) VALUES ('preserved-output','preserved-assignment',%s,'preserved-miner',%s,2,0,1,'execute','accepted','accepted',75,'{\"terminal_status\":\"accepted\"}','immutable-output-ref')", (rid, MINER_A))
        c.execute("INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,amount_microusd,entry_doc) VALUES ('uncertain',%s,%s,'preserved-miner','preserved-output',2,%s,'openrouter','test',123,'{\"call\":{\"call_succeeded\":false}}')", (MINER_A, rid, "sha256:" + "4" * 64))
        c.execute("ALTER TABLE lab_arena_rounds ENABLE TRIGGER USER")
        c.execute("SELECT to_jsonb(r) FROM lab_arena_runs r WHERE status='accepted' ORDER BY run_id")
        accepted = c.fetchall()
        c.execute("SELECT to_jsonb(l) FROM lab_arena_ledger l ORDER BY entry_id")
        ledger = c.fetchall()
    # Use the same current store/RPC and service cancellation path. This test
    # connection sees its controlled fixtures without committing user data.
    class SharedConnectionTransport(PsycopgTransport):
        def _acquire(self): return connection
        def _release(self, _connection): pass
    obj = object.__new__(ArenaService)
    obj._store = ArenaStore(SharedConnectionTransport(lambda: connection))
    obj._round = lambda _rid: obj._store.get_round(_rid)
    now = [datetime(2026, 10, 13, 20, 30, 2, tzinfo=timezone.utc)]
    obj.now = lambda: now[0]
    assert obj.publish(rid)["reason"] == "baseline_incomplete"
    assert obj._store.get_round(rid)["status"] == "scored"
    now[0] = datetime(2026, 10, 13, 20, 30, 3, tzinfo=timezone.utc)
    assert obj.publish(rid)["status"] == "cancelled"
    row = obj._store.get_round(rid)
    assert row["cancel_reason"] == "scoring_incomplete"
    assert row["publication_doc"] is None and row["reward_basis_doc"] is None
    assert row["baseline_promoted_at"] is None and row["reward_activated_at"] is None
    with connection.cursor() as c:
        c.execute("SELECT to_jsonb(r) FROM lab_arena_runs r WHERE status='accepted' ORDER BY run_id")
        assert c.fetchall() == accepted
        c.execute("SELECT to_jsonb(l) FROM lab_arena_ledger l ORDER BY entry_id")
        assert c.fetchall() == ledger


def test_duplicate_and_archived_judgment_cannot_witness_baseline(connection):
    with connection.cursor() as c:
        c.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,per_icp_score,result_doc,output_ref) SELECT 'duplicate','duplicate-assignment',round_id,submission_id,miner_hotkey,stage,icp_position,2,kind,status,terminal_cause,per_icp_score,result_doc,output_ref FROM lab_arena_runs WHERE run_id='execute-0'")
        assert not ready(c)

        c.execute("UPDATE lab_arena_runs SET status='failed',terminal_cause='provider_error' WHERE run_id='duplicate'")
        assert ready(c)
        c.execute("INSERT INTO lab_arena_rounds(round_id,status,configuration_doc) SELECT 'arena-2026-10-12-archived','published',configuration_doc FROM lab_arena_rounds WHERE round_id='arena-2026-10-13-finalgate'")
        c.execute("UPDATE lab_arena_runs SET round_id='arena-2026-10-12-archived' WHERE run_id='score-0'")
        assert not ready(c)


@pytest.mark.parametrize("crowned,rewards_enabled,day", [(False, True, 13), (True, True, 14), (False, False, 15)])
def test_normal_complete_baseline_publishes_and_activates_with_all_guards(normal_database, tmp_path, monkeypatch, crowned, rewards_enabled, day):
    psycopg2, dsn = normal_database
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                               challengers=["FinalMiner"] if rewards_enabled else [], runners=["alpha"])
    harness.service.config.defaults = replace(harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True, rewards_enabled=rewards_enabled)
    if not rewards_enabled:
        harness.service.config.defaults = replace(harness.service.config.defaults, benchmark_icp_count=10)
        harness.service.config.daily_icp_source = lambda **kwargs: {
            "status": "ready", "set_id": int(kwargs["set_id"]), "icps": daily_icps()[:10],
        }
    _enable_verified_proxy_runtime(harness)
    def judge(companies, icp, reference):
        rows = _baseline_first_judge(companies, icp, reference)
        if crowned:
            # A measured zero baseline remains a valid comparison. Only failed
            # execution/scoring zeros are excluded by the new authority gate.
            for row in rows:
                if row["final_score"] == 80:
                    row.update(final_score=0.0, company_qualified=False,
                               verifier_gate_receipts=[{"gate": "company_fit", "decision": "mismatch"}])
                else:
                    row["final_score"] = 80.0
        return rows
    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    assert fixtures._start_round(harness, day=day, epoch=62400 + day) == (2 if rewards_enabled else 1)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("published", runners=1)
    row = harness.service.store.get_round(harness.round_id)
    baseline = next(r for r in row["publication_doc"]["final_ranking"] if r["is_baseline"])
    assert baseline["final_score"] is not None
    assert row["publication_doc"]["king_decision"]["outcome"] == ("crowned" if crowned else "no_king")
    if not rewards_enabled:
        assert harness.service.activate_reward(harness.round_id)["status"] == "disabled"
        assert harness.service.store.get_round(harness.round_id)["reward_basis_doc"] is None
        return
    if crowned:
        root = tmp_path / "local-promotion"
        root.mkdir()
        remote = fixtures.promotion_repository(root)
        harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-cache")
        assert harness.service.promote_baseline(harness.round_id)["status"] == "promoted"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    assert harness.service.store.get_round(harness.round_id)["reward_basis_doc"]["king_outcome"] == ("crowned" if crowned else "no_king")


def test_migration_replay_preserves_rows_and_private_helper_acl(connection):
    with connection.cursor() as c:
        c.execute("UPDATE lab_arena_rounds SET status='published',reward_activated_at=now(),reward_basis_doc='{\"old\":true}'::jsonb WHERE round_id='arena-2026-10-13-finalgate'")
        c.execute("SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id='arena-2026-10-13-finalgate'")
        before = c.fetchone()[0]
        c.execute(MIGRATION.read_text())
        c.execute(MIGRATION.read_text())
        c.execute("SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id='arena-2026-10-13-finalgate'")
        assert c.fetchone()[0] == before
        c.execute("SELECT has_function_privilege('anon','public.lab_arena__baseline_final_ready_v1(text)','execute'),has_function_privilege('lab_arena_service','public.lab_arena__baseline_final_ready_v1(text)','execute')")
        assert c.fetchone() == (False, False)
