"""Exact quality-ref authority migration, including malformed and frozen controls."""
from copy import deepcopy
import json

import pytest

from tests.lab_arena import company_only_lifecycle_test as lifecycle
from tests.lab_arena.lab_arena_pg_harness import SCRIPTS

MIGRATION = SCRIPTS / "589-lab-arena-company-quality-judgment-authority.sql"
# Deliberately pinned: the failure control and migration preimage must not move
# when the current application fixture gains later migrations.
PRE_589_TAIL = (
    "443-lab-arena-competition-publication-projection.sql",
    "575-lab-arena-score-host-continuation.sql",
    "576-lab-arena-champion-score-funding.sql",
    "577-lab-arena-judge-continuation-runner-handoff.sql",
    "578-lab-arena-closed-provider-ordered-candidate.sql",
    "579-lab-arena-close-covered-credit-candidates.sql",
    "581-lab-arena-judge-generation-drain.sql",
    "583-lab-arena-promotion-rpc-privileges.sql",
    "584-lab-arena-stage-quota-run-lookup.sql",
    "585-lab-arena-champion-baseline-reasoning.sql",
    "585-lab-arena-review-deadline-recovery.sql",
    "586-lab-arena-publication-restart-drain.sql",
    "587-lab-arena-atomic-review-restart-guard.sql",
    "588-lab-arena-qualifier-training-publication.sql",
)


@pytest.fixture
def before_database():
    from scripts.verify_arena_full_thinking_e2e import full_thinking_database
    database = full_thinking_database(production_order=True)
    try:
        pg, dsn = next(database)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            for migration in PRE_589_TAIL:
                cur.execute((SCRIPTS / migration).read_text())
            cur.execute("TRUNCATE public.lab_arena_rounds CASCADE")
        yield pg, dsn
    finally:
        database.close()


@pytest.fixture(scope="module")
def migrated_database():
    from tests.lab_arena.lab_arena_pg_harness import current_service_database
    database = current_service_database()
    try:
        pg, dsn = next(database)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute(MIGRATION.read_text())
        yield pg, dsn
    finally:
        database.close()


def canonical():
    configuration = {"integrity_policy": "arena_integrity_v1", "company_quality_policy": "company_quality_v1",
        "scorer_image_digest": "sha256:" + "a" * 64, "scorer_image_reference": "registry/scorer@sha256:" + "a" * 64,
        "scorer_policy": {"company_quality_policy": "company_quality_v1", "scoring_adapter_version": "qualification_integrity_v2"}}
    row = {"round_id": "arena-2026-10-29-quality", "arena_network_name": "finney", "arena_netuid": 71,
        "evaluation_date": "2026-10-29", "configuration_doc": configuration}
    ref = {"schema_version": "leadpoet.lab_arena.company_judgment_ref.v1", "company_index": 0,
        "cache_key": "sha256:" + "b" * 64, "company_input_hash": "sha256:" + "c" * 64}
    ref["scope_doc"] = {"round_id": row["round_id"], "network_name": row["arena_network_name"],
        "netuid": row["arena_netuid"], "evaluation_date": row["evaluation_date"],
        **{key: configuration[key] for key in ("integrity_policy", "company_quality_policy", "scorer_image_digest", "scorer_image_reference")},
        "cache_key": ref["cache_key"], "company_input_hash": ref["company_input_hash"]}
    return row, [ref]


def valid(cur, row, refs):
    cur.execute("SELECT public.lab_arena__company_quality_refs_v1(jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb),%s::jsonb)",
        (json.dumps(row), json.dumps(refs)))
    return cur.fetchone()[0]


@pytest.mark.parametrize("adapter", ["qualification_integrity_v2", "qualification_contacts_v3", "qualification_integrity_v5", "qualification_contacts_v6"])
@pytest.mark.parametrize("empty", [False, True])
def test_ordered_quality_refs_preserve_existing_adapter_variants(migrated_database, adapter, empty):
    pg, dsn = migrated_database
    row, refs = canonical()
    # The new validator binds the same policy/scope as217. Adapter validity is
    # retained in the existing configuration guards, not narrowed here.
    row["configuration_doc"]["scorer_policy"]["scoring_adapter_version"] = adapter
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert valid(cur, row, [] if empty else refs) is True


@pytest.mark.parametrize("path,value", [
    (("configuration_doc", "company_quality_policy"), None),
    (("configuration_doc", "company_quality_policy"), "future_quality"),
    (("configuration_doc", "scorer_policy", "company_quality_policy"), "future_quality"),
    (("configuration_doc", "scorer_policy", "company_quality_policy"), None),
    (("configuration_doc", "company_judgment_policy"), "company_judgment_v1"),
    (("configuration_doc", "scorer_policy", "company_judgment_policy"), "company_judgment_v1"),
    (("configuration_doc", "integrity_policy"), None),
    (("configuration_doc", "scorer_policy"), []),
    (("arena_netuid",), None),
    (("evaluation_date",), None),
])
def test_invalid_quality_policies_never_acquire_ref_authority(migrated_database, path, value):
    pg, dsn = migrated_database
    row, refs = canonical()
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert valid(cur, row, refs) is False


@pytest.mark.parametrize("field,value", [
    ("schema_version", None), ("company_index", "0"), ("company_index", 1),
    ("cache_key", None), ("cache_key", "sha256:invalid"),
    ("company_input_hash", 7), ("scope_doc", None), ("scope_doc", []),
])
def test_malformed_ordered_refs_fail_closed(migrated_database, field, value):
    pg, dsn = migrated_database
    row, refs = canonical()
    refs[0][field] = value
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert valid(cur, row, refs) is False


@pytest.mark.parametrize("field,value", [
    ("cache_key", "sha256:" + "d" * 64), ("company_input_hash", "sha256:" + "d" * 64),
    ("round_id", "other-round"), ("network_name", "test"), ("netuid", "71"),
    ("evaluation_date", "2026-10-28"), ("integrity_policy", None),
    ("company_quality_policy", "future_quality"), ("scorer_image_digest", "sha256:" + "d" * 64),
    ("scorer_image_reference", None), ("company_judgment_policy", "company_judgment_v1"),
])
def test_refs_remain_bound_to_frozen_scope(migrated_database, field, value):
    pg, dsn = migrated_database
    row, refs = canonical()
    refs[0]["scope_doc"][field] = value
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert valid(cur, row, refs) is False


@pytest.mark.parametrize("refs", [None, {}, "refs", [None], [0], [canonical()[1][0]] * 6])
def test_invalid_ref_containers_fail_without_cast_errors(migrated_database, refs):
    pg, dsn = migrated_database
    row, _ = canonical()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert valid(cur, row, refs) is False


def test_quality_lifecycle_fails_on_exact_previous_schema_and_passes_after_migration(before_database, tmp_path, monkeypatch):
    from lab_arena.service import ArenaService
    from lab_arena.store import ArenaStoreError
    pg, dsn = before_database
    original = ArenaService.publish
    repaired = []

    def publish(instance, round_id):
        if repaired:
            return original(instance, round_id)
        before = deepcopy(instance.store.list_runs(round_id))
        with pytest.raises(ArenaStoreError, match="lab_arena_(late_judge_score_invalid|baseline_incomplete)"):
            original(instance, round_id)
        assert instance.store.list_runs(round_id) == before
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_get_functiondef('public.lab_arena__all_judgments_ready_v1(text)'::regprocedure)")
            old_ready = cur.fetchone()[0]
            cur.execute("SELECT pg_get_functiondef('public.lab_arena__participant_incomplete_reason_v1(text,text)'::regprocedure)")
            old_incomplete = cur.fetchone()[0]
            cur.execute("SELECT pg_get_functiondef('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure)")
            old_continuation = cur.fetchone()[0]
            cur.execute(MIGRATION.read_text())
        result = original(instance, round_id)
        assert result["status"] == "ok"
        repaired.append(round_id)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (round_id,))
            assert cur.fetchone()[0] is True
            # Negative persisted-state controls run only in rolled-back local
            # savepoints. Bypass write guards to test the independent reader;
            # every trigger and accepted row is restored by the rollback.
            mutations = [
                "judgment_cache_key='sha256:" + "d" * 64 + "'",
                "judgment_input_hash='sha256:" + "d" * 64 + "'",
                "judgment_scope_doc='{}'::jsonb",
                "scoring_generation_floor=scoring_generation_floor+1",
                "miner_hotkey='unbound-miner'",
                "company_judgment_refs=jsonb_set(company_judgment_refs,'{0,company_index}','\"0\"'::jsonb)",
            ]
            for mutation in mutations:
                cur.execute("SAVEPOINT invalid_retained_binding")
                cur.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
                cur.execute("UPDATE public.lab_arena_runs SET " + mutation + " WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE kind='score' AND status='accepted')")
                cur.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (round_id,))
                assert cur.fetchone()[0] is False, mutation
                cur.execute("ROLLBACK TO SAVEPOINT invalid_retained_binding")
            for path in ("{company_quality_policy}", "{scorer_policy,company_quality_policy}"):
                cur.execute("SAVEPOINT invalid_quality_policy")
                cur.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,%s::text[],'\"future_quality\"'::jsonb) WHERE round_id=%s", (path, round_id))
                cur.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (round_id,))
                assert cur.fetchone()[0] is False
                cur.execute("ROLLBACK TO SAVEPOINT invalid_quality_policy")
            sources = instance.store.list_runs(round_id, kind="execute")
            judges = instance.store.list_runs(round_id, kind="score")
            source = next(run for run in sources if run["status"] == "accepted")
            judge = next(run for run in judges if run["scored_run_id"] == source["run_id"])
            assert judge["company_judgment_refs"]
            empty_refs = {**judge, "company_judgment_refs": []}
            assert not instance._judgment_matches_execution(instance.store.get_round(round_id), empty_refs, source)
            miner = next(run["submission_id"] for run in sources if run["submission_id"].startswith("sub-"))
            missing = next(run for run in sources if run["submission_id"] == miner and run["icp_position"] == 9)
            retained_peer = next(run for run in judges if run["submission_id"] == miner and run["icp_position"] == 0)
            cur.execute("SAVEPOINT incomplete_quality_participant")
            cur.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
            cur.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cur.execute("UPDATE public.lab_arena_rounds SET status='scored',publication_doc=NULL,configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((clock_timestamp()-interval '1 minute')::text)) WHERE round_id=%s", (round_id,))
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',output_ref=NULL,per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s OR scored_run_id=%s", (missing["run_id"], missing["run_id"]))
            cur.execute("SELECT public.lab_arena__participant_incomplete_reason_v1(%s,%s)", (round_id, miner))
            assert cur.fetchone()[0] == "stage_closed"
            cur.execute("SAVEPOINT previous_incomplete_authority")
            cur.execute(old_incomplete)
            cur.execute("SELECT public.lab_arena__participant_incomplete_reason_v1(%s,%s)", (round_id, miner))
            assert cur.fetchone()[0] is None
            cur.execute("ROLLBACK TO SAVEPOINT previous_incomplete_authority")
            cur.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc='{}'::jsonb WHERE run_id=%s", (retained_peer["run_id"],))
            # The malformed retained accepted peer cannot grant omission.
            cur.execute("SELECT public.lab_arena__participant_incomplete_reason_v1(%s,%s)", (round_id, miner))
            assert cur.fetchone()[0] is None
            cur.execute("ROLLBACK TO SAVEPOINT incomplete_quality_participant")
            cur.execute("SAVEPOINT quality_retry_authority")
            cur.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
            cur.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring' WHERE round_id=%s", (round_id,))
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',output_ref=NULL WHERE run_id=%s", (retained_peer["run_id"],))

            def continuation():
                cur.execute("SELECT public.lab_arena__judge_continuation_v1(r,j) FROM public.lab_arena_rounds r JOIN public.lab_arena_runs j USING(round_id) WHERE j.run_id=%s", (retained_peer["run_id"],))
                return cur.fetchone()[0]

            assert continuation() is True
            cur.execute("SAVEPOINT previous_retry_authority")
            cur.execute(old_continuation)
            assert continuation() is False
            cur.execute("ROLLBACK TO SAVEPOINT previous_retry_authority")
            for mutation in ("miner_hotkey='unbound-miner'", "scoring_generation_floor=scoring_generation_floor+1", "judgment_scope_doc='{}'::jsonb"):
                cur.execute("SAVEPOINT invalid_retry_binding")
                cur.execute("UPDATE public.lab_arena_runs SET " + mutation + " WHERE run_id=%s", (retained_peer["run_id"],))
                assert continuation() is False, mutation
                cur.execute("ROLLBACK TO SAVEPOINT invalid_retry_binding")
            cur.execute("SAVEPOINT invalid_retry_source")
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE run_id=%s", (retained_peer["scored_run_id"],))
            assert continuation() is False
            cur.execute("ROLLBACK TO SAVEPOINT invalid_retry_source")
            # Real retry insertion must run every normal write/generation guard.
            cur.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            cur.execute("SAVEPOINT nonclear_retry_billing")
            cur.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,0,%s::jsonb FROM public.lab_arena_ledger WHERE run_id=%s AND call_identity IS NOT NULL ORDER BY entry_id DESC LIMIT 1", (json.dumps({"call": {"call_succeeded": True}}), retained_peer["run_id"]))
            assert cur.rowcount == 1
            cur.execute("SELECT public.lab_arena__judge_calls_clear_v1(%s)", (retained_peer["assignment_id"],))
            assert cur.fetchone()[0] is False
            cur.execute("SELECT public.lab_arena_expire_leases(%s)", (round_id,))
            assert cur.fetchone()[0]["retried"] == 0
            cur.execute("ROLLBACK TO SAVEPOINT nonclear_retry_billing")
            cur.execute("SELECT public.lab_arena_expire_leases(%s)", (round_id,))
            assert cur.fetchone()[0]["retried"] == 1
            cur.execute("SELECT attempt,status,scored_run_id,company_judgment_refs,clock_timestamp()<created_at+interval '30 seconds' FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=2", (retained_peer["assignment_id"],))
            assert cur.fetchone() == (2, "pending", retained_peer["scored_run_id"], retained_peer["company_judgment_refs"], True)
            cur.execute("ROLLBACK TO SAVEPOINT quality_retry_authority")
            # Same accepted data, exact prior finality function: no authority.
            cur.execute("SAVEPOINT previous_finality")
            cur.execute(old_ready)
            cur.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (round_id,))
            assert cur.fetchone()[0] is False
            cur.execute("ROLLBACK TO SAVEPOINT previous_finality")
            def evidence():
                values = []
                for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger"):
                    cur.execute(f"SELECT md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text)::text,'')) FROM public.{table} t")
                    values.append(cur.fetchone()[0])
                cur.execute("SELECT pg_get_functiondef('public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)'::regprocedure)")
                values.append(cur.fetchone()[0])
                return values
            before_replay = evidence()
            cur.execute(MIGRATION.read_text())
            assert evidence() == before_replay
        return result

    monkeypatch.setattr(ArenaService, "publish", publish)
    # Continue the same real round after its exact migration. Preserve every
    # full lifecycle/public verifier/promotion/reward assertion in that test.
    lifecycle.test_company_only_ten_icp_baseline_and_miner_complete_full_lifecycle(
        before_database, tmp_path, True, False)
    assert repaired == ["arena-2026-09-24-companyonly"]


@pytest.mark.parametrize("drift", ["source", "acl", "helper_source", "helper_acl", "continuation_source", "continuation_acl"])
def test_source_and_security_drift_roll_back_the_whole_migration(before_database, drift):
    pg, dsn = before_database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if drift.startswith("helper"):
            cur.execute(MIGRATION.read_text())
            signature = "public.lab_arena__company_quality_refs_v1(public.lab_arena_rounds,jsonb)"
        elif drift.startswith("continuation"):
            signature = "public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)"
        else:
            signature = "public.lab_arena__all_judgments_ready_v1(text)"
        cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
        definition = cur.fetchone()[0]
        if drift.endswith("acl"):
            cur.execute("GRANT EXECUTE ON FUNCTION " + signature + " TO anon")
        else:
            cur.execute(definition.replace("BEGIN", "BEGIN -- deliberate source drift589", 1) if drift.startswith(("helper", "continuation"))
                else definition.replace("WITH validation_round548", "WITH /* deliberate source drift589 */ validation_round548", 1))
        conn.commit()
        def snapshot():
            cur.execute("SELECT jsonb_agg(to_jsonb(p) ORDER BY p.oid) FROM pg_proc p WHERE p.proname IN ('lab_arena__company_quality_refs_v1','lab_arena__all_judgments_ready_v1','lab_arena_record_run_scores','lab_arena__participant_incomplete_reason_v1','lab_arena__judge_continuation_v1')")
            return cur.fetchone()[0]
        before = snapshot()
        conn.commit()
        with pytest.raises(pg.Error, match="Arena quality judgment (preimage|security|helper) differs"):
            cur.execute(MIGRATION.read_text())
        conn.rollback()
        assert snapshot() == before
