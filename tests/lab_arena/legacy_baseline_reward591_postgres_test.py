"""Legacy training exemption is comparison authority, never a training pass."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.slot_decay_postgres_test import MIGRATIONS, POLICY
from tests.lab_arena.reward_slots_postgres_test import BASELINE, MINER, _insert, _publication
from tests.lab_arena.reward_shares_oct22_test import _activation_service
from tests.lab_arena.champion_baseline_reasoning_test import baseline_round
from lab_arena.service import ArenaService
from types import SimpleNamespace

MIGRATION = Path(__file__).parents[2] / "scripts/591-lab-arena-legacy-baseline-reward-comparison.sql"
PREIMAGE = "78bf61278a796e0c7b06c025148bee5e0ad9bc8d249b27b2528ae337684bba33"
POSTIMAGE = "670235bc84b77ebc11ca69be3729a1bdb430bb5119a32021db862e90f453f499"
REPORT = {"eligible": False, "reasons": ["baseline_reasoning_not_required"],
          "exemption": "champion_baseline"}
CONFIG = {"scorer_policy": {"scoring_adapter_version": "qualification_integrity_v2"},
          "cost_per_company_microusd": 6_000_000,
          "baseline_reasoning_configuration": {},
          "reasoning_capture_policy": "full_generated_thinking_v1",
          "integrity_policy": "arena_integrity_v1",
          "sourcing_cost_eligibility_policy": "successful_calls_flat_per_icp_v1"}
CHAINS = itertools.count(36100)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture()
def state(database):
    pg, dsn = database
    control = pg.connect(**dsn)
    control.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        yield store, control, next(CHAINS)
    finally:
        store.close()
        control.close()


def _definition(cursor):
    cursor.execute("SELECT pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
    return cursor.fetchone()[0]


def _insert_legacy(control, chain, name, *, day="2026-10-08", activated=False,
                   eligible=False, config=None, mutate=None):
    rid = f"arena-{day}-{chain}{name}"
    pub = _publication(rid, delta=11.72, baseline_score=8.55)
    baseline = pub["final_ranking"][1]
    baseline.update(submission_id="baseline-" + rid.removeprefix("arena-"),
                    eligible=eligible,
                    eligibility_reason="eligible" if eligible else "training_data_incomplete",
                    training_data_eligibility=deepcopy(REPORT))
    pub["participants"][1]["submission_id"] = baseline["submission_id"]
    if mutate:
        mutate(pub)
    _insert(control, chain, name, day=day, activated=activated, publication=pub)
    # Historical slot fixture inserts already-published rows. Only its frozen
    # policy fields are installed here; production write-once guards are intact.
    with control.cursor() as cursor:
        cursor.execute("BEGIN")
        try:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                           "configuration_doc || %s::jsonb WHERE round_id=%s",
                           (json.dumps(CONFIG if config is None else config), rid))
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    return rid


def _service(store, rid, chain):
    service, _, _ = _activation_service("2026-10-08")
    service._store = store
    service._round = lambda _rid: store.get_round(_rid)
    service._chain_scope = lambda: ("finney", chain)
    # This focused consumer proof starts after established complete baseline
    # and promotion readiness; the guarded publication test below proves costs.
    return service


def test_n_minus_one_rejects_actual_report_then_exact_fix_activates(database, state):
    store, control, chain = state
    row, baseline = baseline_round({})
    assessor = object.__new__(ArenaService)
    assessor._store = SimpleNamespace(get_submission=lambda _sid: baseline)
    assert assessor._submission_training_eligibility(row, baseline["submission_id"], []) == REPORT
    rid = _insert_legacy(control, chain, "current")
    service = _service(store, rid, chain)
    with control.cursor() as cursor:
        # Restore the exact predecessor after other module tests apply591.
        predecessor = _definition(cursor)
        if hashlib.sha256(predecessor.encode()).hexdigest() == POSTIMAGE:
            sql = MIGRATION.read_text()
            import re
            for seam in ("candidate", "baseline"):
                old = re.search(r"v_old_" + seam + r" TEXT := \$old\$(.*?)\$old\$;", sql, re.S)[1]
                new = re.search(r"v_new_" + seam + r" TEXT := \$new\$(.*?)\$new\$;", sql, re.S)[1]
                predecessor = predecessor.replace(new, old)
            cursor.execute(predecessor)
        assert hashlib.sha256(_definition(cursor).encode()).hexdigest() == PREIMAGE
    with pytest.raises(ArenaStoreError, match="lab_arena_reward_current_achievement_invalid"):
        service.activate_reward(rid)
    assert store.get_round(rid)["reward_basis_doc"] is None
    before = store.get_round(rid)
    with control.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        assert hashlib.sha256(_definition(cursor).encode()).hexdigest() == POSTIMAGE
    assert store.get_round(rid) == before
    assert service.activate_reward(rid)["status"] == "activated"
    after = store.get_round(rid)
    assert after["publication_doc"] == before["publication_doc"]
    assert after["configuration_doc"] == before["configuration_doc"]
    assert after["publication_doc"]["final_ranking"][1]["eligible"] is False
    assert after["publication_doc"]["final_ranking"][1]["training_data_eligibility"] == REPORT
    assert after["reward_basis_doc"]["reward_slots"][0]["baseline_score"] == 8.55
    assert after["reward_basis_doc"]["champion_reward_factor_ppm"] == 1000000
    assert service.activate_reward(rid)["status"] == "existing"


@pytest.mark.parametrize("policy", ["successful_calls_per_icp_v1", "successful_calls_flat_per_icp_v1"])
def test_both_existing_cost_policies_and_old_eligible_path(state, policy):
    store, control, chain = state
    with control.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    old = _insert(control, chain, "old", day="2026-10-07", delta=6)
    frozen = {**CONFIG, "sourcing_cost_eligibility_policy": policy}
    rid = _insert_legacy(control, chain, "current", config=frozen)
    slots = store.reward_slot_snapshot(rid, POLICY)
    assert [slot["round_id"] for slot in slots] == [rid] * 3
    assert store.get_round(old)["reward_basis_doc"] == {"king_outcome": "no_king"}
    # A normal true baseline has exactly the original path and no new config
    # requirement. Old valid historical achievements also keep their slots.
    ordinary = _insert(control, chain, "ordinary", activated=False, delta=0.5)
    history_slots = store.reward_slot_snapshot(ordinary, POLICY)
    assert history_slots[0] is None
    assert [slot["round_id"] for slot in history_slots[1:]] == [old, old]


@pytest.mark.parametrize("field,value", [
    ("baseline_reasoning_configuration", None),
    ("baseline_reasoning_configuration", []),
    ("baseline_reasoning_configuration", {"reasoning_capture_policy": "full_generated_thinking_v1"}),
    ("baseline_reasoning_configuration", {"reasoning_capture_policy": "wrong"}),
    ("reasoning_capture_policy", None), ("reasoning_capture_policy", "wrong"),
    ("integrity_policy", None), ("integrity_policy", "wrong"),
    ("sourcing_cost_eligibility_policy", None), ("sourcing_cost_eligibility_policy", "wrong"),
])
def test_only_explicit_empty_frozen_legacy_policy_is_exempt(state, field, value):
    store, control, chain = state
    with control.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    config = {**CONFIG, field: value}
    rid = _insert_legacy(control, chain, "invalid", config=config)
    with pytest.raises(ArenaStoreError, match="current_achievement_invalid"):
        store.reward_slot_snapshot(rid, POLICY)


def test_absent_legacy_contract_cannot_enter_exemption(state):
    store, control, chain = state
    config = deepcopy(CONFIG)
    del config["baseline_reasoning_configuration"]
    rid = _insert_legacy(control, chain, "missing", config=config)
    with pytest.raises(ArenaStoreError, match="current_achievement_invalid"):
        store.reward_slot_snapshot(rid, POLICY)


def _mutate(pub, kind):
    baseline = pub["final_ranking"][1]
    if kind == "noncanonical":
        baseline["submission_id"] = pub["participants"][1]["submission_id"] = "other-baseline"
    elif kind == "wrong_hotkey": pub["participants"][1]["miner_hotkey"] = MINER
    elif kind == "not_baseline": baseline["is_baseline"] = False
    elif kind == "duplicate_baseline": pub["final_ranking"].append(deepcopy(baseline))
    elif kind == "duplicate_participant": pub["participants"].append(deepcopy(pub["participants"][1]))
    elif kind == "wrong_round": pub["round_id"] = "arena-other"
    elif kind == "ineligible_winner": pub["final_ranking"][0]["eligible"] = False
    elif kind == "missing_score": del baseline["final_score"]
    elif kind == "cost_failure": baseline["eligibility_reason"] = "provider_cost_uncertain"
    elif kind == "not_assessed": baseline["training_data_eligibility"] = {"status":"not_assessed", "eligible":None, "reasons":[]}
    elif kind == "strict_capture_failure": baseline["training_data_eligibility"] = {"eligible":False, "reasons":["selected_run_export_missing"]}
    elif kind == "missing_report": del baseline["training_data_eligibility"]
    elif kind == "missing_exemption": del baseline["training_data_eligibility"]["exemption"]
    elif kind == "extra_report": baseline["training_data_eligibility"]["extra"] = "untrusted"
    elif kind == "extra_reason": baseline["training_data_eligibility"]["reasons"].append("capture_failed")
    elif kind == "null_eligible": baseline["eligible"] = None
    elif kind == "string_eligible": baseline["eligible"] = "false"
    elif kind == "missing_eligible": del baseline["eligible"]
    else: raise AssertionError(kind)


@pytest.mark.parametrize("kind", [
    "noncanonical", "wrong_hotkey", "not_baseline", "duplicate_baseline",
    "duplicate_participant", "wrong_round", "ineligible_winner", "missing_score",
    "cost_failure", "not_assessed", "strict_capture_failure", "missing_report",
    "missing_exemption", "extra_report", "extra_reason", "null_eligible",
    "string_eligible", "missing_eligible",
])
def test_legacy_exception_rejects_invalid_current_and_historical_rows(state, kind):
    store, control, chain = state
    with control.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    _insert_legacy(control, chain, "old", day="2026-10-07", activated=True,
                   mutate=lambda pub: _mutate(pub, kind))
    target = _insert(control, chain, "normal", activated=False, delta=0.5)
    assert store.reward_slot_snapshot(target, POLICY) == [None] * 3
    invalid = _insert_legacy(control, chain + 10000, "bad", mutate=lambda pub: _mutate(pub, kind))
    with pytest.raises(ArenaStoreError, match="current_achievement_invalid"):
        store.reward_slot_snapshot(invalid, POLICY)


def test_exact_replay_security_and_drift_fail_closed(state):
    _, control, chain = state
    _insert_legacy(control, chain, "immutable")
    with control.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc "
                       "WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
        identity = cursor.fetchone()
        cursor.execute("SELECT pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure)")
        activation = cursor.fetchone()[0]
        cursor.execute("CREATE ROLE legacy591_migrator NOLOGIN NOSUPERUSER INHERIT")
        cursor.execute("GRANT lab_arena_owner TO legacy591_migrator")
        cursor.execute("GRANT USAGE,CREATE ON SCHEMA public TO legacy591_migrator")
        cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
        acl = cursor.fetchone()[0]
        cursor.execute("SET ROLE legacy591_migrator")
        cursor.execute(MIGRATION.read_text())
        cursor.execute("RESET ROLE")
        cursor.execute("SELECT proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc "
                       "WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
        assert cursor.fetchone() == identity
        cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
        assert cursor.fetchone()[0] == acl
        cursor.execute("SELECT pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure)")
        assert cursor.fetchone()[0] == activation
        for drift in ("definition", "search_path", "volatility", "acl"):
            cursor.execute("BEGIN")
            try:
                if drift == "definition": cursor.execute(_definition(cursor).replace("DECLARE", "-- drift\nDECLARE", 1))
                elif drift == "search_path": cursor.execute("ALTER FUNCTION public.lab_arena_reward_slot_snapshot(text,jsonb) SET search_path=pg_catalog")
                elif drift == "volatility": cursor.execute("ALTER FUNCTION public.lab_arena_reward_slot_snapshot(text,jsonb) VOLATILE")
                elif drift == "acl": cursor.execute("GRANT EXECUTE ON FUNCTION public.lab_arena_reward_slot_snapshot(text,jsonb) TO anon")
                with pytest.raises(Exception, match="591 reward slot (definition|security) differs"):
                    cursor.execute(MIGRATION.read_text())
            finally:
                cursor.execute("ROLLBACK")


def _measured_publication(cursor, rid, config, participants):
    """Use actual ledger and integrity helpers for every report scalar."""
    pub = _publication(rid, delta=20, baseline_score=0)
    pub.update(stage1_ranking=[], finalists=[])
    pub["participants"] = [{"submission_id": p["submission_id"],
                            "miner_hotkey": p["miner_hotkey"],
                            "is_baseline": p["is_king"]} for p in participants]
    ranking = []
    for participant in participants:
        sid = participant["submission_id"]
        per_icp = []
        for position in range(2):
            cursor.execute("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,0)",
                           (rid, sid, position))
            per_icp.append({**cursor.fetchone()[0], "qualified_company_count": 0,
                            "returned_company_count": 0})
        cursor.execute("SELECT public.lab_arena_submission_costs(%s)", (sid,))
        costs = cursor.fetchone()[0]
        hard_reason = next((r["eligibility_reason"] for r in per_icp
                            if r["eligibility_reason"] in
                            ("provider_calls_inflight", "provider_cost_uncertain")), None)
        report = deepcopy(REPORT) if participant["is_king"] else {
            "eligible": True, "reasons": [], "source_review": {"eligible": True},
            "capture": {"eligible": True}}
        if hard_reason is None and not report["eligible"]:
            hard_reason = "training_data_incomplete"
        summary = {
            "sourcing_cost_eligibility_policy": config["sourcing_cost_eligibility_policy"],
            "execution_icp_cap_microusd": config["execution_icp_cap_microusd"],
            "cost_per_company_cap_microusd": config["cost_per_company_microusd"],
            "competition_sourcing_microusd": sum(r["competition_sourcing_microusd"] for r in per_icp),
            "qualified_company_count": 0, "returned_company_count": 0,
            "eligible_icp_count": sum(r["eligible"] for r in per_icp),
            "per_icp": per_icp,
            "execution": ArenaService._cost_kind_summary(costs, "execute", include_successful=True),
            "judge": ArenaService._cost_kind_summary(costs, "score", include_successful=True),
        }
        score = 0 if participant["is_king"] else 20
        ranking.append({"submission_id": sid, "is_baseline": participant["is_king"],
                        "final_score": score, "eligible": hard_reason is None,
                        "eligibility_reason": hard_reason or "eligible",
                        "training_data_eligibility": report, "cost_summary": summary})
    pub["final_ranking"] = ranking
    return pub


def _transition_publication(cursor, rid, publication):
    cursor.execute("SAVEPOINT guarded_publication591")
    try:
        cursor.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",
                       (rid, json.dumps({"publication_doc": publication,
                                         "published_at": publication["published_at"]})))
        result = cursor.fetchone()[0]
        if result["status"] == "retry":
            assert result["reason"] == "evaluation_draining"
            cursor.execute("RELEASE SAVEPOINT guarded_publication591")
            return False
        assert result["status"] == "ok"
        cursor.execute("RELEASE SAVEPOINT guarded_publication591")
        return True
    except Exception as exc:
        cursor.execute("ROLLBACK TO SAVEPOINT guarded_publication591")
        cursor.execute("RELEASE SAVEPOINT guarded_publication591")
        assert exc.pgcode in ("22023", "55000"), str(exc)
        assert ("lab_arena_publication_cost_report_mismatch" in str(exc)
                or "lab_arena_final_baseline_incomplete" in str(exc)), str(exc)
        return False


def test_original_publication_guard_proves_cost_reason_cannot_be_forged():
    from tests.lab_arena.lab_arena_pg_harness import current_service_database
    from tests.lab_arena.training_publication569_postgres_test import _verify

    database = current_service_database()
    try:
        pg, dsn = next(database)
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute("BEGIN")
            rid = "arena-2026-10-08-guardedlegacy591"
            row, _ = baseline_round({})
            config = {**row["configuration_doc"], **CONFIG, "round_id": rid,
                      "stage_1_icp_count": 1, "stage_2_icp_count": 1,
                      "benchmark_icp_count": 2, "promotion_margin": 1,
                      "execution_sequence_policy": "baseline_scored_first_v1",
                      "mode": "live", "network_name": "finney", "netuid": 591,
                      "rewards_enabled": True, "baseline_hotkey": BASELINE,
                      "execution_icp_cap_microusd": 6_000_000,
                      "execution_cap_microusd": 12_000_000}
            participants = [{"submission_id": "baseline-" + rid.removeprefix("arena-"),
                             "miner_hotkey": BASELINE, "is_king": True},
                            {"submission_id": rid + "-winner", "miner_hotkey": MINER,
                             "is_king": False}]
            # Seed accepted measured rows only in the disposable fixture. Restore
            # all production triggers before testing publication and activation.
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,"
                           "participants,rewards_enabled,evaluation_date,finalists) VALUES "
                           "(%s,'scored',%s::jsonb,%s::jsonb,true,'2026-10-08','[]'::jsonb)",
                           (rid, json.dumps(config), json.dumps(participants)))
            for p in participants:
                sid, hotkey = p["submission_id"], p["miner_hotkey"]
                cursor.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,"
                               "miner_hotkey,status,is_king) VALUES (%s,%s,%s,'frozen',%s)",
                               (sid, rid, hotkey, p["is_king"]))
                for position in range(2):
                    execute = sid + "-execute-" + str(position)
                    for kind in ("execute", "score"):
                        run = execute if kind == "execute" else execute + "-judge"
                        cursor.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
                                       "submission_id,miner_hotkey,stage,icp_position,attempt,kind,scored_run_id,"
                                       "status,terminal_cause,per_icp_score,qualification_doc,result_doc,output_ref,scoring_generation_floor) "
                                       "VALUES (%s,%s,%s,%s,%s,1,%s,1,%s,%s,'accepted','accepted',%s,"
                                       "'{\"companies\":[]}'::jsonb,'{\"terminal_status\":\"accepted\"}'::jsonb,'measured-output',0)",
                                       (run, run + "-assignment", rid, sid, hotkey, position,
                                        kind, execute if kind == "score" else None,
                                        0 if p["is_king"] else 20))
            cursor.execute("SET LOCAL session_replication_role=origin")
            publication = _measured_publication(cursor, rid, config, participants)
            baseline = publication["final_ranking"][0]
            assert baseline["eligible"] is False
            assert baseline["eligibility_reason"] == "training_data_incomplete"
            assert _verify(cursor, rid, baseline)
            assert _transition_publication(cursor, rid, publication)
            plan = {"commit": "1" * 40, "main_before": "2" * 40,
                    "lab_before": "3" * 40, "timestamp": "2026-10-08T12:00:01Z"}
            for function, expected in (("prepare_promotion", "prepared"),
                                       ("complete_promotion", "promoted")):
                cursor.execute("SELECT public.lab_arena_" + function + "(%s,%s::jsonb)",
                               (rid, json.dumps(plan)))
                assert cursor.fetchone()[0]["status"] == expected
            class SharedTransport(PsycopgTransport):
                def _acquire(self): return connection
                def _release(self, _connection): pass
            store = ArenaStore(SharedTransport(lambda: connection))
            service = _service(store, rid, 591)
            assert service._baseline_fully_succeeded(store.get_round(rid))
            def measured_baseline_ready(_row):
                cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s)", (rid,))
                return cursor.fetchone()[0]
            service._baseline_final_ready = measured_baseline_ready
            assert service.activate_reward(rid)["status"] == "activated"
            assert service.activate_reward(rid)["status"] == "existing"
            assert store.get_round(rid)["publication_doc"] == publication
            # Publication succeeds with the original full score/cost guards.
            # Reopen only the disposable row to test original guarded writes.
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET status='scored',publication_doc=NULL,king_outcome=NULL,king_hotkey=NULL,published_at=NULL,"
                           "promotion_doc=NULL,baseline_promoted_at=NULL,reward_basis_doc=NULL,reward_basis_hash=NULL,"
                           "reward_activated_at=NULL,effective_reward_epoch=NULL,king_start_epoch=NULL,signing_key_doc=NULL "
                           "WHERE round_id=%s", (rid,))
            cursor.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,"
                           "submission_id,run_id,stage,call_identity,provider,operation_id,"
                           "amount_microusd,entry_doc,terminal_response) VALUES "
                           "('dispatch',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat',1,"
                           "'{\"call\":{\"call_succeeded\":true}}'::jsonb,"
                           "'{\"call_succeeded\":true}'::jsonb)",
                           (BASELINE, rid, participants[0]["submission_id"],
                            participants[0]["submission_id"] + "-execute-0", "sha256:" + "9" * 64))
            cursor.execute("SET LOCAL session_replication_role=origin")
            actual = _measured_publication(cursor, rid, config, participants)
            actual_baseline = actual["final_ranking"][0]
            assert actual_baseline["eligible"] is False
            assert actual_baseline["eligibility_reason"] == "provider_calls_inflight"
            assert _verify(cursor, rid, actual_baseline)
            forged = deepcopy(actual_baseline)
            forged["eligibility_reason"] = "training_data_incomplete"
            assert not _verify(cursor, rid, forged)
            forged_summary = deepcopy(baseline)
            assert not _verify(cursor, rid, forged_summary)
            # Both fixed reason and fabricated old summary fail original SQL,
            # before either can become the immutable publication slots trust.
            for forged_row in (forged, forged_summary):
                attack = deepcopy(actual)
                attack["final_ranking"][0] = forged_row
                assert not _transition_publication(cursor, rid, attack)
    finally:
        database.close()


def test_strict_baseline_capture_failure_stays_ineligible(state):
    from lab_arena import baseline_reasoning
    store, control, chain = state
    row, _ = baseline_round({})
    strict = baseline_reasoning.from_configuration(row["configuration_doc"])
    assert strict["reasoning_capture_policy"] == "full_generated_thinking_v1"
    config = {**CONFIG, "baseline_reasoning_configuration": strict}
    rid = _insert_legacy(control, chain, "strict", config=config,
                         mutate=lambda pub: _mutate(pub, "strict_capture_failure"))
    with pytest.raises(ArenaStoreError, match="current_achievement_invalid"):
        store.reward_slot_snapshot(rid, POLICY)


def test_non_superuser_first_apply_replay_keeps_old_bases_and_no_winner_slots():
    database = database_with_lab_arena_migration(MIGRATIONS)
    try:
        pg, dsn = next(database)
        with pg.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                chain = next(CHAINS)
                old = _insert(connection, chain, "prior", day="2026-10-07", delta=11.72)
                rid = f"arena-2026-10-08-{chain}noking"
                pub = _publication(rid, delta=0.5)
                pub["king_decision"] = {"outcome": "no_king"}
                _insert(connection, chain, "noking", activated=False, publication=pub)
                cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r ORDER BY round_id")
                rows = cursor.fetchall()
                cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",
                               (rid, json.dumps(POLICY)))
                before = cursor.fetchone()[0]
                assert [slot["round_id"] for slot in before["reward_slots"]] == [old] * 3
                cursor.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),proowner,proacl,"
                               "prosecdef,provolatile,proconfig FROM pg_proc WHERE pronamespace="
                               "'public'::regnamespace AND proname LIKE 'lab_arena_%' AND prokind='f' ORDER BY oid")
                functions = cursor.fetchall()
                cursor.execute("CREATE ROLE legacy591_first_migrator NOLOGIN NOSUPERUSER INHERIT")
                cursor.execute("GRANT lab_arena_owner TO legacy591_first_migrator")
                cursor.execute("GRANT USAGE,CREATE ON SCHEMA public TO legacy591_first_migrator")
                cursor.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
                cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
                acl = cursor.fetchone()[0]
                cursor.execute("SET ROLE legacy591_first_migrator")
                connection.commit()
                # A wrong computed postimage must roll back before replacement.
                with pytest.raises(pg.Error, match="591 reward slot target differs"):
                    cursor.execute(MIGRATION.read_text().replace(POSTIMAGE, "0" * 64))
                cursor.execute("ROLLBACK")
                assert hashlib.sha256(_definition(cursor).encode()).hexdigest() == PREIMAGE
                for _ in range(2):
                    cursor.execute("SELECT current_user,rolsuper FROM pg_roles WHERE rolname=current_user")
                    assert cursor.fetchone() == ("legacy591_first_migrator", False)
                    cursor.execute(MIGRATION.read_text())
                cursor.execute("RESET ROLE")
                cursor.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),proowner,proacl,"
                               "prosecdef,provolatile,proconfig FROM pg_proc WHERE pronamespace="
                               "'public'::regnamespace AND proname LIKE 'lab_arena_%' AND prokind='f' ORDER BY oid")
                after = cursor.fetchall()
                assert len(functions) == len(after)
                changed = []
                for original, updated in zip(functions, after):
                    assert original[0] == updated[0]
                    assert original[2:] == updated[2:]
                    if original[1] != updated[1]: changed.append(original[0])
                assert changed == ["lab_arena_reward_slot_snapshot(text,jsonb)"]
                cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r ORDER BY round_id")
                assert cursor.fetchall() == rows
                cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
                assert cursor.fetchone()[0] == acl
                cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",
                               (rid, json.dumps(POLICY)))
                assert cursor.fetchone()[0] == before
    finally:
        database.close()
