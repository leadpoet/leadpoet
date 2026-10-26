"""Busy or stale alternatives cannot strand an authenticated score continuation."""

import copy
import base64
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import champion_score_funding576_postgres_test as base
from tests.lab_arena.test_lab_arena_service_round import keypair
from tests.lab_arena.test_lab_arena_migration_postgres import claim as database_claim
from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import legacy_company_cache501_postgres_test as company
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import legacy_floor_zero


ROOT = Path(__file__).parents[2]
SQL = ROOT / "scripts/577-lab-arena-judge-continuation-runner-handoff.sql"
SIG = base.SIGNATURES[1]
PRE = "11f8669d16966fbe41e5c2530c3ed273fea2abc1aacfc4264e2854f017f5b6f3"
scalar = base.scalar


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    fixture = base.scenario.__wrapped__(tmp_path_factory)
    try:
        result = next(fixture)
        with result[0][0].connect(**result[0][1]) as conn, conn.cursor() as cur:
            cur.execute(base.SQL.read_text())
        yield result
    finally:
        fixture.close()


@pytest.fixture(autouse=True)
def restore(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        original = scalar(cur, "pg_get_functiondef(%s::regprocedure)", (SIG,))
        cur.execute("TRUNCATE public.lab_arena_trajectory_events CASCADE")
        cur.execute("TRUNCATE public.lab_arena_company_judgment_reservations CASCADE")
    yield
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(original)


def install(cur, run):
    cur.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (run["run_id"],))
    cur.execute("INSERT INTO public.lab_arena_runs SELECT "
                "(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*",
                (json.dumps(run),))


def prepare(scenario, *, target="baseline", alternative="stale", legacy=False, cause="judge_error"):
    db, h, _, saved = scenario
    rid, baseline, pending = base.prepare(scenario)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        if target == "miner":
            # Preserve the completed baseline while restoring one real admitted
            # miner's accepted execution and its frozen score input from the
            # full service fixture. No fabricated scorer scope is used.
            by_id = {r["run_id"]: r for r in saved["lab_arena_runs"]}
            for name in (pending["run_id"], pending["scored_run_id"]):
                install(cur, by_id[name])
            judge = next(r for r in saved["lab_arena_runs"]
                         if r["round_id"] == rid and r["kind"] == "score" and r["submission_id"] != baseline
                         and r["status"] == "accepted")
            install(cur, by_id[judge["scored_run_id"]])
            install(cur, judge)
            pending = copy.deepcopy(judge)
            pending.update(status="pending", terminal_cause=None, result_doc=None,
                output_ref=None, claim_request_id=None, claim_request_hash=None,
                claim_response=None, lease_token_hash=None, lease_expires_at=None,
                stage_generation=3, champion_funding_sources=None)
            install(cur, pending)
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',"
                        "stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
                        (json.dumps(next(r for r in saved["lab_arena_rounds"] if r["round_id"] == rid)["stage2_scoring_plan_doc"]), rid))
        cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s", (pending["run_id"],))
        cur.execute("DELETE FROM public.lab_arena_judgment_cache WHERE cache_key=%s",
                    (pending["judgment_cache_key"],))
        cur.execute("SET LOCAL session_replication_role=origin")
    h.service._invalidate_hot_round()
    runner = h.runner(0)
    claimed, token, _, _ = database_claim(h.service.store, rid, h.runner_keys[0], parallelism=8, ceiling=20)
    lease = dict(claimed, lease_token=token)
    assert lease["run_id"] == pending["run_id"], lease
    parent = h.service.store.get_run(lease["run_id"])
    if cause in ("lease_expired", "host_fault"):
        append_start(h.service.store, lease)
        if cause == "host_fault":
            append_error(h.service.store, lease)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease["run_id"],))
        result = h.service.store.expire_leases(rid)
        assert result["retried"] == 1, result
    else:
        result = h.service.store.complete_attempt(run_id=lease["run_id"],
            lease_token_hash=hash_lease_token(lease["lease_token"]),
            result={"terminal_status": cause}, terminal_cause=cause, output_ref=None)
        assert result["status"] == "failed", result
    successor_id = parent["assignment_id"] + ":2"
    successor = h.service.store.get_run(successor_id)
    assert successor is not None, result
    primary = parent["runner_hotkey"]
    other = keypair("handoff577-alternative").ss58_address
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        generation = scalar(cur, "stage_generation FROM public.lab_arena_rounds WHERE round_id=%s", (rid,))
        if legacy:
            cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-01' WHERE round_id=%s", (rid,))
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                    "'{runner_hotkeys}',configuration_doc->'runner_hotkeys'||to_jsonb(%s::text)) WHERE round_id=%s",
                    (other, rid))
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s", (successor_id,))
        # The alternative owns unrelated work. Its busy lease is not a promise
        # to take this retry. The stale case mirrors a replaced validator whose
        # old leases remain valid; neither case authorizes their expiry.
        alternate = copy.deepcopy(next(r for r in saved["lab_arena_runs"] if r["round_id"] == rid and r["kind"] == "execute"))
        alternate.update(run_id=rid+":handoff-other:1", assignment_id=rid+":handoff-other",
            status="leased", terminal_cause=None, terminal_doc=None, result_doc=None,
            output_ref=None, runner_hotkey=other, previous_runner_hotkey=None,
            claim_request_id=uuid.uuid4().hex, claim_request_hash="sha256:"+"c"*64,
            lease_token_hash="sha256:"+"c"*64, lease_generation=1,
            stage_generation=generation, champion_funding_sources=None,
            champion_restart_required=False, participation_accepted_at=None)
        install(cur, alternate)
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+interval '50 minutes',"
                    "updated_at=clock_timestamp()-make_interval(mins=>%s) WHERE run_id=%s",
                    (25 if alternative == "stale" else 0, alternate["run_id"]))
        cur.execute("SET LOCAL session_replication_role=origin")
    return rid, successor_id, parent, runner, alternate["run_id"]


def event(store, lease, kind, content):
    return store.append_trajectory_events(lease["run_id"], hash_lease_token(lease["lease_token"]),
        [{"event_id": str(uuid.uuid4()), "kind": kind, "content": content,
          "occurred_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}])


def append_start(store, lease):
    assert event(store, lease, "runtime.started", {"status": "starting",
        "lease_generation": lease["lease_generation"]})["status"] == "accepted"


def append_error(store, lease):
    assert event(store, lease, "runtime.error", {"status": "abandoned",
        "failure_stage": "runtime", "error_class": "RuntimeHostError",
        "runtime_host_reason": "sandbox_launch_failed", "launch_exit_code": 128,
        "launch_timed_out": False})["status"] == "accepted"


def claim(h, runner):
    request = uuid.uuid4().hex
    return h.service.store.claim_assignment(round_id=h.round_id, runner_hotkey=runner,
        declared_parallelism=8, slot_ceiling=20, excluded_miner_hotkeys=[],
        request_id=request, request_hash="sha256:"+request*2,
        lease_token_hash="sha256:"+request*2)


@pytest.fixture(scope="module")
def paid_database():
    from scripts.verify_arena_full_thinking_e2e import full_thinking_database
    fixture = full_thinking_database()
    try:
        db = next(fixture)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for migration in ("575-lab-arena-score-host-continuation.sql",
                              "576-lab-arena-champion-score-funding.sql"):
                cur.execute((ROOT / "scripts" / migration).read_text())
        yield db
    finally:
        fixture.close()


def test_paid_timeout_reuses_three_real_company_judgments_and_keeps_receipts(paid_database):
    db = paid_database
    store = ArenaStore(PsycopgTransport(lambda: db[0].connect(**db[1])))
    store._test_database = db
    rid = "arena-2099-10-20-handoff577"
    try:
        # This older immutable-input seeder disables triggers and predates the
        # required floor column. Supply its explicit floor-zero default only
        # during seeding, then restore the current schema before every RPC.
        with legacy_floor_zero(db):
            _, items = company.open_new(store, round_id=rid,
                markers_by_submission=[["alpha", "bravo", "charlie", "delta", "echo"]])
        # The historical fixture updates scoring deadlines but predates the
        # final publication deadline fence. Keep that fixture deadline open.
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                "'{schedule,publication_deadline}',to_jsonb(%s::text)) WHERE round_id=%s",
                ((datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(), rid))
        primary = keypair("handoff577-paid-primary").ss58_address
        other = keypair("handoff577-paid-other").ss58_address
        parent, token, _, _ = database_claim(store, rid, primary, excluded=[primary])
        assert parent["status"] == "leased"
        identity = contracts.document_hash({"paid-parent": parent["run_id"]})
        token_hash = hash_lease_token(token)
        assert store.reserve_call(run_id=parent["run_id"], lease_token_hash=token_hash,
            call_identity=identity, operation_id="openrouter.chat", provider="openrouter",
            funding_source="miner_key", amount_microusd=2000,
            call_doc={"model": "fixture", "request_hash": identity})["status"] == "reserved"
        assert store.mark_dispatched(run_id=parent["run_id"], lease_token_hash=token_hash,
                                     call_identity=identity)["status"] == "dispatched"
        assert store.settle_call(run_id=parent["run_id"], lease_token_hash=token_hash,
            call_identity=identity, actual_microusd=1000, terminal_response={"status": 200,
                "headers": {"content-type": "application/json"}, "call_succeeded": True,
                "body_b64": base64.b64encode(b'{"choices":[]}').decode()})["status"] == "settled"
        refs = items[0]["company_judgment_refs"]
        partial = company.complete_subset(store, parent, token, refs,
                                          limit=3, terminal="judge_timeout")
        assert partial["status"] == "failed" and partial["company_judgments_stored"] == 3
        successor_id = parent["assignment_id"] + ":2"
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s", (successor_id,))
            source = store.get_run(parent["scored_run_id"])
            alternate = copy.deepcopy(source)
            alternate.update(run_id=rid+":busy-other:1", assignment_id=rid+":busy-other",
                status="leased", terminal_cause=None, output_ref=None, result_doc=None,
                runner_hotkey=other, lease_generation=1, claim_request_id=uuid.uuid4().hex,
                stage_generation=store.get_round(rid)["stage_generation"],
                claim_request_hash="sha256:"+"e"*64, lease_token_hash="sha256:"+"e"*64)
            install(cur, alternate)
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+interval '50 minutes' WHERE run_id=%s", (alternate["run_id"],))
            cur.execute("SET LOCAL session_replication_role=origin")
        assert database_claim(store, rid, primary, excluded=[primary])[0]["status"] == "no_pending"
        ledger_before = store.list_ledger(run_id=parent["run_id"])
        parent_before = store.get_run(parent["run_id"])
        source_before = store.get_run(parent["scored_run_id"])
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(j) FROM public.lab_arena_company_judgments j WHERE source_score_run_id=%s ORDER BY cache_key,authority_slot", (parent["run_id"],))
            cached = cur.fetchall()
            assert len(cached) == 3
            cur.execute(SQL.read_text())
        retry, retry_token, request_id, request_hash = database_claim(store, rid, primary, excluded=[primary])
        assert retry["status"] == "leased" and retry["run_id"] == successor_id
        assert sorted(r["company_index"] for r in retry["company_judgment_cache"]["hits"]) == [0, 1, 2]
        assert sorted(r["company_index"] for r in retry["company_judgment_cache"]["misses"]) == [3, 4]
        assert store.claim_assignment(round_id=rid, runner_hotkey=primary, declared_parallelism=1,
            slot_ceiling=8, excluded_miner_hotkeys=[primary], request_id=request_id,
            request_hash=request_hash, lease_token_hash=hash_lease_token(retry_token)) == retry
        accepted = company.complete_subset(store, retry, retry_token, refs)
        assert accepted["status"] == "accepted"
        assert store.list_ledger(run_id=parent["run_id"]) == ledger_before
        assert store.get_run(parent["run_id"]) == parent_before
        assert store.get_run(parent["scored_run_id"]) == source_before
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(j) FROM public.lab_arena_company_judgments j WHERE source_score_run_id=%s ORDER BY cache_key,authority_slot", (parent["run_id"],))
            assert cur.fetchall() == cached
        assert store.expire_leases(rid)["retried"] == 0
    finally:
        store.close()


@pytest.mark.parametrize("accounting", ["dispatched", "uncertain_success"])
def test_unresolved_provider_work_cannot_create_a_continuation(paid_database, accounting):
    db = paid_database
    store = ArenaStore(PsycopgTransport(lambda: db[0].connect(**db[1])))
    store._test_database = db
    rid = "arena-2099-10-20-billing577"
    try:
        with legacy_floor_zero(db):
            company.open_new(store, round_id=rid, markers_by_submission=[["alpha"]])
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                "'{schedule,publication_deadline}',to_jsonb(%s::text)) WHERE round_id=%s",
                ((datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(), rid))
            cur.execute("SET LOCAL session_replication_role=origin")
            cur.execute(SQL.read_text())
        primary = keypair("handoff577-billing").ss58_address
        parent, token, _, _ = database_claim(store, rid, primary, excluded=[primary])
        assert parent["status"] == "leased"
        identity = contracts.document_hash({"unclear": parent["run_id"]})
        token_hash = hash_lease_token(token)
        assert store.reserve_call(run_id=parent["run_id"], lease_token_hash=token_hash,
            call_identity=identity, operation_id="openrouter.chat", provider="openrouter",
            funding_source="miner_key", amount_microusd=2000,
            call_doc={"model": "fixture", "request_hash": identity})["status"] == "reserved"
        assert store.mark_dispatched(run_id=parent["run_id"], lease_token_hash=token_hash,
                                     call_identity=identity)["status"] == "dispatched"
        if accounting == "uncertain_success":
            assert store.mark_uncertain(run_id=parent["run_id"], lease_token_hash=token_hash,
                call_identity=identity, call_doc={"reason": "missing_provider_cost", "provider_status": 200})["status"] == "uncertain"
        before = store.list_ledger(run_id=parent["run_id"])
        result = store.complete_attempt(run_id=parent["run_id"], lease_token_hash=token_hash,
            result={"terminal_status": "judge_timeout"}, terminal_cause="judge_timeout", output_ref=None)
        assert result["status"] == ("accounting_open" if accounting == "dispatched" else "failed")
        assert store.get_run(parent["assignment_id"]+":2") is None
        assert store.expire_leases(rid)["retried"] == 0
        assert store.get_run(parent["assignment_id"]+":2") is None
        assert store.list_ledger(run_id=parent["run_id"]) == before
    finally:
        store.close()


@pytest.mark.parametrize("target", ["baseline", "miner"])
@pytest.mark.parametrize("alternative", ["stale", "busy"])
def test_real_continuation_fails_before_and_claims_after_with_state_preserved(scenario, target, alternative):
    db, h, _, _ = scenario
    rid, successor_id, parent, runner, other_id = prepare(scenario, target=target, alternative=alternative)
    try:
        primary = parent["runner_hotkey"]
        before = h.service.store.get_run(successor_id)
        assert before["previous_runner_hotkey"] == primary
        assert claim(h, primary)["status"] == "no_pending"
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert scalar(cur, "public.lab_arena__judge_continuation_v1(d,r) FROM public.lab_arena_rounds d "
                          "JOIN public.lab_arena_runs r USING(round_id) WHERE r.run_id=%s", (successor_id,)) is True
            saved = base.rows(cur, rid)
            metadata = scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (SIG,))
            original = scalar(cur, "pg_get_functiondef(%s::regprocedure)", (SIG,))
            assert hashlib.sha256(original.encode()).hexdigest() == PRE
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            assert base.rows(cur, rid) == saved
            assert scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (SIG,)) == metadata
        after = claim(h, primary)
        assert after["status"] == "leased" and after["run_id"] == successor_id, after
        stored = h.service.store.get_run(successor_id)
        assert h.service.store.provider_funding(successor_id, "deepline")["funding_source"] == (
            "host" if target == "baseline" else "miner_key")
        for field in ("assignment_id", "scored_run_id", "judgment_input_hash", "judgment_cache_key",
                      "judgment_scope_doc", "company_judgment_refs", "scoring_generation_floor"):
            assert stored[field] == before[field]
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            current = base.rows(cur, rid)
            assert current["lab_arena_rounds"] == saved["lab_arena_rounds"]
            assert current["lab_arena_ledger"] == saved["lab_arena_ledger"]
            assert [r for r in current["lab_arena_runs"] if r["run_id"] != successor_id] == [r for r in saved["lab_arena_runs"] if r["run_id"] != successor_id]
            assert scalar(cur, "status FROM public.lab_arena_runs WHERE run_id=%s", (other_id,)) == "leased"
    finally:
        runner.close()


@pytest.mark.parametrize("control", ["backoff", "legacy", "lease_expired", "host_fault"])
def test_backoff_legacy_and_authenticated_host_handoffs_still_deny(scenario, control):
    db, h, _, _ = scenario
    rid, successor, parent, runner, _ = prepare(scenario, legacy=control == "legacy",
        cause=control if control in ("lease_expired", "host_fault") else "judge_error")
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            if control == "backoff":
                cur.execute("SET LOCAL session_replication_role=replica")
                cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp() WHERE run_id=%s", (successor,))
            continuation = scalar(cur, "public.lab_arena__judge_continuation_v1(d,r) "
                "FROM public.lab_arena_rounds d JOIN public.lab_arena_runs r USING(round_id) WHERE r.run_id=%s", (successor,))
            assert continuation is (control != "legacy")
        assert claim(h, parent["runner_hotkey"])["status"] == "no_pending"
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            before = base.rows(cur, rid)
            cur.execute(SQL.read_text())
            assert base.rows(cur, rid) == before
        assert claim(h, parent["runner_hotkey"])["status"] == "no_pending"
    finally:
        runner.close()


def test_ordinary_execute_confirmation_still_prefers_other_runner(scenario):
    db, h, _, _ = scenario
    rid, successor, parent, runner, other_id = prepare(scenario, target="miner")
    try:
        primary = parent["runner_hotkey"]
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            source = h.service.store.get_run(parent["scored_run_id"])
            generation = h.service.store.get_round(rid)["stage_generation"]
            assignment = rid + ":ordinary-execute-confirmation"
            failed = copy.deepcopy(source)
            failed.update(run_id=assignment+":1", assignment_id=assignment, status="failed",
                terminal_cause="model_error", result_doc={"terminal_status": "model_error"},
                output_ref=None, stage_generation=generation, runner_hotkey=primary,
                claim_request_id=None, claim_request_hash=None, claim_response=None,
                lease_token_hash=None, lease_expires_at=None, per_icp_score=None,
                qualification_doc=None, participation_accepted_at=None)
            install(cur, failed)
            pending = copy.deepcopy(failed)
            pending.update(run_id=assignment+":2", attempt=2, status="pending",
                terminal_cause=None, result_doc=None, runner_hotkey=None,
                previous_runner_hotkey=primary, claim_request_id=None, claim_request_hash=None,
                claim_response=None, lease_token_hash=None, lease_expires_at=None)
            install(cur, pending)
            cur.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (successor,))
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id=%s", (rid,))
            cur.execute("SET LOCAL session_replication_role=origin")
        assert claim(h, primary)["status"] == "no_pending"
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
        assert claim(h, primary)["status"] == "no_pending"
        other = h.service.store.get_run(other_id)["runner_hotkey"]
        claimed = claim(h, other)
        assert claimed["status"] == "leased" and claimed["run_id"] == pending["run_id"]
    finally:
        runner.close()


def test_three_authenticated_host_faults_still_quarantine_new_score_work(scenario):
    db, h, _, saved = scenario
    rid, baseline, pending = base.prepare(scenario)
    selected = [r for r in saved["lab_arena_runs"] if r["round_id"] == rid
                and r["submission_id"] == baseline and r["kind"] == "score"
                and r["icp_position"] in (6, 7, 8)]
    assert len(selected) == 3
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        for old in selected:
            new = copy.deepcopy(old)
            new.update(status="pending", terminal_cause=None, result_doc=None, output_ref=None,
                claim_request_id=None, claim_request_hash=None, claim_response=None,
                lease_token_hash=None, lease_expires_at=None, stage_generation=3,
                champion_funding_sources=None)
            install(cur, new)
            cur.execute("DELETE FROM public.lab_arena_judgment_cache WHERE cache_key=%s", (new["judgment_cache_key"],))
            cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s", (new["run_id"],))
    primary = h.runner_keys[0]
    for _ in range(3):
        run, token, _, _ = database_claim(h.service.store, rid, primary, parallelism=8, ceiling=20)
        assert run["status"] == "leased"
        lease = dict(run, lease_token=token)
        append_start(h.service.store, lease)
        append_error(h.service.store, lease)
    # Position 9 is fresh, not a retry of a faulty assignment. Only the round
    # quarantine should reject it after three distinct authenticated faults.
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        assert scalar(cur, "public.lab_arena__judge_continuation_v1(d,r) "
            "FROM public.lab_arena_rounds d JOIN public.lab_arena_runs r USING(round_id) WHERE r.run_id=%s", (pending["run_id"],)) is True
    assert claim(h, primary)["status"] == "no_pending"
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
    assert claim(h, primary)["status"] == "no_pending"


def test_existing_funding576_effective_provider_and_replay_control(scenario, monkeypatch):
    original = base.SQL.read_text()

    class BothMigrations:
        def read_text(self):
            return original + "\n" + SQL.read_text()

    monkeypatch.setattr(base, "SQL", BothMigrations())
    base.test_real_score_broker_uses_frozen_provider_payer_and_replays_once(scenario, migrated=True)


def test_concurrent_continuation_claims_allocate_exactly_once(scenario):
    db, h, _, _ = scenario
    _, successor, parent, runner, _ = prepare(scenario)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: claim(h, parent["runner_hotkey"]), range(2)))
        assert sorted(r["status"] for r in results) == ["leased", "no_pending"]
        assert next(r for r in results if r["status"] == "leased")["run_id"] == successor
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert scalar(cur, "count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND status='leased'",
                          (parent["assignment_id"],)) == 1
    finally:
        runner.close()


@pytest.mark.parametrize("drift", ["source", "acl", "security"])
@pytest.mark.parametrize("installed", [False, True])
def test_exact_migration_rejects_drift_before_and_after_install(scenario, drift, installed):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        if installed:
            cur.execute(SQL.read_text())
            conn.commit()
        original = scalar(cur, "pg_get_functiondef(%s::regprocedure)", (SIG,))
        if drift == "source":
            cur.execute(original.replace("BEGIN", "BEGIN\n  NULL;", 1))
        elif drift == "acl":
            cur.execute("GRANT EXECUTE ON FUNCTION " + SIG + " TO anon")
        else:
            cur.execute("ALTER FUNCTION " + SIG + " SECURITY INVOKER")
        with pytest.raises(db[0].Error):
            cur.execute(SQL.read_text())
        conn.rollback()
        assert scalar(cur, "pg_get_functiondef(%s::regprocedure)", (SIG,)) == original
