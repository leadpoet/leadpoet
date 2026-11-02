"""Count genuine judge attempts; authenticated launcher losses retain recovery."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import bounded_judge_attempts596_postgres_test as prior
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL = Path(__file__).parents[2] / "scripts/597-lab-arena-zero-call-judge-attempt-credit.sql"
HELPER = "public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)"
scenario = prior.scenario


@pytest.fixture(autouse=True)
def reset_functions(scenario):
    prior.original_functions.__wrapped__(scenario)
    db = scenario[0][0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        for signature in prior.SIGNATURES:
            cur.execute("REVOKE ALL ON FUNCTION " + signature + " FROM authenticated")
        cur.execute("DROP FUNCTION IF EXISTS " + HELPER)


def prepare(scenario, attempt=1, *, floor=0):
    context, rid, pending = prior.prepare(scenario, attempt, floor=floor)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        # The historical fixture retains the replaced score's provider rows.
        # Remove only this synthetic assignment's rows to model a fresh launch.
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id IN "
                    "(SELECT run_id FROM public.lab_arena_runs WHERE assignment_id=%s)",
                    (pending["assignment_id"],))
    prior.upgrade(context)
    return context, rid, pending


def upgrade(context):
    db = context[0]
    before = prior.snapshots(context)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        schema_acl = prior.scalar(cur, "nspacl::text FROM pg_namespace WHERE nspname='public'")
        cur.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p "
                    "WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        definitions = cur.fetchall()
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        changed = []
        for oid, definition, metadata in definitions:
            assert prior.scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s", (oid,)) == metadata
            if prior.scalar(cur, "pg_get_functiondef(%s)", (oid,)) != definition:
                changed.append(str(oid))
        assert set(changed) == {str(prior.scalar(cur, "%s::regprocedure::oid", (sig,))) for sig in prior.SIGNATURES}
        assert prior.scalar(cur, "proacl::text FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,)) == "{lab_arena_owner=X/lab_arena_owner}"
        assert prior.scalar(cur, "nspacl::text FROM pg_namespace WHERE nspname='public'") == schema_acl
    assert prior.snapshots(context) == before


def claim(context, runner="bad-host", *, age=True, pending=None):
    if age:
        assert pending is not None
        for run in context[1].service.store.list_runs(context[1].round_id):
            if run["assignment_id"] == pending["assignment_id"] and run["status"] == "pending":
                prior.age(context, run["run_id"])
    db = context[0]
    token = uuid.uuid4().hex
    token_hash = hash_lease_token(token)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET ROLE lab_arena_service")
        cur.execute("SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)",
                    (context[1].round_id, hotkey("score597-" + runner), uuid.uuid4().hex,
                     "sha256:" + uuid.uuid4().hex * 2, token_hash))
        lease = cur.fetchone()[0]
    if lease.get("status") == "leased":
        lease["lease_token"] = token
    return lease


def zero_loss(context, pending, *, runner="bad-host", started=False):
    lease = claim(context, runner, pending=pending)
    assert lease["status"] == "leased", lease
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    events = []
    if started:
        events.append({"event_id": str(uuid.uuid4()), "kind": "runtime.started", "occurred_at": now,
                       "content": {"status": "starting", "runtime": "runsc", "lease_generation": lease["lease_generation"]}})
    events.append({"event_id": str(uuid.uuid4()), "kind": "runtime.error", "occurred_at": now,
                   "content": {"status": "abandoned", "failure_stage": "runtime", "error_class": "RuntimeHostError",
                               "runtime_host_reason": "sandbox_launch_failed", "launch_exit_code": 128,
                               "launch_timed_out": False}})
    assert context[1].service.store.append_trajectory_events(
        lease["run_id"], hash_lease_token(lease["lease_token"]), events)["status"] == "accepted"
    expiry = context[1].service.store.expire_leases(context[1].round_id)
    row = context[1].service.store.get_run(lease["run_id"])
    assert row["terminal_doc"]["recovery_reason"] == "authenticated_zero_call_runtime_host_error"
    return lease, expiry


def count(context, run_id):
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        return prior.scalar(cur, "public.lab_arena__score_attempt_count_v1(r) "
                            "FROM public.lab_arena_runs r WHERE run_id=%s", (run_id,))


def test_five_same_host_zero_call_losses_reproduce_then_recover_without_changing_accepted_work(scenario):
    context, rid, pending = prepare(scenario)
    accepted = [r for r in context[1].service.store.list_runs(rid) if r["status"] == "accepted"]
    assert len(accepted) == 19
    for number in range(1, 6):
        lease, expiry = zero_loss(context, pending)
        assert lease["attempt"] == number
        assert expiry["retried"] == (1 if number < 5 else 0)
    sixth = pending["assignment_id"] + ":6"
    assert sixth not in prior.run_ids(context, pending)  # failing-before witness
    upgrade(context)
    # Historical failed materialization is fixed without rewriting history.
    assert context[1].service.store.expire_leases(rid)["retried"] == 1
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert count(context, pending["assignment_id"] + ":5") == 0
    assert count(context, sixth) == 1
    assert claim(context, age=False) == {"status": "no_pending"}  # physical backoff remains
    # With no started event the existing handoff does not exclude this runner.
    # The separate quarantine still needs three DISTINCT assignments.
    lease = claim(context, pending=pending)
    assert lease["attempt"] == 6
    assert [r for r in context[1].service.store.list_runs(rid) if r["status"] == "accepted"] == accepted
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        assert prior.scalar(cur, "count(*) FROM public.lab_arena_ledger WHERE run_id IN "
                            "(SELECT run_id FROM public.lab_arena_runs WHERE assignment_id=%s)",
                            (pending["assignment_id"],)) == 0


@pytest.mark.parametrize("path", ["complete", "expire"])
@pytest.mark.parametrize("layout", ["prefix", "intervening"])
def test_five_genuine_attempts_remain_bounded_after_five_zero_call_losses(scenario, path, layout):
    context, rid, pending = prepare(scenario)
    upgrade(context)
    steps = ["loss"] * 5 + ["real"] * 5 if layout == "prefix" else ["loss", "real"] * 5
    actual = 0
    for number, step in enumerate(steps, 1):
        if step == "loss":
            lease, expiry = zero_loss(context, pending)
            assert lease["attempt"] == number and expiry["retried"] == 1
            continue
        actual += 1
        lease = claim(context, pending=pending)
        assert lease["attempt"] == number
        assert count(context, lease["run_id"]) == actual
        if path == "complete":
            result = prior.fail(context, lease)
            assert ("confirmation_attempt" in result) is (actual < 5)
        else:
            db = context[0]
            with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
                cur.execute("SET LOCAL session_replication_role=replica")
                cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease["run_id"],))
            assert context[1].service.store.expire_leases(rid)["retried"] == (1 if actual < 5 else 0)
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert pending["assignment_id"] + ":11" not in prior.run_ids(context, pending)
    assert claim(context, pending=pending) == {"status": "no_pending"}


@pytest.mark.parametrize("mutation", [
    "receipt", "error", "original_expiry", "runner", "assignment", "lease_generation",
    "scoring_generation", "stale", "late", "class", "setup", "provider", "finished", "cleanup",
    "ledger", "result", "output", "retired", "policy", "date",
])
def test_unproved_paid_or_mismatched_failure_never_gets_attempt_credit(scenario, mutation):
    context, _, pending = prepare(scenario)
    upgrade(context)
    lease, _ = zero_loss(context, pending, started=True)
    assert count(context, pending["assignment_id"] + ":2") == 1
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        run = lease["run_id"]
        if mutation == "receipt":
            cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=NULL WHERE run_id=%s", (run,))
        elif mutation == "error":
            cur.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=%s AND event_kind='runtime.error'", (run,))
        elif mutation == "original_expiry":
            cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=jsonb_set(terminal_doc,'{original_lease_expires_at}',to_jsonb((clock_timestamp()+interval '2 hours')::text)) WHERE run_id=%s", (run,))
        elif mutation in ("runner", "assignment"):
            column = "runner_hotkey" if mutation == "runner" else "assignment_id"
            cur.execute(f"UPDATE public.lab_arena_trajectory_events SET {column}='mismatched' WHERE run_id=%s AND event_kind='runtime.error'", (run,))
        elif mutation in ("lease_generation", "scoring_generation"):
            column = "lease_generation" if mutation == "lease_generation" else "scoring_generation_floor"
            cur.execute(f"UPDATE public.lab_arena_runs SET {column}={column}+1 WHERE run_id=%s", (run,))
        elif mutation in ("stale", "late"):
            column, offset = ("occurred_at", "-interval '2 hours'") if mutation == "stale" else ("created_at", "+interval '2 hours'")
            cur.execute(f"UPDATE public.lab_arena_trajectory_events SET {column}=clock_timestamp(){offset} WHERE run_id=%s AND event_kind='runtime.error'", (run,))
        elif mutation in ("class", "setup"):
            key, value = ("error_class", "RunnerError") if mutation == "class" else ("failure_stage", "setup")
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s,to_jsonb(%s::text)) WHERE run_id=%s AND event_kind='runtime.error'", ([key], value, run))
        elif mutation in ("provider", "finished", "cleanup"):
            kind = {"provider": "provider.request", "finished": "runtime.finished", "cleanup": "runtime.cleanup_error"}[mutation]
            cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE run_id=%s AND event_kind='runtime.started'", (kind, run))
        elif mutation == "ledger":
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) "
                        "SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,'sha256:'||repeat('e',64),'openrouter','openrouter.chat','host',1 FROM public.lab_arena_runs WHERE run_id=%s", (run,))
        elif mutation in ("result", "output", "retired"):
            setting = {"result": "result_doc='{}'", "output": "output_ref='saved-output'", "retired": "champion_restart_required=true"}[mutation]
            cur.execute("UPDATE public.lab_arena_runs SET " + setting + " WHERE run_id=%s", (run,))
        elif mutation == "policy":
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'infrastructure_retry_cost_policy'")
        elif mutation == "date":
            cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13'")
    # Moving the old receipt to a different scoring generation must not buy
    # credit, even though the current generation now starts at physical two.
    assert count(context, pending["assignment_id"] + ":2") == (1 if mutation == "scoring_generation" else 2)


def test_supersession_uses_only_its_own_zero_call_receipts_and_five_attempts(scenario):
    context, rid, pending = prepare(scenario, 36, floor=1)
    upgrade(context)
    loss, expiry = zero_loss(context, pending, started=False)
    assert loss["attempt"] == 36 and expiry["retried"] == 1
    for number in range(37, 42):
        lease = claim(context, pending=pending)
        assert lease["attempt"] == number and count(context, lease["run_id"]) == number - 36
        result = prior.fail(context, lease)
        assert ("confirmation_attempt" in result) is (number < 41)
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert pending["assignment_id"] + ":42" not in prior.run_ids(context, pending)


def test_deadline_stops_repeated_same_host_launch_failures(scenario):
    context, rid, pending = prepare(scenario)
    upgrade(context)
    for _ in range(6):
        assert zero_loss(context, pending)[1]["retried"] == 1
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s", (rid,))
    before = prior.snapshots(context)
    assert claim(context, age=False) == {"status": "no_pending"}
    # Age before a second check so only the deadline can reject this attempt.
    prior.age(context, pending["assignment_id"] + ":7")
    assert claim(context, age=False) == {"status": "no_pending"}
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert prior.snapshots(context)["lab_arena_ledger"] == before["lab_arena_ledger"]


def test_signed_start_keeps_existing_same_host_handoff_exclusion(scenario):
    context, _, pending = prepare(scenario)
    upgrade(context)
    assert zero_loss(context, pending, started=True)[1]["retried"] == 1
    assert claim(context, pending=pending) == {"status": "no_pending"}
    lease = claim(context, "healthy-alternative", pending=pending)
    assert lease["attempt"] == 2 and count(context, lease["run_id"]) == 1


@pytest.mark.parametrize("cause", ["accepted", "judge_error"])
def test_existing_paid_lease_above_bound_can_settle_and_complete(scenario, cause):
    context, rid, pending = prior.prepare(scenario, 36)
    lease = prior.claim(context)  # admitted before both bounded migrations
    assert lease["attempt"] == 36
    h = context[1]
    token_hash = hash_lease_token(lease["lease_token"])
    identity = "sha256:" + "a" * 64
    assert h.service.store.reserve_call(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity, operation_id="openrouter.chat", provider="openrouter",
        funding_source="host", amount_microusd=100, call_doc={})["status"] == "reserved"
    assert h.service.store.mark_dispatched(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity)["status"] == "dispatched"
    prior.upgrade(context)
    upgrade(context)
    assert h.service.store.settle_call(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity, actual_microusd=100,
        terminal_response={"call_succeeded": True, "status": 200})["status"] == "settled"
    result = h.service.store.complete_attempt(run_id=lease["run_id"], lease_token_hash=token_hash,
        result={"terminal_status": cause}, terminal_cause=cause,
        output_ref="saved-judge-output" if cause == "accepted" else None)
    assert result["status"] == ("accepted" if cause == "accepted" else "failed")
    assert "confirmation_attempt" not in result
    assert h.service.store.expire_leases(rid)["retried"] == 0
    assert pending["assignment_id"] + ":37" not in prior.run_ids(context, pending)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (identity,))
        assert cur.fetchall() == [("reservation", 0), ("dispatch", 0), ("settlement", 100)]


def test_expiry_retains_partial_company_refs_and_settled_calls(scenario):
    context, rid, pending = prepare(scenario)
    upgrade(context)
    for _ in range(5):
        zero_loss(context, pending)
    lease = claim(context, pending=pending)
    assert lease["attempt"] == 6
    h = context[1]
    token_hash = hash_lease_token(lease["lease_token"])
    identity = "sha256:" + "b" * 64
    assert h.service.store.reserve_call(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity, operation_id="openrouter.chat", provider="openrouter",
        funding_source="host", amount_microusd=100, call_doc={})["status"] == "reserved"
    assert h.service.store.mark_dispatched(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity)["status"] == "dispatched"
    assert h.service.store.settle_call(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity, actual_microusd=100,
        terminal_response={"call_succeeded": False, "status": 503})["status"] == "settled"
    frozen = h.service.store.get_round(rid)
    scope = {"cache_key": "sha256:" + "c" * 64, "company_input_hash": "sha256:" + "d" * 64,
             "round_id": rid, "network_name": frozen["arena_network_name"], "netuid": frozen["arena_netuid"],
             "evaluation_date": frozen["evaluation_date"], "integrity_policy": "arena_integrity_v1",
             "company_judgment_policy": "company_judgment_v1",
             "scorer_image_digest": frozen["configuration_doc"]["scorer_image_digest"],
             "scorer_image_reference": frozen["configuration_doc"]["scorer_image_reference"]}
    refs = [{"schema_version": "leadpoet.lab_arena.company_judgment_ref.v1", "company_index": 0,
             "cache_key": scope["cache_key"], "company_input_hash": scope["company_input_hash"], "scope_doc": scope}]
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        # Preserve one consistent frozen company scope through the synthetic
        # predecessor chain, as required by the existing continuation proof.
        cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb WHERE assignment_id=%s",
                    (json.dumps(refs), pending["assignment_id"]))
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease["run_id"],))
    ledger = prior.snapshots(context)["lab_arena_ledger"]
    assert h.service.store.expire_leases(rid)["retried"] == 1
    assert count(context, lease["run_id"]) == 1  # paid work is a real attempt
    successor = h.service.store.get_run(pending["assignment_id"] + ":7")
    assert successor["company_judgment_refs"] == refs
    assert prior.snapshots(context)["lab_arena_ledger"] == ledger


def test_recovered_queue_finishes_real_scorer_and_publication(scenario):
    context, rid, pending = prepare(scenario)
    for _ in range(5):
        zero_loss(context, pending)
    upgrade(context)
    assert context[1].service.store.expire_leases(rid)["retried"] == 1
    prior.age(context, pending["assignment_id"] + ":6")
    runner = context[1].runner(0)
    try:
        lease = runner.claim_one()
        assert lease["attempt"] == 6
        assert runner._slots.acquire(blocking=False)
        runner._run_lease(lease)
        assert context[1].service.store.get_run(lease["run_id"])["status"] == "accepted"
    finally:
        runner.close()
    context[1].advance_until("scored", runners=1)
    assert context[1].service.publish(rid)["status"] == "ok"
    assert context[1].status() == "published"


def test_concurrent_expiry_materializes_exactly_one_successor(scenario):
    context, rid, pending = prepare(scenario)
    for _ in range(5):
        zero_loss(context, pending)
    upgrade(context)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: context[1].service.store.expire_leases(rid), range(2)))
    assert sum(r["retried"] for r in results) == 1
    assert prior.run_ids(context, pending).count(pending["assignment_id"] + ":6") == 1


def test_pending_real_six_stays_unclaimable_and_private_helper_stays_private(scenario):
    context, _, pending = prepare(scenario, 6)
    upgrade(context)
    assert count(context, pending["run_id"]) == 6
    assert claim(context, pending=pending) == {"status": "no_pending"}
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET ROLE lab_arena_service")
        with pytest.raises(db[0].Error, match="permission denied for function"):
            cur.execute("SELECT public.lab_arena__score_attempt_count_v1(r) FROM public.lab_arena_runs r WHERE run_id=%s", (pending["run_id"],))
        conn.rollback()


def test_existing_owner_migrator_with_broad_default_grants_keeps_helper_private(scenario):
    context, _, _ = prepare(scenario)
    db = context[0]
    before = prior.snapshots(context)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("CREATE ROLE judge597_migrator NOLOGIN NOSUPERUSER INHERIT")
        cur.execute("GRANT lab_arena_owner TO judge597_migrator")
        cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO judge597_migrator WITH GRANT OPTION")
        cur.execute("ALTER DEFAULT PRIVILEGES FOR ROLE judge597_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role")
        cur.execute("SET ROLE judge597_migrator")
        schema_acl = prior.scalar(cur, "nspacl::text FROM pg_namespace WHERE nspname='public'")
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            assert prior.scalar(cur, "has_function_privilege(%s,%s,'EXECUTE')", (role, HELPER)) is False
        assert prior.scalar(cur, "nspacl::text FROM pg_namespace WHERE nspname='public'") == schema_acl
        cur.execute("RESET ROLE")
    assert prior.snapshots(context) == before


@pytest.mark.parametrize("drift", ["body", "acl", "postimage", "helper_body", "helper_acl"])
def test_migration_fails_closed_atomically_and_never_overwrites_drift(scenario, drift):
    context, _, _ = prepare(scenario)
    db = context[0]
    if drift.startswith("helper_"):
        upgrade(context)
    before = prior.snapshots(context)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        sql = SQL.read_text()
        target = HELPER if drift.startswith("helper_") else prior.SIGNATURES[0]
        if drift in ("body", "helper_body"):
            definition = prior.scalar(cur, "pg_get_functiondef(%s::regprocedure)", (target,))
            marker = "SELECT GREATEST" if drift == "helper_body" else "BEGIN"
            cur.execute(definition.replace(marker, "-- unreviewed source\n " + marker, 1))
        elif drift in ("acl", "helper_acl"):
            cur.execute("GRANT EXECUTE ON FUNCTION " + target + " TO authenticated")
        else:
            sql = sql.replace("df4e15dc9008ff0774e60f56ac7780b85e6783b7c12add61adac94ba8cbf68aa", "0" * 64)
        conn.commit()
        definition = prior.scalar(cur, "pg_get_functiondef(%s::regprocedure)", (target,))
        with pytest.raises(db[0].Error, match="(source|preimage|postimage|security).*differs"):
            cur.execute(sql)
        conn.rollback()
        assert prior.scalar(cur, "pg_get_functiondef(%s::regprocedure)", (target,)) == definition
        if not drift.startswith("helper_"):
            assert prior.scalar(cur, "to_regprocedure(%s)", (HELPER,)) is None
    assert prior.snapshots(context) == before
