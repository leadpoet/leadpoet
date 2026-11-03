"""Five judge attempts per frozen scoring generation, without cancelling leases."""
from pathlib import Path
import json

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import company_judgment_validation_projection548_postgres_test as completed
from tests.lab_arena import concurrent_baseline_judging572_postgres_test as queue
from tests.lab_arena.lab_arena_pg_harness import current_service_database

SQL = Path(__file__).parents[2] / "scripts/596-lab-arena-bounded-judge-attempts.sql"
SIGNATURES = (
    "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)",
    "public.lab_arena_complete_attempt(text,text,jsonb,text,text)",
    "public.lab_arena_expire_leases(text)",
)


def scalar(cur, expression, params=()):
    cur.execute("SELECT " + expression, params)
    return cur.fetchone()[0]


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    database = current_service_database()
    db = next(database)
    fixture = completed.completed.__wrapped__(db, tmp_path_factory)
    try:
        h, row = next(fixture)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            saved = {}
            for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger", "lab_arena_judgment_cache"):
                saved[table] = scalar(cur, "coalesce(jsonb_agg(to_jsonb(t)),'[]'::jsonb) FROM public." + table + " t")
            definitions = [scalar(cur, "pg_get_functiondef(%s::regprocedure)", (sig,)) for sig in SIGNATURES]
        yield (db, h, row, saved), definitions
    finally:
        fixture.close()
        database.close()


@pytest.fixture(autouse=True)
def original_functions(scenario):
    db = scenario[0][0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        for definition in scenario[1]:
            cur.execute(definition)
        cur.execute("TRUNCATE public.lab_arena_trajectory_events CASCADE")
        cur.execute("TRUNCATE public.lab_arena_company_judgment_reservations CASCADE")


def prepare(scenario, attempt, *, floor=0):
    context = scenario[0]
    rid, _, pending = queue.prepare(context, attempt=attempt, floor=floor)
    if floor:
        # The historical overlap fixture updates all other score rows. A
        # supersession instead leaves physical predecessors at the old floor.
        db = context[0]
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_runs SET scoring_generation_floor=0 "
                        "WHERE assignment_id=%s AND attempt<%s",
                        (pending["assignment_id"], attempt))
    return context, rid, pending


def upgrade(context):
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        metadata = [scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (sig,)) for sig in SIGNATURES]
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert [scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (sig,)) for sig in SIGNATURES] == metadata


def snapshots(context):
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        return {table: scalar(cur, "coalesce(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text),'[]'::jsonb) FROM public." + table + " t")
                for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger", "lab_arena_judgment_cache")}


def run_ids(context, pending):
    return [r["run_id"] for r in context[1].service.store.list_runs(context[1].round_id)
            if r["assignment_id"] == pending["assignment_id"]]


def claim(context):
    runner = context[1].runner(0)
    try:
        return runner.claim_one()
    finally:
        runner.close()


def fail(context, lease, cause="judge_error"):
    return context[1].service.store.complete_attempt(
        run_id=lease["run_id"], lease_token_hash=hash_lease_token(lease["lease_token"]),
        result={"terminal_status": cause}, terminal_cause=cause, output_ref=None)


def age(context, run_id):
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '30 minutes' WHERE run_id=%s", (run_id,))


def mark_failed(context, pending, cause="judge_error"):
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=%s,result_doc=%s::jsonb WHERE run_id=%s",
                    (cause, json.dumps({"terminal_status": cause}), pending["run_id"]))


def test_reproduces_unbounded_materialization_then_preserves_19_accepted_rows(scenario):
    context, rid, pending = prepare(scenario, 5)
    mark_failed(context, pending)
    h = context[1]
    accepted = [r for r in h.service.store.list_runs(rid) if r["status"] == "accepted"]
    assert len(accepted) == 19
    # Failing-before witness: this exact deployed queue creates attempt six.
    assert h.service.store.expire_leases(rid)["retried"] == 1
    sixth = pending["assignment_id"] + ":6"
    assert sixth in run_ids(context, pending)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (sixth,))
    before = snapshots(context)
    upgrade(context)
    assert snapshots(context) == before
    assert h.service.store.expire_leases(rid)["retried"] == 0
    assert sixth not in run_ids(context, pending)
    assert [r for r in h.service.store.list_runs(rid) if r["status"] == "accepted"] == accepted
    assert snapshots(context) == before


@pytest.mark.parametrize("attempt", [4, 5])
@pytest.mark.parametrize("cause", ["judge_error", "judge_timeout", "credential_error"])
def test_completion_four_can_create_five_but_five_never_creates_six(scenario, attempt, cause):
    context, rid, pending = prepare(scenario, attempt)
    upgrade(context)
    lease = claim(context)
    assert lease["run_id"] == pending["run_id"]
    result = fail(context, lease, cause)
    assert result["status"] == "failed"
    if attempt == 4:
        assert result["confirmation_attempt"] == 5
        successor = pending["assignment_id"] + ":5"
        assert successor in run_ids(context, pending)
        assert claim(context) == {"status": "no_pending"}  # existing backoff still applies
        age(context, successor)
        assert claim(context)["run_id"] == successor
    else:
        assert "confirmation_attempt" not in result
        assert context[1].service.store.expire_leases(rid)["retried"] == 0
        assert pending["assignment_id"] + ":6" not in run_ids(context, pending)


@pytest.mark.parametrize("attempt", [4, 5, 36])
def test_lease_expiry_and_rematerialization_share_bound(scenario, attempt):
    context, rid, pending = prepare(scenario, attempt)
    lease = claim(context) if attempt > 5 else None
    upgrade(context)
    lease = lease or claim(context)
    assert lease["run_id"] == pending["run_id"]
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease["run_id"],))
    expired = context[1].service.store.expire_leases(rid)
    assert expired["expired"] == 1
    assert expired["retried"] == (1 if attempt == 4 else 0)
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert (pending["assignment_id"] + ":" + str(max(6, attempt + 1))) not in run_ids(context, pending)


@pytest.mark.parametrize("attempt", [6, 37])
def test_preexisting_oversized_pending_chain_cannot_be_claimed(scenario, attempt):
    context, _, pending = prepare(scenario, attempt)
    before = snapshots(context)
    upgrade(context)
    assert claim(context) == {"status": "no_pending"}
    assert snapshots(context) == before


@pytest.mark.parametrize("cause", ["accepted", "judge_error"])
def test_already_leased_attempt_36_can_settle_and_finish_after_upgrade(scenario, cause):
    context, rid, pending = prepare(scenario, 36)
    h = context[1]
    lease = claim(context)
    assert lease["run_id"] == pending["run_id"]
    token_hash = hash_lease_token(lease["lease_token"])
    identity = "sha256:" + "a" * 64
    # Admit an existing call before upgrade; its settlement remains valid.
    reserved = h.service.store.reserve_call(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity, operation_id="openrouter.chat", provider="openrouter", funding_source="host",
        amount_microusd=100, call_doc={})
    assert reserved["status"] == "reserved"
    assert h.service.store.mark_dispatched(run_id=lease["run_id"], lease_token_hash=token_hash, call_identity=identity)["status"] == "dispatched"
    upgrade(context)
    assert h.service.store.settle_call(run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity, actual_microusd=100, terminal_response={"call_succeeded": True, "status": 200})["status"] == "settled"
    result = h.service.store.complete_attempt(run_id=lease["run_id"], lease_token_hash=token_hash,
        result={"terminal_status": cause}, terminal_cause=cause,
        output_ref="accepted-output" if cause == "accepted" else None)
    assert result["status"] == ("accepted" if cause == "accepted" else "failed")
    assert "confirmation_attempt" not in result
    assert h.service.store.expire_leases(rid)["retried"] == 0
    assert pending["assignment_id"] + ":37" not in run_ids(context, pending)


def test_supersession_physical_attempt_36_gets_five_current_generation_attempts(scenario):
    context, rid, pending = prepare(scenario, 36, floor=1)
    upgrade(context)
    for number in range(36, 41):
        age(context, pending["assignment_id"] + ":" + str(number))
        lease = claim(context)
        assert lease["attempt"] == number
        result = fail(context, lease)
        if number < 40:
            assert result["confirmation_attempt"] == number + 1
        else:
            assert "confirmation_attempt" not in result
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert pending["assignment_id"] + ":41" not in run_ids(context, pending)
    assert claim(context) == {"status": "no_pending"}



def test_preupgrade_attempt_36_finishes_real_scorer_completion_and_publication(scenario):
    context, rid, pending = prepare(scenario, 36)
    h = context[1]
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        assert lease["run_id"] == pending["run_id"]
        upgrade(context)
        assert runner._slots.acquire(blocking=False)
        runner._run_lease(lease)
        assert h.service.store.get_run(lease["run_id"])["status"] == "accepted"
    finally:
        runner.close()
    h.advance_until("scored", runners=1)
    assert h.service.publish(rid)["status"] == "ok"
    assert h.status() == "published"


@pytest.mark.parametrize("cause", ["judge_error", "judge_timeout", "lease_expired", "stage_closed", "credential_error"])
def test_fifth_failed_terminal_never_rematerializes(scenario, cause):
    context, rid, pending = prepare(scenario, 5)
    mark_failed(context, pending, cause)
    upgrade(context)
    before = snapshots(context)
    assert context[1].service.store.expire_leases(rid)["retried"] == 0
    assert snapshots(context) == before

def test_publication_cutoff_and_successful_control_are_preserved(scenario):
    context, rid, pending = prepare(scenario, 5)
    upgrade(context)
    db, h, _, _ = context
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                    "'{schedule,publication_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s", (rid,))
    h.service._invalidate_hot_round()
    assert claim(context) == {"status": "no_pending"}
    # Restore the same unexpired frozen-fixture schedule and run the real
    # scorer/complete/publication route with the existing offline provider.
    context, rid, pending = prepare(scenario, 5)
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        assert lease["run_id"] == pending["run_id"]
        assert runner._slots.acquire(blocking=False)
        runner._run_lease(lease)
        assert h.service.store.get_run(lease["run_id"])["status"] == "accepted"
    finally:
        runner.close()
    h.advance_until("scored", runners=1)
    assert h.service.publish(rid)["status"] == "ok"
    assert h.status() == "published"


@pytest.mark.parametrize("drift", ["body", "acl"])
def test_exact_upgrade_fails_closed_for_unreviewed_function_shape(scenario, drift):
    context, _, _ = prepare(scenario, 5)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        if drift == "body":
            original = scalar(cur, "pg_get_functiondef(%s::regprocedure)", (SIGNATURES[0],))
            cur.execute(original.replace("BEGIN", "BEGIN\n  -- unexpected source", 1))
        else:
            cur.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURES[0] + " TO authenticated")
        with pytest.raises(db[0].Error, match="(preimage|security) differs"):
            cur.execute(SQL.read_text())
        conn.rollback()
