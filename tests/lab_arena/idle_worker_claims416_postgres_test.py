"""Current PostgreSQL claim policy uses free slots without splitting ownership."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery
from tests.lab_arena import execute_host_cooldown396_postgres_test as identity
from tests.lab_arena import proxy_model_capacity401_postgres_test as historical
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as retry
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete


ROOT = Path(__file__).parents[2]
SQL = ROOT / "scripts/416-lab-arena-idle-worker-claims.sql"
PREHASH = "59a5e9dc26f9e3d5157beb251f55c08db46e020dd30103c8a38db18ee18d1992"
POSTHASH = "836d8a6594e8c61dca9db084a5311f2a5bce86c1f94b137768aa07c7f0aba467"
database = recovery.database


@pytest.fixture(scope="module")
def migrated(database, tmp_path_factory):
    # Keep migration 401's fixture historical. Install every subsequent current
    # Arena migration here before testing the exact deployed claim preimage.
    recovery.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for number in (404, 413, 414, 415):
                paths = list((ROOT / "scripts").glob(f"{number}-*.sql"))
                assert len(paths) == 1
                cur.execute(paths[0].read_text())
            before = identity._security(cur)
            assert identity._hash(cur) == PREHASH
            h, frozen = _stage2(lambda: psycopg.connect(**dsn),
                                tmp_path_factory.mktemp("pre416"), "pre416")
            old_leases = [_claim(h) for _ in range(10)]
            assert all(r["status"] == "leased" for r, _ in old_leases)
            _finish(h, old_leases[0])
            assert _claim(h)[0] == {"status": "no_pending"}
            for bad_sql, message in (
                (SQL.read_text().replace(PREHASH, "0" * 64, 1), "preimage differs"),
                (SQL.read_text().replace(POSTHASH, "0" * 64, 1), "postimage differs"),
            ):
                with pytest.raises(psycopg.Error, match=message):
                    cur.execute(bad_sql)
                cur.execute("ROLLBACK")
                assert identity._hash(cur) == PREHASH
            cur.execute(SQL.read_text())
            assert identity._hash(cur) == POSTHASH
            freed_slot = _claim(h)[0]
            assert freed_slot["status"] == "leased"
            assert freed_slot["submission_id"] != old_leases[0][0]["submission_id"]
            assert h.service.store.get_round(h.round_id)["configuration_doc"] == frozen["configuration_doc"]
            cur.execute(SQL.read_text())
            assert identity._hash(cur) == POSTHASH
            assert identity._security(cur) == before
            # A replay must still reject a changed owner or public ACL.
            for mutation, restoration in (
                (f"ALTER FUNCTION {identity.SIGNATURE} OWNER TO postgres",
                 f"ALTER FUNCTION {identity.SIGNATURE} OWNER TO lab_arena_owner"),
                (f"GRANT EXECUTE ON FUNCTION {identity.SIGNATURE} TO PUBLIC",
                 f"REVOKE EXECUTE ON FUNCTION {identity.SIGNATURE} FROM PUBLIC"),
            ):
                cur.execute(mutation)
                with pytest.raises(psycopg.Error, match="security shape differs"):
                    cur.execute(SQL.read_text())
                cur.execute("ROLLBACK")
                cur.execute(restoration)
            assert identity._security(cur) == before
    return True


@pytest.fixture(scope="module")
def connect(database, migrated):
    return lambda: database[0].connect(**database[1])


def _stage2(connect, tmp_path, label, *, ceiling=251, miners=3, runners=("alpha",)):
    harness = historical._start(
        connect, tmp_path, label.replace("-", ""), icps=10, ceiling=ceiling,
        challengers=tuple(f"SlotMiner{i}" for i in range(miners)), runners=runners,
    )
    # Isolate SQL queue behavior with stage-2 pending assignments. Only this
    # disposable test fixture bypasses frozen-row triggers. The separate signed
    # HTTP end-to-end test proves the real baseline-first stage transitions.
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute(
            "DELETE FROM public.lab_arena_runs WHERE round_id=%s AND submission_id="
            "(SELECT p->>'submission_id' FROM public.lab_arena_rounds r,"
            "jsonb_array_elements(r.participants) p WHERE r.round_id=%s "
            "AND (p->>'is_king')::boolean)", (harness.round_id, harness.round_id),
        )
        cur.execute(
            "UPDATE public.lab_arena_rounds SET status='stage2',configuration_doc="
            "(configuration_doc-'parallel_twenty_icp_execution') || "
            "jsonb_build_object('execution_sequence_policy','baseline_scored_first_v1') "
            "WHERE round_id=%s", (harness.round_id,),
        )
        cur.execute("UPDATE public.lab_arena_runs SET stage=2 WHERE round_id=%s",
                    (harness.round_id,))
    frozen = harness.service.store.get_round(harness.round_id)
    assert frozen["configuration_doc"]["stage_1_icp_count"] == 5
    assert frozen["configuration_doc"]["stage_2_icp_count"] == 5
    assert "parallel_twenty_icp_execution" not in frozen["configuration_doc"]
    return harness, frozen


def _claim(harness, slots=10, runner=0, ceiling=251):
    return claim(harness.service.store, harness.round_id,
                 harness.runner_keys[runner], parallelism=slots, ceiling=ceiling)[:2]


def _finish(harness, lease):
    run, token = lease
    assert complete(harness.service.store, run["run_id"], hash_lease_token(token),
                    "accepted", output_ref="arena/test/" + run["run_id"])["status"] == "accepted"


def test_idle_tail_uses_free_slots_and_finishes_older_pending_model_first(connect, tmp_path):
    h, frozen = _stage2(connect, tmp_path, "idle-tail")
    first = [_claim(h) for _ in range(10)]
    assert {r["icp_position"] for r, _ in first} == set(range(10))
    assert len({r["submission_id"] for r, _ in first}) == 1
    assert _claim(h)[0] == {"status": "no_free_slot", "active_leases": 10, "slot_limit": 10}
    for lease in first[:-1]:
        _finish(h, lease)
    second = [_claim(h) for _ in range(9)]
    assert all(r["status"] == "leased" for r, _ in second)
    assert {r["icp_position"] for r, _ in second} == set(range(9))
    assert len({r["submission_id"] for r, _ in second}) == 1
    assert second[0][0]["submission_id"] != first[0][0]["submission_id"]
    _finish(h, second[0])
    older_pending = _claim(h)
    assert older_pending[0]["submission_id"] == second[0][0]["submission_id"]
    assert older_pending[0]["icp_position"] == 9
    # More than two model tails must not recreate another arbitrary model cap.
    _finish(h, second[1])
    third = _claim(h)
    assert third[0]["status"] == "leased"
    assert third[0]["submission_id"] not in {
        first[0][0]["submission_id"], second[0][0]["submission_id"]}
    after = h.service.store.get_round(h.round_id)
    for key in ("configuration_doc", "participants", "benchmark_ref", "benchmark_hash"):
        assert after.get(key) == frozen.get(key)


@pytest.mark.parametrize("slots,ceiling", [(3, 251), (10, 10), (19, 20)])
def test_partial_capacity_and_frozen_ceiling_stay_physical(connect, tmp_path, slots, ceiling):
    h, _ = _stage2(connect, tmp_path, f"partial-{slots}", ceiling=ceiling)
    leases = [_claim(h, slots, ceiling=ceiling) for _ in range(slots)]
    assert all(r["status"] == "leased" for r, _ in leases)
    assert _claim(h, slots, ceiling=ceiling)[0]["status"] == "no_free_slot"
    _finish(h, leases[0])
    assert _claim(h, slots, ceiling=ceiling)[0]["status"] == "leased"
    assert _claim(h, 1, ceiling=ceiling)[0]["status"] == "no_free_slot"


def test_concurrent_claims_serialize_at_physical_limit(connect, tmp_path):
    h, _ = _stage2(connect, tmp_path, "concurrent")
    start = threading.Barrier(16)
    def take(_):
        start.wait(timeout=10)
        return _claim(h, 3)[0]
    with ThreadPoolExecutor(max_workers=16) as pool:
        responses = list(pool.map(take, range(16)))
    leased = [r for r in responses if r["status"] == "leased"]
    assert len(leased) == 3
    assert len({r["run_id"] for r in leased}) == 3
    assert all(r["status"] == "no_free_slot" for r in responses if r not in leased)


def test_cross_runner_initial_ownership_and_retry_handoff_remain(connect, tmp_path):
    h, _ = _stage2(connect, tmp_path, "ownership", runners=("alpha", "beta"))
    a = _claim(h, 3)
    b = _claim(h, 3, runner=1)
    assert a[0]["submission_id"] != b[0]["submission_id"]
    next_a = _claim(h, 3)
    assert next_a[0]["submission_id"] == a[0]["submission_id"]
    assert next_a[0]["icp_position"] == 1
    # Retain the existing authenticated retry guard; it does not become a new
    # initial-ownership claim merely because model admission is less strict.
    with connect() as conn:
        retry._seed(conn, retry._result("model_error", 0), "model_error", kind="execute")
        assert retry._claim(conn, retry.RUNNER_A, "a")["status"] == "no_pending"
        assert retry._claim(conn, retry.RUNNER_B, "b")["status"] == "leased"


def test_host_fault_guard_is_unchanged(database, migrated):
    recovery.test_recovery_hands_off_without_ttl_wait_and_preserves_audit(database, migrated, "execute")
