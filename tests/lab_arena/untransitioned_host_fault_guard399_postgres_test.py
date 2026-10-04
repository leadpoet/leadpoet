"""A host fault remains in the claim guard until its normal expiry transition."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.lab_arena import active_host_fault_guard397_postgres_test as prior397
from tests.lab_arena import execute_host_cooldown396_postgres_test as prior396
from tests.lab_arena import score_host_retry398_postgres_test as prior398
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior


ROOT = Path(__file__).parents[2]
SQL399 = ROOT / "scripts/399-lab-arena-untransitioned-host-fault-guard.sql"
LIVE_PREIMAGE = "6c977cc24ce500a76ba97e94ab0be50572d6395172109267c156d2de6f461cbc"
database = prior397.database


@pytest.fixture(scope="module")
def migrated(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for path, expected in (
                (prior396.SQL390, prior396.LIVE_390_PREIMAGE),
                (prior396.SQL396, prior396.LIVE_396_PREIMAGE),
                (prior397.SQL397, prior397.LIVE_PREIMAGE),
            ):
                before = prior396._security(cursor)
                old_hash = prior396._hash(cursor)
                sql = path.read_text()
                assert expected in sql
                sql = sql.replace(expected, old_hash)
                cursor.execute("BEGIN")
                with pytest.raises(psycopg.Error, match="preimage differs"):
                    cursor.execute(sql.replace(old_hash, "0" * 64, 1))
                cursor.execute("ROLLBACK")
                assert prior396._hash(cursor) == old_hash
                cursor.execute(sql)
                applied_hash = prior396._hash(cursor)
                cursor.execute(sql)
                assert prior396._hash(cursor) == applied_hash
                assert prior396._security(cursor) == before
            expiry_before = prior398._security(cursor)
            expiry_hash = prior398._hash(cursor, prior398.EXPIRY)
            sql398 = prior398.SQL398.read_text().replace(
                prior398.LIVE_398_PREIMAGE, expiry_hash
            )
            cursor.execute(sql398)
            cursor.execute(sql398)
            assert prior398._security(cursor) == expiry_before
            before = prior396._security(cursor)
            old_hash = prior396._hash(cursor)
            sql399 = SQL399.read_text()
            assert LIVE_PREIMAGE in sql399
            sql399 = sql399.replace(LIVE_PREIMAGE, old_hash)
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql399.replace(old_hash, "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert prior396._hash(cursor) == old_hash
            cursor.execute(sql399)
            applied_hash = prior396._hash(cursor)
            cursor.execute(sql399)
            assert prior396._hash(cursor) == applied_hash
            assert prior396._security(cursor) == before
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (prior396.SIGNATURE,))
            definition = cursor.fetchone()[0]
            assert definition.count("lab_arena_untransitioned_host_fault_guard_v1") == 1
            assert definition.count("lab_arena_active_host_fault_claim_guard_v1") == 1
    return True


def _set_deadline(conn, runs, *, age=1, transition=False):
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET lease_expires_at=now()-(%s::interval)"
            + (",status='failed',terminal_cause='lease_expired'" if transition else "")
            + " WHERE run_id=ANY(%s)",
            (f"{age} seconds", runs),
        )
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_two_active_one_expired_still_leased_blocks(database, migrated, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind=kind)
        _set_deadline(conn, faults[:1])
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        with conn.cursor() as cursor:
            cursor.execute("SELECT status FROM public.lab_arena_runs WHERE run_id=%s", (faults[0],))
            assert cursor.fetchone()[0] == "leased"
        assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh


def test_five_expired_still_leased_and_two_active_reproduces_live_gap(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind="score", positions=tuple(range(5)))
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            for index, letter in ((5, "J"), (6, "K")):
                submission = f"gap-score-sub-{index}"
                miner = "5" + letter * 47
                source = f"arena/{prior.ROUND}/sources/{submission}.tar.gz"
                assignment = f"{prior.ROUND}:{submission}:1:{index}"
                run = assignment + ":1"
                cursor.execute(
                    "INSERT INTO public.lab_arena_submissions("
                    "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
                    "source_ref,source_size_bytes,consent,frozen_at) "
                    "SELECT %s,round_id,%s,status,FALSE,"
                    "jsonb_set(submission_doc,'{source_ref}',to_jsonb(%s::text)),"
                    "%s,source_size_bytes,consent,frozen_at "
                    "FROM public.lab_arena_submissions WHERE submission_id=%s",
                    (submission, miner, source, source, prior.SUBMISSION),
                )
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                    "lease_generation,stage_generation,runner_hotkey,lease_token_hash,lease_expires_at) "
                    "VALUES (%s,%s,%s,%s,%s,1,%s,1,'score','leased',1,1,%s,%s,"
                    "now()+interval '10 minutes')",
                    (run, assignment, prior.ROUND, submission, miner, index,
                     prior.RUNNER_A, "sha256:" + "a" * 64),
                )
                prior396._event(cursor, run)
                faults.append(run)
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        _set_deadline(conn, faults[:5])
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_guard_has_no_gap_during_phased_expiry_transition(database, migrated, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind=kind)
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        for run in faults:
            _set_deadline(conn, [run])
            assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
            _set_deadline(conn, [run], transition=True)
            assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        _set_deadline(conn, faults[:1], age=421, transition=True)
        assert prior._claim(conn, prior.RUNNER_A, "c")["run_id"] == fresh


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_untransitioned_fault_ages_out_after_frozen_ttl(database, migrated, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind=kind)
        _set_deadline(conn, faults[:1], age=421)
        assert prior._claim(conn, prior.RUNNER_A, "a")["run_id"] == fresh


def test_2560_pending_jobs_keep_blocked_claim_bounded(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind="execute")
        _set_deadline(conn, faults[:1])
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "lease_generation,stage_generation) "
                "SELECT 'gap-load-'||i,'gap-load-'||i,%s,%s,%s,1,i%%10,1,"
                "'execute','pending',0,1 FROM generate_series(1,2560) AS i",
                (prior.ROUND, prior.SUBMISSION, prior.MINER),
            )
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        blocked = time.monotonic() - start
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_B, "b")["status"] == "leased"
        healthy = time.monotonic() - start
        assert blocked < 5.0 and healthy < 5.0
        print(f"399 2560-job claim seconds: blocked={blocked:.3f}, healthy={healthy:.3f}")
