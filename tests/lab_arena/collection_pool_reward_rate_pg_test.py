"""Exercise the rate change across real enrollments, assignments, native bills and epoch quotes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest

from tests.lab_arena.collection_standing_pool_pg_test import db, digest, pool


ORIGINAL_TERMS = (
    "sha256:1655fcba4ba721aa1559c073e1430853450b1771ebf2dea7c74c5a74c4768df1"
)
NEW_TERMS = "sha256:29c76e6f5d945c6e68821a331423395a3445d37c2f9d8943dab79433e8b608bc"


def migrate(p):
    # Exercise the same unprivileged membership and schema grants as the RDS migrator.
    with p.driver.connect(**p.dsn) as admin, admin.cursor() as cur:
        cur.execute("CREATE ROLE rate_test_migrator NOINHERIT")
        cur.execute(
            "GRANT lab_arena_owner TO rate_test_migrator WITH INHERIT TRUE, SET TRUE"
        )
        cur.execute(
            "GRANT USAGE, CREATE ON SCHEMA public TO rate_test_migrator WITH GRANT OPTION"
        )
    with p.driver.connect(
        **p.dsn, application_name="standing-rate-772-test"
    ) as conn, conn.cursor() as cur:
        cur.execute("SET ROLE rate_test_migrator")
        sql = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "772-standing-pool-reward-rate.sql"
        )
        cur.execute(sql.read_text())


def test_assignment_rate_survives_cutover_retry_and_settlement(pool):
    p = pool
    p.activate()
    original = p.register(0, 0, 200_000, alter={"terms_sha256": ORIGINAL_TERMS})
    fixed_quote = p.rpc("reward_quotes", 101)
    token = "sha256:" + digest("rate-cutover")

    def claim(batch):
        return p.claim(batch, token=token)

    def bill(job, name, cost):
        assert p.reserve(job, token, name, 999) == "reserved"
        p.record(job, token, name, cost, "gen-" + name)

    old_done = claim("existing-closed")
    bill(old_done, "already-settled", 101)
    p.finish(old_done, token, passed=False, epoch=201)
    pending_basis = p.rpc("reward_quotes", 202)
    old_rows = p.rpc("batch_jobs", old_done["batch_id"])
    old_statement = p.rpc("statement", original[1], "live")["jobs"][0]
    assert pending_basis["quotes_microusd"] == {original[1]: 127}

    old_inflight = [claim("inflight-pass"), claim("inflight-fail")]
    bill(old_inflight[0], "dispatch-before-cutover", 101)
    migrate(p)
    assert p.rpc("pool_reward_rate_bps") == 15_000
    assert p.rpc("reward_quotes", 202) == pending_basis
    assert p.rpc("reward_quotes", 101) == fixed_quote
    for role in ("anon", "authenticated", "service_role"):
        assert not p.sql(
            "SELECT has_function_privilege(%s,'public.collection_pool_reward_rate_bps_v1()', 'EXECUTE')",
            (role,),
        )
    assert p.rpc("batch_jobs", old_done["batch_id"]) == old_rows
    assert old_statement in p.rpc("statement", original[1], "live")["jobs"]
    assert p.rpc("register", *original)["registration_id"] == original[0]
    other_old_enrollment = p.enrollment(
        3, 1, 20_000, alter={"terms_sha256": ORIGINAL_TERMS}
    )
    assert (
        p.rpc("register", *other_old_enrollment)["error"]
        == "collection_consent_invalid"
    )
    for kwargs in (
        {"terms_sha256": NEW_TERMS, "spend_limit_microusd": 50_000},
        {"terms_sha256": NEW_TERMS, "priority": 9},
    ):
        assert (
            p.rpc("register", *p.enrollment(3, 1, 20_000, alter=kwargs))["error"]
            == "collection_consent_invalid"
        )

    assert (
        p.claim("inflight-pass", job=old_inflight[0]["job_id"], token=token)
        == old_inflight[0]
    )
    assert all("reward_rate_bps" not in job for job in old_inflight)
    bill(old_inflight[1], "dispatch-after-cutover", 101)
    for i, job in enumerate(old_inflight):
        assert p.finish(job, token, passed=i == 0, epoch=202)["status"] == "accepted"
        assert p.rpc("batch_jobs", job["batch_id"])[0]["price_microusd"] == 127

    # Existing grant, same cap and manifest, new independent job at the new rate.
    new_on_old = claim("existing-closed")
    assert (
        new_on_old["registration_id"] == original[0]
        and new_on_old["reward_rate_bps"] == 15_000
    )
    assert (
        p.claim("existing-closed", job=new_on_old["job_id"], token=token) == new_on_old
    )
    fresh = p.register(1, 1, 60_000, alter={"terms_sha256": NEW_TERMS})
    new_on_new = claim("existing-closed")
    assert (
        new_on_new["registration_id"] == fresh[0]
        and new_on_new["reward_rate_bps"] == 15_000
    )
    for granted, job, passed in (
        (original, new_on_old, False),
        (fresh, new_on_new, True),
    ):
        assert p.reserve(job, token, "generator", 999) == "reserved"
        # A stale worker cannot set its own rate and a pending receipt cannot earn a credit.
        with pytest.raises(p.driver.Error, match="immutable"):
            p.sql(
                "UPDATE public.collection_jobs SET reward_rate_bps=NULL WHERE job_id=%s",
                (job["job_id"],),
                owner=True,
            )
        with pytest.raises(p.driver.Error, match="missing verified collection funding"):
            p.finish(job, token, passed=passed, epoch=202)
        p.record(job, token, "generator", 101, "gen-" + job["job_id"])
        if not passed:
            # Two 101-microdollar calls earn 303 once on their sum, not 152 twice.
            bill(job, "second-generator", 101)
        assert p.finish(job, token, passed=passed, epoch=202)["status"] == "accepted"
        p.finish(
            job, token, passed=passed, epoch=203
        )  # Replay does not create a later credit.
        rows = p.rpc("statement", granted[1], "live")["jobs"]
        mine = next(row for row in rows if row["job_id"] == job["job_id"])
        assert (mine["price_microusd"], mine["reward_rate_bps"]) == (
            152 if passed else 303,
            15_000,
        )

    # Reconciliation after an interrupted run releases its reservation without creating a reward.
    incomplete = claim("interrupted")
    assert incomplete["reward_rate_bps"] == 15_000
    assert p.reserve(incomplete, token, "g", 500) == "reserved"
    p.record(incomplete, token, "g", None, "gen-interrupted")
    with pytest.raises(p.driver.Error, match="missing verified collection funding"):
        p.finish(incomplete, token, epoch=202)
    assert p.finish(incomplete, token, status="failed", epoch=202)["status"] == "failed"
    assert (
        p.rpc(
            "settle_generation",
            incomplete["job_id"],
            "g",
            "gen-interrupted",
            digest("gresponse"),
            digest("verified-late-bill"),
            101,
        )["status"]
        == "settled"
    )
    assert p.rpc("batch_jobs", incomplete["batch_id"])[0]["price_microusd"] == 0

    mixed_batch = {
        row["job_id"]: row for row in p.rpc("batch_jobs", old_done["batch_id"])
    }
    assert mixed_batch[old_done["job_id"]] == old_rows[0]
    assert mixed_batch[new_on_old["job_id"]]["reward_rate_bps"] == 15_000
    assert mixed_batch[new_on_new["job_id"]]["reward_rate_bps"] == 15_000
    assert p.rpc("reward_quotes", 202) == pending_basis
    assert p.rpc("reward_quotes", 203)["quotes_microusd"] == {
        original[1]: 557,
        fresh[1]: 152,
    }
    assert p.rpc("reward_quotes", 204)["job_count"] == 0
    assert p.rpc("reward_quotes", 101) == fixed_quote
    assert (
        p.rpc("statement", original[1], "live")["enrollments"][0]["available_microusd"]
        == 200_000 - 5 * 101
    )
    assert (
        p.rpc("statement", fresh[1], "live")["enrollments"][0]["available_microusd"]
        == 60_000 - 2 * 101
    )


def test_claim_waiting_at_cutover_receives_immutable_new_rate(pool):
    p = pool
    p.activate()
    p.register(0, 1, 30_000, alter={"terms_sha256": ORIGINAL_TERMS})
    first = p.claim("locked-boundary")

    with ThreadPoolExecutor(max_workers=2) as threads, p.driver.connect(
        **p.dsn
    ) as conn:
        with conn.cursor() as cur:
            # Represent a claim transaction already past the production dispatch lock.
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0))"
            )
            migration = threads.submit(migrate, p)
            until = time.monotonic() + 10
            while True:
                cur.execute("SELECT pg_stat_clear_snapshot()")
                cur.execute(
                    """SELECT EXISTS(SELECT FROM pg_stat_activity WHERE datname=current_database()
                               AND application_name='standing-rate-772-test' AND wait_event='advisory')"""
                )
                if cur.fetchone()[0]:
                    break
                if time.monotonic() >= until:
                    if migration.done():
                        migration.result()
                    pytest.fail(
                        "migration did not wait on the existing claim transaction"
                    )
                time.sleep(0.01)
            next_claim = threads.submit(p.claim, "locked-boundary")
            conn.commit()
            migration.result(timeout=8)
            second = next_claim.result(timeout=8)
    assert first["job_id"] != second["job_id"]
    assert [j.get("reward_rate_bps", 12500) for j in (first, second)] == [12500, 15000]
    assert [
        j.get("reward_rate_bps", 12500) for j in p.rpc("batch_jobs", first["batch_id"])
    ] == [12500, 15000]
