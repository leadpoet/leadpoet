"""Owner-only rejudgment of completed launch evidence."""

from uuid import uuid4

import pytest
from psycopg2.extras import Json

from lab_arena.collection_broker import CollectionCallError, VERSIONS
from lab_arena.collection_launch_review import LaunchReviewBroker


def verify_recovery_contract(q, pg, job, source, budget):
    """Exercise the real staged enrollment/claim fixture before its first cutover."""
    attempt, other = str(uuid4()), str(uuid4())
    rid = "collection-" + "a" * 64 + "-1"
    rid2 = "collection-" + "b" * 64 + "-1"
    out = "sha256:" + "c" * 64
    good = Json({"training_status": "accepted"})
    payload = "a" * 64
    reserve = "SELECT public.collection_reserve_launch_judge_v1(%s,%s,%s,%s,%s,%s)"
    record = "SELECT public.collection_record_launch_judge_v1(%s,%s,%s,%s,%s,%s,%s,%s)"
    finish = "SELECT public.collection_finish_launch_review_v1(%s,%s,%s,%s,%s,%s)"
    q.execute("SET LOCAL ROLE lab_arena_service")
    q.execute(reserve, (job, source, attempt, rid, payload, budget))
    assert q.fetchone()[0] == {"status": "budget_exhausted"}
    q.execute(reserve, (job, source, attempt, rid, payload, 29))
    assert q.fetchone()[0] == {"status": "reserved"}
    q.execute(reserve, (job, source, attempt, rid, payload, 29))
    assert q.fetchone()[0] == {"status": "already_reserved"}
    q.execute("SAVEPOINT owner_only")
    with pytest.raises(pg.Error, match="already in progress|in progress"):
        q.execute(reserve, (job, source, other, rid, payload, 1))
    q.execute("ROLLBACK TO owner_only")
    with pytest.raises(pg.Error, match="verified owner judgment"):
        q.execute(finish, (job, source, attempt, out, good, Json([rid])))
    q.execute("ROLLBACK TO owner_only")
    q.execute(record, (job, source, attempt, rid, payload, "gen-owner-429", None, 429))
    assert q.fetchone()[0]["status"] == "recorded"
    with pytest.raises(pg.Error, match="verified owner judgment"):
        q.execute(finish, (job, source, attempt, out, good, Json([rid])))
    q.execute("ROLLBACK TO owner_only")
    # The unknown cost may ultimately exceed its original reservation.
    q.execute(record, (job, source, attempt, rid, payload, "gen-owner-429", None, 429))
    q.execute(reserve, (job, source, attempt, rid2, payload, 1))
    held = q.fetchone()[0]
    assert held == {
        "status": "unresolved_prior_charge",
        "judge_receipts": [
            {"request_id": attempt + ":" + rid, "generation_id": "gen-owner-429"}
        ],
    }
    reconcile = "SELECT public.collection_reconcile_launch_judge_v1(%s,%s,%s,%s,%s,%s)"
    q.execute("SAVEPOINT unresolved")
    with pytest.raises(pg.Error, match="generation does not match"):
        q.execute(
            reconcile, (job, source, attempt + ":" + rid, "unrelated", 0, payload)
        )
    q.execute("ROLLBACK TO unresolved")
    q.execute(
        reconcile, (job, source, attempt + ":" + rid, "gen-owner-429", 105, payload)
    )
    assert q.fetchone()[0] == {"status": "reconciled"}
    q.execute(reserve, (job, source, attempt, rid2, payload, budget - 100 - 104))
    assert q.fetchone()[0] == {"status": "budget_exhausted"}
    q.execute("RESET ROLE")
    q.execute(
        "UPDATE public.collection_launch_reviews SET expires_at=clock_timestamp()-interval '1 minute' "
        "WHERE job_id=%s AND attempt_id=%s",
        (job, attempt),
    )
    q.execute("SET LOCAL ROLE lab_arena_service")
    q.execute("SAVEPOINT expired")
    with pytest.raises(pg.Error, match="expired"):
        q.execute(reserve, (job, source, attempt, rid2, payload, 1))
    q.execute("ROLLBACK TO expired")
    q.execute(reserve, (job, source, other, rid2, payload, 15))
    assert q.fetchone()[0] == {"status": "reserved"}
    q.execute(
        record,
        (job, source, other, rid2, payload, "confirmed-owner-generation", 12, 200),
    )
    assert q.fetchone()[0]["status"] == "recorded"
    final_args = (job, source, other, out, good, Json([rid2]))
    q.execute(finish, final_args)
    assert q.fetchone()[0]["artifact_sha256"] == out
    q.execute(finish, final_args)
    assert q.fetchone()[0]["artifact_sha256"] == out
    q.execute("SAVEPOINT immutable")
    with pytest.raises(pg.Error, match="conflicting launch review"):
        q.execute(
            finish, (job, source, other, "sha256:" + "d" * 64, good, Json([rid2]))
        )
    q.execute("ROLLBACK TO immutable")
    q.execute("RESET ROLE")
    q.execute(
        "SELECT c.provider,c.key_id,c.reserved_microusd FROM public.collection_calls c "
        "WHERE job_id=%s AND request_id LIKE '%%:collection-%%' ORDER BY c.created_at",
        (job,),
    )
    assert q.fetchall() == [("judge", None, 29), ("judge", None, 15)]
    q.execute(
        "SELECT j.status,j.artifact_sha256,r.source_sha256,r.artifact_sha256 FROM public.collection_jobs j "
        "JOIN public.collection_launch_reviews r USING(job_id) WHERE j.job_id=%s AND r.attempt_id=%s",
        (job, other),
    )
    assert q.fetchone() == ("ready", out, source, out)
    q.execute("SELECT public.collection_reward_start_v1()")
    assert q.fetchone()[0] == {"first_epoch": None}


def test_recovery_broker_never_uses_a_miner_key_or_normal_lease():
    pending = {
        "job_id": str(uuid4()),
        "artifact_sha256": "sha256:" + "a" * 64,
        "model": next(iter(VERSIONS)),
    }

    class DB:
        def __init__(self):
            self.calls = []

        def rpc(self, name, args):
            self.calls.append((name, args))
            return {"status": "budget_exhausted"}

    db = DB()
    attempt = str(uuid4())
    broker = LaunchReviewBroker(
        db,
        None,
        pending,
        {},
        routes=dict.fromkeys(VERSIONS, "fixture"),
        judge_key="owner-test-key",
        attempt_id=attempt,
    )
    assert broker.intake is None and "lease_token" not in broker.job
    with pytest.raises(CollectionCallError, match="only owner-funded"):
        broker._call("fixture", "openrouter", "", b"", 1)
    assert not db.calls
    request = "collection-" + "b" * 64 + "-1"
    got = broker._rpc(
        "collection_reserve_call_v1",
        request,
        p_provider="judge",
        p_hash="b" * 64,
        p_amount=2,
    )
    assert got == {"status": "budget_exhausted"}
    assert db.calls == [
        (
            "collection_reserve_launch_judge_v1",
            {
                "p_job": pending["job_id"],
                "p_attempt": attempt,
                "p_source": pending["artifact_sha256"],
                "p_request": request,
                "p_hash": "b" * 64,
                "p_amount": 2,
            },
        )
    ]


@pytest.mark.parametrize(
    "response, status, expected",
    [
        ({"data": {"id": "gen-original", "total_cost": 0}}, 200, 0),
        ({"data": {"id": "gen-original", "total_cost": 0.000105}}, 200, 105),
        ({"data": {"id": "gen-other", "total_cost": 0}}, 200, None),
        ({"data": {"id": "gen-original"}}, 200, None),
        ({"data": {"id": "gen-original", "total_cost": 0}}, 429, None),
    ],
)
def test_only_the_original_authenticated_generation_receipt_can_release_a_hold(
    response, status, expected
):
    import json
    from types import SimpleNamespace

    pending = {"job_id": str(uuid4()), "artifact_sha256": "sha256:" + "a" * 64}
    records, reads = [], []
    generation_id = "gen-original"

    class Database:
        def rpc(self, name, arguments):
            if name == "collection_reconcile_launch_judge_v1":
                records.append(arguments)
                return {"status": "reconciled"}
            assert name == "collection_reserve_launch_judge_v1"
            if records:
                return {"status": "reserved"}
            return {
                "status": "unresolved_prior_charge",
                "judge_receipts": [
                    {"generation_id": generation_id, "request_id": "earlier"}
                ],
            }

    class Readback:
        def send(self, **kwargs):
            assert kwargs["method"] == "GET"
            assert (
                kwargs["url"]
                == "https://openrouter.ai/api/v1/generation?id=gen-original"
            )
            assert (
                kwargs["headers"] == {"authorization": "Bearer owner-fixture"}
                and not kwargs["body"]
            )
            reads.append(kwargs["url"])
            return SimpleNamespace(status=status, body=json.dumps(response).encode())

    meter = LaunchReviewBroker(
        Database(),
        Readback(),
        pending,
        {},
        routes=dict.fromkeys(VERSIONS, "fixture"),
        judge_key="owner-fixture",
        attempt_id=uuid4(),
    )
    result = meter._rpc(
        "collection_reserve_call_v1",
        "collection-" + "f" * 64 + "-1",
        p_provider="judge",
        p_hash="a" * 64,
        p_amount=15,
    )
    assert len(reads) == 1
    if expected is None:
        assert result["status"] == "unresolved_prior_charge" and not records
    else:
        assert result["status"] == "reserved"
        assert len(records) == 1 and records[0]["p_cost"] == expected
        assert (
            records[0]["p_job"] == pending["job_id"]
            and records[0]["p_generation"] == generation_id
        )
