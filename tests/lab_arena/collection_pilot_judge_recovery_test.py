"""The owner can finish a missing pilot judge without changing or rerunning the paid generator."""

from pathlib import Path
from uuid import uuid4

import pytest
from lab_arena import store
from lab_arena.collection_launch_review import LaunchReviewBroker
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.locked_collection_pipeline_test import registered, claim
from tests.lab_arena.collection_batch_test import pilot


def test_owner_judge_can_recover_unavailable_pilot_without_changing_settlement():
    directory = Path(__file__).resolve().parents[2] / "scripts"
    numbers = (
        720,
        721,
        722,
        723,
        724,
        725,
        726,
        727,
        728,
        729,
        732,
        733,
        738,
        740,
        741,
        742,
        743,
        744,
        745,
        746,
        747,
        748,
        749,
    )
    migrations = tuple(next(directory.glob(f"{n}-*.sql")).name for n in numbers)
    for pg, dsn in database_with_lab_arena_migration(DEFAULT_MIGRATIONS + migrations):

        def connect():
            c = pg.connect(**dsn)
            c.autocommit = True
            return c

        db = store.PsycopgTransport(connect, role="lab_arena_service")
        miner = registered(db, "M")
        batch = "owner-judge-pilot"
        job = claim(miner, batch, "P1-s1001")
        rid = job["job_id"]
        genargs = dict(p_job=rid, p_lease=store.hash_lease_token(job["lease_token"]))
        for request, provider, maxfee, price, status in [
            ("generate-1", "openrouter", 1000, 100, 200),
            ("prior-owner-error", "judge", 9000, None, 429),
        ]:
            assert (
                db.rpc(
                    "collection_reserve_call_v1",
                    dict(
                        genargs,
                        p_request=request,
                        p_provider=provider,
                        p_hash="c" * 64,
                        p_amount=maxfee,
                    ),
                )["status"]
                == "reserved"
            )
            db.rpc(
                "collection_record_call_v1",
                dict(
                    genargs,
                    p_request=request,
                    p_hash="d" * 64,
                    p_provider_id="gen-" + request,
                    p_cost=price,
                    p_status=status,
                ),
            )
        original = "sha256:" + "a" * 64
        revised = "sha256:" + "e" * 64
        miner.complete(
            rid,
            job["lease_token"],
            "accepted",
            {"training_status": "cannot_judge"},
            original,
            7,
        )
        scope = {"p_job": rid, "p_source": original, "p_attempt": str(uuid4())}
        request = "collection-" + "3" * 64 + "-1"

        def reserve(amount):
            return db.rpc(
                "collection_reserve_launch_judge_v1",
                {**scope, "p_request": request, "p_hash": "d" * 64, "p_amount": amount},
            )

        assert reserve(job["budget_microusd"] - 9099)["status"] == "budget_exhausted"
        assert reserve(300)["status"] == "reserved"
        assert reserve(300)["status"] == "already_reserved"
        db.rpc(
            "collection_record_launch_judge_v1",
            {
                **scope,
                "p_request": request,
                "p_hash": "b" * 64,
                "p_provider_id": "gen-recovered",
                "p_cost": 120,
                "p_status": 200,
            },
        )
        args = dict(
            scope,
            p_artifact=revised,
            p_quality={"training_status": "accepted"},
            p_requests=[request],
        )
        assert (
            db.rpc("collection_finish_launch_review_v1", args)["artifact_sha256"]
            == revised
        )
        assert (
            db.rpc("collection_finish_launch_review_v1", args)["artifact_sha256"]
            == revised
        )
        rows = db.rpc("collection_batch_jobs_v1", {"p_batch": batch})
        assert len(rows) == 1
        assert (
            rows[0]["artifact_sha256"],
            rows[0]["pilot_review_sha256"],
            rows[0]["price_microusd"],
        ) == (original, revised, 0)
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute(
                "SELECT status,quality,artifact_sha256 FROM public.collection_jobs WHERE job_id=%s",
                (rid,),
            )
            assert q.fetchone() == (
                "accepted",
                {"training_status": "cannot_judge"},
                original,
            )
            q.execute(
                "SELECT count(*),sum(r.cost_microusd) FROM public.collection_calls c JOIN public.collection_call_receipts r USING(job_id,request_id) WHERE c.job_id=%s AND c.key_id IS NOT NULL",
                (rid,),
            )
            assert q.fetchone() == (1, 100)
        db.close()


@pytest.mark.parametrize("pilot", ["training-pack"], indirect=True)
def test_only_the_unavailable_vote_is_replayed_and_an_authentic_overlay_is_required(
    pilot, monkeypatch
):
    import json
    from copy import deepcopy
    from lab_arena import (
        collection_judge as j,
        collection_judge_review as review,
        collection_launch_review as recovery,
    )
    from lab_arena.collection_workflow import comparison_revisions, store_artifact
    from scripts import run_locked_collection as worker, collection_batch as cb
    from tests.lab_arena.collection_judge_test import PROMPT, pass_verdict

    raw, db, objects, bundle, receipt, key = pilot
    policy = worker.load_batch(raw, PROMPT)
    batch = policy.batch_id
    item = next(
        x
        for x in bundle["runs"]
        if len(x["assessment"]["records"]) == 2
        and "control_sha256" not in x
        and not any(r["cannot_judge"] for r in x["assessment"]["records"])
    )
    rid = item["packet"]["run_id"]
    job = next(x for x in db.rows if x["job_id"] == rid)
    full = deepcopy(item["assessment"])
    first, missing = full["records"]
    prior = dict(
        full,
        records=[first],
        status="cannot_judge",
        claim_review="missing",
        keep_steps=0,
        hard_steps=[],
        verifier_unavailable={
            "model": missing["judge_version"],
            "reason": "http_error",
            "request_id": "collection-" + "a" * 64 + "-1",
        },
    )
    orig = cb.read_artifact(
        objects, batch, rid, "manifest", {"sha256": job["artifact_sha256"]}, 1048576
    )
    prior_manifest = dict(
        orig, assessment=store_artifact(objects, batch, rid, "judge-output", prior)
    )
    job["artifact_sha256"] = store_artifact(
        objects, batch, rid, "manifest", prior_manifest
    )["sha256"]
    old_hash = job["artifact_sha256"]
    called = []

    def judge_one(self, **kw):
        called.append(kw["model"])
        assert kw["model"] == missing["judge_version"]
        self.completed_requests.append(kw["request_id"])
        identity = json.loads(kw["messages"][-1]["content"])["identity"]
        return json.dumps(pass_verdict(identity))

    monkeypatch.setattr(recovery.LaunchReviewBroker, "judge", judge_one)
    monkeypatch.setattr(worker, "endpoint_prices", lambda *a, **kw: {})

    def finish(database, pending, attempt, artifact, quality, requests):
        assert pending["artifact_sha256"] == old_hash and len(requests) == 1
        job["pilot_review_sha256"] = artifact
        return {"artifact_sha256": artifact}

    monkeypatch.setattr(recovery, "finish_review", finish)
    attempt = str(uuid4())
    worker.rejudge_ready(
        db, objects, None, raw, policy, PROMPT, "standin-owner", rid, attempt
    )
    worker.rejudge_ready(
        db, objects, None, raw, policy, PROMPT, "standin-owner", rid, attempt
    )
    assert called == [missing["judge_version"]]
    revised = cb.read_artifact(
        objects, batch, rid, "manifest", {"sha256": job["pilot_review_sha256"]}, 1048576
    )
    assessment = cb.read_artifact(
        objects, batch, rid, "judge-output", revised["assessment"], 1048576
    )
    assert (
        assessment["records"][0] == first
        and len(assessment["records"]) == 2
        and not assessment.get("verifier_unavailable")
    )
    assert (
        revised["judge_recovery"]["prior_manifest_sha256"] == old_hash
        and revised["generator"] == orig["generator"]
        and revised["packet"] == orig["packet"]
    )
    job["pilot_review_sha256"] = old_hash
    with pytest.raises(j.InvalidVerdict, match="overlay"):
        worker.rejudge_ready(
            db, objects, None, raw, policy, PROMPT, "standin-owner", rid, attempt
        )
