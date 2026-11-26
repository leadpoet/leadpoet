"""A real fixed quote can settle while the separate judge study is incomplete."""

from pathlib import Path
import time
from uuid import uuid4

from psycopg2.extras import Json
import pytest

from lab_arena.collection_intake import MODELS, PAID_WORKFLOW_TERMS_HASH, TASK_SCOPE
from tests.lab_arena.collection_rewards_test import MINER, insert_legacy_state
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)


@pytest.fixture
def db():
    root = Path(__file__).resolve().parents[2] / "scripts"
    migrations = [
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
        750,
        751,
        752,
        753,
        754,
        755,
    ]
    yield from database_with_lab_arena_migration(
        (
            *DEFAULT_MIGRATIONS,
            *(next(root.glob(f"{number}-*.sql")).name for number in migrations),
        )
    )


def test_paid_offer_through_first_epoch(db):
    pg, dsn = db
    commit, artifact, token = "d" * 40, "sha256:" + "b" * 64, "sha256:" + "d" * 64
    source, offer = "measured-p7", "dv-" + "c" * 64
    model = MODELS[1]
    reg, key, job, live_job, live_key = (str(uuid4()) for _ in range(5))
    with pg.connect(**dsn) as conn, conn.cursor() as q:

        def rpc(name, *args):
            q.execute(
                "SELECT public." + name + "(" + ",".join(["%s"] * len(args)) + ")", args
            )
            return q.fetchone()[0]

        def bill(job_id, key_id, cost, request="generate", provider="openrouter"):
            q.execute(
                "INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd) "
                "VALUES(%s,%s,%s,%s,%s,20000)",
                (job_id, request, provider, key_id, "1" * 64),
            )
            q.execute(
                "INSERT INTO public.collection_call_receipts VALUES (%s,%s,%s,%s,%s,200)",
                (job_id, request, "3" * 64, job_id + request, cost),
            )

        prior = insert_legacy_state(q, 100)
        q.execute(
            "INSERT INTO public.collection_champion_retirement(first_epoch,previous_state_hash) VALUES (101,%s)",
            (prior["state_hash"],),
        )
        q.execute(
            "INSERT INTO public.collection_batches VALUES (%s,'pilot',%s,%s,%s,'{}','{}')",
            (source, commit, Json(["P7-s40401"]), Json(MODELS)),
        )
        q.execute(
            "INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent) "
            "VALUES (%s,%s,%s,'pilot','{}')",
            (reg, MINER, MINER),
        )
        q.execute(
            "INSERT INTO public.collection_keys(key_id,registration_id,provider,miner_hotkey,submission_id,"
            "daily_cap_microusd,fingerprint,ciphertext_b64) VALUES (%s,%s,'openrouter',%s,'pilot',500000,%s,'mock')",
            (key, reg, MINER, "sha256:" + "1" * 64),
        )
        q.execute(
            """INSERT INTO public.collection_jobs(job_id,batch_id,mode,registration_id,miner_hotkey,openrouter_key_id,
                     slot,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,lease_token_hash,
                     lease_expires_at,status,accepted_epoch,artifact_sha256,finished_at)
                     VALUES (%s,%s,'pilot',%s,%s,%s,0,current_date-1,'P7-s40401',%s,%s,20000,0,%s,now(),
                       'accepted',99,%s,now())""",
            (job, source, reg, MINER, key, model, commit, token, artifact),
        )
        bill(job, key, 6537)
        bill(job, None, 900000, request="judge", provider="judge")
        measured = rpc("collection_verified_p7_fees_v1", source, Json([job]))
        assert measured["prices"] == {"P": {model: 8172}}
        assert measured["budgets"] == {"P": {model: 20000}}
        q.execute("SAVEPOINT sample")
        for observations in ([job, job], [str(uuid4())]):
            with pytest.raises(pg.Error, match="repeated, incomplete or unsettled"):
                rpc("collection_verified_p7_fees_v1", source, Json(observations))
            q.execute("ROLLBACK TO sample")
        bill(job, key, None, "unresolved")
        with pytest.raises(pg.Error, match="unsettled"):
            rpc("collection_verified_p7_fees_v1", source, Json([job]))
        q.execute("ROLLBACK TO sample")
        anchor = {
            "epoch": 100,
            "state_hash": prior["state_hash"],
            "head_block": 25,
            "head_hash": "0x" + "a" * 64,
            "observed_at": int(time.time()),
        }
        approval = {
            **measured,
            "payment_rule": "verified_p7_v1",
            "schema": "leadpoet.collection.verified_p7_prices.v1",
            "live_batch_id": offer,
            "batch_id": source,
            "measurement_jobs": [job],
            "capture_version": "collection.workflow-events.v2",
            "routes": {model: "coreweave/fp8"},
            "terms_sha256": PAID_WORKFLOW_TERMS_HASH,
            "harness_commit": commit,
            "builder_commit": "2" * 40,
            "split_manifest_sha256": "d" * 64,
            "judge_prompt_sha256": "1" * 64,
            "cutover_anchor": anchor,
        }
        stage_sql = (
            "INSERT INTO public.collection_reward_launch VALUES ('finney',71,101,%s)"
        )
        with pytest.raises(pg.Error, match="unverified P7 price"):
            q.execute(stage_sql, (Json(dict(approval, prices={"P": {model: 100000}})),))
        q.execute("ROLLBACK TO sample")
        q.execute(stage_sql, (Json(approval),))
        assert rpc("collection_reward_quotes_v1", 101)["job_count"] == 0
        terms = PAID_WORKFLOW_TERMS_HASH
        consent = {
            "round_id": "finney:71:locked-collection:live",
            "hotkey": MINER,
            "scope": "lab_arena.collection.register.v1",
            "body": {
                "daily_cap_microusd": 100000,
                "terms_sha256": terms,
                "task_scope": TASK_SCOPE,
                "offer_id": offer,
                "max_slots_per_utc_day": 1,
            },
        }
        args = (
            str(uuid4()),
            MINER,
            MINER,
            "live",
            Json(consent),
            100000,
            Json(
                {
                    "key_id": live_key,
                    "fingerprint": "sha256:" + "4" * 64,
                    "ciphertext_b64": "mock",
                }
            ),
            None,
        )
        claim = (
            live_job,
            token,
            offer,
            "live",
            commit,
            Json(["P7-s40402", "P7-s40403"]),
            Json([model]),
            Json(measured["prices"]["P"]),
            Json(measured["budgets"]["P"]),
            terms,
        )
        assert (
            rpc("collection_claim_v1", *claim) is None
        )  # pilot grant cannot fund paid work
        assert rpc("collection_register_v1", *args)["status"] == "enrolled"
        assert rpc("collection_priced_claim_v1", *claim, None) == {"model": model}
        assert rpc("collection_claim_v1", *claim)["price_microusd"] == 8172
        assert (
            rpc("collection_claim_v1", *claim)["job_id"] == live_job
        )  # retry returns existing lease
        quality = {"training_status": "needs_review", "payment_rule": "verified_p7_v1"}
        finish = (live_job, token, Json(quality), artifact)
        q.execute("SAVEPOINT claim")
        with pytest.raises(pg.Error, match="settled"):
            rpc("collection_prepare_launch_v1", *finish)
        q.execute("ROLLBACK TO claim")
        bill(live_job, live_key, 6500)
        q.execute("SAVEPOINT funding")
        with pytest.raises(pg.Error, match="verified P7 execution"):
            rpc(
                "collection_prepare_launch_v1",
                live_job,
                token,
                Json({"training_status": "needs_review"}),
                artifact,
            )
        q.execute("ROLLBACK TO funding")
        assert rpc("collection_prepare_launch_v1", *finish)["status"] == "ready"
        assert rpc("collection_reward_quotes_v1", 101)["job_count"] == 0
        conn.commit()
        # Fresh transaction resumes a ready launch without another model call.
        signed_job_anchor = Json(dict(anchor, miner_hotkey=MINER))
        for _ in range(2):
            assert (
                rpc("collection_launch_finish_v1", live_job, signed_job_anchor)[
                    "status"
                ]
                == "accepted"
            )
        q.execute("SET CONSTRAINTS ALL IMMEDIATE")
        assert rpc("collection_reward_quotes_v1", 101)["quotes_microusd"] == {
            MINER: 8172
        }
        assert rpc("collection_reward_quotes_v1", 102)["job_count"] == 0
        assert rpc("collection_reward_start_v1") == {"first_epoch": 101}
