"""Economic authorization joins the real 713–716 database contracts."""

from uuid import uuid4

from bittensor_wallet import Keypair
import pytest
from psycopg2.extras import Json

from lab_arena.collection_intake import MODELS
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)


@pytest.fixture(scope="module")
def db():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS
        + (
            "713-locked-collection-intake.sql",
            "714-locked-collection-reward-epoch.sql",
            "715-collection-call-budget.sql",
            "716-collection-pilot-prices.sql",
        )
    )


def test_freeze_reviewed_fees_uses_each_actual_miner_charge_once(db):
    pg, dsn = db
    miner = Keypair.create_from_uri("//Calibration").ss58_address
    batch = "test-pilot"
    reg, orr, dee = map(str, (uuid4(), uuid4(), uuid4()))
    commit = "1" * 40
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute(
            "INSERT INTO public.collection_batches VALUES (%s,'pilot',%s,%s,%s,%s,%s)",
            (batch, commit, Json(["C01-s2001"]), Json(MODELS), Json({}), Json({})),
        )
        q.execute(
            "INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent) "
            "VALUES (%s,%s,%s,'pilot','{}')",
            (reg, miner, miner),
        )
        for key, provider, fingerprint in (
            (orr, "openrouter", "1"),
            (dee, "deepline", "2"),
        ):
            q.execute(
                "INSERT INTO public.collection_keys(key_id,registration_id,provider,miner_hotkey,submission_id,"
                "daily_cap_microusd,fingerprint,ciphertext_b64) VALUES (%s,%s,%s,%s,'fixture',10000000,%s,'fixture')",
                (key, reg, provider, miner, "sha256:" + fingerprint * 64),
            )
        for slot in range(102):
            job = str(uuid4())
            model = MODELS[slot % len(MODELS)]
            q.execute(
                """INSERT INTO public.collection_jobs(job_id,batch_id,mode,miner_hotkey,registration_id,openrouter_key_id,
                      deepline_key_id,slot,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,
                      status,lease_token_hash,lease_expires_at,finished_at,accepted_epoch,artifact_sha256)
                      VALUES (%s,%s,'pilot',%s,%s,%s,%s,%s,current_date-%s,'C01-s2001',%s,%s,3000000,0,'accepted',%s,
                      now(),now(),100,%s)""",
                (
                    job,
                    batch,
                    miner,
                    reg,
                    orr,
                    dee,
                    slot,
                    slot,
                    model,
                    commit,
                    "sha256:" + "a" * 64,
                    "sha256:" + "b" * 64,
                ),
            )
            for provider, key, actual in [
                ("openrouter", orr, 100),
                ("deepline", dee, 20),
                ("judge", None, 2000),
            ]:
                q.execute(
                    """INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd)
                           VALUES (%s,%s,%s,%s,%s,10000)""",
                    (job, provider, provider, key, "a" * 64),
                )
                q.execute(
                    """INSERT INTO public.collection_call_receipts(job_id,request_id,response_hash,provider_id,cost_microusd,http_status)
                           VALUES (%s,%s,%s,%s,%s,200)""",
                    (job, provider, "c" * 64, job + provider, actual),
                )
        q.execute("SELECT public.collection_pilot_fees_v1(%s,'[]')", (batch,))
        result = q.fetchone()[0]
        assert result["prices"] == {"C": dict.fromkeys(MODELS, 150)}
        assert (
            result["budgets"] == {"C": dict.fromkeys(MODELS, 3000000)}
            and result["job_count"] == 102
        )
        proof = {
            **result,
            "schema": "leadpoet.collection.approved_pilot.v1",
            "batch_id": batch,
            "harness_commit": commit,
            "builder_commit": "2" * 40,
            "control_ids": [],
            "review_sha256": "a" * 64,
            "review_bundle_sha256": "b" * 64,
            "judge_policy_sha256": "d" * 64,
            "judge_prompt_sha256": "e" * 64,
        }
        q.execute("SAVEPOINT altered")
        with pytest.raises(pg.Error, match="prices or budget differs"):
            q.execute(
                "INSERT INTO public.collection_reward_cutover VALUES ('finney',71,101,%s)",
                (Json(dict(proof, prices={"C": dict.fromkeys(MODELS, 10000)})),),
            )
        q.execute("ROLLBACK TO altered")
        q.execute(
            "INSERT INTO public.collection_reward_cutover VALUES ('finney',71,101,%s)",
            (Json(proof),),
        )
        q.execute("SELECT public.collection_reward_schedule_v1()")
        assert q.fetchone()[0]["pilot_evidence"] == proof
        q.execute(
            "INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd) "
            "VALUES (%s,'missing','deepline',%s,%s,30)",
            (job, dee, "a" * 64),
        )
        with pytest.raises(pg.Error, match="incomplete"):
            q.execute("SELECT public.collection_pilot_fees_v1(%s,'[]')", (batch,))
