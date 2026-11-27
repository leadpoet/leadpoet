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
        756,
        757,
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
        # The operator reads this exact RPC over the shared transport allowlist.
        from lab_arena.store import FUNCTION_SIGNATURES

        assert FUNCTION_SIGNATURES["collection_verified_p7_fees_v1"] == (
            ("p_batch", "text"),
            ("p_jobs", "jsonb"),
        )
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
        # Provider and verifier receipts must not multiply either the job count or the fee.
        q.execute("SAVEPOINT contributor_check")
        bill(live_job, live_key, 100, request="second-funded-call")
        bill(live_job, None, 900, provider="judge", request="owner-judge")
        [row] = rpc("collection_public_miners_v1", None)
        assert row["hotkey"] == MINER and row["funded_runs"] == 1
        assert row["last_funded_at"] and row["epochs"] == {"101": 8172}
        q.execute("ROLLBACK TO contributor_check")
        conn.commit()
        exercise_successor_offers(
            db,
            q,
            rpc,
            approval,
            consent,
            model,
            offer,
            live_job,
            commit,
            token,
            artifact,
        )


def exercise_successor_offers(
    db, q, rpc, initial, consent, model, original, source_job, commit, token, artifact
):
    """Exercise same-day selection, historical consent and first-epoch invariants."""
    from bittensor_wallet import Keypair
    from contextlib import closing
    from scripts.collection_price_successor import publish

    pg, dsn = db
    samples = rpc("collection_offer_samples_v1", original, 0)
    assert (
        len(samples) == 1
        and samples[0]["job_id"] == source_job
        and samples[0]["cost_microusd"] == 6500
    )
    new = "dv-" + "2" * 64
    next_offer = {
        **initial,
        "schema": "leadpoet.collection.verified_task_prices.v1",
        "live_batch_id": new,
        "batch_id": original,
        "observations": samples,
        "measurement_kind": "settled_production",
        "through_slot": 0,
        "job_count": 1,
        "prices": {"P": {model: 8125}},
    }
    next_offer["historical_realization"] = {
        "quote_epoch": 101,
        "payment_block": 450,
        "block_hash": "0x" + "7" * 64,
        "quoted_microusd": 8172,
        "received_microusd_at_signed_price": 6280,
    }
    for field in ("cutover_anchor", "measurement_jobs", "measurement_sha256"):
        next_offer.pop(field, None)
    with closing(pg.connect(**dsn)) as signer:
        with pytest.raises(pg.Error, match="differ from observations"):
            publish(signer, dict(next_offer, prices={"P": {model: 8225}}), original)
        with pytest.raises(pg.Error, match="settled production source"):
            publish(
                signer,
                dict(next_offer, observations=[dict(samples[0], cost_microusd=6501)]),
                original,
            )
        incorrect_history = dict(
            next_offer["historical_realization"], quoted_microusd=9000
        )
        with pytest.raises(pg.Error, match="invalid prior payout observation"):
            publish(
                signer,
                dict(next_offer, historical_realization=incorrect_history),
                original,
            )
        assert publish(signer, next_offer, original) == "published"
        assert publish(signer, next_offer, original) == "already_published"
        with pytest.raises(ValueError, match="different measurements"):
            publish(signer, dict(next_offer, job_count=3), original)
        with pytest.raises(pg.Error, match="predecessor"):
            publish(signer, dict(next_offer, live_batch_id="dv-" + "3" * 64), original)
    assert rpc("collection_offer_schedule_v1", None)["pilot_evidence"] == {
        k: v for k, v in next_offer.items() if k != "observations"
    }
    assert (
        rpc("collection_offer_schedule_v1", original)["pilot_evidence"]["live_batch_id"]
        == original
    )
    assert rpc("collection_offer_schedule_v1", "dv-" + "f" * 64) == {}
    assert (
        rpc("collection_reward_schedule_v1")["pilot_evidence"]["live_batch_id"]
        == original
    )
    assert rpc("collection_reward_quotes_v1", 101)["quotes_microusd"] == {MINER: 8172}

    for i, (offer_id, fee) in enumerate([(original, 8172), (new, 8125)], 1):
        key = Keypair.create_from_uri(f"//Priced{i}").ss58_address
        registration = str(uuid4())
        credential = str(uuid4())
        signed = dict(
            consent, hotkey=key, body=dict(consent["body"], offer_id=offer_id)
        )
        grant = rpc(
            "collection_register_v1",
            registration,
            key,
            key,
            "live",
            Json(signed),
            100000,
            Json(
                {
                    "key_id": credential,
                    "fingerprint": "sha256:" + str(6 + i) * 64,
                    "ciphertext_b64": "mock",
                }
            ),
            None,
        )
        assert grant["status"] == "enrolled"
        scenarios = (
            ["P7-s40402", "P7-s40403"] if offer_id == original else ["P7-s40410"]
        )
        claim = (
            str(uuid4()),
            token,
            offer_id,
            "live",
            commit,
            Json(scenarios),
            Json([model]),
            Json({model: fee}),
            Json({model: 20000}),
            PAID_WORKFLOW_TERMS_HASH,
        )
        job = rpc("collection_claim_v1", *claim)
        assert (job["miner_hotkey"], job["mode"], job["price_microusd"]) == (
            key,
            "live",
            fee,
        )
        rpc("collection_finish_v1", claim[0], token, "failed", Json({}), None, None)

    q.connection.commit()
    # Owner references have no collection job or grant. They may price only the separate P8 task.
    research = "dv-" + "a" * 64
    sample = dict(
        samples[0],
        job_id=str(uuid4()),
        cost_microusd=1000,
        scenario_id="P8-s1",
        provider_receipts_sha256="d" * 64,
    )
    research_offer = dict(
        next_offer,
        live_batch_id=research,
        batch_id="owner-p8-" + "c" * 64,
        measurement_kind="owner_provider_receipts",
        payment_rule="verified_sales_evidence_v1",
        observations=[sample],
        prices={"P8": {model: 1250}},
        budgets={"P8": {model: 20000}},
    )
    research_offer.pop("through_slot")
    with closing(pg.connect(**dsn)) as signer:
        with pytest.raises(pg.Error, match="invalid settled price observations"):
            publish(
                signer,
                dict(research_offer, observations=[dict(sample, scenario_id="P7-s1")]),
                new,
            )
        assert publish(signer, research_offer, new) == "published"
    current = rpc("collection_offer_schedule_v1", None)
    assert current["available_offers"] == [
        {"task": "P8", "offer_id": research},
        {"task": "P7", "offer_id": new},
    ]
    assert not any(
        r["job_id"] == sample["job_id"]
        for r in rpc("collection_batch_jobs_v1", research)
    )
    key = Keypair.create_from_uri("//PaidSales").ss58_address
    grant = dict(consent, hotkey=key, body=dict(consent["body"], offer_id=research))
    rpc(
        "collection_register_v1",
        str(uuid4()),
        key,
        key,
        "live",
        Json(grant),
        100000,
        Json(
            {
                "key_id": str(uuid4()),
                "fingerprint": "sha256:" + "9" * 64,
                "ciphertext_b64": "mock",
            }
        ),
        None,
    )
    job = rpc(
        "collection_claim_v1",
        str(uuid4()),
        token,
        research,
        "live",
        commit,
        Json(["P8-s2"]),
        Json([model]),
        Json({model: 1250}),
        Json({model: 20000}),
        PAID_WORKFLOW_TERMS_HASH,
    )
    assert (
        job["miner_hotkey"] == key
        and job["scenario_id"] == "P8-s2"
        and job["price_microusd"] == 1250
    )
    assert (
        rpc(
            "collection_reserve_call_v1",
            job["job_id"],
            token,
            "generate-1",
            "openrouter",
            "a" * 64,
            1800,
        )["status"]
        == "reserved"
    )
    rpc(
        "collection_record_call_v1",
        job["job_id"],
        token,
        "generate-1",
        "f" * 64,
        "gen-research",
        1400,
        200,
    )
    q.execute("SAVEPOINT native_quality")
    with pytest.raises(pg.Error, match="settled verified"):
        rpc(
            "collection_finish_v1",
            job["job_id"],
            token,
            "accepted",
            Json({"payment_rule": "verified_p7_v1", "training_status": "needs_review"}),
            artifact,
            101,
        )
    q.execute("ROLLBACK TO native_quality")
    rpc(
        "collection_finish_v1",
        job["job_id"],
        token,
        "accepted",
        Json(
            {
                "payment_rule": "verified_sales_evidence_v1",
                "training_status": "needs_review",
            }
        ),
        artifact,
        101,
    )
    assert rpc("collection_reward_quotes_v1", 102)["quotes_microusd"] == {key: 1250}
    assert rpc("collection_reward_quotes_v1", 101)["quotes_microusd"] == {MINER: 8172}
    q.execute("SAVEPOINT immutability")
    with pytest.raises(pg.Error, match="immutable"):
        q.execute(
            "UPDATE public.collection_paid_offers SET approval='{}' WHERE offer_id=%s",
            (research,),
        )
    q.execute("ROLLBACK TO immutability")


def test_public_miner_cursor_preserves_new_and_returning_miners(db):
    pg, dsn = db
    keys = ["5" + char * 47 for char in "123456789ABCDEFGHJKLMNPQRSTUVWXYZ"]
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        def enroll(key, mode, date, revoked=False):
            q.execute("""INSERT INTO public.collection_registrations
                (registration_id,miner_hotkey,owner_coldkey,mode,consent,created_at,revoked_at)
                VALUES (%s,%s,%s,%s,'{}',%s,CASE WHEN %s THEN now() END)""",
                (str(uuid4()), key, key, mode, date, revoked))
        for key in keys:
            enroll(key, "live", "2026-01-02", revoked=True)
        enroll(keys[0], "live", "2026-02-01")
        enroll("5" + "a" * 47, "pilot", "2026-03-01")
        q.execute("SET LOCAL ROLE lab_arena_service")
        def page(after):
            q.execute("SELECT public.collection_public_miners_v1(%s)", (after,))
            return q.fetchone()[0]
        first = page(None)
        assert len(first) == 26
        joined = first[:25] + page(first[24]["hotkey"])
        assert [r["hotkey"] for r in joined] == sorted(keys, reverse=True)
        assert all(r["funded_runs"] == 0 and r["last_funded_at"] is None and not r["epochs"] for r in joined)
        assert joined[-1]["joined_at"].startswith("2026-01-02")
        assert not page(keys[0])
        q.execute("SELECT has_function_privilege('anon', 'public.collection_public_miners_v1(text)', 'execute')")
        assert q.fetchone() == (False,)
