"""Economic authorization joins the real collection database contracts."""

import time
from uuid import uuid4

from bittensor_wallet import Keypair
import pytest
from psycopg2.extras import Json

from lab_arena.collection_intake import MODELS, TASK_SCOPE, TERMS_HASH
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)


@pytest.fixture(scope="module")
def db():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS
        + (
            "720-locked-collection-intake.sql",
            "721-locked-collection-reward-epoch.sql",
            "722-collection-call-budget.sql",
            "723-collection-pilot-prices.sql",
            "724-collection-scenario-ranking.sql",
            "725-collection-generator-roster.sql",
            "726-collection-first-paid-job.sql",
            "727-collection-synthetic-funding.sql",
            "728-collection-pilot-attempts.sql",
            "729-collection-ready-judge-recovery.sql",
        )
    )


def test_freeze_reviewed_fees_uses_each_actual_miner_charge_once(db):
    exercise_ordinary_cutover(db)


def exercise_ordinary_cutover(db, *, paid_panel=None, after_cutover=None, retired_anchor=None, template="C01"):
    pg, dsn = db
    miner = Keypair.create_from_uri("//Calibration").ss58_address
    batch = "test-pilot"
    reg, orr, dee = map(str, (uuid4(), uuid4(), uuid4()))
    commit = "1" * 40
    family = template[0]
    scenarios = [f"{template}-s{2001+i}" for i in range(17)]
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute(
            "INSERT INTO public.collection_batches VALUES (%s,'pilot',%s,%s,%s,%s,%s)",
            (batch, commit, Json(scenarios), Json(MODELS), Json({}), Json({})),
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
                      VALUES (%s,%s,'pilot',%s,%s,%s,%s,%s,current_date-%s,%s,%s,%s,3000000,0,'accepted',%s,
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
                    scenarios[slot // 6],
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
        assert result["prices"] == {family: dict.fromkeys(MODELS, 150)}
        assert (
            result["budgets"] == {family: dict.fromkeys(MODELS, 3000000)}
            and result["job_count"] == 102
        )
        # Retry liability comes from the original budget. Neither the failed miner charge
        # nor the verifier charge buys an additional completed run or inflates its quote.
        q.execute("SAVEPOINT with_retry")
        q.execute("UPDATE public.collection_jobs SET status='failed',accepted_epoch=NULL,artifact_sha256=NULL WHERE job_id=%s",(job,))
        retry = str(uuid4())
        q.execute("""INSERT INTO public.collection_jobs(job_id,batch_id,mode,miner_hotkey,registration_id,openrouter_key_id,
                  deepline_key_id,slot,attempt,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,
                  status,lease_token_hash,lease_expires_at,finished_at,accepted_epoch,artifact_sha256)
                  SELECT %s,batch_id,mode,miner_hotkey,registration_id,openrouter_key_id,deepline_key_id,slot,2,current_date-500,
                  scenario_id,model,harness_commit,budget_microusd-2120,0,'accepted',lease_token_hash,now(),now(),100,%s
                  FROM public.collection_jobs WHERE job_id=%s""",(retry,"sha256:"+"f"*64,job))
        q.execute("INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd) VALUES (%s,'retry','openrouter',%s,%s,100)",(retry,orr,"e"*64))
        q.execute("INSERT INTO public.collection_call_receipts VALUES (%s,'retry',%s,%s,50,200)",(retry,"c"*64,retry))
        q.execute("SELECT public.collection_pilot_fees_v1(%s,'[]')",(batch,))
        counted = q.fetchone()[0]
        assert counted["prices"][family][model] == 145  # ceil(1.25 * (16 * 120 + 50) / 17)
        assert counted["job_count"] == 102 and counted["budgets"] == result["budgets"]
        q.execute("INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd) VALUES (%s,'unknown','openrouter',%s,%s,20)",(job,orr,"f"*64))
        with pytest.raises(pg.Error, match="incomplete"):
            q.execute("SELECT public.collection_pilot_fees_v1(%s,'[]')",(batch,))
        q.execute("ROLLBACK TO with_retry")
        from tests.lab_arena.collection_rewards_test import insert_legacy_state

        predecessor = retired_anchor(q, 100) if retired_anchor else insert_legacy_state(q, 100)
        before_launch = {"first_epoch": 100 if retired_anchor else None}
        proof = {
            **result,
            "cutover_anchor": {
                "epoch": 100,
                "state_hash": predecessor["state_hash"],
                "head_block": 25,
                "head_hash": "0x" + "b" * 64,
                "observed_at": int(time.time()),
            },
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
        if paid_panel is not None:
            proof["panel_qualification_sha256"] = paid_panel
        q.execute("SAVEPOINT altered")
        for bad_anchor in (
            {"observed_at": int(time.time()) - 150},
            {"head_block": 51},
            {"epoch": 99},
            {"state_hash": "sha256:" + "f" * 64},
        ):
            with pytest.raises(pg.Error, match="finalized legacy epoch readback"):
                q.execute(
                    "INSERT INTO public.collection_reward_cutover VALUES ('finney',71,101,%s)",
                    (
                        Json(
                            dict(
                                proof,
                                cutover_anchor={
                                    **proof["cutover_anchor"],
                                    **bad_anchor,
                                },
                            )
                        ),
                    ),
                )
            q.execute("ROLLBACK TO altered")

        with pytest.raises(pg.Error, match="prices or budget differs"):
            q.execute(
                "INSERT INTO public.collection_reward_cutover VALUES ('finney',71,101,%s)",
                (Json(dict(proof, prices={family: dict.fromkeys(MODELS, 10000)})),),
            )
        q.execute("ROLLBACK TO altered")
        # Even an approved pilot cannot activate payments without an earned live quote.
        with pytest.raises(pg.Error, match="first paid collection job"):
            q.execute(
                "INSERT INTO public.collection_reward_cutover VALUES ('finney',71,101,%s)",
                (Json(proof),),
            )
            q.execute("SET CONSTRAINTS ALL IMMEDIATE")
        q.execute("ROLLBACK TO altered")
        q.execute(
            "INSERT INTO public.collection_reward_launch VALUES ('finney',71,101,%s)",
            (Json(proof),),
        )
        q.execute("SELECT public.collection_reward_start_v1(), public.collection_reward_schedule_v1()")
        start, staged = q.fetchone()
        assert start == before_launch
        assert staged["staged"] and staged["pilot_evidence"] == proof
        q.execute("SAVEPOINT staged")
        q.execute("DELETE FROM public.collection_reward_launch")
        q.execute("SELECT public.collection_reward_schedule_v1(), public.collection_reward_start_v1()")
        assert q.fetchone() == ({}, before_launch)
        q.execute("ROLLBACK TO staged")
        live_job, live_batch, token = str(uuid4()), "test-live", "sha256:" + "b" * 64
        live_reg, live_orr = str(uuid4()), str(uuid4())
        consent = {"schema_version":1,"scope":"lab_arena.collection.register.v1",
                   "round_id":"finney:71:locked-collection:live","hotkey":miner,
                   "timestamp":1795540000,"request_id":"fixture-unified-terms",
                   "body":{"daily_cap_microusd":10000000,"task_scope":TASK_SCOPE,
                           "terms_sha256":TERMS_HASH,"credentials_sha256":"sha256:"+"1"*64},
                   "signature":"a"*128}
        q.execute("SET LOCAL ROLE lab_arena_service")
        q.execute("SELECT public.collection_register_v1(%s,%s,%s,'live',%s,10000000,%s,NULL)",
                  (live_reg, miner, miner, Json(consent),
                   Json({"key_id": live_orr, "fingerprint": "sha256:" + "3"*64, "ciphertext_b64": "fixture"})))
        assert q.fetchone()[0]["status"] == "enrolled"
        fee = result["prices"][family][MODELS[0]]
        q.execute("SELECT public.collection_claim_v1(%s,%s,%s,'live',%s,%s,%s,%s,%s,%s)",
                  (live_job, token, live_batch, commit, Json([scenarios[0]]), Json(MODELS),
                   Json(result["prices"][family]), Json(result["budgets"][family]), TERMS_HASH))
        claimed = q.fetchone()[0]
        assert claimed["job_id"] == live_job and claimed["deepline_key_id"] is None
        assert claimed["price_microusd"] == fee and claimed["openrouter_key_id"] == live_orr
        q.execute("RESET ROLE")
        q.execute("SAVEPOINT leased")
        with pytest.raises(pg.Error, match="running collection job"):
            q.execute("DELETE FROM public.collection_reward_launch")
        q.execute("ROLLBACK TO leased")
        args = (live_job, token, Json({"training_status":"cannot_judge"}), "sha256:" + "f" * 64)
        prepare_sql = "SELECT public.collection_prepare_launch_v1(%s,%s,%s,%s)"
        finish_sql = "SELECT public.collection_launch_finish_v1(%s,%s)"
        anchor = dict(proof["cutover_anchor"], miner_hotkey=miner)
        with pytest.raises(pg.Error, match="settled first paid collection job"):
            q.execute(prepare_sql, args)
        q.execute("ROLLBACK TO leased")
        q.execute("INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd) "
                  "VALUES (%s,'first','openrouter',%s,%s,500)", (live_job,live_orr,"a"*64))
        q.execute("INSERT INTO public.collection_call_receipts(job_id,request_id,response_hash,provider_id,cost_microusd,http_status) "
                  "VALUES (%s,'first',%s,%s,100,200)", (live_job,"b"*64,live_job))
        q.execute(prepare_sql, args)
        assert q.fetchone()[0]["status"] == "ready"
        q.execute(prepare_sql, args)
        assert q.fetchone()[0]["status"] == "ready"
        # A new worker can resume only the terminal acceptance without any provider key or old lease token.
        q.execute("SELECT public.collection_reward_schedule_v1(),public.collection_reward_start_v1()")
        pending, champion = q.fetchone()
        assert pending["pending"] == {
            "job_id": live_job, "miner_hotkey": miner, "owner_coldkey": miner, "status": "ready",
            "batch_id": live_batch, "model": MODELS[0], "scenario_id": scenarios[0],
            "harness_commit": commit, "artifact_sha256": args[-1],
        }
        assert champion == before_launch
        conn.commit()
        with pg.connect(**dsn) as independent, independent.cursor() as fresh:
            fresh.execute("SET LOCAL ROLE lab_arena_service")
            fresh.execute("SELECT public.collection_reward_schedule_v1(),public.collection_reward_start_v1()")
            resumed, before = fresh.fetchone()
            assert resumed["pending"] == pending["pending"] and before == before_launch
            fresh.execute("SELECT has_table_privilege(current_user,'public.collection_reward_launch','INSERT')")
            assert fresh.fetchone() == (False,)
        source = args[-1]
        q.execute("SAVEPOINT pending_outage")
        with pytest.raises(pg.Error, match="owner verification"):
            q.execute(finish_sql, (live_job, Json(anchor)))
        q.execute("ROLLBACK TO pending_outage")
        from tests.lab_arena.collection_launch_review_test import verify_recovery_contract
        verify_recovery_contract(q, pg, live_job, source, claimed["budget_microusd"])
        q.execute("SAVEPOINT paid")
        with pytest.raises(pg.Error, match="finalized legacy epoch readback"):
            q.execute(finish_sql, (live_job, Json(dict(anchor, observed_at=int(time.time()) - 150))))
        q.execute("ROLLBACK TO paid")
        q.execute("SELECT public.collection_reward_start_v1(),status FROM public.collection_jobs WHERE job_id=%s", (live_job,))
        assert q.fetchone() == (before_launch, "ready")
        q.execute(finish_sql, (live_job, Json(anchor)))
        assert q.fetchone()[0]["status"] == "accepted"
        q.execute("SET CONSTRAINTS ALL IMMEDIATE")
        q.execute(finish_sql, (live_job, Json(anchor)))
        assert q.fetchone()[0]["status"] == "accepted"
        q.execute("SELECT public.collection_reward_start_v1(),public.collection_reward_quotes_v1(101),"
                  "public.collection_reward_quotes_v1(102),public.collection_reward_schedule_v1()")
        active, first, second, schedule = q.fetchone()
        assert active == {"first_epoch": 100 if retired_anchor else 101}
        assert schedule["first_epoch"] == 101
        assert first["job_count"] == 1 and first["quotes_microusd"] == {miner: fee}
        assert second["job_count"] == 0
        assert not schedule.get("staged")
        if after_cutover:
            after_cutover(q)
        q.execute(
            "INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd) "
            "VALUES (%s,'missing','deepline',%s,%s,30)",
            (job, dee, "a" * 64),
        )
        with pytest.raises(pg.Error, match="incomplete"):
            q.execute("SELECT public.collection_pilot_fees_v1(%s,'[]')", (batch,))


def test_live_enrollment_preserves_champion_rounds_until_activation():
    from datetime import timedelta
    from tests.lab_arena.reasoning_activation_creation_test import CUTOFF, service_and_store
    from lab_arena.service import ServiceError

    service, store = service_and_store()
    service.collection_mode = lambda: "live"
    service._collection_reward_start = lambda: None
    service.open_round = lambda **_: {"round_id": "current", "status": "open"}
    assert service.ensure_daily_round()["round_id"] == "current"
    before = service.create_round(CUTOFF)
    service._collection_reward_start = lambda: 101
    assert service.ensure_daily_round() == {"status": "disabled"}
    assert service.create_round(CUTOFF) == before
    with pytest.raises(ServiceError, match="champion_rounds_replaced"):
        service.create_round(CUTOFF + timedelta(days=1))


def test_worker_resumes_completed_launch_before_claiming_another_job(monkeypatch):
    from types import SimpleNamespace
    from scripts import run_locked_collection as worker

    pending = {"status": "ready", "job_id": str(uuid4())}
    db = SimpleNamespace(rpc=lambda name, args: {"first_epoch": 101, "staged": True, "pending": pending, "pilot_evidence": {}})
    resumed = []
    monkeypatch.setattr(worker, "finish_launch", lambda database, job, *args: resumed.append(job))
    assert worker.run_cohort(raw={"budgets_microusd": {}}, policy=SimpleNamespace(pilot=False, qualification=False), prompt={},
                             intake=None, database=db, objects=None, transport=None, pricing=None, epoch=lambda: 100,
                             sources=None, owner_lookup=None, judge_key="fixture", max_jobs=1) == 0
    assert resumed == [pending]


def test_burn_target_cannot_enroll_even_with_a_different_baseline(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from lab_arena import contracts
    from lab_arena.service import ArenaService, ServiceError
    from tests.lab_arena.collection_intake_test import registration

    signed, creds = registration("X", "live")
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime.now(timezone.utc)
    service._config = SimpleNamespace(verify_signature=lambda *_: True, credential_manager=None,
                                      defaults=SimpleNamespace(baseline_hotkey="5" + "Y" * 47),
                                      accepted_burn_hotkey=signed["hotkey"], banned_hotkeys_source=lambda: ())
    service._store = SimpleNamespace(_transport=None)
    service.collection_mode = lambda: "live"
    service._enforce_submission_request_limit = lambda _: None
    monkeypatch.setattr(contracts, "validate_signed_request", lambda *a, **kw: signed)
    with pytest.raises(ServiceError, match="collection_ineligible_hotkey"):
        service.handle_collection_request("register", {"authorization": signed, "credentials": creds})


def test_train_only_p7_quote_through_paid_job():
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    history = (720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 732, 733, 738, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751)
    steps = tuple(next(scripts.glob(f"{n}-*.sql")).name for n in history)
    for pg, dsn in database_with_lab_arena_migration((*DEFAULT_MIGRATIONS, *steps)):
        with pg.connect(**dsn) as conn, conn.cursor() as cursor:
            cursor.execute("SELECT public.collection_fee_class_v1(id) FROM unnest(%s::text[]) AS s(id)",
                           (["P7-s40401", "P6-s40401", "P8-s40401", "P7-s0", "C04-s3012"],))
            assert [r[0] for r in cursor.fetchall()] == ["P", "P", None, None, "C"]
        def check_public_rate(cursor):
            from types import SimpleNamespace
            from lab_arena.collection_intake import approved_comparison_quotes

            def read_schedule(name, args):
                assert (name, args) == ("collection_reward_schedule_v1", {})
                cursor.execute("SELECT public.collection_reward_schedule_v1()")
                return cursor.fetchone()[0]

            prices = approved_comparison_quotes(SimpleNamespace(rpc=read_schedule))
            cursor.execute("SELECT model,price_microusd,budget_microusd FROM public.collection_jobs "
                           "WHERE mode='live' AND status='accepted'")
            model, fee, cap = cursor.fetchone()
            assert len(prices) == 6
            assert {"task_class": "P", "model": model, "quote_microusd": fee, "cap_microusd": cap} in prices

        exercise_ordinary_cutover((pg, dsn), template="P7", after_cutover=check_public_rate)
