"""Exercise UTC funding quotas through the real claim, receipt and revocation RPCs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from lab_arena import collection_intake as funding, contracts, store
from tests.lab_arena import collection_intake_test as kit
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.locked_collection_pipeline_test import claim


@pytest.fixture
def database():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    numbers = (720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 732, 733, 738, 740, 743, 744, 745)
    files = tuple(next(scripts.glob(f"{n}-*.sql")).name for n in numbers)
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS + files)


@pytest.fixture
def api(database):
    pg, dsn = database

    def connect():
        connection = pg.connect(**dsn)
        connection.autocommit = True
        return connection

    db = store.PsycopgTransport(connect, role="lab_arena_service")
    yield db
    db.close()


def enroll(db, key, owner=None, funding_key=None, cap=kit.CAP, mode="pilot", scope=funding.TASK_SCOPE):
    client = funding.CollectionIntake(db, kit.manager(), mode)
    signed, credentials = kit.registration(key, mode)
    credentials = kit.credentials_for(funding_key or key)
    signed = contracts.build_signed_request(
        scope=funding.SCOPE["register"], round_id=funding.scope_id(mode), hotkey=signed["hotkey"],
        body=dict(signed["body"], daily_cap_microusd=cap, task_scope=scope,
                  terms_sha256=funding.terms(mode, scope)["terms_sha256"],
                  credentials_sha256=contracts.document_hash(credentials)),
        timestamp=1, sign_message=lambda _: "a" * 128)
    result = client.register(signed, owner or signed["hotkey"], credentials)
    return client, result["registration_id"]


def take(client, batch="quota-A"):
    return claim(client, batch, "C01-s2071")


def finish(client, job, status):
    client.complete(job["job_id"], job["lease_token"], status, {}, None, 25)


def test_settled_failure_may_replace_same_slot_without_resetting_cost_or_daily_limit(database, api):
    client, _ = enroll(api, "A", cap=1000)
    original = take(client)
    lease = store.hash_lease_token(original["lease_token"])
    call = dict(p_job=original["job_id"], p_lease=lease, p_request="generation")
    assert api.rpc("collection_reserve_call_v1", dict(call, p_provider="openrouter",
                   p_amount=1000, p_hash="a"*64))["status"] == "reserved"
    finish(client, original, "failed")
    assert take(client) is None
    api.rpc("collection_record_call_v1", dict(call, p_hash="b"*64, p_status=200,
            p_cost=800, p_provider_id=str(uuid4())))
    replacement = take(client)
    assert replacement and (replacement["slot"], replacement["attempt"]) == (original["slot"], 2)
    assert replacement["model"] == original["model"]
    assert replacement["budget_microusd"] == original["budget_microusd"] - 800
    retry_call = dict(p_job=replacement["job_id"], p_lease=store.hash_lease_token(replacement["lease_token"]),
                      p_request="replacement-call", p_provider="openrouter", p_hash="c" * 64)
    assert api.rpc("collection_reserve_call_v1", dict(retry_call, p_amount=201))["status"] == "budget_exhausted"
    assert api.rpc("collection_reserve_call_v1", dict(retry_call, p_amount=200))["status"] == "reserved"
    assert take(client, "different-pack") is None
    finish(client, replacement, "failed")
    assert take(client) is None
    assert take(client, "different-pack") is None


def sql(database, statement, parameters=()):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(statement, parameters)
        return cursor.fetchone() if cursor.description else None


def test_quota_independently_follows_coldkey_and_funding_fingerprint(database, api):
    sql(database, "UPDATE public.collection_dispatch_settings SET daily_slots_per_owner=2")
    owner_a, owner_b, owner_c = ("5" + letter * 47 for letter in "XYZ")

    def run_and_revoke(key, owner, funding_key, batch):
        client, rid = enroll(api, key, owner, funding_key)
        job = take(client, batch)
        if job:
            assert job["miner_hotkey"] == "5" + key * 47
            finish(client, job, "failed")
        client.revoke("5" + key * 47, rid)
        return job

    assert run_and_revoke("A", owner_a, "A", "one")
    assert run_and_revoke("B", owner_b, "B", "two")
    # The owner used one slot and the reused runtime key used a different slot.
    assert run_and_revoke("C", owner_a, "B", "three")
    assert run_and_revoke("D", owner_a, "D", "owner-exhausted") is None
    assert run_and_revoke("E", owner_c, "B", "key-exhausted") is None
    assert sql(database, "SELECT count(*) FROM public.collection_jobs")[0] == 3


def test_hotkey_cannot_reset_daily_slot_by_changing_funding_key_and_account_owner(database, api):
    client, rid = enroll(api, "A", "5" + "Y" * 47)
    first = take(client)
    finish(client, first, "failed")
    client.revoke(first["miner_hotkey"], rid)
    replacement, _ = enroll(api, "A", "5" + "Z" * 47, funding_key="B")
    assert take(replacement, "new-owner-and-key") is None


def test_ordinary_and_experiment_scopes_share_the_owner_allowance(database, api):
    from lab_arena.collection_experiment import load_experiment
    from lab_arena.collection_workflow import CORRECTION_CHAMPION_REV
    from tests.lab_arena.collection_experiment_test import settings

    owner = "5" + "Z" * 47
    client, _ = enroll(api, "A", owner)
    expert, _ = enroll(api, "B", owner, scope=funding.CORRECTION_SCOPE)
    ordinary = take(client)
    assert ordinary["miner_hotkey"] == "5" + "A" * 47
    finish(client, ordinary, "failed")
    experiment = load_experiment(settings(), {})
    assert expert.claim_experiment(experiment, CORRECTION_CHAMPION_REV) is None
    sql(database, "UPDATE public.collection_dispatch_settings SET daily_slots_per_owner=2")
    correction = expert.claim_experiment(experiment, CORRECTION_CHAMPION_REV)
    assert correction and correction["miner_hotkey"] == "5" + "B" * 47


def test_concurrent_hotkeys_share_one_active_lease_and_one_daily_quota(database, api):
    owner = "5" + "Z" * 47
    sql(database, "UPDATE public.collection_dispatch_settings SET daily_slots_per_owner=2")
    client, _ = enroll(api, "A", owner)
    enroll(api, "B", owner)

    def race(batch):
        with ThreadPoolExecutor(max_workers=4) as pool:
            return [job for job in pool.map(lambda _: take(client, batch), range(4)) if job]

    first, = race("first")
    assert race("busy") == []
    finish(client, first, "failed")
    second, = race("second")
    assert first["miner_hotkey"] != second["miner_hotkey"]
    finish(client, second, "failed")
    assert race("daily-limit") == []


def test_running_lease_crosses_utc_midnight_but_finished_slot_uses_its_original_day(database, api):
    client, _ = enroll(api, "A")
    old = take(client, "yesterday")
    assert str(old["collection_day"]) == str(sql(database, "SELECT (now() AT TIME ZONE 'UTC')::date")[0])
    sql(database, "UPDATE public.collection_jobs SET collection_day=collection_day-1 WHERE job_id=%s",
        (old["job_id"],))
    assert take(client, "today") is None
    finish(client, old, "failed")
    today = take(client, "today")
    assert today
    finish(client, today, "failed")
    assert take(client, "another") is None


def test_running_cross_mode_exclusion_and_separate_finished_allowances(database, api):
    from tests.lab_arena.collection_calibration_test import exercise_ordinary_cutover

    exercise_ordinary_cutover(database, after_cutover=lambda cursor: cursor.connection.commit())
    # The previous fixture donors have finished. Preserve their bills and use two new wallets.
    sql(database, "UPDATE public.collection_registrations SET revoked_at=now() WHERE revoked_at IS NULL")
    proof = sql(database, "SELECT public.collection_reward_schedule_v1()")[0]["pilot_evidence"]
    owner = "5" + "Z" * 47
    pilot, _ = enroll(api, "A", owner)
    live, _ = enroll(api, "B", owner, mode="live")
    paid = live.claim("live-quota-test", ["C01-s2101"], proof["harness_commit"],
                      proof["prices"]["C"], proof["budgets"]["C"])
    assert paid
    assert take(pilot, "other-mode") is None
    finish(live, paid, "failed")
    job = take(pilot, "other-mode")
    assert job and job["owner_coldkey"] == owner


def test_dispatch_limit_is_bounded_and_writable_only_by_operator(database):
    pg, _ = database
    assert sql(database, "SELECT daily_slots_per_owner FROM public.collection_dispatch_settings") == (1,)
    for bad in (0, 4097):
        with pytest.raises(pg.errors.CheckViolation):
            sql(database, "UPDATE public.collection_dispatch_settings SET daily_slots_per_owner=%s", (bad,))
    for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
        assert sql(database, "SELECT has_table_privilege(%s,'public.collection_dispatch_settings','UPDATE')",
                   (role,)) == (False,)
