"""Exercise signed pool funding, overlapping leases and actual-cost credits through the real RPCs."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from uuid import uuid4

import pytest
from psycopg2.extras import Json

from lab_arena.collection_intake import MODELS
from lab_arena.collection_pool import POOL_SCOPE, POOL_PAYMENT_RULE, POOL_TERMS_HASH
from tests.lab_arena.collection_verified_p7_fees_test import (
    db,
    test_paid_offer_through_first_epoch as seed_paid,
)


def digest(s):
    return sha256(s.encode()).hexdigest()


class Pool:
    model = MODELS[1]
    harness = "c" * 40
    epochs = 201
    budget = 10000

    def __init__(self, database):
        self.driver, self.dsn = database

    def sql(self, query, args=(), *, owner=False):
        with self.driver.connect(**self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SET LOCAL ROLE "
                + ("lab_arena_owner" if owner else "lab_arena_service")
            )
            cur.execute(query, args)
            if cur.description:
                return cur.fetchone()[0]

    def rpc(self, fn, *args):
        return self.sql(
            "select public.collection_"
            + fn
            + "_v1("
            + ",".join(["%s"] * len(args))
            + ")",
            args,
        )

    def activate(self):
        self.sql(
            "insert into public.collection_pool_activation(singleton) values(true)",
            owner=True,
        )

    def enrollment(self, index, priority, limit, *, fingerprint=None, alter=None):
        key = str(uuid4())
        rid = str(uuid4())
        hk = "5" + ("ABCDEFGHJKLMNPQRSTUVWXYZ"[index] * 47)
        body = dict(
            task_scope=POOL_SCOPE,
            terms_sha256=POOL_TERMS_HASH,
            credentials_sha256="sha256:" + digest(key),
            spend_limit_microusd=limit,
            priority=priority,
        )
        if alter:
            body.update(alter)
        consent = Json(
            dict(
                round_id="finney:71:locked-collection:live",
                hotkey=hk,
                scope="lab_arena.collection.register.v1",
                body=body,
            )
        )
        cred = Json(
            dict(
                key_id=key,
                fingerprint=fingerprint or "sha256:" + digest(key),
                ciphertext_b64="local-fixture",
            )
        )
        return rid, hk, hk, "live", consent, limit, cred, None

    def register(self, index, priority, limit, **kwargs):
        args = self.enrollment(index, priority, limit, **kwargs)
        assert self.rpc("register", *args)["status"] == "enrolled"
        return args

    def claim(
        self,
        batch,
        *,
        budget=None,
        cases=("P9-s1001",) * 8,
        expected="*",
        job=None,
        token=None
    ):
        return self.rpc(
            "priced_claim",
            job or str(uuid4()),
            token or "sha256:" + digest(str(uuid4())),
            "pool-" + digest(batch),
            "live",
            self.harness,
            Json(list(cases)),
            Json([self.model] * len(cases)),
            Json({self.model: 0}),
            Json({self.model: budget or self.budget}),
            POOL_TERMS_HASH,
            expected,
        )

    def reserve(self, job, token, request, amount, provider="openrouter"):
        return self.rpc(
            "reserve_call",
            job["job_id"],
            token,
            request,
            provider,
            digest(request),
            amount,
        )["status"]

    def record(self, job, token, request, cost, generation, status=200):
        return self.rpc(
            "record_call",
            job["job_id"],
            token,
            request,
            digest(request + "response"),
            generation,
            cost,
            status,
        )

    def finish(self, job, token, passed=True, *, epoch=201, status="accepted"):
        return self.rpc(
            "finish",
            job["job_id"],
            token,
            status,
            Json(
                dict(
                    payment_rule=POOL_PAYMENT_RULE,
                    training_status="accepted" if passed else "code_rejected",
                )
            ),
            "sha256:" + digest(job["job_id"]),
            epoch,
        )


@pytest.fixture
def pool(db):
    # Start at the previous live configuration with an independently exercised fixed payment.
    seed_paid(db)
    return Pool(db)


def test_priority_cap_time_reuse_and_pass_fail_rewards(pool):
    p = pool
    reserve = p.enrollment(0, 0, 90000)
    assert not p.rpc("pool_active")
    assert p.rpc("register", *reserve)["error"] == "collection_consent_invalid"
    assert p.claim("early") is None
    p.activate()
    assert p.rpc("pool_active") and p.rpc("register", *reserve)["status"] == "enrolled"
    a = p.register(1, 1, 12000)
    b = p.register(2, 1, 12000)
    c = p.register(3, 1, 16000)
    assert p.rpc("register", *a)["status"] == "enrolled"
    assert p.claim("a", expected=None) == {"model": p.model}
    tok = "sha256:" + digest("t")
    jobs = [p.claim(k, token=tok) for k in ("x", "y", "z", "w")]
    assert [j["registration_id"] for j in jobs] == [c[0], a[0], b[0], reserve[0]]
    assert all(j["price_microusd"] == 0 for j in jobs)
    assert p.claim("x", job=jobs[0]["job_id"], token=tok)["job_id"] == jobs[0]["job_id"]
    for i, value in enumerate((101, 205)):
        job = jobs[i]
        assert p.reserve(job, tok, "generate", 400) == "reserved"
        p.record(job, tok, "generate", value, "gen-paid-" + str(i))
        p.record(job, tok, "generate", value, "gen-paid-" + str(i))
        assert p.reserve(job, tok, "private", 600, "judge") == "reserved"
        p.record(job, tok, "private", 500, "owner-judge-" + str(i))
        assert p.finish(job, tok, passed=i == 0)["status"] == "accepted"
        assert p.finish(job, tok, passed=i == 0, epoch=202)["status"] == "accepted"
    targets = p.rpc("reward_quotes", 202)
    assert targets["job_count"] == 2 and targets["quotes_microusd"] == {
        c[1]: 127,
        a[1]: 257,
    }
    assert p.rpc("reward_quotes", 203)["job_count"] == 0
    miners = {r["hotkey"]: r for r in p.rpc("public_miners", None)}
    assert miners[c[1]]["epochs"] == {"202": 127} and miners[a[1]]["epochs"] == {
        "202": 257
    }
    # Same signed key can fund a distinct manifest immediately; its old daily quota did not reset.
    assert p.claim("new", cases=["P10-s1001"], token=tok)["registration_id"] == c[0]
    assert p.rpc("reward_quotes", 101)["job_count"] == 1
    auth = p.rpc("training_consents", jobs[0]["batch_id"])[0]
    assert (
        auth["spend_limit_microusd"],
        auth["priority"],
        auth["miner_cost_microusd"],
    ) == (16000, 1, 101)
    statement = p.rpc("statement", c[1], "live")
    grant = statement["enrollments"][0]
    assert (
        grant["available_microusd"] == 16000 - 101 - 10000
        and grant["priority"] == 1
        and "effective_max_slots_per_utc_day" not in grant
    )
    assert (
        next(
            row["price_microusd"]
            for row in statement["jobs"]
            if row["job_id"] == jobs[0]["job_id"]
        )
        == 127
    )
    participation = p.rpc("participation")
    assert next(x for x in participation["work"] if x["arm"] == "pool")["accepted"] == 2
    assert (
        next(x for x in participation["billing"] if x["arm"] == "pool")[
            "miner_microusd"
        ]
        == 306
    )


def test_lifetime_unknown_late_charge_withdrawal_and_explicit_reenrollment(pool):
    p = pool
    p.activate()
    fingerprint = "sha256:" + digest("same-funding-key")
    n = p.register(6, 1, 12000, fingerprint=fingerprint)
    standby = p.register(7, 0, 30000)
    tok = "sha256:" + digest("other")
    j = p.claim("hold", token=tok)
    assert j["registration_id"] == n[0]
    # Seed an outstanding request older than the legacy 24-hour rolling window.
    p.sql(
        """insert into public.collection_calls
        (job_id,request_id,provider,key_id,request_hash,reserved_microusd,created_at)
        values (%s,'gen','openrouter',%s,%s,7000,now()-interval '2 days')""",
        (j["job_id"], j["openrouter_key_id"], digest("gen")),
        owner=True,
    )
    p.record(j, tok, "gen", None, "gen-late")
    assert p.finish(j, tok, status="failed")["status"] == "failed"
    assert (
        p.rpc("statement", n[1], "live")["enrollments"][0]["available_microusd"] == 5000
    )
    backup_job = p.claim("next", token=tok)
    assert backup_job["registration_id"] == standby[0]
    p.rpc("revoke", n[0], n[1], "live")
    replacement = p.register(6, 1, 11000, fingerprint=fingerprint)
    assert (
        p.rpc("statement", n[1], "live")["enrollments"][0]["available_microusd"] == 4000
    )
    response_hash = digest("genresponse")
    assert (
        p.rpc(
            "settle_generation",
            j["job_id"],
            "gen",
            "gen-late",
            response_hash,
            digest("lookup"),
            3000,
        )["status"]
        == "settled"
    )
    assert (
        p.rpc("statement", n[1], "live")["enrollments"][0]["available_microusd"]
        == 11000
    )
    assert p.rpc("reward_quotes", 202)["job_count"] == 0
    live = p.claim("reuse", token=tok)
    assert live["registration_id"] == replacement[0]
    p.rpc("revoke", replacement[0], n[1], "live")
    with pytest.raises(p.driver.Error):
        p.reserve(live, tok, "after-revoke", 1)


def test_concurrent_lease_liability_and_receipt_integrity(pool):
    p = pool
    p.activate()
    normal = p.register(5, 1, 20000)
    standby = p.register(4, 0, 10000)
    tok = "sha256:" + digest("concurrency")
    with ThreadPoolExecutor(max_workers=4) as threads:
        result = list(
            threads.map(lambda x: p.claim("concurrent" + str(x), token=tok), range(4))
        )
    jobs = [j for j in result if j]
    assert len(jobs) == 3
    assert [j["registration_id"] for j in jobs].count(normal[0]) == 2
    assert [j["registration_id"] for j in jobs].count(standby[0]) == 1
    first, second = jobs[:2]
    assert p.reserve(first, tok, "g", 5000) == "reserved"
    assert p.reserve(first, tok, "another", 5001) == "budget_exhausted"
    p.record(first, tok, "g", 42, "gen-one")
    assert p.finish(first, tok)["status"] == "accepted"
    assert p.reserve(second, tok, "g", 1000) == "reserved"
    with pytest.raises(
        p.driver.Error, match="ambiguous collection provider generation"
    ):
        p.record(second, tok, "g", 42, "gen-one")
    with pytest.raises(p.driver.Error):
        p.finish(second, tok)
    p.record(second, tok, "g", 1001, "gen-over")
    with pytest.raises(p.driver.Error, match="verified settled"):
        p.finish(second, tok)
    assert p.rpc("reward_quotes", 202)["job_count"] == 1


def test_refuses_invalid_grants_and_heldout_legacy_jobs(pool):
    p = pool
    p.activate()
    for alter in (
        {"priority": 2},
        {"offer_id": "unrelated"},
        {"max_slots_per_utc_day": 6},
        {"spend_limit_microusd": 5000},
        {"terms_sha256": "sha256:" + "a" * 64},
    ):
        bad = p.enrollment(9, 1, 20000, alter=alter)
        assert p.rpc("register", *bad)["error"] == "collection_consent_invalid"
    p.register(9, 1, 20000)
    assert p.claim("eval", cases=["K01-s101"])["error"] == "invalid_pool_experiment"
    assert p.claim("preview", expected="another-model") is None
    assert p.claim("correct", cases=["P9-s101", "P9-s101"])["model"] == p.model
    with pytest.raises(p.driver.Error):
        p.sql(
            "update public.collection_pool_activation set activated_at=now()",
            owner=True,
        )


def test_pool_executes_beyond_six_daily_slots_and_stops_exactly_at_limit(pool):
    p = pool
    p.activate()
    normal = p.register(10, 1, 72)
    reserve = p.register(11, 0, 72)
    token = "sha256:" + digest("eight")
    for index in range(9):
        run = p.claim(
            "first-pack" if index < 4 else "second-pack", budget=9, token=token
        )
        assert run["registration_id"] == (normal if index < 8 else reserve)[0]
        assert p.reserve(run, token, "gen", 9) == "reserved"
        p.record(run, token, "gen", 9, "gen-eight-" + str(index))
        if index == 0:
            bad = Json(
                {"payment_rule": "verified_p7_v1", "training_status": "accepted"}
            )
            with pytest.raises(p.driver.Error, match="verified complete"):
                p.rpc(
                    "finish",
                    run["job_id"],
                    token,
                    "accepted",
                    bad,
                    "sha256:" + digest("wrong"),
                    201,
                )
        assert p.finish(run, token, passed=bool(index % 2))["status"] == "accepted"
    statement = p.rpc("statement", normal[1], "live")
    assert statement["enrollments"][0]["available_microusd"] == 0
    assert p.rpc("reward_quotes", 202)["quotes_microusd"] == {
        normal[1]: 96,
        reserve[1]: 12,
    }


@pytest.mark.parametrize(
    "http_status,should_fall_back", [(401, True), (402, True), (429, False)]
)
def test_provider_auth_or_credit_refusal_falls_back_without_releasing_unknown_bill(
    pool, http_status, should_fall_back
):
    p = pool
    p.activate()
    normal = p.register(12, 1, 25000)
    reserve = p.register(13, 0, 50000)
    token = "sha256:" + digest("provider")
    j = p.claim("provider", token=token)
    assert j["registration_id"] == normal[0]
    assert p.reserve(j, token, "g", 2000) == "reserved"
    p.record(j, token, "g", None, None, http_status)
    p.finish(j, token, status="failed")
    remaining = p.rpc("statement", normal[1], "live")["enrollments"][0][
        "available_microusd"
    ]
    assert remaining == 23000
    second = p.claim("after", token=token)
    assert second["registration_id"] == (reserve if should_fall_back else normal)[0]


def test_staggered_paid_slot_uses_healthy_pack_only_and_enforces_cap(pool, tmp_path):
    from lab_arena.collection_experiment import load_experiment
    from scripts.prepare_collection_experiment import store_private_json
    from scripts.run_locked_collection import poll_pool_experiments
    from tests.lab_arena.collection_pool_worker_test import SMALL, EXPERT, pool_raw
    from tests.lab_arena.collection_judge_test import PROMPT

    p = pool
    p.activate()
    p.register(1, 1, 100000)
    packs = []
    for family, first in (("P8", SMALL), ("P9", EXPERT)):
        raw = pool_raw(family, repeats=4)
        raw["arms"].append(dict(raw["arms"][0], model=EXPERT, id="stronger"))
        raw.update(max_slots=8, budget_microusd=800000)
        raw["routes"][EXPERT] = "coreweave/fp8"
        if first == EXPERT:
            raw["series_arm_order"] = [a["id"] for a in raw["arms"]]
            raw["arms"].reverse()
        spec = load_experiment(raw, PROMPT)
        store_private_json(tmp_path / (spec.batch_id + ".json"), raw)
        packs.append(spec)
    from lab_arena.collection_experiment import pool_program_id

    # Give the unavailable DeepSeek pack the first probe.
    cursor = tmp_path / ".last-pool-program"
    cursor.write_text(pool_program_id(packs[0].raw))
    cursor.chmod(0o600)

    def native_claim(spec, expected, job_id, token):
        arms = [a for _, a, _ in spec.slots]
        return p.rpc(
            "priced_claim",
            job_id,
            token,
            spec.batch_id,
            "live",
            p.harness,
            Json([case["scenario_id"] for case, _, _ in spec.slots]),
            Json([a["model"] for a in arms]),
            Json(dict.fromkeys(spec.raw["routes"], 0)),
            Json({a["model"]: a["budget_microusd"] for a in arms}),
            POOL_TERMS_HASH,
            expected,
        )

    observed = []

    def run(raw, *, before_claim):
        spec = load_experiment(raw, PROMPT)
        id_ = str(uuid4())
        token = "sha256:" + digest(id_)
        preview = native_claim(spec, None, id_, token)
        if not preview:
            return 0
        model = preview["model"]
        observed.append(model)
        if model == EXPERT:
            raise ValueError(
                "approved route unavailable"
            )  # No lease or cost on this pack.
        before_claim()
        assert native_claim(spec, model, id_, token)["model"] == SMALL
        return 1

    assert poll_pool_experiments(PROMPT, tmp_path, None, run) == 1 and observed == [
        EXPERT,
        SMALL,
    ]
    # The healthy job's full ceiling now holds the sole key's signed balance.
    for spec in packs:
        assert (
            native_claim(spec, None, str(uuid4()), "sha256:" + digest("preview"))
            is None
        )
    assert cursor.read_text() == pool_program_id(packs[0].raw)
