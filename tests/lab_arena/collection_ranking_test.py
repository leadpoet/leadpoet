from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import pytest
from lab_arena import (
    collection_intake as intake,
    collection_judge as j,
    collection_ranking as ranking,
    store,
)
from lab_arena.service import LocalObjectStore
from lab_arena.collection_workflow import store_artifact
from tests.lab_arena.collection_intake_test import (
    manager,
    registration,
    SAMPLE,
    COMMIT,
    PRICES,
    BUDGETS,
)
from tests.lab_arena.collection_judge_test import POLICY, PROMPT, Caller, packet
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        (
            *DEFAULT_MIGRATIONS,
            "720-locked-collection-intake.sql",
            "724-collection-scenario-ranking.sql",
            "727-collection-synthetic-funding.sql",
        )
    )


def test_last_concurrent_finisher_owns_one_immutable_paid_ranking(database, tmp_path):
    driver, dsn = database

    def connect():
        c = driver.connect(**dsn)
        c.autocommit = True
        return c

    transport = store.PsycopgTransport(connect, role="lab_arena_service")
    app = intake.CollectionIntake(transport, manager(), "live")
    for name in "ABCDEF":
        signed, creds = registration(name)
        app.register(signed, signed["hotkey"], creds)
    jobs = [app.claim("b-1", SAMPLE, COMMIT, PRICES, BUDGETS) for i in range(6)]
    policy = replace(POLICY, batch_id="b-1")
    objects = LocalObjectStore(tmp_path)
    broker = type("Broker", (), {})()
    broker.judge = Caller()
    items = []
    for job in jobs:
        rid = job["job_id"]
        p = packet(rid)
        p["scenario_id"] = job["scenario_id"]
        a = j.judge_run(
            policy,
            dict(run_id=rid, primary="fable", overlap=False),
            p,
            Caller(),
            prompt=PROMPT,
        )
        store_artifact(objects, "b-1", rid, "judge-input", p)
        store_artifact(objects, "b-1", rid, "judge-output", a)
        items.append((job, p, a, policy, PROMPT, broker, objects, transport))
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(lambda args: ranking.rank_pack_if_ready(*args), items))
    (idx,) = [i for i, r in enumerate(outputs) if r]
    assert len(broker.judge.requests) == 2
    record = json.loads(objects.get(outputs[idx]["key"]))
    assert record["status"] == "ranked" and len(record["labels"]) == 3
    assert (
        ranking.rank_pack_if_ready(*items[idx]) == outputs[idx]
        and len(broker.judge.requests) == 2
    )
    old = items[idx]
    p = dict(old[1], facts="edited")
    with pytest.raises(Exception, match="changed"):
        ranking.rank_pack_if_ready(old[0], p, *old[2:])
    with pytest.raises(Exception, match="inactive_collection_lease"):
        ranking.rank_pack_if_ready(dict(old[0], lease_token="9" * 64), *old[1:])
    transport.close()
