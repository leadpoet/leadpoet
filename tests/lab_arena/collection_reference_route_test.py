from copy import deepcopy
from pathlib import Path

import pytest
from lab_arena import collection_judge as judge, store
from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
from lab_arena.collection_intake import (
    GENERATOR_ROUTES,
    MODELS,
    SYNTHETIC_TRAINING_TERMS_HASH,
    IntakeError,
)
from lab_arena.collection_workflow import COMPARISON_CAPTURE
from scripts.run_locked_collection import load_batch
from tests.lab_arena.collection_judge_test import PROMPT, SPLIT
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.locked_collection_pipeline_test import claim, registered


def settings():
    routes = {m: r.tag for m, r in GENERATOR_ROUTES.items()}
    routes.update({m: r.route for m, r in judge.QUALIFICATION_JUDGES.items()})
    return dict(
        name="private-reference",
        seed="seed",
        scenario_ids=["P7-s40401"],
        mode="pilot",
        routes=routes,
        budgets_microusd=dict.fromkeys(MODELS, 4000000),
        capture_version=COMPARISON_CAPTURE,
        terms_sha256=SYNTHETIC_TRAINING_TERMS_HASH,
        split_manifest=SPLIT,
        control_slots={},
        drift_slots={},
        control_artifacts={},
    )


def test_private_route_requires_same_origin_pack_model_and_predeclared_frontier():
    from scripts import collection_batch

    original = settings()
    parent = load_batch(original, PROMPT)
    new = deepcopy(original)
    new["routes"][REFERENCE_MODEL] = REFERENCE_ROUTE
    with pytest.raises(ValueError, match="unverified generator"):
        load_batch(new, PROMPT)
    new["reference_successor"] = {
        "parent": parent.batch_id,
        "next_slot": 1,
        "owner_witness_sha256": "sha256:" + "b" * 64,
    }
    successor = load_batch(new, PROMPT)
    assert (
        successor.batch_id.startswith("rs-") and successor.batch_id != parent.batch_id
    )
    for mutation in (
        {"mode": "live"},
        {"scenario_ids": ["P7-s40402"]},
        {"budgets_microusd": dict.fromkeys(MODELS, 10000)},
        {"reference_successor": dict(new["reference_successor"], next_slot=2)},
    ):
        with pytest.raises(ValueError):
            load_batch(dict(new, **mutation), PROMPT)
    modified = deepcopy(new)
    modified["routes"][REFERENCE_MODEL] = "another/route"
    with pytest.raises(ValueError, match="unverified generator"):
        load_batch(modified, PROMPT)
    with pytest.raises(ValueError, match="private reference"):
        collection_batch.source_bundle(new, PROMPT, None, None)
    with pytest.raises(ValueError, match="independent reviewer"):
        collection_batch.write_exports(new, PROMPT, None, None, None)


def test_successor_keeps_parent_slot_and_daily_funding_history():
    directory = Path(__file__).resolve().parents[2] / "scripts"
    migrations = tuple(
        next(directory.glob(f"{n}-*.sql")).name
        for n in (
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
        )
    )
    for pg, dsn in database_with_lab_arena_migration(DEFAULT_MIGRATIONS + migrations):

        def connect():
            c = pg.connect(**dsn)
            c.autocommit = True
            return c

        db = store.PsycopgTransport(connect, role="lab_arena_service")
        payer = registered(db, "K")
        parent = "reference-short"
        new = "rs-" + "b" * 64
        scenario = "P7-s40401"
        j = claim(payer, parent, scenario)
        assert j["slot"] == 0
        insert = "INSERT INTO public.collection_reference_successions(parent_id,successor_id,next_slot,model,provider_tag,model_revision,witness_sha256) VALUES (%s,%s,%s,%s,%s,%s,%s)"
        params = [
            parent,
            new,
            1,
            REFERENCE_MODEL,
            REFERENCE_ROUTE,
            GENERATOR_ROUTES[REFERENCE_MODEL].revision,
            "sha256:" + "c" * 64,
        ]
        with pg.connect(**dsn) as c, c.cursor() as q:
            with pytest.raises(pg.Error, match="parent frozen slot"):
                q.execute(insert, params)
        payer.complete(j["job_id"], j["lease_token"], "failed", {}, None, 1)
        # Finish the parent's one replacement, preserving the actual historical slot.
        retry = claim(payer, parent, scenario)
        assert retry["attempt"] == 2 and retry["slot"] == 0
        payer.complete(retry["job_id"], retry["lease_token"], "failed", {}, None, 1)
        with pg.connect(**dsn) as c, c.cursor() as q:
            q.execute(insert, params)
        unknown = "rs-" + "a" * 64
        with pytest.raises(IntakeError, match="unwitnessed_reference_successor"):
            claim(payer, unknown, scenario)
        assert claim(payer, parent, scenario) is None
        assert (
            claim(payer, new, scenario) is None
        )  # Already used today's distinct slot.
        other = registered(db, "L")
        successor = claim(other, new, scenario)
        assert (
            successor["batch_id"],
            successor["slot"],
            successor["attempt"],
            successor["model"],
        ) == (new, 1, 1, REFERENCE_MODEL)
        assert (
            successor["price_microusd"] == 0
            and successor["budget_microusd"] == j["budget_microusd"]
        )
        with pg.connect(**dsn) as c, c.cursor() as q:
            q.execute(
                "SELECT count(*),max(slot) FROM public.collection_jobs WHERE batch_id=%s",
                (parent,),
            )
            assert q.fetchone() == (2, 0)
            with pytest.raises(pg.Error, match="immutable"):
                q.execute("DELETE FROM public.collection_reference_successions")
        db.close()


def test_partial_successor_only_exports_its_real_absolute_slots():
    from scripts.collection_batch import assigned_export_slots

    assert assigned_export_slots(6, 1, [1, 2], [1, 2, 3]) == {1, 2}
    assert assigned_export_slots(6, 1, [1], [1, 2]) == {1}
    for actual, registered in [
        ([0, 1], [0, 1]),
        ([2], [2]),
        ([1, 1], [1, 2]),
        ([1, 3], [1, 3]),
        ([1, 6], [1, 6]),
    ]:
        with pytest.raises(ValueError):
            assigned_export_slots(6, 1, actual, registered)


def test_complete_measured_route_is_frozen_before_slot_zero_and_rechecks_immutable_witness(
    tmp_path, monkeypatch
):
    import hashlib
    from lab_arena import collection_route_witness as w
    from lab_arena.collection_workflow import store_artifact, comparison_revisions
    from lab_arena.service import LocalObjectStore

    objects = LocalObjectStore(tmp_path / "objects")
    raw = settings()
    raw["routes"][REFERENCE_MODEL] = REFERENCE_ROUTE
    agent, builder = comparison_revisions(raw)

    def store(kind, doc):
        return store_artifact(objects, w.OWNER_BATCH, w.OWNER_RUN, kind, doc)

    generator = {
        "run": w.OWNER_RUN,
        "teacher": REFERENCE_MODEL,
        "task": "P7-s40401",
        "evidence": {"grade": {"score": 1, "hard_fail": None}},
    }
    links = {
        "generator": store("generator", generator),
        "workflow-output": store(
            "workflow-output",
            {
                "run_id": w.OWNER_RUN,
                "model": REFERENCE_MODEL,
                "scenario_id": "P7-s40401",
                "finish": "answered",
                "harness_commit": agent,
                "builder_commit": builder,
            },
        ),
        "provider-0001": store(
            "provider-0001",
            {
                "run_id": w.OWNER_RUN,
                "request_id": "generate-1",
                "response": {
                    "id": "abc",
                    "model": REFERENCE_MODEL,
                    "provider": "CoreWeave",
                },
            },
        ),
    }
    doc = {
        "schema": "collection.owner-route-witness.v1",
        "owner_only": True,
        "run_id": w.OWNER_RUN,
        "model": REFERENCE_MODEL,
        "revision": GENERATOR_ROUTES[REFERENCE_MODEL].revision,
        "provider": "CoreWeave",
        "route": REFERENCE_ROUTE,
        "quantization": "fp8",
        "native_checks_passed": True,
        "finish": "answered",
        "scope": "P7-s40401",
        "objects": links,
        "cap_microusd": 2000,
        "usage_microusd": 22,
        "receipts": [
            {
                "request": "generate-1",
                "status": 200,
                "model": REFERENCE_MODEL,
                "source_revision": GENERATOR_ROUTES[REFERENCE_MODEL].revision,
                "provider": "CoreWeave",
                "route": REFERENCE_ROUTE,
                "gen_sha256": hashlib.sha256(b"abc").hexdigest(),
                "known_microusd": 22,
                "returned_reasoning": True,
            }
        ],
        "strict_terms": {
            "author_distillable": True,
            "fallback": False,
            "request_fee": 0,
            "safety_retention": False,
            "zdr": True,
        },
    }
    sha = store("route-witness", doc)["sha256"]
    monkeypatch.setattr(w, "WITNESS", sha)
    raw["route_witness_sha256"] = sha
    complete = load_batch(raw, PROMPT)
    assert (
        complete.batch_id.startswith("rq-")
        and w.verified_witness(raw, objects)["provider_route"] == REFERENCE_ROUTE
    )
    with pytest.raises(ValueError, match="witnessed"):
        load_batch(dict(raw, mode="live", panel_qualification_sha256="b" * 64), PROMPT)
    changed = dict(doc, strict_terms=dict(doc["strict_terms"], fallback=True))
    sha = store("route-witness", changed)["sha256"]
    monkeypatch.setattr(w, "WITNESS", sha)
    raw["route_witness_sha256"] = sha
    with pytest.raises(ValueError, match="approved private route"):
        w.verified_witness(raw, objects)
    assert load_batch(raw, PROMPT).batch_id != complete.batch_id
