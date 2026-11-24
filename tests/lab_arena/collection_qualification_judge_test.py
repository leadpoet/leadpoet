"""The qualification pair can collect calibration evidence but cannot release a batch."""

from dataclasses import asdict

import pytest

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import GENERATOR_ROUTES, MODELS
from scripts.run_locked_collection import load_batch
from tests.lab_arena.collection_judge_test import Caller, PROMPT, packet
from tests.lab_arena.locked_collection_pipeline_test import database, transport


PAIR = {
    "moonshotai/kimi-k3": "deepinfra/mxfp4",
    "mistralai/mistral-large-2512": "mistral/zdr",
}


def manifest():
    return dict(
        mode="pilot",
        name="cal",
        seed="blind",
        scenario_ids=["test"],
        routes={**{m: r.tag for m, r in GENERATOR_ROUTES.items()}, **PAIR},
        budgets_microusd=dict.fromkeys(MODELS, 1_000_000),
        control_slots={},
        drift_slots={},
    )


def test_candidates_have_truthful_independent_judgments_and_a_release_hold():
    policy = load_batch(manifest(), PROMPT)
    assert judge.BatchPolicy(**asdict(policy)) == policy
    run = packet()
    assignments = [
        dict(run_id=run["run_id"], miner="miner", generator=MODELS[0], harness="a")
    ]
    schedule = judge.assign_judges(policy, assignments)
    call = Caller()
    result = judge.judge_run(policy, schedule[run["run_id"]], run, call, prompt=PROMPT)
    assert {v["judge_version"] for v in result["records"]} == {policy.version("mistral")}
    assert {v["judge"] for v in result["records"]} == {"mistral"}
    assert {v["max_tokens"] for v in call.requests} == {600}
    report = judge.quality_report(
        policy,
        assignments=assignments,
        assessments=[result],
        human_marks={},
        planted={},
        drift={},
    )
    assert "judge_qualification" in report["holds"]
    assert not report["quality_ready"]


def test_candidate_pair_cannot_enter_live_or_change_route_or_grade_own_model():
    with pytest.raises(ValueError, match="qualification"):
        load_batch({**manifest(), "mode": "live"}, PROMPT)
    raw = manifest()
    raw["routes"][next(iter(PAIR))] = "any"
    with pytest.raises(ValueError, match="judge route"):
        load_batch(raw, PROMPT)
    policy = load_batch(manifest(), PROMPT)
    independent = judge.assign_judges(
        policy, [dict(run_id="r", miner="m", generator=next(iter(PAIR)), harness="h")]
    )["r"]
    assert policy.version(independent["primary"]) != next(iter(PAIR)) and not independent["overlap"]
    mixed = dict(asdict(policy), astra_version="openai/gpt-6.2-astra")
    with pytest.raises(ValueError, match="judge models"):
        judge.BatchPolicy(**mixed)


@pytest.mark.parametrize("qualification", [True, False])
def test_metered_judges_use_owner_funding_and_strict_routes(
    transport, database, qualification
):
    import json
    from lab_arena import broker
    from lab_arena.collection_broker import (
        CollectionBroker,
        CollectionCallError,
        batch_versions,
    )
    from tests.lab_arena.locked_collection_pipeline_test import claim, registered

    intake = registered(transport, "W" if qualification else "X")
    job = claim(
        intake,
        "qualification-broker" if qualification else "frontier-broker",
        "B1-s2027",
    )
    raw = manifest()
    if not qualification:
        raw["routes"] = {
            **{m: r.tag for m, r in GENERATOR_ROUTES.items()},
            "anthropic/claude-fable-5.2": "test-vertex",
            "openai/gpt-6.2-astra": "test-azure",
        }
    policy = load_batch(raw, PROMPT)
    routes = raw["routes"]
    prices = broker.price_table_from_models_response(
        {
            "data": [
                {
                    "id": model,
                    "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                }
                for model in routes
            ]
        },
        tuple(routes),
        fetched_at="2026-11-23T00:00:00Z",
    )
    prices["endpoint_providers"] = dict.fromkeys(routes, "test provider")
    sent = []

    class LocalTransport:
        def send(self, **request):
            assert request["headers"]["authorization"] == "Bearer test-judge-account"
            body = json.loads(request["body"])
            sent.append(body)
            model = body["model"]
            selected = {
                "selected": True,
                "provider": "test provider",
                "model": batch_versions(routes)[model],
            }
            verdict = Caller()(**body)
            return broker.ProviderResponse(
                200,
                {},
                json.dumps(
                    {
                        "id": "gen-" + job["job_id"] + "-" + str(len(sent)),
                        "provider": selected["provider"],
                        "model": model,
                        "openrouter_metadata": {"endpoints": {"available": [selected]}},
                        "usage": {"cost": 0.00001},
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": verdict},
                            }
                        ],
                    }
                ).encode(),
            )

    channel = CollectionBroker(
        intake,
        transport,
        LocalTransport(),
        job,
        prices,
        routes=routes,
        judge_key="test-judge-account",
    )
    run = packet(job["job_id"])
    assignment = judge.assign_judges(
        policy,
        [
            dict(
                run_id=job["job_id"],
                harness="public",
                miner="miner",
                generator=job["model"],
            )
        ],
    )[job["job_id"]]
    result = judge.judge_run(policy, assignment, run, channel.judge, prompt=PROMPT)
    assert result["status"] == "accepted"
    assert channel.usage()["settled_microusd"] == 0
    driver, dsn = database
    with driver.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT c.provider,r.cost_microusd FROM public.collection_calls c "
            "JOIN public.collection_call_receipts r USING(job_id, request_id) "
            "WHERE c.job_id = %s",
            (job["job_id"],),
        )
        assert cursor.fetchall() == [("judge", 10)] * (1 if qualification else 2)
    for body in sent:
        settings = body["provider"]
        assert settings["zdr"] and settings["enforce_distillable_text"]
        if qualification:
            assert settings["require_parameters"]
        assert settings["data_collection"] == "deny" and not settings["allow_fallbacks"]
        assert settings["only"] == [routes[body["model"]]]
    channel.job = dict(job, mode="live")
    with pytest.raises(CollectionCallError, match="qualification"):
        channel.judge(
            request_id="live", model="moonshotai/kimi-k3", messages=[], max_tokens=100
        )
    assert len(sent) == (1 if qualification else 2)

    channel.job = job

    class Unavailable:
        def send(self, **request):
            if status == 200:
                malformed = {
                    "id": "gen-no-metadata-" + job["job_id"],
                    "model": json.loads(request["body"])["model"],
                    "usage": {"cost": 0.00001},
                    "openrouter_metadata": {"endpoints": ["invalid"]},
                }
                return broker.ProviderResponse(
                    status, {}, json.dumps(malformed).encode()
                )
            return broker.ProviderResponse(
                status, {}, b'{"error":{"message":"unavailable"}}'
            )

    channel.transport = Unavailable()
    for status in (401, 402, 403, 404, 429, 503, 200):
        evidence = packet(job["job_id"])
        evidence["steps"][0]["evidence"] = f"Different scoped control {status}"
        gap = judge.judge_run(
            policy, assignment, evidence, channel.judge, prompt=PROMPT
        )
        expected = "identity_mismatch" if status == 200 else "http_error"
        assert gap["verifier_unavailable"]["reason"] == expected and not gap["records"]
        with driver.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT http_status, cost_microusd FROM public.collection_call_receipts "
                "WHERE job_id=%s AND request_id=%s",
                (job["job_id"], gap["verifier_unavailable"]["request_id"]),
            )
            assert cursor.fetchone() == (status, 10 if status == 200 else None)
        assert channel.usage()["settled_microusd"] == 0


def test_restart_requires_the_original_generator_before_finishing_launch(monkeypatch):
    from types import SimpleNamespace
    from scripts import run_locked_collection as worker
    from scripts import collection_batch

    policy = judge.BatchPolicy("restart-batch", "seed", judge.digest(PROMPT))
    pending = {
        "job_id": "launchjob",
        "status": "ready",
        "batch_id": policy.batch_id,
        "model": MODELS[0],
        "artifact_sha256": "sha256:" + "4" * 64,
        "harness_commit": worker.CHAMPION_REV,
        "miner_hotkey": "miner",
        "scenario_id": "B1-s2027",
        "owner_coldkey": "owner",
    }
    writes = []

    def rpc(name, args):
        if name == "collection_reward_schedule_v1":
            return {
                "first_epoch": 101,
                "staged": True,
                "pending": pending,
                "pilot_evidence": {
                    "harness_commit": worker.CHAMPION_REV,
                    "builder_commit": worker.POET_REV,
                    "judge_prompt_sha256": policy.prompt_sha256,
                },
            }
        writes.append(name)
        return {"status": "accepted"}

    def missing(ref, size):
        raise FileNotFoundError("launch object unavailable")

    monkeypatch.setattr(collection_batch, "cutover_anchor", lambda *a, **kw: {})
    with pytest.raises(FileNotFoundError, match="launch object"):
        worker.run_cohort(
            raw={"budgets_microusd": {}},
            policy=policy,
            prompt=PROMPT,
            intake=None,
            database=SimpleNamespace(rpc=rpc),
            objects=SimpleNamespace(get_bounded=missing),
            transport=None,
            pricing=None,
            epoch=lambda: 100,
            sources=None,
            owner_lookup=None,
            judge_key="fixture",
            max_jobs=1,
        )
    assert not writes


def test_first_live_job_can_resume_after_its_generator_is_restored(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from lab_arena.collection_workflow import store_artifact
    from lab_arena.service import LocalObjectStore
    from scripts import collection_batch, run_locked_collection as worker

    policy = judge.BatchPolicy("restart-positive", "seed", judge.digest(PROMPT))
    rid = "ready-example"
    row = packet(rid)
    row["checks"] = dict.fromkeys(
        ("answered", "shape", "privacy", "training_split"), True
    )
    pending = dict(
        job_id=rid,
        batch_id=policy.batch_id,
        scenario_id=row["scenario_id"],
        model=MODELS[0],
        harness_commit=worker.CHAMPION_REV,
        miner_hotkey="miner",
        owner_coldkey="owner",
        status="ready",
    )
    assessment = judge.judge_run(
        policy,
        {"run_id": rid, "primary": "fable", "overlap": True},
        row,
        Caller(),
        prompt=PROMPT,
    )
    objects = LocalObjectStore(tmp_path)

    def store(kind, obj):
        return store_artifact(objects, policy.batch_id, rid, kind, obj)

    generator = {"run": rid, "teacher": MODELS[0]}
    manifest = {
        **pending,
        "builder_commit": worker.POET_REV,
        "judge_policy": asdict(policy),
        "generator": store("generator", generator),
        "packet": store("judge-input", row),
        "assessment": store("judge-output", assessment),
    }
    pending["artifact_sha256"] = store("manifest", manifest)["sha256"]
    approved = dict(
        harness_commit=worker.CHAMPION_REV,
        builder_commit=worker.POET_REV,
        judge_prompt_sha256=policy.prompt_sha256,
    )
    calls = []
    db = SimpleNamespace(
        rpc=lambda name, args: calls.append(name) or {"status": "accepted"}
    )
    monkeypatch.setattr(collection_batch, "cutover_anchor", lambda *a, **kw: {})

    def unavailable_generator(ref, limit):
        if "/generator-" in ref:
            raise FileNotFoundError("generator was not uploaded")
        return objects.get_bounded(ref, limit)

    unavailable = SimpleNamespace(get_bounded=unavailable_generator)
    with pytest.raises(FileNotFoundError):
        worker.finish_launch(db, pending, unavailable, policy, approved)
    assert not calls
    worker.finish_launch(db, pending, objects, policy, approved)
    assert calls == ["collection_launch_finish_v1"]


def test_rejudge_ready_reuses_the_generator_and_the_completed_primary(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from uuid import uuid4
    import json
    from lab_arena import broker
    from lab_arena.collection_broker import VERSIONS
    from lab_arena.collection_workflow import store_artifact
    from lab_arena.service import LocalObjectStore
    from scripts import collection_batch, run_locked_collection as worker

    policy = judge.BatchPolicy("frozen-live", "seed", judge.digest(PROMPT))
    rid, attempt = str(uuid4()), str(uuid4())
    p = packet(rid)
    p["checks"] = dict.fromkeys(
        ("answered", "shape", "privacy", "training_split"), True
    )
    pending = dict(
        job_id=rid,
        batch_id=policy.batch_id,
        scenario_id=p["scenario_id"],
        model=MODELS[0],
        harness_commit=worker.CHAMPION_REV,
        miner_hotkey="miner",
        owner_coldkey="owner",
        status="ready",
    )
    source = dict(
        run_id=rid,
        generator=MODELS[0],
        harness=pending["harness_commit"],
        miner="miner",
    )
    scheduled = judge.assign_judges(policy, [source])[rid]
    scheduled["overlap"] = True
    answers = Caller()

    def first_attempt(**request):
        if answers.requests:
            raise judge.JudgeUnavailable(
                request["model"], request["request_id"], "http_error"
            )
        return answers(**request)

    prior = judge.judge_run(policy, scheduled, p, first_attempt, prompt=PROMPT)
    missing = prior["verifier_unavailable"]["model"]
    obj = LocalObjectStore(tmp_path)

    def store(kind, value):
        return store_artifact(obj, policy.batch_id, rid, kind, value)

    m = {
        **pending,
        "builder_commit": worker.POET_REV,
        "judge_policy": asdict(policy),
        "generator": store("generator", {"run": rid, "teacher": MODELS[0]}),
        "packet": store("judge-input", p),
        "assessment": store("judge-output", prior),
    }
    old = store("manifest", m)["sha256"]
    approved = dict(
        harness_commit=worker.CHAMPION_REV,
        builder_commit=worker.POET_REV,
        judge_prompt_sha256=policy.prompt_sha256,
    )
    pending["artifact_sha256"] = old
    calls, requests = [], []
    routes = {m: r.tag for m, r in GENERATOR_ROUTES.items()}
    routes.update({policy.version(r): "test" for r in policy.roles})

    def rpc(name, args):
        if name == "collection_reward_schedule_v1":
            return {
                "staged": True,
                "pending": dict(pending),
                "pilot_evidence": approved,
            }
        if name == "collection_launch_finish_v1":
            calls.append("finish")
            return {"status": "accepted"}
        assert (
            args["p_source"] == old
            and args["p_attempt"] == attempt
            and args["p_job"] == rid
        )
        if name == "collection_reserve_launch_judge_v1":
            requests.append(args["p_request"])
            return {"status": "reserved"}
        if name == "collection_record_launch_judge_v1":
            assert (
                args["p_request"] == requests[-1]
                and args["p_cost"] == 10
                and args["p_status"] == 200
            )
            return {"status": "recorded"}
        assert (
            name == "collection_finish_launch_review_v1"
            and args["p_requests"] == requests
        )
        pending["artifact_sha256"] = args["p_artifact"]
        return {"artifact_sha256": args["p_artifact"]}

    db = SimpleNamespace(rpc=rpc)

    def prices(routes, *, models):
        assert tuple(models) == (missing,)
        pricing = broker.price_table_from_models_response(
            {
                "data": [
                    {
                        "id": missing,
                        "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                    }
                ]
            },
            tuple(models),
            fetched_at="2026-11-23T00:00:00Z",
        )
        pricing["endpoint_providers"] = {missing: "test"}
        return pricing

    monkeypatch.setattr(worker, "endpoint_prices", prices)

    def send(**req):
        assert req["headers"]["authorization"] == "Bearer owner-test"
        body = json.loads(req["body"])
        assert body["model"] == missing and body["provider"]["enforce_distillable_text"]
        calls.append(missing)
        return broker.ProviderResponse(
            200,
            {},
            json.dumps(
                {
                    "id": "gen-recovered",
                    "model": missing,
                    "provider": "test",
                    "openrouter_metadata": {
                        "endpoints": {
                            "available": [
                                {
                                    "selected": True,
                                    "provider": "test",
                                    "model": VERSIONS[missing],
                                }
                            ]
                        }
                    },
                    "usage": {"cost": 0.00001},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": answers(**body)},
                        }
                    ],
                }
            ).encode(),
        )

    transport = SimpleNamespace(send=send)
    monkeypatch.setattr(collection_batch, "cutover_anchor", lambda *args, **kwargs: {})
    worker.rejudge_ready(
        db,
        obj,
        transport,
        {"routes": routes},
        policy,
        PROMPT,
        "owner-test",
        rid,
        attempt,
    )
    assert calls == [missing, "finish"] and len(requests) == 1
    updated = worker.ready_launch_evidence(pending, obj, policy, approved)
    assert updated["assessment"]["records"][:1] == prior["records"]
    assert len(updated["assessment"]["records"]) == 2 and updated["packet"] == p
    with pytest.raises(ValueError, match="only an unavailable verifier"):
        worker.rejudge_ready(
            db,
            obj,
            None,
            {"routes": {}},
            policy,
            PROMPT,
            "owner-test",
            rid,
            str(uuid4()),
        )
    assert calls == [missing, "finish"]
