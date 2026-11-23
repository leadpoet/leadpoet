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


def test_candidates_have_truthful_ids_double_judgments_and_a_release_hold():
    policy = load_batch(manifest(), PROMPT)
    assert judge.BatchPolicy(**asdict(policy)) == policy
    run = packet()
    assignments = [
        dict(run_id=run["run_id"], miner="miner", generator=MODELS[0], harness="a")
    ]
    schedule = judge.assign_judges(policy, assignments)
    call = Caller()
    result = judge.judge_run(policy, schedule[run["run_id"]], run, call, prompt=PROMPT)
    assert {v["judge_version"] for v in result["records"]} == set(PAIR)
    assert {v["judge"] for v in result["records"]} == {"kimi", "mistral"}
    assert {v["max_tokens"] for v in call.requests} == {600, 1200}
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
    with pytest.raises(ValueError, match="own generator"):
        judge.assign_judges(
            policy,
            [dict(run_id="r", miner="m", generator=next(iter(PAIR)), harness="h")],
        )
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
        assert cursor.fetchall() == [("judge", 10), ("judge", 10)]
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
    assert len(sent) == 2


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
