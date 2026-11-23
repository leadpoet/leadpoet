"""Cross-repository exercise. Use the exact committed generator and scenario source trees."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lab_arena import broker, collection_judge as judge, contracts, store
from lab_arena.collection_broker import CollectionBroker, CollectionCallError, VERSIONS
from lab_arena.collection_intake import CollectionIntake, MODELS
from lab_arena.collection_worker import process_job
from lab_arena.collection_workflow import CHAMPION_REV, build_scenario, workflow_sources
from lab_arena.service import LocalObjectStore
from tests.lab_arena import collection_intake_test as intake_fixture
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

MIGRATIONS = (
    "179-lab-arena-v1.sql",
    "720-locked-collection-intake.sql",
    "722-collection-call-budget.sql",
    "724-collection-scenario-ranking.sql",
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture(scope="module")
def transport(database):
    driver, dsn = database

    def connect():
        c = driver.connect(**dsn)
        c.autocommit = True
        return c

    t = store.PsycopgTransport(connect, role="lab_arena_service")
    yield t
    t.close()


def registered(db, hotkeychar):
    intake = CollectionIntake(db, intake_fixture.manager(), "pilot")
    signed, credentials = intake_fixture.registration(hotkeychar, "pilot")
    intake.register(signed, "5" + "Y" * 47, credentials)
    return intake


def claim(enrolled, batch, scenario):
    return enrolled.claim(
        batch,
        [scenario],
        CHAMPION_REV,
        dict.fromkeys(MODELS, 0),
        dict.fromkeys(MODELS, 4000000),
    )


def setup_broker(enrolled, db, job, transport):
    models = (*MODELS, "anthropic/claude-fable-5.2", "openai/gpt-6.2-astra")
    pricing = broker.price_table_from_models_response(
        {
            "data": [
                {"id": m, "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}}
                for m in models
            ]
        },
        fetched_at="2026-11-23T00:00:00Z",
    )
    return CollectionBroker(
        enrolled,
        db,
        transport,
        job,
        pricing,
        routes=dict.fromkeys(models, "test"),
        judge_key="private-test-credential-only",
    )


class StandIn:
    def __init__(self, calls=(), answer="Done."):
        self.calls = list(calls)
        self.answer = answer
        self.sends = []

    def send(self, **request):
        body = json.loads(request["body"])
        self.sends.append(body)
        model = body["model"]
        if model in ("openai/gpt-6.2-astra", "anthropic/claude-fable-5.2"):
            env = json.loads(body["messages"][-1]["content"])
            if "runs" in env:
                labels = sorted(env["runs"])
                result = dict(
                    env["identity"],
                    order=labels,
                    margin="narrow",
                    why_first="Fewer calls.",
                    loser_tags={x: "needless_search" for x in labels[1:]},
                )
            else:
                result = dict(
                    env["identity"],
                    scores=dict.fromkeys(judge.PARTS, 3),
                    evidence={},
                    failure_tags=[],
                    hard_steps=[],
                    first_bad_step=None,
                    reasoning="none_shown",
                    reasoning_matches_actions=True,
                    unsupported_claims=[],
                    confidence="high",
                    cannot_judge=False,
                    cannot_judge_reason=None,
                )
            message = {"role": "assistant", "content": json.dumps(result)}
        elif self.calls:
            action, params = self.calls.pop(0)
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "action" + str(len(self.sends)),
                        "type": "function",
                        "function": {"name": action, "arguments": json.dumps(params)},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": self.answer}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        data = {
            "id": "gen-" + str(len(self.sends)),
            "provider": "test",
            "model": model,
            "openrouter_metadata": {
                "endpoints": {
                    "available": [
                        {"selected": True, "model": VERSIONS[model], "provider": "test"}
                    ]
                }
            },
            "usage": {"cost": 0.00001},
            "choices": [{"message": message, "finish_reason": finish}],
        }
        return broker.ProviderResponse(200, {}, json.dumps(data).encode())


def test_replayed_call_never_spends_twice_and_cap_cannot_be_bypassed(transport):
    enrolled = registered(transport, "A")
    job = claim(enrolled, "budget-proof", "B1-s2027")
    standin = StandIn()
    b = setup_broker(enrolled, transport, job, standin)
    body = {
        "model": job["model"],
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 16,
    }
    b.model_call("step-1", body)
    with pytest.raises(CollectionCallError, match="already_reserved"):
        b.model_call("step-1", body)
    with pytest.raises(CollectionCallError, match="assignment"):
        b.model_call("step-2", dict(body, model="unassigned/model"))
    assert len(standin.sends) == 1
    assert b.usage()["settled_microusd"] == 10
    denied = b._rpc(
        "collection_reserve_call_v1",
        "overspend",
        p_provider="openrouter",
        p_hash="a" * 64,
        p_amount=4000000,
    )
    assert denied["status"] == "budget_exhausted"


def test_full_six_slot_cohort_and_public_review_projection(transport, tmp_path):
    agent, poet = (
        os.getenv(k)
        for k in ("COLLECTION_TEST_AGENT_REPO", "COLLECTION_TEST_POET_REPO")
    )
    if not agent or not poet:
        pytest.skip("requires both authorized source git repositories")
    prompt = {
        "rubric_version": "control",
        "tag_version": "control",
        "run_instructions": "Grade synthetic controls.",
        "ranking_instructions": "Order the examples.",
        "rubric": "Use the five score dimensions.",
        "examples": {
            str(i): ["Synthetic control A.", "Synthetic control B."] for i in range(4)
        },
    }
    policy = judge.BatchPolicy(
        "six-slot-control", "fixed", judge.digest(prompt), pilot=True
    )
    observed = []
    with workflow_sources(agent, poet) as sources:
        for key in "BCDEFG":
            enrolled = registered(transport, key)
            job = claim(enrolled, policy.batch_id, "E4-s2027")
            observed.append(job["model"])
            t = build_scenario(sources[1], job["scenario_id"])
            ref = sources[1].perfect(t)
            calls = list(ref["calls"]) + [
                ("write_file", {"path": p, "content": v})
                for p, v in ref["files"].items()
            ]
            standin = StandIn(calls, ref["answer"])
            metered = setup_broker(enrolled, transport, job, standin)
            assignment = {
                "run_id": job["job_id"],
                "generator": job["model"],
                "harness": CHAMPION_REV,
                "miner": job["miner_hotkey"],
                "slot": job["slot"],
                "cohort": job["slot"],
            }
            result = process_job(
                job=job,
                assignment=judge.assign_judges(policy, [assignment])[job["job_id"]],
                policy=policy,
                prompt=prompt,
                intake=enrolled,
                broker=metered,
                objects=LocalObjectStore(tmp_path),
                sources=sources,
                epoch=lambda: 10,
            )
            generator = json.loads((tmp_path / result["generator"]["key"]).read_bytes())
            assert (
                generator["teacher"] == job["model"]
                and len(generator["label_turns"]) > 1
            )
            assert "judge" not in generator and "scores" not in generator
            assessment = json.loads(
                (tmp_path / result["assessment"]["key"]).read_bytes()
            )
            assert (
                assessment["status"] == "accepted" and len(assessment["records"]) == 2
            )
            statement = enrolled.statement(job["miner_hotkey"])
            assert statement["jobs"][0]["status"] == "accepted" and "prompt" not in str(
                statement
            )
        assert observed == list(MODELS)
