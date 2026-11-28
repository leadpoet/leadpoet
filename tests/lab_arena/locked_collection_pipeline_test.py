"""Cross-repository exercise. Use the exact committed generator and scenario source trees."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lab_arena import broker, collection_judge as judge, contracts, store
from lab_arena.collection_broker import CollectionBroker, CollectionCallError, VERSIONS
from lab_arena.collection_intake import CollectionIntake, GENERATOR_ROUTES, MODELS
from lab_arena.collection_worker import process_job
from lab_arena.collection_workflow import CHAMPION_REV, build_scenario, workflow_sources
from lab_arena.service import LocalObjectStore
from tests.lab_arena import collection_intake_test as intake_fixture
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration

MIGRATIONS = (
    *DEFAULT_MIGRATIONS,
    *(
        next(Path("scripts").glob(f"{n}-*.sql")).name
        for n in (720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 732, 733, 738, 740, 741)
    ),
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
    intake.register(signed, signed["hotkey"], credentials)
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
    pricing["endpoint_providers"] = dict.fromkeys(models, "test")
    return CollectionBroker(
        enrolled,
        db,
        transport,
        job,
        pricing,
        routes={
            **dict.fromkeys(models, "test"),
            **{m: r.tag for m, r in GENERATOR_ROUTES.items()},
        },
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
            if "sentences" in env:
                result = {"labels": ["supported"] * len(env["sentences"])}
            elif "runs" in env:
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


def test_workflow_preserves_only_provider_exposed_reasoning_text(transport):
    enrolled = registered(transport, "Q")
    job = claim(enrolled, "visible-thought", "B1-s2027")

    class VisibleParts(StandIn):
        def send(self, **request):
            original = super().send(**request)
            data = json.loads(original.body)
            data["choices"][0]["message"]["reasoning_details"] = [
                {"type": "reasoning.encrypted", "data": "opaque bytes"},
                {"type": "reasoning.text", "text": "Read the source."},
                {"type": "reasoning.text", "text": "Compare the dates."},
            ]
            return broker.ProviderResponse(200, {}, json.dumps(data).encode())

    upstream = VisibleParts()
    adapter = setup_broker(enrolled, transport, job, upstream)
    response = adapter.model_call("visible", {
        "model": job["model"], "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 128,
    })
    assert upstream.sends[0]["reasoning"] == {"effort": "low", "exclude": False}
    assert response["choices"][0]["message"]["reasoning"] == "Read the source.\nCompare the dates."


def test_provider_receipt_must_match_the_admitted_endpoint(transport):
    enrolled = registered(transport, "Z")
    job = claim(enrolled, "wrong-provider-receipt", "B1-s2027")
    adapter = setup_broker(enrolled, transport, job, StandIn())
    adapter.pricing["endpoint_providers"][job["model"]] = "Other Service"
    with pytest.raises(CollectionCallError, match="assigned model"):
        adapter.model_call("wrong-provider", {
            "model": job["model"],
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 16,
        })
    assert adapter.usage()["settled_microusd"] == 10


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


def test_funding_statement_reports_only_own_confirmed_and_pending_costs(transport):
    enrolled = registered(transport, "H")
    job = claim(enrolled, "statement-proof", "B1-s2027")
    miner = job["miner_hotkey"]
    reader = setup_broker(enrolled, transport, job, StandIn())
    reader.model_call("step-1", {
        "model": job["model"], "messages": [{"role": "user", "content": "Test"}], "max_tokens": 16,
    })
    for provider, amount in (("openrouter", 50), ("judge", 100)):
        reader._rpc("collection_reserve_call_v1", "pending-" + provider,
                    p_provider=provider, p_hash="a" * 64, p_amount=amount)
    statement = enrolled.statement(miner)
    assert statement["enrollments"][0]["task_scope"] == "synthetic_workflows"
    assert statement["enrollments"][0]["daily_cap_microusd"] == intake_fixture.CAP
    run = statement["jobs"][0]
    assert run["confirmed_miner_cost_microusd"] == 10
    assert run["pending_miner_reservations_microusd"] == 50
    assert run["unresolved_miner_calls"] == 1
    assert run["price_microusd"] == 0
    assert enrolled.statement("5" + "z" * 47) == {"enrollments": [], "jobs": []}
    assert not any(name in json.dumps(statement) for name in ("fingerprint", "ciphertext", "key_id", "credentials", "lease_token"))



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


def test_checkpointed_comparison_keeps_the_exact_prefix_on_generation_failure(
    transport, tmp_path
):
    from lab_arena.collection_workflow import (
        CORRECTION_CHAMPION_REV,
        TRAIN_ONLY_POET_REV,
    )

    agent = os.environ.get("COLLECTION_TEST_AGENT_REPO")
    poet = os.environ.get("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("requires both approved workflow repositories")
    from tests.lab_arena.collection_judge_test import PROMPT

    batch = "capture-failure-proof"
    enrolled = registered(transport, "J")
    job = enrolled.claim(
        batch,
        ["P7-s60001"],
        CORRECTION_CHAMPION_REV,
        dict.fromkeys(MODELS, 0),
        dict.fromkeys(MODELS, 4000000),
    )
    policy = judge.BatchPolicy(batch, "fixed", judge.digest(PROMPT), pilot=True)

    class Broken:
        def __init__(self):
            self.calls = 0

        def model_call(self, request_id, body):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError()
            return {
                "model": body["model"],
                "id": "one",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Inspect the workspace.",
                            "tool_calls": [
                                {
                                    "id": "step1",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ],
            }

    with workflow_sources(
        agent, poet, corrections=True, builder_revision=TRAIN_ONLY_POET_REV
    ) as sources:
        with pytest.raises(TimeoutError):
            process_job(
                job=job,
                policy=policy,
                prompt=PROMPT,
                intake=enrolled,
                broker=Broken(),
                objects=LocalObjectStore(tmp_path),
                sources=sources,
                epoch=lambda: 10,
                assignment={"run_id": job["job_id"]},
                capture_version="collection.workflow-events.v1",
            )
    base = tmp_path / "collection" / batch / job["job_id"]
    events = [
        json.loads(f.read_text()) for f in sorted((base / "events").glob("[0-9]*.json"))
    ]
    assert [e["kind"] for e in events] == [
        "start",
        "before",
        "reply",
        "tool",
        "before",
        "interrupted",
    ]
    assert events[2]["message"]["reasoning"] == "Inspect the workspace."
    assert events[3]["output"] == json.dumps(sorted(events[1]["files"]))
    assert (
        events[4]["state"] == events[1]["state"] and events[4]["prefix_messages"] == 4
    )
    assert all("code_checks" in e for e in (events[1], events[4]))
    assert not list(base.glob("manifest-*.json"))


@pytest.mark.parametrize("checkpointed", [False, True])
def test_funded_runner_holds_missing_or_contradictory_split_before_claim(
    checkpointed, capsys
):
    from types import SimpleNamespace
    from scripts.run_locked_collection import run_cohort
    from lab_arena.collection_workflow import (
        COMPARISON_CAPTURE,
        TRAIN_ONLY_POET_REV,
        POET_REV,
    )

    agent = os.environ.get("COLLECTION_TEST_AGENT_REPO")
    poet = os.environ.get("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("requires both approved workflow repositories")
    attempted = []

    class NoClaim:
        def claim(self, *args):
            attempted.append(True)
            raise AssertionError("attempted to charge excluded scenario")

    with workflow_sources(
        agent,
        poet,
        corrections=checkpointed,
        builder_revision=TRAIN_ONLY_POET_REV if checkpointed else POET_REV,
    ) as sources:
        raw = {
            "mode": "pilot",
            "scenario_ids": ["C07-s3013"],
            "budgets_microusd": dict.fromkeys(MODELS, 1),
        }
        if checkpointed:
            raw["capture_version"] = COMPARISON_CAPTURE
        args = dict(
            raw=raw,
            policy=SimpleNamespace(batch_id="test-split", pilot=True),
            prompt={},
            intake=NoClaim(),
            database=None,
            objects=None,
            transport=None,
            pricing=None,
            epoch=lambda: 1,
            sources=sources,
            owner_lookup=None,
            judge_key="fixture",
            max_jobs=1,
        )
        assert (
            run_cohort(**args) == 0
            and "missing_split_manifest" in capsys.readouterr().out
        )
        # A fresh seed and an invented held-out list cannot reassign a known test template.
        raw["split_manifest"] = {
            "schema": "collection.split.v1",
            "training": {
                "template_id": ["C07"],
                "rule_pool": ["poet:train_rules:chain"],
                "tool_variant": ["contacts,functions,plain"],
            },
            "heldout": {
                "template_id": ["unrelated"],
                "rule_pool": ["separate"],
                "tool_variant": ["another"],
            },
        }
        assert (
            run_cohort(**args) == 0
            and "scenario_not_in_training_partition" in capsys.readouterr().out
        )
        assert attempted == []


def test_hard_stop_retains_an_unsealed_private_prefix_without_a_payable_manifest(tmp_path):
    from lab_arena.collection_journal import WorkflowJournal

    objects = LocalObjectStore(tmp_path)
    journal = WorkflowJournal(objects, "batch", "run")
    journal.append({"index": 0, "kind": "start", "model": MODELS[0], "messages": [], "tools": []})
    journal.append({"index": 1, "kind": "before", "turn": 0, "state": {"example": "before"}})
    readback = WorkflowJournal(objects, "batch", "run")
    assert readback.comparison_index(complete=False, model=MODELS[0]) == {
        "status": "unsealed", "events": journal.hashes
    }
    with pytest.raises(ValueError, match="complete event index"):
        readback.comparison_index(complete=True, model=MODELS[0], manifest_events=journal.hashes)
    assert not list(tmp_path.rglob("manifest-*.json"))
