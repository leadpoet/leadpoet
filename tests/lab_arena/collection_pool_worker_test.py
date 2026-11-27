"""Execute the pool's real pinned experiment runner. Only provider I/O and ledger are fixtures."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4
import json, os
import pytest

from lab_arena import collection_experiment as xp, collection_judge as judge, contracts
from lab_arena.collection_generation import CAPTURE_V2
from lab_arena.collection_intake import GENERATOR_ROUTES, SCOPE, scope_id
from lab_arena.collection_pool import POOL_PAYMENT_RULE, POOL_SCOPE, POOL_TERMS_HASH
from lab_arena.collection_workflow import (
    CORRECTION_CHAMPION_REV,
    OUTREACH_POET_REV,
    NATIVE_PARTITIONS,
    workflow_sources,
)
from lab_arena.service import LocalObjectStore
from tests.lab_arena.collection_judge_test import PROMPT
from tests.lab_arena.collection_sales_evidence_test import OriginalReply

SMALL = "nvidia/nemotron-3.5-super-vl"
EXPERT = "deepseek/deepseek-v4-pro-0813"


def pool_raw(family="P9", model=SMALL, repeats=4):
    return {
        "schema": xp.SCHEMA,
        "id": "native-eligibility",
        "payment_rule": POOL_PAYMENT_RULE,
        "mode": "live",
        "question": "Which source-stop errors occur?",
        "pass_mark": "All native decisions and evidence checks pass.",
        "max_slots": repeats,
        "budget_microusd": repeats * 100000,
        "cases": [{"scenario_id": family + "-s81000"}],
        "split_sha256": "sha256:" + NATIVE_PARTITIONS[family],
        "terms_sha256": POOL_TERMS_HASH,
        "routes": {
            model: ("coreweave/fp8" if model == EXPERT else GENERATOR_ROUTES[model].tag)
        },
        "capture_version": CAPTURE_V2,
        "successor_seeds": True,
        "arms": [
            {
                "id": "student",
                "role": "expert",
                "model": model,
                "repeats": repeats,
                "max_steps": 32,
                "max_tokens": 8192,
                "max_run_tokens": 65536,
                "reasoning_effort": "low",
                "capture_reasoning": True,
                "budget_microusd": 100000,
            }
        ],
    }


def test_pool_identity_repeats_and_safe_renewal(tmp_path):
    from scripts.run_locked_collection import poll_pool_experiments
    from scripts.prepare_collection_experiment import store_private_json

    raw = pool_raw()
    spec = xp.load_experiment(raw, PROMPT)
    assert spec.batch_id.startswith("pool-") and [r for _, _, r in spec.slots] == [
        1,
        2,
        3,
        4,
    ]
    bad = deepcopy(raw)
    bad["arms"][0].pop("max_run_tokens")
    with pytest.raises(ValueError):
        xp.load_experiment(bad, PROMPT)
    for corrupt in (
        dict(raw, terms_sha256="sha256:" + "a" * 64),
        dict(raw, pricing_pilot="xp-" + "a" * 64),
        dict(raw, routes={SMALL: "anything"}),
    ):
        with pytest.raises(ValueError):
            xp.load_experiment(corrupt, PROMPT)
    path = tmp_path / (spec.batch_id + ".json")
    store_private_json(path, raw)
    db = SimpleNamespace(
        rpc=lambda *a: [{"status": "accepted", "slot": i} for i in range(4)]
    )
    got = []

    def dispatch(manifest, *, before_claim):
        got.append(manifest["cases"][0]["scenario_id"])
        if manifest == raw:
            return 0
        before_claim()
        return 1

    assert poll_pool_experiments(PROMPT, tmp_path, db, dispatch) == 1
    assert got == ["P9-s81000", "P9-s81001"] and path.is_file()
    got.clear()
    assert poll_pool_experiments(PROMPT, tmp_path, db, dispatch) == 1 and got == [
        "P9-s81001"
    ]


@pytest.fixture
def pool_fixture(tmp_path, monkeypatch, request):
    from bittensor_wallet import Keypair
    import lab_arena.collection_broker as cb

    agent = os.environ.get("COLLECTION_TEST_AGENT_REPO")
    poet = os.environ.get("COLLECTION_TEST_POET_REPO")
    if not (agent and poet):
        pytest.skip("private scenario checkout unavailable")
    key = Keypair.create_from_uri("//StandingComputeFixture")
    consent = contracts.build_signed_request(
        scope=SCOPE["register"],
        round_id=scope_id("live"),
        hotkey=key.ss58_address,
        body={
            "task_scope": POOL_SCOPE,
            "spend_limit_microusd": 1000000,
            "priority": 1,
            "terms_sha256": POOL_TERMS_HASH,
            "credentials_sha256": "sha256:" + "b" * 64,
        },
        timestamp=1,
        sign_message=lambda m: key.sign(m).hex(),
    )
    env = SimpleNamespace(
        objects=LocalObjectStore(tmp_path / "bucket"),
        jobs=[],
        grants={},
        scripts=[],
        run_costs={},
        inputs=tmp_path / "inputs",
        key=key,
        consent=consent,
    )
    env.inputs.mkdir()
    from scripts.collection_review import write_new

    class Intake:
        def claim_pool(self, spec, harness, *, expected_model="*"):
            slot = sum(j["batch_id"] == spec.batch_id for j in env.jobs)
            if slot >= len(spec.slots):
                return None
            case, arm, _ = spec.slots[slot]
            if expected_model is None:
                return {"model": arm["model"]}
            rid = str(uuid4())
            j = {
                "job_id": rid,
                "batch_id": spec.batch_id,
                "mode": "live",
                "slot": slot,
                "attempt": 1,
                "scenario_id": case["scenario_id"],
                "model": arm["model"],
                "price_microusd": 0,
                "budget_microusd": arm["budget_microusd"],
                "lease_token": "f" * 64,
                "miner_hotkey": key.ss58_address,
                "owner_coldkey": key.ss58_address,
                "harness_commit": harness,
                "status": "running",
            }
            env.jobs.append(j)
            return j

        def complete(self, rid, token, status, quality, artifact, epoch):
            j = next(j for j in env.jobs if j["job_id"] == rid)
            count = env.run_costs.get(rid, 0)
            cost = count * 10
            j.update(
                status=status,
                quality=quality,
                artifact_sha256=artifact,
                price_microusd=(cost * 5 + 3) // 4 if status == "accepted" else 0,
            )
            env.grants[rid] = dict(
                job_id=rid,
                mode="live",
                miner_hotkey=key.ss58_address,
                registration_id=str(UUID(hex=consent["request_id"])),
                consent=consent,
                task_scope=POOL_SCOPE,
                terms_sha256=POOL_TERMS_HASH,
                spend_limit_microusd=1000000,
                priority=1,
                unresolved=0,
                generator_calls=count,
                miner_cost_microusd=cost,
            )

    class Db:
        def rpc(self, name, params):
            group = [j for j in env.jobs if j["batch_id"] == params["p_batch"]]
            if name == "collection_batch_jobs_v1":
                return group
            if name == "collection_training_consents_v1":
                return [env.grants[j["job_id"]] for j in group]
            raise AssertionError(name)

    def broker(_, db, transport, job, pricing, *, routes, judge_key):
        rid = job["job_id"]
        ad = OriginalReply(env.scripts.pop(0), rid)
        ad.versions = {m: r.revision for m, r in GENERATOR_ROUTES.items()}

        def usage():
            used = len(ad.prefixes)
            env.run_costs[rid] = used
            return {
                "settled_microusd": 10 * used,
                "reserved_or_uncertain_microusd": 0,
                "inflight_calls": 0,
                "providers": {"openrouter": {"used": used, "inflight": 0}},
            }

        ad.usage = usage
        return ad

    monkeypatch.setattr(cb, "CollectionBroker", broker)
    env.db = Db()
    env.intake = Intake()
    from lab_arena.collection_workflow import SALES_EVIDENCE_POET_REV
    research = getattr(request, "param", None) == "P8"
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=SALES_EVIDENCE_POET_REV if research else OUTREACH_POET_REV
    ) as src:
        env.sources = src
        for family in (("P8",) if research else ("P9", "P10")):
            split = src[1].training_partition(family)
            write_new(env.inputs / (judge.digest(split) + ".json"), split)
        yield env


def simulation(env, raw, scripts, *, on_pool_failure=None):
    spec = xp.load_experiment(raw, PROMPT)
    env.scripts.extend(scripts)
    sources = env.sources
    inputs = xp.read_inputs(
        spec, env.inputs, sources, objects=env.objects, database=env.db, prompt=PROMPT
    )
    count = xp.run_experiment(
        spec=spec,
        sources=sources,
        inputs=inputs,
        intake=env.intake,
        database=env.db,
        objects=env.objects,
        transport=None,
        pricing=lambda model: {},
        owner_lookup=lambda _: SimpleNamespace(coldkey=env.key.ss58_address),
        judge_key=None,
        max_jobs=len(scripts),
        epoch=lambda: 123,
        prompt=PROMPT,
        on_pool_failure=on_pool_failure,
    )
    assert count == len(scripts)
    return spec


def scripts_for(env, family):
    t = env.sources[1].build(family, 81000)
    ref = env.sources[1].perfect(t)
    correct = list(ref["calls"]) + [ref["answer"]]
    wrong = deepcopy(correct)
    for r in wrong[-2][1]["plan"]:
        r["disposition"] = "release" if family == "P9" else "keep"
    return correct, wrong


@pytest.mark.parametrize("family", ["P9", "P10"])
def test_pool_pass_and_fail_paid_and_genuine_continuation(pool_fixture, family):
    from lab_arena.collection_pool_source import funded_source
    from scripts.collection_review import write_new

    e = pool_fixture
    raw = pool_raw(family=family, repeats=2)
    correct, bad = scripts_for(e, family)
    spec = simulation(e, raw, [correct, bad])
    for j, passed in zip(e.jobs, [True, False], strict=True):
        manifest, proof = xp.pool_verified_result(e.objects, j, spec)
        assert (
            j["status"] == "accepted"
            and proof["eligible"]
            and proof["complete_verified"]
            and proof["task_pass"] is passed
        )
        assert j["price_microusd"] == (5 * proof["cost_microusd"] + 3) // 4
    j = e.jobs[-1]
    source = funded_source(e.db, e.objects, spec.batch_id, j["job_id"], PROMPT)
    assert (
        source["rejected_fatal"]
        and source["failure"]["reason"] == "incorrect_irreversible_plan"
    )
    assert "commit" in source["rejected"]["tool_calls"][0]["function"]["arguments"]
    sid = xp._hash(source)
    write_new(e.inputs / (sid[7:] + ".json"), source)
    run = pool_raw(family, EXPERT, 1)
    run.pop("successor_seeds")
    run["arms"][0]["role"] = "continuation"
    run["cases"][0]["source_sha256"] = sid
    resumed = simulation(e, run, [correct[-2:]])
    m, proof = xp.pool_verified_result(e.objects, e.jobs[-1], resumed)
    assert proof["task_pass"] and proof["calls"] == 2
    from lab_arena.collection_workflow import read_artifact

    row = read_artifact(
        e.objects,
        resumed.batch_id,
        e.jobs[-1]["job_id"],
        "generator",
        m["generator"],
        16 << 20,
    )
    assert all(row["correction"]["correction_step_checks"].values())
    assert (
        row["messages"][: row["correction"]["prefix_messages"]]
        == source["checkpoint"]["messages"]
    )


def test_healthy_program_survives_preflight_failure_but_never_replays_ambiguous_claim(
    tmp_path,
):
    from scripts.prepare_collection_experiment import store_private_json
    from scripts.run_locked_collection import poll_pool_experiments

    for family in ("P9", "P10"):
        raw = pool_raw(family=family)
        path = tmp_path / (xp.load_experiment(raw, PROMPT).batch_id + ".json")
        store_private_json(path, raw)
    store_private_json(tmp_path / ("pool-" + "a" * 64 + ".json"), {})
    calls = []

    def available(raw, *, before_claim):
        calls.append(raw["cases"][0]["scenario_id"])
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")
        before_claim()
        return 1

    assert (
        poll_pool_experiments(PROMPT, tmp_path, None, available) == 1
        and len(calls) == 2
    )
    calls.clear()

    def ambiguous(raw, *, before_claim):
        before_claim()
        calls.append(raw["id"])
        raise RuntimeError("unknown claim")

    with pytest.raises(RuntimeError, match="unknown claim"):
        poll_pool_experiments(PROMPT, tmp_path, None, ambiguous)
    assert len(calls) == 1


def test_completed_receipt_retry_never_repays_a_lost_response():
    record = {
        "job_id": "id",
        "lease_token": "lease",
        "batch_id": "batch",
        "status": "running",
    }
    calls = []

    def complete(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionError("lost")
        return True

    db = SimpleNamespace(rpc=lambda *args: [record])
    epochs = iter((200, 201))
    assert xp.finish_pool_job(
        SimpleNamespace(complete=complete),
        db,
        record,
        "accepted",
        {},
        "artifact",
        lambda: next(epochs),
    )
    assert [args[-1] for args in calls] == [200, 201]
    record.update(status="accepted", artifact_sha256="artifact")
    calls.clear()
    assert (
        xp.finish_pool_job(
            SimpleNamespace(complete=complete),
            db,
            record,
            "accepted",
            {},
            "artifact",
            lambda: 200,
        )
        == record
    )
    assert len(calls) == 1


def test_automatic_independent_correction_staged_once_and_runs_with_funding(
    pool_fixture, tmp_path
):
    from functools import partial
    from scripts.run_locked_collection import stage_failed_pool_continuation

    e = pool_fixture
    raw = pool_raw(repeats=1)
    raw["correction_model"] = EXPERT
    raw["routes"][EXPERT] = "coreweave/fp8"
    raw["arms"].append(dict(raw["arms"][0], id="stronger", model=EXPERT))
    raw["max_slots"] = 2
    raw["budget_microusd"] = 200000
    out = tmp_path / "offers"
    out.mkdir()
    callback = partial(
        stage_failed_pool_continuation,
        database=e.db,
        objects=e.objects,
        directory=out,
        inputs=e.inputs,
        prompt=PROMPT,
    )
    good, bad = scripts_for(e, "P9")
    parent = simulation(e, raw, [bad, good], on_pool_failure=callback)
    [path] = out.glob("pool-*.json")
    new = json.loads(path.read_bytes())
    assert (
        new["arms"][0]["role"] == "continuation" and new["arms"][0]["model"] == EXPERT
    )
    assert len(xp.load_experiment(new, PROMPT).slots) == 1
    assert (
        callback(parent, e.jobs[0]) is None
    )  # immutable candidate for this case already exists
    assert (
        callback(parent, dict(e.jobs[0], scenario_id="P9-s81001")) is None
    )  # no second outstanding correction
    corrected = simulation(
        e,
        new,
        [good[-2:]],
        on_pool_failure=lambda *a: pytest.fail("automatic correction recursed"),
    )
    _, evidence = xp.pool_verified_result(e.objects, e.jobs[-1], corrected)
    assert (
        evidence["task_pass"]
        and evidence["complete_verified"]
        and evidence["calls"] == 2
    )


def test_native_provider_usage_divergence_is_rejected(pool_fixture):
    from lab_arena.collection_worker import _paid_reply
    from lab_arena.collection_workflow import read_artifact

    e = pool_fixture
    raw = pool_raw(repeats=1)
    good, _ = scripts_for(e, "P9")
    spec = simulation(e, raw, [good])
    job = e.jobs[0]
    manifest, proof = xp.pool_verified_result(e.objects, job, spec)
    row = read_artifact(
        e.objects,
        spec.batch_id,
        job["job_id"],
        "generator",
        manifest["generator"],
        128 << 20,
    )
    original = next(v for v in row["evidence"]["events"] if v["kind"] == "reply")
    message = next(m for m in row["messages"] if m["role"] == "assistant")
    with pytest.raises(ValueError, match="original generator receipt differs"):
        _paid_reply(
            e.objects,
            job,
            1,
            dict(original, usage={"completion_tokens": 99999}),
            message,
        )


@pytest.mark.parametrize("pool_fixture", ["P8"], indirect=True)
@pytest.mark.parametrize(
    "later",
    [
        "unchanged",
        "fully_repaired",
        "different_error",
        "malformed",
        "manual",
        "expert_repaired_late",
    ],
)
def test_standing_grounding_followup_requires_persistent_error(
    pool_fixture, tmp_path, later, monkeypatch
):
    from functools import partial
    from tests.lab_arena.collection_sales_evidence_test import script
    from scripts.run_locked_collection import stage_failed_pool_continuation
    from lab_arena.collection_workflow import read_artifact
    from lab_arena.collection_experiment_export import _pair

    env = pool_fixture
    raw = pool_raw("P8", repeats=1)
    raw["arms"].append(dict(raw["arms"][0], id="stronger", model=EXPERT))
    raw.update(correction_model=EXPERT, max_slots=2, budget_microusd=200000)
    raw["routes"][EXPERT] = "coreweave/fp8"
    _, _, correct = script(env.sources[1], seed=81000)
    original_memo = json.loads(correct[-2][1]["content"])

    def inaccurate(index):
        memo = deepcopy(original_memo)
        field = memo["findings"][index]
        field["value"] = "unsupported status"
        field["citations"] = []
        return (
            "write_file",
            {"path": "qualification.json", "content": json.dumps(memo)},
        )

    failed = correct[:-2] + [inaccurate(0)]
    if later == "malformed":
        failed[-1] = (
            "write_file",
            {"path": "qualification.json", "content": "invalid JSON"},
        )
    if later == "fully_repaired":
        failed += [correct[-2], "Wrong terminal response."]
    elif later == "different_error":
        failed += [inaccurate(1), correct[-1]]
    else:
        failed += [correct[-1]]
    offers = tmp_path / "offers"
    offers.mkdir()
    callback = partial(
        stage_failed_pool_continuation,
        database=env.db,
        objects=env.objects,
        directory=offers,
        inputs=env.inputs,
        prompt=PROMPT,
    )
    spec = simulation(
        env,
        raw,
        [failed, correct],
        on_pool_failure=None if later == "manual" else callback,
    )
    original, evidence = xp.pool_verified_result(env.objects, env.jobs[0], spec)
    assert evidence["complete_verified"] and not evidence["task_pass"]
    followups = list(offers.glob("pool-*.json"))
    if later == "manual":
        from lab_arena.collection_pool_source import funded_source, verify_funded_source
        from scripts.prepare_collection_experiment import store_private_json

        # A previously staged manual source may predate the new field labels.
        previous = funded_source(
            env.db, env.objects, spec.batch_id, env.jobs[0]["job_id"], PROMPT
        )
        assert "unrepaired_grounding" not in previous["failure"]
        verify_funded_source(env.db, env.objects, previous, PROMPT)
        reference = "sha256:" + judge.digest(previous)
        manual = dict(
            raw,
            id="manual-continuation",
            question="Can an expert correct the memo?",
            max_slots=1,
            budget_microusd=100000,
            routes={EXPERT: raw["routes"][EXPERT]},
            arms=[dict(raw["arms"][1], role="continuation")],
            cases=[dict(raw["cases"][0], source_sha256=reference)],
        )
        del manual["successor_seeds"], manual["correction_model"]
        frozen = xp.load_experiment(manual, PROMPT)
        store_private_json(env.inputs / (reference[7:] + ".json"), previous)
        manual_path = offers / (frozen.batch_id + ".json")
        store_private_json(manual_path, manual)
        assert callback(spec, env.jobs[0]) is None and list(
            offers.glob("pool-*.json")
        ) == [manual_path]
        return
    if later not in {"unchanged", "expert_repaired_late"}:
        assert not followups
        return
    assert len(followups) == 1
    if later == "unchanged":
        from scripts.prepare_collection_experiment import main, store_private_json
        import scripts.run_locked_collection as worker

        parent_path = tmp_path / "source-program.json"
        prompt_path = tmp_path / "prompt.json"
        store_private_json(parent_path, raw)
        store_private_json(prompt_path, PROMPT)
        monkeypatch.setattr(worker, "evidence_stores", lambda: (env.db, env.objects))
        with pytest.raises(
            ValueError, match="continuation from this paid source already exists"
        ):
            main(
                [
                    "prepare-pool",
                    "--experiment-id",
                    "manual-followup",
                    "--family",
                    "P8",
                    "--first-seed",
                    "81000",
                    "--scenario-count",
                    "1",
                    "--repeat",
                    "1",
                    "--model",
                    EXPERT,
                    "--cap-microusd",
                    "100000",
                    "--champion-repo",
                    os.environ["COLLECTION_TEST_AGENT_REPO"],
                    "--poet-repo",
                    os.environ["COLLECTION_TEST_POET_REPO"],
                    "--prompt",
                    str(prompt_path),
                    "--source-batch",
                    str(parent_path),
                    "--source-job",
                    env.jobs[0]["job_id"],
                    "--directory",
                    str(offers),
                    "--inputs",
                    str(env.inputs),
                ]
            )
        assert list(offers.glob("pool-*.json")) == followups
    plan = json.loads(followups[0].read_bytes())
    src_hash = plan["cases"][0]["source_sha256"]
    student = json.loads((env.inputs / (src_hash[7:] + ".json")).read_bytes())
    assert not student["rejected_fatal"]
    assert student["failure"]["unrepaired_grounding"] == {
        original_memo["findings"][0]["field"]: ["claim", "citation"]
    }
    # All evidence was opened by the student before the mistake. The expert may fix
    # the evidence-supported memo directly and must not have to buy another search.
    expert_script = correct[-2:]
    if later == "expert_repaired_late":
        # Submitting an equivalent wrong memo under a different tool-call ID must never be a positive action.
        expert_script = [inaccurate(0)] + expert_script
    resumed = simulation(env, plan, [expert_script])
    job = env.jobs[-1]
    m, e = xp.pool_verified_result(env.objects, job, resumed)
    assert (
        e["complete_verified"]
        and e["task_pass"]
        and e["verified_decision"]
        and e["calls"] == len(expert_script)
    )
    row = read_artifact(
        env.objects,
        resumed.batch_id,
        job["job_id"],
        "generator",
        m["generator"],
        128 << 20,
    )
    packet = read_artifact(
        env.objects,
        resumed.batch_id,
        job["job_id"],
        "judge-input",
        m["packet"],
        128 << 20,
    )
    pair = _pair(row, packet, student, native=e)
    if later == "expert_repaired_late":
        assert (
            e["first_failed_decision"]["turn"] == student["branch_turn"]
            and e["observed_repair"]
        )
        assert pair is None
        return
    from lab_arena.collection_worker import native_payment_evidence

    historical = dict(m)
    del historical["uses_checkpoint_evidence"]
    prior = native_payment_evidence(env.objects, job, historical)
    assert (
        prior["complete_verified"]
        and not prior["task_pass"]
        and not prior["verified_decision"]
    )
    assert pair["checkpoint_sha256"] == student["checkpoint_sha256"]
    assert (
        pair["context"] == student["checkpoint"]["messages"]
        and pair["rejected"] != pair["chosen"]
    )
    assert (
        not pair["step_checks"]["student_action_fatal"]
        and pair["failure_kind"] == "unrepaired_grounding"
    )


def test_stagger_only_rotates_fresh_paid_successors(tmp_path):
    from scripts.prepare_collection_experiment import store_private_json
    from scripts.run_locked_collection import poll_pool_experiments
    from lab_arena.collection_experiment import pool_program_id

    raw = pool_raw()
    raw.update(
        id="consent-and-suppression",
        correction_model=EXPERT,
        max_slots=8,
        budget_microusd=800000,
    )
    raw["routes"][EXPERT] = "coreweave/fp8"
    raw["arms"].append(dict(raw["arms"][0], id="stronger", model=EXPERT))
    prior = xp.load_experiment(raw, PROMPT)
    p = tmp_path / (prior.batch_id + ".json")
    store_private_json(p, raw)
    original_bytes = p.read_bytes()
    series = judge.digest({k: v for k, v in raw.items() if k != "cases"})
    assert pool_program_id(raw) == series
    goal = (raw["id"], EXPERT)
    called = []
    settled = set(range(7))
    db = SimpleNamespace(
        rpc=lambda op, p: (
            [dict(slot=i, status="accepted") for i in settled]
            if p["p_batch"] == prior.batch_id
            else []
        )
    )

    def run(doc, *, before_claim):
        called.append(doc)
        return 0  # Exhausted authorized budget, or no eligible key; never a real lease.

    assert poll_pool_experiments(PROMPT, tmp_path, db, run, successor_leader=goal) == 0
    assert called == [raw] and len(list(tmp_path.glob("pool-*.json"))) == 1
    settled.add(7)
    assert poll_pool_experiments(PROMPT, tmp_path, db, run, successor_leader=goal) == 0
    nxt = called[-1]
    fresh = xp.load_experiment(nxt, PROMPT)
    assert (
        nxt["cases"] == [{"scenario_id": "P9-s81001"}]
        and pool_program_id(nxt) == series
    )
    assert [(a["model"], r) for _, a, r in fresh.slots] == [
        (m, n) for n in range(1, 5) for m in (EXPERT, SMALL)
    ]
    assert p.read_bytes() == original_bytes and fresh.batch_id != prior.batch_id
    from scripts.run_locked_collection import _freeze_pool_successor

    assert (
        _freeze_pool_successor(prior, tmp_path, PROMPT, None).batch_id == fresh.batch_id
    )
    assert (
        _freeze_pool_successor(prior, tmp_path, PROMPT, goal).batch_id == fresh.batch_id
    )
    # A different operator order racing at the same boundary cannot create a second pack.
    assert (
        _freeze_pool_successor(prior, tmp_path, PROMPT, (raw["id"], SMALL)).batch_id
        == fresh.batch_id
    )
    assert poll_pool_experiments(PROMPT, tmp_path, db, run) == 0
    assert called[-1] == nxt and len(list(tmp_path.glob("pool-*.json"))) == 2

    def claim(doc, *, before_claim):
        assert doc == nxt
        before_claim()
        return 1

    assert (
        poll_pool_experiments(PROMPT, tmp_path, db, claim, successor_leader=goal) == 1
    )
    assert (tmp_path / ".last-pool-program").read_text() == series
    shifted = deepcopy(nxt)
    shifted["series_arm_order"] = ["stronger", "stronger"]
    with pytest.raises(ValueError):
        xp.load_experiment(shifted, PROMPT)
    shifted = deepcopy(nxt)
    shifted.pop("successor_seeds")
    with pytest.raises(ValueError):
        xp.load_experiment(shifted, PROMPT)


def test_prelease_price_diagnostic_redacts_provider_details(tmp_path, capsys):
    from scripts.prepare_collection_experiment import store_private_json
    from scripts.run_locked_collection import poll_pool_experiments
    from lab_arena.collection_prices import endpoint_prices
    from urllib.error import HTTPError

    raw = pool_raw()
    store_private_json(
        tmp_path / (xp.load_experiment(raw, PROMPT).batch_id + ".json"), raw
    )

    def refused(request, timeout):
        raise HTTPError(
            "https://sensitive.invalid/key", 429, "credentials are private", None, None
        )

    def run(doc, *, before_claim):
        endpoint_prices(
            xp.load_experiment(doc, PROMPT).routes, models=[SMALL], open_url=refused
        )
        pytest.fail("an unavailable route cannot authorize a lease")

    with pytest.raises(ValueError, match="could not be polled"):
        poll_pool_experiments(PROMPT, tmp_path, None, run)
    [line] = capsys.readouterr().err.splitlines()
    record = json.loads(line)
    assert (
        record["reason"] == "provider_catalog_throttled"
        and not (tmp_path / ".last-pool-program").exists()
    )
    assert "sensitive" not in line and "credentials" not in line


def test_productive_burst_finishes_work_then_obeys_time_funding_and_drain():
    from scripts.run_locked_collection import poll_funded_burst

    now = [100.0]
    issued = []

    def advance():
        issued.append(now[0])
        now[0] += 200
        return 1

    # The ninth authorized run may finish exactly at the deadline; no tenth lease is started.
    assert (
        poll_funded_burst(
            64,
            advance,
            productive=True,
            monotonic=lambda: now[0],
            timer_active=lambda: True,
        )
        == 9
    )
    assert len(issued) == 9
    issued.clear()
    assert (
        poll_funded_burst(
            64,
            lambda: issued.append(1) or 1,
            productive=True,
            monotonic=lambda: 0,
            timer_active=lambda: False,
        )
        == 8
    )
    assert len(issued) == 8
    issued.clear()
    assert (
        poll_funded_burst(
            64,
            lambda: issued.append(1) or 0,
            productive=True,
            monotonic=lambda: 0,
            timer_active=lambda: True,
        )
        == 0
    )
    assert len(issued) == 1
    issued.clear()

    def failing():
        issued.append(1)
        raise RuntimeError("ambiguous claim")

    with pytest.raises(RuntimeError, match="ambiguous claim"):
        poll_funded_burst(
            64, failing, productive=True, monotonic=lambda: 0, timer_active=lambda: True
        )
    assert len(issued) == 1

    def no_timer():
        pytest.fail("ordinary operators never inspect the service timer")

    assert (
        poll_funded_burst(
            3, lambda: 1, productive=False, monotonic=lambda: 0, timer_active=no_timer
        )
        == 3
    )
    assert (
        poll_funded_burst(
            64,
            lambda: 1,
            productive=True,
            monotonic=lambda: 0,
            timer_active=lambda: True,
        )
        == 64
    )


@pytest.mark.parametrize("terminal", ["accepted", "rejected"])
def test_installed_worker_extends_only_a_paid_burst(tmp_path, monkeypatch, terminal):
    import scripts.run_locked_collection as worker

    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    calls = []
    monkeypatch.setattr(
        worker, "evidence_stores", lambda: (SimpleNamespace(rpc=lambda *a: True), None)
    )

    def completed(args, parser, raw, *, on_pool_settled, before_claim=None):
        calls.append(1)
        on_pool_settled(terminal)
        return 1

    monkeypatch.setattr(worker, "_run_options", completed)
    monkeypatch.setattr(
        worker,
        "poll_pool_experiments",
        lambda prompt, directory, database, run, **kw: run({}),
    )
    assert (
        worker.main(
            [
                "--batch",
                str(cfg),
                "--prompt",
                str(cfg),
                "--offer-directory",
                str(tmp_path),
                "--productive-burst",
                "--max-jobs",
                "5",
            ]
        )
        == 0
    )
    assert len(calls) == (5 if terminal == "accepted" else 1)
