"""Exercise deterministic paid work against the actual pinned sales harness and P7 grader."""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from lab_arena import collection_judge as judge
from lab_arena.collection_generation import encoded
from lab_arena.collection_worker import P7_PAYMENT_RULE, process_job, verify_paid_job
from lab_arena.collection_workflow import (
    COMPARISON_CAPTURE,
    CORRECTION_CHAMPION_REV,
    TRAIN_ONLY_POET_REV,
    build_scenario,
    workflow_sources,
)
from lab_arena.collection_intake import MODELS
from lab_arena.service import LocalObjectStore
from scripts.run_locked_collection import load_batch, pack_split
from tests.lab_arena.collection_judge_test import PROMPT


@pytest.fixture
def native(tmp_path):
    agent, poet = (
        os.getenv(k)
        for k in ("COLLECTION_TEST_AGENT_REPO", "COLLECTION_TEST_POET_REPO")
    )
    if not agent or not poet:
        pytest.skip("requires the pinned public harness and approved P7 producer")
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=TRAIN_ONLY_POET_REV
    ) as sources:
        job = dict(
            job_id="native-payment",
            batch_id="dv-" + "a" * 64,
            model=MODELS[1],
            scenario_id="P7-s40401",
            harness_commit=CORRECTION_CHAMPION_REV,
            lease_token="local-lease",
            mode="live",
        )
        from lab_arena.collection_intake import (
            GENERATOR_ROUTES,
            PAID_WORKFLOW_TERMS_HASH,
        )
        from lab_arena.collection_broker import VERSIONS

        raw = dict(
            mode="live",
            name="verified-paid-test",
            seed="fixed",
            routes={
                model: (
                    GENERATOR_ROUTES[model].tag if model in GENERATOR_ROUTES else "test"
                )
                for model in VERSIONS
            },
            scenario_ids=[job["scenario_id"]],
            control_slots={},
            drift_slots={},
            split_manifest=sources[1].training_partition("P7"),
            payment_rule=P7_PAYMENT_RULE,
            budgets_microusd={job["model"]: 200000},
            fixed_quotes_microusd={job["model"]: 38},
            terms_sha256=PAID_WORKFLOW_TERMS_HASH,
            capture_version=COMPARISON_CAPTURE,
        )
        policy = load_batch(raw, PROMPT)
        job["batch_id"] = policy.batch_id
        p7 = build_scenario(sources[1], job["scenario_id"])
        answer = sources[1].perfect(p7)
        split = pack_split(
            {"split_manifest": sources[1].training_partition("P7")},
            sources[1],
            job["scenario_id"],
        )
        objects = LocalObjectStore(tmp_path)
        completed = []

        def run(*, wrong=False, thoughts=True, duplicate=False):
            actions = list(answer["calls"])
            if wrong:
                actions[-1] = (
                    "partner_credit_exchange",
                    {"envelope": dict(actions[-1][1]["envelope"], registrations=[])},
                )

            class Replies:
                record_generation = None

                def model_call(self, request_id, body):
                    receipt = "gen-duplicate" if duplicate else "gen-" + request_id
                    message = {"role": "assistant", "content": answer["answer"]}
                    if thoughts:
                        message["reasoning"] = "I checked the revision."
                    if actions:
                        name, args = actions.pop(0)
                        message.update(
                            content="",
                            tool_calls=[
                                dict(
                                    id="tool-" + request_id,
                                    type="function",
                                    function={
                                        "name": name,
                                        "arguments": json.dumps(args),
                                    },
                                )
                            ],
                        )
                    reply = dict(
                        id=receipt,
                        model=job["model"],
                        usage={"cost": 0.000010, "completion_tokens": 12},
                        choices=[
                            dict(
                                message=message,
                                finish_reason=(
                                    "tool_calls"
                                    if message.get("tool_calls")
                                    else "stop"
                                ),
                            )
                        ],
                    )
                    data = encoded(reply)
                    self.record_generation(
                        request_id,
                        hashlib.sha256(encoded(body)).hexdigest(),
                        hashlib.sha256(data).hexdigest(),
                        200,
                        10,
                        receipt,
                        data,
                    )
                    return reply

            return process_job(
                job=job,
                assignment={"run_id": job["job_id"]},
                policy=policy,
                prompt=PROMPT,
                sources=sources,
                broker=Replies(),
                objects=objects,
                intake=SimpleNamespace(
                    complete=lambda *args, **kwargs: completed.append((args, kwargs))
                ),
                epoch=lambda: 10,
                split=split,
                capture_version=COMPARISON_CAPTURE,
                payment_rule=P7_PAYMENT_RULE,
                prepare_launch=True,
            )

        yield run, completed, job, objects, policy, raw


def test_real_pinned_native_run_is_payable_and_recoverable_without_a_judge(native):
    run, completed, job, objects, policy, raw = native
    manifest = run()
    args, kw = completed[0]
    assert args[2:4] == (
        "accepted",
        {"training_status": "needs_review", "payment_rule": P7_PAYMENT_RULE},
    )
    assert kw == {"prepare_launch": True}
    approved = dict(
        payment_rule=P7_PAYMENT_RULE,
        harness_commit=CORRECTION_CHAMPION_REV,
        builder_commit=TRAIN_ONLY_POET_REV,
        capture_version=COMPARISON_CAPTURE,
    )
    result = verify_paid_job(
        dict(job, artifact_sha256=args[4]), objects, policy, approved
    )
    assert result["eligible"] and result["cost_microusd"] == 30 and result["calls"] == 3
    assert "assessment" not in manifest and "verification" in manifest


@pytest.mark.parametrize("failure", ["wrong", "thoughts"])
def test_failed_native_work_keeps_its_original_trajectory_without_claiming_a_fee(
    native, failure
):
    run, completed, *_ = native
    manifest = run(**{failure: failure == "wrong"})
    args, kw = completed[0]
    assert args[2] == "rejected" and not kw
    assert manifest["generator"] and manifest["events"] and manifest["verification"]


def test_duplicate_generation_receipts_are_not_payable(native):
    run, completed, *_ = native
    with pytest.raises(ValueError, match="receipt"):
        run(duplicate=True)
    assert not completed


@pytest.mark.parametrize("wrong", [False, True])
def test_native_export_checks_settlement_and_keeps_only_model_visible_data(
    native, tmp_path, monkeypatch, wrong
):
    from zipfile import ZipFile
    from lab_arena import collection_trajectories
    from scripts.collection_batch import write_exports

    run, completed, job, objects, policy, raw = native
    run(wrong=wrong)
    args, _ = completed[0]
    model = job["model"]
    job = dict(job, slot=0, attempt=1, status=args[2], artifact_sha256=args[4])
    tariffs = dict(
        schema="leadpoet.collection.verified_p7_prices.v1",
        payment_rule=P7_PAYMENT_RULE,
        live_batch_id=policy.batch_id,
        routes={model: raw["routes"][model]},
        prices={"P": raw["fixed_quotes_microusd"]},
        budgets={"P": raw["budgets_microusd"]},
        capture_version=COMPARISON_CAPTURE,
        split_manifest_sha256=judge.digest(raw["split_manifest"]),
        judge_prompt_sha256=policy.prompt_sha256,
        harness_commit=CORRECTION_CHAMPION_REV,
        builder_commit=TRAIN_ONLY_POET_REV,
    )

    def rpc(name, args):
        if name == "collection_reward_schedule_v1":
            return {"pilot_evidence": tariffs}
        assert name == "collection_batch_jobs_v1"
        return [job]

    grant = dict(unresolved=0, miner_cost_microusd=30, generator_calls=3)

    def grants(db, batch, rows, *, mode, scope, digest, allow_pending):
        assert (
            batch == policy.batch_id
            and digest == raw["terms_sha256"]
            and mode == "live"
        )
        return {job["job_id"]: grant}

    monkeypatch.setattr(collection_trajectories, "verified_grants", grants)
    out = tmp_path / "native.zip"
    db = SimpleNamespace(rpc=rpc)
    meta = write_exports(raw, PROMPT, db, objects, out)
    assert meta["rows"] == 1 and "qualification_sha256" not in meta
    with ZipFile(out) as z:
        row = json.loads(z.read("trajectories/native-payment.jsonl"))
        assert row["verified_task_pass"] is (not wrong)
        assert (
            row["judgment_status"] is None
            and row["verification_method"] == P7_PAYMENT_RULE
        )
        assert (
            row["length"]["reasoning_bytes"] > 0
            and row["length"]["tool_result_bytes"] > 0
        )
        assert "initial_state" not in row and "reference_answer" not in row
    grant["miner_cost_microusd"] += 1
    with pytest.raises(ValueError, match="funded outcome"):
        write_exports(raw, PROMPT, db, objects, tmp_path / "mismatch.zip")


def test_missing_original_wire_blocks_first_payment_after_restart(native, tmp_path):
    run, completed, job, objects, policy, _ = native
    run()
    # Remove a test-owned object from the local store to simulate a failed upload.
    pointer = json.loads(
        next(tmp_path.glob("collection/*/*/saved/wire-generate-1.json")).read_bytes()
    )
    (tmp_path / pointer["key"]).unlink()
    approved = dict(
        payment_rule=P7_PAYMENT_RULE,
        harness_commit=CORRECTION_CHAMPION_REV,
        builder_commit=TRAIN_ONLY_POET_REV,
        capture_version=COMPARISON_CAPTURE,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_paid_job(
            dict(job, artifact_sha256=completed[0][0][4]), objects, policy, approved
        )
