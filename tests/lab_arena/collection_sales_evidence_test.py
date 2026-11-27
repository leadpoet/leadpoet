"""Offline full-funnel source work through the exact pinned public workflow and private task producer."""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from lab_arena import collection_judge as judge
from lab_arena.collection_generation import encoded
from lab_arena.collection_intake import MODELS
from lab_arena.collection_sales_evidence import sales_correction_source
from lab_arena.collection_worker import (
    process_job,
    native_payment_evidence,
    registered_native_result,
    SALES_EVIDENCE_RULE,
)
from lab_arena.collection_workflow import (
    COMPARISON_CAPTURE,
    CORRECTION_CHAMPION_REV,
    SALES_EVIDENCE_POET_REV,
    build_scenario,
    comparison_revisions,
    workflow_sources,
)
from lab_arena.collection_quality import admit_built_split, require_release_partition
from lab_arena.service import LocalObjectStore
from tests.lab_arena.collection_judge_test import PROMPT


class OriginalReply:
    """Test-only broker preserving ordinary caller responsibilities: genuine tests never debit a miner."""

    record_generation = None

    def __init__(self, steps, tag):
        self.steps = iter(steps)
        self.tag = tag
        self.prefixes = []

    def model_call(self, req, params):
        self.prefixes.append(params["messages"])
        action = next(self.steps)
        msg = {
            "role": "assistant",
            "reasoning": "Examine the dated source and its issuing account.",
        }
        if isinstance(action, str):
            msg.update(content=action)
        else:
            msg.update(
                content="",
                tool_calls=[
                    {
                        "id": self.tag + req,
                        "type": "function",
                        "function": {
                            "name": action[0],
                            "arguments": json.dumps(action[1]),
                        },
                    }
                ],
            )
        rid = "g-" + self.tag + "-" + req
        out = {
            "model": params["model"],
            "id": rid,
            "choices": [
                {
                    "message": msg,
                    "finish_reason": (
                        "tool_calls" if isinstance(action, tuple) else "stop"
                    ),
                }
            ],
            "usage": {"cost": 0.00001, "completion_tokens": 12},
        }
        raw = encoded(out)
        self.record_generation(
            req,
            hashlib.sha256(encoded(params)).hexdigest(),
            hashlib.sha256(raw).hexdigest(),
            200,
            10,
            rid,
            raw,
        )
        return out


@pytest.fixture(scope="module")
def pinned():
    agent = os.getenv("COLLECTION_TEST_AGENT_REPO")
    poet = os.getenv("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("real pinned sales harness and Poet-9B producer are required")
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=SALES_EVIDENCE_POET_REV
    ) as s:
        yield s


def execute(
    pinned,
    objects,
    raw_steps,
    *,
    seed=61021,
    mode="owner",
    model=MODELS[1],
    rid="student",
    source=None
):
    b = pinned[1]
    scenario = "P8-s" + str(seed)
    task = build_scenario(b, scenario)
    batch = ("owner-p8-" if mode == "owner" else "dv-") + "a" * 64
    p = judge.BatchPolicy(batch, "fixed", judge.digest(PROMPT), pilot=mode == "owner")
    job = dict(
        job_id=rid,
        mode=mode,
        batch_id=batch,
        model=model,
        scenario_id=scenario,
        harness_commit=CORRECTION_CHAMPION_REV,
    )
    calls = []
    if mode == "live":
        job.update(lease_token="local", miner_hotkey="fake-test", price_microusd=5000)
    sourcebroker = OriginalReply(raw_steps, rid)
    m = process_job(
        job=job,
        policy=p,
        prompt=PROMPT,
        assignment={"run_id": rid},
        intake=(
            None
            if mode == "owner"
            else SimpleNamespace(complete=lambda *a, **kw: calls.append((a, kw)))
        ),
        sources=pinned,
        objects=objects,
        broker=sourcebroker,
        epoch=lambda: 10,
        split=admit_built_split(b, task, None, b.training_partition("P8")),
        capture_version=COMPARISON_CAPTURE,
        payment_rule=SALES_EVIDENCE_RULE,
        correction_source_job=source,
    )
    job["artifact_sha256"] = "sha256:" + judge.digest(m)
    check = native_payment_evidence(objects, job, m)
    return job, m, check, calls, sourcebroker, p


def script(b, seed=61021):
    task = build_scenario(b, "P8-s" + str(seed))
    ref = b.perfect(task)
    return (
        task,
        ref,
        list(ref["calls"])
        + [
            (
                "write_file",
                {
                    "path": "qualification.json",
                    "content": ref["files"]["qualification.json"],
                },
            ),
            ref["answer"],
        ],
    )


def bad_write(ref):
    m = json.loads(ref["files"]["qualification.json"])
    m["findings"][0]["citations"][0][
        "quote"
    ] = "A plausible but unsupported account statement."
    return ("write_file", {"path": "qualification.json", "content": json.dumps(m)})


def test_sales_native_lease_reads_real_articles_and_verifies_date_entity_quotes(
    pinned, tmp_path
):
    obj = LocalObjectStore(tmp_path)
    t, ref, steps = script(pinned[1])
    j, m, e, calls, _, policy = execute(pinned, obj, steps, mode="live")
    assert (
        len(calls) == 1
        and calls[0][0][2] == "accepted"
        and calls[0][0][3]["payment_rule"] == SALES_EVIDENCE_RULE
    )
    assert (
        e["eligible"]
        and e["verified_decision"]
        and all(e["field_evidence"]["claim_checks"].values())
    )
    assert (
        all(e["field_evidence"]["citation_checks"].values())
        and len(e["searches"]) == 2
        and len(e["opened_sources"]) == 3
    )
    assert e["first_failed_decision"] is None and not e["observed_repair"]
    assert (
        e["cost_microusd"] == len(steps) * 10
        and e["calls"] == len(steps)
        and not m.get("funding_mode")
    )
    assert (
        registered_native_result(
            j,
            obj,
            policy,
            dict(
                payment_rule=SALES_EVIDENCE_RULE,
                harness_commit=CORRECTION_CHAMPION_REV,
                builder_commit=SALES_EVIDENCE_POET_REV,
                capture_version=COMPARISON_CAPTURE,
            ),
        )[1]
        == e
    )
    require_release_partition(
        {
            "cases": [{"scenario_id": j["scenario_id"]}],
            "split_sha256": "sha256:"
            + judge.digest(pinned[1].training_partition("P8")),
        }
    )
    assert comparison_revisions(
        {"capture_version": COMPARISON_CAPTURE, "scenario_ids": [j["scenario_id"]]}
    ) == (CORRECTION_CHAMPION_REV, SALES_EVIDENCE_POET_REV)


def test_native_repair_is_a_real_observation_and_exact_state_expert_branch_is_distinct(
    pinned, tmp_path
):
    obj = LocalObjectStore(tmp_path)
    t, ref, steps = script(pinned[1])
    j, m, e, calls, _, _ = execute(
        pinned, obj, steps[:-2] + [bad_write(ref), steps[-2], steps[-1]]
    )
    assert (
        e["verified_decision"]
        and e["eligible"]
        and not calls
        and m["funding_mode"] == "owner"
    )
    failed = e["first_failed_decision"]
    repair = e["observed_repair"]
    assert (
        failed["turn"] == len(steps) - 2 and repair["repair_turn"] == failed["turn"] + 1
    )
    original = sales_correction_source(obj, j)
    assert original["turn"] == failed["turn"] and original[
        "checkpoint_sha256"
    ] == judge.digest(original["checkpoint"])
    assert "qualification.json" not in original["checkpoint"]["files"]
    expert, em, c, _, broker, _ = execute(
        pinned, obj, steps[-2:], model=MODELS[2], rid="expert", source=j
    )
    assert (
        c["verified_counterfactual"]
        and c["changed_first_action"]
        and not c["eligible"]
        and c["calls"] == 2
    )
    assert (
        m.get("correction_source") is None
        and em["correction_source"]["job_id"] == j["job_id"]
    )
    visible = [
        {k: v for k, v in msg.items() if k not in {"reasoning", "reasoning_content"}}
        for msg in original["checkpoint"]["messages"]
    ]
    assert broker.prefixes[0] == visible
    assert len(visible) > 2 and "expected" not in broker.prefixes[0][0]["content"]
    assert em["batch_id"].startswith("owner-")
    with pytest.raises(ValueError, match="different|invalid owner-funded"):
        execute(pinned, obj, steps[-2:], rid="same-model", source=j)


def test_native_rejects_guessed_or_wrong_evidence_and_cannot_fabricate_failure(
    pinned, tmp_path
):
    obj = LocalObjectStore(tmp_path)
    t, ref, steps = script(pinned[1])
    j, m, e, calls, _, _ = execute(
        pinned, obj, [steps[-2], *steps[:-2], steps[-1]], rid="posthoc"
    )
    assert (
        not e["verified_decision"]
        and not e["eligible"]
        and e["first_failed_decision"]["turn"] == 0
    )
    j2, m2, e, _, _, _ = execute(pinned, obj, steps, rid="clean")
    assert e["eligible"]
    with pytest.raises(ValueError, match="no observed failed decision"):
        sales_correction_source(obj, j2)
    with pytest.raises(ValueError, match="separately funded"):
        sales_correction_source(obj, dict(j2, mode="live"))
    j3, m3, e3, paid, _, _ = execute(
        pinned, obj, steps[:-2] + [bad_write(ref), steps[-1]], rid="wrong", mode="live"
    )
    assert (
        not e3["eligible"] and paid[0][0][2] == "rejected" and not e3["observed_repair"]
    )
    assert (
        e3["first_failed_decision"]
        and e3["field_evidence"]["citation_checks"][t["state"]["count_field"]] is False
    )


def test_exact_correction_can_fix_only_the_student_final_disposition(pinned, tmp_path):
    obj = LocalObjectStore(tmp_path)
    task, ref, steps = script(pinned[1])
    source, _, e, _, _, _ = execute(
        pinned, obj, steps[:-1] + ["qualify"], rid="bad-conclusion"
    )
    assert (
        not e["eligible"]
        and e["first_failed_decision"]["reason"]
        == "incomplete_or_incorrect_final_decision"
    )
    _, _, result, _, _, _ = execute(
        pinned,
        obj,
        [ref["answer"]],
        model=MODELS[2],
        rid="one-turn-correction",
        source=source,
    )
    assert (
        result["calls"] == 1
        and result["verified_counterfactual"]
        and result["original_final_passed"] is False
    )


def test_no_silent_p8_release_without_checked_holdout(pinned):
    _, builder = pinned
    with pytest.raises(ValueError, match="private audit only"):
        require_release_partition(
            {
                "cases": [{"scenario_id": "P8-s61021"}],
                "split_sha256": "sha256:"
                + judge.digest(builder.training_partition("P7")),
            }
        )
