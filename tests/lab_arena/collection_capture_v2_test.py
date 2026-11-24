from copy import deepcopy
from types import SimpleNamespace

import pytest

from lab_arena import collection_experiment as experiments
from lab_arena import collection_judge as judge
from lab_arena.collection_workflow import comparison_revisions
from tests.lab_arena.collection_experiment_test import settings


def test_separate_call_and_run_limits_meter_unknown_without_fabricating_usage():
    raw = settings()
    arm = raw["arms"][0]
    arm.update(max_tokens=64, max_run_tokens=150)
    experiments.load_experiment(raw, {})
    model = arm["model"]
    limits = []
    usage = [None, 30, 56]

    def model_call(request, body):
        limits.append(body["max_tokens"])
        tokens = usage.pop(0)
        return {
            "model": model,
            "id": request,
            "usage": {} if tokens is None else {"completion_tokens": tokens},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    broker = SimpleNamespace(model_call=model_call, versions={model: "revision"})
    meter = experiments.ExperimentMeter(broker, arm)
    for idx in range(3):
        meter.model_call(str(idx), {"model": model, "max_tokens": 100})
    assert limits == [64, 64, 56]
    assert meter.remaining == 0 and "completion_tokens" not in meter.calls[0]["usage"]
    with pytest.raises(experiments.OutputLimitReached):
        meter.model_call("again", {"model": model, "max_tokens": 64})


def test_the_capture_and_output_budget_are_frozen_into_batch_identity():
    raw = settings()
    before = experiments.load_experiment(raw, {})
    revised = deepcopy(raw)
    revised["capture_version"] = "collection.workflow-events.v2"
    revised["arms"][0]["max_run_tokens"] = 65536
    assert experiments.load_experiment(revised, {}).batch_id != before.batch_id
    assert comparison_revisions(revised) == comparison_revisions(
        {"capture_version": "collection.workflow-events.v1"}
    )
    revised["arms"][0]["max_run_tokens"] = 0
    with pytest.raises(ValueError):
        experiments.load_experiment(revised, {})


def test_chunked_judging_covers_every_byte_and_binds_each_verdict():
    import json
    from lab_arena import collection_judge_segments as segments
    from tests.lab_arena.collection_judge_test import (
        ASSIGN,
        POLICY,
        PROMPT,
        packet,
        pass_verdict,
    )

    p = packet()
    p["steps"][2]["reasoning"] = "Original € evidence. " * 3000
    p["steps"][-1]["evidence"] += " Late source contradicts the price."
    segments.annotate_packet(p)
    received = {}

    def call(**request):
        last = json.loads(request["messages"][-1]["content"])
        if "data" in last:
            received[last["part"]] = last["data"]
            return json.dumps(
                {
                    "part": last["part"],
                    "source": last["source"],
                    "complete": True,
                    "note": "Preserved fragment " + str(last["part"]),
                }
            )
        assert len(last["run"]["segment_evidence"]) == p["judge_segments"]["parts"]
        return json.dumps(pass_verdict(last["identity"]))

    result = judge.judge_run(POLICY, ASSIGN, p, call, prompt=PROMPT)
    assert result["status"] == "accepted"
    assert "".join(received[i] for i in sorted(received)) == judge._json(
        segments.source(p)
    )
    segments.verify_notes(p, result)
    result["segment_reviews"]["fable"].pop()
    with pytest.raises(judge.InvalidVerdict, match="complete segment coverage"):
        segments.verify_notes(p, result)


def test_unread_fragment_holds_the_entire_judgment():
    import json
    from lab_arena import collection_judge_segments as segments
    from tests.lab_arena.collection_judge_test import ASSIGN, POLICY, PROMPT, packet

    p = packet()
    p["facts"] = "recorded datum " * 2400
    segments.annotate_packet(p)

    def call(**request):
        part = json.loads(request["messages"][-1]["content"])
        return json.dumps(
            {
                "source": part["source"],
                "part": part["part"],
                "complete": False,
                "note": "Missing context for this observation.",
            }
        )

    result = judge.judge_run(POLICY, ASSIGN, p, call, prompt=PROMPT)
    assert result["status"] == "cannot_judge"
    assert all(v["cannot_judge"] for v in result["records"])
    assert p["judge_segments"]["bytes"] == len(judge._json(segments.source(p)).encode())


def test_real_pinned_workflow_keeps_a_truncated_reply_and_quarantines_credentials(
    tmp_path,
):
    import json
    import os
    from lab_arena.collection_journal import WorkflowJournal
    from lab_arena.collection_workflow import (
        workflow_sources,
        play_workflow,
        TRAIN_ONLY_POET_REV,
    )
    from lab_arena.collection_generation import CAPTURE_V2, QuarantinedWorkflow
    from lab_arena.service import LocalObjectStore

    agent, poet = (
        os.environ.get(k)
        for k in ("COLLECTION_TEST_AGENT_REPO", "COLLECTION_TEST_POET_REPO")
    )
    if not agent or not poet:
        pytest.skip("requires both authorized pinned source repositories")
    objects = LocalObjectStore(tmp_path)
    job = {
        "job_id": "capture-length",
        "model": "mistralai/mistral-large-4-0",
        "scenario_id": "P7-s40401",
    }
    reply = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "Cut off",
                    "reasoning_content": "Compare the permitted allocations",
                },
            }
        ],
        "id": "synthetic-receipt",
        "model": "mistralai/mistral-large-4-0",
        "usage": {},
    }
    broker = SimpleNamespace(model_call=lambda *_: deepcopy(reply))
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=TRAIN_ONLY_POET_REV
    ) as sources:
        journal = WorkflowJournal(objects, "test", job["job_id"])
        row, packet = play_workflow(
            *sources,
            job,
            broker,
            record=journal.append,
            save=journal.save,
            capture_version=CAPTURE_V2
        )
        assert not packet["checks"]["answered"]
        assert (
            row["messages"][-1]["reasoning_content"]
            == "Compare the permitted allocations"
        )
        summary = row["evidence"]["generation"]
        assert (
            summary["finish_reason"] == "output_limit"
            and not summary["final_turn_present"]
        )
        assert (
            summary["completion_tokens"] is None
            and summary["unknown_completion_calls"] == 1
        )
        assert summary["response_finish_reasons"] == ["length"]
        assert row["evidence"]["step_checks"][0]["after"] == row["evidence"]["grade"]
        assert objects.get_bounded(
            "collection/test/capture-length/saved/generator.json", 4096
        )
        assert objects.get_bounded(
            "collection/test/capture-length/saved/workflow-output.json", 4096
        )
        events = [json.loads(raw) for raw, _ in journal.read(journal.hashes)]
        assert all(
            e["visible_training_safe"]
            for e in events
            if e["kind"] in ("start", "reply", "tool")
        )
        reply["choices"][0]["message"]["content"] = "sk-or-" + "X" * 40
        job["job_id"] = "capture-quarantine"
        journal = WorkflowJournal(objects, "test", job["job_id"])
        with pytest.raises(QuarantinedWorkflow):
            play_workflow(
                *sources,
                job,
                broker,
                record=journal.append,
                save=journal.save,
                capture_version=CAPTURE_V2
            )
        assert objects.get_bounded(
            "collection/test/capture-quarantine/saved/quarantine.json", 4096
        )
        assert not objects.get_optional_bounded(
            "collection/test/capture-quarantine/saved/workflow-output.json", 4096
        )
        assert all(
            b"sk-or-" not in file.read_bytes() for file in tmp_path.rglob("*.json")
        )


def test_final_grade_exception_preserves_the_original_answer(tmp_path):
    import json
    import os
    from lab_arena.collection_journal import WorkflowJournal
    from lab_arena.collection_workflow import (
        workflow_sources,
        play_workflow,
        TRAIN_ONLY_POET_REV,
    )
    from lab_arena.service import LocalObjectStore

    agent, poet = (
        os.environ.get(k)
        for k in ("COLLECTION_TEST_AGENT_REPO", "COLLECTION_TEST_POET_REPO")
    )
    if not agent or not poet:
        pytest.skip("requires both authorized pinned source repositories")
    objects = LocalObjectStore(tmp_path)
    journal = WorkflowJournal(objects, "case", "grader-outage")
    job = {
        "job_id": "grader-outage",
        "model": "mistralai/mistral-large-4-0",
        "scenario_id": "P7-s40401",
    }

    def call(request_id, body):
        return {
            "model": job["model"],
            "id": "receipt-test",
            "usage": {"completion_tokens": 4},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Original model answer.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    with workflow_sources(
        agent, poet, corrections=True, builder_revision=TRAIN_ONLY_POET_REV
    ) as (h, b):
        grade = b.grade

        def fail_on_answer(task, out):
            if out.get("answer"):
                raise RuntimeError("injected final grader interruption")
            return grade(task, out)

        b.grade = fail_on_answer
        with pytest.raises(RuntimeError, match="injected final grader"):
            play_workflow(
                h,
                b,
                job,
                SimpleNamespace(model_call=call),
                record=journal.append,
                save=journal.save,
            )
    ref = json.loads(
        objects.get_bounded(
            "collection/case/grader-outage/saved/workflow-output.json", 4096
        )
    )
    doc = json.loads(objects.get_bounded(ref["key"], 1024 * 1024))
    assert (
        doc["out"]["answer"]
        == doc["messages"][-1]["content"]
        == "Original model answer."
    )
    assert (
        doc["generation"]["final_turn_present"]
        and doc["generation"]["completion_tokens"] == 4
    )
    assert any(
        json.loads(event)["kind"] == "reply"
        for event, _ in journal.read(journal.hashes)
    )
