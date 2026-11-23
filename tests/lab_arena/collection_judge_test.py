"""Collection judgments use scripted provider payloads and synthetic evidence."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import collection_judge as j


PROMPT = dict(
    rubric_version="fixture",
    tag_version="fixture",
    run_instructions="scripted run",
    ranking_instructions="scripted ranking",
    rubric="test rubric",
    examples={str(i): ["fixture one", "fixture two"] for i in range(4)},
)
POLICY = j.BatchPolicy("batch-one", "secret-seed", j.digest(PROMPT))
ASSIGN = {"run_id": "hidden-miner/run-one", "primary": "fable", "overlap": False}


def packet(rid=ASSIGN["run_id"]):
    return {
        "run_id": rid,
        "scenario_id": "job-123",
        "task": "Check the mock account",
        "facts": "Mock joined Tuesday",
        "reference_answer": "Tuesday",
        "grading_notes": "The account was suppressed on Wednesday.",
        "final_output": "Tuesday",
        "steps": [
            {
                "step": i,
                "action": "read",
                "evidence": "Tuesday",
                "reasoning": "test thought",
            }
            for i in range(1, 6)
        ],
        "checks": {"account": True},
        "claim_reader": {"units": [], "results": []},
        "miner": "hidden-miner",
        "model": "hidden-generator",
        "private_key": "never-read",
    }


def pass_verdict(identity):
    return {
        **identity,
        "scores": dict.fromkeys(j.PARTS, 3),
        "evidence": {},
        "failure_tags": [],
        "first_bad_step": None,
        "hard_steps": [1, 3],
        "reasoning": "sound",
        "reasoning_matches_actions": True,
        "unsupported_claims": [],
        "confidence": "high",
        "cannot_judge": False,
        "cannot_judge_reason": None,
    }


class Caller:
    def __init__(self, alter=None):
        self.requests = []
        self.alter = alter

    def __call__(self, **req):
        self.requests.append(req)
        last = json.loads(req["messages"][-1]["content"])
        if "runs" in last:
            order = list(last["runs"])[::-1]
            out = dict(
                last["identity"],
                order=order,
                margin="clear",
                why_first="Grounded claim.",
                loser_tags={k: "needless_search" for k in order[1:]},
            )
        else:
            out = pass_verdict(last["identity"])
        if self.alter:
            replacement = self.alter(out, len(self.requests))
            if replacement is not None:
                return replacement
        return json.dumps(out)


def test_assignment_is_order_independent_balanced_and_reproducible():
    rows = [
        dict(
            run_id=str(i),
            generator="g" + str(i % 6),
            miner="m" + str(i % 10),
            harness="h" + str(i % 3),
        )
        for i in range(100)
    ]
    got = j.assign_judges(POLICY, rows)
    assert j.assign_judges(POLICY, list(reversed(rows))) == got
    assert Counter(x["primary"] for x in got.values()) == {"fable": 50, "astra": 50}
    assert sum(x["overlap"] for x in got.values()) == 10
    assert all(
        x["overlap"]
        for x in j.assign_judges(replace(POLICY, pilot=True), rows).values()
    )
    with pytest.raises(ValueError):
        j.assign_judges(POLICY, [*rows, rows[0]])


def test_code_fail_cannot_spend_and_untrusted_inputs_are_excluded():
    c = Caller()
    p = packet()
    p["checks"]["account"] = False
    assert (
        j.judge_run(POLICY, ASSIGN, p, c, prompt=PROMPT)["status"] == "code_rejected"
        and not c.requests
    )
    p["checks"]["account"] = True
    result = j.judge_run(POLICY, ASSIGN, p, c, prompt=PROMPT)
    assert result["status"] == "accepted" and len(c.requests) == 1
    wire = json.dumps(c.requests[0])
    assert all(
        s not in wire
        for s in ["hidden-miner", "hidden-generator", "never-read", "job-123"]
    )
    assert (
        c.requests[0]["model"] == POLICY.fable_version
        and c.requests[0]["max_tokens"] == 400
    )
    assert result["records"][0]["run_id"] == p["run_id"]
    assert j.public_result(result)["judges"][0].keys() == {"scores", "failure_tags"}
    with pytest.raises(j.InvalidVerdict):
        j.judge_run(POLICY, ASSIGN, p, c, prompt=dict(PROMPT, rubric="changed"))
    assert len(c.requests) == 1


@pytest.mark.parametrize(
    "alter",
    [
        lambda v, n: "bad json",
        lambda v, n: v["scores"].update(truth=True),
        lambda v, n: v.update(first_bad_step=6),
        lambda v, n: v.update(judge_version="other"),
        lambda v, n: v.update(failure_tags=["forbidden"]),
        lambda v, n: v.update(ideal_answer="Leak"),
        lambda v, n: v.update(hard_steps=[1, 1]),
        lambda v, n: v.update(
            unsupported_claims=[{"step": 1, "claim": "bad", "kind": "unknown"}]
        ),
        lambda v, n: v["scores"].update(efficiency=1),
        lambda v, n: '{"a":2,"a":3}',
    ],
)
def test_invalid_evidence_retried_once_per_judge_then_unusable(alter):
    c = Caller(alter)
    result = j.judge_run(POLICY, ASSIGN, packet(), c, prompt=PROMPT)
    assert result["status"] == "cannot_judge" and result["keep_steps"] == 0
    assert len(c.requests) == 2
    assert len({r["request_id"] for r in c.requests}) == 2
    assert (
        result["records"] == []
        and result["verifier_unavailable"]["reason"] == "invalid_verdict"
    )


def test_budget_and_transport_failures_propagate_without_new_paid_identity():
    calls = []

    def fail(**req):
        calls.append(req["request_id"])
        raise TimeoutError("transport unavailable")

    for _ in range(2):
        with pytest.raises(TimeoutError):
            j.judge_run(POLICY, ASSIGN, packet(), fail, prompt=PROMPT)
    assert len(set(calls)) == 1


def fail_truth(v):
    v["scores"]["truth"] = 0
    v["failure_tags"] = ["made_up_fact"]
    v["first_bad_step"] = 4
    v["evidence"]["truth"] = {
        "step": 4,
        "note": "The cited day contradicts the record.",
    }


def test_secondary_rechecks_low_confidence_and_disagreement_holds():
    def disagreement(v, n):
        if n == 1:
            v["confidence"] = "low"
        else:
            fail_truth(v)

    c = Caller(disagreement)
    result = j.judge_run(POLICY, ASSIGN, packet(), c, prompt=PROMPT)
    assert result["status"] == "needs_review" and result["keep_steps"] == 0
    assert [r["model"] for r in c.requests] == [
        POLICY.fable_version,
        POLICY.astra_version,
    ]


def test_salvage_only_original_prefix_with_agreed_hard_steps():
    def rejected(v, n):
        fail_truth(v)
        v["hard_steps"] = [1, 3, 4] if n == 1 else [3, 4]

    r = j.judge_run(
        POLICY, dict(ASSIGN, overlap=True), packet(), Caller(rejected), prompt=PROMPT
    )
    assert (r["status"], r["keep_steps"], r["hard_steps"]) == ("salvaged", 3, [3])


def test_rank_top_three_blind_and_bind_to_original_accepted_packet():
    c = Caller()
    candidates = []
    for i in range(4):
        p = packet("hidden-run-" + str(i))
        a = dict(ASSIGN, run_id=p["run_id"])
        candidates.append((p, j.judge_run(POLICY, a, p, c, prompt=PROMPT)))
    rank = j.rank_scenario(POLICY, candidates, c, prompt=PROMPT)
    assert rank["status"] == "ranked" and len(rank["labels"]) == 3
    assert "hidden-run" not in json.dumps(c.requests[-1])
    assert rank == j.rank_scenario(POLICY, list(reversed(candidates)), c, prompt=PROMPT)
    p, _ = candidates[0]
    p["final_output"] = "substituted after assessment"
    with pytest.raises(ValueError, match="content changed"):
        j.rank_scenario(POLICY, candidates, c, prompt=PROMPT)


def test_batch_release_holds_missing_human_marks_drift_and_planted_failures():
    policy = replace(POLICY, pilot=True)
    ids = [f"job-{i}" for i in range(132)]
    assignments = [
        dict(run_id=r, generator=f"model-{i%6}", harness="variant", miner="payer")
        for i, r in enumerate(ids)
    ]
    schedules = j.assign_judges(policy, assignments)
    packets = {r: packet(r) for r in ids}
    results = [
        j.judge_run(
            policy,
            schedules[r],
            packets[r],
            Caller((lambda v, n: fail_truth(v)) if r == ids[-1] else None),
            prompt=PROMPT,
        )
        for r in ids
    ]
    planted = {ids[-1]: False, ids[-2]: True}
    drift = {
        r: {name: dict.fromkeys(j.PARTS, 3) for name in j.JUDGES} for r in ids[100:130]
    }
    aliases, jobs = j.blind_human_sample(policy, [packets[r] for r in ids[:100]])
    assert len(jobs) == 50 and all(
        "job-" not in json.dumps(job) for job in jobs.values()
    )
    marks = {
        a: dict(
            source="human_blind",
            reviewer="test fixture reviewer",
            scores=dict.fromkeys(j.PARTS, 3),
        )
        for a in aliases
    }
    inputs = dict(
        assignments=assignments,
        assessments=results,
        human_marks=marks,
        planted=planted,
        drift=drift,
    )
    report = j.quality_report(policy, **inputs)
    assert (
        report["quality_ready"]
        and report["random_overlap_count"] == 100
        and report["blind_human_marks"] == 50
    )
    assert (
        "human_sample_incomplete"
        in j.quality_report(policy, **dict(inputs, human_marks={}))["holds"]
    )
    drift[ids[100]]["fable"]["truth"] = 0
    planted[ids[-1]] = True
    holds = j.quality_report(policy, **inputs)["holds"]
    assert "drift_detected" in holds and "planted_control_mismatch" in holds
    with pytest.raises(ValueError, match="provenance"):
        j.quality_report(
            policy,
            **dict(
                inputs,
                human_marks={a: dict(m, source="model") for a, m in marks.items()},
            ),
        )


def test_sealed_cohorts_reconstruct_independently_and_agreement_uses_batch_slots():
    jobs = [
        dict(
            run_id=str(i),
            generator=str(i % 6),
            harness="standard",
            miner=str(i % 10),
            cohort=i // 6,
            slot=i,
        )
        for i in range(60)
    ]
    scheduled = {}
    for i in range(0, 60, 6):
        cohort = j.assign_judges(POLICY, jobs[i : i + 6])
        assert sum(a["primary"] == "fable" for a in cohort.values()) == 3
        scheduled.update(cohort)
    assert j.assign_judges(POLICY, list(reversed(jobs))) == scheduled
    assert [
        sum(scheduled[str(i)]["overlap"] for i in range(at, at + 10))
        for at in range(0, 60, 10)
    ] == [1] * 6
    jobs[-1]["slot"] = 0
    with pytest.raises(ValueError, match="slots"):
        j.assign_judges(POLICY, jobs)


def test_ranking_uses_each_runs_own_trimmed_evidence_and_ignores_code_failure_before_steps():
    pairs = []
    c = Caller()
    for rid in ("left", "right", "broken"):
        p = packet(rid)
        p["facts"] = "matched source " + rid
        if rid == "broken":
            p["checks"]["shape"] = False
            p.pop("steps")
        a = j.judge_run(POLICY, dict(ASSIGN, run_id=rid), p, c, prompt=PROMPT)
        pairs.append((p, a))
    assert j.rank_scenario(POLICY, pairs, c, prompt=PROMPT)["status"] == "ranked"
    message = json.loads(c.requests[-1]["messages"][-1]["content"])
    assert {r["facts"] for r in message["runs"].values()} == {
        "matched source left",
        "matched source right",
    }


def test_claim_reader_refuses_unbound_disagreement_and_requires_missing_written_checks():
    c = Caller()
    p = packet()
    unit = {
        "id": "reply",
        "text": "The renewal is Friday.",
        "sources": {"crm": "Renewal is Monday."},
    }
    p["claim_reader"] = {"units": [unit], "results": []}
    asses = lambda: j.judge_run(POLICY, ASSIGN, p, c, prompt=PROMPT)
    a = asses()
    assert (
        a["status"] == "accepted"
        and a["claim_review"] == "missing"
        and not j.training_selection(a)
    )
    wrapped = {
        "unit_sha256": j.digest(unit),
        "record": {
            "id": "reply",
            "strict": True,
            "sentences": [{"text": unit["text"], "level": "invented", "unread": 0}],
        },
    }
    p["claim_reader"]["results"].append(wrapped)
    a = asses()
    assert a["claim_review"] == "disputed" and not j.training_selection(a)
    wrapped["unit_sha256"] = "wrong"
    with pytest.raises(ValueError, match="unbound"):
        asses()


def test_blinded_evidence_keeps_cited_and_contradictory_late_rows_or_explicitly_abstains():
    from lab_arena.collection_workflow import _steps, IncompleteEvidence

    filler = "irrelevant filler. " * 700
    raw = (
        filler
        + "\nTicket C-18 still pending.\n"
        + filler
        + "\nC-18 superseded; quote rejected."
    )
    msgs = [
        {"role": "assistant", "content": "check record"},
        {"role": "tool", "content": raw},
        {"role": "assistant", "content": "C-18 is closed."},
    ]
    steps = _steps(msgs, "Confirm C-18 from source", "C-18 is closed.")
    assert (
        "still pending" in steps[0]["evidence"]
        and "quote rejected" in steps[0]["evidence"]
    )
    assert len(steps[0]["evidence"]) < 8192 and msgs[1]["content"] == raw
    with pytest.raises(IncompleteEvidence):
        _steps(
            [
                {"role": "assistant", "content": "read"},
                {"role": "tool", "content": "C-18 " * 4000},
            ],
            "C-18",
            "C-18",
        )
    p = packet()
    p["projection_error"] = "selected_evidence_exceeds_limit"
    p["steps"] = []
    caller = Caller()
    a = j.judge_run(POLICY, ASSIGN, p, caller, prompt=PROMPT)
    assert (
        a["status"] == "cannot_judge" and not caller.requests and a["keep_steps"] == 0
    )


def test_judge_receives_complete_long_reasoning_from_an_open_generator():
    from lab_arena.collection_workflow import _steps, IncompleteEvidence

    reasoning = "a" * 9857
    msgs = [{"role": "assistant", "content": "Wait.", "reasoning": reasoning}]
    p = packet()
    p["steps"] = _steps(msgs, "Wait.", "Wait.")
    _, run = j._packet(p)
    assert run["steps"][0]["reasoning"] == reasoning
    msgs[0]["reasoning"] = "b" * 16385
    with pytest.raises(IncompleteEvidence, match="visible_reasoning_exceeds_limit"):
        _steps(msgs, "Wait.", "Wait.")


@pytest.mark.parametrize("completed", [0, 1])
def test_owner_verifier_outage_keeps_only_actual_votes(completed):
    run, call = packet(), Caller()
    attempts = []

    def throttled(**request):
        attempts.append(request)
        if len(attempts) > completed:
            raise j.JudgeUnavailable(
                request["model"], request["request_id"], "http_error"
            )
        return call(**request)

    a = j.judge_run(POLICY, dict(ASSIGN, overlap=True), run, throttled, prompt=PROMPT)
    assert a["status"] == "cannot_judge" and not j.training_selection(a)
    assert (
        len(a["records"]) == completed and not a["keep_steps"] and not a["hard_steps"]
    )
    assert a["verifier_unavailable"] == {
        "model": attempts[-1]["model"],
        "request_id": attempts[-1]["request_id"],
        "reason": "http_error",
    }
