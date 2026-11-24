"""A training row must retain negative/partial outcomes without importing hidden grader state."""
import json
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import pytest


def examples():
    messages = [{"role":"system","content":"sell"}, {"role":"user","content":"task"},
                {"role":"assistant","content":"","reasoning_content":"check","tool_calls":[{"id":"x","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"x","content":"record"}, {"role":"assistant","content":"answer"}]
    z = {"score":0.0,"credits":{"task":0.0},"hard_fail":None,"v2":{"hidden":"secret gold answer"}}
    fail = {**z,"hard_fail":"suppressed_contact","score":0.0}
    g = {"teacher":"test/model","run":"r1","messages":messages,"tools":[],"label_turns":[2,4],"no_loss_turns":[],
         "evidence":{"finish":"answered","grade":fail,"initial_state":{"gold":"secret gold answer"},"step_checks":[
             {"turn":0,"before":z,"after":fail},{"turn":1,"before":fail,"after":fail}]}}
    packet={"checks":{"answered":True,"privacy":True,"training_split":True,"shape":True,"code_grade":False,"no_hard_failure":False},
            "claim_reader":{"units":[],"results":[]}}
    job={"job_id":"r1","batch_id":"b","model":"test/model","scenario_id":"P7-s10","slot":0,"attempt":1,"status":"accepted",
         "artifact_sha256":"sha256:"+"a"*64,"generator_calls":2,"miner_cost_microusd":47,"unresolved":0}
    kw=dict(job=job,generator=g,packet=packet,calls=[{"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}},
            {"usage":{"prompt_tokens":8}}],settings_sha256="sha256:"+"b"*64,harness="a"*40,builder="c"*40,revision="test/model-v1",route="test/zdr",
            split={"template_id":"P7","rule_pool":"train","tool_variant":"ledger"},terms_sha256="sha256:"+"c"*64)
    return kw


def test_full_negative_has_aligned_labels_unknown_usage_and_no_private_state():
    from lab_arena.collection_trajectories import trajectory_row
    row=trajectory_row(**examples())
    assert row["messages"][2]["reasoning_content"]=="check" and row["messages"][3]["content"]=="record"
    assert row["verified_task_pass"] is False and row["finish_reason"]=="answered" and row["final_turn_present"]
    assert row["first_failed_check"]=={"turn":0,"check":"suppressed_contact","kind":"native_hard_failure"}
    assert row["tokens"]["completion_tokens"] is None and row["tokens"]["known_completion_tokens"]==3
    assert row["miner_cost_microusd"]==47 and row["length"]["message_count"]==5
    assert row["step_checks"][0]["message_index"]==2 and "secret gold answer" not in json.dumps(row)


def test_unanswered_and_cannot_judge_are_not_verified_preferences():
    from lab_arena.collection_trajectories import trajectory_row, TrajectoryArchive
    kw=examples()
    kw["generator"]["messages"].pop()
    kw["generator"]["label_turns"]=[2]
    kw["generator"]["evidence"].update(finish="max_steps",step_checks=[])
    kw["packet"]["checks"]["answered"]=False
    row=trajectory_row(**kw,assessment={"status":"cannot_judge"})
    assert row["verified_task_pass"] is None and not row["final_turn_present"] and row["finish_reason"]=="max_steps"
    assert row["judgment_status"]=="cannot_judge"
    with ZipFile(BytesIO(),"w") as z:
        a=TrajectoryArchive(z);a.write(row);summary=a.finish()
        assert summary["unanswered_runs"]==1 and summary["cannot_judge_runs"]==1 and summary["preference_pairs"]==0


def test_repeat_preferences_reference_both_complete_verified_rows():
    from lab_arena.collection_trajectories import trajectory_row,TrajectoryArchive
    kw=examples();fail=trajectory_row(**kw)
    good={**fail,"run_id":"r2","verified_task_pass":True}
    buf=BytesIO()
    with ZipFile(buf,"w") as z:
        a=TrajectoryArchive(z);a.write(fail);a.write(good);meta=a.finish()
    with ZipFile(buf) as z:
        preference=json.loads(z.read("preferences.jsonl"))
        for key,rid in [('chosen','r2'),('rejected','r1')]:
            target=preference[key]; data=z.read(target['path'])
            assert json.loads(data)['run_id']==rid
            assert target['sha256']=="sha256:"+sha256(data).hexdigest()
        assert meta["preference_pairs"]==1
