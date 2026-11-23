import base64
import io
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from lab_arena import collection_research as research, contracts


def job():
    return dict(job_id=str(uuid.uuid4()),model=research.collection_intake.MODELS[0],batch_id="test",
                harness_commit=research.CHAMPION_REV,scenario_id="research-case")


def params(model):
    return {"model":model,"input":[{"type":"message","role":"user","content":"Find the company."}],
            "tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}}],
            "max_output_tokens":2048}


def test_gateway_projects_only_bound_model_turns_and_keeps_labels_separate(monkeypatch):
    j=job()
    c={"icp":{"output_schema_version":"leadpoet.lab_arena.output.v6"},"deepline_catalog":None,"evaluation_date":"2026-11-23"}
    source=io.BytesIO()
    req=params(j["model"])
    research.send(source,{"type":"provider","frame":{"operation_id":"openrouter.responses",
                  "parameters":req,"action_sequence":0,"timeout_ms":90000}})
    research.send(source,{"type":"research_result","job_id":j["job_id"],"companies":[]})
    source.seek(0); upstream=io.BytesIO()
    @contextmanager
    def channel(_):
        yield source,upstream
    def fetch(request_id,body,*,api):
        assert body["model"]==j["model"] and api=="responses"
        return {"model":j["model"],"status":"completed","usage":{"cost":0.4},"output":[
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Found none."}]}]}
    run=research.RemoteResearch(channel,lambda _:c,lambda _j,x,companies: {
        "icp_sha256":contracts.document_hash(x["icp"]),"output_sha256":contracts.document_hash(companies),
        "facts":"No evidence","reference_answer":"No example","grading_notes":"No verified native result",
        "checks":{"native_coverage":False}})
    row,packet=run(j,SimpleNamespace(job=j,model_call=fetch))
    assert row["teacher"]==j["model"] and row["native_review_required"] and not all(packet["checks"].values())
    assert row["research_rows"][0]["messages"][-1]["content"]=="Found none."
    assert not any("reference_answer" in x for x in row["research_rows"])
    upstream.seek(0);started=research.receive(upstream);reply=research.receive(upstream)
    assert "lease_token" not in started and reply["document"]["status"]==200


def test_agent_request_cannot_borrow_judge_role_or_change_provider(monkeypatch):
    j=job()
    calls=[]
    b=SimpleNamespace(catalog=None,job=j,model_call=lambda *a,**k:calls.append(a))
    frame={"operation_id":"openrouter.responses","action_sequence":1,"timeout_ms":100000,
           "parameters":params("openai/gpt-6.2-astra")}
    with pytest.raises(ValueError,match="unassigned"):
        research._public_call(frame,b,{})
    assert not calls
    frame.update(operation_id="judge")
    assert research._public_call(frame,b,{})["status"]==403


def test_research_exports_one_sample_for_each_independent_conversation():
    j=job()
    row={"run":j["job_id"],"research_rows":[]}
    for n in (1,2):
        segment=research.decision_row({"op":"openrouter.responses","request":params(j["model"]),
            "response":{"status":"completed","output":[
                    {"type":"reasoning","content":[{"text":"Reasoning "+str(n)}]},
                    {"type":"function_call","name":"lookup","call_id":str(n),"arguments":"{}"}]}})
        row["research_rows"].append(segment)
    accepted={"run_id":j["job_id"],"status":"accepted","keep_steps":2,"hard_steps":[1]}
    short=research.export_research_rows(row,accepted,"hard_steps")
    assert len(short)==2
    assert short[0]["messages"][-1]["reasoning_content"]=="Reasoning 1"
    assert "reasoning_content" not in short[1]["messages"][-1]
    assert len(short[1]["messages"])==2 # does not learn a sibling worker's history
    assert len(research.export_research_rows(row,dict(accepted,status="salvaged",keep_steps=1),"decisions"))==1
    assert not research.export_research_rows(row,dict(accepted,status="rejected"),"reasoning")


def test_frozen_native_labels_require_identical_evidence():
    c={"icp":{},"labels":{contracts.document_hash({"company_name":"Exact","company_website":"a.com","intent":"funded"}):
        {"company":"Exact","class":"qualified","source_evidence":["independently verified source"]}}}
    checker=object.__new__(research.FrozenResearchCases)
    same=[{"company_name":"Exact","company_website":"a.com","intent":"funded"}]
    assert all(checker.grade(job(),c,same)["checks"].values())
    assert not checker.grade(job(),c,[dict(same[0],intent="hiring")])["checks"]["native_coverage"]
    assert not checker.grade(job(),c,same+same)["checks"]["distinct_companies"]


def test_wire_rejects_bad_frames_and_image_identity(tmp_path):
    from scripts import collection_research_worker as worker
    with pytest.raises(ValueError,match="protocol"):
        research.receive(io.BytesIO(b'{"model":"malicious"}\n'))
    j=job()
    with pytest.raises(ValueError,match="assignment"):
        worker.run_assignment(None,{**j,"harness_commit":"b"*40},None,None)
