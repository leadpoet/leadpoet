"""Positive fit checkpoints are exact private evidence, never final approvals."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lab_arena import company_judgments as cj, contracts, scoring, signal_cache_control, signal_judgments
from qualification.scoring import intent_details, lead_scorer
from qualification.scoring.company_fit_decision import COMPANY_FIT_DIMENSIONS, company_fit_match
from qualification.scoring.signal_judgment_cache import install_exchange, SignalJudgmentCacheError
from tests.test_arena_intent_retry_cache import _company_model, _icp_model, _identity
from tests.lab_arena import legacy_company_judgments_test as legacy


def native_fit():
    return lead_scorer._complete_company_fit_result(
        "match", "company fit verified from independent identity and web evidence",
        dimensions={dimension: "match" for dimension in COMPANY_FIT_DIMENSIONS},
        dimension_evidence={
            dimension: {"decision": "match", "web_evidence": {
                "url": "https://acme.example/about", "quote": "Acme provides software.",
            }, **({"web_identity_receipt": _identity()} if dimension == "identity" else {})}
            for dimension in COMPANY_FIT_DIMENSIONS
        }, stage_required=True,
    )


def snapshot():
    return {
        "schema_version": cj.FIT_CHECKPOINT_SCHEMA_VERSION,
        "company_fit_receipt": native_fit().receipt("company_fit"),
        "linkedin_profile_source": {}, "required_attribute_sources": {},
        "matched_company_sources": {"https://acme.example/about": {
            "final_url": "https://acme.example/about", "text": "Acme provides software.",
        }},
    }


def context(document):
    return {
        "assignment_id": "execute:score", "scored_run_id": document["scored_run_id"],
        "scoring_generation_floor": 0, "execution_output_ref": "arena/output.json",
        "execution_output_hash": "sha256:" + "b" * 64,
        "scoring_input_hash": contracts.document_hash(document),
        "runtime_image_digest": "sha256:" + "a" * 64,
        "runtime_image_reference": "registry/scorer@sha256:" + "a" * 64,
    }


def fit_scope(document=None, **kwargs):
    document = document or legacy._input(("Acme",))
    document["scorer_policy"]["signal_judgment_policy"] = signal_judgments.POLICY
    return signal_judgments.build_fit_scope(
        scoring_input=document, company_index=0, checkpoint_context=kwargs.pop("checkpoint_context", context(document)),
        round_id="arena-2026-10-20", network_name="finney", netuid=71,
        scorer_image_digest=kwargs.pop("scorer_image_digest", "sha256:" + "a" * 64),
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64, **kwargs,
    )


def test_snapshot_deep_copy_and_not_a_company_judgment():
    value = snapshot()
    retained = cj.validate_verified_fit_checkpoint(value)
    value["company_fit_receipt"]["dimension_evidence"]["identity"]["web_identity_receipt"]["observed_name"] = "wrong"
    assert retained["company_fit_receipt"]["dimension_evidence"]["identity"]["web_identity_receipt"]["observed_name"] == "Acme"
    assert not cj.raw_judgment_is_cacheable(retained)
    assert cj.validate_verified_fit_checkpoint(json.loads(contracts.canonical_json(retained))) == retained


@pytest.mark.parametrize("mutation", [
    lambda v: v["company_fit_receipt"].update(decision="unavailable"),
    lambda v: v["company_fit_receipt"].update(decision="mismatch"),
    lambda v: v["company_fit_receipt"]["company_fit_dimensions"].pop("industry"),
    lambda v: v["company_fit_receipt"]["company_fit_dimensions"].update(industry="unavailable"),
    lambda v: v["company_fit_receipt"].update(required_attribute_decision="mismatch"),
    lambda v: v["company_fit_receipt"].update(contract_version="other-verifier"),
    lambda v: v["company_fit_receipt"]["dimension_evidence"]["identity"].pop("web_identity_receipt"),
    lambda v: v["matched_company_sources"]["https://acme.example/about"].update(final_url="http://127.0.0.1/private"),
    lambda v: v["matched_company_sources"]["https://acme.example/about"].update(text="x" * 24001),
    lambda v: v["required_attribute_sources"].update({"https://acme.example/failed": {"status": "unavailable", "final_url": "https://acme.example/failed", "text": "source"}}),
])
def test_corrupt_negative_or_partial_snapshot_is_not_reusable(mutation):
    value = snapshot(); mutation(value)
    with pytest.raises(cj.CompanyJudgmentError):
        cj.validate_verified_fit_checkpoint(value)


def test_optional_stage_preserves_current_aggregate_rule():
    value = snapshot(); receipt = value["company_fit_receipt"]
    receipt["company_fit_stage_required"] = False
    receipt["company_fit_dimensions"]["stage"] = "unavailable"
    receipt["dimension_evidence"]["stage"]["decision"] = "unavailable"
    assert cj.validate_verified_fit_checkpoint(value) == value


@pytest.mark.parametrize("field", ["company", "icp", "evidence", "policy", "date", "execution", "assignment", "generation", "artifact", "runtime", "image"])
def test_scope_changes_for_every_authoritative_input(field):
    document = legacy._input(("Acme",)); document["scorer_policy"]["signal_judgment_policy"] = signal_judgments.POLICY
    before = fit_scope(deepcopy(document))
    kwargs = {}
    if field == "company": document["companies"][0]["company_name"] = "Other"
    elif field == "icp": document["icp"]["industry"] = "Manufacturing"
    elif field == "evidence": document["companies"][0]["intent_signals"][0]["snippet"] = "Changed evidence"
    elif field == "policy": document["scorer_policy"]["industry_evidence_model"] = "different-verifier"
    elif field == "date": document["evaluation_date"] = "2026-10-21"
    elif field == "execution": document["scored_run_id"] = "another-execution"
    else:
        changed = context(document)
        mapping = {"assignment": "assignment_id", "generation": "scoring_generation_floor", "artifact": "execution_output_hash", "runtime": "runtime_image_digest"}
        if field == "image": kwargs["scorer_image_digest"] = "sha256:" + "e" * 64
        else: changed[mapping[field]] = 1 if field == "generation" else "sha256:" + "e" * 64
        kwargs["checkpoint_context"] = changed
    assert fit_scope(document, **kwargs)["cache_key"] != before["cache_key"]


def test_typed_control_rejects_foreign_or_normal_positions():
    request = {"operation": "claim", "company_index": 0, "signal_index": -1, "checkpoint_kind": cj.FIT_CHECKPOINT_KIND}
    assert signal_cache_control.validate_request(request) == request
    for changed in ({"signal_index": 0}, {"signal_index": -2}, {"checkpoint_kind": "unknown"}, {"scored_run_id": "foreign"}):
        with pytest.raises(ValueError): signal_cache_control.validate_request({**request, **changed})


def test_nonserializable_and_multibyte_snapshot_release_without_changing_score(monkeypatch):
    async def signal(*_args, **_kwargs): return 80, 80, 1, 100, False, []
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal)
    completions = []
    async def exchange(request):
        if request["operation"] == signal_cache_control.FIT_CAPABILITY_OPERATION:
            return signal_cache_control.fit_checkpoint_capability()
        if request["operation"] == "claim": return {"status": "miss", "reference": {"authority_slot": 0}}
        completions.append(request["judgment"])
        return {"status": "released"}
    for variant in ("multibyte", "nonserializable"):
        async def verify(*_args, **kwargs):
            value = native_fit()
            if variant == "multibyte":
                # Every individual page fits its native limit; the exact
                # aggregate UTF8 envelope exceeds the worker control budget.
                for index in range(14):
                    url = f"https://acme.example/page{index}"
                    kwargs["matched_company_source_sink"][url] = {"final_url": url, "text": "界" * 24000}
            else: value.details["unserializable"] = object()
            return value
        monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify)
        baseline = asyncio.run(lead_scorer.score_company_competition_intent(_company_model(), _icp_model(), 0, 0, set(), integrity_policy=True))
        with install_exchange(exchange):
            result = asyncio.run(lead_scorer.score_company_competition_intent(_company_model(), _icp_model(), 0, 0, set(), integrity_policy=True))
        assert result.final_score == baseline.final_score == 80
    assert completions == [None, None]


def process_score(mode, count_path, document):
    """Fresh-interpreter driver; provider verifiers are deterministic substitutes."""
    from lab_arena import scorer_entrypoint
    from tests.test_arena_intent_retry_cache import _record_positive, _record_unavailable
    if mode.startswith("before_"):
        mode = mode.removeprefix("before_")
        original = Path(__file__).parents[1] / "fixtures/arena_company_fit_before_checkpoint.py"
        exec(compile(original.read_text(), str(original), "exec"), lead_scorer.__dict__)
    async def fit(*_args, **kwargs):
        if mode == "resume": raise AssertionError("a completed fit was paid twice")
        with Path(count_path).open("a") as stream: stream.write("fit\n")
        kwargs["matched_company_source_sink"].update(snapshot()["matched_company_sources"])
        return native_fit()
    async def signal(value, *_args, evidence_signals, verdict_out, **_kwargs):
        if mode == "intent_fail": return _record_unavailable(verdict_out)
        result = _record_positive(value, evidence_signals, verdict_out)
        for evaluation in verdict_out[-1]["verification_trace"]["intent_verdict"]["signal_evaluations"]:
            evaluation["verification_mode"] = "source_grounded"
        return result
    async def paragraph(*_args, **_kwargs):
        return {"gate": "intent_details", "decision": "unavailable" if mode == "paragraph_fail" else "match"}
    lead_scorer._verify_company_fit = fit
    lead_scorer._score_single_intent_signal = signal
    intent_details.review_intent_details = paragraph
    from arena_reviewer_runtime import checkpoint_io
    from lab_arena import quality_policy
    import os
    import tempfile
    with tempfile.TemporaryDirectory(prefix="native-fit-output-") as directory:
        input_path,output_path=Path(directory)/"input.json",Path(directory)/"output.json"
        input_path.write_text(json.dumps(document))
        os.environ["LAB_ARENA_INPUT_PATH"]=str(input_path)
        os.environ["LAB_ARENA_OUTPUT_PATH"]=str(output_path)
        checkpoint_io.run_scorer(score_input=scorer_entrypoint.score_input,scoring=scoring,quality_policy=quality_policy)
        return json.loads(output_path.read_text())


def _fit_training_values(*, whole_company_hit=False):
    from tests.lab_arena.training_export_signal_test import _signal_fixture
    from lab_arena.training_input import read_input
    values=_signal_fixture(); execute,judge,source=values["runs"]
    cfg=values["round_row"]["configuration_doc"]
    captured=read_input(values["objects"],judge["run_id"],"score",judge["lease_generation"],judge["claim_request_hash"])["input"]
    fields=("schema_version","scored_run_id","icp","companies","scorer_policy","evaluation_date")
    document={key:captured[key] for key in fields}
    origin=judge;origin_index=0;company_rows=[]
    source["scored_run_id"]=execute["run_id"]
    source["assignment_id"]=judge.get("assignment_id")
    source.update(status="failed",terminal_cause="judge_error",output_ref="")
    values["signal_judgment_rows"][0]["source_scored_run_id"]=execute["run_id"]
    if whole_company_hit:
        origin=deepcopy(judge);origin.update(run_id="original-company-judge",scored_run_id="original-execution",assignment_id="original-execution:score")
        original_execute={**execute,"run_id":origin["scored_run_id"],"output_ref":"original-execution-output"}
        values["runs"] += [origin,original_execute]
        original_document=deepcopy(document);original_document["scored_run_id"]=origin["scored_run_id"]
        other=deepcopy(document["companies"][0]);other["company_name"]="Other"
        original_document["companies"]=[other,document["companies"][0]]
        refs=cj.build_company_scopes(scoring_input=document,round_id=judge["round_id"],network_name="finney",netuid=71,scorer_image_digest=cfg["scorer_image_digest"],scorer_image_reference=cfg["scorer_image_reference"],integrity_policy="arena_integrity_v1",company_judgment_policy=cj.POLICY)
        original_refs=cj.build_company_scopes(scoring_input=original_document,round_id=judge["round_id"],network_name="finney",netuid=71,scorer_image_digest=cfg["scorer_image_digest"],scorer_image_reference=cfg["scorer_image_reference"],integrity_policy="arena_integrity_v1",company_judgment_policy=cj.POLICY)
        judge["company_judgment_refs"]=refs;origin["company_judgment_refs"]=original_refs
        judge["claim_response"]={"company_judgment_cache":{"hits":[{**refs[0],"authority_slot":0}],"misses":[]}}
        company_rows=[{"cache_key":refs[0]["cache_key"],"authority_slot":0,"source_score_run_id":origin["run_id"]}]
        source["assignment_id"]=origin["assignment_id"];source["scored_run_id"]=origin["scored_run_id"]
        values["signal_judgment_rows"][0]["source_scored_run_id"]=origin["scored_run_id"]
        values["signal_judgment_uses"][0].update(run_id=origin["run_id"],company_index=1)
        origin_index=1;document_for_fit=original_document;execute_for_fit=original_execute
    else:
        document_for_fit=document;execute_for_fit=execute
    fit=cj.validate_verified_fit_checkpoint(snapshot())
    scope=signal_judgments.build_fit_scope(scoring_input=document_for_fit,company_index=origin_index,
        checkpoint_context=signal_judgments.fit_checkpoint_context(origin,execute_for_fit,cfg,document_for_fit,contracts.document_hash({"schema_version":"leadpoet.lab_arena.output.v1","companies":document_for_fit["companies"]})),
        round_id=judge["round_id"],network_name="finney",netuid=71,scorer_image_digest=cfg["scorer_image_digest"],scorer_image_reference=cfg["scorer_image_reference"])
    row={"cache_key":scope["cache_key"],"authority_slot":0,"signal_input_hash":scope["signal_input_hash"],"scope_doc":scope,"judgment_doc":fit,"judgment_hash":contracts.document_hash(fit),"source_score_run_id":source["run_id"],"source_scored_run_id":source["scored_run_id"],"source_runner_hotkey":source["runner_hotkey"],"source_claim_request_hash":source["claim_request_hash"],"source_lease_generation":source["lease_generation"],"runner_authority_exclusions":[source["runner_hotkey"]]}
    values["signal_judgment_rows"].append(row)
    values["signal_judgment_uses"].append({**{key:row[key] for key in ("cache_key","authority_slot","signal_input_hash","source_score_run_id","judgment_hash")},"run_id":origin["run_id"],"company_index":origin_index,"signal_index":-1})
    saved=json.loads(values["objects"].values[judge["output_ref"]])
    saved["breakdowns"][0]["verifier_gate_receipts"]=[fit["company_fit_receipt"]]
    values["objects"].put(judge["output_ref"],contracts.canonical_json(saved).encode())
    return values,document,document_for_fit,company_rows,saved["breakdowns"]


@pytest.mark.parametrize("whole_company_hit",[False,True])
def test_training_fit_authority_uses_original_execution_after_completed_company_reuse(whole_company_hit):
    from lab_arena.training_export import _signal_authority_evidence
    values,current,original,company_rows,breakdowns=_fit_training_values(whole_company_hit=whole_company_hit)
    judge=values["runs"][1]
    evidence=_signal_authority_evidence(judge=judge,breakdowns=breakdowns,scored_company_indexes=[0],scoring_input=current,
        uses=values["signal_judgment_uses"],authority_rows=values["signal_judgment_rows"],company_rows=company_rows,
        runs={row["run_id"]:row for row in values["runs"]},round_row=values["round_row"],source_input_reader=lambda _:original)
    assert len(evidence)==2
    fit=next(item for item in evidence if item["signal_index"]==-1)
    assert fit["use"]["company_index"]==int(whole_company_hit)
    assert fit["authority"]["scope_doc"]["checkpoint_context"]["scored_run_id"] == original["scored_run_id"]
    assert fit["authority"]["judgment_doc"]["company_fit_receipt"] == snapshot()["company_fit_receipt"]


def test_training_fit_and_signal_reuse_count_original_provider_cost_once():
    from lab_arena.training_export import assemble_example
    from tests.lab_arena.training_export_test import save_input
    values,document,*_=_fit_training_values()
    source=values["runs"][2]
    save_input(values["objects"],source["run_id"],"score",document)
    ledger=values["signal_source_ledger_rows"]
    values["ledger_rows"]=ledger;values["signal_source_ledger_rows"]=[]
    result=assemble_example(**values)
    assert "judgment" in result,result["eligibility"]
    authority=result["judgment"]["signal_authority_evidence"]
    assert len(authority)==2 and {item["signal_index"] for item in authority} == {-1,0}
    costs=result["judgment"]["signal_source_cost_attribution"]
    assert costs["source_score_run_ids"] == ["judge:source"]
    assert result["judgment"]["signal_source_provider_calls"] == []
    assert len(result["provider_calls"]) == 1
    assert costs["confirmed_actual_microusd"] == sum(row["amount_microusd"] for row in ledger if row["entry_kind"]=="settlement")


def test_fit_control_exact_outer_frame_limit(monkeypatch):
    from lab_arena import shim
    sent=[]
    class Socket:
        def __init__(self):
            body=b'{"status":"released"}';self.reply=len(body).to_bytes(4,"big")+body
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def settimeout(self,*_):pass
        def connect(self,*_):pass
        def sendall(self,value):sent.append(value)
        def recv(self,length):
            value,self.reply=self.reply[:length],self.reply[length:];return value
    monkeypatch.setenv(shim.TRUSTED_SCORER_ENV,"1")
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV,"/tmp/unused-test-socket")
    monkeypatch.setattr(signal_cache_control.socket,"socket",lambda *_args,**_kwargs:Socket())
    request={"operation":"complete","company_index":0,"signal_index":-1,"checkpoint_kind":cj.FIT_CHECKPOINT_KIND,"reference":{},"judgment":{"padding":""}}
    size=lambda:len(contracts.canonical_json({"schema_version":signal_cache_control.SCHEMA,"request":request}).encode())
    request["judgment"]["padding"]="x"*(signal_cache_control.MAX_BYTES-size())
    assert size()==signal_cache_control.MAX_BYTES
    assert signal_cache_control._exchange_once(request)["status"]=="released"
    assert len(sent[0])-4==signal_cache_control.MAX_BYTES
    request["judgment"]["padding"]+="x"
    with pytest.raises(SignalJudgmentCacheError,match="frame too large"):
        signal_cache_control._exchange_once(request)
    assert len(sent)==1


def test_training_compares_existing_visible_quote_projection_without_changing_raw_fit():
    from arena_reviewer_runtime.checkpoint_io import normalize_evidence_quotes
    from lab_arena.training_export import _signal_authority_evidence
    values,current,original,company_rows,breakdowns=_fit_training_values()
    row=values["signal_judgment_rows"][-1]
    row["judgment_doc"]["company_fit_receipt"]["dimension_evidence"]["industry"]["web_evidence"]["quote"]="Acme\x00provides software."
    row["judgment_hash"]=contracts.document_hash(row["judgment_doc"])
    values["signal_judgment_uses"][-1]["judgment_hash"]=row["judgment_hash"]
    breakdowns[0]["verifier_gate_receipts"]=[normalize_evidence_quotes(row["judgment_doc"]["company_fit_receipt"])]
    evidence=_signal_authority_evidence(judge=values["runs"][1],breakdowns=breakdowns,scored_company_indexes=[0],scoring_input=current,
        uses=values["signal_judgment_uses"],authority_rows=values["signal_judgment_rows"],company_rows=company_rows,
        runs={run["run_id"]:run for run in values["runs"]},round_row=values["round_row"],source_input_reader=lambda _:original)
    fit=next(item for item in evidence if item["signal_index"]==-1)
    assert "\x00" in fit["authority"]["judgment_doc"]["company_fit_receipt"]["dimension_evidence"]["industry"]["web_evidence"]["quote"]


@pytest.mark.parametrize("failure",["intent_mismatch","intent_unavailable","paragraph_mismatch"])
def test_reused_fit_cannot_autoaccept_later_intent_or_paragraph_failure(monkeypatch,failure):
    async def no_fit(*_args,**_kwargs):pytest.fail("verified fit must not be repeated")
    async def hit(request):
        if request["operation"] == signal_cache_control.FIT_CAPABILITY_OPERATION:
            return signal_cache_control.fit_checkpoint_capability()
        return {"status":"hit","reference":{"authority_slot":0},"judgment":snapshot()}
    async def signal(*_args,**_kwargs):
        if failure=="intent_unavailable":return 0,0,0,0,False,[{"judge_verdict":{"decision":"rejected_verifier_error","pipeline_decision":"unavailable","error_class":"provider_error"}}]
        return (0,0,0,0,True,[]) if failure=="intent_mismatch" else (80,80,1,100,False,[])
    async def paragraph(*_args,**_kwargs):return {"gate":"intent_details","decision":"mismatch"}
    monkeypatch.setattr(lead_scorer,"_verify_company_fit",no_fit)
    monkeypatch.setattr(lead_scorer,"score_company_competition_intent_signal",signal)
    monkeypatch.setattr(intent_details,"review_intent_details",paragraph)
    company=_company_model().model_copy(update={"intent_details":"Acme is hiring."})
    with install_exchange(hit):
        result=asyncio.run(lead_scorer.score_company_competition_intent(company,_icp_model(),0,0,set(),integrity_policy=True))
    assert result.final_score==0
    assert result.verifier_gate_receipts[0]==snapshot()["company_fit_receipt"]


@pytest.mark.parametrize("failure",["forced","duplicate"])
def test_invocation_guards_precede_durable_fit_reuse(monkeypatch,failure):
    async def unexpected(_request):pytest.fail("forced and duplicate guards must precede cache claim")
    with install_exchange(unexpected):
        result=asyncio.run(lead_scorer.score_company_competition_intent(_company_model(),_icp_model(),0,0,{"acme"} if failure=="duplicate" else set(),integrity_policy=True,force_fail_reason="forced" if failure=="forced" else None))
    assert result.final_score==0
    assert ("duplicate" in result.failure_reason.lower()) if failure=="duplicate" else result.failure_reason=="forced"
