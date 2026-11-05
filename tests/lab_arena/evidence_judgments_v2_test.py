"""Evidence identities separate current source bytes from submitted claims."""
from copy import deepcopy
import pytest
from lab_arena import company_judgments as cj, contracts, evidence_judgments as ej, signal_cache_control
from tests.lab_arena import legacy_company_judgments_test as legacy
from tests.lab_arena.verified_fit_checkpoint_test import snapshot


def document():
    doc=legacy._input(("Acme",))
    doc["scorer_policy"].update(signal_judgment_policy="signal_judgment_v1",evidence_judgment_policy=ej.POLICY)
    doc["scorer_policy"]["env_bindings"]["ARENA_EVIDENCE_JUDGMENT_POLICY"]=ej.POLICY
    return doc


def scope(doc=None, *, fit=True, evidence=None, **kwargs):
    doc=doc or document()
    if not fit and evidence is not None:
        signal=cj.effective_company_inputs(doc)[0]["company"]["intent_signals"][0]
        evidence={"schema_version":"leadpoet.intent_evidence_input.v1", "evaluation_date":doc["evaluation_date"],
            "model":doc["scorer_policy"]["judge_models"]["intent_fetched_evidence"],
            "row":{"signal_date":signal["date"],"claimed_source_urls":[signal["url"]]},**evidence}
    return ej.build_scope(scoring_input=doc,company_index=0,signal_index=-1 if fit else 0,
        round_id="arena-2026-10-20",network_name="finney",netuid=71,
        scorer_image_digest="sha256:"+"a"*64,scorer_image_reference="registry/scorer@sha256:"+"a"*64,
        runtime_image_digest="sha256:"+"a"*64,runtime_image_reference="registry/scorer@sha256:"+"a"*64,
        verified_company_identity={"decision":"match","observed_domain":"acme.example"},
        verified_funding_stage_context=[], evidence_input=evidence,
        evidence_hash=contracts.document_hash(evidence) if evidence is not None else None,**kwargs)


def fit_snapshot(manifest=None, *, negative=False):
    value=snapshot()
    value.update(schema_version=ej.FIT_SCHEMA,evidence_manifest=manifest or {"pages":{"https://acme.example/about":{"text":"Acme provides software.","status":"fetched"}}})
    value["source_snapshot_hash"]=contracts.document_hash(value["evidence_manifest"])
    if negative:
        receipt=value["company_fit_receipt"]
        receipt.update(decision="mismatch",company_fit_decision="mismatch")
        receipt["company_fit_dimensions"]["industry"]="mismatch"
        receipt["dimension_evidence"]["industry"]["decision"]="mismatch"
    return value


def test_fit_identity_ignores_execution_and_intent_prose_but_preserves_fit_conflicts():
    a=document();b=deepcopy(a);b["scored_run_id"]="different-execution"
    b["companies"][0]["intent_signals"][0].update(description="Different sales prose",snippet="Different snippet")
    assert scope(a)==scope(b)
    for field, value in (("company_name","Other"),("employee_count",999),("industry","Mining"),("required_attribute",{"text":"Other fact","passed":True,"evidence_url":"https://acme.example/proof","evidence_quote":"Fact","explanation":"Other source fact"})):
        changed=deepcopy(a);changed["companies"][0][field]=value
        assert scope(changed)["family_hash"]!=scope(a)["family_hash"]
    b=deepcopy(a);b["companies"][0]["intent_signals"][0]["url"]+="/changed"
    assert scope(a)["family_hash"]!=scope(b)["family_hash"]


def test_whole_company_v2_is_execution_bound_and_v1_bytes_unchanged():
    a=legacy._input();b=deepcopy(a);b["scored_run_id"]="another-execution"
    assert legacy._refs(a)==legacy._refs(b)
    a=document();b=deepcopy(a);b["scored_run_id"]="another-execution"
    assert legacy._refs(a)[0]["cache_key"]!=legacy._refs(b)[0]["cache_key"]
    assert scope(a)==scope(b)


def test_signal_family_preserves_source_and_buyer_but_ignores_claim_words():
    evidence={"source":{"text":"Original fetched text","http_status":200}}
    a=document();b=deepcopy(a);b["companies"][0]["intent_signals"][0]["description"]="Equivalent wording"
    first,second=scope(a,fit=False,evidence=evidence),scope(b,fit=False,evidence=evidence)
    assert first["family_hash"]==second["family_hash"]
    assert first["cache_key"]!=second["cache_key"]
    altered=scope(a,fit=False,evidence={"source":{"text":"Changed text","http_status":200}})
    assert altered["cache_key"]!=first["cache_key"]
    with pytest.raises(ej.Error,match="snapshot is required"):
        scope(a,fit=False)
    assert first["claim"]==ej.normalize_claim(first["family_doc"]["native_input"]["signal"] | {"description":a["companies"][0]["intent_signals"][0]["description"],"snippet":a["companies"][0]["intent_signals"][0]["snippet"]})


def test_bound_equivalence_receipt_and_exact_claim_equality():
    a={"description":"Acme launched a product"};b={"description":"Acme introduced a product"}
    policy=document()["scorer_policy"]
    receipt={"schema_version":ej.EQUIVALENCE_SCHEMA,"source_claim_hash":contracts.document_hash(a),
        "current_claim_hash":contracts.document_hash(b),"decision":"equivalent","material_differences":[],
        "model":policy["judge_models"]["intent_fetched_evidence"]}
    assert ej.validate_equivalence(None,a,a,policy) is None
    assert ej.validate_equivalence(receipt,a,b,policy)==receipt
    for field,wrong in (("current_claim_hash",contracts.document_hash(a)),("model","unapproved"),("material_differences",["date"])):
        with pytest.raises(ej.Error):ej.validate_equivalence({**receipt,field:wrong},a,b,policy)
    assert ej.normalize_claim({"description":" A  B ","snippet":" x "})["description"]=="A  B"


@pytest.mark.parametrize("negative",[False,True])
def test_native_positive_and_negative_fit_are_retained_with_complete_manifest(negative):
    value=fit_snapshot(negative=negative)
    current=scope(evidence=value["evidence_manifest"])
    assert ej.validate_judgment(value,current)==value
    value["company_fit_receipt"]["decision"]="unavailable"
    with pytest.raises(ej.Error):ej.validate_judgment(value,current)


def test_fit_requires_current_revalidated_snapshot_and_wire_bounds():
    value=fit_snapshot();current=scope(evidence=value["evidence_manifest"])
    value["evidence_manifest"]["pages"]["https://acme.example/about"]["text"]="Changed content"
    with pytest.raises(ej.Error):ej.validate_judgment(value,current)
    request={"operation":"candidates","company_index":0,"signal_index":-1,"checkpoint_kind":ej.FIT_KIND}
    assert signal_cache_control.validate_request(request)==request
    with pytest.raises(ValueError):signal_cache_control.validate_request({**request,"operation":"claim"})
    with pytest.raises(ValueError):signal_cache_control.validate_request({**request,"signal_index":0})


def test_fit_projection_preserves_native_intent_details_presence_policy():
    original={"company":{"intent_signals":[],"intent_details":None}}
    supplied=deepcopy(original);supplied["company"]["intent_details"]="private prose"
    empty=deepcopy(original);empty["company"]["intent_details"]=""
    assert ej.fit_projection(original)!=ej.fit_projection(supplied)
    assert ej.fit_projection(supplied)==ej.fit_projection(empty)


def test_v2_bulk_reader_includes_all_five_fit_slots_and_rejects_invalid_positions():
    from lab_arena.store import ArenaStore, ArenaStoreError
    from tests.lab_arena.public_result_authority_batch_test import Transport
    rows=[{"run_id":"score","company_index":company,"signal_index":signal,"acceptance_doc":None}
          for company in range(5) for signal in range(-1,100)]
    transport=Transport({"lab_arena_signal_judgment_uses":rows})
    store=ArenaStore(transport)
    assert len(store.get_signal_judgment_uses_batch(["score"],evidence=True)["score"])==505
    assert all("acceptance_doc" in call[1]["columns"] for call in transport.calls)
    transport.tables["lab_arena_signal_judgment_uses"][0]["signal_index"]=-2
    with pytest.raises(ArenaStoreError):store.get_signal_judgment_uses_batch(["score"],evidence=True)


def test_signal_envelope_retains_full_native_result_and_rejects_unbound_native_payload():
    from tests.test_intent_evidence_terminal_admission import _inputs
    raw,evidence,expected=_inputs(positive=True)
    trace=raw["judge_verdict"]["verification_trace"]
    trace["stage1"]={"status":"skipped_source_grounded"}
    trace["stage3"]={"status":"supported","decision":"approve"}
    native={"decision":"approve","client_ready":True,"stage1":deepcopy(trace["stage1"]),
            "stage3":deepcopy(trace["stage3"]),"verdict":deepcopy(trace["intent_verdict"])}
    claim={"description":"Acme introduced a product"}
    value={"schema_version":ej.SIGNAL_SCHEMA,"raw_judgment":raw,"verifier_result":native,
           "claim":claim,"evidence_input":evidence}
    current={**expected,"claim":claim,"evidence_input":evidence}
    assert ej.validate_judgment(value,current)==value
    for changed in ({}, {**native,"decision":"reject"}, {**native,"verdict":{"signal_evaluations":[]}}):
        with pytest.raises(ej.Error):ej.validate_judgment({**value,"verifier_result":changed},current)
