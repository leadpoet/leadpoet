"""Real private PostgreSQL authority, leases and current-evidence reuse."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading
import pytest
from lab_arena import contracts, evidence_judgments as ej, signal_judgments
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, hash_lease_token
from tests.lab_arena import verified_fit_checkpoint625_postgres_test as base
from tests.lab_arena import signal_judgments503_postgres_test as prior
from tests.lab_arena.evidence_judgments_v2_test import fit_snapshot

SQL=Path(__file__).parents[2]/"scripts/631-lab-arena-evidence-judgments-v2.sql"


@pytest.fixture(scope="module")
def database():
    generator=base.database.__wrapped__();db=next(generator)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute("CREATE ROLE evidence631_migrator NOLOGIN NOSUPERUSER INHERIT")
            cur.execute("GRANT lab_arena_owner TO evidence631_migrator")
            cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO evidence631_migrator WITH GRANT OPTION")
            cur.execute("ALTER DEFAULT PRIVILEGES FOR ROLE evidence631_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role")
            cur.execute("SET ROLE evidence631_migrator")
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
            cur.execute("RESET ROLE")
        yield db
    finally:generator.close()


@pytest.fixture()
def store(database):
    value=ArenaStore(PsycopgTransport(lambda:database[0].connect(**database[1])))
    value._test_database=database
    try:yield value
    finally:value.close()


def setup(store,database,tmp_path,*,two=False,baseline=False):
    rid="arena-2026-09-11-evidence631"
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor SET DEFAULT 0")
    try:participants,_=prior.prior.open_new(store,round_id=rid,markers_by_submission=[["one"],["two"]] if two else [["one"]])
    finally:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor DROP DEFAULT")
    config=prior._enable_round(database,rid)
    config["scorer_policy"].update(evidence_judgment_policy=ej.POLICY)
    config["scorer_policy"]["env_bindings"]["ARENA_EVIDENCE_JUDGMENT_POLICY"]=ej.POLICY
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s",(json.dumps(config),rid))
    if baseline:prior._make_baseline(database,rid,participants)
    def lease(label):
        judge=prior.company.hotkey(label)
        leased,token,_,_=prior.company.claim(store,rid,judge,excluded=[judge])
        service,request,raw=prior._signal_service(store,leased,tmp_path)
        return service,leased,token,request,raw
    return lease("evidence631-source"),lease if two else None


def call(case,request):
    service,leased,token,*_=case
    return service.handle_signal_judgment(leased["run_id"],token,request)


def fit_request(manifest=None,operation="claim"):
    value={"operation":operation,"company_index":0,"signal_index":-1,"checkpoint_kind":ej.FIT_KIND}
    if manifest is not None:value.update(evidence_input=manifest,evidence_hash=contracts.document_hash(manifest))
    return value


def signal_request(case,*,operation="claim",snapshot=None):
    raw=case[4]
    trace=raw["judge_verdict"]["verification_trace"]
    store=case[0]._store
    run=store.get_run(case[1]["run_id"])
    policy=store.get_round(run["round_id"])["configuration_doc"]["scorer_policy"]
    snapshot=snapshot or {"schema_version":"leadpoet.intent_evidence_input.v1",
        "model":policy["judge_models"]["intent_fetched_evidence"],"evaluation_date":"2099-10-20",
        "row":{"claimed_source_urls":raw["evidence_urls"],"signal_date":trace["extracted_signal_date"]},
        "contents":{"results":deepcopy(trace["verified_source_context"]),"statuses":[]}}
    return {**case[3],"operation":operation,"judgment_kind":ej.SIGNAL_KIND,
            "evidence_input":snapshot,"evidence_hash":contracts.document_hash(snapshot)}


def store_fit(case,negative=False):
    value=fit_snapshot(negative=negative);request=fit_request(value["evidence_manifest"])
    miss=call(case,request);assert miss["status"]=="miss"
    assert call(case,{**request,"operation":"complete","reference":miss["reference"],"judgment":value})["status"]=="stored"
    return request,miss,value


def test_security_shape_and_private_index(database):
    signature="public.lab_arena_evidence_judgment_v2(text,text,text,integer,integer,jsonb,jsonb,jsonb)"
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT pg_get_userbyid(proowner),proacl::text,prosecdef,proconfig FROM pg_proc WHERE oid=%s::regprocedure",(signature,))
        assert cur.fetchone()==("lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",True,["search_path=pg_catalog, public"])
        cur.execute("SELECT indexdef FROM pg_indexes WHERE indexname='lab_arena_signal_judgments_evidence_family_v2'")
        assert "family_hash" in cur.fetchone()[0]
        for role in ("anon","authenticated","service_role"):
            cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')",(role,signature));assert cur.fetchone()==(False,)
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)'::regprocedure)")
        assert "retired621.request_id='nov04-baseline-source621-v1'" in cur.fetchone()[0]


@pytest.mark.parametrize("negative",[False,True])
@pytest.mark.parametrize("baseline",[False,True])
def test_fit_reuses_across_executions_after_current_snapshot_and_retains_original_authority(store,database,tmp_path,negative,baseline):
    source,lease=setup(store,database,tmp_path,two=True,baseline=baseline)
    request,miss,value=store_fit(source,negative)
    target=lease("evidence631-target")
    # Candidate discovery is read-only; no authority use exists on target.
    discovered=call(target,fit_request(operation="candidates"))
    assert len(discovered["candidates"])==1
    assert store.get_signal_judgment_uses(target[1]["run_id"],evidence=True)==[]
    with pytest.raises(ServiceError):call(target,fit_request())
    hit=call(target,{**request,"candidate_reference":miss["reference"]})
    assert hit["status"]=="hit" and hit["judgment"]==value
    use=store.get_signal_judgment_uses(target[1]["run_id"],evidence=True)[0]
    assert use["source_score_run_id"]==source[1]["run_id"]
    assert use["acceptance_doc"]["current_scope"]==hit["current_scope"]
    assert use["acceptance_doc"]["candidate_reference"]==miss["reference"]
    assert store.get_run(source[1]["run_id"])["runner_hotkey"] in store.signal_authority_exclusions(target[1]["run_id"])
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.lab_arena_ledger");assert cur.fetchone()==(0,)
        cur.execute("SELECT count(*) FROM public.lab_arena_signal_judgments");assert cur.fetchone()==(1,)


def test_same_claim_does_not_skip_current_evidence_hash_or_source_revalidation(store,database,tmp_path):
    source,lease=setup(store,database,tmp_path,two=True)
    request,miss,_=store_fit(source)
    target=lease("evidence631-target")
    changed={"pages":{"https://acme.example/about":{"text":"Now a different entity","status":"fetched"}}}
    with pytest.raises(ServiceError):call(target,{**fit_request(changed),"candidate_reference":miss["reference"]})
    assert call(target,fit_request(changed))["status"]=="miss"
    assert store.get_signal_judgment_uses(target[1]["run_id"] ,evidence=True)==[]
    with pytest.raises(ServiceError):call(target,{**request,"evidence_hash":"sha256:"+"0"*64})


@pytest.mark.parametrize("negative",[False,True])
def test_signal_changed_claim_requires_bound_equivalence_and_keeps_source_envelope(store,database,tmp_path,negative):
    source,lease=setup(store,database,tmp_path,two=True)
    request=signal_request(source);miss=call(source,request);scope=miss["current_scope"]
    raw=deepcopy(source[4]);trace=raw["judge_verdict"]["verification_trace"]
    trace["intent_verdict"]["signal_evaluations"][0]["signal_id"]="signal-1"
    trace["stage1"]={"status":"skipped_source_grounded"}
    trace["stage3"]={"status":"supported","decision":"approve"}
    if negative:
        raw.update(raw=0.0,after_decay=0.0,decay=0.0,confidence=0,date_status="verified",date_verdict="not_evaluated",claim_support_verdict="contradicted")
        raw["judge_verdict"].update(decision="rejected_three_stage",pipeline_decision="reject",client_ready=False,claim_support_verdict="contradicted")
        evaluation=trace["intent_verdict"]["signal_evaluations"][0]
        evaluation.update(signal_status="contradicted",contradicting_quotes=evaluation.pop("supporting_quotes"),supporting_quotes=[])
        trace.update(final_disposition="reject",stage3={"status":"contradicted","decision":"reject"})
    native={"decision":"reject" if negative else "approve","client_ready":not negative,"stage1":trace["stage1"],"stage3":trace["stage3"],"verdict":trace["intent_verdict"]}
    envelope={"schema_version":ej.SIGNAL_SCHEMA,"raw_judgment":raw,"verifier_result":native,"evidence_input":request["evidence_input"],"claim":scope["claim"]}
    assert call(source,{**request,"operation":"complete","reference":miss["reference"],"judgment":envelope})["status"]=="stored"
    target=lease("evidence631-target")
    service=target[0];executed=store.get_run(target[1]["scored_run_id"])
    output=json.loads(service._objects.get(executed["output_ref"]))
    output["companies"][0]["intent_signals"][0]["description"]="Equivalent product introduction wording"
    variant_ref=executed["output_ref"]+"-variant"
    service._objects.put(variant_ref,json.dumps(output).encode())
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET output_ref=%s WHERE run_id=%s",(variant_ref,executed["run_id"]))
    target_request=signal_request(target)
    candidates=call(target,{**target_request,"operation":"candidates"})["candidates"]
    assert len(candidates)==1 and store.get_signal_judgment_uses(target[1]["run_id"],evidence=True)==[]
    claim_request={**target_request,"candidate_reference":miss["reference"]}
    with pytest.raises(ServiceError):call(target,claim_request)
    target_scope=call(target,target_request)["current_scope"]
    receipt={"schema_version":ej.EQUIVALENCE_SCHEMA,"source_claim_hash":contracts.document_hash(scope["claim"]),"current_claim_hash":contracts.document_hash(target_scope["claim"]),"decision":"equivalent","material_differences":[],"model":scope["family_doc"]["scorer_policy"]["judge_models"]["intent_fetched_evidence"]}
    for key,value in (("model","wrong"),("current_claim_hash",contracts.document_hash(scope["claim"])),("material_differences",["buyer"])):
        with pytest.raises(ServiceError):call(target,{**claim_request,"equivalence_receipt":{**receipt,key:value}})
    hit=call(target,{**claim_request,"equivalence_receipt":receipt})
    assert hit["judgment"]==envelope
    proof=store.get_signal_judgment_uses(target[1]["run_id"],evidence=True)[0]["acceptance_doc"]
    assert proof["equivalence_receipt"]==receipt and proof["current_scope"]["claim"]==target_scope["claim"]


@pytest.mark.parametrize("corrupt",["judgment_hash","claim_request_hash","lease_generation","runtime"])
def test_candidate_source_authority_corruption_is_rejected(store,database,tmp_path,corrupt):
    source,lease=setup(store,database,tmp_path,two=True)
    request,miss,_=store_fit(source);target=lease("evidence631-target")
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        if corrupt=="judgment_hash":cur.execute("UPDATE public.lab_arena_signal_judgments SET judgment_hash=%s",("sha256:"+"0"*64,))
        elif corrupt=="runtime":cur.execute("UPDATE public.lab_arena_runs SET reviewer_runtime_doc='{}'::jsonb WHERE run_id=%s",(source[1]["run_id"],))
        elif corrupt=="lease_generation":cur.execute("UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s",(source[1]["run_id"],))
        else:cur.execute("UPDATE public.lab_arena_runs SET claim_request_hash=%s WHERE run_id=%s",("sha256:"+"0"*64,source[1]["run_id"]))
    if corrupt=="runtime":
        # Default reviewer_runtime_doc {} can already select the same image;
        # set an explicit different runtime to prove the intended gate.
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_runs SET reviewer_runtime_doc=%s::jsonb WHERE run_id=%s",(json.dumps({"runtime_image_digest":"sha256:"+"0"*64,"runtime_image_reference":"other@sha256:"+"0"*64}),source[1]["run_id"]))
    with pytest.raises((ServiceError,ValueError)):call(target,{**request,"candidate_reference":miss["reference"]})


@pytest.mark.parametrize("failure",["expired","generation","excluded"])
def test_sql_candidate_acceptance_preserves_lease_and_exclusion_guards(store,database,tmp_path,failure):
    source,lease=setup(store,database,tmp_path,two=True)
    request,miss,_=store_fit(source);target=lease("evidence631-target")
    scope=call(source,request)["current_scope"]
    args=dict(operation="claim",run_id=target[1]["run_id"],lease_token_hash=hash_lease_token(target[2]),company_index=0,signal_index=-1,scope=scope,candidate_reference=miss["reference"])
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        if failure=="expired":cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(target[1]["run_id"],))
        elif failure=="generation":cur.execute("UPDATE public.lab_arena_runs SET lease_token_hash=%s WHERE run_id=%s",("sha256:"+"0"*64,target[1]["run_id"]))
        else:cur.execute("UPDATE public.lab_arena_signal_judgments SET runner_authority_exclusions=runner_authority_exclusions||%s::jsonb",(json.dumps([store.get_run(target[1]["run_id"])["miner_hotkey"]]),))
    with pytest.raises(ArenaStoreError):store.evidence_judgment(**args)
    assert store.get_signal_judgment_uses(target[1]["run_id"],evidence=True)==[]


def test_concurrent_exact_reservation_has_one_owner_and_no_duplicate_use(store,database,tmp_path):
    source,lease=setup(store,database,tmp_path,two=True);target=lease("evidence631-target")
    value=fit_snapshot();request=fit_request(value["evidence_manifest"])
    barrier=threading.Barrier(2)
    def claim(case):barrier.wait();return call(case,request)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(claim,[source,target]))
    assert sorted(item["status"] for item in results)==["busy","miss"]
    winner=source if results[0]["status"]=="miss" else target
    miss=next(item for item in results if item["status"]=="miss")
    assert call(winner,{**request,"operation":"complete","reference":miss["reference"],"judgment":value})["status"]=="stored"
    loser=target if winner is source else source
    assert call(loser,request)["status"]=="hit"
    assert len(store.get_signal_judgment_uses(winner[1]["run_id"],evidence=True))==1
    assert len(store.get_signal_judgment_uses(loser[1]["run_id"],evidence=True))==1


def test_original_exact_completion_can_be_reclaimed_via_candidates_without_new_use(store,database,tmp_path):
    case,_=setup(store,database,tmp_path)
    request,miss,value=store_fit(case)
    discovered=call(case,fit_request(operation="candidates"))["candidates"]
    assert discovered[0]["reference"]==miss["reference"]
    assert call(case,{**request,"candidate_reference":miss["reference"]})["judgment"]==value
    assert call(case,request)["judgment"]==value
    uses=store.get_signal_judgment_uses(case[1]["run_id"],evidence=True)
    assert len(uses)==1 and uses[0]["acceptance_doc"] is None


@pytest.mark.parametrize("invalid",["unavailable","missing_decision","native_mismatch_missing","positive_partial"])
def test_direct_sql_fit_admission_rejects_incomplete_native_evidence(store,database,tmp_path,invalid):
    case,_=setup(store,database,tmp_path)
    value=fit_snapshot();request=fit_request(value["evidence_manifest"])
    miss=call(case,request);scope=miss["current_scope"]
    if invalid=="unavailable":value["company_fit_receipt"].update(decision="unavailable",company_fit_decision="unavailable")
    elif invalid=="missing_decision":value["company_fit_receipt"].pop("decision")
    elif invalid=="native_mismatch_missing":value["company_fit_receipt"].update(decision="mismatch",company_fit_decision="mismatch")
    else:value["company_fit_receipt"]["company_fit_dimensions"]["industry"]="unavailable"
    with pytest.raises(ArenaStoreError,match="evidence_judgment_invalid"):
        store.complete_signal_judgment(run_id=case[1]["run_id"],lease_token_hash=hash_lease_token(case[2]),
            company_index=0,signal_index=-1,scope=scope,reference=miss["reference"],judgment=value,judgment_hash=contracts.document_hash(value))
    assert store.get_signal_judgment_uses(case[1]["run_id"],evidence=True)==[]
