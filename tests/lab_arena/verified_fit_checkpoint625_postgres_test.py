"""Real private signal RPC/worker transport survives a fresh judge interpreter."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from lab_arena import company_judgments as cj, contact_policy, contracts, integrity, intent_details_policy, runner, scoring, shim
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, hash_lease_token
from lab_arena.service import ServiceError
from tests.lab_arena import signal_judgments503_postgres_test as prior
from tests.lab_arena import model_policy_supersession517_postgres_test as current
from tests.lab_arena.verified_fit_checkpoint_test import snapshot

SQL = Path(__file__).parents[2] / "scripts/625-lab-arena-verified-fit-checkpoint.sql"


def compose_current_claim(cur):
    """Install621's function-only delta, never its historical recovery DML."""
    # The retained517 harness predates536's nullable recovery receipt column.
    cur.execute("ALTER TABLE public.lab_arena_scoring_supersessions ADD COLUMN IF NOT EXISTS recovery_doc JSONB")
    signature = "public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)"
    cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
    definition = cur.fetchone()[0]
    assert hashlib.sha256(definition.encode()).hexdigest() == "47a767d79cc8a47ad3d653f940e5987ce1ae210701ec41f97e781b498b1ce250"
    old = "  AND j.signal_input_hash=v_hash"
    retired = old + """
  AND NOT EXISTS (SELECT 1 FROM public.lab_arena_scoring_supersessions retired621
    WHERE retired621.round_id='arena-2026-11-04'
      AND retired621.request_id='nov04-baseline-source621-v1'
      AND retired621.recovery_doc->>'operation'='baseline_source_supersession_v1'
      AND retired621.recovery_doc->'superseded_runs' ? j.source_scored_run_id
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'kind'='execute'
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'round_id'='arena-2026-11-04'
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'='baseline-2026-11-04')"""
    assert definition.count(old) == 1
    definition = definition.replace(old, retired)
    assert hashlib.sha256(definition.encode()).hexdigest() == "b2f39b095a1596b35244d9da68ab591b3c50b733d8d1e1b5dcfbcd5668f9ee32"
    cur.execute(definition)


@pytest.fixture(scope="module")
def database():
    fixture = current.database.__wrapped__(); db = next(fixture)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            compose_current_claim(cur)
            cur.execute("CREATE ROLE fit625_migrator NOLOGIN NOSUPERUSER INHERIT")
            cur.execute("GRANT lab_arena_owner TO fit625_migrator")
            cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO fit625_migrator WITH GRANT OPTION")
            cur.execute("SELECT jsonb_agg(to_jsonb(r)) FROM public.lab_arena_signal_judgments r"); before = cur.fetchone()[0]
            cur.execute("SET ROLE fit625_migrator")
            cur.execute(SQL.read_text()); cur.execute(SQL.read_text())
            cur.execute("RESET ROLE")
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)'::regprocedure)")
            claim = cur.fetchone()[0]
            assert hashlib.sha256(claim.encode()).hexdigest() == "5397c48a740fddef5ea7c9e5aa0a319450567407308267c5ebc6ed56bf87b6ef"
            assert claim.count("retired621.request_id='nov04-baseline-source621-v1'") == 1
            cur.execute("SELECT jsonb_agg(to_jsonb(r)) FROM public.lab_arena_signal_judgments r")
            assert cur.fetchone()[0] == before
            for signature in ("public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)", "public.lab_arena_complete_signal_judgment_v1(text,text,integer,integer,jsonb,jsonb,jsonb,text)"):
                cur.execute("SELECT pg_get_userbyid(proowner),proacl::text FROM pg_proc WHERE oid=%s::regprocedure", (signature,))
                assert cur.fetchone() == ("lab_arena_owner", "{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}")
        yield db
    finally: fixture.close()


@pytest.fixture()
def store(database):
    value = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    value._test_database = database
    try: yield value
    finally: value.close()


def setup(store, database, tmp_path, *, paragraph=False, baseline=False):
    rid = "arena-2026-09-11-fit625"
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor SET DEFAULT 0")
    try:
        participants, _ = prior.prior.open_new(store, round_id=rid, markers_by_submission=[["one"]])
    finally:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor DROP DEFAULT")
    config = prior._enable_round(database, rid)
    if baseline:
        prior._make_baseline(database,rid,participants)
    if paragraph:
        policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2", company_judgment_cache=True, intent_details=True)
        policy["signal_judgment_policy"] = "signal_judgment_v1"
        config.update(scorer_policy=policy, intent_details_policy=intent_details_policy.POLICY)
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s", (json.dumps(config),rid))
    # A non-miner judge can continue through a later independent judge process.
    judge = prior.company.hotkey("fit625-original-judge")
    leased, token, _, _ = prior.company.claim(store, rid, judge, excluded=[judge])
    service, _, _ = prior._signal_service(store, leased, tmp_path)
    executed = store.get_run(leased["scored_run_id"])
    output = json.loads(service._objects.get(executed["output_ref"]))
    document = scoring.build_scoring_input(
        scored_run_id=executed["run_id"],
        icp=integrity.agent_visible_icp(prior._icp(), contacts_required=False),
        companies=output["companies"], policy=config["scorer_policy"], evaluation_date="2099-10-20",
    )
    document["company_judgment_cache"] = leased["company_judgment_cache"]
    return service, leased, token, document


def run_process(service, leased, token, document, mode, tmp_path):
    count_path = tmp_path / "fit-calls.txt"
    api = SimpleNamespace(signal_judgment=lambda run, lease, request: service.handle_signal_judgment(run, lease, request))
    state = runner.RunState(lease={**leased, "kind": "score", "scorer_policy": document["scorer_policy"]}, lease_token=token)
    with tempfile.TemporaryDirectory(prefix="fit625-",dir="/tmp") as directory:
        worker = runner.WorkerSocketServer(Path(directory)/"worker.sock", api, state)
        worker.start()
        try:
            env = {**os.environ, shim.WORKER_SOCKET_ENV: str(Path(directory)/"worker.sock")}
            result = subprocess.run([sys.executable, "-c", "import json,sys; from tests.lab_arena.verified_fit_checkpoint_test import process_score; print(json.dumps(process_score(sys.argv[1],sys.argv[2],json.load(sys.stdin))))", mode, str(count_path)], input=json.dumps(document), capture_output=True, text=True, timeout=45, env=env)
            assert result.returncode == 0, result.stderr[-2000:]
            output = json.loads(result.stdout)
        finally: worker.stop()
    assert state.action_sequence == 0 and state.calls == []
    return output


@pytest.mark.parametrize("failure", ["intent_fail", "paragraph_fail"])
def test_fresh_process_keeps_complete_fit_after_later_failure(store, database, tmp_path, failure):
    service, leased, token, document = setup(store,database,tmp_path,paragraph=True)
    first = run_process(service,leased,token,document,failure,tmp_path)
    assert first.get("failure") == "judge_error", first
    assert not first.get("company_judgments")
    uses = store.get_signal_judgment_uses(leased["run_id"])
    fit_use = next(use for use in uses if use["signal_index"] == -1)
    retained = store.get_signal_judgment(fit_use["cache_key"],fit_use["authority_slot"])
    assert retained["judgment_doc"]["company_fit_receipt"] == snapshot()["company_fit_receipt"]
    resumed = run_process(service,leased,token,document,"resume",tmp_path)
    assert not resumed.get("failure"), resumed
    assert resumed["breakdowns"][0]["final_score"] > 0
    assert (tmp_path/"fit-calls.txt").read_text().splitlines() == ["fit"]
    # The final fit receipt is byte-for-byte the original; it does not acquire
    # an autoaccept marker or a new provider/cost receipt during reuse.
    assert resumed["breakdowns"][0]["verifier_gate_receipts"][0] == retained["judgment_doc"]["company_fit_receipt"]
    assert store.get_signal_judgment(fit_use["cache_key"],fit_use["authority_slot"]) == retained


def test_typed_position_and_positive_only_sql_guards(store,database,tmp_path):
    service,leased,token,document = setup(store,database,tmp_path)
    request = {"operation":"claim","company_index":0,"signal_index":-1,"checkpoint_kind":cj.FIT_CHECKPOINT_KIND}
    claimed = []
    original_claim = store.claim_signal_judgment
    def capture_claim(**kwargs):
        claimed.append(kwargs)
        return original_claim(**kwargs)
    store.claim_signal_judgment = capture_claim
    miss = service.handle_signal_judgment(leased["run_id"],token,request)
    args = claimed[0]
    # Service validates full intermediate content before the same existing RPC.
    for mutation in (lambda value:value["company_fit_receipt"].update(decision="unavailable"), lambda value:value["company_fit_receipt"]["company_fit_dimensions"].update(industry="mismatch")):
        bad=snapshot(); mutation(bad)
        with pytest.raises(ServiceError,match="signal_judgment_result_invalid"):
            service.handle_signal_judgment(leased["run_id"],token,{**request,"operation":"complete","reference":miss["reference"],"judgment":bad})
        with pytest.raises(ArenaStoreError,match="fit_checkpoint_judgment_invalid"):
            store.complete_signal_judgment(**args,reference=miss["reference"],judgment=bad,judgment_hash=contracts.document_hash(bad))
    for wrong in ({"signal_index":0},{"checkpoint_kind":"unknown"}):
        with pytest.raises(ServiceError,match="signal_judgment_request_invalid"):
            service.handle_signal_judgment(leased["run_id"],token,{**request,**wrong})
    assert service.handle_signal_judgment(leased["run_id"],token,{**request,"operation":"complete","reference":miss["reference"],"judgment":None})["status"] == "released"
    for position,kind in ((-1,None),(0,cj.FIT_CHECKPOINT_KIND),(-2,cj.FIT_CHECKPOINT_KIND)):
        scope = deepcopy(args["scope"])
        if kind is None: scope.pop("checkpoint_kind")
        with pytest.raises(ArenaStoreError):
            original_claim(**{**args,"signal_index":position,"scope":scope})


def test_later_lease_and_judge_process_reuse_fit_with_original_authority(store,database,tmp_path):
    service,leased,token,document = setup(store,database,tmp_path,paragraph=True)
    source_run = store.get_run(leased["run_id"])
    assert run_process(service,leased,token,document,"intent_fail",tmp_path)["failure"] == "judge_error"
    assert store.complete_attempt(
        run_id=leased["run_id"],lease_token_hash=hash_lease_token(token),
        result={"terminal_status":"judge_error"},terminal_cause="judge_error",output_ref="",
        company_judgment_evidence=[],output_hash="",completion_request_hash="sha256:"+"e"*64,
    )["status"] == "failed"
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND kind='score' AND status='pending'",(source_run["round_id"],))
    next_judge=prior.company.hotkey("fit625-next-judge")
    following,next_token,_,_=prior.company.claim(store,source_run["round_id"],next_judge,excluded=[next_judge])
    assert following["run_id"] != leased["run_id"] and following["status"] == "leased"
    following_document={**document,"company_judgment_cache":following["company_judgment_cache"]}
    result=run_process(service,following,next_token,following_document,"resume",tmp_path)
    assert "failure" not in result, result
    assert (tmp_path/"fit-calls.txt").read_text().splitlines() == ["fit"]
    use=next(use for use in store.get_signal_judgment_uses(following["run_id"]) if use["signal_index"] == -1)
    assert use["source_score_run_id"] == leased["run_id"]
    row=store.get_signal_judgment(use["cache_key"],use["authority_slot"])
    assert row["source_runner_hotkey"] == store.get_run(leased["run_id"])["runner_hotkey"]
    assert row["source_runner_hotkey"] in store.signal_authority_exclusions(following["run_id"])


@pytest.mark.parametrize("baseline",[False,True])
def test_original_failure_costs_two_fits_before_one_after_with_exact_output_parity(store,database,tmp_path,baseline):
    results=[]
    for before in (True,False):
        case=tmp_path/("before" if before else "after");case.mkdir()
        service,leased,token,document=setup(store,database,case,paragraph=True,baseline=baseline)
        failure_mode="before_intent_fail" if before else "intent_fail"
        healthy_mode="before_healthy" if before else "resume"
        assert run_process(service,leased,token,document,failure_mode,case)["failure"] == "judge_error"
        results.append(run_process(service,leased,token,document,healthy_mode,case))
        assert "failure" not in results[-1]
        assert (case/"fit-calls.txt").read_text().splitlines() == ["fit"]*(2 if before else 1)
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s",(leased["run_id"],))
            assert cur.fetchone()[0] == 0
    assert results[0] == results[1]


@pytest.mark.parametrize("corrupt",["judgment_hash","claim_hash","generation","runtime"])
def test_persisted_fit_rejects_corrupt_or_changed_source_authority(store,database,tmp_path,corrupt):
    service,leased,token,document=setup(store,database,tmp_path,paragraph=True)
    assert run_process(service,leased,token,document,"intent_fail",tmp_path)["failure"]=="judge_error"
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        if corrupt=="judgment_hash":cur.execute("UPDATE public.lab_arena_signal_judgments SET judgment_hash=%s",("sha256:"+"0"*64,))
        elif corrupt=="claim_hash":cur.execute("UPDATE public.lab_arena_runs SET claim_request_hash=%s WHERE run_id=%s",("sha256:"+"0"*64,leased["run_id"]))
        elif corrupt=="generation":cur.execute("UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s",(leased["run_id"],))
        else:cur.execute("UPDATE public.lab_arena_runs SET reviewer_runtime_doc='{}'::jsonb WHERE run_id=%s",(leased["run_id"],))
    request={"operation":"claim","company_index":0,"signal_index":-1,"checkpoint_kind":cj.FIT_CHECKPOINT_KIND}
    with pytest.raises(ServiceError,match="signal_judgment_(cache|input)_invalid"):
        service.handle_signal_judgment(leased["run_id"],token,request)
    assert (tmp_path/"fit-calls.txt").read_text().splitlines()==["fit"]


@pytest.mark.parametrize("stale",["expired","generation"])
def test_fit_completion_keeps_current_lease_and_reservation_proof(store,database,tmp_path,stale):
    service,leased,token,document=setup(store,database,tmp_path)
    request={"operation":"claim","company_index":0,"signal_index":-1,"checkpoint_kind":cj.FIT_CHECKPOINT_KIND}
    args=[];original=store.claim_signal_judgment
    def capture(**value):args.append(value);return original(**value)
    store.claim_signal_judgment=capture
    miss=service.handle_signal_judgment(leased["run_id"],token,request)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        if stale=="expired":cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(leased["run_id"],))
        else:cur.execute("UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s",(leased["run_id"],))
    with pytest.raises(ArenaStoreError,match="signal_(completion_invalid|reservation_stale)"):
        store.complete_signal_judgment(**args[0],reference=miss["reference"],judgment=snapshot(),judgment_hash=contracts.document_hash(snapshot()))
    assert store.get_signal_judgment_uses(leased["run_id"])==[]


@pytest.mark.parametrize("baseline",[False,True])
def test_old_worker_validator_keeps_new_scorer_native_output_without_fit_reuse(store,database,tmp_path,monkeypatch,baseline):
    from lab_arena import signal_cache_control
    from tests.lab_arena.fit_checkpoint_host_compatibility_test import old_validator
    service,leased,token,document=setup(store,database,tmp_path,paragraph=True,baseline=baseline)
    before=run_process(service,leased,token,document,"before_healthy",tmp_path)
    assert "failure" not in before
    # Only the worker process uses old validation. The fresh scorer interpreter
    # retains the new sender and capability logic, as on an external host.
    monkeypatch.setattr(signal_cache_control,"validate_request",old_validator())
    current=run_process(service,leased,token,document,"healthy",tmp_path)
    assert current==before
    assert (tmp_path/"fit-calls.txt").read_text().splitlines()==["fit","fit"]
    assert all(use["signal_index"]>=0 for use in store.get_signal_judgment_uses(leased["run_id"]))


def test_supported_new_worker_translated_gateway_failure_does_not_repeat_fit(store,database,tmp_path):
    from lab_arena.runner import RunnerError
    service,leased,token,document=setup(store,database,tmp_path,paragraph=True)
    calls=[]
    def rejected(_run,_token,request):
        calls.append(request)
        raise RunnerError("accepted checkpoint source is corrupt")
    service.handle_signal_judgment=rejected
    result=run_process(service,leased,token,document,"healthy",tmp_path)
    assert result["failure"]=="judge_error"
    assert not (tmp_path/"fit-calls.txt").exists()
    assert calls and all(request["operation"]=="claim" and request["signal_index"]==-1 for request in calls)
