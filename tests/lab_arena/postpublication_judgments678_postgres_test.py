"""General finite postpublication judging retains sealed reward authority."""
from tests.lab_arena.historical_slot_policy_fixture import historical_slot_policy  # noqa: F401
import copy,json,re,hashlib
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from tests.lab_arena import finite_judge_compatible_runtime677_postgres_test as f
old=f.old
SQL=Path(__file__).parents[2]/'scripts/678-lab-arena-audited-postpublication-judgments.sql'
database=f.database
prepared=f.prepared
prepared_company=f.prepared_company
case=f.case
base_extended=f.base_extended
extended=f.extended
provider_case=f.provider_case
correctable=f.correctable
corrected=f.corrected
finite=f.finite
compatible=f.compatible
SIG='lab_arena_recover_published_judgments_v1(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)'

@pytest.fixture
def before(compatible):
 p=compatible
 with p.c.cursor()as q:
  q.execute('SET LOCAL session_replication_role=replica')
  q.execute("UPDATE lab_arena_rounds SET rewards_enabled=TRUE,configuration_doc=jsonb_set(jsonb_set(jsonb_set(configuration_doc,'{cost_per_company_microusd}',to_jsonb(10000000)),'{schedule,final_scoring_close}',to_jsonb(to_char((clock_timestamp()-interval '2 minutes')AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'))),'{schedule,publication_deadline}',to_jsonb(to_char((clock_timestamp()-interval '1 minute')AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'))),status='published',published_at=clock_timestamp(),publication_doc=%s::jsonb,king_outcome='no_king',reward_activated_at=clock_timestamp(),effective_reward_epoch=1,reward_basis_doc=%s::jsonb,reward_basis_hash=%s,signing_key_doc='{}'::jsonb WHERE round_id=%s",(json.dumps({'king_decision':{'outcome':'no_king','king_hotkey':'','king_submission_id':None,'winner_submission_id':None}}),json.dumps({'fixture':'sealed-no-king'}),'sha256:'+'a'*64,old.old.ROUND))
  q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{rewards_enabled}','true'::jsonb)WHERE round_id=%s",(old.old.ROUND,))
  q.execute('SET LOCAL session_replication_role=origin')
  p.round_hash='sha256:'+old.old.scalar(q,"SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex')FROM lab_arena_rounds r WHERE round_id=%s",(old.old.ROUND,))

 p.c.commit();return p

@pytest.fixture
def post(before):
 p=before
 freshness=SQL.with_name('627-lab-arena-champion-baseline-freshness.sql').read_text()
 patches=re.findall(r"\('([^']+)','([^']+)','([^']+)',\d+,\$old627\$(.*?)\$old627\$,\$new627\$(.*?)\$new627\$\)",freshness,re.S)
 canonical=['public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)',
  'public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)',
  'public.lab_arena_freeze_champion_funding(text)']
 with p.c.cursor()as q:
  for sig in canonical[1:]:
   changes=[x for x in patches if x[0]==sig]
   definition=old.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))
   if hashlib.sha256(definition.encode()).hexdigest()!=changes[0][2]:
    assert hashlib.sha256(definition.encode()).hexdigest()==changes[0][1]
    for _,_,_,a,b in changes:
     assert definition.count(a)==1;definition=definition.replace(a,b)
    assert hashlib.sha256(definition.encode()).hexdigest()==changes[0][2];q.execute(definition)
  q.execute(SQL.with_name('640-lab-arena-prior-reward-baseline-gate.sql').read_text())
  p.originals=[old.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(s,))for s in [f.COMMON,old.prior.provider.HELPERS[2]]+canonical]
  q.execute(SQL.read_text())
 p.c.commit()
 try:yield p
 finally:
  p.c.rollback()
  with p.c.cursor()as q:
   q.execute('DROP TRIGGER IF EXISTS lab_arena_postpublication_reward_seal ON lab_arena_rounds')
   for d in p.originals:q.execute(d)
   for s in ['lab_arena_postpublication_reward_seal_v1()','lab_arena_recover_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)','lab_arena__postpublication_judge_v1(public.lab_arena_rounds,public.lab_arena_runs)','lab_arena__postpublication_receipt_v1(public.lab_arena_scoring_supersessions)']:q.execute('DROP FUNCTION public.'+s)
  p.c.commit()

def args(p,**changes):
 now=datetime.now(timezone.utc).replace(microsecond=0)
 values=dict(round=old.old.ROUND,sha=p.round_hash,runtime=old.prior.RUNTIME,final=(now+timedelta(hours=2)).isoformat(),publication=(now+timedelta(hours=3)).isoformat(),requests=[dict(parent_run_id=p.parent['run_id'],parent_sha256=p.parent_hash,request_id='postpub-finite-fixture',checkpoints=p.checkpoints,signal_checkpoints=p.signal_checkpoints)],request='postpub-owner-fixture',actor='fixture-owner',approval='approval:postpublication-judging',dry=False)
 values.update(changes);return values

def recover(p,values):
 wire=values.copy();wire['requests']=json.dumps(wire['requests'])
 with p.c.cursor()as q:result=old.old.scalar(q,'SELECT public.'+SIG,tuple(wire.values()))
 p.c.commit();return result

def test_missing_owner_entrypoint_before_migration(before):
 with before.c.cursor()as q:assert old.old.scalar(q,"SELECT to_regprocedure('public.lab_arena_recover_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)')")is None
 before.c.commit()

def test_native_atomic_reopen_grant_claim_preserves_reward_and_every_old_run(post):
 p=post;values=args(p);before=old.base.snapshot(p)
 preview=recover(p,{**values,'dry':True});assert preview['status']=='dry_run'and old.base.snapshot(p)==before
 result=recover(p,values);assert result['status']=='recovered';assert len(result['grants'])==1
 after=old.base.snapshot(p);r=after['lab_arena_rounds'][0];original=before['lab_arena_rounds'][0]
 assert r['status']=='stage2_judged'and r['published_at']is None and r['publication_doc']is None
 for field in ['reward_basis_doc','reward_basis_hash','signing_key_doc','reward_activated_at','effective_reward_epoch','king_outcome','stage_generation','scoring_generation_floor']:assert r[field]==original[field]
 records={x['run_id']:x for x in before['lab_arena_runs']};assert {x['run_id']:x for x in after['lab_arena_runs']if x['run_id']in records}==records
 assert recover(p,values)['idempotent']is True and old.base.snapshot(p)==after
 child=result['grants'][0]['child_identity']['run_id'];old.prior.claim_child(p,child)

@pytest.mark.parametrize('fault',['round_hash','runtime','parent_hash','checkpoints','signals','approval','deadline','duplicate','already_final_budget','active','guard'])
def test_invalid_authority_is_atomic_without_reopening(post,fault):
 p=post;values=args(p)
 if fault=='round_hash':values['sha']='sha256:'+'0'*64
 elif fault=='runtime':values['runtime']='sha256:'+'f'*64
 elif fault=='parent_hash':values['requests'][0]['parent_sha256']='sha256:'+'0'*64
 elif fault=='checkpoints':values['requests'][0]['checkpoints']={}
 elif fault=='signals':values['requests'][0]['signal_checkpoints']={'wrong':'value'}
 elif fault=='approval':values['approval']=' '
 elif fault=='deadline':values['final']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
 elif fault=='duplicate':values['requests']*=2
 else:
  with p.c.cursor()as q:
   if fault=='guard':q.execute('UPDATE lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
   elif fault=='active':
    q.execute('SET LOCAL session_replication_role=replica');q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
    q.execute("UPDATE lab_arena_runs SET status='pending'WHERE run_id=%s",(p.original_parent['run_id'],))
    q.execute('SET LOCAL session_replication_role=origin');q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
   else:
    # Disposable exact immutable history fixture: simulate an already spent finite grant.
    q.execute('SET LOCAL session_replication_role=replica')
    row=old.old.scalar(q,'SELECT to_jsonb(h)FROM lab_arena_scoring_supersessions h WHERE round_id=%s ORDER BY generation_floor DESC LIMIT 1',(old.old.ROUND,))
    row.update(generation_floor=row['generation_floor']+1,request_id='earlier-spent-finite-budget')
    row['recovery_doc']['recovery_kind']='operator_judge_budget_exception_v1'
    old.old.insert(q,'lab_arena_scoring_supersessions',row);q.execute('SET LOCAL session_replication_role=origin')
  p.c.commit()
 before=old.base.snapshot(p)
 with pytest.raises(p.pg.Error):recover(p,values)
 p.c.rollback();assert old.base.snapshot(p)==before

@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_terminal_children_and_owner_replay_never_add_a_successor(post,outcome):
 p=post;values=args(p);child=recover(p,values)['grants'][0]['child_identity']['run_id'];old.prior.claim_child(p,child)
 if outcome=='expired':
  with p.c.cursor()as q:q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second'WHERE run_id=%s",(child,))
  p.c.commit();old.old.rpc(p,'lab_arena_expire_leases(%s)',(old.old.ROUND,))
 else:old.old.rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(child,old.old.TOKEN,json.dumps({'terminal_status':outcome,'failure_diagnostic':{'stage':'scorer','error_class':'judge_error','reason':'malformed_response'}}),outcome,child+'-output'if outcome=='accepted'else None))
 assert recover(p,values)['idempotent']is True
 with p.c.cursor()as q:assert old.old.scalar(q,'SELECT count(*)FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',(p.parent['assignment_id'],p.parent['attempt']+1))==0
 p.c.commit()

@pytest.mark.parametrize('field',['reward_basis_doc','reward_basis_hash','reward_activated_at','effective_reward_epoch','king_outcome','champion_hotkey'])
def test_recovery_keeps_sealed_reward_and_champion_fields_immutable(post,field):
 p=post;recover(p,args(p));before=old.base.snapshot(p)
 with p.c.cursor()as q:
  with pytest.raises(p.pg.Error):
   if field in ('reward_basis_doc',):q.execute('UPDATE lab_arena_rounds SET '+field+"='{}'::jsonb WHERE round_id=%s",(old.old.ROUND,))
   elif field in ('effective_reward_epoch',):q.execute('UPDATE lab_arena_rounds SET '+field+'=2 WHERE round_id=%s',(old.old.ROUND,))
   elif field=='champion_hotkey':q.execute('UPDATE lab_arena_rounds SET champion_hotkey=%s WHERE round_id=%s',(p.parent['miner_hotkey'],old.old.ROUND))
   elif field=='king_outcome':q.execute("UPDATE lab_arena_rounds SET king_outcome='crowned'WHERE round_id=%s",(old.old.ROUND,))
   else:q.execute('UPDATE lab_arena_rounds SET '+field+'=NULL WHERE round_id=%s',(old.old.ROUND,))
 p.c.rollback();assert old.base.snapshot(p)==before


def test_service_role_cannot_open_recovery(post):
 p=post;values=args(p);wire=values.copy();wire['requests']=json.dumps(wire['requests'])
 with pytest.raises(p.pg.errors.InsufficientPrivilege):old.old.rpc(p,SIG,tuple(wire.values()))


def test_migration_replay_preserves_all_state(post):
 p=post;before=old.base.snapshot(p)
 with p.c.cursor()as q:q.execute(SQL.read_text())
 p.c.commit();assert old.base.snapshot(p)==before


@pytest.mark.parametrize('unexpected_crown',[False,True])
def test_real_service_late_judgments_wait_for_deadline_then_republish_sealed_no_king(post,tmp_path,unexpected_crown):
    from dataclasses import replace
    import time
    from urllib.parse import urlsplit
    from copy import deepcopy
    from lab_arena import contracts,runtime,scoring,intent_details_policy,quality_policy
    from qualification.scoring.competition import apply_company_judgment_context
    from tests.test_arena_company_quality_policy import _positive_breakdown
    from tests.lab_arena.company_quality_round_test import QualityHarness
    from tests.lab_arena.icp_fixtures import daily_icps
    from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
    from tests.lab_arena import test_lab_arena_service_round as fixtures
    p=post;company_quality=True;normalized_scale=False
    if p.organizer:pytest.skip('canonical three-miner lifecycle uses ordinary payer; both payer lineages have separate native controls')
    with p.c.cursor()as q:q.execute('TRUNCATE lab_arena_rounds,lab_arena_ledger,lab_arena_trajectory_events CASCADE')
    p.c.commit()
    psycopg2, dsn = p.pg,p.dsn
    connect = lambda: psycopg2.connect(**dsn)
    harness = QualityHarness(
        connect,
        tmp_path,
        challengers=["MinerA","MinerB","MinerC"],
        runners=["alpha", "beta"],
    )
    frozen_bank = []
    for position, source in enumerate(daily_icps()[:10]):
        row = deepcopy(source)
        row.update(
            icp_id=f"icp_20260923_{position + 1:03d}",
            prompt=(
                f"Public fixture ICP {position + 1}."
                " Target contacts: VP Engineering in the United States."
            ),
            contact_policy="contacts_v1",
            target_roles=["VP Engineering"],
            target_seniority="VP",
            contact_geography={
                "countries": ["US"],
                "regions": [],
                "cities": [],
            },
        )
        frozen_bank.append(row)
    bank_snapshot = deepcopy(frozen_bank)
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        promotion_margin=0.5,
        runner_slot_ceiling=10,
        pool_percent=30,
        rewards_enabled=True,
        per_icp_cost_policy=True,
        normalize_intent_scale=normalized_scale,
        contacts_from=None,
        company_quality_from=(
            "2026-01-01T00:00:00Z" if company_quality else None
        ),
        intent_details_from="2026-01-01T00:00:00Z",
        execution_sequence_from="2026-01-01T00:00:00Z",
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready",
        "set_id": int(kwargs["set_id"]),
        "icps": deepcopy(frozen_bank),
    }
    original_runner = harness.runner

    def verified_runner(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = verified_runner
    original_run_icp = harness.sandbox.run_icp
    observed_execution_inputs = []
    observed_scoring_inputs = []
    recovering = False
    broken_sources = set()
    kept_broken = None
    boundary_checks = []

    def run_icp(spec, **kwargs):
        document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(encoding="utf-8")
        )
        base = original_run_icp(spec, **kwargs)
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            observed_scoring_inputs.append(document)
            assert "contact_source_evidence" not in document
            assert all("contact" not in row for row in document["companies"])
            assert ("company_judgment_cache" in document) is company_quality

            is_miner = document["companies"] and document["companies"][0]["company_name"].startswith("Miner")
            position = int(document["icp"]["icp_id"].rsplit("_",1)[-1])-1
            if recovering and not boundary_checks:
                # At the deadline a leased final judgment must drain; activated
                # authority and the reopened phase must remain intact.
                boundary_checks.append(document["scored_run_id"])
                scheduled_now=harness.clock.now
                try:
                    harness.clock.now=publication+timedelta(seconds=1)
                    held=harness.service.advance_round(round_id)
                    assert held['status']=='retry'and held['reason']=='judge_incomplete',held
                    active=harness.service.store.get_round(round_id)
                    assert active['status']=='stage2_judged'
                    for field in ['reward_basis_doc','reward_basis_hash','reward_activated_at','effective_reward_epoch','king_outcome']:
                        assert active[field]==sealed[field],field
                    assert harness.service.store.get_run(document['scored_run_id'])['status']=='accepted'
                finally:harness.clock.now=scheduled_now
            must_fail = is_miner and position == 0 and (not recovering or document["scored_run_id"] == kept_broken)
            if must_fail: broken_sources.add(document["scored_run_id"])
            def judge(companies, _buyer, _reference):
                judgments = []
                for company in companies:
                    name = str(company["company_name"])
                    if must_fail and name.endswith(" Company 1"):
                        raise scoring.ScoringError("fixture malformed judgment",failure_reason="malformed_response")
                    slug = name.casefold().replace(" ", "-")
                    score = (1.0 if unexpected_crown else 60.0) if name.startswith("PublicBaseline") else (100.0 if recovering else 10.0)
                    raw = _positive_breakdown(
                        name,
                        str(urlsplit(company["company_website"]).hostname),
                        slug,
                    )
                    raw.update(
                        final_score=score,
                        intent_signal_raw=score,
                        intent_signal_final=score,
                    )
                    raw["intent_signals_detail"][0].update(
                        raw=score, after_decay=score
                    )
                    raw["verifier_gate_receipts"].append(
                        {"gate": "intent_details", "decision": "match"}
                    )
                    judgments.append(raw)
                return apply_company_judgment_context(
                    companies, judgments, contacts_required=False
                )

            judge.company_quality = company_quality
            judge.integrity_policy = True
            judge.contacts_required = False
            item = {"scored_run_id": document["scored_run_id"]}
            if company_quality:
                completed=[]
                try:
                    full, new = scoring.score_quality_work_item(
                        item,icp=document["icp"],companies=document["companies"],scorer=judge,
                        cache_context=document["company_judgment_cache"],
                        on_company_judgments=lambda rows:completed.__setitem__(slice(None),rows))
                except scoring.ScoringError:
                    return runtime.fake_result(exit_code=0,output_bytes=json.dumps(scoring.build_scoring_failure(
                        document["scored_run_id"],"judge_error",reason="malformed_response",company_judgments=completed)).encode())
            else:
                full = scoring.score_work_item(
                    item,
                    icp=document["icp"],
                    companies=document["companies"],
                    scorer=judge,
                )
                new = None
            output = scoring.build_scoring_output(
                document["scored_run_id"],
                full,
                **({"company_judgments": new} if new is not None else {}),
            )
            return runtime.fake_result(
                exit_code=0, output_bytes=json.dumps(output).encode()
            )

        observed_execution_inputs.append(document)
        assert document["output_schema_version"] == (
            intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
        )
        assert "contact_policy" not in document["icp"]
        assert "target_roles" not in document["icp"]
        assert "Target contacts:" not in document["icp"]["prompt"]
        rows = json.loads(base.output_bytes)["companies"]
        flavor = rows[0]["company_name"].split(" Company ", 1)[0]
        for row in rows:
            slug = row["company_name"].casefold().replace(" ", "-")
            row.update(
                company_linkedin=f"https://linkedin.com/company/{slug}",
                state="CA",
                intent_details=(
                    f"{row['company_name']} announced a product launch that "
                    "matches the requested signal."
                ),
                company_stage_evidence=[],
                # V6 accepts old agents that still emit a contact, then removes
                # it before persistence and scoring.
                contact={"legacy": flavor, "invalid": True},
            )
            row.pop("fit_summary", None)
            row.pop("fit_evidence_urls", None)
            for signal in row["intent_signals"]:
                signal.pop("why_now", None)
                signal.pop("snippet", None)
        return runtime.fake_result(
            exit_code=0,
            output_bytes=json.dumps(
                {
                    "schema_version": (
                        intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
                    ),
                    "companies": rows,
                }
            ).encode(),
        )

    harness.sandbox.run_icp = run_icp
    harness.chain.epoch = 40_000
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-11-09-postpub"
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(seconds=5), round_id=round_id
    )
    harness.round_id = round_id
    # Register the unchanged frozen image as its own selected compatible runtime.
    with p.c.cursor()as q:
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)',(configuration['scorer_image_digest'],configuration['scorer_image_reference'],configuration['scorer_image_digest'],configuration['scorer_image_reference']))
    p.c.commit()
    miners=[harness.submit(flavor,round_id)for flavor in ('MinerA','MinerB','MinerC')]
    def age_pending():
        with harness.connect()as c,c.cursor()as q:
            q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour'WHERE round_id=%s AND kind='score'AND status='pending'AND attempt>1",(round_id,))
    harness.clock.advance_to(harness.schedule()['submission_cutoff'])
    harness.advance_until('stage2_scoring',runners=2,max_steps=120)
    for _ in range(8):
        waiting=harness.service.store.list_runs(round_id,kind='score')
        if any(r['status']=='pending'and r['attempt']>1 for r in waiting):age_pending()
        harness.run_stage_with_runners(2)
        pending=[r for r in harness.service.store.list_runs(round_id,kind='score')if r['status']=='pending']
        if not pending:break
    assert len(broken_sources)==3
    # Complete any normal closure step, but do not fabricate a score for failed judges.
    if harness.status()=='stage2_scoring':harness.service.advance_round(round_id)
    assert harness.status()=='stage2_judged'
    judges=harness.service.store.list_runs(round_id,kind='score')
    parents=[]
    for source in sorted(broken_sources):parents.append(max((r for r in judges if r['scored_run_id']==source),key=lambda r:r['attempt']))
    assert len({r['submission_id']for r in parents})==3
    kept_broken=parents[-1]['scored_run_id']
    def native_parent_request(parent,request):
        with p.c.cursor()as q:
            saved=old.old.scalar(q,'SELECT to_jsonb(r)FROM lab_arena_runs r WHERE run_id=%s',(parent['run_id'],))
            sha='sha256:'+old.old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex')FROM lab_arena_runs r WHERE run_id=%s",(parent['run_id'],))
            cp=old.old.scalar(q,"SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::text,encode(extensions.digest(to_jsonb(j)::text,'sha256'),'hex')),'{}')FROM lab_arena_company_judgments j WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(%s::jsonb)ref WHERE ref->>'cache_key'=j.cache_key)",(json.dumps(saved['company_judgment_refs']),))
            signals=old.old.scalar(q,"SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::text,encode(extensions.digest(to_jsonb(j)::text,'sha256'),'hex')),'{}')FROM lab_arena_signal_judgments j WHERE source_scored_run_id=%s",(saved['scored_run_id'],))
        p.c.commit();return dict(parent_run_id=saved['run_id'],parent_sha256=sha,request_id=request,checkpoints=cp,signal_checkpoints=signals)
    third=native_parent_request(parents[-1],'native-third-final-budget')
    with p.c.cursor()as q:third_grant=old.old.scalar(q,'SELECT public.lab_arena_grant_judge_budget_exception_v1(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,FALSE)',(third['parent_run_id'],third['parent_sha256'],configuration['scorer_image_digest'],third['request_id'],'native-owner','native-earlier-approval',json.dumps(third['checkpoints']),json.dumps(third['signal_checkpoints'])))
    p.c.commit();age_pending();harness.run_stage_with_runners(2)
    assert harness.service.store.get_run(third_grant['child_identity']['run_id'])['status']=='failed'
    # Advance only the disposable scheduling clock to its real deadline boundary.
    with p.c.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb(to_char((clock_timestamp()-interval '2 seconds')AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'))),'{schedule,publication_deadline}',to_jsonb(to_char((clock_timestamp()-interval '1 second')AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"')))WHERE round_id=%s",(round_id,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit();harness.clock.now=max(harness.clock.now,datetime.now(timezone.utc))
    harness.advance_until('published',runners=2,max_steps=10)
    published=harness.service.store.get_round(round_id);assert published['king_outcome']=='no_king'
    assert harness.service.activate_reward(round_id)['status']=='activated'
    sealed=harness.service.store.get_round(round_id)
    requests=[native_parent_request(parent,'native-late-'+str(i))for i,parent in enumerate(parents[:2])]
    with p.c.cursor()as q:round_sha='sha256:'+old.old.scalar(q,"SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex')FROM lab_arena_rounds r WHERE round_id=%s",(round_id,))
    p.c.commit();now=datetime.now(timezone.utc).replace(microsecond=0);close=now+timedelta(seconds=12);publication=now+timedelta(seconds=20)
    values=dict(round=round_id,sha=round_sha,runtime=configuration['scorer_image_digest'],final=close.isoformat(),publication=publication.isoformat(),requests=requests,request='native-postpublication-owner',actor='native-owner',approval='native-late-approval',dry=False)
    all_before=harness.service.store.list_runs(round_id)
    recovered=recover(p,values);assert len(recovered['grants'])==2
    recovering=True;age_pending();harness.run_stage_with_runners(2)
    assert len(boundary_checks)==1
    for g in recovered['grants']:assert harness.service.store.get_run(g['child_identity']['run_id'])['status']=='accepted'
    early=harness.service.advance_round(round_id)
    assert harness.status()=='stage2_judged'and early['status']=='retry',early
    assert datetime.now(timezone.utc)<close
    time.sleep(max(0,(publication-datetime.now(timezone.utc)).total_seconds())+0.1)
    harness.clock.now=datetime.now(timezone.utc)
    harness.advance_until('scored',runners=2,max_steps=10)
    if unexpected_crown:
        from lab_arena.store import ArenaStoreError
        with pytest.raises(ArenaStoreError,match='postpublication'):
            harness.service.publish(round_id)
        still=harness.service.store.get_round(round_id)
        assert still['status']=='scored'and still['publication_doc']is None
        assert still['reward_basis_hash']==sealed['reward_basis_hash']and still['reward_activated_at']==sealed['reward_activated_at']
        return
    harness.advance_until('published',runners=2,max_steps=10)
    amended=harness.service.store.get_round(round_id)
    for field in ['reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at','king_outcome','champion_hotkey','champion_submission_id']:
        assert amended[field]==sealed[field],field
    assert amended['publication_doc']['king_decision']==sealed['publication_doc']['king_decision']
    assert harness.service.activate_reward(round_id)['status']=='existing'
    assert recover(p,values)['idempotent']is True
    after_by_id={r['run_id']:r for r in harness.service.store.list_runs(round_id)}
    for r in all_before:
        assert after_by_id[r['run_id']]['status']==r['status']
        if r.get('per_icp_score')is not None:assert after_by_id[r['run_id']]['per_icp_score']==r['per_icp_score']
    assert len(harness.service.store.list_runs(round_id,kind='execute'))==40
    ranking={r['submission_id']:r for r in amended['publication_doc']['final_ranking']}
    for parent in parents[:2]:assert ranking[parent['submission_id']]['final_score']is not None
    assert ranking[parents[-1]['submission_id']]['final_score']is None


def test_latest640_next_day_defender_is_current_but_ordinary_pending_day_blocks(post):
 p=post
 with p.c.cursor()as q:
  r=old.old.scalar(q,'SELECT to_jsonb(r)FROM lab_arena_rounds r WHERE round_id=%s',(old.old.ROUND,))
  day=datetime.fromisoformat(r['evaluation_date'])+timedelta(days=1)
  rid='arena-'+day.strftime('%Y-%m-%d')+'-future'
  cfg=copy.deepcopy(r['configuration_doc']);cfg['round_id']=rid;cfg['reasoning_capture_policy']='full_generated_thinking_v1'
  q.execute('SET LOCAL session_replication_role=replica')
  old.old.insert(q,'lab_arena_rounds',dict(round_id=rid,status='committed',configuration_doc=cfg,rewards_enabled=True,evaluation_date=day.strftime('%Y-%m-%d'),participants=[]))
  q.execute('SET LOCAL session_replication_role=origin')
  before=old.old.scalar(q,'SELECT to_jsonb(r)FROM lab_arena_rounds r WHERE round_id=%s',(rid,))
  assert old.old.scalar(q,'SELECT public.lab_arena__baseline_source_current_v1(r)FROM lab_arena_rounds r WHERE round_id=%s',(rid,))is True
 p.c.commit();recover(p,args(p))
 with p.c.cursor()as q:
  assert old.old.scalar(q,'SELECT public.lab_arena__baseline_source_current_v1(r)FROM lab_arena_rounds r WHERE round_id=%s',(rid,))is True
  assert old.old.scalar(q,'SELECT to_jsonb(r)FROM lab_arena_rounds r WHERE round_id=%s',(rid,))==before
  other=copy.deepcopy(cfg);other['round_id']=old.old.ROUND+'-blocked'
  q.execute('SET LOCAL session_replication_role=replica')
  old.old.insert(q,'lab_arena_rounds',dict(round_id=other['round_id'],status='stage2_judged',configuration_doc=other,rewards_enabled=True,evaluation_date=r['evaluation_date'],participants=[]))
  q.execute('SET LOCAL session_replication_role=origin')
  assert old.old.scalar(q,'SELECT public.lab_arena__baseline_source_current_v1(r)FROM lab_arena_rounds r WHERE round_id=%s',(rid,))is False
 p.c.commit()


@pytest.mark.parametrize('diagnostic',['scorer','timeout'])
def test_actual_compatible_parent_variants_keep_frozen_checkpoints(post,diagnostic):
 p=post;f.change_parent(p,diagnostic,null_runtime=diagnostic=='timeout')
 values=args(p,runtime=f.RUNTIME)
 result=recover(p,values);assert result['grants'][0]['runtime_image_digest']==f.RUNTIME
 old.prior.claim_child(p,result['grants'][0]['child_identity']['run_id'])


@pytest.mark.parametrize('fault',['source','acl','owner','search_path','trigger'])
def test_new_install_security_drift_refuses_atomically(post,fault):
 p=post;sig='public.lab_arena__postpublication_receipt_v1(public.lab_arena_scoring_supersessions)'
 with p.c.cursor()as q:
  if fault=='source':
   definition=old.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))
   source=old.old.scalar(q,'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure',(sig,))
   q.execute(definition.replace(source,source+'\n-- drift',1))
  elif fault=='acl':q.execute('GRANT EXECUTE ON FUNCTION '+sig+' TO lab_arena_service')
  elif fault=='owner':q.execute('ALTER FUNCTION '+sig+' OWNER TO CURRENT_USER')
  elif fault=='search_path':q.execute('ALTER FUNCTION '+sig+' SET search_path=public')
  else:q.execute('ALTER TABLE lab_arena_rounds DISABLE TRIGGER lab_arena_postpublication_reward_seal')
  with pytest.raises(p.pg.Error,match='installed (function|trigger) differs'):q.execute(SQL.read_text())
 p.c.rollback()
 with p.c.cursor()as q:q.execute(SQL.read_text())
 p.c.commit()


def test_disposable_catalogue_matches_all_nine_reviewed_definitions(post):
 p=post;sources=re.findall(r"\('public\.(lab_arena_[^']+)','([0-9a-f]{64})','([^']+)'",SQL.read_text())
 expected={}
 for sig,first,second in sources:expected['public.'+sig]=second if re.fullmatch('[0-9a-f]{64}',second)else first
 assert len(expected)==9
 with p.c.cursor()as q:
  catalogue=[]
  for sig,sha in expected.items():
   q.execute("SELECT pg_get_function_identity_arguments(p.oid),encode(extensions.digest(p.prosrc,'sha256'),'hex'),pg_get_userbyid(p.proowner),p.proacl::text,p.prosecdef,p.provolatile,p.proconfig,l.lanname,p.proisstrict,p.proleakproof,p.proparallel FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=%s::regprocedure",(sig,))
   args_,actual,owner,acl,definer,volatility,config,language,strict,leakproof,parallel=q.fetchone()
   assert actual==sha and owner=='lab_arena_owner'and definer and config==['search_path=pg_catalog, public']
   assert not strict and not leakproof and parallel=='u'
   catalogue.append(dict(signature=sig,identity_arguments=args_,sha256=actual,owner=owner,acl=acl,security_definer=definer,volatility=volatility,config=config,language=language,strict=strict,leakproof=leakproof,parallel=parallel))
  q.execute("SELECT tgname,tgenabled,tgtype,tgqual::text,tgnargs,tgattr::text,tgfoid::regprocedure::text FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_postpublication_reward_seal'")
  trigger=q.fetchone()
 p.c.commit();print('CATALOGUE678='+json.dumps(dict(functions=catalogue,trigger=trigger),sort_keys=True))


@pytest.mark.parametrize('check',[old.base.test_prior_costs_still_exhaust_the_same_cumulative_budget,old.base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_postpublication_grant_keeps_existing_money_and_call_quotas(post,monkeypatch,check):
 monkeypatch.setattr(old.base,'grant',lambda p:recover(p,args(p))['grants'][0])
 check(post)
