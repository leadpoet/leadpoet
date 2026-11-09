"""Finite exceptions preserve frozen grading across compatible runtime changes."""
import json
from pathlib import Path
import pytest
from tests.lab_arena import owner_judge_budget_exception676_postgres_test as old

SQL=Path(__file__).parents[2]/'scripts/677-lab-arena-finite-judge-compatible-runtime.sql'
RUNTIME='sha256:'+'4'*64
database=old.database
prepared=old.prepared
prepared_company=old.prepared_company
case=old.case
base_extended=old.base_extended
extended=old.extended
provider_case=old.provider_case
correctable=old.correctable
corrected=old.corrected
finite=old.finite
plain_finite=old.plain_finite
COMMON=old.prior.COMMON

@pytest.fixture
def compatible(finite):
 p=finite
 with p.c.cursor()as q:
  original=old.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(COMMON,));q.execute(SQL.read_text())
 p.c.commit()
 try:yield p
 finally:
  p.c.rollback()
  with p.c.cursor()as q:q.execute(original)
  p.c.commit()

def change_parent(p,diagnostic='scorer',null_runtime=False):
 with p.c.cursor()as q:
  q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal');q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime');q.execute('SET LOCAL session_replication_role=replica')
  if diagnostic!='scorer':
   q.execute("UPDATE lab_arena_runs SET result_doc=jsonb_set(result_doc,'{failure_diagnostic}',%s::jsonb)WHERE run_id=%s",(json.dumps({'stage':'sandbox','error_class':'judge_timeout'if diagnostic=='timeout'else'unknown','reason':'sandbox_judge_timeout'}),p.parent['run_id']))
  if null_runtime:q.execute('UPDATE lab_arena_runs SET reviewer_runtime_doc=NULL WHERE run_id=%s',(p.parent['run_id'],))
  q.execute('SET LOCAL session_replication_role=origin');q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal');q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
 p.c.commit();old.prior.saved_parent(p,p.parent['run_id']);old.prior.mapping(p,RUNTIME)

@pytest.mark.parametrize('diagnostic',['scorer','timeout'])
def test_before_migration_refuses_actual_parent_variants(finite,diagnostic):
 p=finite;change_parent(p,diagnostic,null_runtime=diagnostic=='timeout');before=old.base.snapshot(p)
 with pytest.raises(p.pg.Error):old.grant(p,runtime=RUNTIME)
 p.c.rollback();assert old.base.snapshot(p)==before

@pytest.mark.parametrize('diagnostic',['scorer','timeout'])
@pytest.mark.parametrize('null_runtime',[False,True])
def test_selected_compatible_runtime_admits_once_preserving_native_state(compatible,diagnostic,null_runtime):
 p=compatible;change_parent(p,diagnostic,null_runtime);before=old.base.snapshot(p)
 preview=old.grant(p,runtime=RUNTIME,dry=True);assert old.base.snapshot(p)==before
 result=old.grant(p,runtime=RUNTIME);assert result['child_identity']==preview['child_identity'];assert result['runtime_image_digest']==RUNTIME
 assert old.grant(p,runtime=RUNTIME)['idempotent']is True
 after=old.base.snapshot(p)
 originals={r['run_id']:r for r in before['lab_arena_runs']}
 assert {r['run_id']:r for r in after['lab_arena_runs']if r['run_id']in originals}==originals
 for table in set(before)-{'lab_arena_runs','lab_arena_scoring_supersessions'}:assert after[table]==before[table]
 child=result['child_identity']['run_id'];assert old.old.funding(p,child)['funding_source']==('host'if p.organizer else'miner_key')
 old.prior.claim_child(p,child)
 with p.c.cursor()as q:assert old.old.scalar(q,"SELECT reviewer_runtime_doc->>'runtime_image_digest'FROM lab_arena_runs WHERE run_id=%s",(child,))==RUNTIME
 p.c.commit();old.prior.complete_failed(p,child);old.prior.saved_parent(p,child)
 with pytest.raises(p.pg.Error):old.grant(p,runtime=RUNTIME,request_id='no-second-finite-exception')
 p.c.rollback()

@pytest.mark.parametrize('diagnostic',['unknown','timeout'])
def test_alternate_diagnostic_remains_exact_and_legacy_kind_refuses(compatible,diagnostic):
 p=compatible;change_parent(p,diagnostic);before=old.base.snapshot(p)
 if diagnostic=='unknown':
  with pytest.raises(p.pg.Error):old.grant(p,runtime=RUNTIME)
  p.c.rollback();assert old.base.snapshot(p)==before
 with pytest.raises(p.pg.Error):old.prior.grant(p,runtime=RUNTIME,request_id='not-a-scorer-correction')
 p.c.rollback();assert old.base.snapshot(p)==before

@pytest.fixture
def compatible_plain(plain_finite):
 p=plain_finite
 with p.c.cursor()as q:
  original=old.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(COMMON,));q.execute(SQL.read_text())
 p.c.commit()
 try:yield p
 finally:
  p.c.rollback()
  with p.c.cursor()as q:q.execute(original)
  p.c.commit()


def test_legacy_scorer_kind_keeps_its_changed_runtime_rule(compatible_plain):
 old.test_legacy_corrected_runtime_grant_still_works_after_extension(compatible_plain)


def test_legacy_provider_kind_keeps_same_runtime_and_nonrecursion(compatible_plain):
 old.test_legacy_provider_grant_and_nonrecursion_stay_exact(compatible_plain)


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_native_finite_terminal_paths_still_have_no_successor(compatible,outcome):
 old.test_native_finality_and_no_second_exception(compatible,outcome)


@pytest.mark.parametrize('fault',['stale_hash','runtime','approval','checkpoints','signal_checkpoints','accepted','later','live','calls_pending','expired','published','generation','frozen_source'])
def test_finite_safety_controls_remain_exact(compatible,fault):
 old.test_refusals_preserve_every_row(compatible,fault)


@pytest.mark.parametrize('check',[old.base.test_prior_costs_still_exhaust_the_same_cumulative_budget,old.base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_existing_cost_and_quota_denials_remain_exact(compatible,monkeypatch,check):
 old.test_existing_money_and_provider_quotas(compatible,monkeypatch,check)


def test_migration_replay_and_unknown_function_state_fail_closed(compatible):
 p=compatible;before=old.base.snapshot(p)
 with p.c.cursor()as q:q.execute(SQL.read_text())
 p.c.commit();assert old.base.snapshot(p)==before
 with p.c.cursor()as q:
  definition=old.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(COMMON,))
  source=old.old.scalar(q,'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure',(COMMON,))
  q.execute(definition.replace(source,source+'\n-- unexpected source',1))
  with pytest.raises(p.pg.Error,match='preimage differs'):q.execute(SQL.read_text())
 p.c.rollback();assert old.base.snapshot(p)==before
