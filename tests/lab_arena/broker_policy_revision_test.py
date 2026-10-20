"""A round policy revision cannot retain a previous broker model allowlist."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import pytest
from lab_arena import broker,scoring,service as svc

OLD='google/gemini-2.5-flash'
NEW='google/gemini-3.5-flash-lite'

def harness():
    policy=scoring.build_scorer_policy()
    policy['judge_models']['role_batch_check']=OLD
    row={'round_id':'arena-policy-cache','configuration_doc':{
        'scorer_policy':policy,'lease_ttl_seconds':3600,
        'baseline_source_url':'https://example.invalid/source.tar.gz',
        'mode':'live','rewards_enabled':True}}
    rows={row['round_id']:row};built=[]
    def factory(_service,snapshot):
        b=object.__new__(broker.Broker)
        b._judge_models=tuple(snapshot['configuration_doc']['scorer_policy']['judge_models'].values())
        # This test exercises the real first allowlist gate before any
        # price lookup, reservation, transport or funding resolver can run.
        b._price_table={'models':{OLD:{},NEW:{}}}
        b.snapshot=deepcopy(snapshot);built.append(b)
        return b
    s=object.__new__(svc.ArenaService);s._lock=threading.RLock();s._brokers={}
    s._hot_round=lambda rid:rows.get(rid)
    s._round=lambda _rid:pytest.fail('broker must reuse the hot snapshot')
    s._config=SimpleNamespace(broker_factory=factory)
    return s,rows,built

def test_cached_old_policy_refreshes_on_model_revision_without_widening_allowlist():
    s,rows,built=harness();rid='arena-policy-cache'
    old=s._broker_for(rid)
    with pytest.raises(broker.BrokerError,match='model_not_allowed'):
        old._openrouter_parameters({'model':NEW},kind='score')
    revised=deepcopy(rows[rid]);revised['configuration_doc']['scorer_policy']['judge_models']['role_batch_check']=NEW
    revised['configuration_doc']['scorer_policy']['env_bindings']['ARENA_ROLE_BATCH_MODEL']=NEW
    rows[rid]=revised
    current=s._broker_for(rid)
    assert current is not old and NEW in current._judge_models and OLD not in current._judge_models
    with pytest.raises(broker.BrokerError,match='model_not_allowed'):
        current._openrouter_parameters({'model':OLD},kind='score')
    # An active holder retains its own unchanged policy/credential-independent
    # broker; replacing the cache never mutates an in-flight broker object.
    assert OLD in old._judge_models and NEW not in old._judge_models
    normalized,_=current._openrouter_parameters({'model':NEW,'messages':[{'role':'user','content':'control'}],'max_tokens':1},kind='score')
    assert normalized['model']==NEW
    assert len(built)==2 and s._broker_for(rid) is current

@pytest.mark.parametrize('field,value',[
    ('lease_ttl_seconds',7200),('baseline_source_url','https://example.invalid/other'),
    ('mode','shadow'),('rewards_enabled',False),
])
def test_factory_routing_inputs_invalidate_broker(field,value):
    s,rows,_=harness();rid='arena-policy-cache';old=s._broker_for(rid)
    rows[rid]['configuration_doc'][field]=value
    assert s._broker_for(rid) is not old

def test_unchanged_context_reuses_broker_and_rounds_do_not_share_instances():
    s,rows,built=harness();rid='arena-policy-cache';first=s._broker_for(rid)
    rows[rid]=deepcopy(rows[rid]);rows[rid]['configuration_doc']['deepline_catalog']={'large':'not a factory input'}
    assert s._broker_for(rid) is first
    other=deepcopy(rows[rid]);other['round_id']='arena-other';rows['arena-other']=other
    assert s._broker_for('arena-other') is not first and len(built)==2

def test_missing_round_cannot_reuse_a_cached_broker():
    s,rows,_=harness();s._broker_for('arena-policy-cache');rows.clear()
    with pytest.raises(svc.ServiceError,match='round_missing'):s._broker_for('arena-policy-cache')


def test_existing_hot_round_ttl_bounds_revision_staleness_without_duplicate_reads(monkeypatch):
    s,rows,_=harness();rid='arena-policy-cache';now=[0.0];reads=[]
    s._hot_rounds={};s._hot_round_lock=threading.RLock()
    s._require_round_ownership=lambda _rid:None;s._require_round_mode=lambda row:row
    def read(_rid):reads.append(_rid);return deepcopy(rows[_rid])
    s._store=SimpleNamespace(get_round=read)
    s._hot_round=svc.ArenaService._hot_round.__get__(s)
    monkeypatch.setattr(svc.time,'monotonic',lambda:now[0])
    snapshot=s._hot_round(rid);old=s._broker_for(rid)
    assert len(reads)==1 and snapshot['configuration_doc']==old.snapshot['configuration_doc']
    rows[rid]['configuration_doc']['scorer_policy']['judge_models']['role_batch_check']=NEW
    now[0]=svc.HOT_ROUND_TTL_SECONDS-0.01
    assert s._broker_for(rid) is old and len(reads)==1
    now[0]=svc.HOT_ROUND_TTL_SECONDS+0.01
    assert s._broker_for(rid) is not old and len(reads)==2
