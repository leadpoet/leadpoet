"""Prepare ordinary capture without holding the run lock; commit atomically."""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading

import pytest

from lab_arena import runner, training_observations as obs
from tests.lab_arena.training_observations_test import observation, response


def test_prepared_evidence_is_detached_and_keeps_privacy():
    value=observation()
    value['request']['admitted_parameters']['messages'][0]['content']='exact synthetic-private-value'
    prepared=obs.Buffer.prepare_observation(value,secrets=('synthetic-private-value',))
    assert prepared is not None
    value['request']['admitted_parameters']['messages'][0]['content']='changed after preparation'
    buffer=obs.Buffer();buffer.append_prepared(prepared,local_refusal=False)
    assert buffer.attempted==1 and not buffer.failures
    stored=buffer.observations[0]
    assert stored['request']['admitted_parameters']['messages'][0]['content']=='exact [REDACTED]'
    assert stored['privacy']=={'redacted':True,'redactions':['credential']}
    assert buffer.size_bytes==len(obs._canonical(stored))+1


@pytest.mark.parametrize('limit',['count','bytes'])
def test_prepared_commits_recheck_current_limits(monkeypatch,limit):
    first=obs.Buffer.prepare_observation(observation(ordinal=0))
    second=obs.Buffer.prepare_observation(observation(ordinal=1))
    assert first and second
    if limit=='count':monkeypatch.setattr(obs,'MAX_OBSERVATIONS',1)
    else:monkeypatch.setattr(obs,'MAX_OBJECT_BYTES',65536+first[1])
    buffer=obs.Buffer()
    buffer.append_prepared(first,local_refusal=False);buffer.append_prepared(second,local_refusal=False)
    assert buffer.attempted==2 and len(buffer.observations)==1
    assert buffer.size_bytes==first[1] and buffer.failures=={'capture_incomplete'}


def test_failed_preparation_keeps_attempt_and_refusal_counts():
    value=observation(kind='local_refusal',identity=None,contributes=False)
    value['request']={}
    prepared=obs.Buffer.prepare_observation(value)
    assert prepared is None
    split=obs.Buffer();split.append_prepared(prepared,local_refusal=True)
    wrapper=obs.Buffer();wrapper.append(value)
    assert split.document()==wrapper.document()
    assert split.attempted==split.local_refusals==1
    assert split.failures=={'capture_incomplete'} and not split.observations


def worker():
    state=runner.RunState(lease={'kind':'execute'},lease_token='synthetic-local-lease')
    server=runner.WorkerSocketServer('/tmp/not-started-capture.sock',object(),state)
    pending=server._begin_observation('openrouter.chat',{'messages':[{'role':'user','content':'synthetic'}]},1000,'frame')
    return state,server,pending


def test_freeze_after_prepare_starts_rejects_commit_without_holding_lock(monkeypatch):
    state,server,pending=worker()
    entered=threading.Event();finish=threading.Event();real_clean=obs.clean_observation
    def blocked(value,secrets):
        assert not state.lock.locked()
        entered.set();assert finish.wait(5)
        return real_clean(value,secrets)
    monkeypatch.setattr(obs,'clean_observation',blocked)
    document=response()
    with ThreadPoolExecutor(max_workers=1)as pool:
        task=pool.submit(server._finish_observation,pending,{k:document[k]for k in ('status','headers','body_b64')},document=document)
        assert entered.wait(5)
        assert state.lock.acquire(timeout=1)
        try:state.worker_observations_frozen=True
        finally:state.lock.release();finish.set()
        task.result(timeout=5)
    assert not state.worker_observations.observations
    assert state.worker_observations.attempted==0
    assert not state.worker_observations.failures


def test_preparation_failure_still_commits_failure_and_clears_pending(monkeypatch):
    state,server,pending=worker()
    def fail(value,secrets):
        assert not state.lock.locked()
        raise ValueError('synthetic capture validation failure')
    monkeypatch.setattr(obs,'clean_observation',fail)
    pending['kind']='local_refusal'
    document=response()
    server._finish_observation(pending,{k:document[k]for k in ('status','headers','body_b64')},document=document)
    assert not state.worker_observations_pending
    assert state.worker_observations.attempted==state.worker_observations.local_refusals==1
    assert state.worker_observations.failures=={'capture_incomplete'}


def test_concurrent_prepare_does_not_overshoot_commit_limit(monkeypatch):
    state,server,first=worker()
    second=server._begin_observation('openrouter.chat',{},1000,'frame')
    barrier=threading.Barrier(2);real_clean=obs.clean_observation
    def concurrent(value,secrets):
        assert not state.lock.locked()
        cleaned=real_clean(value,secrets);barrier.wait(timeout=5)
        return cleaned
    monkeypatch.setattr(obs,'clean_observation',concurrent)
    monkeypatch.setattr(obs,'MAX_OBSERVATIONS',1)
    document=response();wire={k:document[k]for k in ('status','headers','body_b64')}
    with ThreadPoolExecutor(max_workers=2)as pool:
        tasks=[pool.submit(server._finish_observation,pending,copy.deepcopy(wire),document=copy.deepcopy(document))for pending in (first,second)]
        for task in tasks:task.result(timeout=5)
    assert not state.worker_observations_pending
    assert state.worker_observations.attempted==2
    assert len(state.worker_observations.observations)==1
    assert state.worker_observations.failures=={'capture_incomplete'}


def test_paid_success_does_not_use_ordinary_preparation(monkeypatch):
    state,server,pending=worker();pending['action_sequence']=0
    def forbidden(*args,**kwargs):raise AssertionError('paid receipt path changed')
    monkeypatch.setattr(obs.Buffer,'prepare_observation',forbidden)
    document=response(200,{'synthetic':'response'},outcome='settled')
    server._finish_observation(pending,{k:document[k]for k in ('status','headers','body_b64')},document=document)
    assert state.worker_observations.response_receipt_count==1
    assert not state.worker_observations.observations
    assert not state.worker_observations.failures and not state.worker_observations_pending
