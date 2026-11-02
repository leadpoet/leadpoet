"""One complete structured correction for a strictly invalid initial verdict."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import time

import pytest

from qualification.scoring import intent_details as review, verification_helpers

FIXTURE=Path(__file__).with_name('fixtures')/'intent_details_aggregate_protocol_captures.json'


def captured(name):
    value=json.loads(FIXTURE.read_text())['cases'][name]
    return value['document'],value['initial_response']


def run(monkeypatch,document,answers):
    calls=[]
    monkeypatch.setattr(review,'review_evidence',lambda *a,**k:copy.deepcopy(document))
    async def judge(prompt,**kwargs):
        calls.append((prompt,kwargs))
        assert len(calls)<=len(answers),'unexpected extra paragraph call'
        answer=answers[len(calls)-1]
        return json.dumps(answer)
    monkeypatch.setattr(verification_helpers,'openrouter_chat',judge)
    return asyncio.run(review.review_intent_details(None,None,[],{})),calls


def test_complete_correction_reconciles_actual_ahf_with_fresh_negative_checks(monkeypatch):
    document,initial=captured('AHF');fresh=copy.deepcopy(initial);fresh['facts_supported']=True
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2
    assert receipt['decision']=='mismatch'
    assert receipt['checks']['facts_supported']is True
    assert receipt['checks']['relevance_grounded']is False
    assert receipt['checks']['connects_icp']is False
    assert calls[0][0]==calls[1][0]  # Same admitted document and unit boundaries.
    for _,kwargs in calls:
        assert kwargs['model']==review.REVIEW_MODEL
        assert kwargs['max_tokens']==800 and kwargs['max_retries']==0
        assert kwargs['response_format']==review._RESPONSE_FORMAT


def test_complete_correction_can_reclassify_actual_caci_hidden_factual_premise(monkeypatch):
    document,initial=captured('CACI');fresh=copy.deepcopy(initial)
    assert fresh['unit_grounding'][2]['contains_factual_claim']is False
    fresh['unit_grounding'][2].update(contains_factual_claim=True,status='UNPROVEN',evidence=[])
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2 and receipt['decision']=='mismatch'
    assert receipt['checks']['facts_supported']is False
    assert any(x['unit_id']==2 and x['status']=='UNPROVEN'for x in receipt['failed_factual_units'])


def test_complete_correction_keeps_exact_nitra_date_negative(monkeypatch):
    document,initial=captured('Nitra');fresh=copy.deepcopy(initial)
    for unit in fresh['unit_grounding'][:2]:unit['status']='UNPROVEN'
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2 and receipt['decision']=='mismatch'
    assert receipt['checks']['facts_supported']is False
    assert {x['unit_id']for x in receipt['failed_factual_units']}=={0,1}


def test_invalid_initial_independent_flags_are_not_authority(monkeypatch):
    document,initial=captured('CACI');fresh=copy.deepcopy(initial)
    # Contract seam only: this supplied fresh answer is not a semantic prediction.
    for name in review._CHECKS:fresh[name]=True
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2 and receipt['decision']=='match'
    assert all(receipt['checks'].values())


def simple(*,relative=False):
    text='Example recently launched a product.'if relative else'Example launched a product.'
    doc={'intent_details_units':[{'unit_id':0,'text':text}],
        'verified_signals':[{'matched_icp_signal':0}],
        'admitted_evidence':[{'source_index':0,'evidence_kind':'verified_source_context',
            'admitted_text':['Example launched a product.','March 1, 2026']} ]}
    answer={'unit_grounding':[{'unit_id':0,'contains_factual_claim':True,'status':'VERIFIED',
                'evidence':[{'source_index':0,'quote':'Example launched a product.'}]}],
            'signal_coverage':[{'matched_icp_signal':0,'covered':True}],
            **{name:True for name in review._CHECKS}}
    return doc,answer


@pytest.mark.parametrize('negative',[False,True])
def test_first_valid_response_receives_no_complete_correction(monkeypatch,negative):
    document,answer=simple()
    if negative:
        answer['facts_supported']=False
        answer['unit_grounding'][0].update(status='UNPROVEN',evidence=[])
    receipt,calls=run(monkeypatch,document,[answer])
    assert len(calls)==1 and receipt['decision']==('mismatch'if negative else'match')


@pytest.mark.parametrize('defect',['citation','coverage_aggregate','missing_unit','all_nonfactual','wrong_boolean'])
def test_mixed_or_other_initial_defects_are_not_full_correction(monkeypatch,defect):
    document,answer=simple();answer['facts_supported']=False
    if defect=='citation':answer['unit_grounding'][0]['evidence'][0]['quote']='unbound text'
    elif defect=='coverage_aggregate':answer['verified_signals_covered']=False
    elif defect=='missing_unit':answer['unit_grounding']=[]
    elif defect=='all_nonfactual':answer['unit_grounding'][0].update(contains_factual_claim=False,evidence=[])
    else:answer['natural_paragraph']='true'
    receipt,calls=run(monkeypatch,document,[answer])
    assert len(calls)==1 and receipt['decision']=='unavailable'


@pytest.mark.parametrize('defect',['same_conflict','citation','missing_unit','coverage','wrong_boolean'])
def test_invalid_second_full_reply_never_dispatches_a_third_call(monkeypatch,defect):
    document,initial=simple();initial['facts_supported']=False;fresh=copy.deepcopy(initial);fresh['facts_supported']=True
    if defect=='same_conflict':fresh['facts_supported']=False
    elif defect=='citation':fresh['unit_grounding'][0]['evidence'][0]['quote']='unbound text'
    elif defect=='missing_unit':fresh['unit_grounding']=[]
    elif defect=='coverage':fresh['signal_coverage']=[]
    else:fresh['natural_paragraph']='true'
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2 and receipt['decision']=='unavailable'
    assert receipt['failure_reason_code']=='malformed_response'


def test_fresh_factual_classification_is_not_forced_by_invalid_initial(monkeypatch):
    document,initial=captured('CACI');fresh=copy.deepcopy(initial)
    # Classification change is a schema seam, not a correct-model prediction.
    fresh['unit_grounding'][1].update(contains_factual_claim=False,evidence=[])
    fresh['facts_supported']=True
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2 and receipt['decision']=='mismatch'
    assert receipt['checks']['facts_supported']is True


@pytest.mark.parametrize('date_proof',[False,True])
def test_fresh_relative_time_proof_uses_existing_same_source_guard(monkeypatch,date_proof):
    document,initial=simple(relative=True);initial['facts_supported']=False
    fresh=copy.deepcopy(initial);fresh['facts_supported']=True
    if date_proof:fresh['unit_grounding'][0]['evidence'].append({'source_index':0,'quote':'March 1, 2026'})
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2
    assert receipt['decision']==('match'if date_proof else'unavailable')


def test_newly_factual_relative_unit_cannot_skip_date_proof(monkeypatch):
    document,initial=simple()
    document['intent_details_units'].append({'unit_id':1,'text':'Example recently launched a product.'})
    initial['unit_grounding'].append({'unit_id':1,'contains_factual_claim':False,'status':'VERIFIED','evidence':[]})
    initial['facts_supported']=False
    fresh=copy.deepcopy(initial);fresh['facts_supported']=True
    fresh['unit_grounding'][1].update(contains_factual_claim=True,evidence=[{'source_index':0,'quote':'Example launched a product.'}])
    receipt,calls=run(monkeypatch,document,[initial,fresh])
    assert len(calls)==2 and receipt['decision']=='unavailable'


def test_second_provider_failure_remains_unavailable(monkeypatch):
    document,initial=simple();initial['facts_supported']=False;calls=[]
    monkeypatch.setattr(review,'review_evidence',lambda *a,**k:copy.deepcopy(document))
    async def judge(prompt,**kwargs):
        calls.append(kwargs)
        if len(calls)==2:raise RuntimeError('provider failed')
        return json.dumps(initial)
    monkeypatch.setattr(verification_helpers,'openrouter_chat',judge)
    receipt=asyncio.run(review.review_intent_details(None,None,[],{}))
    assert len(calls)==2 and receipt['failure_class']=='intent_details_provider_unavailable'
    assert all(x['max_retries']==0 and x['max_tokens']==800 for x in calls)


def test_first_real_timeout_does_not_dispatch_correction(monkeypatch):
    document,initial=simple();initial['facts_supported']=False;calls=[]
    monkeypatch.setattr(review,'review_evidence',lambda *a,**k:copy.deepcopy(document))
    monkeypatch.setattr(review,'REVIEW_TIMEOUT_SECONDS',0.03)
    async def judge(prompt,**kwargs):
        calls.append(kwargs);await asyncio.sleep(0.1);return json.dumps(initial)
    monkeypatch.setattr(verification_helpers,'openrouter_chat',judge)
    receipt=asyncio.run(review.review_intent_details(None,None,[],{}))
    assert len(calls)==1 and receipt['failure_class']=='intent_details_provider_unavailable'


def test_expired_budget_after_validation_does_not_dispatch_second_provider(monkeypatch):
    document,initial=simple();initial['facts_supported']=False;calls=[]
    monkeypatch.setattr(review,'review_evidence',lambda *a,**k:copy.deepcopy(document))
    monkeypatch.setattr(review,'REVIEW_TIMEOUT_SECONDS',0.02)
    native_validate=review._validate_review_response
    def slow_validation(*args,**kwargs):
        if kwargs.get('initial_review',True):time.sleep(0.04)
        return native_validate(*args,**kwargs)
    monkeypatch.setattr(review,'_validate_review_response',slow_validation)
    async def judge(prompt,**kwargs):calls.append(kwargs);return json.dumps(initial)
    monkeypatch.setattr(verification_helpers,'openrouter_chat',judge)
    receipt=asyncio.run(review.review_intent_details(None,None,[],{}))
    assert len(calls)==1 and receipt['failure_class']=='intent_details_provider_unavailable'


def test_second_real_timeout_uses_remainder_of_original_deadline(monkeypatch):
    document,initial=simple();initial['facts_supported']=False;fresh=copy.deepcopy(initial);fresh['facts_supported']=True
    monkeypatch.setattr(review,'review_evidence',lambda *a,**k:copy.deepcopy(document))
    monkeypatch.setattr(review,'REVIEW_TIMEOUT_SECONDS',0.15)
    calls=[];timeouts=[];wait_for=asyncio.wait_for
    async def observed(awaitable,*,timeout):
        timeouts.append(timeout);return await wait_for(awaitable,timeout=timeout)
    async def judge(prompt,**kwargs):
        calls.append(kwargs);await asyncio.sleep(0.1)
        return json.dumps(initial if len(calls)==1 else fresh)
    monkeypatch.setattr(verification_helpers,'openrouter_chat',judge)
    monkeypatch.setattr(review.asyncio,'wait_for',observed)
    receipt=asyncio.run(review.review_intent_details(None,None,[],{}))
    assert len(calls)==2 and receipt['failure_class']=='intent_details_provider_unavailable'
    assert 0<timeouts[1]<timeouts[0]<=0.15
    assert all(x['max_tokens']==800 and x['max_retries']==0 for x in calls)
