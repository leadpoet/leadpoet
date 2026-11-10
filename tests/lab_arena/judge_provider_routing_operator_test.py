"""The operator command is inert by default and passes one exact CAS request."""
from copy import deepcopy
import json

import pytest

from scripts import configure_arena_judge_provider_routing as operator


def request():
    model='z-ai/glm-5.3-flash'
    return {'round_id':'arena-2026-11-10','expected_configuration':{
        'round_id':'arena-2026-11-10','scorer_policy':{'judge_models':{'industry_research_controller':model}},
        'keep':{'original':True}},'routing':{model:['fireworks']},
        'request_id':'routing683-cli-request','actor_ref':'fixture'}


def test_offline_preview_has_no_connection_or_configuration_dump(tmp_path,capsys):
    path=tmp_path/'request.json';path.write_text(json.dumps(request()))
    assert operator.main(['--request-file',str(path)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['writes']is False
    assert 'expected_configuration'not in result
    assert result['judge_provider_routing']==request()['routing']


@pytest.mark.parametrize('fault',['unknown','wrong_round','wrong_model','empty','duplicates','too_many'])
def test_invalid_request_fails_before_connection(fault):
    value=request()
    if fault=='unknown':value['extra']=True
    elif fault=='wrong_round':value['round_id']='arena-2026-11-11'
    elif fault=='wrong_model':value['routing']={'other/model':['fireworks']}
    elif fault=='empty':value['routing']={}
    elif fault=='duplicates':value['routing']['z-ai/glm-5.3-flash']=['fireworks','fireworks']
    else:value['routing']['z-ai/glm-5.3-flash']=['a','b','c']
    with pytest.raises(ValueError):operator.request_arguments(value)


def test_apply_requires_explicit_operator_environment(tmp_path,monkeypatch):
    monkeypatch.delenv(operator.APPLY_ENV,raising=False)
    path=tmp_path/'request.json';path.write_text(json.dumps(request()))
    with pytest.raises(ValueError,match='operator environment'):
        operator.main(['--request-file',str(path),'--apply'])


def test_exact_cas_arguments_and_transaction_boundary():
    arguments=operator.request_arguments(request());original=deepcopy(arguments)
    class Connection:
        def __init__(self):self.statements=[];self.commits=0;self.rollbacks=0
        def cursor(self):return self
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def execute(self,*args):self.statements.append(args)
        def fetchone(self):return [{'status':'dry_run','request_id':request()['request_id'],
                                   'judge_provider_routing':request()['routing']}]
        def commit(self):self.commits+=1
        def rollback(self):self.rollbacks+=1
    connection=Connection()
    operator.execute(connection,arguments,apply=False)
    _,values=connection.statements[-1]
    assert values==(arguments['p_round_id'],json.dumps(arguments['p_expected_configuration']),
                   json.dumps(arguments['p_routing']),arguments['p_request_id'],arguments['p_actor_ref'],True)
    assert arguments==original and connection.commits==0 and connection.rollbacks==1
