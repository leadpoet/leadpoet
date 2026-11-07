"""Local host tool outcomes survive without another provider turn; never SFT gold."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import pytest
from lab_arena import contracts, lab_arena_checkpoint as checkpoint, runner, shim, training_observations, training_thinking
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease


def event(**changes):
    return {'investigation_id': 'local-investigation-1', 'sequence': 0,
            'tool_call_id': 'tool-1', 'tool_name': 'submit_findings',
            'arguments': '{"findings":[]}', 'result': {'claims': {}, 'failure_reason': ''},
            'disposition': 'accepted_terminal', 'host_validation': {'accepted': True},
            'request_sha256': None, 'response_sha256': None, **changes}


def frame(value):
    return {'schema_version': checkpoint.TRAINING_SCHEMA_VERSION,
            'control': checkpoint.TRAINING_CONTROL, 'kind': 'judge_local_tool',
            'depends_on': [], 'data': value}


def worker(tmp_path, *, kind='score', marker=True):
    issued = lease(); issued['kind'] = kind
    if marker:
        issued['judge_reasoning_capture_policy'] = shim.FULL_THINKING_POLICY
        issued['reasoning_capture_policy'] = shim.FULL_THINKING_POLICY
    api = FakeApi([])
    state = runner.RunState(lease=issued, lease_token='host-lease-token')
    return runner.WorkerSocketServer(tmp_path/'worker.sock', api, state), state, api


def test_closed_frame_and_whole_event_privacy():
    assert checkpoint.validate_training_frame(frame(event()))['data'] == event()
    row = {'id': 0, 'kind': 'judge_local_tool', 'observed_at': '2026-11-05T00:00:00+00:00',
           'after_action_sequence': None, 'depends_on': [],
           'data': event(arguments='private-synthetic-unit-secret'),
           'privacy': {'redacted': False, 'redactions': []}}
    cleaned = training_observations.clean_agent_observation(row, ('private-synthetic-unit-secret',))
    assert cleaned['privacy'] == {'redacted': True, 'redactions': ['credential']}
    assert 'private-synthetic-unit-secret' not in json.dumps(cleaned)


@pytest.mark.parametrize('kind,marker,accepted', [('score',True,True),('score',False,False),('execute',True,False),('execute',False,False)])
def test_frozen_judge_only_and_zero_provider_calls(tmp_path,kind,marker,accepted):
    server,state,api = worker(tmp_path,kind=kind,marker=marker)
    reply = json.loads(server._handle_training_control(json.dumps(frame(event())).encode()))
    assert reply['recorded'] is accepted
    assert len(state.worker_observations.agent_observations) == int(accepted)
    assert api.provider_frames == []


@pytest.mark.parametrize('disposition', ['executed','repair','accepted_terminal','rejected_not_executed'])
def test_host_outcome_survives_without_subsequent_model_turn(tmp_path,disposition):
    server,state,api = worker(tmp_path)
    original = event(disposition=disposition)
    assert json.loads(server._handle_training_control(json.dumps(frame(original)).encode()))['recorded']
    assert state.worker_observations.agent_observations[0]['data'] == original
    assert len(state.calls) == len(api.provider_frames) == 0


def test_supported_unicode_result_exceeds_generic_state_limit_without_chunking(tmp_path):
    server,state,_ = worker(tmp_path)
    value = event(tool_name='fetch_page', result='界'*32000, disposition='executed')
    assert len(json.dumps(value,ensure_ascii=False).encode()) > checkpoint.MAX_TRAINING_STATE_BYTES
    assert json.loads(server._handle_training_control(json.dumps(frame(value),ensure_ascii=False).encode()))['recorded']
    assert state.worker_observations.agent_observations[0]['data']['result'] == value['result']
    generic = {**frame(value),'kind':'state_snapshot','data':{'name':'arbitrary','value':value}}
    with pytest.raises(ValueError): checkpoint.validate_training_frame(generic)


@pytest.mark.parametrize('damage',[{'invented': True},{'disposition':'correct_action'},{'sequence':True},{'request_sha256':'not-a-hash'},{'tool_name':''},{'host_validation':None}])
def test_closed_schema_rejects_invalid_or_gold_claims(damage):
    with pytest.raises(ValueError): checkpoint.validate_training_frame(frame(event(**damage)))


def test_generic_score_snapshot_still_rejected(tmp_path):
    server,state,_ = worker(tmp_path)
    value = {**frame(event()),'kind':'state_snapshot','data':{'name':'arbitrary','value':{}}}
    assert json.loads(server._handle_training_control(json.dumps(value).encode())) == {'recorded':False}
    assert state.worker_observations.agent_observations == []


def test_real_socket_callback_and_invalid_event_are_captured_without_provider(tmp_path,monkeypatch):
    short = tempfile.TemporaryDirectory(prefix='judge-tool-',dir='/tmp')
    server,state,api = worker(Path(short.name))
    server.start()
    try:
        monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV,str(Path(short.name)/'worker.sock'))
        monkeypatch.setenv(checkpoint.JUDGE_CAPTURE_POLICY_ENV,checkpoint.JUDGE_CAPTURE_POLICY)
        assert checkpoint.record_judge_tool_event(event()) == 0
        assert checkpoint.record_judge_tool_event(event(sequence=1, tool_name='fetch_page', disposition='executed', result='last fetched text')) == 1
        assert checkpoint.record_judge_tool_event(event(sequence=2, invented=True)) is None
        assert state.worker_observations.agent_observations[-1]['kind'] == 'capture_failure'
        assert 'capture_incomplete' in state.worker_observations.failures
        assert len(state.calls) == len(api.provider_frames) == 0
    finally:
        server.stop();server.wait_for_handlers();short.cleanup()


def test_callback_marker_off_never_opens_socket(monkeypatch):
    monkeypatch.delenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY',raising=False)
    monkeypatch.setattr(checkpoint.socket,'socket',lambda *a,**k: pytest.fail('socket opened without frozen marker'))
    assert checkpoint.record_judge_tool_event(event()) is None


def test_order_retry_idempotency_and_gap_are_explicit(tmp_path):
    server,state,_ = worker(tmp_path)
    first = {**frame(event()), 'retry_token':'a'*32}
    raw = json.dumps(first).encode()
    assert json.loads(server._handle_training_control(raw)) == {'recorded':True,'observation_id':0}
    assert json.loads(server._handle_training_control(raw)) == {'recorded':True,'observation_id':0}
    assert len(state.worker_observations.agent_observations) == 1
    gap = frame(event(sequence=2))
    assert json.loads(server._handle_training_control(json.dumps(gap).encode())) == {'recorded':False}
    assert 'capture_incomplete' in state.worker_observations.failures


def test_oversize_rejected_and_existing_frame_caps_preserved(monkeypatch):
    monkeypatch.setenv(checkpoint.JUDGE_CAPTURE_POLICY_ENV,checkpoint.JUDGE_CAPTURE_POLICY)
    value=event(result='x'*checkpoint.MAX_TRAINING_FRAME_BYTES)
    with pytest.raises(ValueError): checkpoint.validate_training_frame(frame(value))
    assert checkpoint.MAX_TRAINING_FRAME_BYTES == 8*1024*1024
    assert checkpoint.MAX_TRAINING_STATE_BYTES == 65536


def test_readable_trace_preserves_terminal_host_result_as_observation_not_target():
    from tests.lab_arena.judge_thinking_export_test import judge_example, records
    from tests.lab_arena.training_steps_test import seal
    example=judge_example(); before,_=records(example,role='judge')
    artifact=example['worker_observation_artifacts'][example['score_run_id']]
    rows=artifact['payload']['agent_observations']
    rows.append({'id':len(rows),'kind':'judge_local_tool','observed_at':'2026-11-05T00:00:00+00:00',
                 'after_action_sequence':0,'depends_on':[], 'data':event(),
                 'privacy':{'redacted':False,'redactions':[]}})
    artifact['payload_sha256']=contracts.document_hash(artifact['payload'])
    for call in example['provider_calls']:
        if call['run_id'] == example['score_run_id']:
            for ref in call['response_provenance']['agent_receipt_refs']+[call['response_provenance']['worker_receipt_ref']]:
                ref['payload_sha256']=artifact['payload_sha256']
    example=seal(example)
    after,_=records(example,role='judge')
    assert len(before)==len(after)==1
    assert before[0]['generated_thinking'] == after[0]['generated_thinking']
    assert after[0]['quality'] == {'status':'unreviewed','correct_action':'not_established'}
    report=training_thinking.readable_trace(example)
    local=[row for row in report['observations'] if row['kind']=='judge_local_tool']
    assert len(local)==1 and local[0]['value']['data']==event()
    assert local[0]['execution']=='reported_host_tool_validation_not_gold'
    assert local[0]['delivery']=='not_proof_of_subsequent_model_consumption'
    assert 'accepted_terminal' in training_thinking.readable_trace_markdown(report)
