"""Only a bound final broker failure can refine a generic scorer provider error."""
import base64
import json
import os

import pytest

from lab_arena import contracts, operations, runner, runtime, scoring, shim
from tests.lab_arena.test_lab_arena_runner import FakeApi, make_config, scoring_lease


def provider_response(sequence, *, success=False, outcome=None, identity=True, error='provider_unavailable'):
    raw = (json.dumps({'ok': True}).encode() if success else operations.GENERIC_UNAVAILABLE_BODY
           if error == 'provider_unavailable' else json.dumps({'error': {'code': error}}).encode())
    call = {'operation_id': 'openrouter.chat', 'provider': 'openrouter',
            'action_sequence': sequence, 'funding_source': 'host',
            'provider_status': 200 if success else 502,
            'outcome': outcome or ('settled' if success else 'uncertain')}
    if identity: call['call_identity'] = contracts.document_hash(['final-provider', sequence])
    if not success: call['error_code'] = error
    return {'status': 200 if success else 502, 'headers': {'content-type': 'application/json'},
            'body_b64': base64.b64encode(raw).decode(), 'call': call}


class ProviderApi(FakeApi):
    def __init__(self, lease, documents):
        super().__init__([lease])
        self.documents = documents

    def provider(self, run_id, token, frame):
        self.provider_frames.append(dict(frame))
        return self.documents[len(self.provider_frames) - 1]


class ProviderRuntime:
    def __init__(self, documents, output, *, timed_out=False):
        self.documents, self.output, self.timed_out = documents, output, timed_out

    def run_icp(self, spec, **_kwargs):
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        previous = {key: os.environ.get(key) for key in (shim.TRUSTED_SCORER_ENV, 'ARENA_JUDGE_REASONING_CAPTURE_POLICY')}
        os.environ[shim.TRUSTED_SCORER_ENV] = '1'
        os.environ['ARENA_JUDGE_REASONING_CAPTURE_POLICY'] = shim.FULL_THINKING_POLICY
        try:
            for document in self.documents:
                op = document['call']['operation_id']
                params = ({'model': 'openai/gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': 'Verify evidence.'}]} if op == 'openrouter.chat'
                    else {'tool': 'exa_search', 'payload': {}})
                shim.dispatch(op, params, 1000)
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            for key, value in previous.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
        raw = self.output if isinstance(self.output, bytes) else json.dumps(self.output).encode()
        return runtime.fake_result(exit_code=0, timed_out=self.timed_out, output_bytes=raw)


def run(tmp_path, documents, *, reason='provider_error', output=None, timed_out=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'work').mkdir()
    leased = scoring_lease()
    leased['judge_reasoning_capture_policy'] = shim.FULL_THINKING_POLICY
    output = output if output is not None else scoring.build_scoring_failure('r1', 'judge_error', reason=reason)
    api = ProviderApi(leased, documents)
    worker = runner.Runner(make_config(tmp_path, api, ProviderRuntime(documents, output, timed_out=timed_out)))
    try: assert worker.run_once() == 1
    finally: worker.close()
    assert len(api.provider_frames) == len(documents)
    result = api.completions[0]['body']['result']
    contracts.validate_run_result(result)
    return result


@pytest.mark.parametrize('outcome', ['uncertain', 'settled'])
def test_final_bound_actual_provider_failure_refines_generic_scorer_reason(tmp_path, outcome):
    result = run(tmp_path, [provider_response(0, outcome=outcome)])
    assert result['failure_diagnostic'] == {
        'stage': 'provider_call', 'error_class': 'provider_unavailable', 'reason': 'provider_error'}
    assert result['terminal_status'] == 'judge_error'
    assert result['resource_summary']['provider_unresolved_failure_count'] == 1


@pytest.mark.parametrize('fault', ['earlier_unrelated', 'healed', 'missing_identity', 'refused',
                                  'not_dispatched', 'unknown', 'broker_unavailable'])
def test_earlier_or_unbound_failure_does_not_claim_final_provider_cause(tmp_path, fault):
    documents = [provider_response(0)]
    if fault in ('earlier_unrelated', 'healed'):
        documents.append(provider_response(1, success=True))
        if fault == 'earlier_unrelated': documents[1]['call']['operation_id'] = 'deepline.execute'
    elif fault == 'missing_identity': documents = [provider_response(0, identity=False)]
    elif fault == 'broker_unavailable': documents = [provider_response(0, error='broker_unavailable')]
    else: documents = [provider_response(0, outcome=fault)]
    result = run(tmp_path, documents)
    assert result['failure_diagnostic'] == {'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'provider_error'}


@pytest.mark.parametrize('reason', sorted(scoring.FAILURE_REASON_CODES - {'provider_error'}))
def test_final_provider_failure_does_not_replace_specific_scorer_reason(tmp_path, reason):
    result = run(tmp_path, [provider_response(0)], reason=reason)
    assert result['failure_diagnostic'] == {'stage': 'scorer', 'error_class': 'judge_error', 'reason': reason}


def test_final_provider_failure_preserves_sandbox_timeout_diagnostic(tmp_path):
    result = run(tmp_path, [provider_response(0)], timed_out=True)
    assert result['failure_diagnostic'] == {'stage': 'sandbox', 'error_class': 'judge_timeout'}


def test_final_provider_failure_preserves_invalid_scoring_output(tmp_path):
    result = run(tmp_path, [provider_response(0)], output=b'not json')
    assert result['failure_diagnostic'] == {'stage': 'scoring_output', 'error_class': 'scoring_output_invalid'}


def test_accepted_output_is_not_reclassified_after_provider_failure(tmp_path):
    output = scoring.build_scoring_output('r1', [{'final_score': 50.0, 'failure_reason': ''}])
    result = run(tmp_path, [provider_response(0)], output=output)
    assert result['terminal_status'] == 'accepted' and 'failure_diagnostic' not in result


@pytest.mark.parametrize('fault', ['missing_ack', 'missing_receipt', 'socket_write_failed',
    'receipt_hash', 'receipt_bytes', 'receipt_status', 'ack_hash', 'ack_bytes', 'ack_status',
    'later_issued_success', 'later_completed_success', 'recovered'])
def test_final_provider_requires_exact_consumed_receipt_and_no_later_healing(fault):
    failed = provider_response(1)['call']
    receipt = {'call_identity': failed['call_identity'], 'operation_id': failed['operation_id'],
        'action_sequence': 1, 'write_state': 'socket_written', 'status': 502,
        'body_sha256': contracts.hash_bytes(operations.GENERIC_UNAVAILABLE_BODY),
        'body_bytes': len(operations.GENERIC_UNAVAILABLE_BODY)}
    data = {key: receipt[key] for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')}
    completed, latest, receipts, observations, recovered = [failed], [failed], [receipt], [{'kind': 'response_received', 'data': data}], set()
    if fault == 'missing_ack': observations.clear()
    elif fault == 'missing_receipt': receipts.clear()
    elif fault == 'socket_write_failed': receipt['write_state'] = 'socket_write_failed'
    elif fault.startswith('receipt_'):
        key = {'receipt_hash': 'body_sha256', 'receipt_bytes': 'body_bytes', 'receipt_status': 'status'}[fault]
        receipt[key] = 'sha256:' + 'f' * 64 if key == 'body_sha256' else 1
    elif fault.startswith('ack_'):
        key = {'ack_hash': 'body_sha256', 'ack_bytes': 'body_bytes', 'ack_status': 'status'}[fault]
        data[key] = 'sha256:' + 'f' * 64 if key == 'body_sha256' else 1
    elif fault == 'later_issued_success':
        success = provider_response(2, success=True)['call']
        latest.append(success); completed.insert(0, success)  # older failed call completed after newer success
    elif fault == 'later_completed_success':
        success = provider_response(0, success=True)['call']
        latest.append(success); completed.append(success)  # earlier call completed after the failed final issue
    else: recovered.add(failed['call_identity'])
    assert not runner._final_unresolved_provider_failure(completed, latest, recovered, receipts, observations)
