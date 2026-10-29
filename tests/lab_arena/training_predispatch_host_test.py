"""Proven pre-reservation host errors are observations, never paid ledger calls."""
import base64
from copy import deepcopy
from pathlib import Path
import tempfile
import threading

import pytest

from lab_arena import contracts, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_export_test import add_call, per_icp_cost_fixture, per_icp_cost_reader
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_observations_test import observation, save
from tests.lab_arena.training_optional_capture_test import summary


IDENTITY = contracts.hash_bytes(b'host-only admission timeout')
BODY = b'{"error":{"code":"provider_unavailable"}}'
FALSE_LINKS = {'worker_response_receipt_link_invalid', 'agent_response_receipt_link_invalid',
               'worker_call_accounting_unverified', 'provider_capture_count_unverified'}


def host_only_values(*, paid=True, privacy=False, legacy=False, acknowledged=True):
    values = per_icp_cost_fixture()
    cfg = values['round_row']['configuration_doc']
    native = without_client_callbacks()['configuration']
    cfg.update({key: deepcopy(native[key]) for key in ('reasoning_capture_policy', 'reasoning_route_registry')})
    cfg['training_minimum_icp_coverage_percent'] = 80
    if not legacy:
        cfg['training_maximum_missing_reasoning_percent'] = 1
    if paid:
        add_call(values)
    buffer = training_observations.Buffer()
    if paid:
        raw = base64.b64decode(values['ledger_rows'][-1]['terminal_response']['body_b64'])
        write = {'ordinal': 0, 'action_sequence': 0, 'operation_id': 'openrouter.chat',
            'call_identity': values['ledger_rows'][0]['call_identity'], 'protocol': 'http', 'status': 200,
            'body_sha256': contracts.hash_bytes(raw), 'body_bytes': len(raw), 'write_state': 'socket_written',
            'socket_written_at': '2026-10-28T00:00:00Z'}
        assert buffer.append_response_receipt(write)
    row = observation(kind='nonstandard_response', ordinal=5, action=5, identity=IDENTITY)
    row['request']['admitted_parameters'] = {'messages': [{'role': 'user', 'content': 'Synthetic request'}]}
    if privacy:
        row['request']['admitted_parameters']['api_key'] = '[REDACTED]'
        row['privacy'] = {'redacted': True, 'redactions': ['credential']}
    row['outcome'] = {'protocol': 'http', 'response': {'status': 502, 'headers': {},
        'body_b64': base64.b64encode(BODY).decode()},
        'broker_disposition': {'outcome': 'not_dispatched', 'reason': 'provider_admission_deadline'}}
    buffer.append(row)
    write = {'ordinal': 5, 'action_sequence': 5, 'operation_id': 'openrouter.chat', 'call_identity': IDENTITY,
        'protocol': 'http', 'status': 502, 'body_sha256': contracts.hash_bytes(BODY), 'body_bytes': len(BODY),
        'write_state': 'socket_written', 'socket_written_at': '2026-10-28T00:00:00Z'}
    assert buffer.append_response_receipt(write)
    ack = {'id': 0, 'kind': 'response_received', 'observed_at': '2026-10-28T00:00:00Z',
        'after_action_sequence': 5, 'depends_on': [], 'privacy': {'redacted': False, 'redactions': []},
        'data': {key: write[key] for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')}}
    if acknowledged:
        assert buffer.append_agent_observation(ack) == 0
    buffer.failures.add('capture_incomplete')
    optional = summary(); optional['id'] = int(acknowledged)
    assert buffer.append_agent_observation(optional) == int(acknowledged)
    values['runs'][0]['result_doc']['resource_summary']['provider_call_count'] = int(paid) + 1
    return values, buffer.document()


def export(values, payload):
    save(values, payload)
    return training_export.assemble_example(**values,
        cost_eligibility_reader=per_icp_cost_reader(spend=42, reason='eligible'))


@pytest.mark.parametrize('paid', [False, True])
@pytest.mark.parametrize('privacy', [False, True])
@pytest.mark.parametrize('acknowledged', [False, True])
def test_actual_export_classifies_exact_predispatch_host_error_without_a_paid_call(paid, privacy, acknowledged):
    values, payload = host_only_values(paid=paid, privacy=privacy, acknowledged=acknowledged)
    before = deepcopy(values['ledger_rows'])
    result = export(values, payload)
    reasons = set(result['completeness']['incomplete_reasons'])
    assert not reasons & FALSE_LINKS, reasons
    assert len(result['provider_calls']) == int(paid)
    assert result['costs']['confirmed_actual_microusd'] == (42 if paid else 0)
    assert values['ledger_rows'] == before
    assert result['worker_observation_artifacts'][values['runs'][0]['run_id']]['payload'] == payload
    assert 'worker_observations_incomplete' in reasons
    if privacy:
        assert 'worker_observation_privacy_nonexact' in reasons
    assert result['configuration']['training_minimum_icp_coverage_percent'] == 80
    assert result['configuration']['training_maximum_missing_reasoning_percent'] == 1
    # Existing generic/required privacy gaps remain strict; this fix grants no new allowance.
    assert training_thinking.selected_research_evidence(result) is None
    strict, _ = training_thinking.build([result], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('frontier_kind', ['local_catalog', 'local_refusal', 'response_receipt'])
def test_later_callback_frontier_requires_an_actual_host_action(frontier_kind):
    values, payload = host_only_values()
    payload['agent_observations'][0]['after_action_sequence'] = 7
    if frontier_kind == 'response_receipt':
        # A normal paid success is represented only by its host receipt.
        payload['response_receipts'][0]['action_sequence'] = 7
        for entry in values['ledger_rows']:
            entry['entry_doc']['action_sequence'] = 7
    else:
        row = observation(kind=frontier_kind, ordinal=7, action=7,
            operation='deepline.tools.list' if frontier_kind == 'local_catalog' else 'openrouter.chat',
            contributes=frontier_kind == 'local_catalog')
        payload['observations'].append(training_observations.clean_observation(row))
        payload['observation_count'] += 1
        if frontier_kind == 'local_refusal':
            payload['local_refusal_count'] += 1
            values['runs'][0]['result_doc']['resource_summary']['provider_local_refusal_count'] = 1
        else:
            values['runs'][0]['result_doc']['resource_summary']['provider_call_count'] += 1
    before = deepcopy(payload)
    result = export(values, payload)
    assert not set(result['completeness']['incomplete_reasons']) & FALSE_LINKS
    assert result['worker_observation_artifacts']['execute:1']['payload'] == before
    assert result['costs']['confirmed_actual_microusd'] == 42


@pytest.mark.parametrize('frontier', [4, None, True, 7.0, '7', -1, 6, 100])
def test_callback_cannot_prove_a_premature_or_unwitnessed_frontier(frontier):
    values, payload = host_only_values()
    payload['agent_observations'][0]['after_action_sequence'] = frontier
    if frontier is not None and (type(frontier) is not int or frontier < 0):
        with pytest.raises(training_observations.ObservationError):
            export(values, payload)
    else:
        result = export(values, payload)
        assert set(result['completeness']['incomplete_reasons']) & FALSE_LINKS


def test_runner_parallel_callbacks_use_global_frontier_and_export_without_false_links(monkeypatch):
    from lab_arena import lab_arena_checkpoint as checkpoint, runner, shim
    from tests.lab_arena.test_lab_arena_runner import lease
    from tests.lab_arena.worker_final_capture_drain_test import _request_socket
    from tests.lab_arena.training_worker_capture_v2_test import _pre_reservation_document

    class ParallelApi:
        def __init__(self):
            self.condition = threading.Condition()
            self.frames = []
            self.release = threading.Event()

        def provider(self, _run_id, _token, frame):
            with self.condition:
                self.frames.append(deepcopy(frame))
                self.condition.notify_all()
            assert self.release.wait(5)
            document = _pre_reservation_document('provider_admission_deadline')
            document['call'].update(action_sequence=frame['action_sequence'],
                call_identity=contracts.document_hash(['parallel local error', frame['action_sequence']]))
            return document

    api = ParallelApi()
    state = runner.RunState(lease=lease(), lease_token='local-test-token')
    with tempfile.TemporaryDirectory(prefix='ar-parallel-', dir='/tmp') as directory:
        socket_path = Path(directory) / 'worker.sock'
        server = runner.WorkerSocketServer(socket_path, api, state)
        server.start()
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
        connections = []
        try:
            for number in range(4):
                connections.append(_request_socket(socket_path))
                with api.condition:
                    assert api.condition.wait_for(lambda: len(api.frames) == number + 1, timeout=2)
            api.release.set()
            for connection in connections:
                response = connection.getresponse()
                raw = response.read()
                assert response.status == 502 and raw == BODY
                assert checkpoint.record_response_received(
                    call_identity=response.getheader(shim.CALL_IDENTITY_HEADER), status=502, body=raw) is not None
        finally:
            api.release.set()
            for connection in connections:
                connection.close()
            server.stop()
    payload = state.worker_observations.document()
    assert state.action_sequence == 4 and len(state.calls) == 4
    assert [row['action_sequence'] for row in payload['response_receipts']] == list(range(4))
    assert [row['after_action_sequence'] for row in payload['agent_observations']] == [3] * 4
    values, _ = host_only_values(paid=False, acknowledged=False)
    values['runs'][0]['result_doc']['resource_summary']['provider_call_count'] = 4
    before = deepcopy(payload)
    result = export(values, payload)
    assert not set(result['completeness']['incomplete_reasons']) & FALSE_LINKS
    assert result['provider_calls'] == [] and result['costs']['confirmed_actual_microusd'] == 0
    assert result['worker_observation_artifacts']['execute:1']['payload'] == before
    readable = training_thinking.readable_trace(result)
    assert [row['value'] for row in readable['observations'] if row['kind'] == 'response_received'] == payload['agent_observations']
    strict, _ = training_thinking.build([result], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('damage', ['missing_tag', 'missing_observation', 'ack_hash',
    'ack_status', 'ack_sequence', 'ack_privacy', 'contradictory_ack', 'duplicate_identity', 'duplicate_sequence',
    'body_hash', 'body_bytes', 'body_status', 'write_failed', 'operation', 'ordinal', 'protocol',
    'paid_sequence', 'ledger_identity', 'legacy'])
def test_unexplained_or_contradictory_host_only_event_stays_a_hard_export_gap(damage, parallel):
    values, payload = host_only_values()
    row = payload['observations'][0]
    write = payload['response_receipts'][-1]
    ack = payload['agent_observations'][0]
    if parallel:
        later = observation(kind='local_catalog', ordinal=7, action=7, operation='deepline.tools.list')
        payload['observations'].append(training_observations.clean_observation(later))
        payload['observation_count'] += 1
        values['runs'][0]['result_doc']['resource_summary']['provider_call_count'] += 1
        ack['after_action_sequence'] = 7
    wrong = contracts.hash_bytes(b'wrong')
    if damage == 'missing_tag': row['outcome'].pop('broker_disposition')
    elif damage == 'missing_observation': payload['observations'] = []; payload['observation_count'] = 0
    elif damage == 'ack_hash': ack['data']['body_sha256'] = wrong
    elif damage == 'ack_status': ack['data']['status'] = 503
    elif damage == 'ack_sequence': ack['after_action_sequence'] += 1
    elif damage == 'ack_privacy': ack['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'contradictory_ack':
        copied = deepcopy(ack); copied['id'] = 2; copied['data']['body_sha256'] = wrong
        payload['agent_observations'].append(copied); payload['agent_observation_count'] += 1
    elif damage in ('duplicate_identity', 'duplicate_sequence'):
        copied = deepcopy(row); copied['ordinal'] = 6
        if damage == 'duplicate_identity': copied['action_sequence'] = 6
        else: copied['call_identity'] = wrong
        payload['observations'].append(copied); payload['observation_count'] += 1
    elif damage == 'body_hash': write['body_sha256'] = wrong
    elif damage == 'body_bytes': write['body_bytes'] += 1
    elif damage == 'body_status': write['status'] = 503
    elif damage == 'write_failed': write.update(write_state='socket_write_failed', socket_written_at=None)
    elif damage == 'operation': write['operation_id'] = 'openrouter.responses'
    elif damage == 'ordinal': write['ordinal'] += 1
    elif damage == 'protocol': write['protocol'] = 'frame'
    elif damage == 'paid_sequence': row['action_sequence'] = write['action_sequence'] = ack['after_action_sequence'] = 0
    elif damage == 'ledger_identity':
        # An unassembled row is still accounting evidence; the tag cannot erase it.
        shadow = deepcopy(values['ledger_rows'][0]); shadow.update(entry_id=100, run_id='unrelated', call_identity=IDENTITY)
        values['ledger_rows'].append(shadow)
    else: values['round_row']['configuration_doc'].pop('training_maximum_missing_reasoning_percent')
    payload['observations'].sort(key=lambda item: item['ordinal'])
    if damage in {'duplicate_sequence', 'body_hash', 'body_bytes', 'body_status', 'operation', 'protocol'}:
        with pytest.raises(training_observations.ObservationError):
            export(values, payload)
        return
    result = export(values, payload)
    assert set(result['completeness']['incomplete_reasons']) & FALSE_LINKS, damage


@pytest.mark.parametrize('collision', ['missing_ledger', 'identity', 'base_identity', 'nested_base', 'sequence', 'nested_sequence'])
def test_raw_ledger_absence_is_required_even_for_unassembled_rows(collision):
    values, payload = host_only_values(paid=False)
    save(values, payload)
    run = values['runs'][0]
    snapshot = training_observations.read_snapshot(values['objects'], run['run_id'], run['kind'],
        run.get('lease_generation'), run.get('claim_request_hash'))
    ledger = []
    if collision == 'missing_ledger': ledger = None
    else:
        row = {'run_id': 'foreign', 'call_identity': contracts.hash_bytes(b'other'), 'entry_doc': {}}
        if collision == 'identity': row['call_identity'] = IDENTITY
        elif collision == 'base_identity': row['entry_doc']['base_call_identity'] = IDENTITY
        elif collision == 'nested_base': row['entry_doc']['call'] = {'base_call_identity': IDENTITY}
        elif collision == 'sequence': row.update(run_id=run['run_id'], entry_doc={'action_sequence': 5})
        else: row.update(run_id=run['run_id'], entry_doc={'call': {'action_sequence': 5}})
        ledger.append(row)
    assert not training_export._not_dispatched_host_observations(
        run, [], snapshot, values['round_row']['configuration_doc'], ledger)


def gate_example(*, mode='gate', privacy=False, acknowledged=False, complete=False):
    from tests.lab_arena.training_composed_capture_gaps_test import composed
    from tests.lab_arena.training_failed_delivery_coverage_test import failed_delivery, refresh
    example = composed('triple' if mode == 'triple' else 'delivery') if mode in ('cost', 'triple') else failed_delivery()
    if mode in ('gate', 'cost'):
        for write in example['worker_observation_artifacts']['run']['payload']['response_receipts']:
            write.update(write_state='socket_written', socket_written_at='2026-10-28T00:00:00Z')
    artifact = example['worker_observation_artifacts']['run']
    payload = artifact['payload']
    if complete:
        payload.update(complete=True, failures=[])
        example['completeness'] = {'complete': True, 'incomplete_reasons': [],
            'dimensions': {key: True for key in example['completeness']['dimensions']},
            'reasons_by_dimension': {key: [] for key in example['completeness']['dimensions']}}
        example['related_run_gap_attribution']['run'].update(research_process_gaps=[], cost_gaps=[])
    _, local = host_only_values(paid=False, privacy=privacy, acknowledged=acknowledged)
    row = deepcopy(local['observations'][0]); row.update(ordinal=1000, action_sequence=1000)
    write = deepcopy(local['response_receipts'][0]); write.update(ordinal=1000, action_sequence=1000)
    payload['observations'].append(row)
    payload['response_receipts'].append(write)
    payload['observation_count'] += 1
    payload['response_receipt_count'] += 1
    if acknowledged:
        ack = deepcopy(local['agent_observations'][0]); ack.update(id=len(payload['agent_observations']), after_action_sequence=1000)
        payload['agent_observations'].append(ack); payload['agent_observation_count'] += 1
    if privacy:
        for reasons in (example['completeness']['incomplete_reasons'],
                example['completeness']['reasons_by_dimension']['research_process'],
                example['related_run_gap_attribution']['run']['research_process_gaps']):
            reasons.append('worker_observation_privacy_nonexact')
    refresh(example)
    ledger = [{'run_id': call['run_id'], 'call_identity': call['call_identity'], 'entry_doc': entry.get('accounting') or {}}
              for call in example['provider_calls'] for entry in call['lifecycle']]
    assert not training_export._link_worker_receipts({'run_id': 'run', 'kind': 'execute'}, example['provider_calls'],
        artifact, require_client_ack=False, configuration=example['configuration'], ledger_rows=ledger)
    refresh(example)
    return example


@pytest.mark.parametrize('damage', [None, 'premature', 'future', 'unwitnessed_gap', 'callback_only', 'foreign_run', 'hash', 'status', 'bytes', 'privacy'])
def test_parallel_callback_frontier_composes_with_frozen_gate_and_remains_fail_closed(damage):
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    from tests.lab_arena.training_failed_delivery_coverage_test import refresh
    example = gate_example(mode='triple', acknowledged=True)
    payload = example['worker_observation_artifacts']['run']['payload']
    ack = payload['agent_observations'][-1]
    later = observation(kind='local_catalog', ordinal=1001, action=1001, operation='deepline.tools.list')
    payload['observations'].append(training_observations.clean_observation(later))
    payload['observation_count'] += 1
    ack['after_action_sequence'] = 1001
    if damage == 'premature': ack['after_action_sequence'] = 999
    elif damage == 'future': ack['after_action_sequence'] = 1002
    elif damage == 'unwitnessed_gap': payload['observations'][-1]['action_sequence'] = 1002
    elif damage in {'callback_only', 'foreign_run'}:
        payload['observations'].pop(); payload['observation_count'] -= 1
        if damage == 'foreign_run':
            extra = deepcopy(example['provider_calls'][0])
            extra.update(run_id='foreign', call_identity=contracts.hash_bytes(b'foreign run'), action_sequence=1001)
            example['provider_calls'].append(extra)
    elif damage == 'hash': ack['data']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'status': ack['data']['status'] = 503
    elif damage == 'bytes': ack['data']['body_bytes'] += 1
    elif damage == 'privacy': ack['privacy'] = {'redacted': True, 'redactions': ['credential']}
    refresh(example)
    examples = frozen_bank(example)
    before = deepcopy(example)
    result = assess(examples)
    assert result['eligible'] is (damage is None), result
    assert example == before
    if damage is None:
        assert result['counts']['minimum_coverage_percent'] == 80
        assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
        assert example['configuration']['training_maximum_missing_reasoning_percent'] == 1
        assert training_thinking.selected_research_evidence(example) is None


@pytest.mark.parametrize('mode', ['gate', 'cost', 'delivery', 'triple'])
@pytest.mark.parametrize('privacy', [False, True])
@pytest.mark.parametrize('acknowledged', [False, True])
def test_exact_predispatch_error_composes_with_existing_incomplete_icp_allowance(mode, privacy, acknowledged):
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    from lab_arena import training_eligibility
    example = gate_example(mode=mode, privacy=privacy, acknowledged=acknowledged)
    examples = frozen_bank(example)
    before = deepcopy(example)
    result = assess(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    assert training_eligibility._failed_delivery_gap_verified(example) is not None
    assert training_thinking.selected_research_evidence(example) is None
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    assert example == before
    examples['run-8'] = None
    assert assess(examples)['eligible']
    examples['run-7'] = None
    assert 'training_coverage_incomplete' in assess(examples)['reasons']


def test_exact_complete_capture_keeps_paid_targets_and_creates_no_target_for_local_error():
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    example = gate_example(complete=True)
    result = assess(frozen_bank(example))
    assert result['eligible'] and result['counts']['complete_runs'] == 10, result
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert strict['train'] or strict['eval']
    assert all(ref['call_identity'] != IDENTITY for split in ('train', 'eval')
               for candidate in strict[split] for ref in candidate['source_refs'])


@pytest.mark.parametrize('damage', ['host_hash', 'body', 'missing_tag', 'ledger_collision', 'extra_private_observation',
    'ack_hash', 'source', 'input', 'cost_authority', 'route', 'unknown_reason', 'legacy', 'counter'])
def test_predispatch_coverage_requires_every_existing_constituent_proof(damage):
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    from tests.lab_arena.training_failed_delivery_coverage_test import refresh
    example = gate_example(mode='triple', privacy=True, acknowledged=True)
    examples = frozen_bank(example)
    payload = example['worker_observation_artifacts']['run']['payload']
    row = payload['observations'][-1]; write = payload['response_receipts'][-1]
    wrong = contracts.hash_bytes(b'wrong')
    if damage == 'host_hash': write['body_sha256'] = wrong
    elif damage == 'body': row['outcome']['response']['body_b64'] = base64.b64encode(b'{}').decode()
    elif damage == 'missing_tag': row['outcome'].pop('broker_disposition')
    elif damage == 'ledger_collision': example['provider_calls'][0]['lifecycle'][0].setdefault('accounting', {})['base_call_identity'] = IDENTITY
    elif damage == 'extra_private_observation':
        extra = deepcopy(row); extra.update(ordinal=1001, action_sequence=1001, call_identity=wrong)
        extra['outcome'].pop('broker_disposition')
        payload['observations'].append(extra); payload['observation_count'] += 1
    elif damage == 'ack_hash': payload['agent_observations'][-1]['data']['body_sha256'] = wrong
    elif damage == 'source': example['source']['sha256'] = wrong
    elif damage == 'input': example['input_snapshots'][example['score_run_id']]['input_sha256'] = wrong
    elif damage == 'cost_authority': example['competition_cost_eligibility']['source_rpc'] = 'untrusted'
    elif damage == 'route': example['provider_calls'][2]['reasoning_capture']['provider_route'] = 'wrong'
    elif damage == 'unknown_reason': example['completeness']['incomplete_reasons'].append('unknown_failure')
    elif damage == 'counter': payload['response_receipt_count'] += 1
    else: example['configuration'].pop('training_maximum_missing_reasoning_percent')
    refresh(example)
    assert not assess(examples)['eligible'], damage


@pytest.mark.parametrize('damage', ['foreign_ack', 'unknown_dependency'])
def test_existing_linker_rejects_unbound_callbacks_in_gate_coverage_without_mutation(damage):
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    from tests.lab_arena.training_failed_delivery_coverage_test import refresh
    example = gate_example(acknowledged=True)
    payload = example['worker_observation_artifacts']['run']['payload']
    if damage == 'foreign_ack':
        extra = deepcopy(payload['agent_observations'][-1]); extra['id'] += 1
        extra['data']['call_identity'] = contracts.hash_bytes(b'unknown callback identity')
        payload['agent_observations'].append(extra); payload['agent_observation_count'] += 1
    else:
        payload['agent_observations'][-1]['depends_on'] = [contracts.hash_bytes(b'unknown dependency')]
    refresh(example)
    examples = frozen_bank(example)
    before = deepcopy(example)
    assert not assess(examples)['eligible']
    assert example == before


@pytest.mark.parametrize('foreign_callback', [False, True])
@pytest.mark.parametrize('privacy', [False, True])
def test_linker_recheck_preserves_only_already_proven_optional_binary_diagnostic(foreign_callback, privacy):
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    from tests.lab_arena.training_failed_delivery_coverage_test import refresh
    example = gate_example(privacy=privacy)
    payload = example['worker_observation_artifacts']['run']['payload']
    raw = b'%PDF-1.7 synthetic optional audit'
    digest = contracts.hash_bytes(raw)
    payload['agent_observations'].append({
        'id': 0, 'kind': 'web_fetch_binary', 'observed_at': '2026-10-28T00:00:00Z',
        'after_action_sequence': 0, 'depends_on': [], 'privacy': {'redacted': False, 'redactions': []},
        'data': {'url': 'https://example.com/a.pdf', 'final_url': 'https://example.com/a.pdf',
            'status': 200, 'retrieved_at': '2026-10-28T00:00:00Z', 'content_type': 'application/pdf',
            'body_b64': base64.b64encode(raw).decode(), 'body_ref': None,
            'body_sha256': digest, 'body_bytes': len(raw), 'stored_body_sha256': None, 'stored_body_bytes': None,
            'privacy_review': 'unreviewed_binary', 'capture': {'source': 'runtime_http_v1', 'complete': True,
                'received_body_sha256': digest, 'received_body_bytes': len(raw)}}})
    payload['agent_observations'][-1] = training_observations.clean_agent_observation(payload['agent_observations'][-1])
    payload['agent_observation_count'] += 1
    reason = 'binary_privacy_review_required'
    for reasons in (example['completeness']['incomplete_reasons'],
            example['completeness']['reasons_by_dimension']['research_process'],
            example['related_run_gap_attribution']['run']['research_process_gaps']):
        reasons.append(reason)
    if foreign_callback:
        write = payload['response_receipts'][-1]
        payload['agent_observations'].append({'id': 1, 'kind': 'response_received',
            'observed_at': '2026-10-28T00:00:00Z', 'after_action_sequence': 1000, 'depends_on': [],
            'privacy': {'redacted': False, 'redactions': []}, 'data': {
                **{key: write[key] for key in ('status', 'body_sha256', 'body_bytes')},
                'call_identity': contracts.hash_bytes(b'unknown callback')}})
        payload['agent_observation_count'] += 1
    refresh(example)
    before = deepcopy(example)
    result = assess(frozen_bank(example))
    assert result['eligible'] is (not foreign_callback), result
    assert example['worker_observation_artifacts'] == before['worker_observation_artifacts']
    assert reason in example['completeness']['incomplete_reasons']


@pytest.mark.parametrize('damage', [None, 'foreign_callback', 'unknown_dependency', 'other_required_privacy'])
def test_proven_local_privacy_preserves_known_web_diagnostic_and_required_privacy(damage):
    from tests.lab_arena.training_composed_capture_gaps_test import assess, frozen_bank
    from tests.lab_arena.training_failed_delivery_coverage_test import refresh
    example = gate_example(mode='triple', privacy=True, acknowledged=True)
    payload = example['worker_observation_artifacts']['run']['payload']
    reason = 'direct_web_coverage_unverified'
    for reasons in (example['completeness']['incomplete_reasons'],
            example['completeness']['reasons_by_dimension']['research_process'],
            example['related_run_gap_attribution']['run']['research_process_gaps']):
        reasons.append(reason)
    # Optional redacted text must not erase the required local privacy reason.
    from tests.lab_arena.training_optional_capture_test import summary
    optional = summary(); optional.update(id=1, privacy={'redacted': True, 'redactions': ['credential']})
    payload['agent_observations'].append(optional); payload['agent_observation_count'] += 1
    payload['failures'].append(training_observations.OPTIONAL_CAPTURE_INCOMPLETE)
    if damage == 'foreign_callback':
        extra = deepcopy(payload['agent_observations'][0]); extra['id'] = 2
        extra['data']['call_identity'] = contracts.hash_bytes(b'unknown callback')
        payload['agent_observations'].append(extra); payload['agent_observation_count'] += 1
    elif damage == 'unknown_dependency': payload['agent_observations'][0]['depends_on'] = [contracts.hash_bytes(b'unknown dependency')]
    elif damage == 'other_required_privacy':
        extra = deepcopy(payload['observations'][-1]); extra.update(ordinal=1001, action_sequence=1001, call_identity=contracts.hash_bytes(b'unproven'))
        extra['outcome'].pop('broker_disposition')
        payload['observations'].append(extra); payload['observation_count'] += 1
    refresh(example)
    examples = frozen_bank(example); before = deepcopy(example)
    result = assess(examples)
    assert result['eligible'] is (damage is None), result
    assert example == before
    assert 'worker_observation_privacy_nonexact' in example['completeness']['incomplete_reasons']
    assert training_thinking.selected_research_evidence(example) is None
