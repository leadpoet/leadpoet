"""Assessments prove retained worker data without rehashing it per call."""

from copy import deepcopy
import gc
import json
import weakref

import pytest

from lab_arena import contracts, training_eligibility, training_observations, training_steps, training_worker_proof
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_steps_test import seal


def large_example(count=64):
    example = without_client_callbacks()
    base = example['provider_calls'][1]
    source_write = example['worker_observation_artifacts']['run']['payload']['response_receipts'][1]
    calls, writes = [], []
    for index in range(count):
        call, write = deepcopy(base), deepcopy(source_write)
        identity = contracts.hash_bytes(('worker-proof-' + str(index)).encode())
        terminal = 3 * index + 3
        call.update(call_identity=identity, base_call_identity=identity, action_sequence=index,
                    reservation_entry_id=terminal - 2, terminal_entry_id=terminal)
        for offset, row in enumerate(call['lifecycle']):
            row['entry_id'] = terminal - 2 + offset
        call['lifecycle'][0]['accounting'].update(action_sequence=index, base_call_identity=identity)
        call['response_receipts'][0].update(entry_id=terminal,
                                          response_ref={'field': 'response', 'entry_id': terminal})
        provenance = call['response_provenance']
        provenance['worker_received_receipt_entry_ids'] = [terminal]
        provenance['worker_receipt_ref']['ordinal'] = index
        write.update(call_identity=identity, action_sequence=index, ordinal=index,
                     protocol='frame', socket_written_at='2026-10-28T00:00:00Z')
        calls.append(call)
        writes.append(write)
    payload = {'response_receipts': writes, 'response_receipt_count': len(writes),
               'observations': [], 'observation_count': 0, 'local_refusal_count': 0,
               'complete': True, 'failures': [], 'agent_observation_count': 1,
               'agent_observations': [{
        'kind': 'state_snapshot', 'id': 0, 'after_action_sequence': 0, 'depends_on': [],
        'observed_at': '2026-10-28T00:00:00Z', 'privacy': {'redacted': False, 'redactions': []},
        'data': {'name': 'synthetic_page', 'value': 'Synthetic evidence. ' * 2000}}]}
    digest = contracts.document_hash(payload)
    for call in calls:
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = digest
    example['provider_calls'] = calls
    example['worker_observation_artifacts']['run'] = {
        'schema_version': training_observations.SCHEMA_VERSION, 'run_id': 'run', 'kind': 'execute',
        'lease_generation': 1, 'claim_request_hash': contracts.hash_bytes(b'proof-claim'),
        'payload': payload, 'payload_sha256': digest}
    return seal(example)


def assess(example):
    return training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)


def test_scaling_fixture_has_a_complete_valid_v2_worker_payload():
    example = large_example()
    payload = example['worker_observation_artifacts']['run']['payload']
    assert training_observations._validate_payload(payload) == payload
    assert payload['complete'] and not payload['failures']


def test_assessment_worker_validation_work_is_bounded_by_snapshots_not_calls(monkeypatch):
    example = large_example()
    before = deepcopy(example)
    original = contracts.document_hash
    worker_hashes = []

    def counted(value):
        if isinstance(value, dict) and 'response_receipts' in value and 'agent_observations' in value:
            worker_hashes.append(len(value['response_receipts']))
        return original(value)

    monkeypatch.setattr(contracts, 'document_hash', counted)
    result = assess(example)
    assert result['eligible'], result
    assert result['counts']['full_thinking_calls'] == 64
    assert len(worker_hashes) <= 6, worker_hashes
    assert example == before


@pytest.mark.parametrize('mutate', [
    lambda p: p.__init__({'changed': True}),
    lambda p: p.__setitem__('changed', True),
    lambda p: p.__delitem__('response_receipts'),
    lambda p: p.clear(), lambda p: p.pop('response_receipts'), lambda p: p.popitem(),
    lambda p: p.setdefault('changed', True), lambda p: p.update(changed=True),
    lambda p: p.__ior__({'changed': True}),
    lambda p: p['response_receipts'].__init__([]),
    lambda p: p['response_receipts'].__setitem__(0, {}),
    lambda p: p['response_receipts'].__setitem__(slice(None), []),
    lambda p: p['response_receipts'].__delitem__(0),
    lambda p: p['response_receipts'].__delitem__(slice(None)),
    lambda p: p['response_receipts'].append({}), lambda p: p['response_receipts'].clear(),
    lambda p: p['response_receipts'].extend([{}]), lambda p: p['response_receipts'].insert(0, {}),
    lambda p: p['response_receipts'].pop(), lambda p: p['response_receipts'].remove(p['response_receipts'][0]),
    lambda p: p['response_receipts'].reverse(), lambda p: p['response_receipts'].sort(),
    lambda p: p['response_receipts'].__iadd__([{}]), lambda p: p['response_receipts'].__imul__(2),
    lambda p: p['response_receipts'][0].update(body_bytes=0),
])
def test_owned_worker_payload_cannot_be_mutated_or_reinitialized(mutate):
    example = large_example(2)
    owner = training_worker_proof.WorkerProofs()
    view = owner.prepare(example)
    payload = view['worker_observation_artifacts']['run']['payload']
    before = contracts.canonical_json(payload)
    with pytest.raises(TypeError, match='read-only'):
        mutate(payload)
    assert contracts.canonical_json(payload) == before
    assert before == contracts.canonical_json(example['worker_observation_artifacts']['run']['payload'])
    assert owner.payload_hash(payload) == contracts.document_hash(payload)
    assert deepcopy(payload) is payload


class MutableDict(dict):
    pass


class MutableList(list):
    pass


@pytest.mark.parametrize('path', [(), ('response_receipts',), ('response_receipts', 0),
                                 ('agent_observations', 0, 'depends_on')])
@pytest.mark.parametrize('operation', ['class_assignment', 'attribute_assignment', 'attribute_deletion'])
def test_owned_containers_cannot_replace_their_class_or_attributes(path, operation):
    owner = training_worker_proof.WorkerProofs()
    view = owner.prepare(large_example(2))
    payload = view['worker_observation_artifacts']['run']['payload']
    node = payload
    for key in path:
        node = node[key]
    before = contracts.canonical_json(payload)
    with pytest.raises(TypeError, match='read-only'):
        if operation == 'class_assignment':
            node.__class__ = MutableDict if isinstance(node, dict) else MutableList
        elif operation == 'attribute_assignment':
            node.extra_attribute = 'must not be retained'
        else:
            del node.__dict__
    assert contracts.canonical_json(payload) == before
    assert owner.payload_hash(payload) == contracts.document_hash(payload)


def test_context_holds_exact_objects_and_releases_them_with_its_lifetime():
    owner = training_worker_proof.WorkerProofs()
    view = owner.prepare(large_example(2))
    payload = view['worker_observation_artifacts']['run']['payload']
    reference = weakref.ref(payload)
    del payload, view
    gc.collect()
    assert reference() is not None
    del owner
    gc.collect()
    assert reference() is None


def test_original_foreign_and_mutable_view_payloads_keep_stateless_hashing(monkeypatch):
    example = large_example(2)
    first, second = training_worker_proof.WorkerProofs(), training_worker_proof.WorkerProofs()
    view, other_view = first.prepare(example), second.prepare(example)
    owned = view['worker_observation_artifacts']['run']['payload']
    foreign = other_view['worker_observation_artifacts']['run']['payload']
    assert owned is not foreign
    original = example['worker_observation_artifacts']['run']['payload']
    mutable = json.loads(contracts.canonical_json(owned))
    actual_hash = contracts.document_hash
    calls = []

    def counted(value):
        calls.append(value)
        return actual_hash(value)

    monkeypatch.setattr(contracts, 'document_hash', counted)
    for payload in (original, foreign, mutable):
        assert first.payload_hash(payload) == actual_hash(payload)
        assert first.payload_hash(payload) == actual_hash(payload)
    assert len(calls) == 6
    assert first.payload_hash(owned) == actual_hash(owned)
    assert len(calls) == 6
    mutable['response_receipts'][0]['body_bytes'] += 1
    assert first.payload_hash(mutable) != first.payload_hash(owned)
    original['response_receipts'][0]['body_bytes'] += 1
    assert first.payload_hash(owned) == actual_hash(owned)


@pytest.mark.parametrize('damage', ['recorded_hash', 'nested_write', 'missing_write'])
def test_fresh_assessment_never_reuses_a_previous_or_supplied_proof(damage):
    example = large_example(2)
    assert assess(example)['eligible']
    artifact = example['worker_observation_artifacts']['run']
    if damage == 'recorded_hash':
        artifact['payload_sha256'] = contracts.hash_bytes(b'forged')
    elif damage == 'nested_write':
        artifact['payload']['response_receipts'][0]['body_bytes'] += 1
    else:
        artifact['payload']['response_receipts'].pop()
    seal(example)
    assert not assess(example)['eligible']


def test_changing_an_artifact_header_cannot_use_the_owned_payload_as_a_proof_flag():
    example = large_example(2)
    owner = training_worker_proof.WorkerProofs()
    view = owner.prepare(example)
    view['worker_observation_artifacts']['run']['payload_sha256'] = contracts.hash_bytes(b'forged')
    with pytest.raises(ValueError, match='worker_payload_hash_mismatch'):
        training_steps._host_result(view, view['provider_calls'][0], worker_proofs=owner)


def test_standalone_calls_revalidate_mutable_payloads_after_each_change():
    example = large_example(2)
    call = example['provider_calls'][0]
    assert training_steps._host_result(example, call)
    example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['body_bytes'] += 1
    with pytest.raises(ValueError, match='worker_payload_hash_mismatch'):
        training_steps._host_result(example, call)


@pytest.mark.parametrize('variant', ['full_v2', 'legacy', 'optional', 'failed_tool',
                                    'failed_model', 'judge_privacy', 'judge_cost',
                                    'selected_privacy', 'undispatched'])
def test_owned_and_stateless_assessments_have_identical_results_and_flags(monkeypatch, variant):
    from tests.lab_arena.training_eligibility_test import (
        captured_example, judge_only_gap_example, selected_privacy_cascade_example)
    from tests.lab_arena.training_optional_capture_test import captured_timeout
    from tests.lab_arena.training_failed_delivery_coverage_test import failed_delivery
    from tests.lab_arena.training_predispatch_host_test import gate_example

    factories = {'full_v2': large_example, 'legacy': captured_example, 'optional': captured_timeout,
                 'failed_tool': failed_delivery, 'failed_model': lambda: failed_delivery(model=True),
                 'judge_privacy': judge_only_gap_example,
                 'judge_cost': lambda: judge_only_gap_example(cost_gap=True),
                 'selected_privacy': selected_privacy_cascade_example,
                 'undispatched': gate_example}
    example = factories[variant]()
    before = deepcopy(example)
    result = assess(example)

    class Stateless:
        def prepare(self, value):
            return value

        def payload_hash(self, value):
            return contracts.document_hash(value)

        def exact_privacy(self, value, *, response, check):
            return check(value, response=response)

    monkeypatch.setattr(training_worker_proof, 'WorkerProofs', Stateless)
    assert assess(example) == result
    assert example == before
    assert contracts.document_hash(example) == contracts.document_hash(before)
