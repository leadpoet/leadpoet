"""Journal payload ownership preserves partial authority and exact native outputs."""
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
import weakref
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking, training_worker_proof
from tests.lab_arena.training_journal_candidates_test import journal_values
from tests.lab_arena.training_steps_test import seal


PARTIAL = 'partial_worker_observation_artifacts'
FINAL = 'worker_observation_artifacts'
UNVERIFIED = 'unverified_worker_observation_artifacts'


def partial_example(count=8):
    values, _ = journal_values()
    example = training_export.assemble_example(**values)
    rid = example['run_id']
    artifact = example[PARTIAL][rid]
    base_call = example['provider_calls'][0]
    base_write = artifact['payload']['response_receipts'][0]
    calls, writes = [], []
    for index in range(count):
        call, write = deepcopy(base_call), deepcopy(base_write)
        identity = contracts.hash_bytes(('journal-proof-' + str(index)).encode())
        terminal = 3 * index + 3
        call.update(call_identity=identity, base_call_identity=identity, action_sequence=index,
                    reservation_entry_id=terminal-2, terminal_entry_id=terminal)
        for offset, lifecycle in enumerate(call['lifecycle']):
            lifecycle['entry_id'] = terminal-2+offset
        call['lifecycle'][0]['accounting'].update(action_sequence=index, base_call_identity=identity)
        call['response_receipts'][0].update(entry_id=terminal,
                                          response_ref={'field':'response','entry_id':terminal})
        call['response_provenance']['worker_received_receipt_entry_ids'] = [terminal]
        call['response_provenance']['worker_receipt_ref']['ordinal'] = index
        write.update(call_identity=identity, action_sequence=index, ordinal=index)
        calls.append(call); writes.append(write)
    payload = artifact['payload']
    payload['response_receipts'] = writes; payload['response_receipt_count'] = count
    digest = contracts.document_hash(payload); artifact['payload_sha256'] = digest
    for call in calls: call['response_provenance']['worker_receipt_ref']['payload_sha256'] = digest
    example['provider_calls'] = calls
    return seal(example), values['submission_row']


@contextmanager
def final_only_ownership():
    """Keep final proofs but restore the previous stateless partial-payload path."""
    prepare = training_worker_proof.WorkerProofs.prepare
    def final_only(self, example):
        view = prepare(self, {key:value for key,value in example.items() if key != PARTIAL})
        if PARTIAL in example: view[PARTIAL] = example[PARTIAL]
        return view
    with patch.object(training_worker_proof.WorkerProofs, 'prepare', final_only):
        yield


def native_outputs(example, submission):
    rules = training_steps.Rules(production_rounds_only=False)
    return (training_thinking.coverage([example]),
            training_thinking.build([example], rules),
            training_thinking.build([example], rules, role='judge'),
            training_eligibility.assess_capture(submission, {example['run_id']:example}, [example['run_id']],
                enabled=True, expected_run_count=1, minimum_coverage_percent=80,
                maximum_missing_reasoning_percent=1))


def test_journal_builder_hashes_owned_payload_per_episode_not_per_pair(monkeypatch):
    example, _ = partial_example()
    before = deepcopy(example); hashes = []; original = contracts.document_hash
    def counted(value):
        if isinstance(value, dict) and 'response_receipts' in value and 'agent_observations' in value:
            hashes.append(len(value['response_receipts']))
        return original(value)
    monkeypatch.setattr(contracts, 'document_hash', counted)
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    rows = splits['train'] + splits['eval']
    assert rows and sum(len(row['source_refs']) for row in rows) == 8
    assert hashes == [8,8], hashes  # One native coverage owner and one shared builder owner.
    assert example == before


@pytest.mark.parametrize('final_value', ['absent', None, [], 'invalid', {}])
def test_partial_only_is_owned_even_without_a_valid_final_mapping(final_value):
    payload = {'response_receipts':[{'body_bytes':1}],'agent_observations':[]}
    example = {PARTIAL:{'r':{'payload':payload,'payload_sha256':contracts.document_hash(payload)}}}
    if final_value != 'absent': example[FINAL] = final_value
    owner = training_worker_proof.WorkerProofs(); view = owner.prepare(example)
    owned = view[PARTIAL]['r']['payload']
    assert owned is not payload
    assert contracts.canonical_json(owned) == contracts.canonical_json(payload)
    with pytest.raises(TypeError, match='read-only'): owned['response_receipts'][0]['body_bytes'] = 2
    if final_value != 'absent': assert view[FINAL] == final_value


@pytest.mark.parametrize('damage', ['none','payload_hash','lease','claim','segment_ref','missing_input',
                                    'request_privacy','response_privacy','response_body'])
def test_partial_native_outputs_equal_final_only_checks_and_never_qualify_icp(damage):
    example, submission = partial_example(3); rid = example['run_id']; artifact = example[PARTIAL][rid]
    if damage == 'payload_hash': artifact['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'lease': artifact['lease_generation'] += 1
    elif damage == 'claim': artifact['claim_request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'segment_ref': artifact['segment_refs'][0] = 'wrong-ref'
    elif damage == 'missing_input': example['input_snapshots'].pop(rid)
    elif damage == 'request_privacy': example['provider_calls'][0]['request_capture']['redacted'] = True
    elif damage == 'response_privacy': example['provider_calls'][0]['response_privacy_redactions'] = ['credential']
    elif damage == 'response_body': example['provider_calls'][0]['response']['choices'][0]['message']['content'] = 'Changed fixture response'
    seal(example); before = deepcopy(example)
    with final_only_ownership(): expected = native_outputs(example, submission)
    actual = native_outputs(example, submission)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert example == before
    assert not actual[-1]['eligible'] and actual[-1]['counts']['complete_runs'] == 0
    if damage == 'none':
        assert actual[1][0]['train'] or actual[1][0]['eval']
        assert example[PARTIAL][rid]['complete'] is False


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(changed=True),
    lambda p: p['response_receipts'].append({}),
    lambda p: p['response_receipts'][0].update(body_bytes=0),
    lambda p: p.__init__({}),
    lambda p: p['response_receipts'].__iadd__([{}]),
    lambda p: setattr(p, '__class__', dict),
])
def test_partial_owned_containers_reject_mutation(mutate):
    example, _ = partial_example(2); owner = training_worker_proof.WorkerProofs(); view = owner.prepare(example)
    payload = view[PARTIAL][example['run_id']]['payload']; before = contracts.canonical_json(payload)
    with pytest.raises(TypeError, match='read-only'): mutate(payload)
    assert contracts.canonical_json(payload) == before
    assert deepcopy(payload) is payload


def test_partial_owner_isolation_mutable_fallback_and_lifetime(monkeypatch):
    example, _ = partial_example(2); rid = example['run_id']
    first, second = training_worker_proof.WorkerProofs(), training_worker_proof.WorkerProofs()
    one, two = first.prepare(example), second.prepare(example)
    owned = one[PARTIAL][rid]['payload']; foreign = two[PARTIAL][rid]['payload']
    assert owned is not foreign
    original = contracts.document_hash; seen = []
    def counted(value): seen.append(1); return original(value)
    monkeypatch.setattr(contracts,'document_hash',counted)
    assert first.payload_hash(owned) == original(owned); assert seen == []
    source = example[PARTIAL][rid]['payload']
    for value in (source, foreign, json.loads(contracts.canonical_json(owned))):
        first.payload_hash(value); first.payload_hash(value)
    assert len(seen) == 6
    source['response_receipts'][0]['body_bytes'] += 1
    assert first.payload_hash(source) != first.payload_hash(owned)
    reference = weakref.ref(owned); del owned, one
    gc.collect(); assert reference() is not None
    del first; gc.collect(); assert reference() is None


@pytest.mark.parametrize('artifacts', [None, [], 'invalid', {'r':None}, {'r':{'payload':None}}, {'r':{'payload':[]}}])
def test_invalid_partial_fields_keep_previous_behavior(artifacts):
    example = {PARTIAL:artifacts}; before = deepcopy(example)
    owner = training_worker_proof.WorkerProofs(); view = owner.prepare(example)
    assert view == before and example == before


def test_custom_nested_values_stay_stateless_and_unverified_fields_are_untouched():
    class CustomDict(dict): pass
    payload = {'response_receipts':[], 'agent_observations':[], 'custom':CustomDict(value=1)}
    unverified = {'r':{'payload':{'response_receipts':[],'agent_observations':[]}}}
    example = {PARTIAL:{'r':{'payload':payload}},UNVERIFIED:unverified}
    owner = training_worker_proof.WorkerProofs(); view = owner.prepare(example)
    assert view[PARTIAL]['r']['payload'] is payload
    assert view[UNVERIFIED] is unverified
    with patch.object(contracts, 'document_hash', wraps=contracts.document_hash) as hashed:
        owner.payload_hash(unverified['r']['payload'])
        owner.payload_hash(unverified['r']['payload'])
    assert hashed.call_count == 2
    previous = owner.payload_hash(payload); payload['custom']['value'] = 2
    assert owner.payload_hash(payload) != previous
