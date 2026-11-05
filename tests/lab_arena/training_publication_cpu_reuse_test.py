"""Publication validation reuses one immutable registry within each selected run."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest

from lab_arena import contracts, reasoning_capture, training_eligibility
from lab_arena import training_thinking, training_worker_proof
from tests.lab_arena.training_eligibility_test import captured_example, submission
from tests.lab_arena.training_steps_test import seal


@contextmanager
def original_validation_work():
    """Restore repeated registry validation without replacing outcome checks."""
    policy = training_thinking._policy

    def uncached_registry(call, registry=None, **kwargs):
        return policy(call, registry)

    with patch.object(training_thinking, '_policy', side_effect=uncached_registry):
        yield


def growing_example(history_turns=4, registry_routes=95):
    example = captured_example()
    registry = example['configuration']['reasoning_route_registry']
    original = deepcopy(registry['approved_routes'][0])
    for index in range(registry_routes - len(registry['approved_routes'])):
        registry['approved_routes'].append(dict(deepcopy(original), model_id=f'synthetic/model-{index}'))
    registry_hash = contracts.document_hash(registry).split(':', 1)[1]
    for index, call in enumerate(example['provider_calls']):
        call['reasoning_capture']['registry_sha256'] = registry_hash
        request = call['request']
        request['reasoning_capture']['registry_sha256'] = registry_hash
        body = request['outbound']['body']
        history = []
        for turn in range(history_turns * (index + 1)):
            history.extend([{'role': 'user', 'content': f'Company evidence {turn}: ' + 'Public business evidence. ' * 800},
                            {'role': 'assistant', 'content': 'Public summary.'}])
        body['messages'] = history + body['messages']
        request['admitted_parameters'] = deepcopy(body)
        request['admitted_parameters'].pop('provider', None)
        request['effective_parameters'] = deepcopy(body)
        request['outbound']['body_sha256'] = contracts.document_hash(body)
    return seal(example)


def assess(example):
    run_id = example['run_id']
    return training_eligibility.assess_capture(submission(), {run_id: example}, [run_id], enabled=True)


@pytest.mark.parametrize('damage', ['none', 'privacy', 'bad_route', 'malformed', 'bad_hash'])
def test_actual_assessment_bytes_and_decision_equal_original_scans(damage):
    example = growing_example(history_turns=1, registry_routes=32)
    call = example['provider_calls'][0]
    if damage == 'privacy':
        call['request']['outbound']['body']['messages'][0]['content'] += ' Bearer synthetic-private-token'
        call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    elif damage == 'bad_route':
        call['reasoning_capture']['provider_route'] = 'unreviewed-route'
    elif damage == 'malformed':
        call['request']['outbound']['body']['messages'][0]['role'] = 'unsupported-role'
        call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    elif damage == 'bad_hash':
        call['request']['outbound']['body_sha256'] = 'bad'
    seal(example)
    before = deepcopy(example)
    with original_validation_work():
        expected = assess(example)
    actual = assess(example)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert actual['eligible'] is (damage == 'none')
    assert example == before


def test_actual_assessment_validates_registry_once_and_rechecks_each_assessment():
    example = growing_example(history_turns=1, registry_routes=32)
    with original_validation_work():
        with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
            assert assess(example)['eligible']
            assert validate.call_count == 9
    with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
        assert assess(example)['eligible']
        assert validate.call_count == 1
        assert assess(deepcopy(example))['eligible']
        assert validate.call_count == 2
    example['configuration']['reasoning_route_registry']['approved_routes'][0]['status'] = 'unapproved'
    seal(example)
    assert not assess(example)['eligible']


@pytest.mark.parametrize('kind', ['undeclared', 'empty_chat'])
@pytest.mark.parametrize('bad_route', [False, True])
def test_output_fault_assessment_reuses_owned_registry_and_keeps_route_rejection(kind, bad_route):
    from tests.lab_arena.training_provider_output_coverage_test import output_fault
    from tests.lab_arena.training_eligibility_test import refresh_response_links

    example = output_fault(kind)
    if bad_route:
        example['provider_calls'][0]['response_receipts'][0]['metadata']['reasoning_route_observation']['status'] = 'mismatched'
        refresh_response_links(example, 0)
    before = deepcopy(example)
    validations, outcomes = [], []
    attempt = training_thinking.attempt_coverage

    def omit_owner(*args, **kwargs):
        kwargs.pop('worker_proofs', None)
        return attempt(*args, **kwargs)

    for forward in (False, True):
        owner = training_worker_proof.WorkerProofs()
        owned = owner.prepare(deepcopy(example))
        call = owned['provider_calls'][0]
        # readable_trace warms this same selected-run registry before the
        # assessment reaches an unusable generated target.
        attempt(call, owned['role'], owned['configuration']['reasoning_route_registry'], worker_proofs=owner)
        with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
            with (nullcontext() if forward else patch.object(training_thinking, 'attempt_coverage', side_effect=omit_owner)):
                outcomes.append(training_eligibility._verified_provider_output_fault(owned, call, worker_proofs=owner))
            validations.append(validate.call_count)
    assert outcomes == [not bad_route, not bad_route]
    assert validations[1] == validations[0] - 1
    with original_validation_work():
        expected = assess(example)
    with patch.object(training_eligibility, '_verified_provider_output_fault',
                      wraps=training_eligibility._verified_provider_output_fault) as fault:
        actual = assess(example)
        assert fault.call_count > 0
        assert all(type(call.kwargs.get('worker_proofs')) is training_worker_proof.WorkerProofs
                   for call in fault.call_args_list)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert example == before


def test_registry_reuse_rehashes_mutations_and_keeps_snapshot_immutable():
    example = captured_example()
    call, registry = example['provider_calls'][0], example['configuration']['reasoning_route_registry']
    owner = training_worker_proof.WorkerProofs()
    with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
        assert training_thinking._policy(call, registry, worker_proofs=owner)
        assert training_thinking._policy(deepcopy(call), deepcopy(registry), worker_proofs=owner)
        assert validate.call_count == 1
        with pytest.raises(TypeError, match='read-only'):
            owner._reasoning_registry[1]['approved_routes'][0]['status'] = 'unapproved'
        entry = next(row for row in registry['approved_routes']
                     if row['model_id'] == call['reasoning_capture']['model_id'])
        entry['model_version'] = 'synthetic/changed-model'
        assert not training_thinking._policy(call, registry, worker_proofs=owner)
        assert validate.call_count == 2
        call['reasoning_capture']['registry_sha256'] = contracts.document_hash(registry).split(':', 1)[1]
        assert not training_thinking._policy(call, registry, worker_proofs=owner)
        assert validate.call_count == 2


def test_invalid_unsupported_and_digest_failure_keep_stateless_registry_validation():
    class CustomDict(dict):
        pass

    example = captured_example()
    call, registry = example['provider_calls'][0], example['configuration']['reasoning_route_registry']
    for value in (CustomDict(registry), dict(registry, baseline_route='malformed')):
        owner = training_worker_proof.WorkerProofs()
        with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
            for _ in range(2):
                assert training_thinking._policy(call, value, worker_proofs=owner) == training_thinking._policy(call, value)
            assert validate.call_count == 4
        assert owner._reasoning_registry is None
    owner = training_worker_proof.WorkerProofs()
    with patch.object(training_worker_proof, '_exact_privacy_digest', side_effect=RuntimeError('digest failed')):
        with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
            for _ in range(2):
                assert training_thinking._policy(call, registry, worker_proofs=owner)
            assert validate.call_count == 2
    assert owner._reasoning_registry is None
    with patch.object(reasoning_capture, 'validate_registry', side_effect=ValueError('invalid')) as validate:
        for _ in range(2):
            assert not training_thinking._policy(call, registry, worker_proofs=owner)
        assert validate.call_count == 2
    assert owner._reasoning_registry is None


@pytest.mark.parametrize('field,value', [('model_version', 'synthetic/unreviewed'),
                                         ('provider_route', 'unreviewed'),
                                         ('full_text_fields', ['reasoning_details.text'])])
def test_warmed_registry_never_reuses_call_annotation_or_route_decisions(field, value):
    example = captured_example()
    call, registry = example['provider_calls'][0], example['configuration']['reasoning_route_registry']
    owner = training_worker_proof.WorkerProofs()
    assert training_thinking._policy(call, registry, worker_proofs=owner)
    call['reasoning_capture'][field] = value
    assert not training_thinking._policy(call, registry, worker_proofs=owner)
    assert training_thinking._policy(call, registry, worker_proofs=owner) == training_thinking._policy(call, registry)


def test_registry_cache_keeps_only_the_current_single_snapshot():
    example = captured_example()
    registry = example['configuration']['reasoning_route_registry']
    other = deepcopy(registry)
    other['baseline_route'] = {'model_id': other['approved_routes'][0]['model_id'],
                               'api': other['approved_routes'][0]['api']}
    owner = training_worker_proof.WorkerProofs()
    with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
        for value in (registry, other, registry):
            owner.reasoning_registry(value, validate=reasoning_capture.validate_registry)
        assert validate.call_count == 3


def benchmark():
    """Run with python3.11 -m tests.lab_arena.training_publication_cpu_reuse_test."""
    for turns in (2, 4, 8):
        example = growing_example(history_turns=turns, registry_routes=95)
        outcomes, timings, counts = [], [], []
        for original in (True, False):
            samples = []
            for _ in range(3):
                with (original_validation_work() if original else nullcontext()):
                    start = perf_counter()
                    result = assess(example)
                    samples.append(perf_counter() - start)
            outcomes.append(result)
            timings.append(round(median(samples), 4))
            with (original_validation_work() if original else nullcontext()):
                with patch.object(reasoning_capture, 'validate_registry', wraps=reasoning_capture.validate_registry) as validate:
                    assess(example)
                    counts.append(validate.call_count)
        assert outcomes[0] == outcomes[1] and outcomes[1]['eligible']
        print({'history_turns_per_call': [turns * n for n in (1, 2, 3)],
               'registry_routes': 95, 'example_bytes': len(contracts.canonical_json(example).encode()),
               'before_seconds': timings[0], 'after_seconds': timings[1],
               'registry_validations_before_after': counts,
               'result_sha256': contracts.document_hash(outcomes[1])})


if __name__ == '__main__':
    benchmark()
