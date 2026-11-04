"""Publication validation reuses text and registry work within existing bounds."""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest

from lab_arena import contracts, reasoning_capture, training_capture, training_eligibility
from lab_arena import training_steps, training_thinking, training_worker_proof, trajectory
from tests.lab_arena.training_eligibility_test import captured_example, submission
from tests.lab_arena.training_steps_test import seal


@contextmanager
def original_validation_work():
    """Restore the pre-change scans without replacing any outcome checks."""
    sanitize, policy = training_capture.sanitize_training_value, training_thinking._policy

    def uncached_text(value, *args, **kwargs):
        kwargs.pop('reuse_large_text', None)
        return sanitize(value, *args, **kwargs)

    def uncached_registry(call, registry=None, **kwargs):
        return policy(call, registry)

    with patch.object(training_capture, 'sanitize_training_value', side_effect=uncached_text):
        with patch.object(training_thinking, '_policy', side_effect=uncached_registry):
            yield


def growing_example(history_turns=4, registry_routes=128):
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


def test_text_reuse_is_opt_in_and_exact_validator_scans_large_plain_text_once():
    text = 'Public synthetic evidence. ' * 800
    request = {'admitted_parameters': {'messages': [text]}, 'effective_parameters': {'messages': [text]},
               'outbound': {'body': {'messages': [text]}}}
    with patch.object(trajectory, 'redact_text', wraps=trajectory.redact_text) as redact:
        assert training_capture.sanitize_training_value(request) == (request, [])
        assert sum(call.args[0] == text for call in redact.call_args_list) == 3
        redact.reset_mock()
        assert training_steps._exact_stateless(request)
        assert sum(call.args[0] == text for call in redact.call_args_list) == 1
        assert training_steps._exact_stateless(request)
        assert sum(call.args[0] == text for call in redact.call_args_list) == 2


@pytest.mark.parametrize('suffix', ['', ' Bearer synthetic-private-token',
                                    ' https://user:pass@example.test/?token=private-value'])
def test_opt_in_text_reuse_preserves_redactions_paths_and_node_limits(suffix):
    text = 'Public synthetic evidence. ' * 800 + suffix
    value = {'public': [text, text], 'authorization': text, 'hidden_reasoning': text}
    expected = training_capture.sanitize_training_value(value)
    assert training_capture.sanitize_training_value(value, reuse_large_text=True) == expected
    assert expected[0]['authorization'] == expected[0]['hidden_reasoning'] == '[REDACTED]'
    with pytest.raises(training_capture.CaptureError, match='request_structure_limit'):
        training_capture.sanitize_training_value([text] * 5, max_nodes=5, reuse_large_text=True)
    with pytest.raises(training_capture.CaptureError, match='request_redaction_key_collision'):
        training_capture.sanitize_training_value({text + 'private': 1, text + '[REDACTED]': 2},
                                                 secrets=('private',), reuse_large_text=True)


def test_opt_in_text_reuse_keeps_byte_entry_and_custom_value_bounds(monkeypatch):
    class ObservedCache(OrderedDict):
        def __setitem__(self, key, value):
            if isinstance(value, tuple):
                assert len(self) < 512
                assert sum(v[1] for v in self.values()) + value[1] <= 8 * 1024 * 1024
            super().__setitem__(key, value)

    class CustomText(str):
        def __hash__(self):
            raise AssertionError('custom hash must not be used')

    monkeypatch.setattr(training_capture, 'OrderedDict', ObservedCache)
    texts = [f'{index}: ' + '界' * 17_000 for index in range(100)]
    assert training_capture.sanitize_training_value(texts + [texts[0]], reuse_large_text=True) == (texts + [texts[0]], [])
    custom = CustomText('Public synthetic evidence. ' * 800)
    assert training_capture.sanitize_training_value([custom, custom], reuse_large_text=True) == ([custom, custom], [])


@pytest.mark.parametrize('text,secrets', [('𐀀' * 1_050_000, ()),
                                        ('aaaa' * 650_000, ('aaaa',))], ids=['wide-unicode', 'expanded-secret'])
def test_opt_in_does_not_retain_text_and_output_above_existing_byte_bound(text, secrets):
    with patch.object(trajectory, 'redact_text', wraps=trajectory.redact_text) as redact:
        cleaned, _ = training_capture.sanitize_training_value([text, text], secrets=secrets,
                                                              reuse_large_text=True)
        assert cleaned[0] == cleaned[1]
        assert sum(len(call.args[0]) > 16_384 for call in redact.call_args_list) == 2


def benchmark():
    """Run with python3.11 -m tests.lab_arena.training_publication_cpu_reuse_test."""
    for turns in (2, 4, 8):
        example = growing_example(history_turns=turns, registry_routes=256)
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
               'registry_routes': 256, 'example_bytes': len(contracts.canonical_json(example).encode()),
               'before_seconds': timings[0], 'after_seconds': timings[1],
               'registry_validations_before_after': counts,
               'result_sha256': contracts.document_hash(outcomes[1])})


if __name__ == '__main__':
    benchmark()
