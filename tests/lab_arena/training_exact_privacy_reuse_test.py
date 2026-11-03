"""Per-example privacy reuse rechecks content and preserves independent proof."""

from copy import deepcopy
from decimal import Decimal
from unittest.mock import patch

import pytest

from lab_arena import training_steps as steps, training_worker_proof as proof
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import fixture


def test_mutable_content_and_distinct_equal_objects_are_checked_by_current_content():
    owner = proof.WorkerProofs()
    value = {'messages': [{'role': 'user', 'content': 'ordinary'}]}
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        assert steps._exact(value, worker_proofs=owner)
        assert steps._exact(deepcopy(value), worker_proofs=owner)
        assert scan.call_count == 1
        value['messages'][0]['content'] = 'Bearer synthetic-secret'
        assert not steps._exact(value, worker_proofs=owner)
        assert not steps._exact(deepcopy(value), worker_proofs=owner)
        assert scan.call_count == 2
        value['messages'][0]['content'] = 'ordinary'
        assert steps._exact(value, worker_proofs=owner)
        assert scan.call_count == 2


def test_every_call_rehashes_current_json_and_response_mode():
    owner = proof.WorkerProofs()
    value = {'choices': [{'message': {'reasoning_content': None}}]}
    with patch.object(proof, '_exact_privacy_digest', wraps=proof._exact_privacy_digest) as digest:
        with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
            for _ in range(2):
                assert not steps._exact(value, worker_proofs=owner)
                assert steps._exact(value, response=True, worker_proofs=owner)
            assert scan.call_count == 2
        assert digest.call_count == 4
    assert proof._exact_privacy_digest(value, False) != proof._exact_privacy_digest(value, True)


@pytest.mark.parametrize('value,expected', [
    ({'text': 'ordinary'}, True),
    ({'authorization': 'Bearer synthetic-secret'}, False),
    ({'type': 'reasoning', 'encrypted_content': 'opaque'}, False),
    ({'type': 'reasoning', 'text': 'Readable thought', 'encrypted_content': None}, True),
    ('[REDACTED]', True),
])
@pytest.mark.parametrize('response', [False, True])
def test_clean_dirty_hidden_and_empty_privacy_results_match_stateless(value, expected, response):
    owner = proof.WorkerProofs()
    assert steps._exact(value, response=response) is expected
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        for _ in range(2):
            assert steps._exact(value, response=response, worker_proofs=owner) is expected
        assert scan.call_count == 1


class CustomDict(dict):
    pass


class CustomString(str):
    pass


@pytest.mark.parametrize('value', [
    CustomDict(text='ordinary'), {'nested': CustomDict(text='ordinary')},
    {'text': CustomString('ordinary')}, {1: 'numeric key'},
    {'value': (1, 2)}, {'value': Decimal('1.2')},
    {'value': float('nan')}, {'value': float('inf')}, {'value': float('-inf')},
    {'text': '\ud83d\ude00'},
])
def test_unsupported_custom_and_nonfinite_inputs_use_original_scan_each_time(value):
    owner = proof.WorkerProofs()
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        for _ in range(2):
            try:
                expected = steps._exact_stateless(value)
            except Exception as exc:
                with pytest.raises(type(exc), match=str(exc)):
                    steps._exact(value, worker_proofs=owner)
            else:
                assert steps._exact(value, worker_proofs=owner) == expected
        assert scan.call_count == 4
    assert not owner._exact_privacy


def test_cycles_preserve_original_recursion_error():
    owner = proof.WorkerProofs()
    value = []
    value.append(value)
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        for _ in range(2):
            with pytest.raises(RecursionError):
                steps._exact(value, worker_proofs=owner)
        assert scan.call_count == 2
    assert not owner._exact_privacy


def test_digest_failure_falls_back_and_scanner_exceptions_are_never_cached():
    owner = proof.WorkerProofs()
    value = {'text': 'ordinary'}
    with patch.object(proof, '_exact_privacy_digest', side_effect=RuntimeError('digest unavailable')):
        with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
            assert steps._exact(value, worker_proofs=owner)
            assert steps._exact(value, worker_proofs=owner)
            assert scan.call_count == 2
    assert not owner._exact_privacy
    with patch.object(steps, '_exact_stateless', side_effect=ValueError('scan failed')) as scan:
        for _ in range(2):
            with pytest.raises(ValueError, match='scan failed'):
                steps._exact(value, worker_proofs=owner)
        assert scan.call_count == 2
    assert not owner._exact_privacy


def test_separate_owners_standalone_and_foreign_contexts_do_not_share_results():
    first, second = proof.WorkerProofs(), proof.WorkerProofs()
    value = {'text': 'ordinary'}
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        assert steps._exact(value, worker_proofs=first)
        assert steps._exact(value, worker_proofs=first)
        assert steps._exact(value, worker_proofs=second)
        assert steps._exact(value)
        assert steps._exact(value)
        assert steps._exact(value, worker_proofs=object())
        assert scan.call_count == 5


def test_original_scan_order_is_not_canonicalized_for_reuse():
    owner = proof.WorkerProofs()
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        assert steps._exact({'first': 'a', 'second': 'b'}, worker_proofs=owner)
        assert steps._exact({'second': 'b', 'first': 'a'}, worker_proofs=owner)
        assert scan.call_count == 2


def test_unicode_scalar_and_surrogate_pair_cannot_share_a_content_digest():
    # ASCII JSON escaping represents these different Python strings identically.
    # Invalid UTF-8 strings must instead keep the original stateless scan.
    assert proof._exact_privacy_digest({'text': '\U0001f600'}, False)
    with pytest.raises(UnicodeEncodeError):
        proof._exact_privacy_digest({'text': '\ud83d\ude00'}, False)


def test_entry_bound_fifo_eviction_and_no_retained_input_values(monkeypatch):
    monkeypatch.setattr(proof, 'MAX_EXACT_PRIVACY_PROOFS', 2)
    owner = proof.WorkerProofs()
    with patch.object(steps, '_exact_stateless', wraps=steps._exact_stateless) as scan:
        for text in ('first private fixture', 'second private fixture', 'third private fixture'):
            assert steps._exact({'text': text}, worker_proofs=owner)
        assert len(owner._exact_privacy) == 2
        assert steps._exact({'text': 'third private fixture'}, worker_proofs=owner)
        assert scan.call_count == 3
        assert steps._exact({'text': 'first private fixture'}, worker_proofs=owner)
        assert scan.call_count == 4
        assert len(owner._exact_privacy) == 2
    for (digest, response), result in owner._exact_privacy.items():
        assert type(digest) is bytes and len(digest) == 32
        assert type(response) is bool and type(result) is bool
    assert owner._proofs == {}


@pytest.mark.parametrize('host', [False, True])
def test_actual_candidate_path_reuses_requests_and_responses_without_changing_output(host):
    example = without_client_callbacks() if host else fixture()
    call = example['provider_calls'][0]
    before = deepcopy(example)
    expected = steps._candidate(example, call)
    owner = proof.WorkerProofs()
    with patch.object(steps.training_capture, 'sanitize_training_value',
                      wraps=steps.training_capture.sanitize_training_value) as scan:
        assert steps._candidate(example, call, worker_proofs=owner) == expected
        assert steps._candidate(example, call, worker_proofs=owner) == expected
        # One request scan and one response scan; _model_request and
        # _host_response share the request result in the host-result variant.
        assert scan.call_count == 2
    assert example == before


@pytest.mark.parametrize('damage,reason', [
    ('request_capture_flag', 'request_privacy_nonexact'),
    ('receipt_privacy_flag', 'response_privacy_nonexact'),
    ('request_hash', 'request_identity_mismatch'),
    ('response_hash', 'response_hash_mismatch'),
    ('worker_identity', 'worker_write_identity_mismatch'),
])
def test_warmed_reuse_does_not_bypass_capture_receipt_or_identity_checks(damage, reason):
    example = without_client_callbacks()
    call = example['provider_calls'][0]
    owner = proof.WorkerProofs()
    assert steps._candidate(example, call, worker_proofs=owner)
    if damage == 'request_capture_flag':
        call['request_capture'] = {'redacted': True}
    elif damage == 'receipt_privacy_flag':
        call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'request_hash':
        call['request_hash'] = 'bad'
    elif damage == 'response_hash':
        call['response_receipts'][0]['response_sha256'] = 'bad'
    else:
        call['action_sequence'] += 1
    with pytest.raises(ValueError, match=reason):
        steps._candidate(example, call, worker_proofs=owner)
