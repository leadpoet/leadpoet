"""Finite retry/capacity policy independent of private provider diagnostics."""
import pytest
from tests.lab_arena.test_lab_arena_code_review import _archive


@pytest.mark.parametrize('global_attempts', [1, 7, 20])
def test_finite_invalid_episode_ignores_earlier_global_attempts(global_attempts):
    from lab_arena import code_review_policy as policy
    row = {'code_review_status': 'error', 'code_review_attempts': global_attempts,
           'code_review_doc': {'error_code': 'review_response_invalid'}}
    assert policy.review_retry_available(row)
    row['code_review_doc']['review_invalid_attempts'] = 2
    assert policy.review_retry_available(row)
    row['code_review_doc']['review_invalid_attempts'] = 3
    assert not policy.review_retry_available(row)
    row['code_review_doc']['error_code'] = 'code_review_transport_error'
    assert not policy.review_retry_available(row)
    row['code_review_doc']['refusal_recovery'] = True
    row['code_review_doc']['review_primary_attempts'] = 1
    row['code_review_doc']['review_alternate_attempts'] = 0
    assert not policy.review_retry_available(row)
    row['code_review_doc']['review_invalid_attempts'] = 2
    assert policy.review_retry_available(row)



def test_output_capacity_uses_authoritative_failure_and_ignores_private_diagnostic():
    from lab_arena import code_review_policy as policy
    row = {'code_review_doc': {'error_code': 'review_response_invalid',
           'provider_diagnostic': {'finish_reason': 'length', 'max_output_tokens': 32768}}}
    assert policy.review_output_tokens_for(row) == 32768
    row['code_review_doc']['error_reason'] = 'truncated'
    assert policy.review_output_tokens_for(row) == 65536
    row['code_review_doc'] = {'error_code': 'code_review_transport_error',
                             'review_output_token_recommendation': 65536}
    assert policy.review_output_tokens_for(row) == 65536



@pytest.mark.parametrize('recommendation', [None, 65536, '32768', True])
def test_claim_output_plan_mismatch_settles_zero_without_sending(recommendation):
    from tests.lab_arena.test_lab_arena_code_review_runtime import _reviewer, _source, ReviewTransport
    transport = ReviewTransport(lambda _parameters: pytest.fail('mismatched plan dispatched'))
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    original = store.begin_submission_review
    def begin(*args):
        return {**original(*args), 'review_output_token_recommendation': recommendation}
    store.begin_submission_review = begin
    assert reviewer.review(row)['status'] == 'error'
    assert not transport.sent
    assert store.finish_calls[-1][5] == 0
    assert store.finish_calls[-1][4]['error_code'] == 'code_review_preparation_unavailable'



@pytest.mark.parametrize('elevated', [False, True])
def test_missing_recommendation_is_legacy_default_only(elevated):
    from tests.lab_arena.test_lab_arena_code_review_runtime import _reviewer, _source, _review_body, ReviewTransport
    transport = ReviewTransport(lambda parameters: {'body': _review_body(parameters)})
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    original = store.begin_submission_review
    def begin(*args):
        result = original(*args)
        result.pop('review_output_token_recommendation')
        return result
    store.begin_submission_review = begin
    if elevated:
        row['code_review_doc'] = {'review_output_token_recommendation': 65536}
    assert reviewer.review(row)['status'] == ('error' if elevated else 'passed')
    assert len(transport.sent) == (0 if elevated else 1)
    if elevated:
        assert store.finish_calls[-1][5] == 0
