"""An invalid episode changes its last route without gaining reservations."""
import pytest

from lab_arena import code_review, code_review_policy

PRIMARY = code_review.DEFAULT_REVIEW_MODEL
ALTERNATE = code_review.FALLBACK_REVIEW_MODEL


@pytest.mark.parametrize('count,expected', [(None, PRIMARY), (1, PRIMARY), (2, ALTERNATE), (3, PRIMARY)])
def test_only_last_existing_invalid_slot_selects_alternate(count, expected):
    document = {'error_code': 'review_response_invalid', 'model': PRIMARY}
    if count is not None:
        document['review_invalid_attempts'] = count
    assert code_review_policy.review_model_for({'code_review_attempts': 17, 'code_review_doc': document}) == expected


@pytest.mark.parametrize('count', [False, True, '2', -1, 0])
def test_unproved_invalid_count_cannot_authorize_alternate(count):
    assert code_review_policy.review_model_for({'code_review_doc': {
        'error_code': 'review_response_invalid', 'review_invalid_attempts': count,
    }}) == PRIMARY


@pytest.mark.parametrize('primary,expected', [(1, PRIMARY), (2, ALTERNATE)])
def test_refusal_route_keeps_its_episode_allowance(primary, expected):
    assert code_review_policy.review_model_for({'code_review_doc': {
        'error_code': 'review_response_invalid', 'refusal_recovery': True,
        'review_primary_attempts': primary, 'review_alternate_attempts': 0,
        'review_invalid_attempts': 2,
    }}) == expected


def test_third_reservation_still_exhausts_after_alternate_failure():
    row = {'code_review_status': 'error', 'code_review_attempts': 10, 'code_review_doc': {
        'model': ALTERNATE, 'error_code': 'code_review_transport_error',
        'review_invalid_attempts': 3, 'retryable': True,
    }}
    assert not code_review_policy.review_retry_available(row)
