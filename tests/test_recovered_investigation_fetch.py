"""A recovered source must not keep a completed negative review retryable."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer

URL = 'https://www.businesswire.com/news/home/20260505824389/en/Paymentus-to-Participate-in-Upcoming-Investor-Conferences-in-May-and-June'


def outcome(url=URL, *, ok=True, error=''):
    return investigator._fetch_outcome(
        url, {'ok': ok, 'error': error, 'text': 'Company announcement.' if ok else ''},
    )


def terminal(outcomes):
    receipt = {
        'gate': 'company_evidence_investigation',
        'completed_submitted_findings': True,
        'failure_reason': '',
        'targets': ['geography'],
        'claims': {'geography': {
            'target': 'geography', 'status': 'UNPROVEN',
            'evidence_url': '', 'evidence_quote': '',
        }},
        'usage': {'fetch_outcomes': outcomes},
    }
    return lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ('geography',), investigation_receipt=receipt,
    )


def test_exact_url_fallback_recovers_complete_unproven_review():
    # fetch_fresh_page records both the primary502 and successful Exa fallback.
    assert terminal([outcome(ok=False, error='http_502'), outcome()])


def test_normalized_same_url_recovers():
    assert terminal([
        outcome(URL + '#section', ok=False, error='http_502'), outcome(),
    ])


@pytest.mark.parametrize('outcomes', [
    [outcome(ok=False, error='http_502'), outcome('https://another.example/about')],
    [outcome(), outcome(ok=False, error='http_502')],
    [outcome(ok=False, error='http_502'), outcome(), outcome('https://another.example/about', ok=False, error='http_503')],
    [{'ok': False, 'error_class': 'http_error'}, outcome()],
    [{'url_sha256': 'bad', 'ok': False, 'error_class': 'http_error'}, outcome()],
    [outcome('invalid', ok=False, error='invalid_url'), outcome('invalid')],
    [outcome(ok=False, error='http_502'), {**outcome(), 'loaded_text_length': 0}],
    [outcome(ok=False, error='http_502'), {**outcome(), 'loaded_text_length': True}],
    [outcome(ok=False, error='http_502'), {**outcome(), 'loaded_text_length': '100'}],
])
def test_unresolved_or_unbound_source_failure_stays_retryable(outcomes):
    assert not terminal(outcomes)


def test_receipt_scope_does_not_recover_another_company_obligation():
    assert terminal([outcome()])
    assert not terminal([outcome(ok=False, error='http_502')])


@pytest.mark.parametrize('error', ['http_404', 'http_410', 'provider_request_refused'])
def test_existing_company_local_source_failures_stay_terminal(error):
    assert terminal([outcome(ok=False, error=error)])
