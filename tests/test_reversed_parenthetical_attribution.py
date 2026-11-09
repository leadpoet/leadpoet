"""Minimal anonymized MEC capture: exact legal/acronym labels reverse order."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer
from tests.test_company_evidence_attribute_reuse import _finding

URL = 'https://acmebearings.example/about'
SUBMITTED = 'ABC (Acme Bearing Company, Inc.)'
OBSERVED = 'Acme Bearing Company, Inc. (ABC)'
QUOTE = ('Acme Bearing Company (NYSE: ABC) operates twenty manufacturing plants '
         'and produces metal components for industrial customers.')


def anchor():
    return {
        'submitted_name': SUBMITTED, 'observed_name': OBSERVED, 'verified_name': SUBMITTED,
        **{key: 'acmebearings.example' for key in
           ('submitted_domain', 'observed_domain', 'verified_domain')},
        **{key: 'acme-bearing-company' for key in
           ('submitted_linkedin_slug', 'observed_linkedin_slug', 'verified_linkedin_slug')},
    }


def validated(identity=None, *, quote=QUOTE, text=QUOTE, url=URL, final_url=URL):
    identity = identity or anchor()
    positive = _finding('required_attribute', url=url, quote=quote)
    positive.update(observed_value='Own plants producing industrial components',
                    old_name=SUBMITTED, new_name=SUBMITTED)
    negative = _finding('industry', 'UNPROVEN', url=url, quote=quote)
    names = {inv._company_name(identity[key]) for key in
             ('submitted_name', 'observed_name', 'verified_name')}
    return inv._validated_findings(
        {'findings': [negative, positive]},
        targets=('industry', 'required_attribute'), fetched_pages={url: text},
        fetched_final_urls={url: final_url}, first_party_domains={'acmebearings.example'},
        identity_names=names, identity_anchor=identity,
    )


def test_exact_reversed_legal_initialism_retains_fully_bound_first_party_identity():
    identity = anchor()
    before = deepcopy(identity)
    assert inv._complete_verified_first_party_identity(URL, {'acmebearings.example'}, identity)
    assert identity == before


def test_actual_validation_keeps_quote_and_industry_conditions_separate():
    result = validated()
    assert result['required_attribute']['status'] == 'VERIFIED'
    assert result['industry']['status'] == 'UNPROVEN'
    assert result['required_attribute']['evidence_quote'] == QUOTE
    assert validated(json.loads(json.dumps(anchor()))) == result


@pytest.mark.parametrize('update', [
    {'observed_name': 'Acme Battery Company, Inc. (ABC)'},
    {'observed_name': 'Acme Bearing Holdings, Inc. (ABC)'},
    {'observed_name': 'Acme Bearing Company Asia, Inc. (ABC)'},
    {'observed_name': 'Acme Bearing Company, Inc. (XYZ)'},
    {'observed_name': 'ABC'},
    {'observed_name': 'ABC (part of Acme Bearing Company, Inc.)'},
    {'observed_name': 'Acme Bearing Company, Inc. (ABC subsidiary)'},
    {'observed_name': 'Acme Bearing Company, Inc. (ABC) customer services'},
    {'observed_name': 'Acme (Bearing) Company, Inc. (ABC)'},
    {'observed_name': 'Acme Bearing Company, Inc. (abc)'},
    {'submitted_name': 'XYZ (Acme Bearing Company, Inc.)',
     'verified_name': 'XYZ (Acme Bearing Company, Inc.)',
     'observed_name': 'Acme Bearing Company, Inc. (XYZ)'},
    {'verified_name': 'ABC (Acme Bearing Holdings, Inc.)'},
    {'verified_name': 'ABC'},
    {'observed_domain': 'different.example'},
    {'verified_domain': ''},
    {'observed_linkedin_slug': 'other-company'},
    {'verified_linkedin_slug': ''},
    {'submitted_linkedin_slug': ''},
])
def test_reversal_never_supplies_missing_or_conflicting_identity(update):
    identity = {**anchor(), **update}
    assert not inv._complete_verified_first_party_identity(URL, {'acmebearings.example'}, identity)
    unnamed = 'It operates twenty manufacturing plants and produces industrial components.'
    assert validated(identity, quote=unnamed, text=unnamed)['required_attribute']['status'] != 'VERIFIED'


@pytest.mark.parametrize('kwargs', [
    {'text': 'The page does not contain the submitted quote.'},
    {'quote': QUOTE + ' This invented clause is unsupported.'},
    {'url': 'https://different.example/about'},
    {'final_url': 'https://different.example/about'},
])
def test_exact_quote_and_independent_requested_final_domains_still_required(kwargs):
    assert validated(**kwargs)['required_attribute']['status'] != 'VERIFIED'


def recovery_context():
    # Attribution-consumer fixture, not a fabricated signed/verified receipt.
    # Its caller-owned identity_decision is a separate existing precondition.
    receipt = {**anchor(), 'reason_code': 'test_retained_identity',
               'source_request_hash': 'sha256:' + '1' * 64}
    prior = SimpleNamespace(details={'identity_decision': 'match', 'identity_receipt': receipt})
    company = SimpleNamespace(company_name=SUBMITTED,
        company_website='https://acmebearings.example/',
        company_linkedin='https://www.linkedin.com/company/acme-bearing-company/')
    homepage = {'normalized_name': SUBMITTED, 'registrable_dns_domain': 'acmebearings.example',
                'linkedin_company_slug': 'acme-bearing-company'}
    return company, prior, homepage


def test_saved_attribute_recovery_consumer_roundtrip_preserves_identity_metadata():
    company, prior, homepage = recovery_context()
    prior.details = json.loads(json.dumps(prior.details))
    before = deepcopy((prior.details, homepage))
    assert lead_scorer._complete_verified_attribute_recovery_identity(company, prior, homepage, URL)
    assert (prior.details, homepage) == before


@pytest.mark.parametrize('mutation', ['unavailable', 'mismatch', 'wrong_entity', 'wrong_domain', 'wrong_profile'])
def test_saved_consumer_never_promotes_an_unavailable_or_conflicting_identity(mutation):
    company, prior, homepage = recovery_context()
    if mutation in {'unavailable', 'mismatch'}:
        prior.details['identity_decision'] = mutation
    elif mutation == 'wrong_entity':
        prior.details['identity_receipt']['observed_name'] = 'Acme Battery Company, Inc. (ABC)'
    elif mutation == 'wrong_domain':
        prior.details['identity_receipt']['observed_domain'] = 'other.example'
    else:
        prior.details['identity_receipt']['observed_linkedin_slug'] = 'other-company'
    assert not lead_scorer._complete_verified_attribute_recovery_identity(company, prior, homepage, URL)
