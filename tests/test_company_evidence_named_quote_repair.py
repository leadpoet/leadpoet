"""Public source text tests citation repair; mock replies do not prove model accuracy."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_judgment_reserve import _Clock, _finding, _response

SOURCE = json.loads((Path(__file__).parent / 'fixtures/company_evidence/kratos_named_hardware_source.json').read_text())
URL, TEXT = SOURCE['source_url'], SOURCE['source_text']
QUOTE = TEXT[TEXT.index('Kratos’ Pennsylvania operations engineer'):]
NAME = 'Kratos Defense & Security Solutions, Inc.'
NAMES = {'kratosdefenseandsecuritysolutions', 'kratosdefensesecuritysolutionsinc', 'kratosdefense'}
IDENTITY = {
    'submitted_name': 'Kratos Defense and Security Solutions',
    'observed_name': NAME, 'verified_name': 'kratosdefense',
    'submitted_domain': 'kratosdefense.com', 'observed_domain': 'kratosdefense.com',
    'verified_domain': 'kratosdefense.com',
}


def finding(quote=QUOTE, **changes):
    return {**_finding(url=URL, quote=quote), 'target': 'required_attribute',
            'observed_value': 'Manufactures military hardware', **changes}


def validate(raw, pages=None):
    return inv._validated_findings(
        {'findings': [raw]}, targets=('required_attribute',),
        fetched_pages=pages or {URL: TEXT}, fetched_final_urls={URL: URL},
        first_party_domains={'kratosdefense.com'}, identity_names=NAMES,
        identity_anchor=IDENTITY,
    )['required_attribute']


def test_retained_nov8_short_brand_requires_named_contiguous_repair():
    assert len(TEXT) == 1169
    assert validate(finding())['reason'] == 'source quote did not identify the investigated company'
    candidate = inv._company_bound_quote_candidate(QUOTE, TEXT, NAMES)
    assert candidate == TEXT
    assert inv._quote_occurs(candidate, TEXT)
    assert validate(finding(candidate))['status'] == 'VERIFIED'
    assert validate(finding())['status'] == 'UNPROVEN'  # gate stays strict


@pytest.mark.parametrize('quote,text,names', [
    (QUOTE, QUOTE, NAMES),  # a naked brand is not new identity authority
    (QUOTE + ' invented', TEXT, NAMES),
    (QUOTE, NAME + (' x' * 1100) + QUOTE, NAMES),
    (QUOTE, TEXT, {'anothercompany'}),
    (QUOTE, TEXT, set()),
    (QUOTE, 'Another Company customer: ' + QUOTE, NAMES),
], ids=['short-brand', 'invented', 'too-distant', 'wrong-company', 'no-identity', 'customer'])
def test_candidate_never_invents_or_joins_an_attribution_span(quote, text, names):
    assert inv._company_bound_quote_candidate(quote, text, names) == ''


@pytest.mark.parametrize('final_status', ['VERIFIED', 'UNPROVEN'])
def test_exact_candidate_is_context_not_automatic_acceptance(monkeypatch, final_status):
    requests = []
    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        assert len(requests) <= 2
        quote = QUOTE
        if len(requests) == 2:
            feedback = next(json.loads(m['content']) for m in reversed(payload['messages']) if m.get('role') == 'tool')
            rejected = feedback['rejected_findings'][0]
            assert rejected['reason'] == 'source quote did not identify the investigated company'
            quote = rejected['company_bound_quote_candidate']
            assert quote == TEXT
            assert 'not prove the claim' in feedback['instruction']
        revised = finding(quote)
        if len(requests) == 2 and final_status == 'UNPROVEN':
            revised.update(status='UNPROVEN', evidence_url='', evidence_quote='',
                           reason='A separate mandatory capability is not proved.')
        return 200, _response('submit_findings', {'findings': [revised]}, call=len(requests))
    async def no_research(*_args, **_kwargs):
        raise AssertionError('Existing exact source needs no additional research')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    monkeypatch.setenv('EXA_API_KEY', 'test')
    monkeypatch.setattr(inv, '_post_json', post)
    monkeypatch.setattr(inv, '_search_web', no_research)
    monkeypatch.setattr(inv, '_fetch_page', no_research)
    monkeypatch.setattr(inv, 'time', _Clock())
    monkeypatch.setattr(inv, 'MAX_REASONING_TURNS', 1)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={'name': IDENTITY['submitted_name'], 'website': 'https://kratosdefense.com'},
        targets=('required_attribute',), requested_attribute='Manufactures military hardware.',
        prior_observations={'observed_company_name': NAME, 'observed_company_website': 'https://kratosdefense.com', 'submitted_source_urls': [URL]},
        verified_homepage_identity={'normalized_name': 'kratosdefense', 'registrable_dns_domain': 'kratosdefense.com'},
        prefetched_pages={URL: {'final_url': URL, 'text': TEXT}},
    ))
    assert result['_completed_submit'] is True
    assert result['claims']['required_attribute']['status'] == final_status
    assert result['claims']['required_attribute']['evidence_quote'] == (TEXT if final_status == 'VERIFIED' else '')
    assert len(requests) == 2
    assert result['usage']['search_calls'] == result['usage']['fetch_calls'] == 0


def test_candidate_is_generic_and_can_include_a_following_name():
    quote = 'Anduril builds autonomous defense systems.'
    text = quote + ' About Anduril Industries'
    assert inv._company_bound_quote_candidate(quote, text, {'andurilindustries'}) == text
    assert inv._company_bound_quote_candidate(quote, quote, {'andurilindustries'}) == ''


def test_optional_named_candidates_respect_existing_message_limit():
    document = {
        'instruction': 'x' * 29000,
        'rejected_findings': [{
            'target': 'required_attribute', 'reason': inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,
            'company_bound_quote_candidate': TEXT,
            'rejected_supporting_evidence': [{
                'supporting_evidence_slot': slot,
                'reason': inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,
                'source_url': URL, 'source_context': TEXT,
                'company_bound_quote_candidate': TEXT,
            } for slot in (1, 2)],
        }],
    }
    encoded = inv._bounded_message_json(document)
    assert len(encoded) < inv.OPENROUTER_MAX_CONTENT_CHARS
    feedback = json.loads(encoded)
    assert [r['supporting_evidence_slot'] for r in feedback['rejected_findings'][0]['rejected_supporting_evidence']] == [1, 2]
    assert all(r['reason'] == inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON for r in feedback['rejected_findings'][0]['rejected_supporting_evidence'])
    assert document['rejected_findings'][0]['company_bound_quote_candidate'] == TEXT


@pytest.mark.parametrize('hint_key', ['company_bound_quote_candidate', 'source_context'])
def test_optional_repair_hint_never_displaces_existing_source_text(hint_key):
    source_text = 'original source evidence ' * 100
    rejection = {
        'supporting_evidence_slot': 2,
        'reason': inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,
        hint_key: 'optional hint ' * 150,
    }
    document = {
        'instruction': 'x' * 28000,
        'prefetched_sources': [{'url': URL, 'text': source_text}],
        'rejected_findings': [{'target': 'required_attribute',
                               'rejected_supporting_evidence': [rejection]}],
    }
    feedback = json.loads(inv._bounded_message_json(document))
    assert feedback['prefetched_sources'][0]['text'] == source_text
    assert feedback['rejected_findings'][0]['rejected_supporting_evidence'] == [{
        'supporting_evidence_slot': 2,
        'reason': inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,
    }]
    assert rejection[hint_key] == 'optional hint ' * 150


def test_both_support_failures_are_exposed_in_one_existing_correction(monkeypatch):
    requests = []
    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        assert len(requests) <= 2
        if len(requests) == 1:
            revised = finding(TEXT, supporting_evidence_url_1=URL,
                              supporting_evidence_quote_1=QUOTE,
                              supporting_evidence_url_2=URL,
                              supporting_evidence_quote_2=QUOTE)
        else:
            feedback = next(json.loads(m['content']) for m in reversed(payload['messages']) if m.get('role') == 'tool')
            rejected = feedback['rejected_findings'][0]
            assert rejected['supporting_evidence_slot'] == 1
            supports = rejected['rejected_supporting_evidence']
            assert [s['supporting_evidence_slot'] for s in supports] == [1, 2]
            assert all(s['company_bound_quote_candidate'] == TEXT for s in supports)
            revised = finding(TEXT, supporting_evidence_url_1=URL,
                              supporting_evidence_quote_1=supports[0]['company_bound_quote_candidate'],
                              supporting_evidence_url_2=URL,
                              supporting_evidence_quote_2=supports[1]['company_bound_quote_candidate'])
        return 200, _response('submit_findings', {'findings': [revised]}, call=len(requests))
    async def no_research(*_args, **_kwargs):
        raise AssertionError('Citation repair must use the existing source')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    monkeypatch.setenv('EXA_API_KEY', 'test')
    monkeypatch.setattr(inv, '_post_json', post)
    monkeypatch.setattr(inv, '_search_web', no_research)
    monkeypatch.setattr(inv, '_fetch_page', no_research)
    monkeypatch.setattr(inv, 'time', _Clock())
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={'name': IDENTITY['submitted_name'], 'website': 'https://kratosdefense.com'},
        targets=('required_attribute',), requested_attribute='Manufactures military hardware.',
        prior_observations={'observed_company_name': NAME, 'observed_company_website': 'https://kratosdefense.com', 'submitted_source_urls': [URL]},
        verified_homepage_identity={'normalized_name': 'kratosdefense', 'registrable_dns_domain': 'kratosdefense.com'},
        prefetched_pages={URL: {'final_url': URL, 'text': TEXT}},
    ))
    assert result['_completed_submit'] is True
    assert result['claims']['required_attribute']['status'] == 'VERIFIED'
    assert len(result['claims']['required_attribute']['supporting_evidence']) == 2
    assert len(requests) == 2
    assert result['usage']['search_calls'] == result['usage']['fetch_calls'] == 0
