"""A second search index supplies locators, never evidence or a verdict."""
import asyncio
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_refusal_fallback import URL

SOURCE = 'https://www.coreworks.ai/workflows'


def test_alternative_search_keeps_only_bound_provider_citation_locators(monkeypatch):
    seen = []
    async def post(session, url, *, headers, payload):
        validate_operation_request('openrouter.chat', payload)
        seen.append(payload)
        urls = [SOURCE, SOURCE, 'https://other.example/facts',
                'https://coreworks.ai.evil.example/facts', 'http://coreworks.ai/facts',
                'https://user@coreworks.ai/facts', 'https://127.0.0.1/facts']
        return 200, {'choices': [{'message': {
            'content': 'VERIFIED. Invented proof: https://coreworks.ai/not-cited',
            'annotations': [{'type': 'url_citation', 'url_citation': {
                'url': value, 'title': 'Workflow documentation'}} for value in urls],
        }}]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_industry_locators(
        None, 'account workflows', key='test',
        company_locator={'name': 'Coreworks', 'website': URL},
        include_domains=('coreworks.ai',)))
    assert result == {'results': [{'url': SOURCE, 'title': 'Workflow documentation'}],
                      'notice': 'discovery_only_not_evidence'}
    assert seen[0]['model'] == 'perplexity/sonar'
    assert 'VERIFIED' not in json.dumps(result)


@pytest.mark.parametrize('status, body, expected', [
    (503, {}, RuntimeError),
    (200, {'error': 'failed'}, RuntimeError),
    (200, {'choices': []}, ValueError),
    (200, {'choices': [{'message': {'refusal': 'No'}}]}, ValueError),
    (200, {'choices': [{'message': {'annotations': 'invalid'}}]}, ValueError),
])
def test_alternative_search_errors_are_not_negative_company_evidence(monkeypatch, status, body, expected):
    async def post(*args, **kwargs):
        return status, body
    monkeypatch.setattr(c, '_post_json', post)
    with pytest.raises(expected):
        asyncio.run(c._search_industry_locators(
            None, 'account workflows', key='test',
            company_locator={'name': 'Coreworks', 'website': URL},
            include_domains=('coreworks.ai',)))
