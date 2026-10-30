"""Industry discovery uses provider filters, not unsupported query operators."""
import asyncio

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as investigator


@pytest.mark.parametrize('scoped', [False, True])
def test_native_search_keeps_query_budget_and_discovery_only_evidence(monkeypatch, scoped):
    requests = []
    query = 'site:coreworks.ai cloud migration mainframe services'

    async def post(session, url, *, headers, payload):
        requests.append(payload)
        validate_operation_request('exa.search', payload)
        return 200, {'results': [
            {'url': 'https://investor.coreworks.ai/project', 'title': 'Migration project',
             'text': 'Search text is not proof'},
            {'url': 'https://unrelated.example/project', 'title': 'Different supplier'},
            {'url': 'https://coreworks.ai.attacker.example/project', 'title': 'Lookalike'},
        ]}

    monkeypatch.setattr(investigator, '_post_json', post)
    options = {'include_domains': ('coreworks.ai',)} if scoped else {}
    result = asyncio.run(investigator._search_web(None, query, key='test-key', **options))
    expected = {'query': query, 'type': 'auto', 'numResults': investigator.MAX_SEARCH_RESULTS}
    if scoped:
        expected['includeDomains'] = ['coreworks.ai']
    assert requests == [expected]
    assert result['notice'] == 'discovery_only_not_evidence'
    assert len(result['results']) == (1 if scoped else 3)
    assert all(set(item) == {'url', 'title'} for item in result['results'])
    assert result['results'][0]['url'] == 'https://investor.coreworks.ai/project'


@pytest.mark.parametrize('funding_source', ['host', 'miner_key'])
@pytest.mark.parametrize('source_text', [False, True])
@pytest.mark.parametrize('employer_path', [False, True])
def test_domain_filter_survives_native_broker_and_preserves_billing(monkeypatch, funding_source, source_text, employer_path):
    import json
    from tests.lab_arena.score_compat_capability_test import routed_broker, score_context
    from tests.lab_arena.test_deepline_catalog import frozen, row
    from tests.lab_arena.test_lab_arena_broker import FakeTransport

    catalog = frozen(row('exa_search', provider='exa', categories=['research'],
        inputSchema={'type': 'object', 'properties': {
            'query': {'type': 'string'}, 'type': {'type': 'string'},
            'numResults': {'type': 'integer'},
            'includeDomains': {'type': 'array', 'items': {'type': 'string'}},
            'contents': {'type': 'object'},
        }, 'required': ['query']}))
    scope = 'job-boards.greenhouse.io/coreworks' if employer_path else 'coreworks.ai'
    source = 'https://' + scope + ('/jobs/123' if employer_path else '/project')
    transport = FakeTransport([(200, {
        'status': 'completed', 'billing': {'credits_charged': 0.17 if source_text else 0.1},
        'result': {'data': {'results': [{'url': source, 'title': 'Project',
                                       'text':'Coreworks supplies this service.'}]}},
    })])
    broker, ledger, _ = routed_broker(funding_source=funding_source, transport=transport)
    receipts = []

    async def post(session, url, *, headers, payload):
        result = broker.execute(score_context(catalog), operation_id='exa.search',
            parameters=payload, action_sequence=1, timeout_ms=30_000)
        receipts.append(result.call)
        return result.status, json.loads(result.body)

    monkeypatch.setattr(investigator, '_post_json', post)
    result = asyncio.run(investigator._search_web(None, 'Coreworks mainframe services',
        key='test-key', include_domains=(scope,),
        discovery_highlights=source_text, read_source_text=source_text))
    assert result['results'] == [{'url': source, 'title': 'Project'}]
    assert len(transport.sent) == len(ledger.calls) == len(receipts) == 1
    wire = json.loads(transport.sent[0]['body'])
    assert wire['payload']['includeDomains'] == [scope]
    assert wire['payload']['numResults'] == (10 if source_text else investigator.MAX_SEARCH_RESULTS)
    assert wire['payload']['type'] == 'auto'
    if source_text:
        assert wire['payload']['contents'] == {'text':{'maxCharacters':24000},'maxAgeHours':0,
            'extras':{'links':investigator.MAX_HOMEPAGE_NAVIGATION_LOCATORS}}
        assert result['_source_pages'] == {source:'Coreworks supplies this service.'}
    else:
        assert '_source_pages' not in result
    assert receipts[0]['funding_source'] == funding_source
    assert receipts[0]['outcome'] == 'settled'
    assert receipts[0]['actual_microusd'] == (17_000 if source_text else 10_000)
