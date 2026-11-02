"""Composed industry/HQ instructions must pass the unchanged Arena protocol."""
import asyncio
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_native_industry_review import install
from tests.test_structured_linkedin_headquarters_reuse import (
    ANCHOR, PROFILE, HQ_TEXT, _body, _project,
)


@pytest.mark.parametrize('industry_enabled', [False, True])
@pytest.mark.parametrize('observed_state,expected', [('Texas', 'VERIFIED'), ('Massachusetts', 'UNPROVEN')])
def test_combined_industry_and_bound_hq_reaches_native_judgment(
    monkeypatch, industry_enabled, observed_state, expected,
):
    install(monkeypatch)
    if not industry_enabled:
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL')
    original_post = c._post_json
    requests = []
    finding = _finding('geography')
    finding.update(status='VERIFIED', observed_value='', evidence_url=PROFILE, evidence_quote=HQ_TEXT,
                   observed_country='United States', observed_state=observed_state)
    industry = _finding('industry', observed_value='Geothermal power plants',
                        observed_industry='Energy', observed_subindustry='Geothermal power',
                        activity_role='supplier_operator', evidence_url='https://www.quaise.com/',
                        evidence_quote='Quaise Energy supplies geothermal power plants.')

    async def post(session, url, *, headers, payload):
        validate_operation_request('openrouter.chat', payload)
        requests.append(payload)
        if 'response_format' in payload:
            return await original_post(session, url, headers=headers, payload=payload)
        if payload.get('tool_choice') == {'type': 'function', 'function': {'name': 'search_web'}}:
            return 200, _response('search_web', {'query': 'Quaise Energy headquarters and products'})
        tool = next(t['function'] for t in payload['tools'] if t['function']['name'] == 'submit_findings')
        targets = tool['parameters']['properties']['findings']['items']['properties']['target']['enum']
        return 200, _response('submit_findings', {'findings': [f for f in [industry, finding] if f['target'] in targets]})

    monkeypatch.setattr(c, '_post_json', post)
    homepage = 'https://www.quaise.com/'
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={'name': 'Quaise Energy', 'website': homepage, 'linkedin': PROFILE},
        targets=('industry', 'geography'), requested_industry='Energy',
        requested_product_service='Geothermal power plants', requested_geography='United States',
        verified_homepage_identity=ANCHOR,
        prior_observations={'submitted_source_urls': [homepage, PROFILE]},
        prefetched_pages={homepage: {'final_url': homepage, 'text': 'Quaise Energy supplies geothermal power plants.'},
                          PROFILE: {'final_url': PROFILE, 'text': HQ_TEXT}},
        structured_headquarters_evidence=_project(_body()),
    ))
    assert requests, result
    assert result['claims'].get('geography', {}).get('status') == expected, result
    if expected == 'VERIFIED':
        assert result['_completed_submit'], result
    research = [q for q in requests if 'tools' in q]
    assert research
    for request in research:
        system = ''.join(m['content'] for m in request['messages'] if m['role'] == 'system')
        assert c._LITERAL_HEADQUARTERS_INSTRUCTION in system
        assert ('Industry: research the full criterion' in system) is industry_enabled
        if not industry_enabled:
            assert len([m for m in request['messages'] if m['role'] == 'system']) == 1
    assert result['usage']['fetch_calls'] == 0


@pytest.mark.parametrize('text', ['short', 'a' * 32_000, 'word\n' * 16_000, '界' * 64_001])
def test_bounded_system_messages_preserve_every_instruction(text):
    messages = c._bounded_system_messages(text)
    assert ''.join(m['content'] for m in messages) == text
    assert all(m['role'] == 'system' for m in messages)
    validate_operation_request('openrouter.chat', {
        'model': 'openai/gpt-6.1-sol', 'messages': messages, 'max_tokens': 2000,
    })
