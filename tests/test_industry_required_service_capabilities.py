"""Verify delivery of required-service guidance; model accuracy is tested separately."""
import json
from pathlib import Path

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import industry_classification as industry

CONTRASTS = json.loads((Path(__file__).parent / 'fixtures' / 'industry_required_services_contrasts.json').read_text())
RULE = "The manufacturer's descriptive product-line rule does not make separately required services or technical capabilities optional."


@pytest.mark.parametrize('research', [False, True])
@pytest.mark.parametrize('control', CONTRASTS, ids=lambda row: row['case_id'])
def test_required_service_rule_preserves_original_criterion_and_source(control, research):
    criterion = {'industry': 'Technology', 'subindustry': 'Supplied products and services',
                 'product_service': control['criterion']}
    company = {'name': 'Example Systems', 'website': 'https://example.com/', 'linkedin': ''}
    page = {'source_id': 's0', 'source_url': 'https://example.com/products',
            'body': control['source_body']}
    packet = industry.prepare(criterion, company, [page], evaluation_date='2026-10-24')
    request = industry.request_payload(packet, model=industry.DEFAULT_MODEL, research=research)
    validate_operation_request('openrouter.chat', request)
    system = request['messages'][0]['content']
    assert RULE in system
    assert system.index("identifies the manufacturer's product line") < system.index(RULE)
    assert "Examples introduced by 'such as' or 'including' are illustrative" in system
    assert 'proof of products within that line can establish fit without proof of every listed product' in system
    assert 'Explicit logical AND and mandatory wording take precedence' in system
    assert 'Each listed function or capability of a required product or service remains required' in system
    assert 'can supply them through separate offerings' in system
    assert 'do not require one connected service or platform unless the criterion requires that connection' in system
    assert 'must be met by that platform or product' in system
    assert json.loads(request['messages'][1]['content'])['criterion'] == criterion
    assert packet.pages[0][1] == control['source_body']
    assert type(control['expected_match']) is bool
