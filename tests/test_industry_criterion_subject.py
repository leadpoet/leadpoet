"""The research and final requests must preserve the criterion's subject.

These are request-boundary tests. Real-provider controls separately test model
interpretation; no canned model answer is treated as semantic evidence here.
"""
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import industry_classification as industry


@pytest.mark.parametrize('research', [False, True])
@pytest.mark.parametrize('requirement', [
    'Provides managed storage, cloud migration, and database modernization services to enterprise clients.',
    'Provides one connected platform for storage management, cloud migration, and database modernization to enterprise clients.',
])
def test_native_request_preserves_company_and_platform_scope(research, requirement):
    criterion={'industry':'Technology','subindustry':'Enterprise services','product_service':requirement}
    company={'name':'Example Systems','website':'https://example.com','linkedin':''}
    pages=[{'source_id':'s0','source_url':'https://example.com/services','body':
        'Example Systems supplies three independent services to enterprise clients: managed storage, '
        'cloud migration, and database modernization. They do not share one platform.'}]
    packet=industry.prepare(criterion,company,pages,evaluation_date='2026-10-24')
    request=industry.request_payload(packet,model=industry.DEFAULT_MODEL,research=research)
    validate_operation_request('openrouter.chat',request)
    assert json.loads(request['messages'][1]['content'])['criterion']==criterion
    instructions=request['messages'][0]['content']
    assert 'within the connected offering' not in instructions
    assert 'can supply them through separate offerings' in instructions
    assert 'must be met by that platform or product' in instructions
    assert 'supplier or operator role for each required offering' in instructions
    assert 'only the supplied sources' in instructions
    assert 'EXL' not in instructions
    assert packet.pages[0][1]==pages[0]['body']
    assert request['response_format']['json_schema']['schema']['properties']['decision']==industry.SCHEMA['properties']['decision']
