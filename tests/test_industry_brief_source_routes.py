"""Research route disclosure; fixtures do not establish ownership semantics."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena.operations import validate_operation_request, OPENROUTER_MAX_CONTENT_CHARS
from qualification.scoring import company_evidence_investigator as c
from qualification.scoring import industry_classification as industry
from tests.test_industry_conditional_brand_sources import native_loop, RELATION, PROOF, HOME

FIXTURE=Path(__file__).parent/'fixtures/p8_miter_research_wire_contract.json.gz'
FIXTURE_SHA='4e55bff9c227065f14898b56ae601b982fce1034646a1822ed3e665165e82a1a'


def metadata_message(payload):
    return next(m for m in payload['messages'] if m['role']=='user'
        and m['content'].startswith('Investigate this bounded request.'))


def metadata(payload):
    text=metadata_message(payload)['content']
    return json.loads(text[text.index('{'):])


def new_notice(monkeypatch):
    result, reads, research, reviews=native_loop(monkeypatch)
    assert result['claims']['industry']['status']=='VERIFIED'
    assert len(research)==3 and len(reviews)==1 and len(reads)==2
    return metadata(research[0])['industry_source_research_brief']['notice'], research


def test_actual_wire_brief_matches_enabled_guarded_source_schema_and_all_system_chunks(monkeypatch):
    notice, research=new_notice(monkeypatch)
    assert 'fetch_page.industry_brand_source' in notice
    assert 'exact visible parent relation quote' in notice
    assert 'observed link or complete printed official child URL' in notice
    assert 'industry-only provenance, not ownership or other-dimension proof' in notice
    assert 'final review must prove the as-of operating relation and full criterion' in notice
    for payload in research:
        systems=''.join(m['content']for m in payload['messages']if m['role']=='system')
        assert industry.CRITERION_RULES in systems
        assert 'Existing fetch_page accepts a nullable industry_brand_source' in systems
        assert all(len(m.get('content',''))<=OPENROUTER_MAX_CONTENT_CHARS for m in payload['messages'])
        assert 'industry_brand_source' in next(t['function']['parameters']['properties']for t in payload['tools']if t['function']['name']=='fetch_page')
        validate_operation_request('openrouter.chat', payload)


def test_exact_retained_miter_payload_changes_only_notice_without_trimming_evidence(monkeypatch):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==FIXTURE_SHA
    saved=json.loads(gzip.decompress(FIXTURE.read_bytes()))
    before=saved['payload'];after=deepcopy(before)
    assert hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==saved['provenance']['payload_sha256']
    original=metadata(before);notice,_=new_notice(monkeypatch)
    assert 'fetch_page.industry_brand_source'not in original['industry_source_research_brief']['notice']
    changed=metadata(after);changed['industry_source_research_brief']['notice']=notice
    assert 'fetch_page.industry_brand_source' in changed['industry_source_research_brief']['notice']
    message=metadata_message(after);prefix=message['content'][:message['content'].index('{')]
    message['content']=c._bounded_message_json(changed,prefix=prefix)
    assert metadata(after)==changed  # No secondary context/evidence trim.
    restored=deepcopy(after);restored_message=metadata_message(restored)
    restored_message['content']=metadata_message(before)['content']
    assert restored==before
    assert len(metadata_message(after)['content'])<=len(metadata_message(before)['content'])
    assert len(notice)<=len(original['industry_source_research_brief']['notice'])
    # This hash-bound historical request retains its historical criterion rules.
    # The notice change must not replace them with today's rules. The live-wire
    # test above independently checks the current shared criterion instructions.
    assert [m for m in after['messages'] if m['role']=='system']==[
        m for m in before['messages'] if m['role']=='system']
    assert 'industry_brand_source' in ''.join(m['content']for m in before['messages']if m['role']=='system')
    validate_operation_request('openrouter.chat',before)
    validate_operation_request('openrouter.chat',after)


@pytest.mark.parametrize('relation',[
    'Coreworks holds a minority investment in WorkflowCo.',
    'WorkflowCo is an independent dealer serving Coreworks customers.',
])
def test_conditional_disclosure_is_provenance_not_automatic_ownership_or_fit(monkeypatch,relation):
    # The host admits provenance; a hermetic final-review reply remains UNAVAILABLE.
    # This checks data/control flow, not whether a real model classifies the relation correctly.
    result, reads, research, reviews=native_loop(monkeypatch,parent_text=relation,
        proof={**PROOF,'parent_quote':relation},decision='unavailable')
    assert len(reads)==2
    assert result['claims']['industry']['status']=='UNPROVEN'
    assert 'rebrand'not in result['claims']
    assert 'not ownership' in metadata(research[0])['industry_source_research_brief']['notice']
    final=json.loads(reviews[-1]['messages'][1]['content'])
    assert final['company_locator']['website']=='https://www.coreworks.ai/'
    assert final['conditional_industry_sources']
