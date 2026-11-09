"""Keep full-criterion decisions without granting new source or identity authority."""
import asyncio
import json
from pathlib import Path
import time

import pytest

from qualification.scoring import company_evidence_investigator as c, lead_scorer


CAPTURE = json.loads((Path(__file__).parent / 'fixtures/company_evidence/ait_industry_attribution.json').read_text())
URL = CAPTURE['evidence'][0]['url']
QUOTE = CAPTURE['evidence'][0]['quote']
DOMAIN = 'aitworldwide.com'


def _anchor(*, complete=True):
    home, web = CAPTURE['identity'], CAPTURE['web_identity']
    return {**{key: web[key] for key in ('submitted_name', 'observed_name', 'submitted_domain',
                                        'observed_domain', 'submitted_linkedin_slug', 'observed_linkedin_slug')},
            'verified_name': (web if complete else home)['observed_name'],
            'verified_domain': home['observed_domain'],
            'verified_linkedin_slug': (web if complete else home)['observed_linkedin_slug']}


def _review(monkeypatch, *, decision='mismatch', anchor=None, final_url=URL,
            page=QUOTE, source_url=URL, role='supplier_operator'):
    calls = []

    async def post(_session, _url, **kwargs):
        calls.append(kwargs['payload'])
        spans = [span for message in kwargs['payload']['messages'][2:]
                 for source in json.loads(message['content'])['untrusted_source_pages']
                 for span in source['spans']]
        result = {'decision': decision, 'role': role,
                  'observed_activity': 'North American truckload brokerage',
                  'missing_requirements': [] if decision == 'match' else ['Requested activity'],
                  'reason': CAPTURE['reason'], 'evidence_span_ids': [spans[0]['span_id']]}
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}

    monkeypatch.setattr(c, '_post_openrouter_json', post)
    finding, error, count = asyncio.run(c._review_industry_sources(
        None, model='offline-model', openrouter_key='offline-key', started=time.monotonic(),
        company_locator={'name': CAPTURE['company_name'], 'website': f'https://{DOMAIN}/'},
        criteria={'industry': 'Logistics', 'subindustry': '', 'product_service': 'Freight brokerage'},
        evaluation_date_value='2026-11-09', fetched_pages={source_url: page},
        fetched_final_urls={source_url: final_url} if final_url else {},
        first_party_domains={DOMAIN}, identity_anchor=anchor if anchor is not None else _anchor(),
        identity_names={'aitworldwidelogistics'}))
    assert not error and count == len(calls)
    return finding, calls


@pytest.mark.parametrize('decision,status', [('match', 'VERIFIED'), ('mismatch', 'CONTRADICTED'),
                                              ('unavailable', 'UNPROVEN')])
def test_saved_passage_with_controlled_complete_identity_keeps_semantic_decision(monkeypatch, decision, status):
    finding, calls = _review(monkeypatch, decision=decision)
    assert len(calls) == 1
    assert finding['status'] == status
    assert finding['_industry_review']['decision'] == decision
    if status != 'UNPROVEN':
        assert finding['evidence_quote'] == QUOTE
        assert finding['evidence_url'] == URL
        projected = lead_scorer._project_investigator_industry({'attribute_satisfied': False}, finding)
        assert projected['industry_matches'] is (decision == 'match')
        assert projected['attribute_satisfied'] is False
        assert projected['industry_evidence_quote'] == QUOTE


@pytest.mark.parametrize('fault', ['reported_incomplete_homepage', 'missing_identity', 'different_entity',
                                  'foreign_final', 'missing_final', 'foreign_source', 'no_evidence'])
@pytest.mark.parametrize('decision', ['match', 'mismatch'])
def test_semantic_projection_keeps_source_and_identity_guards(monkeypatch, decision, fault):
    options = {'decision': decision, 'anchor': _anchor()}
    if fault == 'reported_incomplete_homepage':
        options['anchor'] = _anchor(complete=False)
    elif fault == 'missing_identity':
        options['anchor'] = {}
    elif fault == 'different_entity':
        options['anchor']['verified_name'] = 'A different logistics company'
    elif fault == 'foreign_final':
        options['final_url'] = 'https://other.example/truckload-solutions/'
    elif fault == 'missing_final':
        options['final_url'] = ''
    elif fault == 'foreign_source':
        options.update(source_url='https://aitworldwide.com.attacker.example/service',
                       final_url='https://aitworldwide.com.attacker.example/service')
    else:
        options['page'] = ''
    finding, _ = _review(monkeypatch, **options)
    assert finding['status'] == 'UNPROVEN'
    assert finding['evidence_quote'] == finding['evidence_url'] == ''


def test_raw_legacy_supplier_mismatch_still_requires_its_original_role_contract():
    finding = {'target': 'industry', 'status': 'CONTRADICTED', 'activity_role': 'supplier_operator',
               'observed_industry': 'Truckload brokerage', 'evidence_url': URL,
               'evidence_quote': QUOTE, 'reason': 'Model assertion',
               'industry_review_decision': 'mismatch', '_industry_review': {'decision': 'mismatch'}}
    checked = c._validated_findings({'findings': [finding]}, targets=('industry',),
        fetched_pages={URL: QUOTE}, fetched_final_urls={URL: URL}, first_party_domains={DOMAIN},
        identity_names={'ait'}, identity_anchor=_anchor())
    assert checked['industry']['status'] == 'UNPROVEN'


def test_full_investigation_carries_frozen_attribute_to_every_industry_packet(monkeypatch):
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import FEATURES, URL as FIXTURE_URL

    seen = []
    def initial_gap(result):
        if not seen:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])
        seen.append(result)

    run, _, _, reviews = install(monkeypatch, mutation=initial_gap)
    original_prepare = c.industry_classification.prepare
    packets = []

    def prepare(*args, **kwargs):
        packet = original_prepare(*args, **kwargs)
        packets.append((kwargs, packet))
        return packet

    monkeypatch.setattr(c.industry_classification, 'prepare', prepare)
    required = 'Supplies analysis and reporting software AND sells recurring subscriptions.'
    result = asyncio.run(run(
        prefetched_pages={FIXTURE_URL: {'final_url': FIXTURE_URL, 'text': FEATURES}},
        prior_observations={'submitted_source_urls': [FIXTURE_URL]}))
    assert result['_completed_submit'] is True
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert len(packets) >= 3
    # Covers initial reviewer, shared research bodies, and final reviewer construction.
    assert all(kwargs['required_attribute'] == packet.required_attribute == required
               for kwargs, packet in packets)
    assert reviews
    system = c._SYSTEM_PROMPT
    assert 'condition that is not separately checked by the required attribute' not in system
    assert 'Do not mark industry UNPROVEN solely because that' not in system
