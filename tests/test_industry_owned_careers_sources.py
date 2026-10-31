"""An observed careers link scopes authorship to one ATS employer, not its host."""
import asyncio
import json
import time

import pytest

from qualification.scoring import company_verification as verification
from qualification.scoring import company_evidence_investigator as investigator


BOARD = 'https://job-boards.greenhouse.io/coreworks'
POSTING = BOARD + '/jobs/12345'
HOME = 'https://coreworks.ai/'
IDENTITY = {'submitted_domain': 'coreworks.ai', 'observed_domain': 'coreworks.ai',
            'observed_name': 'Coreworks'}
LOCATORS = [{'url': BOARD, 'label': 'View all positions'}]


def test_verified_homepage_keeps_observed_employer_careers_link(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    html = (
        f'<a href="{BOARD}">View all positions</a>'
        '<a href="https://job-boards.greenhouse.io/another">Partner company</a>'
        '<a href="https://news.example/jobs">Jobs</a>'
        f'<a href="{POSTING}" hidden>Apply</a>'
    )
    observed = verification._homepage_navigation_locators(
        html, final_url=HOME, verified_domain='coreworks.ai',
    )
    assert observed == [{'url': BOARD, 'label': 'View all positions'}]


@pytest.mark.parametrize('enabled,final_url', [(False, HOME), (True, 'https://another.example/')])
def test_careers_locator_requires_new_policy_and_verified_source(monkeypatch, enabled, final_url):
    if enabled:
        monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    else:
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)
    assert verification._homepage_navigation_locators(
        f'<a href="{BOARD}">Careers</a>', final_url=final_url,
        verified_domain='coreworks.ai',
    ) == []


def bound(url=POSTING, final_url=POSTING, *, identity=IDENTITY, locators=LOCATORS):
    return investigator._industry_source_is_bound(
        url, final_url, domains={'coreworks.ai'}, identity=identity,
        careers_locators=locators,
    )


def test_company_observed_careers_link_binds_only_its_exact_ats_employer():
    assert bound()
    assert not investigator._independently_bound_first_party_url(
        POSTING, {'coreworks.ai'}, IDENTITY,
    )  # No global authority is granted to greenhouse.io.


@pytest.mark.parametrize('changes', [
    {'locators': []},
    {'locators': [{'url': BOARD, 'label': 'Partner company'}]},
    {'locators': [{'url': BOARD + '-other', 'label': 'Careers'}]},
    {'identity': {'submitted_domain': 'coreworks.ai'}},
    {'url': POSTING.replace('/coreworks/', '/another/')},
    {'final_url': POSTING.replace('/coreworks/', '/another/')},
    {'final_url': 'https://news.example/story'},
    {'url': POSTING.replace('https://', 'https://user:password@')},
    {'url': POSTING.replace('greenhouse.io', 'greenhouse.io.evil.example')},
    {'url': BOARD + '/jobs/../another/123'},
])
def test_ats_name_redirect_or_unbound_claim_cannot_establish_company_authorship(changes):
    assert not bound(**changes)


def test_owned_source_extension_preserves_ordinary_first_party_binding():
    assert bound(HOME + 'product', HOME + 'product', locators=[])
    assert not bound(HOME + 'product', 'https://another.example/product')


def test_markdown_board_links_survive_as_locators_but_never_as_quote_proof(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    raw = f'Open positions: [Electrical Engineer]({POSTING}).'
    text = investigator._plain_text(raw)
    assert POSTING in text
    assert investigator._visible_quote_surface(text).strip() == 'Open positions: Electrical Engineer.'
    assert not investigator._quote_occurs(POSTING, text)
    assert not bound(locators=[])  # A link inside a fetched board is not ownership.


def test_markdown_locator_retention_does_not_change_frozen_policy(monkeypatch):
    monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)
    assert investigator._plain_text(f'[Electrical Engineer]({POSTING})') == 'Electrical Engineer'


def test_markdown_locator_retention_excludes_hidden_and_unsafe_links(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    hidden = POSTING + '999'
    raw = (f'<div hidden>[Apply]({hidden})</div><p>[Apply]({POSTING})</p>'
           '<p>[Unsafe](https://user:secret@example.com/job)</p>')
    text = investigator._plain_text(raw)
    locators = text.partition(investigator._IDENTITY_LINK_CONTEXT_MARKER)[2]
    assert POSTING in locators and hidden not in locators
    assert 'user:secret' not in locators


@pytest.mark.parametrize('linked', [False, True])
def test_review_receives_only_bound_job_body_and_preserves_its_exact_support(monkeypatch, linked):
    seen = []
    home_text = 'Coreworks develops computing products.'
    job_text = 'Coreworks engineers design our accelerator silicon and inference servers.'

    async def post(session, url, *, headers, payload):
        pages = [page for message in payload['messages'][2:]
                 for page in json.loads(message['content'])['untrusted_source_pages']]
        seen.extend(page['source_url'] for page in pages)
        has_job = POSTING in seen
        spans = [span['span_id'] for page in pages for span in page['spans']]
        answer = {'observed_activity': 'Computing hardware', 'role': 'supplier_operator',
                  'reason': 'Offline transport and evidence-binding fixture.',
                  'decision': 'match' if has_job else 'unavailable',
                  'missing_requirements': [] if has_job else ['Server hardware'],
                  'evidence_span_ids': list(reversed(spans))}
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}

    monkeypatch.setattr(investigator, '_post_json', post)
    finding, failure, calls = asyncio.run(investigator._review_industry_sources(
        None, model='openai/gpt-6.1-sol', openrouter_key='offline-key', started=time.monotonic(),
        company_locator={'name': 'Coreworks', 'website': HOME},
        criteria={'industry': 'Hardware', 'subindustry': '', 'product_service': 'Computing servers'},
        fetched_pages={HOME: home_text, POSTING: job_text},
        fetched_final_urls={HOME: HOME, POSTING: POSTING},
        first_party_domains={'coreworks.ai'}, identity_anchor=IDENTITY,
        identity_names={'coreworks'}, careers_locators=LOCATORS if linked else (),
        evaluation_date_value='2026-10-24',
    ))
    assert failure == '' and calls == 1
    assert (POSTING in seen) is linked
    assert finding['status'] == ('VERIFIED' if linked else 'UNPROVEN')
    if linked:
        # An ATS citation cannot replace the independently attributable primary.
        assert finding['evidence_url'] == HOME
        assert finding['supporting_evidence'] == [{'url': POSTING, 'quote': job_text}]
        assert finding['_industry_review']['careers_source_locators'] == LOCATORS


@pytest.mark.parametrize('linked', [False, True])
def test_unresolved_retry_reuses_owned_job_text_only_with_current_observed_link(monkeypatch, linked):
    from qualification.scoring import lead_scorer
    from tests.test_unresolved_industry_retry_sources import run_investigation, IDENTITY as saved_identity

    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    job = 'https://job-boards.greenhouse.io/unibuddy/jobs/12345'
    navigation = [{'url': 'https://job-boards.greenhouse.io/unibuddy', 'label': 'Careers'}]
    cache = {}
    pages = {'https://unibuddy.com/': {'final_url': 'https://unibuddy.com/', 'text': 'Unibuddy company.'},
             job: {'final_url': job, 'text': 'Unibuddy engineers build campus software.'}}
    run_investigation(monkeypatch, pages=pages, cache=cache,
                      navigation=navigation if linked else (),
                      review_changes={'careers_source_locators': navigation})
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, saved_identity)
    # A claimed link in review metadata cannot substitute for the caller's
    # current, independently observed homepage navigation.
    assert (job in retained) is linked


@pytest.mark.parametrize('linked', [False, True])
def test_native_loop_and_saved_dimension_keep_only_bound_employer_evidence(monkeypatch, linked):
    from qualification.scoring import lead_scorer
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import FEATURES

    run, calls, _, reviews = install(monkeypatch)
    job_text = 'Coreworks engineers implement our account workflow and reporting product.'
    result = asyncio.run(run(
        requested_product_service='Supplies reporting and account workflows',
        homepage_navigation_locators=LOCATORS if linked else (),
        prior_observations={'submitted_source_urls': [HOME]},
        prefetched_pages={HOME: {'final_url': HOME, 'text': FEATURES}},
        retained_industry_pages={POSTING: {'final_url': POSTING, 'text': job_text}},
    ))
    assert result['_completed_submit'] is True
    source_pages = [page for request in reviews for message in request['messages'][2:]
                    for page in json.loads(message['content'])['untrusted_source_pages']]
    assert any(page['source_url'] == POSTING for page in source_pages) is linked
    finding = result['claims']['industry']
    projected = lead_scorer._project_investigator_industry({'attribute_satisfied': False}, finding)
    saved = json.loads(json.dumps(lead_scorer._dimension_web_evidence(projected, 'industry')))
    assert saved['url'] == HOME and saved['quote'] == FEATURES
    support = saved.get('supporting_evidence', [])
    assert ({'url': POSTING, 'quote': job_text} in support) is linked
    assert projected['attribute_satisfied'] is False
    assert result['usage']['fetch_calls'] <= investigator.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] <= investigator.MAX_SEARCH_CALLS
