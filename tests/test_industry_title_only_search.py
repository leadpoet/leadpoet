"""Title-only search results remain locators, not completed source reads.

The 45-character title/text pair is from a retained public search result.
All transports and model choices below are fixtures, not accuracy evidence.
"""
import asyncio
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_industry_document_wrapper_refresh import loop, TITLE, WRAPPER, FACT
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_refusal_fallback import FEATURES, IDENTITY
from tests.test_native_industry_review import install


_NATIVE_SEARCH = c._search_web


def test_exact_saved_45_character_result_is_locator_only(monkeypatch):
    url = 'https://vytexwindows.com/3d-flip-book/specialty-windows/'
    record = {'id': url, 'url': url, 'title': TITLE, 'text': TITLE,
              'extras': {'links': []}}
    assert len(record['text']) == 45
    async def post(*args, **kwargs):
        return 200, {'results': [record]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'product documentation', key='offline',
        include_domains=('vytexwindows.com',), discovery_highlights=True, read_source_text=True))
    assert result['results'] == [{'url': url, 'title': TITLE}]
    assert result['_source_pages'] == {}


@pytest.mark.parametrize('text', [
    TITLE,
    '\n  ' + TITLE + '\t',
    '# ' + TITLE,
    '## ' + TITLE + ' ##',
    TITLE + '\n\n' + TITLE,
])
def test_saved_title_only_body_stays_unread_and_keeps_link_locators(monkeypatch, text):
    async def post(*args, **kwargs):
        return 200, {'results': [{
            'url': WRAPPER, 'title': TITLE, 'text': text,
            'extras': {'links': ['https://www.coreworks.ai/documents/manual.pdf']},
        }]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'product documentation', key='offline',
        include_domains=('coreworks.ai',), discovery_highlights=True, read_source_text=True))
    assert result['_source_pages'] == {}
    assert result['results'] == [
        {'url': WRAPPER, 'title': TITLE},
        {'url': 'https://www.coreworks.ai/documents/manual.pdf',
         'title': 'Linked from ' + WRAPPER},
    ]
    assert result['notice'] == 'discovery_only_not_evidence'


@pytest.mark.parametrize('title,text', [
    ('Product documentation', FACT),
    ('Company announcement', 'Coreworks launches account workflows.'),
    ('Coreworks launches account workflows.', 'Coreworks launches account workflows. Available today.'),
    ('Product documentation', 'Product documentation\n\n' + FACT),
    ('Product documentation', '# Product documentation\n\n' + FACT),
    # _plain_text loses line breaks. Keep ambiguous internal heading tokens
    # rather than guessing that they have no additional source content.
    ('Product documentation', 'Product documentation\n\n# Product documentation'),
    ('', FACT),
    (None, FACT),
])
def test_short_facts_or_heading_with_body_remain_source_text(monkeypatch, title, text):
    async def post(*args, **kwargs):
        return 200, {'results': [{'url': WRAPPER, 'title': title, 'text': text}]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'product documentation', key='offline',
        include_domains=('coreworks.ai',), discovery_highlights=True, read_source_text=True))
    assert result['_source_pages'] == {WRAPPER: c._plain_text(text)}


def test_meaningful_title_is_kept_as_discovery_not_proof(monkeypatch):
    title = 'Coreworks launches account workflows.'
    async def post(*args, **kwargs):
        return 200, {'results': [{'url': WRAPPER, 'title': title, 'text': title}]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'product documentation', key='offline',
        include_domains=('coreworks.ai',), discovery_highlights=True, read_source_text=True))
    assert result['results'] == [{'url': WRAPPER, 'title': title}]
    assert result['_source_pages'] == {}


@pytest.mark.parametrize('selected', [False, True])
def test_native_unread_locator_has_no_automatic_read_or_extra_budget(monkeypatch, selected):
    result, reads, documents, _, calls, _ = loop(monkeypatch, selected=selected)
    # The title remains in discovery, but not in the actual quote packet.
    first = calls['model'][0]['messages']
    source_urls = [p['source_url'] for msg in first
                   if msg.get('role') == 'user' and msg['content'].startswith('{"untrusted_source_pages"')
                   for p in json.loads(msg['content'])['untrusted_source_pages']]
    assert WRAPPER not in source_urls
    assert TITLE in json.dumps(first)
    if selected:
        assert reads == [WRAPPER] and len(documents) == 1
        assert result[c.PRIVATE_FETCHED_PAGES_KEY][WRAPPER]['text'] != TITLE
        assert result['usage']['fetch_calls'] == 2
    else:
        assert reads == documents == []
        assert WRAPPER not in result[c.PRIVATE_FETCHED_PAGES_KEY]
        assert result['usage']['fetch_calls'] == 0
    assert result['usage']['search_calls'] == 1
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS


def test_retained_exact_read_wins_and_does_not_become_unread(monkeypatch):
    result, reads, documents, _, _, _ = loop(monkeypatch, prefetch=FACT)
    assert reads == documents == []
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][WRAPPER]['text'] == FACT


def test_title_from_an_existing_exact_read_is_preserved(monkeypatch):
    result, reads, documents, _, _, _ = loop(monkeypatch, prefetch=TITLE)
    assert reads == documents == []
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][WRAPPER]['text'] == TITLE


def test_all_target_choices_have_identical_transport_budget_before_and_after(monkeypatch):
    """Replay declared choices, not the model's unobserved counterfactual."""
    outcomes = []
    for old_admission in (True, False):
        with monkeypatch.context() as patch:
            targets = ('industry', 'stage', 'required_attribute')
            findings = [_finding(t, status='UNPROVEN', evidence_url='', evidence_quote='')
                        for t in targets]
            _, calls, _, _ = install(patch, decision='unavailable',
                missing=['Account workflows'], finding=findings)
            review_post = c._post_json
            if old_admission:
                patch.setattr(c, '_search_text_is_title_only', lambda *args: False)
            transports = []
            choices = iter([
                ('fetch_page', {'url': 'https://www.coreworks.ai/owner'}),
                ('fetch_page', {'url': 'https://www.coreworks.ai/ownership-update'}),
                ('fetch_page', {'url': 'https://www.coreworks.ai/operations'}),
                ('search_web', {'query': 'Coreworks remaining supplied capability'}),
                ('submit_findings', {'findings': findings}),
            ])

            async def html(session, url, **kwargs):
                target = kwargs.get('params', {}).get('url', url)
                transports.append(('fetch', target))
                return 200, target, FEATURES

            async def post(session, url, **kwargs):
                payload = kwargs['payload']
                if url == 'https://api.exa.ai/search':
                    transports.append(('search', payload))
                    return 200, {'results': [{'url': WRAPPER, 'title': TITLE, 'text': TITLE}]}
                if 'response_format' in payload:
                    return await review_post(session, url, **kwargs)
                calls['model'].append(payload)
                return 200, _response(*next(choices))

            patch.setattr(c, '_post_json', post)
            patch.setattr(c, '_fetch_bounded_html', html)
            patch.setattr(c, '_search_web', _NATIVE_SEARCH)
            result = asyncio.run(c.investigate_company_evidence(
                company_locator={'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'},
                targets=targets, requested_industry='Software',
                requested_stage='Private Equity', requested_attribute='Supplies account workflows',
                requested_product_service='Supplies account workflows',
                verified_homepage_identity=IDENTITY, positive_semantic_review=True,
                prefetched_pages={'https://www.coreworks.ai/': {
                    'final_url': 'https://www.coreworks.ai/', 'text': FEATURES}},
                prior_observations={'submitted_source_urls': ['https://www.coreworks.ai/']}))
            assert result['_completed_submit'] is True
            assert set(result['claims']) == set(targets)
            assert all(f['status'] == 'UNPROVEN' for f in result['claims'].values())
            usage = result['usage']
            assert usage['fetch_calls'] <= c.MAX_FETCH_CALLS
            assert usage['search_calls'] + usage['fetch_calls'] <= c.MAX_SEARCH_CALLS + c.MAX_FETCH_CALLS
            outcomes.append((transports, usage['fetch_calls'], usage['search_calls']))
    assert outcomes[0] == outcomes[1]
    assert len([x for x in outcomes[1][0] if x[0] == 'fetch']) == 3


@pytest.mark.parametrize('title,body,expected', [
    ('#', ('# ' * 12000)[:-1] + 'X', False),
    (TITLE, (('# ' + TITLE + ' ') * 500)[:23999] + 'X', False),
    (TITLE, ((TITLE + ' ') * 521).ljust(24000), True),
    ('A' * 23998, 'A' * 23998 + ' X', False),
], ids=['hash-near-miss', 'ordinary-near-miss', 'ordinary-repeat', 'long-title-near-miss'])
def test_untrusted_maximum_body_matching_is_bounded(title, body, expected):
    """Kill only the owned offline test child if a regex regression hangs."""
    # Compile the actual two pure functions, without importing provider code.
    source = Path(c.__file__).read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'_visible_quote_surface', '_search_text_is_title_only'}]
    child = '''import ast,json,re,socket,sys,typing
def denied(*a,**k):raise AssertionError('network forbidden')
socket.socket.connect=socket.getaddrinfo=denied
doc=json.loads(sys.stdin.read())
namespace={'Any':typing.Any,'_IDENTITY_LINK_CONTEXT_MARKER':doc['marker'],'re':re}
exec(compile(doc['source'],'actual-source-helper','exec'),namespace)
assert namespace['_search_text_is_title_only'](doc['body'],doc['title']) is doc['expected']
'''
    payload = {'source': ast.unparse(ast.Module(body=nodes, type_ignores=[])),
               'marker': c._IDENTITY_LINK_CONTEXT_MARKER,
               'title': title, 'body': body, 'expected': expected}
    # The timed helper uses only stdlib; exclude image sitecustomize startup.
    result = subprocess.run([sys.executable, '-S', '-c', child], input=json.dumps(payload),
                            text=True, capture_output=True, timeout=2)
    assert result.returncode == 0, result.stderr
