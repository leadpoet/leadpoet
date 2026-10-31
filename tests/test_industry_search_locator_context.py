"""Search prose can suggest a bounded URL, but cannot establish a fact."""
import asyncio

from qualification.scoring import company_evidence_investigator as c


def search(monkeypatch, content, annotations=()):
    async def post(*args, **kwargs):
        return 200, {'choices': [{'message': {
            'content': content, 'annotations': list(annotations),
        }}]}
    monkeypatch.setattr(c, '_post_json', post)
    return asyncio.run(c._search_industry_locators(
        None, 'managed infrastructure', key='test',
        company_locator={'name': 'Coreworks', 'website': 'https://coreworks.ai'},
        include_domains=('coreworks.ai',)))


def test_answer_locator_and_description_survive_without_provider_annotation(monkeypatch):
    url = 'https://coreworks.ai/partners/cloud'
    content = ('1. Cloud partner services\n'
               f'   - URL: `{url}`\n'
               '   - Description: Managed operations and infrastructure maintenance.')
    result = search(monkeypatch, content)
    assert [r['url'] for r in result['results']] == [url]
    assert 'Managed operations' in result['results'][0]['discovery_excerpt']
    assert result['notice'] == 'discovery_only_not_evidence'
    assert '_source_pages' not in result


def test_answer_locators_keep_existing_domain_and_public_url_guards(monkeypatch):
    result = search(monkeypatch, '\n\n'.join([
        'https://other.example/claimed-fit',
        'https://coreworks.ai.evil.example/claimed-fit',
        'https://user@coreworks.ai/claimed-fit',
        'https://127.0.0.1/claimed-fit',
        'http://coreworks.ai/claimed-fit',
        'https://coreworks.ai:444/claimed-fit',
        'https://coreworks.ai/valid',
    ]))
    assert [r['url'] for r in result['results']] == ['https://coreworks.ai/valid']


def test_answer_locator_order_is_bounded_and_citations_are_deduplicated(monkeypatch):
    urls = [f'https://coreworks.ai/source-{n}' for n in range(20)]
    citations = [{'type': 'url_citation', 'url_citation': {
        'url': urls[0], 'title': 'Source title'}}]
    result = search(monkeypatch, '\n\n'.join(f'{u} — possible source' for u in urls), citations)
    assert [r['url'] for r in result['results']] == urls[:c.MAX_INDUSTRY_SEARCH_RESULTS]
    assert result['results'][0]['title'] == 'Source title'
    assert all(len(r['discovery_excerpt']) <= c.MAX_SEARCH_LOCATOR_EXCERPT_CHARS
               for r in result['results'])


def test_annotation_only_sources_still_work(monkeypatch):
    url = 'https://coreworks.ai/verified-url-not-verified-fact'
    result = search(monkeypatch, 'No URL was included in this answer.', [
        {'type': 'url_citation', 'url_citation': {'url': url, 'title': 'Source'}}])
    assert result['results'] == [{'url': url, 'title': 'Source'}]


def test_invalid_annotation_prefix_does_not_hide_valid_source(monkeypatch):
    url = 'https://coreworks.ai/source'
    invalid = [{'type': 'url_citation', 'url_citation': {'url': 'https://other.example'}}] * 20
    result = search(monkeypatch, None, invalid + [
        {'type': 'url_citation', 'url_citation': {'url': url}}])
    assert result['results'] == [{'url': url, 'title': ''}]


def test_discovery_context_uses_existing_excerpt_and_message_bounds(monkeypatch):
    content = '\n\n'.join(f'https://coreworks.ai/{n}\n' + '\x01' * 900
                          for n in range(c.MAX_INDUSTRY_SEARCH_RESULTS))
    result = search(monkeypatch, content)
    assert len(result['results']) == c.MAX_INDUSTRY_SEARCH_RESULTS
    assert all('\x01' not in r['discovery_excerpt'] for r in result['results'])
    assert len(c._bounded_message_json(result)) < c.OPENROUTER_MAX_CONTENT_CHARS


def test_compact_list_binds_each_description_to_its_own_url(monkeypatch):
    first = 'https://coreworks.ai/migration'
    second = 'https://coreworks.ai/operations'
    content = (f'- {first} — Application migration and conversion.\n'
               f'- {second} — Managed infrastructure operations and monitoring.')
    result = search(monkeypatch, content)
    a, b = result['results']
    assert first in a['discovery_excerpt']
    assert 'Application migration' in a['discovery_excerpt']
    assert second not in a['discovery_excerpt']
    assert second in b['discovery_excerpt']
    assert 'Managed infrastructure operations' in b['discovery_excerpt']
    assert first not in b['discovery_excerpt']
    assert 'Application migration' not in b['discovery_excerpt']
    assert '_source_pages' not in result


def test_long_intro_cannot_hide_a_later_locator_description(monkeypatch):
    url = 'https://coreworks.ai/operations'
    result = search(monkeypatch, 'Background. ' * 100 + f'\n{url} — Managed operations.')
    assert url in result['results'][0]['discovery_excerpt']
    assert 'Managed operations' in result['results'][0]['discovery_excerpt']


def test_locator_keeps_its_own_short_leading_description(monkeypatch):
    url = 'https://coreworks.ai/operations'
    result = search(monkeypatch, f'Managed infrastructure operations: {url}')
    assert 'Managed infrastructure operations' in result['results'][0]['discovery_excerpt']
