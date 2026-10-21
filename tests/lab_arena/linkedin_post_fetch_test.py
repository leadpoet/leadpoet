"""Native LinkedIn post text keeps the unchanged page-evidence contract."""
import asyncio
import json

import pytest

from lab_arena import scoring_provider_compat as compat
from qualification.scoring.company_evidence_investigator import _plain_text, _quote_occurs
from qualification.scoring.company_verification import _fetch_bounded_html

ID = '7489978607814144000'
URL = f'https://www.linkedin.com/posts/example-company_growth-activity-{ID}-abcd'
TEXT = 'Example Company expanded its logistics team. <script>native text</script> & more.'


def route(url=URL, operation='scrapingdog.scrape', **kw):
    return compat.route_for(kind='score', funding_source='miner_key',
        round_id='arena-2026-10-20', operation_id=operation,
        parameters={'url': url} if operation.endswith('.scrape') else {'id': ID}, **kw)


def adapt(element=None, **kw):
    body = json.dumps({'status': 'completed', 'result': {'data': {
        'element': {'id': ID, 'content': TEXT} if element is None else element}}}).encode()
    return compat.adapt_response_with_trusted_url(route(), status=200, headers={}, body=body, **kw)


def test_native_post_route_preserves_requested_identity_and_source():
    r = route()
    assert r.requested_operation_id == 'scrapingdog.scrape'
    assert r.requested_parameters == {'url': URL}
    assert r.effective_parameters == {'tool': 'harvestapi_get_post', 'payload': {
        'url': f'https://www.linkedin.com/feed/update/urn:li:activity:{ID}/'}}
    status, headers, body, source = adapt()
    assert status == 200 and source == URL
    assert headers['content-type'] == 'text/html; charset=utf-8'
    assert b'<script>' not in body and b'&lt;script&gt;' in body and b'&amp;' in body
    assert _quote_occurs('Example Company expanded its logistics team.', _plain_text(body.decode()))


@pytest.mark.parametrize('url', [
    f'https://www.linkedin.com/posts/example_activity-{ID}',
    f'https://linkedin.com/feed/update/urn:li:activity:{ID}/',
])
def test_canonical_post_forms(url):
    assert route(url).adapter == 'harvest_linkedin_post_html'


@pytest.mark.parametrize('url', [
    'https://www.linkedin.com/company/example/', 'https://www.linkedin.com/in/example/',
    'https://www.linkedin.com/jobs/view/1234/', URL + '?tracking=1', URL + '#fragment',
    URL.replace('https:', 'http:'), URL.replace('www.linkedin.com', 'www.linkedin.com.evil.test'),
    URL.replace('www.linkedin.com', 'user@www.linkedin.com'), URL.replace(ID, '123'),
    URL.replace('www.linkedin.com', 'www.linkedin.com:443'),
    URL.replace('/posts/', '/posts/nested/'), URL + '-activity-7489978607814144001',
])
def test_unsupported_or_ambiguous_urls_do_not_select_post_route(url):
    assert route(url).adapter != 'harvest_linkedin_post_html'


@pytest.mark.parametrize('element', [
    {'id': '7489978607814144001', 'content': TEXT},
    {'id': ID}, {'id': ID, 'content': ''}, {'id': ID, 'content': ' '},
    {'id': ID, 'content': 'x' * 131073}, {'id': ID, 'content': 'invalid\x00text'},
    {'id': ID, 'content': TEXT, 'url': 'https://evil.test/'},
    {'id': ID, 'content': TEXT, 'linkedinUrl': URL.replace(ID, '7489978607814144001')},
    {'id': ID, 'content': TEXT, 'postUrl': None},
])
def test_native_response_fails_closed(element):
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(element)


def test_same_id_canonical_returned_url_and_legacy_profile_contract():
    assert adapt({'id': ID, 'content': TEXT, 'url': f'https://www.linkedin.com/feed/update/urn:li:activity:{ID}/'})[0] == 200
    r = route(operation='scrapingdog.profile_post')
    status, headers, body, source = compat.adapt_response_with_trusted_url(r, status=200, headers={},
        body=json.dumps({'status': 'completed', 'result': {'data': {'element': {
            'id': ID, 'content': TEXT, 'author': {'name': 'Example'}, 'postedAt': {'date': '2026-10-20'}}}}}).encode())
    assert status == 200 and source == '' and headers == {'content-type': 'application/json'}
    assert json.loads(body) == {'post_results': {'post_text': TEXT, 'author': {'name': 'Example'}, 'activity_date': '2026-10-20'}}


def test_unchanged_bounded_fetch_and_quote_consumer():
    status, headers, body, source = adapt()
    class Content:
        async def read(self, n):
            nonlocal body
            result, body = body[:n], body[n:]
            return result
    class Response:
        _lab_arena_response_url_validated = True
        content = Content()
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
    response = Response()
    response.status, response.headers, response.url = status, headers, source
    class Session:
        def get(self, url, **kwargs):
            assert url == URL and kwargs == {'allow_redirects': True}
            return response
    got_status, got_url, got_html = asyncio.run(_fetch_bounded_html(Session(), URL))
    assert got_status == 200 and got_url == URL
    text = _plain_text(got_html)
    assert _quote_occurs('Example Company expanded its logistics team.', text)
    assert not _quote_occurs('Example Company raised a billion dollars.', text)


def test_no_execute_or_unapproved_host_route():
    assert compat.route_for(kind='execute', funding_source='miner_key', round_id='arena-2026-10-20',
        operation_id='scrapingdog.scrape', parameters={'url': URL}) is None
    assert compat.route_for(kind='score', funding_source='host', round_id='arena-2026-10-20',
        operation_id='scrapingdog.scrape', parameters={'url': URL}) is None


def test_broker_keeps_native_price_source_url_and_same_action_replay():
    from dataclasses import replace
    from lab_arena import operations
    from tests.lab_arena.test_deepline_catalog import frozen, row
    from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker
    catalog = frozen(row('harvestapi_get_post', provider='harvestapi', categories=['research'],
        inputSchema={'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']}))
    context = replace(CONTEXT, kind='score', round_id='arena-2026-10-20', deepline_catalog=catalog)
    envelope = {'status': 'completed', 'job_id': 'post-native-one',
        'billing': {'credits_charged': 0.03, 'cost_usd': 0.003},
        'result': {'data': {'element': {'id': ID, 'content': TEXT}}}}
    transport = FakeTransport([(200, json.dumps(envelope).encode())])
    broker, store, _ = make_broker(transport=transport,
        provider_funding_source_for=lambda *_: 'miner_key')
    args = dict(operation_id='scrapingdog.scrape', parameters={'url': URL},
                action_sequence=1, timeout_ms=30000)
    first = broker.execute(context, **args)
    assert first.status == 200 and first.call['actual_microusd'] == 3000
    assert first.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == URL
    assert len(transport.sent) == 1
    second = broker.execute(context, **args)
    assert second.status == 200 and second.body == first.body
    assert second.call['call_identity'] == first.call['call_identity']
    assert len(transport.sent) == 1


def test_missing_frozen_harvest_capability_never_dispatches():
    from tests.lab_arena.score_compat_capability_test import routed_broker, score_context
    broker, store, transport = routed_broker(funding_source='miner_key')
    result = broker.execute(score_context(), operation_id='scrapingdog.scrape',
        parameters={'url': URL}, action_sequence=1, timeout_ms=30000)
    assert result.status == 400 and json.loads(result.body)['error']['code'] == 'capability_unavailable'
    assert transport.sent == [] and store.calls == {}


@pytest.mark.parametrize('status', [201, 301, 302])
def test_redirect_or_nonfinal_native_status_is_not_page_evidence(status):
    body = json.dumps({'status': 'completed', 'result': {'data': {
        'status': status, 'element': {'id': ID, 'content': TEXT}}}}).encode()
    with pytest.raises(compat.CompatibilityResponseError):
        compat.adapt_response_with_trusted_url(route(), status=200, headers={}, body=body)


def test_target_not_found_has_no_fabricated_page_or_trusted_url():
    body = json.dumps({'status': 'completed', 'result': {'data': {'status': 404}}}).encode()
    assert compat.adapt_response_with_trusted_url(route(), status=200, headers={}, body=body) == (
        404, {'content-type': 'text/html; charset=utf-8'}, b'', '')


def test_maximum_multibyte_content_and_escaping_stay_bounded():
    content = '😀' * 32768
    assert adapt({'id': ID, 'content': content})[0] == 200
    with pytest.raises(compat.CompatibilityResponseError):
        adapt({'id': ID, 'content': content + '😀'})
    assert len(adapt({'id': ID, 'content': '"' * 131072})[2]) < 1048576
