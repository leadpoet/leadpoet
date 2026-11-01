"""The catalog must offer only models this account can actually route to.

INC-964: the public models list prices models the Arena cannot reach under its
mandatory zero-data-retention policy. A miner picks one in good faith and loses
its whole round. The account-scoped route narrows the list; it must never become
a new way for the catalog build to fail.
"""
import json

import pytest

from lab_arena import broker as br
from lab_arena.contracts import ArenaContractError
from urllib.error import HTTPError, URLError

REACHABLE = 'openai/gpt-4o-mini'
UNREACHABLE = 'some-lab/zdr-less-model'
PRICING = {'prompt': '0.00000015', 'completion': '0.0000006'}


def models_document(*model_ids):
    return {'data': [{'id': model, 'pricing': dict(PRICING)} for model in model_ids]}


class Calls:
    """Record every URL requested and reply from a scripted table."""

    def __init__(self, replies):
        self.replies = replies
        self.urls = []
        self.keys = []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        self.keys.append(request.headers.get('Authorization'))
        reply = self.replies[request.full_url]
        if isinstance(reply, Exception):
            raise reply
        return _Body(json.dumps(reply).encode('utf-8'))


class _Body:
    def __init__(self, raw):
        self.raw = raw

    def read(self, _limit=None):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def fetch(replies, *, api_key='or-test-key'):
    calls = Calls(replies)
    table = br.fetch_openrouter_price_table(urlopen=calls, api_key=api_key)
    return table, calls


def test_account_scoped_catalog_is_preferred_and_excludes_unreachable_models():
    table, calls = fetch({
        br.OPENROUTER_USER_MODELS_URL: models_document(REACHABLE),
        br.OPENROUTER_MODELS_URL: models_document(REACHABLE, UNREACHABLE),
    })
    assert calls.urls == [br.OPENROUTER_USER_MODELS_URL]
    assert calls.keys == ['Bearer or-test-key']
    assert set(table['models']) == {REACHABLE}
    assert table['source'] == br.OPENROUTER_USER_MODELS_URL


def test_no_key_still_uses_the_public_catalog_unauthenticated():
    table, calls = fetch({br.OPENROUTER_MODELS_URL: models_document(REACHABLE)},
                         api_key=None)
    assert calls.urls == [br.OPENROUTER_MODELS_URL]
    assert calls.keys == [None]
    assert table['source'] == br.OPENROUTER_MODELS_URL


@pytest.mark.parametrize('refusal', [
    HTTPError(br.OPENROUTER_USER_MODELS_URL, 401, 'Unauthorized', {}, None),
    HTTPError(br.OPENROUTER_USER_MODELS_URL, 404, 'Not Found', {}, None),
    HTTPError(br.OPENROUTER_USER_MODELS_URL, 500, 'Server Error', {}, None),
    URLError('unreachable'),
])
def test_any_refusal_of_the_account_route_falls_back_to_the_public_catalog(refusal):
    table, calls = fetch({
        br.OPENROUTER_USER_MODELS_URL: refusal,
        br.OPENROUTER_MODELS_URL: models_document(REACHABLE, UNREACHABLE),
    })
    assert calls.urls == [br.OPENROUTER_USER_MODELS_URL, br.OPENROUTER_MODELS_URL]
    assert set(table['models']) == {REACHABLE, UNREACHABLE}
    assert table['source'] == br.OPENROUTER_MODELS_URL


@pytest.mark.parametrize('document', [
    {'data': []},
    {'data': [{'id': REACHABLE}]},
    {'not': 'a models document'},
])
def test_an_unusable_account_response_falls_back_rather_than_failing_the_round(document):
    table, calls = fetch({
        br.OPENROUTER_USER_MODELS_URL: document,
        br.OPENROUTER_MODELS_URL: models_document(REACHABLE),
    })
    assert calls.urls == [br.OPENROUTER_USER_MODELS_URL, br.OPENROUTER_MODELS_URL]
    assert set(table['models']) == {REACHABLE}


def test_a_public_catalog_failure_still_raises():
    """Fallback softens the new route only; the existing contract is unchanged."""
    calls = Calls({
        br.OPENROUTER_USER_MODELS_URL: URLError('unreachable'),
        br.OPENROUTER_MODELS_URL: URLError('unreachable'),
    })
    with pytest.raises(ArenaContractError):
        br.fetch_openrouter_price_table(urlopen=calls, api_key='or-test-key')
