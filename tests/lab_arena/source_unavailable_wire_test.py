"""The additive proof is safe on both deployed legacy runner generations."""
import base64
import copy
import json
from pathlib import Path

import pytest

from lab_arena import broker, runner, shim
from tests.lab_arena.firecrawl_blocked_site_refusal_test import blocked_site_body, CAPTURED_URL
from tests.lab_arena.firecrawl_unsupported_source_test import routed_call

LEGACY = json.loads((Path(__file__).parents[1] / 'fixtures/reviewer_seams/source_refusal_legacy_wire.json').read_text())


@pytest.mark.parametrize('revision', list(LEGACY))
@pytest.mark.parametrize('new_shim', [False, True])
def test_new_host_reply_on_exact_old_runner_and_old_or_new_shim(revision, new_shim):
    record = LEGACY[revision]
    assert record['provider_request_refused_status'] == 403
    namespace = dict(vars(runner))
    exec(record['lab_arena/runner.py']['functions'], namespace)
    result, *_ = routed_call(url=CAPTURED_URL, body=blocked_site_body())
    document = result.to_document()
    document['headers'][shim.CALL_IDENTITY_HEADER] = 'forged-provider-identity'
    headers = namespace['_socket_response_headers'](document, 'scrapingdog.scrape', 0)
    assert headers[shim.CALL_IDENTITY_HEADER] == result.call['call_identity']
    assert headers[broker.SOURCE_UNAVAILABLE_HEADER] == broker.SOURCE_UNAVAILABLE_VALUE
    payload = dict(status=result.status, headers=headers, body_b64=base64.b64encode(result.body).decode())
    parser = shim._parse_worker_response_with_url
    if not new_shim:
        namespace = dict(vars(shim))
        exec(record['lab_arena/shim.py']['functions'], namespace)
        parser = namespace['_parse_worker_response_with_url']
    status, received, body, url = parser(payload)
    assert status == 403 and body == result.body and url == ''
    assert received[shim.CALL_IDENTITY_HEADER] == result.call['call_identity']
    assert (broker.SOURCE_UNAVAILABLE_HEADER in received) is new_shim


@pytest.mark.parametrize('headers', [
    {'x-leadpoet-source-unavailable': 'invalid', 'X-Leadpoet-Source-Unavailable': 'verified-source-v1'},
    {'x-leadpoet-source-unavailable': 'verified-source-v1', 'X-Leadpoet-Source-Unavailable': 'invalid'},
])
def test_case_duplicate_marker_is_never_trusted(headers):
    payload = dict(status=403, headers={**headers, shim.CALL_IDENTITY_HEADER: 'sha256:'+'a'*64},
        body_b64=base64.b64encode(b'{"error":{"code":"provider_request_refused"}}').decode())
    with pytest.raises(shim.ShimTransportError, match='invalid_response'):
        shim._parse_worker_response_with_url(payload)


@pytest.mark.parametrize('changed', ['body', 'status', 'identity', 'version'])
def test_marker_is_bound_to_exact_error_envelope(changed):
    payload = dict(status=403, headers={broker.SOURCE_UNAVAILABLE_HEADER: broker.SOURCE_UNAVAILABLE_VALUE,
        shim.CALL_IDENTITY_HEADER: 'sha256:'+'a'*64},
        body_b64=base64.b64encode(b'{"error":{"code":"provider_request_refused"}}').decode())
    if changed == 'body': payload['body_b64'] = base64.b64encode(b'{"error":{}}').decode()
    elif changed == 'status': payload['status'] = 200
    elif changed == 'identity': payload['headers'].pop(shim.CALL_IDENTITY_HEADER)
    else: payload['headers'][broker.SOURCE_UNAVAILABLE_HEADER] = 'invalid'
    assert broker.SOURCE_UNAVAILABLE_HEADER not in shim._parse_worker_response_with_url(payload)[1]
