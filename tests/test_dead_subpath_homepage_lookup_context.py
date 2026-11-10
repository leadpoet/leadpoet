"""Bounded root recovery and lookup context; all network/provider calls are mocked.

The public fixture reconstructs only the exact title and strong metadata tags
from the Nov9 root read; wrappers and empty body are synthetic. Set
DEERE_RETAINED_ROOT_HTML to the private full capture for exact-byte replay.
Neither input is the original judgment response. Typed replies are synthetic.
"""
import asyncio
from copy import deepcopy
import os
import hashlib
import json
from pathlib import Path

import aiohttp
import pytest

from qualification.scoring import company_verification as cv, lead_scorer as s
from qualification.scoring import linkedin_company_size as profiles
from qualification.scoring.company_fit_evidence_reuse import capture_fit_evidence
from tests.test_company_headcount_evidence import company, icp, verdict, install_no_network
from tests.test_company_verification_transport import _Response, _RoutingSession

ORIGINAL = 'http://www.deere.com/AboutUs'
PATH = 'https://www.deere.com/AboutUs'
ROOT = 'https://www.deere.com/'
FINAL = 'https://www.deere.com/en-us/'
PROFILE = 'https://www.linkedin.com/company/john-deere/'
KEY = 'homepage_identity_lookup_context'
META = '<meta property="og:site_name" content="John Deere">'
HTML = '<html><head>' + META + '</head><body>John Deere equipment.</body></html>'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def deny(*_args, **_kwargs):
        raise AssertionError('network denied')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def root_bytes():
    retained = os.environ.get('DEERE_RETAINED_ROOT_HTML')
    if retained:
        raw = Path(retained).read_bytes()
        expected = '7665b6e52b9f01eda89b38803ff705627af8bd655c66da9e8812e7aa52e06446'
    else:
        raw = (Path(__file__).parent / 'fixtures/deere_public_root_identity_excerpt.html').read_bytes()
        expected = '01be8d813753b8a0c11d0aa39ef66c951b747637d86c49a641b028d93ae67cd6'
    assert hashlib.sha256(raw).hexdigest() == expected
    return raw


def candidate():
    return company().model_copy(update={
        'company_name': 'John Deere', 'company_website': ORIGINAL, 'company_linkedin': PROFILE,
    })


def verify(monkeypatch, *, first=404, original=ORIGINAL, root_status=200,
           final=FINAL, html=HTML, name='John Deere', first_final=PATH):
    calls, timeouts = [], []
    routes = {
        PATH: _Response(first, b'not found', first_final),
        ROOT: _Response(root_status, html.encode(), final),
    }
    if original == ROOT:
        routes[ROOT] = _Response(first, b'not found', ROOT)
    def session(**kwargs):
        timeouts.append(kwargs['timeout'])
        return _RoutingSession(routes, calls)
    monkeypatch.setattr(cv.aiohttp, 'ClientSession', session)
    pages = {}
    with capture_fit_evidence() as capture:
        result = asyncio.run(cv.verify_company_exists(
            name, original, company_linkedin=PROFILE, require_https_transport=True,
            homepage_evidence_sink=pages,
        ))
    return result, calls, timeouts, pages, capture


@pytest.mark.parametrize('status', [404, 410])
def test_root_metadata_recover_transport_and_context_without_identity(monkeypatch, status):
    raw = root_bytes()
    result, calls, timeouts, pages, capture = verify(
        monkeypatch, first=status, html=cv._decode_homepage_html(raw, 'text/html;charset=UTF-8'))
    assert calls == [PATH, ROOT]
    assert len(timeouts) == cv._TRANSIENT_FETCH_ATTEMPTS == 2
    assert all(t.total == 5 and t.connect == 3 for t in timeouts)
    assert result.decision == 'unavailable'
    assert result.details['verified_homepage_transport_domain'] == 'deere.com'
    assert result.details['actual_final_url'] == FINAL
    assert result.details[KEY] == {
        'source': 'server_observed_head_metadata_v1', 'observed_name': 'John Deere',
        'registrable_dns_domain': 'deere.com', 'request_url': ROOT, 'final_url': FINAL,
    }
    assert result.details['identity']['submitted_name'] == 'johndeere'
    assert result.details['identity']['decision'] == 'unavailable'
    assert not s._verified_homepage_identity_anchor(result)
    fallback = result.details['homepage_root_fallback']
    assert fallback['original_request_url'] == PATH
    assert fallback['original_status'] == status
    assert fallback['original_actual_final_url'] == PATH
    assert fallback['request_url'] == ROOT and fallback['actual_final_url'] == FINAL
    assert fallback['status'] == 200
    sources = {r['request']['url']: capture.responses[k] for k, r in capture.records.items()}
    assert sources[PATH] == {'status': status, 'final_url': PATH, 'text': 'not found'}
    assert sources[ROOT]['text'] == cv._decode_homepage_html(raw)
    assert sources[ROOT]['final_url'] == FINAL
    assert not capture.complete  # A failed source cannot become reusable proof.
    assert set(pages) == {ROOT} and pages[ROOT]['final_url'] == FINAL
    assert ORIGINAL == candidate().company_website  # Submitted claim stays exact.


@pytest.mark.parametrize('status', [200, 301, 403, 408, 425, 429, 500, 502, 503, 504])
def test_success_or_nondefinitive_failures_never_add_root_call(monkeypatch, status):
    result, calls, *_ = verify(monkeypatch, first=status)
    assert calls == [PATH]
    assert 'homepage_root_fallback' not in result.details


@pytest.mark.parametrize('status', [404, 410])
def test_already_root_does_not_fallback(monkeypatch, status):
    result, calls, *_ = verify(monkeypatch, first=status, original=ROOT)
    assert calls == [ROOT]
    assert 'homepage_root_fallback' not in result.details


@pytest.mark.parametrize('url', [
    'https://deere.com/missing', 'https://careers.deere.com/missing',
    'https://foreign.example/missing', 'http://www.deere.com/missing',
    'https://user:pass@www.deere.com/missing', 'https://127.0.0.1/missing',
    'https://www.deere.com:444/missing',
])
def test_failed_subpath_redirect_must_stay_original_approved_host(monkeypatch, url):
    result, calls, *_ = verify(monkeypatch, first_final=url)
    assert calls == [PATH]
    assert KEY not in result.details


@pytest.mark.parametrize('url', [
    'https://deere.com/', 'https://careers.deere.com/', 'https://foreign.example/',
    'http://www.deere.com/', 'https://user:pass@www.deere.com/',
    'https://127.0.0.1/', 'https://localhost/', 'https://www.deere.com:444/',
    'not-a-url', 'https://www.deere.com/#forged',
])
def test_root_redirect_cannot_supply_foreign_or_unsafe_context(monkeypatch, url):
    result, calls, _, pages, _ = verify(monkeypatch, final=url)
    assert calls == [PATH, ROOT]
    assert result.decision != 'match'
    assert KEY not in result.details and pages == {}
    assert 'verified_homepage_transport_domain' not in result.details
    assert result.details['homepage_root_fallback']['original_status'] == 404


@pytest.mark.parametrize('html', [
    '', 'short provider placeholder', 'This domain is for sale ' + HTML,
    '<html><head><title>John Deere</title></head><body>John Deere</body></html>',
    '<html><head></head><body>' + META + '<h1>John Deere</h1></body></html>',
    '<html><head><!--' + META + '--></head><body>John Deere</body></html>',
    '<html><head><script>var forged = ' + json.dumps(META) + ';</script></head></html>',
    '<html><head><style>' + META + '</style></head></html>',
    '<html><head><template>' + META + '</template></head></html>',
    '<html><head><noscript>' + META + '</noscript></head></html>',
    '<html><head>&lt;meta property="og:site_name" content="John Deere"&gt;</head></html>',
    '<html><head>' + META + '<meta name="application-name" content="Parent Holdings"></head></html>',
    '<html><head><meta property="og:site_name" content="Parent Holdings"></head><body>John Deere</body></html>',
    '<html><head>' + META,  # Incomplete/malformed head.
    '<html><head><div>' + META + '</div></head></html>',
    '<html><head><meta property="og:site_name" content="Other" content="John Deere"></head></html>',
    '%PDF-1.7 binary prefix ' + HTML,
    'Provider placeholder content. ' + HTML,
    '<html>Body content before the head<head>' + META + '</head></html>',
    '<html><head>Body content' + META + '</head></html>',
    '<html><head><div>Body content</div>' + META + '</head></html>',
])
def test_only_unambiguous_strong_head_metadata_can_supply_context(monkeypatch, html):
    result, calls, *_ = verify(monkeypatch, html=html)
    assert calls == [PATH, ROOT]
    assert KEY not in result.details
    assert result.decision != 'match'


@pytest.mark.parametrize('meta', [META, '<meta name="application-name" content="John Deere"/>'])
def test_exact_strong_metadata_and_self_closing_tags_are_supported(monkeypatch, meta):
    result, *_ = verify(monkeypatch, html='<html><head>' + meta + '</head></html>')
    assert result.details[KEY]['observed_name'] == 'John Deere'


def test_parent_root_cannot_supply_child_lookup_context(monkeypatch):
    result, *_ = verify(monkeypatch, name='John Deere Financial', html=HTML.replace(
        'equipment.', 'equipment. John Deere Financial is our subsidiary.'))
    assert KEY not in result.details
    assert result.decision != 'match'


@pytest.mark.parametrize('root_status', [403, 404, 410, 429, 500, 503])
def test_root_failure_keeps_original_failure_and_adds_no_retry(monkeypatch, root_status):
    result, calls, _, pages, capture = verify(monkeypatch, root_status=root_status)
    assert calls == [PATH, ROOT]
    assert result.decision != 'match' and KEY not in result.details and not pages
    assert result.details['homepage_root_fallback']['original_status'] == 404
    assert result.details['homepage_root_fallback']['status'] == root_status
    assert len(capture.records) == 2


@pytest.mark.parametrize('order', ['path_timeout', 'root_timeout', 'root_connection'])
def test_transient_and_root_reads_share_existing_attempt_cap(monkeypatch, order):
    calls = []
    async def fetch(_session, url):
        calls.append(url)
        if order == 'path_timeout' and len(calls) == 1:
            raise asyncio.TimeoutError()
        if url == ROOT:
            raise aiohttp.ClientConnectionError() if order == 'root_connection' else asyncio.TimeoutError()
        return 404, PATH, 'not found'
    monkeypatch.setattr(cv, '_fetch_bounded_html', fetch)
    result = asyncio.run(cv.verify_company_exists('John Deere', ORIGINAL,
        company_linkedin=PROFILE, require_https_transport=True, timeout_secs=.02))
    assert calls == ([PATH, PATH] if order == 'path_timeout' else [PATH, ROOT])
    assert len(calls) == cv._TRANSIENT_FETCH_ATTEMPTS
    assert result.decision == 'unavailable'
    assert KEY not in result.details
    if order != 'path_timeout':
        assert result.details['homepage_root_fallback']['original_status'] == 404


def synthetic_payload(**updates):
    return {'status': 'completed', 'result': {'data': {'status': 200, 'element': {
        'name': 'John Deere', 'website': ORIGINAL, 'linkedinUrl': PROFILE,
        'employeeCountRange': {'start': 11, 'end': 50}, **updates,
    }}}}


def run_model(monkeypatch, homepage, *, observed_name='John Deere', proof_fault=''):
    observed = {**verdict('11-50', 'John Deere has 11-50 employees'),
                'observed_company_name': observed_name,
                'observed_company_website': ROOT, 'observed_company_linkedin': '',
                'observed_industry': '', 'observed_subindustry': '', 'industry_matches': None,
                'industry_activity_role': 'unresolved', 'industry_evidence_url': '',
                'industry_evidence_quote': ''}
    install_no_network(monkeypatch, observed)
    requests, lookups = [], []
    async def request(**kwargs):
        requests.append(kwargs)
        return deepcopy(observed), ''
    async def refresh(value, *_args, **_kwargs):
        return value
    async def structured(domain, url, **kwargs):
        lookups.append((domain, url))
        assert kwargs['preserve_identity_website'] is True
        changes = {'name': 'Deere & Company'} if proof_fault == 'name' else {}
        if proof_fault == 'domain': changes['website'] = 'https://other.example/'
        if proof_fault == 'raw_website': changes['website'] = 'https://careers.deere.com/'
        if proof_fault == 'slug': changes['linkedinUrl'] = PROFILE.rstrip('/') + '-parent'
        if proof_fault == 'no_proof': return None
        payload = synthetic_payload(**changes)
        evidence = profiles.project_structured_linkedin_company_identity(
            domain, url, payload, preserve_identity_website=True)
        if evidence:
            if proof_fault == 'missing_raw': evidence.pop('provider_website_url', None)
            kwargs['company_identity_evidence'].update(evidence)
        return profiles.project_structured_linkedin_company_size(domain, url, payload)
    monkeypatch.setattr(s, '_request_company_reverify_json', request)
    monkeypatch.setattr(s, '_refresh_linkedin_employee_size_observation', refresh)
    monkeypatch.setattr(s, 'fetch_structured_linkedin_company_size', structured)
    # Exercise real identity routing, admission, typed projection and decisions.
    # Unrelated dimension fetches are disabled; this test cannot claim full fit.
    async def no_ground(value, **_kwargs): return value
    monkeypatch.setattr(s, '_ground_employee_size_observation', no_ground)
    result = asyncio.run(s._llm_reverify_company(candidate(), icp(),
        require_company_fit_dimensions=True, verified_homepage_identity=homepage))
    return result, requests, lookups


def test_initial_and_repair_requests_keep_same_lookup_context_without_identity_proof(monkeypatch):
    homepage, *_ = verify(monkeypatch, html=cv._decode_homepage_html(root_bytes()))
    result, requests, lookups = run_model(monkeypatch, homepage, proof_fault='no_proof')
    assert [r['telemetry_purpose'] for r in requests] == [
        'lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    retained = json.loads((Path(__file__).parent / 'fixtures/deere_retained_identity_request_locators.json').read_text())
    assert len(retained['requests']) == 2
    assert not any(row['requested_name_present'] or row['homepage_lookup_context_present']
                   for row in retained['requests'])
    blocks = []
    for request, prior_request in zip(requests, retained['requests']):
        prompt = request['prompt']
        assert '<verified_homepage_identity>' not in prompt
        assert 'identity is NOT verified' in prompt
        locator = json.loads(prompt.split('<untrusted_company_locator>')[1].split('</untrusted_company_locator>')[0])
        assert locator == prior_request['locator'] == {'registrable_dns_domain': 'deere.com'}
        blocks.append(json.loads(prompt.split('<homepage_identity_lookup_context>')[1].split('</homepage_identity_lookup_context>')[0]))
    assert blocks == [homepage.details[KEY]] * 2
    assert lookups == [('deere.com', PROFILE.rstrip('/'))]
    assert result.details['identity_decision'] != 'match'
    assert result.decision != 'match'


@pytest.mark.parametrize('observed_name', ['Deere & Company', 'John Deere Financial'])
def test_context_cannot_replace_independent_model_exact_name_or_inherit_parent(monkeypatch, observed_name):
    homepage, *_ = verify(monkeypatch)
    result, requests, lookups = run_model(monkeypatch, homepage, observed_name=observed_name)
    assert requests and all('lookup context only' in r['prompt'] for r in requests)
    assert lookups == []
    assert result.details['identity_decision'] != 'match'
    assert result.decision != 'match'


@pytest.mark.parametrize('fault', ['', 'no_proof', 'name', 'domain', 'slug', 'raw_website', 'missing_raw'])
def test_existing_typed_identity_gates_decide_after_independent_exact_model_observation(monkeypatch, fault):
    homepage, *_ = verify(monkeypatch)
    result, _, lookups = run_model(monkeypatch, homepage, proof_fault=fault)
    assert lookups == [('deere.com', PROFILE.rstrip('/'))]
    assert (result.details['identity_decision'] == 'match') is (fault == '')
    # Identity alone cannot qualify unrelated/unproven fit dimensions.
    assert result.decision != 'match'


def test_root_fallback_does_not_add_a_linked_identity_source_call(monkeypatch):
    result, calls, *_ = verify(monkeypatch, html=HTML.replace(
        '</body>', '<a href="https://news.deere.com/">News</a></body>'))
    assert calls == [PATH, ROOT]
    assert result.details[KEY]['observed_name'] == 'John Deere'


def test_existing_conflicting_linkedin_path_gets_no_new_lookup_context(monkeypatch):
    result, calls, *_ = verify(monkeypatch, html=HTML.replace(
        '</body>', '<a href="https://www.linkedin.com/company/parent-holdings/">Parent</a></body>'))
    assert calls == [PATH, ROOT]
    assert KEY not in result.details
    assert result.decision != 'match'


@pytest.mark.parametrize('fault', ['name', 'domain', 'source', 'host', 'final', 'extra', 'type', 'mismatch'])
def test_saved_or_malformed_context_is_revalidated_before_prompt_use(monkeypatch, fault):
    homepage, *_ = verify(monkeypatch)
    original_receipt = deepcopy(homepage.details['identity'])
    context = homepage.details[KEY]
    if fault == 'name': context['observed_name'] = 'John Deere Financial'
    elif fault == 'domain': context['registrable_dns_domain'] = 'other.example'
    elif fault == 'source': context['source'] = 'model_claim'
    elif fault == 'host': context['request_url'] = 'https://careers.deere.com/'
    elif fault == 'final': context['final_url'] = 'https://www.deere.com/other'
    elif fault == 'extra': context['verified'] = 'true'
    elif fault == 'type': context['observed_name'] = ['John Deere']
    elif fault == 'mismatch': homepage.decision = 'mismatch'
    result, requests, _ = run_model(monkeypatch, homepage, proof_fault='no_proof')
    assert requests and all('<homepage_identity_lookup_context>' not in r['prompt'] for r in requests)
    assert result.details['identity_decision'] != 'match'
    assert homepage.details['identity'] == original_receipt


def test_json_roundtrip_preserves_context_but_grants_no_identity_authority(monkeypatch):
    homepage, *_ = verify(monkeypatch, html=cv._decode_homepage_html(root_bytes()))
    homepage.details = json.loads(json.dumps(homepage.details))
    result, requests, _ = run_model(monkeypatch, homepage, proof_fault='no_proof')
    assert requests and '<homepage_identity_lookup_context>' in requests[0]['prompt']
    assert not s._verified_homepage_identity_anchor(homepage)
    assert result.details['identity_decision'] != 'match'


@pytest.mark.parametrize('prefix', ['<br>', '<img src="x">', '<input>', '<html><br>'])
def test_body_element_before_head_cannot_supply_lookup_metadata(monkeypatch, prefix):
    html = prefix + '<head>' + META + '</head><body>John Deere equipment</body>'
    assert cv._homepage_lookup_name(html, 'John Deere') == ''
    result, *_ = verify(monkeypatch, html=html)
    assert KEY not in result.details
