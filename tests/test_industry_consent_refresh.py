"""Retained Nov7 Skai texts reproduce an optional refresh losing loaded proof.

Transport and model responses are hermetic controls, not an accuracy claim.
"""
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests import test_industry_document_wrapper_refresh as wrapper

SAVED = json.loads((Path(__file__).parent / 'fixtures/company_evidence/skai_consent_refresh_20261107.json').read_text())


def test_exact_retained_skai_source_survives_consent_only_refresh(monkeypatch):
    result, reads, documents, _, calls, _ = wrapper.loop(monkeypatch,
        search_text=SAVED['loaded_text'], raw=SAVED['refresh_text'])
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text'] == SAVED['loaded_text']
    assert reads == [wrapper.WRAPPER] and documents == []
    assert result['usage']['fetch_calls'] == 1
    assert 'prior_loaded_source_retained' in json.dumps(calls['model'])
    assert result['usage']['fetch_outcomes'][-1]['error_class'] == 'source_access_denied'
    assert not c._has_unresolved_source_fetch_faults(result['usage']['fetch_outcomes'])


@pytest.mark.parametrize('new_body', [
    'We discontinued this product. It is no longer sold.',
    'The platform is now free and does not offer paid plans.',
    'Coreworks now supplies a different product.',
])
def test_valid_changed_page_replaces_the_old_search_body(monkeypatch, new_body):
    result, reads, documents, _, _, _ = wrapper.loop(monkeypatch,
        search_text=SAVED['loaded_text'], raw=new_body)
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text'] == new_body
    assert reads == [wrapper.WRAPPER] and documents == []


def test_consent_like_words_do_not_hide_material_changed_facts(monkeypatch):
    new_body = SAVED['refresh_text'].partition(c._IDENTITY_LINK_CONTEXT_MARKER)[0] + ' This product has been discontinued.'
    result, _, _, _, _, _ = wrapper.loop(monkeypatch,
        search_text=SAVED['loaded_text'], raw=new_body)
    assert 'This product has been discontinued.' in result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text']


def test_failed_refresh_retains_only_fetched_facts_not_an_untrusted_quote(monkeypatch):
    result, _, _, _, _, _ = wrapper.loop(monkeypatch,
        search_text=SAVED['loaded_text'], raw=SAVED['refresh_text'])
    page = result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text']
    assert c._quote_occurs('Our advertising management platform offers flat annual pricing based on the size of your program', page)
    assert not c._quote_occurs('Skai sells every requested product for $49 per month.', page)


def test_shell_without_retained_matching_title_does_not_reuse_another_page(monkeypatch):
    result, _, _, _, _, _ = wrapper.loop(monkeypatch,
        search_text='A different page and title. Other source facts.', raw=SAVED['refresh_text'])
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text'] != 'A different page and title. Other source facts.'


def test_prefetched_source_is_not_newly_refreshable(monkeypatch):
    result, reads, documents, _, _, _ = wrapper.loop(monkeypatch,
        search_text=SAVED['loaded_text'], prefetch=SAVED['loaded_text'], raw=SAVED['refresh_text'])
    assert reads == documents == []
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text'] == SAVED['loaded_text']


def test_consent_shell_has_no_commercial_proof_without_a_loaded_body(monkeypatch):
    result, _, _, _, calls, _ = wrapper.loop(monkeypatch,
        search_text='', raw=SAVED['refresh_text'])
    page = result[c.PRIVATE_FETCHED_PAGES_KEY][wrapper.WRAPPER]['text']
    assert 'prior_loaded_source_retained' not in json.dumps(calls['model'])
    assert not c._quote_occurs('Our advertising management platform offers flat annual pricing', page)
    assert result['claims']['industry']['status'] == 'UNPROVEN'


@pytest.mark.parametrize('option', ['deadline', 'fetch_cap'])
def test_consent_refresh_keeps_existing_admission_limits(monkeypatch, option):
    result, reads, documents, _, _, _ = wrapper.loop(monkeypatch,
        search_text=SAVED['loaded_text'], raw=SAVED['refresh_text'], **{option: True})
    assert documents == []
    assert result['usage']['fetch_calls'] <= 1
    assert reads == ([] if option == 'deadline' else [wrapper.WRAPPER])
