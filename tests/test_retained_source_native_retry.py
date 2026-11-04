"""Actual source and distinct held replies preserve strict evidence/date guards."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from gateway.qualification.models import ICPPrompt, IntentSignal
from qualification.scoring import intent_verification_three_stage as verifier, lead_scorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.competition import _normalized_icp


def _fixture():
    value = json.loads((Path(__file__).parent / 'fixtures/epam_wiz_retained_source_stage3_20261024.json').read_text())
    assert hashlib.sha256(value['source_text'].encode()).hexdigest() == value['source_text_sha256']
    serialized = json.dumps(value['native_answer'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert hashlib.sha256(serialized).hexdigest() == value['native_answer_sha256']
    retry = json.dumps(value['native_retry_answer'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert hashlib.sha256(retry).hexdigest() == value['native_retry_answer_sha256']
    return value


def _install(monkeypatch, fixture, *, fresh, answer=None):
    url = fixture['source_url']
    pages = {url: {'final_url': url, 'text': fixture['source_text']}}
    effective = verifier._verified_submitted_source_pages(pages, [url], fixture['identity'],
        company_name=fixture['company']['company_name'], claim=fixture['miner_claim'])
    text = effective[url]['text']
    fetched = ({'results': [{'url': url, 'text': text, 'meta': {'final_url': url}}], 'statuses': []}
               if fresh else {'results': [], 'statuses': [{'url': url, 'source': 'none',
                   'sd_stage': 'all_tiers_exhausted:http_502', 'sd_error': 'http_502',
                   'exa_stage': 'exa_transient_exhausted', 'exa_error': 'HTTP 502'}]})
    fetch = AsyncMock(return_value=fetched)
    judge = AsyncMock(return_value={'model': 'captured-native-luna', 'usage': {},
        'answer': copy.deepcopy(fixture['native_answer'] if answer is None else answer)})
    monkeypatch.setattr(verifier, '_fetch_sd_then_exa', fetch)
    monkeypatch.setattr(verifier, '_call_openrouter', judge)
    return pages, text, fetch, judge


def _verify(monkeypatch, fixture, *, fresh, claim=None, answer=None):
    pages, text, fetch, judge = _install(monkeypatch, fixture, fresh=fresh, answer=answer)
    with use_evaluation_date('2026-10-24'):
        result = asyncio.run(verifier.verify_three_stage(object(), **fixture['company'],
            source_url=fixture['source_url'], miner_claim=claim or fixture['miner_claim'],
            target_signal_text=fixture['target_signal_text'], buyer_request_context=fixture['buyer_request_context'],
            miner_signal_date=fixture['miner_date'], evidence_type='PARTNERSHIP',
            stage1_soft_reject=True, integrity_policy=True, company_quality=True,
            verified_company_identity=fixture['identity'], verified_source_pages=pages, buyer_max_age_days=365))
    assert fetch.await_count == judge.await_count == 1
    return result, text


def test_actual_epam_bad_date_quote_is_unavailable_and_separate_reply_is_usable(monkeypatch):
    fixture = _fixture()
    first, text = _verify(monkeypatch, fixture, fresh=False)
    first_item = fixture['native_answer']['signal_evaluations'][0]
    first_quotes = first_item['supporting_quotes']
    assert 'NEWTOWN, PA, USA, August 25, 2026' in text
    assert [verifier._grounded_exact_text(text, quote) for quote in first_quotes] == [False, True]
    grounded = [quote for quote in first_quotes if verifier._grounded_exact_text(text, quote)]
    assert not verifier._has_grounded_source_event_date(first_item, '\n'.join(grounded))
    assert first['decision'] == 'unavailable' and first['client_ready'] is False
    second, same_text = _verify(monkeypatch, fixture, fresh=False, answer=fixture['native_retry_answer'])
    second_item = fixture['native_retry_answer']['signal_evaluations'][0]
    assert same_text == text
    assert all(verifier._grounded_exact_text(text, quote) for quote in second_item['supporting_quotes'])
    assert verifier._has_grounded_source_event_date(second_item, '\n'.join(second_item['supporting_quotes']))
    assert second['decision'] == 'approve' and second['client_ready'] is True
    assert fixture['native_answer_sha256'] != fixture['native_retry_answer_sha256']


def test_actual_complete_native_reply_keeps_healthy_and_retained_acceptance_equal(monkeypatch):
    fixture = _fixture()
    results = [_verify(monkeypatch, fixture, fresh=fresh, answer=fixture['native_retry_answer'])[0]
               for fresh in (True, False)]
    assert results[0]['decision'] == results[1]['decision'] == 'approve'
    assert results[0]['client_ready'] is results[1]['client_ready'] is True


def test_actual_epam_wrong_acquisition_cannot_qualify_either_path(monkeypatch):
    fixture = _fixture()
    results = [_verify(monkeypatch, fixture, fresh=fresh, claim=fixture['wrong_acquisition_claim'],
        answer=fixture['wrong_acquisition_answer'])[0] for fresh in (True, False)]
    assert results[0]['decision'] == 'reject'
    assert results[1]['decision'] == 'unavailable'
    assert results[0]['client_ready'] is results[1]['client_ready'] is False


@pytest.mark.parametrize('fresh', [True, False])
def test_actual_old_source_date_cannot_be_rejuvenated_by_new_miner_date(monkeypatch, fresh):
    fixture = _fixture()
    pages, _, fetch, judge = _install(monkeypatch, fixture, fresh=fresh, answer=fixture['native_retry_answer'])
    # The actual source and held model reply retain August25. Only the miner
    # supplies October23; the existing host date gate uses a30day buyer cap.
    signal = IntentSignal(source='news', description=fixture['miner_claim'], url=fixture['source_url'],
        date='2026-10-23', snippet=fixture['miner_claim'], matched_icp_signal=0)
    icp = ICPPrompt.model_validate(_normalized_icp(fixture['icp'])).model_copy(update={'intent_max_age_days': 30})
    receipts = []
    with use_evaluation_date('2026-10-24'):
        score = asyncio.run(lead_scorer._score_single_intent_signal(signal, icp, None,
            fixture['company']['company_name'], fixture['company']['company_website'],
            stage1_soft_reject=True, llm_only_intent_gate=True, integrity_policy=True, company_quality=True,
            verified_company_identity=fixture['identity'], verified_source_pages=pages,
            verdict_out=receipts, evidence_signals=[signal]))
    assert score[0] == 0
    assert receipts[-1]['authoritative_date'] == '2026-08-25'
    assert receipts[-1]['date_verdict'] == 'out_of_window'
    assert fetch.await_count == judge.await_count == 1
