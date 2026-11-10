"""A repaired URL must not launder an unsupported quote into intent credit."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_source_grounded_stage1_skip import _response

URL = 'https://news.example/acme-raises-thirty-million-series-b-led-by-fund-to-expand-enterprise-software'
COPIED = URL.replace('series-b', 'series-c')
FACT = 'Acme raised thirty million in a Series B funding round led by Fund.'
INVENTED = FACT.replace('Series B', 'Series C')


@pytest.mark.asyncio
@pytest.mark.parametrize('quotes', [[INVENTED], [FACT, INVENTED], [], ['Acme raised thirty million ... led by Fund.']])
async def test_miscopied_url_with_ungrounded_quotes_never_approves(monkeypatch, quotes):
    reply = _response()
    item = reply['answer']['signal_evaluations'][0]
    item.update(evidence_urls_used=[COPIED], supporting_quotes=quotes)
    call = AsyncMock(return_value=deepcopy(reply))
    fetch = AsyncMock(return_value={'results': [{'url': URL, 'text': FACT}], 'statuses': []})
    monkeypatch.setattr(intent, '_call_openrouter', call)
    monkeypatch.setattr(intent, '_fetch_sd_then_exa', fetch)
    result = await intent.verify_three_stage(
        None, company_name='Acme', company_website='https://acme.com',
        company_linkedin='https://www.linkedin.com/company/acme', source_url=URL,
        miner_claim=INVENTED, target_signal_text='Raised a Series C funding round',
        stage1_soft_reject=True, integrity_policy=True,
    )
    assert result['client_ready'] is False
    assert result['decision'] != 'approve'
    call.assert_awaited_once()
    fetch.assert_awaited_once()


@pytest.mark.parametrize(('quotes', 'pages', 'restores'), [
    ([FACT], [{'url': URL, 'text': FACT}], True),
    (['"'+FACT+'"'], [{'url': URL, 'text': FACT}], True),
    ([FACT], [{'url': URL, 'text': FACT.replace(' ', '\n')}], True),
    ([FACT], [], False),
    ([FACT], [{'url': 'https://elsewhere.example', 'text': FACT}], False),
    ([FACT], [{'url': URL, 'text': 'Acme raised thirty million'},
              {'url': URL, 'text': 'in a Series B funding round led by Fund.'}], False),
    ([''], [{'url': URL, 'text': FACT}], False),
    ([17], [{'url': URL, 'text': FACT}], False),
])
def test_correction_uses_supplied_page_text_not_just_a_similar_address(quotes, pages, restores):
    reply = _response()['answer']
    item = reply['signal_evaluations'][0]
    item.update(evidence_urls_used=[COPIED], supporting_quotes=quotes)
    result = intent._apply_guardrails({'claimed_source_urls':[URL]}, reply,
        restore_miscopied_citations=True, contents={'results': pages})
    item=result['signal_evaluations'][0]
    assert (item['signal_status']=='supported') is restores
    assert item['evidence_urls_used']==([URL] if restores else [COPIED])


def test_correction_does_not_borrow_quotes_from_another_supplied_page():
    other='https://news.example/acme-history'
    old_fact='Acme was founded in 2010.'
    reply=_response()['answer']
    reply['signal_evaluations'][0].update(evidence_urls_used=[COPIED,other],supporting_quotes=[FACT,old_fact])
    contents={'results':[{'url':URL,'text':FACT},{'url':other,'text':old_fact}]}
    restored=intent._apply_guardrails({'claimed_source_urls':[URL,other]},deepcopy(reply),
        restore_miscopied_citations=True,contents=contents)
    assert restored['signal_evaluations'][0]['signal_status']=='unable_to_verify'
    reply['signal_evaluations'][0]['supporting_quotes']=[old_fact]
    rejected=intent._apply_guardrails({'claimed_source_urls':[URL,other]},reply,
        restore_miscopied_citations=True,contents=contents)
    assert rejected['signal_evaluations'][0]['signal_status']=='unable_to_verify'


@pytest.mark.parametrize('contents', [None, {}, {'results':None}, {'results':17},
                                    {'results':'bad'}, {'results':{}},
                                    {'results':[{'url':17,'text':FACT}]}])
def test_invalid_fetch_shape_leaves_citation_unresolved(contents):
    reply=_response()['answer']
    reply['signal_evaluations'][0].update(evidence_urls_used=[COPIED],supporting_quotes=[FACT])
    result=intent._apply_guardrails({'claimed_source_urls':[URL]},reply,
        restore_miscopied_citations=True,contents=contents)
    assert result['signal_evaluations'][0]['signal_status']=='unable_to_verify'


def test_incidental_company_name_cannot_bind_another_pages_fact():
    reply=_response()['answer']
    reply['signal_evaluations'][0].update(evidence_urls_used=[COPIED],supporting_quotes=['Acme',FACT])
    other='https://news.example/different-event'
    result=intent._apply_guardrails({'claimed_source_urls':[URL,other]},reply,
        restore_miscopied_citations=True,contents={'results':[
            {'url':URL,'text':'Acme company overview'}, {'url':other,'text':FACT}]})
    assert result['signal_evaluations'][0]['signal_status']=='unable_to_verify'
