"""Incomplete careers-link lookup must preserve already accepted source text."""
from unittest.mock import AsyncMock
import pytest
from qualification.scoring import intent_verification_three_stage as intent

URL='https://acme.example/careers'
ORIGINAL='Responsibilities: Build the platform and keep production services operating.'

@pytest.mark.asyncio
@pytest.mark.parametrize('fallback_ok',[False,True])
async def test_incomplete_lookup_retains_body_and_captures_fallback_status(monkeypatch,fallback_ok):
    sd=AsyncMock(return_value={'ok':True,'stage':'sd:baseline','content':ORIGINAL,
        'meta':{'careers_link_lookup_incomplete':True,'observed_ownership_links':[]}})
    exa=AsyncMock(return_value={'ok':fallback_ok,'stage':'exa_scraped' if fallback_ok else 'exa_unavailable',
        'content':'Additional source text with no hiring evidence.' if fallback_ok else ''})
    monkeypatch.setattr(intent,'_scrape_sd_hardened',sd)
    monkeypatch.setattr(intent,'_scrape_exa',exa)
    result=await intent._fetch_sd_then_exa([URL],prefer_dynamic_job_index=True)
    exa.assert_awaited_once_with(URL)
    assert len(result['results'])==1
    assert result['results'][0]['text'].startswith(ORIGINAL)
    assert result['results'][0]['meta']['careers_link_lookup_incomplete'] is True
    status=next(row for row in result['statuses'] if row['source']=='exa_careers_link_fallback')
    assert status['accepted'] is fallback_ok
    assert status['original_text_chars']==len(ORIGINAL)
    assert ORIGINAL not in str(status)

@pytest.mark.asyncio
async def test_healthy_noncareer_body_uses_no_new_fallback(monkeypatch):
    sd=AsyncMock(return_value={'ok':True,'stage':'sd:baseline','content':ORIGINAL,'meta':{}})
    exa=AsyncMock()
    monkeypatch.setattr(intent,'_scrape_sd_hardened',sd)
    monkeypatch.setattr(intent,'_scrape_exa',exa)
    result=await intent._fetch_sd_then_exa(['https://acme.example/news/product'])
    sd.assert_awaited_once_with('https://acme.example/news/product')
    exa.assert_not_awaited()
    assert result['results'][0]['text']==ORIGINAL

@pytest.mark.asyncio
async def test_healthy_careers_links_need_no_extra_fallback(monkeypatch):
    sd=AsyncMock(return_value={'ok':True,'stage':'sd:dynamic_render','content':ORIGINAL,
        'meta':{'kind':'first_party_careers_index','observed_job_link_count':1}})
    exa=AsyncMock()
    monkeypatch.setattr(intent,'_scrape_sd_hardened',sd)
    monkeypatch.setattr(intent,'_scrape_exa',exa)
    await intent._fetch_sd_then_exa([URL],prefer_dynamic_job_index=True)
    exa.assert_not_awaited()
