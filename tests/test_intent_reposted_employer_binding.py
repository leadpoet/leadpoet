"""Employer proof must survive rendering failure and apply to copied postings."""
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_first_party_careers_dynamic import _ScrapingDogClient, _identity, _verdict

ID = "aac35219-c38c-4490-8050-cee52e7dcf6b"
ATS = f"https://jobs.ashbyhq.com/acme/{ID}"
OFFICIAL = "https://acme.example/careers"
BODY = "Responsibilities: Build and operate our platform. Qualifications: software engineering."
COPIES = ["https://builtin.com/job/platform-engineer/12345",
          "https://recruiting.paylocity.com/Recruiting/Jobs/Details/12345",
          "https://careers-acme.icims.com/jobs/12345/platform-engineer/job"]


def page(url, text):
    return {"url": url, "text": text}


def test_official_custom_posting_links_bind_same_uuid_only():
    identity = _identity("Acme", "acme.example")
    source = page(OFFICIAL, f'<a href="/careers/{ID}">Machine Learning Engineer</a>')
    assert intent._ats_ownership_anchors([source], [ATS], identity)
    for html in [f'<a href="https://other.example/careers/{ID}">Machine Learning Engineer</a>',
                 '<a href="/careers/12345">Machine Learning Engineer</a>',
                 f'<a href="/customers/{ID}">Customer story</a>']:
        assert not intent._ats_ownership_anchors([page(OFFICIAL, html)], [ATS], identity)


@pytest.mark.parametrize("url", COPIES)
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.asyncio
async def test_reposted_job_body_can_ground_employer_without_official_link(monkeypatch, url, bound):
    judge = AsyncMock(return_value=_verdict(url, "supported", BODY))
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    contents = {"results": [page(url, BODY)], "statuses": []}
    async def fetch(urls, **kwargs):
        if urls == [url]:
            return contents
        return {"results": [page("https://acme.example/",
                    f'<a href="{url}">Careers: Platform Engineer</a>' if bound else "Our company homepage.")],
                "statuses": []}
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    result = await intent.verify_three_stage(None, company_name="Acme",
        company_website="https://acme.example", company_linkedin="https://linkedin.com/company/acme",
        source_url=url, miner_claim="Acme is hiring a platform engineer.",
        target_signal_text="Hiring platform engineers", evidence_type="HIRING",
        declared_source="job_board", stage1_soft_reject=True, integrity_policy=True,
        company_quality=True, verified_company_identity=_identity("Acme", "acme.example"))
    assert result["client_ready"] is True
    if bound and "builtin.com" not in url:
        assert result["job_publisher_relationship"] == "verified"
    if "builtin.com" not in url:
        assert "OPTIONAL EMPLOYER RELATIONSHIP OBSERVATIONS" in judge.await_args.args[2]


@pytest.mark.asyncio
async def test_failed_render_without_links_uses_existing_exa_fallback(monkeypatch):
    client = _ScrapingDogClient([httpx.ReadTimeout("render timeout"),
        httpx.Response(200, text="<html><body>" + ("Read about our company and our culture. " * 100) + "</body></html>")])
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-placeholder")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda: client)
    fallback = AsyncMock(return_value={"ok":True,"stage":"exa","content":f"[Platform Engineer]({ATS})"})
    monkeypatch.setattr(intent, "_scrape_exa", fallback)
    result = await intent._fetch_sd_then_exa([OFFICIAL], prefer_dynamic_job_index=True,
        trusted_company_domain="acme.example", trusted_company_name="Acme")
    fallback.assert_awaited_once_with(OFFICIAL)
    assert len(client.calls) == 2
    assert intent._ats_ownership_anchors(result["results"], [ATS], _identity("Acme", "acme.example"))


@pytest.mark.parametrize("url", ["https://notbuiltin.com/jobs/123", "https://builtin.com.other.example/job/123"])
def test_job_board_hostname_matching_is_exact(url):
    assert not intent._is_job_board_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_entity,status", [("fail","wrong_entity"),("unclear","unable_to_verify")])
@pytest.mark.parametrize("url", COPIES)
async def test_copied_posting_still_rejects_wrong_or_unresolved_employer(monkeypatch,url,same_entity,status):
    verdict=_verdict(url,status,BODY)
    verdict['answer']['signal_evaluations'][0]['same_entity_check']=same_entity
    monkeypatch.setattr(intent,'_call_openrouter',AsyncMock(return_value=verdict))
    async def fetch(urls,**_kwargs):
        return {'results':[page(url,BODY)] if urls==[url] else [],'statuses':[]}
    monkeypatch.setattr(intent,'_fetch_sd_then_exa',fetch)
    result=await intent.verify_three_stage(None,company_name='Acme',company_website='https://acme.example',
        company_linkedin='',source_url=url,miner_claim='Acme is hiring a platform engineer.',
        target_signal_text='Hiring platform engineers',evidence_type='HIRING',declared_source='job_board',
        stage1_soft_reject=True,integrity_policy=True,company_quality=True,
        verified_company_identity=_identity('Acme','acme.example'))
    assert not result['client_ready']
    if same_entity=='fail':
        assert result['decision']=='reject'

@pytest.mark.asyncio
@pytest.mark.parametrize('url',COPIES)
async def test_unreadable_copy_body_remains_unverified(monkeypatch,url):
    judge=AsyncMock()
    monkeypatch.setattr(intent,'_call_openrouter',judge)
    monkeypatch.setattr(intent,'_fetch_sd_then_exa',AsyncMock(return_value={
        'results':[page(url,'Please enable JavaScript to view this page.')],'statuses':[]}))
    result=await intent.verify_three_stage(None,company_name='Acme',company_website='https://acme.example',
        company_linkedin='',source_url=url,miner_claim='Acme is hiring a platform engineer.',
        target_signal_text='Hiring platform engineers',evidence_type='HIRING',declared_source='job_board',
        stage1_soft_reject=True,integrity_policy=True,company_quality=True,
        verified_company_identity=_identity('Acme','acme.example'))
    judge.assert_not_awaited()
    assert not result['client_ready']
    assert result['verdict']['signal_evaluations'][0]['signal_status']=='unable_to_verify'
