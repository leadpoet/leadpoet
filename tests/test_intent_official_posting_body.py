"""Detailed first-party postings need not use fixed job-body headings."""
from unittest.mock import AsyncMock
import pytest
from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_first_party_careers_dynamic import _identity, _verdict

URL = 'https://acme.example/careers/8b304769-209b-4798-a4bb-9207e042da59'
BODY = '''# Machine Learning Engineer
We would love to meet you if you enjoy turning difficult research into reliable software for real customers.
You have built complex learning systems and measured how their predictions behave under changing data distributions.
The core work will include developing training pipelines and designing efficient model architectures for document understanding.
You will own the complete lifecycle of our inference platform from experimental design through deployment and ongoing optimization.
Collaborate with our engineering team to improve accuracy and latency while keeping operating costs predictable for customers.
Our engineers work closely with researchers and customers to understand failures and translate them into measurable improvements.
We value clear thinking and careful experimentation, and give each engineer substantial autonomy over technical decisions.
The team meets in our office each week to share new findings and plan the next set of experiments.
Compensation includes salary and equity alongside health coverage and flexible time away from the office.
'''

def test_official_body_without_fixed_phrases_can_reach_judge():
    assert not intent._looks_like_job_body(BODY)
    assert intent._substantial_official_posting_body({'url': URL, 'text': BODY}, URL, trusted_official=True)

@pytest.mark.parametrize('url,text,trusted,result_url', [
    (URL, BODY, False, URL),
    ('https://acme.example/careers', BODY, True, 'https://acme.example/careers'),
    (URL, BODY, True, 'https://acme.example/news/story'),
    (URL, 'Machine Learning Engineer', True, URL),
    (URL, 'Learn about our workplace and meet our wonderful people. ' * 50, True, URL),
    (URL, 'Enable JavaScript to view this page. ' + BODY, True, URL),
    (URL, 'Sign in to continue. ' + BODY, True, URL),
    (URL, '', True, URL),
])
def test_fallback_rejects_untrusted_shell_index_and_unrelated_result(url,text,trusted,result_url):
    assert not intent._substantial_official_posting_body({'url':result_url,'text':text},url,trusted_official=trusted)

@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['supported', 'contradicted'])
async def test_first_party_body_reaches_unchanged_semantic_judge(monkeypatch,status):
    judge=AsyncMock(return_value=_verdict(URL,status,BODY.splitlines()[4] if status=='supported' else ''))
    monkeypatch.setattr(intent,'_call_openrouter',judge)
    monkeypatch.setattr(intent,'_fetch_sd_then_exa',AsyncMock(return_value={
        'results':[{'url':URL,'text':BODY}], 'statuses':[]}))
    result=await intent.verify_three_stage(None,company_name='Acme',
        company_linkedin='https://www.linkedin.com/company/acme',company_website='https://acme.example',
        source_url=URL,miner_claim='Acme is hiring a platform engineer.',target_signal_text='Hiring platform engineers',
        evidence_type='HIRING',declared_source='job_board',stage1_soft_reject=True,
        company_quality=True,integrity_policy=True,verified_company_identity=_identity('Acme','acme.example'))
    assert judge.await_count == 1
    assert result['rejection_reason'] != 'job_body_not_in_fetched_content'
    if status=='contradicted':
        assert not result['client_ready']
    else:
        assert result['client_ready'] is True
        assert result['decision'] == 'approve'
        assert result['job_publisher_relationship'] == 'verified'
        assert result['verified_job_source_urls'] == [URL]
        assert result['verdict']['signal_evaluations'][0]['evidence_urls_used'] == [URL]
    gate=next(s for s in result['scrape']['statuses'] if s.get('stage')=='job_body_gate')
    assert gate['reason']=='substantial_official_posting_deferred_to_judge'
    assert gate['text_chars']==len(BODY)
    assert BODY not in str(gate)

ATS_URL = 'https://jobs.ashbyhq.com/acme/8b304769-209b-4798-a4bb-9207e042da59'
ATS_STATUS = {'url': ATS_URL, 'source':'scrapingdog_ashby_api', 'stage':'sd:ashby_api:1'}

@pytest.mark.parametrize('kind,status,text,closed,source_url', [
    ('ashby_job', ATS_STATUS, 'Machine Learning Engineer', False, ATS_URL),
    ('ashby_job', ATS_STATUS, BODY, True, ATS_URL),
    ('ashby_job', None, BODY, False, ATS_URL),
    ('greenhouse_job', ATS_STATUS, BODY, False, ATS_URL),
    ('ashby_job', ATS_STATUS, BODY, False, ATS_URL.replace('8b304769', '7b304769')),
    ('ashby_job', ATS_STATUS, 'Enable JavaScript. '+BODY, False, ATS_URL),
])
def test_native_body_fallback_requires_exact_typed_usable_receipt(kind,status,text,closed,source_url):
    result={'url':source_url,'text':text,'meta':{'kind':kind,'is_closed':closed}}
    assert not intent._substantial_official_posting_body(
        result,source_url,trusted_official=False,native_api_status=status)

@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['supported', 'contradicted'])
async def test_native_parser_body_without_fixed_headings_reaches_judge(monkeypatch,status):
    import httpx
    from tests.test_intent_first_party_careers_dynamic import _ScrapingDogClient
    payload={'jobs':[{'id':ATS_URL.rsplit('/',1)[1], 'jobUrl':ATS_URL,
        'title':'Machine Learning Engineer','descriptionPlain':BODY,'isListed':True}]}
    client=_ScrapingDogClient([httpx.Response(200,json=payload)])
    monkeypatch.setenv('SCRAPINGDOG_API_KEY','test-placeholder')
    monkeypatch.setattr(intent.httpx,'AsyncClient',lambda **_kwargs:client)
    native=await intent._scrape_ashby_job(ATS_URL)
    assert native['ok']
    assert not intent._looks_like_job_body(native['content'])
    fetched={'results':[{'url':ATS_URL,'text':native['content'],'meta':{'kind':'ashby_job'}}],
        'statuses':[{'url':ATS_URL,'source':'scrapingdog_ashby_api','stage':native['stage']}]}
    async def fetch(urls,**_kwargs):
        return fetched if urls==[ATS_URL] else {'results':[],'statuses':[]}
    monkeypatch.setattr(intent,'_fetch_sd_then_exa',fetch)
    judge=AsyncMock(return_value=_verdict(ATS_URL,status,BODY.splitlines()[4] if status=='supported' else ''))
    monkeypatch.setattr(intent,'_call_openrouter',judge)
    result=await intent.verify_three_stage(None,company_name='Acme',company_website='https://acme.example',
        company_linkedin='',source_url=ATS_URL,miner_claim='Acme is hiring a platform engineer.',
        target_signal_text='Hiring platform engineers',evidence_type='HIRING',declared_source='job_board',
        stage1_soft_reject=True,company_quality=True,integrity_policy=True,
        verified_company_identity=_identity('Acme','acme.example'))
    assert judge.await_count==1
    assert result['rejection_reason']==('stage3_identity_unresolved' if status=='supported' else 'stage3_contradicted')
    assert not result['client_ready']
