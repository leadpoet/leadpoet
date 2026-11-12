"""Verified recruiting widgets prove configured board ownership, not observed jobs."""
import pytest
from qualification.scoring import intent_verification_three_stage as v

SOURCE='https://lightedge.com/careers/'
TARGET='https://jobs.lever.co/lightedge/e8c35358-11e9-43c6-9f62-2f7ed719f648'
LOADER='https://andreasmb.github.io/lever-jobs-embed/index.js'
IDENTITY={'observed_domain':'lightedge.com'}
HTML=("<div id='lever-jobs-container'></div>"
      "<script>window.leverJobsOptions = {accountName: 'Lightedge', includeCss: true};</script>"
      f"<script src='{LOADER}'></script>")


def extract(body=HTML, source=SOURCE, identity=IDENTITY):
    return v._configured_lever_embed(body,source,identity)


def test_known_official_lever_embed_emits_configured_board_provenance():
    records=extract()
    assert len(records)==1
    record=records[0]
    assert record['kind']=='configured_lever_board'
    assert record['board_url']=='https://jobs.lever.co/lightedge'
    assert record['source_url']==SOURCE
    assert record['loader_url']==LOADER
    anchors=v._ats_ownership_anchors([
        {'url':SOURCE,'text':'Official careers prose','meta':{'configured_ats_boards':records}},
    ],[TARGET],IDENTITY)
    assert anchors and anchors[0]['kind']=='official_configured_board'
    assert anchors[0]['configured_board_url']=='https://jobs.lever.co/lightedge'
    assert 'linked_url' not in anchors[0]  # Do not claim an observed anchor/job.


@pytest.mark.parametrize('body',[
    HTML.replace("id='lever-jobs-container'", "id='another'"),
    HTML.replace(LOADER,LOADER+'.evil'),
    HTML.replace(LOADER,LOADER.replace('github.io','github.io.evil.example')),
    HTML.replace(LOADER,LOADER+'?tenant=lightedge'),
    HTML.replace("accountName: 'Lightedge'", "accountName: companyName"),
    HTML.replace("accountName: 'Lightedge'", "accountName: 'Lightedge', accountName: 'another'"),
    HTML+"<script>window.leverJobsOptions = {accountName: 'another'};</script>",
    '<!--'+HTML+'-->',
    '<template>'+HTML+'</template>',
    HTML.replace("<div id=", "<div hidden id="),
    HTML.replace("window.leverJobsOptions = {accountName: 'Lightedge', includeCss: true};",
                 "// window.leverJobsOptions = {accountName: 'Lightedge'};"),
    HTML.replace("window.leverJobsOptions = {accountName: 'Lightedge', includeCss: true};",
                 "console.log(\"window.leverJobsOptions = {accountName: 'Lightedge'};\");"),
    HTML.replace("accountName: 'Lightedge'", "accountName: '../lightedge'"),
    HTML.replace("accountName: 'Lightedge'", "accountName: 'lightedge.evil.example'"),
])
def test_missing_conflicting_computed_or_nonactive_embed_is_not_ownership(body):
    assert not extract(body)


@pytest.mark.parametrize('source',[
    'https://unrelated.example/careers/',
    'https://lightedge.com.evil.example/careers/',
    'https://lightedge.com/news/',
])
def test_configured_embed_requires_verified_official_careers_source(source):
    assert not extract(source=source)


def test_names_alone_and_wrong_selected_tenant_are_not_ownership():
    assert not extract('<p>Lightedge uses Lever; accountName Lightedge.</p>')
    records=extract()
    result={'url':SOURCE,'text':'Official careers prose','meta':{'configured_ats_boards':records}}
    assert not v._ats_ownership_anchors([result],[TARGET.replace('/lightedge/','/another/')],IDENTITY)
    assert not v._ats_ownership_anchors([{**result,'url':'https://unrelated.example/careers/'}],[TARGET],IDENTITY)


def test_ambiguous_duplicate_assignment_is_not_a_single_static_configuration():
    assert not extract(HTML+"<script>window.leverJobsOptions = {accountName: 'Lightedge'};</script>")


def test_conflicting_typed_receipts_are_not_an_unambiguous_board_binding():
    records=extract()
    records.append({**records[0],'account_name':'other','board_url':'https://jobs.lever.co/other'})
    assert not v._ats_ownership_anchors([
        {'url':SOURCE,'text':'Official careers prose','meta':{'configured_ats_boards':records}},
    ],[TARGET],IDENTITY)


@pytest.mark.asyncio
async def test_native_careers_fetch_preserves_configured_board_without_extra_call(monkeypatch):
    import httpx
    from unittest.mock import AsyncMock
    body='<html><body>'+HTML+''.join(
        f'<p>Our engineers support customer infrastructure, collaborate with teams and improve operations in region {i}.</p>'
        for i in range(30)
    )+'</body></html>'
    calls=[]
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
        async def get(self,url,**kwargs):
            calls.append(url)
            return httpx.Response(200,text=body,headers={'content-type':'text/html'},request=httpx.Request('GET',url))
    monkeypatch.setenv('SCRAPINGDOG_API_KEY','offline-not-a-key')
    monkeypatch.setattr(v.httpx,'AsyncClient',Client)
    exa=AsyncMock()
    monkeypatch.setattr(v,'_scrape_exa',exa)
    result=await v._fetch_sd_then_exa([SOURCE],prefer_dynamic_job_index=True,
        trusted_company_domain='lightedge.com',trusted_company_name='Lightedge')
    assert len(calls)==1
    exa.assert_not_awaited()
    assert result['results'][0]['meta']['configured_ats_boards'][0]['account_name']=='lightedge'
    assert v._ats_ownership_anchors(result['results'],[TARGET],IDENTITY)


@pytest.mark.parametrize('body',[
    HTML.replace('<script>', '<script type="application/json">'),
    HTML.replace("id='lever-jobs-container'", "id='other' id='lever-jobs-container'"),
    HTML.replace('<script>', '<script type="text/javascript" type="application/json">'),
])
def test_inert_script_and_ambiguous_html_attributes_do_not_bind(body):
    assert not extract(body)
