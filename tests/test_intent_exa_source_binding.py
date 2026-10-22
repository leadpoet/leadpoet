"""Exa page identity binds body, title, and publication date together."""

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as verifier


URL = "https://example.test/news/launch"
OTHER = "https://example.test/news/related"
BODY = "The requested page describes its own launch. " * 12


def _result(url=URL, **extra):
    return {"url": url, "text": BODY, "title": "Launch report",
            "publishedDate": "2026-08-04T12:00:00Z", **extra}


def _transport(monkeypatch, documents):
    replies = iter(documents)
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, endpoint, *, json, **_kwargs):
            assert endpoint == "https://api.exa.ai/contents"
            assert json["ids"] == [URL]
            calls.append(endpoint)
            return httpx.Response(200, json=next(replies))

    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(verifier.httpx, "AsyncClient", lambda **_kwargs: Client())
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    _result(),
    _result("HTTPS://EXAMPLE.TEST/news/launch/#section", id=URL),
    _result(id="opaque-exa-document-123"),
])
async def test_exact_page_admits_all_three_fields(monkeypatch, result):
    calls = _transport(monkeypatch, [{"results": [result]}])
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 1
    assert scraped["ok"] is True
    assert scraped["content"] == BODY
    assert scraped["source_title"] == "Launch report"
    assert scraped["source_publication_date"] == "2026-08-04"


@pytest.mark.asyncio
async def test_other_page_before_exact_page_cannot_supply_any_field(monkeypatch):
    exact = _result(title="Exact launch", publishedDate="2026-07-01")
    calls = _transport(monkeypatch, [{"results": [
        _result(OTHER, text="Wrong article. " * 30), exact,
    ]}])
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 1
    assert scraped["content"] == BODY
    assert scraped["source_title"] == "Exact launch"
    assert scraped["source_publication_date"] == "2026-07-01"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    _result(OTHER),
    _result(url=""),
    _result(url=OTHER, id=URL),
])
async def test_unbound_body_date_and_title_are_rejected_after_existing_retry(monkeypatch, result):
    calls = _transport(monkeypatch, [{"results": [result]}] * 2)
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 2
    assert scraped == {"ok": False, "stage": "exa_source_mismatch",
                       "content": "", "error": "target_result_mismatch"}
    assert "target_absence" not in scraped


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [
    [{"id": URL, "status": "success"}],
    [],
    [{"id": OTHER, "status": "success"}],
    [{"id": URL, "status": "error"}],
])
async def test_off_page_result_stays_unavailable_despite_requested_id_and_status(
    monkeypatch, statuses
):
    calls = _transport(monkeypatch, [{"results": [_result(OTHER, id=URL)],
                                    "statuses": statuses}] * 2)
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 2
    assert scraped["ok"] is False
    assert scraped["stage"] == "exa_source_mismatch"
    assert "target_absence" not in scraped


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [
    [_result(), _result()],
    [_result(), _result(error="crawl failed")],
    [42, _result()],
])
async def test_duplicate_or_malformed_results_are_ambiguous(monkeypatch, results):
    calls = _transport(monkeypatch, [{"results": results}] * 2)
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 2
    assert scraped["stage"] == "exa_source_mismatch"


@pytest.mark.asyncio
async def test_error_status_cannot_admit_even_exact_result(monkeypatch):
    calls = _transport(monkeypatch, [{
        "results": [_result()],
        "statuses": [{"id": URL, "status": "error"}],
    }] * 2)
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 2
    assert scraped["stage"] == "exa_source_mismatch"
    assert scraped["content"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("document", [
    {"results": [_result()], "error": "provider failed"},
    {"results": [_result()], "errors": ["provider failed"]},
    {"results": [_result(error="crawl failed")]},
    {"results": [_result(error="")]},
    {"results": [_result(status="error")]},
    {"results": [_result()], "statuses": [
        {"id": URL, "status": "success", "error": {"tag": "CRAWL_FAILED"}},
    ]},
    {"results": [_result()], "statuses": "success"},
    {"results": [], "error": "provider failed", "statuses": [
        {"id": URL, "status": "error", "error": {
            "tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404,
        }},
    ]},
])
async def test_explicit_or_malformed_errors_never_admit_content_or_absence(
    monkeypatch, document
):
    calls = _transport(monkeypatch, [document] * 2)
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 2
    assert scraped["ok"] is False
    assert scraped["content"] == ""
    assert scraped["stage"] == "exa_source_mismatch"
    assert "target_absence" not in scraped


@pytest.mark.asyncio
async def test_null_or_empty_optional_error_fields_keep_exact_page_valid(monkeypatch):
    calls = _transport(monkeypatch, [{
        "results": [_result(id="opaque-document-id", error=None)],
        "error": None, "errors": [],
        "statuses": [{"id": URL, "status": "success", "error": None}],
    }])
    scraped = await verifier._scrape_exa(URL)
    assert len(calls) == 1
    assert scraped["ok"] is True
    assert scraped["content"] == BODY


@pytest.mark.asyncio
@pytest.mark.parametrize("date,expected", [
    ("2026-08-04", "2026-08-04"), ("", ""),
])
async def test_facebook_route_preserves_bound_publication_date(monkeypatch, date, expected):
    fb_url = "https://www.facebook.com/example/posts/123456"

    async def no_ats_route(_url):
        return {"routed": False}

    async def exa(_url):
        assert _url == fb_url
        return {"ok": True, "stage": "exa_scraped", "content": BODY,
                "source_publication_date": date}

    for name in ("_scrape_ashby_job", "_scrape_greenhouse_job", "_scrape_workday_cxs"):
        monkeypatch.setattr(verifier, name, no_ats_route)
    monkeypatch.setattr(verifier, "_scrape_exa", exa)
    fetched = await verifier._fetch_sd_then_exa([fb_url])
    assert fetched["results"][0]["source_publication_date"] == expected
    assert verifier._project_contents_for_prompt(fetched)["results"][0][
        "source_publication_date"] == expected
