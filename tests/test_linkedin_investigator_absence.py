"""Exact source absence must survive the LinkedIn investigator fetch seam."""
import asyncio
import copy

import pytest

from qualification.scoring import company_evidence_investigator as c, linkedin_company_size as l
from tests.test_current_linkedin_company_size import _install_exa_bodies

URL = "https://www.linkedin.com/company/acme"


def absence(status=404, url=URL):
    return {"results": [], "statuses": [{"id": url, "status": "error",
        "error": {"httpStatusCode": status, "tag": "CRAWL_NOT_FOUND"}}]}


@pytest.mark.parametrize("status,expected", [(404, "source_not_found"), (410, "source_gone")])
@pytest.mark.parametrize("url", [URL, "https://linkedin.com/company/acme", "https://ca.linkedin.com/company/acme"])
def test_exact_absence_retains_source_identity(monkeypatch, status, expected, url):
    _install_exa_bodies(monkeypatch, absence(status, url))
    source = {"url": "https://old.example", "text": "Old evidence", "http_status": "500"}
    result = asyncio.run(l.fetch_current_linkedin_company_size(URL, source_text_sink=source))
    assert result == {"outcome": "insufficient_evidence", "url": url}
    assert source == {"url": url, "text": "", "http_status": str(status)}
    _install_exa_bodies(monkeypatch, absence(status, url))
    fetched = asyncio.run(c._fetch_page(object(), URL))
    assert fetched == {"ok": False, "error": "http_" + str(status)}
    outcome = c._fetch_outcome(URL, fetched)
    assert outcome["error_class"] == expected and outcome["url_sha256"]
    assert outcome["ok"] is False and outcome["loaded_text_length"] == 0


@pytest.mark.parametrize("body,expected", [
    *[(absence(status), "provider_error") for status in (401, 403, 429, 500, 502, 503, 504)],
    (absence(404, "https://www.linkedin.com/company/other"), "provider_error"),
    (absence(404, URL + "/life"), "provider_error"),
    (absence(404, URL + "?other=1"), "provider_error"),
    (absence(404, "https://evil.example/company/acme"), "provider_error"),
    ({"results": [], "statuses": [{"id": URL, "status": "error",
        "error": {"httpStatusCode": "404", "tag": "CRAWL_NOT_FOUND"}}]}, "provider_error"),
    ({"results": [], "statuses": [{"id": URL, "status": "error",
        "error": {"httpStatusCode": 404, "tag": "OTHER_ERROR"}}]}, "provider_error"),
    (dict(absence(), error="account failure"), "provider_error"),
    (dict(absence(), results=[{"url": URL, "text": "conflicting response"}]), "provider_error"),
    ({"results": []}, "malformed_response"),
    ({"results": [{"url": URL, "text": ""}]}, "empty_page"),
])
def test_absence_does_not_clear_other_failures(monkeypatch, body, expected):
    _install_exa_bodies(monkeypatch, copy.deepcopy(body))
    fetched = asyncio.run(c._fetch_page(object(), URL))
    assert fetched == {"ok": False, "error": expected}
    assert c._fetch_outcome(URL, fetched)["error_class"] not in {"source_not_found", "source_gone"}


def test_timeout_remains_provider_failure(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    class Session:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        def post(self, *args, **kwargs): raise asyncio.TimeoutError()
    monkeypatch.setattr(l.aiohttp, "ClientSession", Session)
    assert asyncio.run(c._fetch_page(object(), URL)) == {"ok": False, "error": "provider_error"}


@pytest.mark.parametrize("text", ["# Acme\n# About\nCompany size\n11-50 employees\nWebsite\nhttps://acme.example",
    "# About\nAcme supplies software."])
def test_valid_exact_profile_text_still_loads(monkeypatch, text):
    _install_exa_bodies(monkeypatch, {"results": [{"url": URL, "text": text}]})
    fetched = asyncio.run(c._fetch_page(object(), URL))
    assert fetched["ok"] is True and fetched["text"] == text and fetched["final_url"] == URL


@pytest.mark.parametrize("outer_status", [403, 429, 502, 503])
def test_outer_provider_failure_overrides_inner_absence(monkeypatch, outer_status):
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    class Response:
        status = outer_status
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def json(self): return absence()
    class Session:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        def post(self, *args, **kwargs): return Response()
    monkeypatch.setattr(l.aiohttp, "ClientSession", Session)
    assert asyncio.run(c._fetch_page(object(), URL)) == {"ok": False, "error": "provider_error"}
