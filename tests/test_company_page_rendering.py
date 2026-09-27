from __future__ import annotations

import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as investigator


@pytest.mark.parametrize("stealth", [False, True])
def test_company_fetch_uses_existing_bounded_rendered_transport(monkeypatch, stealth):
    target = "https://acme.example/pricing"
    calls = []

    async def read(_session, url, *, params=None):
        calls.append((url, params))
        return 200, url, """<html><body><main><h1>Workspace plans</h1>
          <table><tr><td>Pro</td><td>$25/month</td></tr></table>
          <p>Manage your production cloud applications.</p>
          <script>Hidden enterprise guarantee</script>
        </main></body></html>"""

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(investigator, "_fetch_bounded_html", read)
    page = asyncio.run(investigator._fetch_page(object(), target, stealth_mode=stealth))

    assert len(calls) == 1
    assert calls[0] == ("https://api.scrapingdog.com/scrape", {
        "api_key": "test-runtime-handle", "url": target,
        "dynamic": "true", "wait": "5000",
        **({"stealth_mode": "true"} if stealth else {}),
    })
    assert page["url"] == page["final_url"] == target
    assert "$25/month" in page["text"]
    assert "Hidden enterprise guarantee" not in page["text"]


def test_rendered_fetch_keeps_observed_redirect_identity(monkeypatch):
    target = "https://acme.example/pricing"

    async def read(_session, _url, *, params=None):
        return 200, "https://other.example/pricing", "Other sells software."

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(investigator, "_fetch_bounded_html", read)
    page = asyncio.run(investigator._fetch_page(object(), target))

    assert page["url"] == target
    assert page["final_url"] == "https://other.example/pricing"


def test_keyless_company_fetch_keeps_direct_transport(monkeypatch):
    target = "https://acme.example/pricing"
    calls = []

    async def read(_session, url):
        calls.append(url)
        return 200, url, "Acme sells software."

    monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setattr(investigator, "_fetch_bounded_html", read)
    assert asyncio.run(investigator._fetch_page(object(), target))["ok"]
    assert calls == [target]
