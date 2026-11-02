"""A failed optional read must not discard independently grounded findings."""
import asyncio

import aiohttp
import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_investigator import _attribute_finding
from tests.test_company_evidence_refusal_fallback import (
    SUBSCRIPTION, URL, _install_loop,
)


OPTIONAL_URL = "https://www.coreworks.ai/additional-details"


@pytest.mark.parametrize("failure", [asyncio.TimeoutError, aiohttp.ClientConnectionError])
@pytest.mark.parametrize("proof", ["verified", "missing", "invented"])
def test_failed_optional_read_preserves_only_validated_proof(monkeypatch, failure, proof):
    finding = (_attribute_finding() if proof == "missing" else
               _attribute_finding("VERIFIED", url=URL, quote=(
                   SUBSCRIPTION if proof == "verified" else "Invented pricing proof.")))
    run, calls, _ = _install_loop(monkeypatch, urls=(URL, OPTIONAL_URL),
                                   finding=finding, contents=asyncio.TimeoutError())

    async def raw(_session, _url, **kwargs):
        url = kwargs.get("params", {}).get("url", _url)
        calls["primary"].append(url)
        if url == OPTIONAL_URL:
            raise failure("source read failed")
        return 200, url, SUBSCRIPTION

    monkeypatch.setattr(c, "_fetch_bounded_html", raw)
    result = asyncio.run(run())
    assert calls["primary"] == [URL, OPTIONAL_URL]
    assert len(calls["contents"]) == (1 if failure is asyncio.TimeoutError else 0)
    assert result["usage"]["fetch_calls"] == len(calls["primary"]) + len(calls["contents"])
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == SUBSCRIPTION
    assert result["usage"]["fetch_outcomes"][-1]["ok"] is False
    if proof == "verified":
        assert result["_completed_submit"] is True
        assert result["failure_reason"] == ""
        assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    else:
        assert result.get("_completed_submit", False) is False
        assert result["failure_reason"] == "provider_error"
        assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("failure", [
    c._ArenaProviderFetchError("unsettled request", broker_status=409),
    RuntimeError("billing guard failed"),
])
def test_financial_or_broker_failure_still_stops_investigation(monkeypatch, failure):
    run, _, _ = _install_loop(monkeypatch)

    async def raw(*args, **kwargs):
        raise failure

    monkeypatch.setattr(c, "_fetch_bounded_html", raw)
    result = asyncio.run(run())
    assert result.get("_completed_submit", False) is False
    assert result["failure_reason"] == "provider_error"
    assert result["claims"] == {}


def test_source_cancellation_is_not_an_optional_failure(monkeypatch):
    run, _, _ = _install_loop(monkeypatch)

    async def raw(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(c, "_fetch_bounded_html", raw)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


@pytest.mark.parametrize("stealth", [False, True])
def test_prefetch_or_stealth_failure_retains_independent_cached_source(monkeypatch, stealth):
    finding = _attribute_finding("VERIFIED", url=URL, quote=SUBSCRIPTION)
    run, calls, _ = _install_loop(monkeypatch, finding=finding,
                                 contents=asyncio.TimeoutError())
    original_post = c._post_json

    async def post(session, url, **kwargs):
        if url == "https://api.exa.ai/contents":
            calls["contents"].append(kwargs["payload"])
            assert kwargs["payload"]["ids"] == [OPTIONAL_URL]
            raise asyncio.TimeoutError()
        return await original_post(session, url, **kwargs)

    async def raw(_session, _url, **kwargs):
        params = kwargs.get("params", {})
        url = params.get("url", _url)
        calls["primary"].append(url)
        assert url == OPTIONAL_URL
        if stealth and params.get("stealth_mode") != "true":
            return 400, url, "Retry with stealth_mode=true"
        raise asyncio.TimeoutError()

    monkeypatch.setattr(c, "_fetch_bounded_html", raw)
    monkeypatch.setattr(c, "_post_json", post)
    # The optional page is selected before the model loop. Its failed read
    # cannot invalidate another page's exact validated evidence.
    result = asyncio.run(run(
        requested_product_service="Subscription software",
        prior_observations={"submitted_source_urls": [OPTIONAL_URL, URL]},
        homepage_navigation_locators=[{"url": OPTIONAL_URL, "label": "Pricing"}],
        prefetched_pages={URL: {"text": SUBSCRIPTION, "final_url": URL}},
    ))
    assert calls["primary"] == [OPTIONAL_URL] * (2 if stealth else 1)
    assert result["_completed_submit"] is True
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == SUBSCRIPTION
    assert result["usage"]["fetch_outcomes"][-1]["ok"] is False
