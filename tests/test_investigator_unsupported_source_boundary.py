"""Exercise source-policy refusals through the real worker response reader."""
import asyncio
import json

import pytest

from lab_arena import shim
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification as verification


SOURCE = "https://acme.example/report.pdf"
UNSUPPORTED = b'{"error":{"code":"unsupported_source"}}'


class Session:
    def __init__(self, status=415, body=UNSUPPORTED, headers=None, final_url=None):
        self.status = status
        self.body = body
        self.headers = headers or {"content-type": "application/json"}
        self.final_url = final_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def get(self, url, **_kwargs):
        return shim._AiohttpResponse(
            request_url=url, response_url=self.final_url,
            status=self.status, headers=self.headers, body=self.body,
        )


@pytest.mark.parametrize("has_call_identity", [False, True])
def test_exact_unsupported_source_is_a_non_content_fetch_result(monkeypatch, has_call_identity):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    headers = {"content-type": "application/json"}
    if has_call_identity:
        headers["x-leadpoet-call-identity"] = "sha256:" + "a" * 64
    result = asyncio.run(investigator._fetch_page(Session(headers=headers), SOURCE))
    assert result == {"ok": False, "error": "unsupported_source"}
    outcome = investigator._fetch_outcome(SOURCE, result)
    assert outcome["error_class"] == "unsupported_source"
    assert outcome["loaded_text_length"] == 0


@pytest.mark.parametrize("status,body", [
    (401, UNSUPPORTED), (403, UNSUPPORTED), (429, UNSUPPORTED),
    (502, UNSUPPORTED), (503, UNSUPPORTED),
    (415, b'{"error":{"code":"provider_error"}}'),
    (415, UNSUPPORTED + b" trailing text"),
    (415, b"<html>unsupported_source</html>"),
])
def test_untyped_or_wrong_status_refusal_remains_an_infrastructure_error(monkeypatch, status, body):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    with pytest.raises(verification._ArenaProviderFetchError) as caught:
        asyncio.run(investigator._fetch_page(Session(status, body), SOURCE))
    assert caught.value.broker_status == status
    assert caught.value.broker_error_code == ""


def test_typed_source_error_does_not_acquire_a_target_url():
    with pytest.raises(verification._ArenaProviderFetchError) as caught:
        asyncio.run(verification._fetch_bounded_html(Session(), SOURCE))
    assert caught.value.broker_status == 415
    assert caught.value.broker_error_code == "unsupported_source"
    assert not hasattr(caught.value, "final_url")


def test_normal_loaded_html_still_retains_its_actual_final_url(monkeypatch):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    final = "https://www.acme.example/product"
    session = Session(200, b"<main>Acme supplies commercial workflow software.</main>",
                      {"content-type": "text/html"}, final)
    result = asyncio.run(investigator._fetch_page(session, "https://acme.example/product"))
    assert result["ok"] is True
    assert result["final_url"] == final
    assert "commercial workflow software" in result["text"]


def test_refusal_constructor_preserves_only_exact_status_code_pairs():
    for status, code, expected in [
        (415, "unsupported_source", "unsupported_source"),
        (403, "unsupported_source", ""),
        (415, "provider_request_refused", ""),
        (403, "provider_request_refused", "provider_request_refused"),
    ]:
        error = verification._ArenaProviderFetchError(
            "unavailable", broker_status=status, broker_error_code=code)
        assert error.broker_error_code == expected


def test_homepage_refusal_is_unavailable_without_identity_or_page_evidence(monkeypatch):
    monkeypatch.setattr(verification.aiohttp, "ClientSession", lambda **_kwargs: Session())
    sources = {}
    result = asyncio.run(verification.verify_company_exists(
        "Acme", "https://acme.example", homepage_evidence_sink=sources))
    assert result.decision == "unavailable"
    assert result.details["failure_reason_code"] == "source_blocked"
    assert sources == {}


@pytest.mark.parametrize("alternative_succeeds", [True, False])
def test_native_tool_loop_continues_after_unsupported_source(monkeypatch, alternative_succeeds):
    from tests.test_company_evidence_investigator import _finding

    product = "https://acme.example/product"
    quote = "Acme supplies commercial workflow software."
    fetched = []

    class Sources(Session):
        def get(self, url, **kwargs):
            target = (kwargs.get("params") or {}).get("url", url)
            fetched.append(target)
            if target == SOURCE:
                return Session().get(url)
            assert alternative_succeeds and target == product
            return Session(200, ("<main>" + quote + "</main>").encode(),
                           {"content-type": "text/html"}, product).get(url)

    turns = [("fetch_page", {"url": SOURCE})]
    if alternative_succeeds:
        turns.append(("fetch_page", {"url": product}))
    turns.append(("submit_findings", {"findings": [_finding(
        "industry", status="VERIFIED" if alternative_succeeds else "UNPROVEN",
        observed_value="", observed_industry="Software",
        activity_role="supplier_operator" if alternative_succeeds else "unresolved",
        evidence_url=product if alternative_succeeds else "",
        evidence_quote=quote if alternative_succeeds else "",
    )]}))

    async def model(_session, _url, *, headers, payload):
        if _url == "https://api.exa.ai/search":
            return 200, {"results": []}
        choice = payload.get("tool_choice")
        forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
        if forced == "search_web":
            name, arguments = "search_web", {"query": "Acme software products"}
        else:
            name, arguments = turns.pop(0) if len(turns) > 1 else turns[0]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "call-" + str(len(turns)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    for name in ("OPENROUTER_API_KEY", "EXA_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(name, "offline-placeholder")
    monkeypatch.setattr(investigator.aiohttp, "ClientSession", lambda **_kwargs: Sources())
    monkeypatch.setattr(investigator, "_post_json", model)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Software",
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example"}))
    assert len(turns) == 1
    assert fetched == ([SOURCE, product] if alternative_succeeds else [SOURCE])
    assert "usage" in result, result
    assert result["usage"]["fetch_outcomes"][0]["error_class"] == "unsupported_source"
    pages = result.get(investigator.PRIVATE_FETCHED_PAGES_KEY, {})
    assert SOURCE not in pages
    if alternative_succeeds:
        assert result["claims"]["industry"]["status"] == "VERIFIED", result["claims"]["industry"]["reason"]
        assert product in pages
    else:
        assert result["claims"]["industry"]["status"] == "UNPROVEN"
        assert result.get("failure_reason") == "provider_error"
