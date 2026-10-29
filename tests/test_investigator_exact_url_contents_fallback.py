import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from lab_arena import operations
from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_investigator import _finding


URL = "https://acme.example/about"
QUOTE = "Acme sells commercial workflow software to enterprise customers."


def _review(monkeypatch, *, error=None, cap=4, expired=False, timeout=False,
            late_settlement=False):
    clock = [0.0]
    calls = []
    turns = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(investigator, "MAX_FETCH_CALLS", cap)
    monkeypatch.setattr(investigator.time, "monotonic", lambda: clock[0])

    async def fetch(_session, url, **_kwargs):
        calls.append(("primary", url))
        if expired:
            clock[0] = investigator.ADMISSION_DEADLINE_SECONDS
        if timeout:
            raise asyncio.TimeoutError
        return ({"ok": False, "error": error} if error else
                {"ok": True, "url": url, "final_url": url, "text": QUOTE})

    async def post(_session, endpoint, *, headers, payload):
        if endpoint == "https://api.exa.ai/contents":
            calls.append(("contents", payload["ids"][0]))
            assert payload == {"ids": [URL],
                               "text": {"maxCharacters": investigator.MAX_PAGE_CHARACTERS},
                               "maxAgeHours": 0,
                               "livecrawlTimeout": investigator.EXA_LIVECRAWL_TIMEOUT_MILLISECONDS}
            # The same admitted call may settle after its admission deadline.
            if late_settlement:
                clock[0] = investigator.ADMISSION_DEADLINE_SECONDS + 1
            return 200, {"results": [{"url": URL, "text": QUOTE}]}
        turns.append(True)
        if len(turns) == 1:
            name, args = "fetch_page", {"url": URL}
        elif isinstance(payload["tool_choice"], dict) and payload["tool_choice"]["function"]["name"] == "search_web":
            name, args = "search_web", {"query": "Acme software"}
        else:
            loaded = not error or any(kind == "contents" for kind, _ in calls)
            name, args = "submit_findings", {"findings": [_finding(
                "industry", status="VERIFIED" if loaded else "UNPROVEN",
                observed_industry="Software",
                activity_role="supplier_operator" if loaded else "unresolved",
                evidence_url=URL if loaded else "",
                evidence_quote=QUOTE if loaded else "",
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(turns)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Software",
        verified_homepage_identity={"normalized_name": "acme",
                                    "registrable_dns_domain": "acme.example"},
    ))
    return result, calls


@pytest.mark.parametrize("error", [None, "http_400", "http_401", "http_402", "http_403", "http_404", "http_422"])
def test_healthy_and_nontransient_paths_do_not_add_contents_calls(monkeypatch, error):
    result, calls = _review(monkeypatch, error=error)
    assert calls == [("primary", URL)]
    assert result["usage"]["fetch_calls"] == 1


@pytest.mark.parametrize("error,timeout", [("http_502", False), ("http_503", False), ("timeout", True)])
def test_one_counted_same_url_recovery_preserves_loaded_evidence(monkeypatch, error, timeout):
    result, calls = _review(monkeypatch, error=error, timeout=timeout)
    assert calls == [("primary", URL), ("contents", URL)]
    assert result["usage"]["fetch_calls"] == 2
    assert len(result["usage"]["fetch_outcomes"]) == 2
    # Expiring admission while a request settles does not erase its evidence.
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == QUOTE
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_admitted_contents_may_settle_without_a_new_cancellation_deadline(monkeypatch):
    result, calls = _review(monkeypatch, error="http_502", late_settlement=True)
    assert calls == [("primary", URL), ("contents", URL)]
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == QUOTE
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["_completed_submit"] is True


@pytest.mark.parametrize("cap,expired", [(1, False), (4, True)])
def test_recovery_never_adds_admission_time_or_fetch_slots(monkeypatch, cap, expired):
    result, calls = _review(monkeypatch, error="http_502", cap=cap, expired=expired)
    assert calls == [("primary", URL)]
    assert result["usage"]["fetch_calls"] == 1


@pytest.mark.parametrize("page", [
    {"url": "https://foreign.example/about", "text": QUOTE},
    {"url": URL, "text": ""},
    {"url": URL, "text": "Provider account capacity details redacted."},
    {"url": URL, "text": "%PDF-1.7 binary content"},
    {"url": URL, "text": QUOTE, "error": "crawl failed"},
])
def test_contents_cannot_relabel_other_or_invalid_evidence(monkeypatch, page):
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    post = AsyncMock(return_value=(200, {"results": [page]}))
    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator._fetch_exa_exact_page(object(), URL))
    assert result["ok"] is False
    assert post.await_count == 1


def test_contents_uses_existing_deepline_accounted_gateway_route():
    parameters = {"ids": [URL], "text": {"maxCharacters": investigator.MAX_PAGE_CHARACTERS}, "maxAgeHours": 0}
    normalized = operations.validate_operation_request("exa.contents", parameters)
    request = operations.build_outbound_request("exa.contents", normalized)
    assert operations.OPERATIONS["exa.contents"].provider == "deepline"
    assert request.url == "https://code.deepline.com/api/v2/integrations/exa_contents/execute"
    assert json.loads(request.body) == {"provider": "exa", "operation": "exa_contents", "payload": parameters}


@pytest.mark.parametrize("extra", [
    {"error": "crawl failed"}, {"errors": ["crawl failed"]},
    {"statuses": [{"id": URL, "status": "error"}]},
    {"statuses": [{"id": "opaque-document-id", "status": "error"}]},
])
def test_contents_rejects_failure_even_with_cached_result_text(monkeypatch, extra):
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(investigator, "_post_json", AsyncMock(return_value=(
        200, {"results": [{"url": URL, "text": QUOTE}], **extra})))
    assert not asyncio.run(investigator._fetch_exa_exact_page(object(), URL))["ok"]


@pytest.mark.parametrize("extra", [{}, {"statuses": [{"id": URL, "status": "success"}]}])
def test_contents_accepts_optional_success_status(monkeypatch, extra):
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(investigator, "_post_json", AsyncMock(return_value=(
        200, {"results": [{"url": URL, "text": QUOTE}], **extra})))
    assert asyncio.run(investigator._fetch_exa_exact_page(object(), URL))["ok"]


def test_linkedin_timeout_cannot_admit_contents_fallback(monkeypatch):
    monkeypatch.setattr(__import__(__name__, fromlist=["URL"]), "URL",
                        "https://www.linkedin.com/company/acme/")
    result, calls = _review(monkeypatch, timeout=True)
    assert calls == [("primary", "https://www.linkedin.com/company/acme/")]
    assert "usage" not in result  # Existing timeout result stays unchanged.
