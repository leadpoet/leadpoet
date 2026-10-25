"""Already admitted product pages must reach the model without losing their tails."""

import asyncio
import copy
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as investigator
from tests.product_coverage_fixtures import native_product_post


CRITERION = "Provides mainframe modernization services to large enterprises"
BASE = "https://acmeproducts.example"
TAIL = "Acme Products provides mainframe modernization services to large enterprises."
SOURCE_PREFIX = "Untrusted fetched source data."


def _run(monkeypatch, pages, *, submitted_status="UNPROVEN"):
    requests = []

    async def model(_session, _url, *, headers, payload):
        del headers
        normalized = validate_operation_request("openrouter.chat", payload)
        assert normalized["messages"] == payload["messages"]
        requests.append(payload)
        last_url = next(reversed(pages))
        finding = {
            "target": "industry", "status": submitted_status,
            "observed_industry": "Technology services" if submitted_status == "VERIFIED" else "",
            "activity_role": "supplier_operator" if submitted_status == "VERIFIED" else "unresolved",
            "evidence_url": last_url if submitted_status == "VERIFIED" else "",
            "evidence_quote": TAIL if submitted_status == "VERIFIED" else "",
            "reason": "Judge from the loaded company pages.",
        }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": [finding]})},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "MAX_SEARCH_CALLS", 0)
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme Products", "website": BASE},
        targets=("industry",), requested_industry="Technology services",
        requested_product_service=CRITERION,
        prior_observations={
            "submitted_source_urls": list(pages),
            "observed_company_name": "Acme Products",
            "observed_company_website": BASE,
        },
        prefetched_pages=pages,
    ))
    return result, requests


def test_product_review_keeps_decisive_tails_from_three_loaded_pages(monkeypatch):
    pages = {
        f"{BASE}/source-{index}": {
            "final_url": f"{BASE}/source-{index}",
            "text": "Background text. " + "x" * 15_500 + (TAIL if index == 2 else ""),
        }
        for index in range(3)
    }
    _result, requests = _run(monkeypatch, pages)
    assert requests
    messages = requests[0]["messages"]
    metadata = next(message["content"] for message in messages
                    if message.get("role") == "user" and message["content"].startswith("Investigate this bounded request."))
    assert "prefetched_sources" not in json.loads(metadata.split("\n", 1)[1])
    sources = [message["content"] for message in messages
               if message.get("role") == "user" and message["content"].startswith(SOURCE_PREFIX)]
    assert len(sources) == 3
    for index, content in enumerate(sources):
        url = f"{BASE}/source-{index}"
        assert f"Requested URL: {url}\nFetched final URL: {url}\n" in content
        assert pages[url]["text"] in content
        assert len(content) < 32_000
    assert TAIL in sources[2]


def test_partition_keeps_non_ascii_max_page_and_broker_request_bounds(monkeypatch):
    pages = {
        f"{BASE}/source-{index}": {
            "final_url": f"{BASE}/source-{index}",
            "text": "界" * investigator.MAX_PAGE_CHARACTERS,
        }
        for index in range(investigator.MAX_PREFETCHED_PAGES)
    }
    _result, requests = _run(monkeypatch, pages)
    assert requests
    sources = [message["content"] for message in requests[0]["messages"]
               if message.get("role") == "user" and message["content"].startswith(SOURCE_PREFIX)]
    assert len(sources) == investigator.MAX_PREFETCHED_PAGES
    assert all("界" * investigator.MAX_PAGE_CHARACTERS in content for content in sources)


def test_eight_max_pages_fit_real_broker_full_history(monkeypatch):
    page_text = "🧪" * investigator.MAX_PAGE_CHARACTERS
    pages = {
        f"{BASE}/source-{index}": {
            "final_url": f"{BASE}/source-{index}", "text": page_text,
        }
        for index in range(investigator.MAX_PREFETCHED_PAGES)
    }
    _result, requests = _run(monkeypatch, pages)
    payload = copy.deepcopy(requests[0])
    for index in range(
        investigator.MAX_RETURNED_FETCHED_PAGES - investigator.MAX_PREFETCHED_PAGES
    ):
        url = f"{BASE}/fresh-{index}"
        call_id = f"fetch-{index}"
        payload["messages"].extend([
            {"role": "assistant", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "fetch_page", "arguments": json.dumps({"url": url})},
            }]},
            {"role": "tool", "tool_call_id": call_id, "name": "fetch_page",
             "content": json.dumps({"ok": True, "url": url, "final_url": url,
                                    "text": page_text}, ensure_ascii=False)},
        ])
    normalized = validate_operation_request("openrouter.chat", payload)
    assert normalized["messages"] == payload["messages"]
    assert len(payload["messages"]) <= 128
    assert len(pages) + investigator.MAX_PRODUCT_FETCH_CALLS == investigator.MAX_RETURNED_FETCHED_PAGES


@pytest.mark.parametrize("final_url,admitted", [
    (f"{BASE}/source", True),
    (f"{BASE}/renamed", True),
    ("http://unbound.example/source", False),
])
def test_unsafe_final_url_is_not_admitted_as_loaded_source(
    monkeypatch, final_url, admitted,
):
    url = f"{BASE}/source"
    pages = {url: {"final_url": final_url, "text": TAIL}}
    result, requests = _run(monkeypatch, pages)
    sources = [message["content"] for message in requests[0]["messages"]
               if message.get("role") == "user" and message["content"].startswith(SOURCE_PREFIX)]
    assert len(sources) == int(admitted)
    if admitted:
        assert f"Fetched final URL: {final_url}\n" in sources[0]
    assert result["claims"]["industry"]["status"] == "UNPROVEN"


def test_same_company_redirect_supports_bound_product_proof(monkeypatch):
    requested_url = f"{BASE}/source"
    final_url = f"{BASE}/renamed"
    result, requests = _run(
        monkeypatch,
        {requested_url: {"final_url": final_url, "text": TAIL}},
        submitted_status="VERIFIED",
    )
    sources = [message["content"] for message in requests[0]["messages"]
               if message.get("role") == "user" and message["content"].startswith(SOURCE_PREFIX)]
    assert len(sources) == 1
    assert f"Requested URL: {requested_url}\nFetched final URL: {final_url}\n" in sources[0]
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_cross_domain_final_url_is_reported_as_untrusted_data(monkeypatch):
    url = f"{BASE}/source"
    final_url = "https://unbound.example/source"
    _result, requests = _run(monkeypatch, {url: {"final_url": final_url, "text": TAIL}})
    sources = [message["content"] for message in requests[0]["messages"]
               if message.get("role") == "user" and message["content"].startswith(SOURCE_PREFIX)]
    assert len(sources) == 1 and f"Fetched final URL: {final_url}\n" in sources[0]


def test_control_character_page_fails_closed_before_investigator_judgment(monkeypatch):
    url = f"{BASE}/source"
    result, requests = _run(
        monkeypatch,
        {url: {"final_url": url, "text": TAIL + "\x00"}},
    )
    assert not requests
    assert result["failure_reason"] == investigator.MALFORMED_RESPONSE_FAILURE_REASON
    assert result["claims"] == {}
