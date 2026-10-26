"""Bounded discovery can expose late, relevant first-party locators."""

import asyncio
import json

import pytest

from lab_arena import operations as arena_operations
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification


def test_late_sector_solution_survives_generic_homepage_menu(monkeypatch):
    # A large product menu precedes the sector solution. The page is a
    # locator source only; it never establishes product coverage itself.
    html = "".join(
        f'<a href="/products/item-{index}">Product item {index}</a>'
        for index in range(45)
    ) + '<a href="/solutions/institutions">Institutions and platforms</a>'
    locators = company_verification._homepage_navigation_locators(
        html, final_url="https://acme.example/", verified_domain="acme.example",
    )
    assert len(locators) == 46
    assert company_verification.MAX_HOMEPAGE_NAVIGATION_LOCATORS == 50
    assert locators[45] == {
        "url": "https://acme.example/solutions/institutions",
        "label": "Institutions and platforms",
    }
    assert sum(len(row["url"]) + len(row["label"]) for row in locators) <= (
        company_verification.MAX_HOMEPAGE_NAVIGATION_TOTAL_CHARACTERS
    )
    monkeypatch.setattr(company_verification, "MAX_HOMEPAGE_NAVIGATION_LOCATORS", 40)
    previous = company_verification._homepage_navigation_locators(
        html, final_url="https://acme.example/", verified_domain="acme.example",
    )
    assert len(previous) == 40
    assert not any(row["url"].endswith("/institutions") for row in previous)


def test_product_search_returns_tenth_safe_locator_without_search_authority(monkeypatch):
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        return 200, {"results": [
            {"url": f"https://acme.example/result/{index}",
             "title": f"Result {index}", "highlights": ["X" * 700],
             "text": "not fetched evidence"}
            for index in range(12)
        ]}

    monkeypatch.setattr(investigator, "_post_json", post)
    ordinary = asyncio.run(investigator._search_web(None, "Acme", key="offline"))
    product = asyncio.run(investigator._search_web(
        None, "Acme", key="offline", max_results=10,
    ))
    assert requests[0]["numResults"] == 5
    assert requests[1]["numResults"] == 10
    assert len(ordinary["results"]) == 5
    assert len(product["results"]) == 10
    assert product["results"][9]["url"].endswith("/9")
    assert all(len(row["discovery_excerpt"]) == 400 for row in product["results"])
    assert all("text" not in row for row in product["results"])
    assert product["notice"] == "discovery_only_not_evidence"


@pytest.mark.parametrize("criterion,expected_count", [
    ("Payment APIs used by institutions", 10), ("", 5),
])
def test_investigator_uses_wider_search_only_for_product_review(
    monkeypatch, criterion, expected_count,
):
    search_requests = []
    model_turns = 0

    async def post(_session, url, *, headers, payload):
        nonlocal model_turns
        if url.endswith("/search"):
            search_requests.append(payload["numResults"])
            return 200, {"results": []}
        model_turns += 1
        if model_turns > 1:
            raise RuntimeError("end offline trace")
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "search-1", "type": "function", "function": {
                "name": "search_web", "arguments": json.dumps({
                    "query": "Acme institutional payments",
                }),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline")
    monkeypatch.setenv("EXA_API_KEY", "offline")
    monkeypatch.setattr(investigator, "_post_json", post)
    asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Payments",
        requested_product_service=criterion,
    ))
    assert search_requests == [expected_count]


def test_tenth_product_locator_can_be_fetched_with_stage_search_preserved(monkeypatch):
    searches, fetches, model_turns = [], [], []
    last_url = "https://acme.example/sector/9"

    async def model(_session, _url, *, headers, payload):
        model_turns.append(payload)
        turn = len(model_turns)
        if turn > 2:
            raise RuntimeError("end offline trace")
        name, arguments = (
            ("search_web", {"query": "Acme institutional payment users"})
            if turn == 1 else ("fetch_page", {"url": last_url})
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(turn), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def search(_session, query, *, key, max_results=5):
        searches.append((query, max_results))
        return {"results": (
            [{"url": "https://acme.example/stage"}]
            if len(searches) == 1 else
            [{"url": f"https://acme.example/sector/{index}"} for index in range(10)]
        )}

    async def fetch(_session, url):
        fetches.append(url)
        return {"ok": True, "url": url, "final_url": url,
                "text": "Acme serves institutional payment customers."}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline")
    monkeypatch.setenv("EXA_API_KEY", "offline")
    monkeypatch.setattr(investigator, "_post_json", model)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage", "industry"), requested_stage="Series C+",
        requested_industry="Payments",
        requested_product_service="Payment APIs for institutional customers",
        positive_semantic_review=True,
    ))

    assert [limit for _query, limit in searches] == [5, 10]
    assert model_turns[0]["tool_choice"]["function"]["name"] == "search_web"
    assert model_turns[1]["tool_choice"]["function"]["name"] == "fetch_page"
    assert fetches == [last_url]
    assert result["usage"]["early_product_research"] == {
        "search": "completed", "fetch": "loaded", "safe_result_count": 10,
    }


def test_ten_worst_bounded_locators_fit_one_broker_tool_message():
    locators = [
        {"url": "https://acme.example/" + "x" * 1979,
         "title": "T" * 300, "discovery_excerpt": "E" * 400}
        for _ in range(10)
    ]
    content = investigator._bounded_message_json({
        "results": locators, "notice": "discovery_only_not_evidence",
    })
    assert len(content) < arena_operations.OPENROUTER_MAX_CONTENT_CHARS
    messages = [
        {"role": "system", "content": investigator._SYSTEM_PROMPT},
        {"role": "system", "content": investigator._PRODUCT_JOINT_EVIDENCE_SUFFIX},
        {"role": "user", "content": "Investigate a bounded request."},
    ]
    for index in range(investigator.MAX_RETURNED_FETCHED_PAGES):
        url = f"https://acme.example/page/{index}"
        messages.append(investigator._untrusted_loaded_page_message(
            url, url, "é" * investigator.MAX_PAGE_CHARACTERS,
        ))
    for index in range(3):
        call_id = f"search-{index}"
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                "function": {"name": "search_web", "arguments": json.dumps({"query": "Acme"})}}]},
            {"role": "tool", "name": "search_web", "tool_call_id": call_id,
             "content": content},
        ])
    payload = {
        "model": investigator.POSITIVE_SEMANTIC_REVIEW_MODEL,
        "messages": messages,
        "tools": [{"type": "function", "function": {
            key: value for key, value in tool.items() if key != "type"
        }} for tool in investigator._tools(("industry",))],
        "tool_choice": "required", "parallel_tool_calls": False,
        "max_tokens": investigator.REASONING_MAX_TOKENS,
    }
    arena_operations.validate_operation_request("openrouter.chat", payload)
    outbound = arena_operations.build_outbound_request("openrouter.chat", payload)
    assert len(outbound.body) < 1_000_000


def test_product_guidance_targets_missing_customer_use_without_proof_shortcut():
    prompt = " ".join(investigator._PRODUCT_COVERAGE_INSTRUCTIONS.split())
    assert "customer or user type" in prompt
    assert "first-party sector solutions or customer stories" in prompt
    assert "Fetch" in prompt
    assert "Each citation must independently pass" in prompt


def test_concurrent_product_search_does_not_expand_ordinary_default(monkeypatch):
    requests = []

    async def run():
        product_started = asyncio.Event()
        ordinary_finished = asyncio.Event()

        async def post(_session, _url, *, headers, payload):
            requests.append((payload["query"], payload["numResults"]))
            if payload["query"] == "product":
                product_started.set()
                await ordinary_finished.wait()
            else:
                ordinary_finished.set()
            return 200, {"results": [
                {"url": f"https://acme.example/result/{index}"}
                for index in range(12)
            ]}

        monkeypatch.setattr(investigator, "_post_json", post)
        product = asyncio.create_task(investigator._search_web(
            None, "product", key="offline", max_results=10,
        ))
        await product_started.wait()
        ordinary = await investigator._search_web(None, "ordinary", key="offline")
        return await product, ordinary

    product, ordinary = asyncio.run(run())
    assert requests == [("product", 10), ("ordinary", 5)]
    assert len(product["results"]) == 10
    assert len(ordinary["results"]) == 5


@pytest.mark.parametrize("max_results", [0, 6, 11, None])
def test_search_refuses_unbounded_result_counts_without_provider_call(
    monkeypatch, max_results,
):
    async def refuse(*_args, **_kwargs):
        pytest.fail("invalid result bound reached the provider")

    monkeypatch.setattr(investigator, "_post_json", refuse)
    with pytest.raises(ValueError, match="invalid bounded search result count"):
        asyncio.run(investigator._search_web(
            None, "Acme", key="offline", max_results=max_results,
        ))
