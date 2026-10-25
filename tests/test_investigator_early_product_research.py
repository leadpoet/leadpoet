"""Early product research within the existing stage and review limits."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.product_coverage_fixtures import native_product_post, trusted_plan


STAGE_URL = "https://acme.example/news/series-c"
PRODUCT_URL = "https://acme.example/products/cloud-platform"
STAGE_QUOTE = "Acme completed a $120 million Series C funding round in 2025."
PRODUCT_QUOTE = "Acme supplies a managed cloud platform to enterprise customers."
PRODUCT_CRITERION = "Supplies a managed cloud platform to enterprise customers"


def _finding(target, *, status="VERIFIED", url="", quote=""):
    return {
        "target": target, "status": status,
        "observed_value": "Series C+" if target == "stage" else "Managed cloud platform",
        "observed_country": "", "observed_state": "",
        "observed_industry": "Software" if target == "industry" else "",
        "observed_subindustry": "Cloud software" if target == "industry" else "",
        "activity_role": "supplier_operator" if target == "industry" else "unresolved",
        "evidence_url": url, "evidence_quote": quote, "supporting_evidence": [],
        "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
        "shared_linkedin_slug": "", "reason": "Source states the fact.",
    }


async def _investigate(*, product=PRODUCT_CRITERION, targets=("stage", "industry")):
    return await investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=targets, requested_stage="Series C+",
        requested_industry="Software", requested_subindustry="Cloud software",
        requested_product_service=product, positive_semantic_review=True,
        verified_homepage_identity={
            "normalized_name": "acme", "registrable_dns_domain": "acme.example",
        },
    )


@pytest.mark.parametrize("fetch_cap", [2, 4])
def test_combined_stage_product_searches_and_fetches_product_before_stage(
    monkeypatch, fetch_cap,
):
    actions = [
        ("search_web", {"query": "Acme managed cloud platform enterprise customers"}),
        ("fetch_page", {"url": PRODUCT_URL}),
        ("fetch_page", {"url": STAGE_URL}),
        ("submit_findings", {"findings": [
            _finding("stage", url=STAGE_URL, quote=STAGE_QUOTE),
            _finding("industry", url=PRODUCT_URL, quote=PRODUCT_QUOTE),
        ]}),
    ]
    requests = []
    searches = []
    fetches = []

    async def model(_session, _url, *, headers, payload):
        del headers
        index = len(requests)
        required = payload["tool_choice"]
        if index < 2:
            assert required["function"]["name"] == actions[index][0]
        elif index == 2:
            assert required == "required"
        requests.append(payload)
        name, arguments = actions[index]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(index), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": STAGE_URL if len(searches) == 1 else PRODUCT_URL}]}

    async def fetch(_session, url):
        fetches.append(url)
        return {"ok": True, "url": url, "final_url": url,
                "text": STAGE_QUOTE if url == STAGE_URL else PRODUCT_QUOTE}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "MAX_FETCH_CALLS", fetch_cap)
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(_investigate())

    assert searches == [searches[0], actions[0][1]["query"]]
    assert "latest funding round" in searches[0]
    assert fetches == [PRODUCT_URL, STAGE_URL]
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["search_calls"] == 2
    assert result["usage"]["fetch_calls"] == 2
    assert result["usage"]["early_product_research"]["search"] == "completed"
    assert result["usage"]["early_product_research"]["fetch"] == "loaded"


@pytest.mark.parametrize("results", [[], [
    {"url": "http://acme.example/insecure"},
    {"url": "https://127.0.0.1/private"},
]])
def test_empty_or_unsafe_product_results_do_not_force_fetch(monkeypatch, results):
    requests = []
    searches = []
    fetches = []

    async def model(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            assert payload["tool_choice"]["function"]["name"] == "search_web"
            return 200, {"choices": [{"message": {"tool_calls": [{
                "id": "product-search", "type": "function",
                "function": {"name": "search_web", "arguments": json.dumps({
                    "query": "Acme managed cloud platform enterprise customers",
                })},
            }]}}]}
        assert payload["tool_choice"] == "required"
        raise RuntimeError("end offline trace")

    async def search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": STAGE_URL}] if len(searches) == 1 else results}

    async def fetch(_session, url):
        fetches.append(url)
        return {"ok": True, "url": url, "final_url": url, "text": PRODUCT_QUOTE}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(_investigate())

    assert len(searches) == 2
    assert fetches == []
    assert result["usage"]["early_product_research"]["fetch"] == "no_safe_result"


def test_product_search_provider_error_does_not_force_fetch(monkeypatch):
    searches = []

    async def model(_session, _url, *, headers, payload):
        del headers
        assert payload["tool_choice"]["function"]["name"] == "search_web"
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "product-search", "type": "function",
            "function": {"name": "search_web", "arguments": json.dumps({
                "query": "Acme managed cloud platform enterprise customers",
            })},
        }]}}]}

    async def search(_session, query, *, key):
        del key
        searches.append(query)
        if len(searches) == 2:
            raise RuntimeError("offline search failure")
        return {"results": [{"url": STAGE_URL}]}

    async def fetch(_session, url):
        raise AssertionError(f"unexpected fetch: {url}")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(_investigate())

    assert len(searches) == 2
    assert result["failure_reason"] == investigator.PROVIDER_ERROR_FAILURE_REASON
    assert result["usage"]["early_product_research"]["search"] == "provider_error"


@pytest.mark.parametrize("product,targets", [
    ("", ("stage", "industry")),
    (PRODUCT_CRITERION, ("stage",)),
])
def test_empty_product_or_stage_only_keeps_unforced_first_turn(
    monkeypatch, product, targets,
):
    requests = []

    async def model(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        assert payload["tool_choice"] == "required"
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "negative", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [_finding(target, status="UNPROVEN") for target in targets],
            })},
        }]}}]}

    async def search(_session, query, *, key):
        del key
        assert "latest funding round" in query
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(_investigate(product=product, targets=targets))

    assert len(requests) == 1
    assert "early_product_research" not in result["usage"]


def test_judgment_reserve_suppresses_early_product_research(monkeypatch):
    async def model(_session, _url, *, headers, payload):
        del headers
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        raise RuntimeError("end offline trace")

    async def search(_session, query, *, key):
        del key
        assert "latest funding round" in query
        return {"results": [{"url": STAGE_URL}]}

    def submit_only(_self, **_kwargs):
        return SimpleNamespace(
            mode="submit_only", serialization_expired=False,
            restrict_tools=True,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator.ReviewerAdmission, "next_turn", submit_only)

    result = asyncio.run(_investigate())

    assert result["usage"]["early_product_research"]["search"] == "judgment_time_reserved"


def test_one_remaining_fetch_is_reserved_for_stage(monkeypatch):
    actions = [
        ("fetch_page", {"url": STAGE_URL}),
        ("fetch_page", {"url": PRODUCT_URL}),
        ("submit_findings", {"findings": [
            _finding("stage", url=STAGE_URL, quote=STAGE_QUOTE),
            _finding("industry", status="UNPROVEN"),
        ]}),
    ]
    requests = []
    fetches = []
    searches = []

    async def model(_session, _url, *, headers, payload):
        del headers
        index = len(requests)
        requests.append(payload)
        if index == 0:
            assert payload["tool_choice"] == "required"
        name, args = actions[index]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(index), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    async def search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": STAGE_URL}]}

    async def fetch(_session, url):
        fetches.append(url)
        return {"ok": True, "url": url, "final_url": url, "text": STAGE_QUOTE}

    monkeypatch.setattr(investigator, "MAX_FETCH_CALLS", 1)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(_investigate())

    assert searches and len(searches) == 1
    assert fetches == [STAGE_URL]
    assert result["usage"]["fetch_calls"] == 1
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["usage"]["early_product_research"]["search"] == "stage_fetch_reserved"
    exhausted = json.loads(requests[2]["messages"][-1]["content"])
    assert exhausted == {"ok": False, "error": "fetch_budget_exhausted"}


def test_forced_fetch_rejects_url_absent_from_product_results(monkeypatch):
    requests = []
    fetched = []
    searches = []

    async def model(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 3:
            assert payload["tool_choice"] == "required"
            raise RuntimeError("end offline trace")
        name, args = (
            ("search_web", {"query": "Acme managed cloud platform enterprise customers"})
            if len(requests) == 1 else
            ("fetch_page", {"url": "https://acme.example/unreturned"})
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    async def search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": STAGE_URL if len(searches) == 1 else PRODUCT_URL}]}

    async def fetch(_session, url):
        fetched.append(url)
        raise AssertionError("unreturned URL must not be fetched")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(_investigate())

    assert fetched == []
    assert result["usage"]["early_product_research"]["fetch"] == "rejected_url"
    rejected = json.loads(requests[2]["messages"][-1]["content"])
    assert rejected["error"] == "product_search_result_url_required"


def test_product_allocation_keeps_three_search_total_cap(monkeypatch):
    actions = [
        ("search_web", {"query": "Acme managed cloud platform enterprise customers"}),
        ("fetch_page", {"url": PRODUCT_URL}),
        ("search_web", {"query": "Acme latest Series C status"}),
        ("search_web", {"query": "Acme another Series C status"}),
        ("submit_findings", {"findings": [
            _finding("stage", status="UNPROVEN"),
            _finding("industry", url=PRODUCT_URL, quote=PRODUCT_QUOTE),
        ]}),
    ]
    requests = []
    searches = []

    async def model(_session, _url, *, headers, payload):
        del headers
        index = len(requests)
        requests.append(payload)
        name, args = actions[index]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(index), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    async def search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": STAGE_URL if len(searches) == 1 else PRODUCT_URL}]}

    async def fetch(_session, url):
        return {"ok": True, "url": url, "final_url": url, "text": PRODUCT_QUOTE}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(_investigate())

    assert len(searches) == 3  # required stage, product, one later model search
    assert result["usage"]["search_calls"] == 3
    exhausted = json.loads(requests[4]["messages"][-1]["content"])
    assert exhausted == {"ok": False, "error": "search_budget_exhausted"}


@pytest.mark.parametrize("leaf_status,page_text", [
    ("UNPROVEN", PRODUCT_QUOTE),
    ("VERIFIED", "Acme offers an unrelated company overview."),
])
def test_early_research_does_not_relax_leaf_or_exact_source_checks(
    leaf_status, page_text,
):
    plan = trusted_plan(PRODUCT_CRITERION)
    finding = _finding("industry", url=PRODUCT_URL, quote=PRODUCT_QUOTE)
    finding["product_coverage"] = json.dumps({
        "plan_id": plan.plan_id,
        "claims": [{
            "id": "p0", "status": leaf_status,
            "activity_role": "supplier_operator",
            "evidence": [] if leaf_status == "UNPROVEN" else [{
                "url": PRODUCT_URL, "quote": PRODUCT_QUOTE,
            }],
        }],
    })

    claims = investigator._validated_findings(
        {"findings": [finding]}, targets=("industry",),
        fetched_pages={PRODUCT_URL: page_text},
        fetched_final_urls={PRODUCT_URL: PRODUCT_URL},
        first_party_domains={"acme.example"}, identity_names={"acme"},
        identity_anchor={"submitted_name": "Acme", "submitted_domain": "acme.example"},
        requested_product_service=PRODUCT_CRITERION, product_plan=plan,
    )

    assert claims is not None
    assert claims["industry"]["status"] == "UNPROVEN"
