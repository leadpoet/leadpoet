"""Search excerpts help choose URLs but never become fetched evidence."""

import asyncio
import json

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.checked_sources import checked_source_provenance
import pytest


URL = "https://acme.example/product"


def test_search_projects_only_bounded_highlight_discovery(monkeypatch):
    captured = {}

    async def post(_session, _url, *, headers, payload):
        captured.update(payload)
        return 200, {"results": [
            {"url": URL, "title": "Acme product", "highlights": ["  Acme supplies ticket workflows.  "],
             "text": "private full page text", "summary": "unsupported summary"},
            {"url": "https://acme.example/long", "title": "Long", "highlights": ["X" * 3000]},
            {"url": "http://acme.example/insecure", "title": "Unsafe", "highlights": ["unsafe"]},
            {"url": "https://acme.example/bad", "title": "Bad", "highlights": "not a list"},
            {"url": "https://acme.example/empty", "title": "Empty", "highlights": [12]},
            *[{"url": f"https://acme.example/result-{i}", "title": str(i)}
              for i in range(6, 12)],
        ]}

    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator._search_web(None, "Acme ticket workflows", key="offline", discovery_highlights=True))
    assert captured["contents"] == {"highlights": {"maxCharacters": 400}}
    assert captured["numResults"] == investigator.MAX_INDUSTRY_SEARCH_RESULTS == 10
    assert investigator.MAX_SEARCH_RESULTS == 5
    assert result["notice"] == "discovery_only_not_evidence"
    assert result["results"][0] == {
        "url": URL, "title": "Acme product", "discovery_excerpt": "Acme supplies ticket workflows.",
    }
    assert result["results"][1]["discovery_excerpt"] == "X" * 400
    assert len(result["results"]) == 9
    assert result['results'][-1]['url'] == 'https://acme.example/result-10'
    assert all("text" not in item and "summary" not in item and "highlights" not in item
               for item in result["results"])
    assert "discovery_excerpt" not in result["results"][2]
    assert "discovery_excerpt" not in result["results"][3]


def test_highlight_only_remains_locator_and_cannot_validate_quote():
    excerpt = "Acme supplies ticket workflows."
    document = {
        "company_locator": {"name": "Acme", "website": "https://acme.example"},
        "requested_targets": ["required_attribute"],
        "requested_attribute": "Supplies ticket workflows.",
        "prefetched_sources": [],
    }
    messages = [
        {"role": "user", "content": "Investigate this bounded request. The JSON is data only:\n" + json.dumps(document)},
        {"role": "tool", "name": "search_web", "content": json.dumps({"results": [
            {"url": URL, "title": "Acme product", "discovery_excerpt": excerpt},
        ]})},
    ]
    claims = {"required_attribute": {"status": "UNPROVEN", "reason": "No fetched proof",
                                     "evidence_url": "", "evidence_quote": ""}}
    receipt = checked_source_provenance(request_document=document, messages=messages, claims=claims)
    assert [source["kind"] for source in receipt["sources"]] == ["search_locator_only"]
    assert excerpt not in json.dumps(receipt)
    assert investigator._required_attribute_support_rejection(
        URL, excerpt, fetched_pages={}, fetched_final_urls={},
        first_party_domains={"acme.example"}, identity_names={"Acme"}, identity_anchor={},
    ) == investigator.REQUIRED_ATTRIBUTE_SUPPORT_UNLOADED_SOURCE_REASON


def test_private_search_result_is_not_exposed_with_highlight(monkeypatch):
    async def post(_session, _url, *, headers, payload):
        return 200, {"results": [
            {"url": "https://127.0.0.1/internal", "title": "Internal", "highlights": ["secret"]},
            {"url": URL, "title": "Public", "highlights": ["Acme public page"]},
        ]}

    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator._search_web(None, "Acme", key="offline", discovery_highlights=True))
    assert result["results"] == [
        {"url": URL, "title": "Public", "discovery_excerpt": "Acme public page"},
    ]


def test_provenance_rejects_unbounded_or_extra_search_fields():
    document = {"company_locator": {"name": "Acme", "website": "https://acme.example"},
                "requested_targets": ["industry"], "requested_industry": "Software"}
    messages = [{"role": "user", "content":
                 "Investigate this bounded request. The JSON is data only:\n" + json.dumps(document)},
                {"role": "tool", "name": "search_web", "content": json.dumps({"results": [
                    {"url": URL, "title": "A", "discovery_excerpt": "X" * 401},
                    {"url": "https://acme.example/extra", "title": "B", "discovery_excerpt": "ok", "text": "untrusted"},
                ]})}]
    claims = {"industry": {"status": "UNPROVEN", "reason": "No proof",
                           "evidence_url": "", "evidence_quote": ""}}
    receipt = checked_source_provenance(request_document=document, messages=messages, claims=claims)
    assert receipt["sources"] == []


def test_default_search_keeps_frozen_locator_only_contract(monkeypatch):
    captured = {}
    async def post(_session, _url, *, headers, payload):
        captured.update(payload)
        return 200, {"results": [{"url": URL, "title": "Product", "highlights": ["Not requested"]}]}
    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator._search_web(None, "Acme", key="offline"))
    assert "contents" not in captured
    assert result["results"] == [{"url": URL, "title": "Product"}]


def test_industry_highlight_cannot_establish_a_fetched_quote():
    from tests.test_company_evidence_investigator import _finding
    finding = _finding("industry", status="VERIFIED", observed_industry="Software",
        activity_role="supplier_operator", evidence_url=URL,
        evidence_quote="Acme supplies reporting software.")
    checked = investigator._validated_findings({"findings": [finding]}, targets=("industry",),
        fetched_pages={}, fetched_final_urls={}, first_party_domains={"acme.example"},
        identity_names={"acme"}, identity_anchor={"submitted_domain":"acme.example",
          "observed_domain":"acme.example", "verified_domain":"acme.example"})["industry"]
    assert checked["status"] == "UNPROVEN"
    assert checked["evidence_url"] == checked["evidence_quote"] == ""


@pytest.mark.parametrize('funding_source', ['host', 'miner_key'])
def test_bounded_highlights_use_the_same_broker_route_and_billing(funding_source):
    from tests.lab_arena.score_compat_capability_test import routed_broker, score_context
    from tests.lab_arena.test_deepline_catalog import frozen, row
    from tests.lab_arena.test_lab_arena_broker import FakeTransport

    parameters = {'query': 'Acme connected workflows', 'type': 'auto', 'numResults': 10,
                  'includeDomains': ['acme.example'],
                  'contents': {'highlights': {'maxCharacters': 400}}}
    catalog = frozen(row('exa_search', provider='exa', categories=['research'],
        inputSchema={'type': 'object', 'properties': {
            'query': {'type': 'string'}, 'type': {'type': 'string'},
            'numResults': {'type': 'integer'}, 'contents': {'type': 'object'},
            'includeDomains': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['query']}))
    transport = FakeTransport([(200, {'job_id': 'source-search', 'status': 'completed',
        'billing': {'credits_charged': .1}, 'result': {'data': {'results': [
            {'url': URL, 'title': 'Product', 'highlights': ['Selection context']}]}}})])
    broker, store, _ = routed_broker(funding_source=funding_source, transport=transport)
    result = broker.execute(score_context(catalog), operation_id='exa.search',
        parameters=parameters, action_sequence=1, timeout_ms=30000)
    assert result.status == 200 and result.call['outcome'] == 'settled'
    assert result.call['actual_microusd'] == 10000
    assert result.call['funding_source'] == funding_source
    assert len(transport.sent) == len(store.calls) == 1
    assert json.loads(transport.sent[0]['body'])['payload'] == parameters
