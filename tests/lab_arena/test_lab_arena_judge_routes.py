"""Layer A of validator scoring: the judge's provider calls through the table.

The Research Lab evaluator calls Exa, Scrapingdog, and OpenRouter with its own
``httpx`` client and placeholder credentials. Inside the scorer sandbox every
call crosses the shim, so this module proves the exact requests the judge
makes match one operation each, reach Deepline on Exa's behalf, come back in
Exa's shape, and that the trusted-scorer mode strips the placeholder
credentials while the normal mode still refuses them.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode

import pytest

from lab_arena import broker as br
from lab_arena import contracts
from lab_arena import operations as ops
from lab_arena import shim

EVALUATOR_SEARCH = {"query": "Acme series B funding announcement", "type": "auto", "numResults": 5, "contents": {"highlights": {"query": "raised a Series B", "maxCharacters": 400}}}
EVALUATOR_CONTENTS = {"ids": ["https://example.com/news"], "text": {"maxCharacters": 12000}, "maxAgeHours": 0}


def test_exa_shaped_judge_requests_route_to_deepline_and_return_exa_shaped_replies():
    operation_id, parameters = ops.match_request("POST", "https://api.exa.ai/search", json.dumps(EVALUATOR_SEARCH).encode(), {"Content-Type": "application/json"})
    assert operation_id == "exa.search" and parameters == EVALUATOR_SEARCH
    outbound = ops.build_outbound_request(operation_id, parameters)
    assert outbound.url == "https://code.deepline.com/api/v2/integrations/exa_search/execute"
    assert outbound.target.host == "code.deepline.com" and outbound.target.path == "/api/v2/integrations/exa_search/execute"
    assert json.loads(outbound.body) == {"provider": "exa", "operation": "exa_search", "payload": EVALUATOR_SEARCH}
    assert dict(outbound.headers) == {"x-deepline-execute-response-intent": "raw"} and outbound.credential.scheme == "Bearer"
    operation_id, parameters = ops.match_request("POST", "https://api.exa.ai/contents", json.dumps(EVALUATOR_CONTENTS).encode(), {"Content-Type": "application/json"})
    assert operation_id == "exa.contents" and parameters == EVALUATOR_CONTENTS
    assert json.loads(ops.build_outbound_request(operation_id, parameters).body)["operation"] == "exa_contents"
    # The Deepline envelope is unwrapped to the raw Exa reply, person entities dropped, cost still readable by the broker.
    envelope = {"job_id": "j", "status": "completed", "result": {"data": {"requestId": "r", "results": [{"id": "u", "url": "https://example.com/news", "title": "t", "text": "body", "highlights": ["raised a Series B"], "publishedDate": "2026-01-01", "entities": [{"type": "person", "properties": {"name": "Jane Roe"}}, {"type": "company", "properties": {"name": "Acme"}}]}], "searchTime": 3.0}}, "billing": {"credits_charged": 0.1, "cost_usd": 0.01}}
    status, headers, body = ops.sanitize_response("exa.search", 200, {"content-type": "application/json"}, json.dumps(envelope).encode())
    reply = json.loads(body)
    assert status == 200 and set(reply) == {"requestId", "results", "searchTime"} and reply["results"][0]["highlights"] == ["raised a Series B"]
    assert reply["results"][0]["entities"] == [{"type": "company", "properties": {"name": "Acme"}}] and b"Jane Roe" not in body
    assert headers["content-length"] == str(len(body))
    assert br.deepline_cost_microusd(json.dumps(envelope).encode()) == 10_000
    # A failed envelope passes through untouched so the judge sees the provider's own error.
    failed = {"job_id": "j", "status": "failed", "error": {"message": "tool failed"}}
    assert json.loads(ops.sanitize_response("exa.search", 200, {}, json.dumps(failed).encode())[2]) == failed


@pytest.mark.parametrize("url, operation_id, expected", [
    ("https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com%2Fjobs&dynamic=true&wait=8000&premium=true&stealth_mode=true", "scrapingdog.scrape", {"url": "https://example.com/jobs", "dynamic": True, "wait": 8000, "premium": True, "stealth_mode": True}),
    ("https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com%2F&dynamic=false", "scrapingdog.scrape", {"url": "https://example.com/", "dynamic": False}),
    ("https://api.scrapingdog.com/x/post?tweetId=1234567890", "scrapingdog.x_post", {"tweetId": "1234567890"}),
    ("https://api.scrapingdog.com/x/profile?profileId=acme", "scrapingdog.x_profile", {"profileId": "acme"}),
    ("https://api.scrapingdog.com/profile?type=company&id=acme-inc", "scrapingdog.profile", {"type": "company", "id": "acme-inc"}),
    ("https://api.scrapingdog.com/profile/post?id=7123456789", "scrapingdog.profile_post", {"id": "7123456789"}),
    ("https://api.scrapingdog.com/linkedinjobs?job_id=4012345678", "scrapingdog.linkedinjobs", {"job_id": "4012345678"}),
    ("https://api.scrapingdog.com/jobs?job_id=abc123", "scrapingdog.jobs", {"job_id": "abc123"}),
    ("https://api.scrapingdog.com/indeed?url=https%3A%2F%2Fwww.indeed.com%2Fviewjob%3Fjk%3D1", "scrapingdog.indeed", {"url": "https://www.indeed.com/viewjob?jk=1"}),
    ("https://api.scrapingdog.com/instagram/profile?username=acme", "scrapingdog.instagram_profile", {"username": "acme"}),
    ("https://api.scrapingdog.com/tiktok/profile?username=acme", "scrapingdog.tiktok_profile", {"username": "acme"}),
    ("https://api.scrapingdog.com/youtube/video?v=dQw4w9WgXcQ", "scrapingdog.youtube_video", {"v": "dQw4w9WgXcQ"}),
    ("https://api.scrapingdog.com/youtube/transcripts?v=dQw4w9WgXcQ", "scrapingdog.youtube_transcripts", {"v": "dQw4w9WgXcQ"}),
    ("https://api.scrapingdog.com/youtube/channel?channel_id=UC123", "scrapingdog.youtube_channel", {"channel_id": "UC123"}),
    ("https://api.scrapingdog.com/youtube/search?search_query=acme+funding", "scrapingdog.youtube_search", {"search_query": "acme funding"}),
])
def test_every_scrapingdog_endpoint_the_judge_uses_is_a_closed_operation(url, operation_id, expected):
    matched, parameters = ops.match_request("GET", url, b"", {})
    assert matched == operation_id and parameters == expected
    outbound = ops.build_outbound_request(matched, parameters)
    assert outbound.url.startswith("https://api.scrapingdog.com" + ops.OPERATIONS[operation_id].path + "?") and outbound.credential.location == "query"
    assert "api_key" not in outbound.query


def test_trusted_scorer_mode_strips_placeholder_credentials_and_normal_mode_refuses_them(monkeypatch):
    url = "https://api.scrapingdog.com/scrape?" + urlencode({"api_key": "placeholder-dog-key", "url": "https://example.com/", "dynamic": "true"})
    headers = {"x-api-key": "placeholder-exa-key", "Authorization": "Bearer placeholder-or-key", "Content-Type": "application/json", "User-Agent": "python-httpx/0.27"}
    monkeypatch.delenv(shim.TRUSTED_SCORER_ENV, raising=False)
    assert shim.trusted_scorer_mode() is False
    with pytest.raises(shim.ShimRequestError) as refused:
        shim.execute(method="GET", url=url, headers=headers, body=b"", timeout_ms=1000)
    assert refused.value.code == "forbidden_header"
    monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, "1")
    assert shim.trusted_scorer_mode() is True
    stripped_url, stripped_headers = shim.strip_caller_credentials(url, headers)
    assert "placeholder" not in stripped_url and stripped_url.startswith("https://api.scrapingdog.com/scrape?")
    assert set(stripped_headers) == {"Content-Type", "User-Agent"}
    operation_id, parameters = ops.match_request("GET", stripped_url, b"", stripped_headers)
    assert operation_id == "scrapingdog.scrape" and parameters == {"url": "https://example.com/", "dynamic": True}
    # Stripping never reaches the worker frame: the frame carries only the operation and its parameters.
    frame = shim.build_operation_frame(operation_id, parameters, 1000)
    assert b"placeholder" not in frame and b"api_key" not in frame


def test_quota_document_pins_judge_quotas_and_the_broker_allows_judge_models_only_for_scoring_runs():
    document = contracts.call_quota_document()
    assert document["schema_version"] == "lab_arena.call_quotas.v2" and document["per_scoring_work_item"] == contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM
    assert ops.CALL_QUOTA_HASH == contracts.document_hash(document) and contracts.ASSIGNMENT_KINDS == ("execute", "score")
    from tests.lab_arena.test_lab_arena_broker import price_table

    table = price_table()
    broker = br.Broker(store=None, key_for=lambda h, p: None, price_table=table, allowed_models=["openai/gpt-4o-mini"], judge_models=["anthropic/claude-3.5-haiku"], transport=None)
    judged = broker._openrouter_parameters({"model": "anthropic/claude-3.5-haiku", "messages": [{"role": "user", "content": "judge"}]}, kind="score")[0]
    assert judged["model"] == "anthropic/claude-3.5-haiku"
    with pytest.raises(br.BrokerError):
        broker._openrouter_parameters({"model": "anthropic/claude-3.5-haiku", "messages": []}, kind="execute")
    with pytest.raises(br.BrokerError):
        broker._openrouter_parameters({"model": "openai/gpt-4o-mini", "messages": []}, kind="score")
    with pytest.raises(contracts.ArenaContractError):
        br.Broker(store=None, key_for=lambda h, p: None, price_table=table, allowed_models=["openai/gpt-4o-mini"], judge_models=["perplexity/sonar"], transport=None)
    context = br.RunContext(run_id="r", assignment_id="a", icp_position=0, lease_token_hash=contracts.document_hash("l"), miner_hotkey="5" + "a" * 47, submission_id="s", stage=1)
    assert context.kind == "execute" and br.RunContext(**{**context.__dict__, "kind": "score"}).kind == "score"
