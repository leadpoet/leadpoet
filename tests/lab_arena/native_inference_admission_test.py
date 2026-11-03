"""Full-thinking execution refuses native model products before economic work."""

import copy
import json
from dataclasses import replace

import pytest

from lab_arena import operations, reasoning_capture as rc
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeLedgerStore, FakeTransport, make_broker
from tests.lab_arena.reasoning_capture_test import _reviewed_registry, _execute as execute_openrouter


DIRECT = [f"dataforseo_ai_optimization_{model}_llm_scraper_live_{format}"
          for model in ("chat_gpt", "gemini") for format in ("advanced", "html")]
CONTROLS = [
    ("dataforseo_serp_google_organic_live_advanced", "dataforseo", {"keyword": "business software"}),
    ("dataforseo_ai_optimization_ai_keyword_data_keywords_search_volume_live", "dataforseo", {"keywords": ["software"]}),
    ("deepseek_search", "deepseek", {"query": "business software"}),
    ("buyercaddy_find_customers_of_products", "buyercaddy", {"prompt": "Companies using software"}),
    ("firecrawl_extract", "firecrawl", {"urls": ["https://example.com"], "prompt": "Extract company name"}),
    ("generic_http_request", "generic_http", {"url": "https://example.com/article", "method": "GET"}),
    ("contextdev_get_web_naics", "contextdev", {"input": "example.com"}),
    ("discolike_run_company_research", "discolike", {"domains": ["example.com"], "query": "Company research"}),
    ("exa_answer", "exa", {"query": "Company research with sources"}),
    ("contextdev_post_web_search", "contextdev", {"query": "business software"}),
]
SCHEMA = {"type": "object", "properties": {
    **{name: {"type": "string"} for name in ("keyword", "query", "prompt", "url", "method", "input")},
    **{name: {"type": "array", "items": {"type": "string"}} for name in ("keywords", "urls", "domains")},
    "force_web_search": {"type": "boolean"},
}, "additionalProperties": False}
OK = (200, {"result": {"answer": "Synthetic offline result"},
            "billing": {"credits_charged": 2, "cost_usd": 0.02}})


def entry(tool, provider="dataforseo", aliases=()):
    return row(tool, provider=provider, categories=["research"], inputSchema=SCHEMA,
               operationAliases=[tool, *aliases],
               description="AI prompt model terminology is not runtime authority.")


def context(*extra, kind="execute", policy=rc.POLICY):
    catalog = frozen(*[entry(tool) for tool in DIRECT],
                     *[entry(tool, provider) for tool, provider, _ in CONTROLS], *extra)
    return replace(CONTEXT, round_id="arena-2026-10-28", kind=kind,
                   deepline_catalog=catalog, reasoning_capture_policy=policy,
                   reasoning_route_registry=_reviewed_registry())


def execute(provider, ctx, tool, payload, *, raw_http=False):
    operation, parameters = "deepline.execute", {"tool": tool, "payload": payload}
    if raw_http:
        operation, parameters = operations.match_request(
            "POST", f"https://code.deepline.com/api/v2/integrations/{tool}/execute",
            json.dumps({"payload": payload}).encode(), {},
            deepline_catalog=ctx.deepline_catalog, allow_full_thinking=True,
        )
    return provider.execute(ctx, operation_id=operation, parameters=parameters,
                            action_sequence=0, timeout_ms=1000)


@pytest.mark.parametrize("tool", DIRECT)
@pytest.mark.parametrize("raw_http", [False, True])
def test_native_model_family_refuses_before_funding_and_preserves_prior_ledger(tool, raw_http):
    funding = []
    ledger = FakeLedgerStore()
    ledger.calls["prior"] = {"kind": "settlement", "identity": "prior", "provider": "openrouter",
                             "amount": 0, "actual": 0, "call_doc": {}}
    original = copy.deepcopy(ledger.calls)
    provider, _, transport = make_broker(store=ledger, transport=FakeTransport([OK]),
                                         funding_source_for=lambda *_: funding.append("called") or "host")
    result = execute(provider, context(), tool, {"keyword": "Compute 2+2."}, raw_http=raw_http)
    assert result.status == 400 and result.call["error_code"] == "invalid_request"
    assert result.call["reason"] == "reasoning_route_unsupported"
    assert result.call["outcome"] == "not_dispatched"
    assert result.call["supported_reasoning_apis"] == ["openrouter.chat", "openrouter.responses"]
    assert funding == [] and transport.sent == [] and ledger.log == []
    assert ledger.calls == original


@pytest.mark.parametrize("tool,aliases", [
    ("dataforseo_ai_optimization_future_model_llm_scraper_live_json", []),
    ("dataforseo_company_lookup", ["ai_optimization_chat_gpt_llm_scraper_live_html"]),
])
@pytest.mark.parametrize("raw_http", [False, True])
def test_reviewed_provider_family_covers_future_formats_and_bound_aliases(tool, aliases, raw_http):
    provider, ledger, transport = make_broker(transport=FakeTransport([OK]))
    result = execute(provider, context(entry(tool, aliases=aliases)), tool,
                     {"keyword": "Compute 2+2.", "force_web_search": True}, raw_http=raw_http)
    assert result.status == 400 and result.call["reason"] == "reasoning_route_unsupported"
    assert ledger.calls == {} and transport.sent == []


def test_new_refusal_does_not_rewrite_an_existing_same_action_settlement():
    provider, ledger, transport = make_broker(transport=FakeTransport([OK]))
    ctx = context()
    # The old implementation admitted this exact native action without an
    # annotation. Seed its same-action receipt through the unchanged path.
    assert execute(provider, replace(ctx, reasoning_capture_policy=None), DIRECT[0],
                   {"keyword": "Company research"}).status == 200
    original, log, sent = copy.deepcopy(ledger.calls), list(ledger.log), list(transport.sent)
    denied = execute(provider, ctx, DIRECT[0], {"keyword": "Company research"})
    assert denied.call["reason"] == "reasoning_route_unsupported"
    assert ledger.calls == original and ledger.log == log and transport.sent == sent


@pytest.mark.parametrize("tool,provider_name,payload", CONTROLS)
@pytest.mark.parametrize("raw_http", [False, True])
def test_source_search_extract_enrichment_and_research_controls_still_settle(tool, provider_name, payload, raw_http):
    provider, ledger, transport = make_broker(transport=FakeTransport([OK]))
    ctx = context()
    result = execute(provider, ctx, tool, payload, raw_http=raw_http)
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert len(ledger.calls) == 1 and len(transport.sent) == 1
    assert next(iter(ledger.calls.values()))["call_doc"].get("reasoning_capture") is None
    replay = execute(provider, ctx, tool, payload, raw_http=raw_http)
    assert replay.call["idempotent"] is True and len(transport.sent) == 1


@pytest.mark.parametrize("kind,policy", [("execute", None), ("score", rc.POLICY), ("score", None)])
@pytest.mark.parametrize("tool", DIRECT)
def test_policy_absent_oct27_and_score_calls_are_unchanged(kind, policy, tool):
    provider, ledger, transport = make_broker(transport=FakeTransport([OK]))
    ctx = context(kind=kind, policy=policy)
    if policy is None:
        ctx = replace(ctx, round_id="arena-2026-10-27")
    result = execute(provider, ctx, tool, {"keyword": "Company research"})
    assert result.status == 200 and len(ledger.calls) == len(transport.sent) == 1


def test_provider_identity_and_public_model_host_get_do_not_imply_inference():
    tool = "other_company_search"
    provider, _, _ = make_broker(transport=FakeTransport([OK, OK]))
    ctx = context(entry(tool, provider="other", aliases=[DIRECT[0]]))
    assert execute(provider, ctx, tool, {"query": "Company research"}).status == 200
    provider, _, _ = make_broker(transport=FakeTransport([OK]))
    assert execute(provider, ctx, "generic_http_request", {
        "url": "https://api.openai.com/v1/chat/completions", "method": "GET",
    }).status == 200


def test_openrouter_approved_and_unreviewed_route_controls_remain_enforced():
    ctx = context()
    accepted, ledger, transport = execute_openrouter(ctx)
    assert accepted.status == 200 and len(ledger.calls) == len(transport.sent) == 1
    assert next(iter(ledger.calls.values()))["call_doc"]["reasoning_capture"]["reviewed_route"] is True
    denied, ledger, transport = execute_openrouter(replace(ctx, reasoning_route_registry=_reviewed_registry("different/model")))
    assert denied.status != 200 and ledger.calls == {} and transport.sent == []


def test_normal_research_uncertain_replay_does_not_gain_an_extra_provider_post():
    provider, ledger, transport = make_broker(transport=FakeTransport([(502, {"error": {"code": "upstream_error"}})]))
    ctx = context()
    first = execute(provider, ctx, "contextdev_post_web_search", {"query": "Company research"})
    assert first.call["outcome"] == "uncertain"
    execute(provider, ctx, "contextdev_post_web_search", {"query": "Company research"})
    assert len(ledger.calls) == 1
    assert next(iter(ledger.calls.values()))["kind"] == "uncertain"
    assert len([r for r in transport.sent if r["method"] == "POST"]) == 1
