"""The site runner's real request shapes through broker validation and pricing.

Shapes are copied from leadpoet-site (branch claude/subnet-lead-runner,
6bde0b11b4): the OpenRouter body that ``callStructuredLLM`` builds in
server/api/outreach/llm.ts for Lead Search's model reads
(server/lead-search/finding/model-read.ts), and Deepline's raw-v2 execute
answers for the finding core's contact check
(server/lead-search/finding/providers.ts and its spec).
"""

from __future__ import annotations

import copy
import json
from decimal import Decimal

import pytest

from lab_arena import signing
from lab_arena.broker import ProviderResponse, price_table_from_models_response
from lab_arena.lead_orders.broker import DEEPLINE_OPERATIONS, BrokerRequest, LeadOrderBroker, LeadOrderBrokerError
from lab_arena.lead_orders.policy import DEFAULT_POLICY_BODY, load_policy, sign_policy

GLM = "z-ai/glm-5.3-flash"
GEMINI = "google/gemini-3.7-flash"
LUNA = "openai/gpt-6-luna"
SIGNER = signing.LocalSigner.generate()
POLICY = load_policy(sign_policy(SIGNER, DEFAULT_POLICY_BODY), public_key_der=SIGNER.public_key_der,
                     expected_public_key_hash=SIGNER.public_key_hash)
# Listed prices (USD per token). GLM's listed price is its cheapest endpoint;
# its dearest endpoint on Nov 14 2026 charged 0.225 in and 0.75 out per million.
PRICE_TABLE = price_table_from_models_response({"data": [
    {"id": GLM, "pricing": {"prompt": "0.00000004", "completion": "0.00000008"}},
    {"id": GEMINI, "pricing": {"prompt": "0.00000075", "completion": "0.00000375"}},
    {"id": LUNA, "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
]}, fetched_at="2026-11-19T00:00:00Z")


def broker() -> LeadOrderBroker:
    return LeadOrderBroker(store=None, policy=POLICY, transport=None, registrar=None, leadpoet_secrets={},
                           price_table=lambda: PRICE_TABLE, current_epoch=lambda: 1)


def read_body(model: str, *, max_price: dict, reasoning: bool, healing: bool, max_tokens: int = 1_500) -> dict:
    """callStructuredLLM's wire body for one Lead Search read (llm.ts wireRequestBody)."""

    body = {
        "models": [model],
        "route": "fallback",
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "lead_search_model_read", "strict": True,
            "schema": {"type": "object", "properties": {"hiring": {"type": "string"}}, "required": ["hiring"],
                       "additionalProperties": False}}},
        "messages": [{"role": "system", "content": "Read the job post."},
                     {"role": "user", "content": json.dumps({"post": "We are hiring a payroll manager."})}],
        "max_tokens": max_tokens,
        "provider": {"data_collection": "deny", "zdr": True, "require_parameters": True, "max_price": max_price},
        "user": "0123456789abcdef",
    }
    if reasoning:
        body["reasoning"] = {"effort": "low", "exclude": True}  # OPEN_MODEL_REASONING
    if healing:
        body["temperature"] = 0
        body["plugins"] = [{"id": "response-healing"}]
    return body


def chat(body: dict) -> BrokerRequest:
    return BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {}, json.dumps(body).encode(), None)


def prepared(body: dict):
    outbound, reserve, models = broker()._prepare(chat(body))
    return json.loads(outbound.body), reserve, models


GLM_READ = read_body(GLM, max_price={"prompt": 0.3, "completion": 1, "request": 0}, reasoning=True, healing=False,
                     max_tokens=1_500 + 3_000)  # open models get OPEN_MODEL_REASONING_HEADROOM_TOKENS
GEMINI_READ = read_body(GEMINI, max_price={"prompt": 1, "completion": 4, "request": 0}, reasoning=False,
                        healing=False)


def test_glm_read_is_accepted_and_reserved_at_its_own_max_price():
    sent, reserve, models = prepared(GLM_READ)
    assert models == (GLM,)
    assert sent["provider"]["data_collection"] == "deny" and sent["provider"]["zdr"] is True
    assert sent["max_tokens"] == 4_500 and sent["reasoning"] == {"effort": "low", "exclude": True}
    assert sent["usage"] == {"include": True} and sent["user"] == "0123456789abcdef"
    # Output (reasoning included) at max_price, not the cheaper listed price.
    assert reserve >= 4_500 * 1  # 1 USD per million output tokens = 1 micro-USD per token
    listed_only = prepared(dict(GLM_READ, provider={"data_collection": "deny", "zdr": True}))[1]
    assert reserve > listed_only


def test_fallback_list_reserves_the_dearest_model():
    both = dict(GEMINI_READ, models=[GLM, GEMINI], provider={"data_collection": "deny", "zdr": True})
    _, reserve, models = prepared(both)
    assert models == (GLM, GEMINI)
    # The longer list adds a few prompt bytes; the dearest model sets the reserve.
    assert reserve >= prepared(dict(both, models=[GEMINI]))[1] > prepared(dict(both, models=[GLM]))[1]
    assert reserve < 2 * prepared(dict(both, models=[GLM]))[1] + prepared(dict(both, models=[GEMINI]))[1]


def test_max_completion_tokens_is_clamped_like_max_tokens_and_reasoning_stays_inside_it():
    body = read_body(LUNA, max_price={"prompt": 3, "completion": 15, "request": 0}, reasoning=False, healing=False)
    del body["max_tokens"]
    body["max_completion_tokens"] = 10_000_000
    body["reasoning"] = {"enabled": True, "max_tokens": 10_000_000}
    sent, reserve, _ = prepared(body)
    cap = POLICY.openrouter_max_output_tokens
    assert sent["max_completion_tokens"] == cap and "max_tokens" not in sent
    assert sent["reasoning"]["max_tokens"] == cap
    assert reserve >= cap * 15


def test_privacy_preferences_are_forced_and_cannot_be_weakened():
    sent, _, _ = prepared(dict(GEMINI_READ, provider={"require_parameters": True}))
    # max_price is always sent: the dearest listed price fills what the runner left out.
    assert sent["provider"] == {"require_parameters": True, "data_collection": "deny", "zdr": True,
                                "max_price": {"prompt": 0.75, "completion": 3.75, "request": 0.0}}
    for weak in ({"data_collection": "allow"}, {"zdr": False}):
        with pytest.raises(LeadOrderBrokerError) as refused:
            prepared(dict(GEMINI_READ, provider=weak))
        assert (refused.value.status, refused.value.code) == (400, "provider_privacy_required")


@pytest.mark.parametrize("change, code", [
    # OpenRouter's docs give response-healing no price and do not say it is free.
    ({"plugins": [{"id": "response-healing"}]}, "field_not_allowed:plugins"),
    ({"models": [GEMINI, "other/model"]}, "model_not_allowed"),
    ({"models": [GEMINI + ":online"]}, "model_variant_not_allowed"),
    ({"models": []}, "models_invalid"),
    ({"route": "price"}, "route_not_allowed"),
    ({"provider": {"order": ["together"]}}, "provider_field_not_allowed:order"),
    ({"provider": {"max_price": {"prompt": -1}}}, "max_price_invalid"),
    ({"provider": {"max_price": {"image": 1}}}, "max_price_invalid"),
    ({"reasoning": {"effort": "low", "budget": 5}}, "reasoning_invalid"),
    ({"max_completion_tokens": 10}, "max_tokens_invalid"),
    ({"user": "x" * 200}, "user_invalid"),
])
def test_unpriced_or_unsafe_variants_fail_closed(change, code):
    body = copy.deepcopy(GEMINI_READ)
    body.update(change)
    with pytest.raises(LeadOrderBrokerError) as refused:
        prepared(body)
    assert (refused.value.status, refused.value.code) == (400, code)


def test_runner_healing_shape_is_refused_until_it_drops_the_plugin():
    healed = read_body(GLM, max_price={"prompt": 0.3, "completion": 1, "request": 0}, reasoning=True, healing=True)
    with pytest.raises(LeadOrderBrokerError):
        prepared(healed)
    del healed["plugins"]
    assert prepared(healed)[2] == (GLM,)


def openrouter_answer(model: str, cost: str) -> ProviderResponse:
    body = {"id": "gen-1", "model": model, "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "cost": float(cost)}}
    return ProviderResponse(status=200, headers={}, body=json.dumps(body).encode())


def test_openrouter_settles_at_the_reported_cost_of_the_model_that_answered():
    models = (GLM, GEMINI)
    assert broker()._cost("openrouter", models, openrouter_answer(GEMINI, "0.000731")) == 731
    assert broker()._cost("openrouter", models, openrouter_answer("z-ai/glm-5.3-flash-1101", "0.000052")) == 52


def deepline_execute(operation: str) -> BrokerRequest:
    return BrokerRequest("deepline", "POST", "api/v2/integrations/%s/execute" % operation, {},
                         json.dumps({"payload": {"email": "ada.one@northwind.com"}}).encode(), None)


def deepline_answer(tool_response: dict, credits) -> ProviderResponse:
    body = {"status": "completed", "toolResponse": tool_response, "billing": {"credits_charged": credits}}
    return ProviderResponse(status=200, headers={}, body=json.dumps(body).encode())


@pytest.mark.parametrize("operation", ["crustdata_v3_job_search", "quickenrich_employee_email_enrich",
                                       "icypeas_email_verification", "icypeas_read_results"])
def test_finding_core_deepline_operations_are_accepted(operation):
    assert operation in DEEPLINE_OPERATIONS
    _, reserve, _ = broker()._prepare(deepline_execute(operation))
    assert reserve == POLICY.deepline_reserve_per_call_microusd


def test_custom_event_tools_stay_refused():
    for operation in ("serper_google_maps_search", "socrata_query_dataset"):
        with pytest.raises(LeadOrderBrokerError) as refused:
            broker()._prepare(deepline_execute(operation))
        assert refused.value.code == "operation_not_allowed"


def test_icypeas_async_costs_read_back_like_the_other_operations():
    # Icypeas bills the verification once; each read of its verdict bills 0,
    # still working or done (providers.spec.ts, Nov 16 2026 billing).
    verify = deepline_answer({"success": True, "item": {"_id": "v1", "status": "NONE"}}, 0.02)
    working = deepline_answer({"success": True, "total": 1, "items": [{"_id": "v1", "status": "IN_PROGRESS"}]}, 0)
    done = deepline_answer({"success": True, "total": 1, "items": [{"_id": "v1", "status": "DEBITED"}]}, 0)
    assert broker()._cost("deepline", None, verify) == int(Decimal("0.02") * Decimal("0.10") * 1_000_000)
    assert broker()._cost("deepline", None, working) == 0
    assert broker()._cost("deepline", None, done) == 0
    # No billing record on a 2xx stays unknown, never guessed.
    silent = ProviderResponse(status=200, headers={}, body=json.dumps({"status": "completed", "toolResponse": {}}).encode())
    assert broker()._cost("deepline", None, silent) is None


def test_firecrawl_scrape_for_a_fresh_job_post_copy_is_accepted_and_reserved():
    # finding/providers.ts fetchEventPageCopy, sent raw-v2.
    request = BrokerRequest("deepline", "POST", "api/v2/integrations/firecrawl_scrape/execute", {},
                            json.dumps({"payload": {"url": "https://jobs.example.com/payroll-manager",
                                                    "formats": ["rawHtml"], "onlyMainContent": False,
                                                    "maxAge": 0, "timeout": 30_000}}).encode(), "application/json",
                            headers={"x-deepline-execute-response-contract": "raw-v2"})
    outbound, reserve, _ = broker()._prepare(request)
    assert reserve == POLICY.deepline_reserve_per_call_microusd
    assert outbound.headers == {"x-deepline-execute-response-contract": "raw-v2"}


def bill(billing) -> ProviderResponse:
    return ProviderResponse(status=200, headers={}, body=json.dumps(
        {"status": "completed", "toolResponse": {}, "billing": billing}).encode())


def test_deepline_cost_reads_cost_usd_first_then_credits_like_the_site():
    from lab_arena import broker as arena_broker

    only_usd = bill({"cost_usd": 0.0031})
    assert broker()._cost("deepline", None, only_usd) == 3_100
    # Both present: cost_usd wins (plan credits can cost more than $0.10 each).
    assert broker()._cost("deepline", None, bill({"cost_usd": 0.0019, "credits_charged": 0.02})) == 1_900
    assert broker()._cost("deepline", None, bill({"cost_usd": 0.0000001})) == 1  # rounded up, never down
    # An unusable cost_usd falls back to credits, as the site does.
    for bad in ("0.5", -1, None, True):
        assert broker()._cost("deepline", None, bill({"cost_usd": bad, "credits_charged": 0.02})) == 2_000
    assert broker()._cost("deepline", None, bill({"cost_usd": "x"})) is None
    # Arena's own reader is unchanged: credits only.
    assert arena_broker.deepline_cost_microusd(bytes(only_usd.body)) is None
