"""The new industry role can use the same model on bounded regional routes."""
from dataclasses import replace
from decimal import Decimal
import json

import pytest

from lab_arena import broker as br, operations
from tests.lab_arena.test_lab_arena_broker import (
    CHAT, CONTEXT, FakeLedgerStore, FakeTransport, HOST_KEYS, price_table,
)

MODEL = "openai/gpt-6.1-sol"


def fixture(*, roles, funding="host"):
    table = price_table()
    # The public model row may advertise Flex. It must not underprice Azure.
    table["models"][MODEL] = dict(
        prompt="0.000001", completion="0.000005", request="0",
        image="0", web_search="0.01", internal_reasoning="0",
    )
    store = FakeLedgerStore()
    transport = FakeTransport([(200, {
        "id": "gen-regional-industry", "model": MODEL,
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}], "usage": {"cost": "0.0002"},
    })])
    broker = br.Broker(
        store=store, key_for=lambda p: HOST_KEYS[p], price_table=table,
        judge_models=roles, transport=transport,
        provider_funding_source_for=lambda _c, _p: funding,
    )
    return broker, store, transport


@pytest.mark.parametrize("funding", ["host", "miner_key"])
def test_frozen_industry_role_routes_same_model_and_prices_regional_ceiling(funding):
    broker, store, transport = fixture(roles={"industry_evidence_review": MODEL}, funding=funding)
    request = {**CHAT, "model": MODEL}
    result = broker.execute(replace(CONTEXT, kind="score"),
                            operation_id="openrouter.chat", parameters=request,
                            action_sequence=0, timeout_ms=65000)
    assert result.status == 200
    sent = json.loads(transport.sent[0]["body"])
    assert sent["model"] == MODEL
    assert sent["messages"] == request["messages"]
    assert sent["provider"] == {
        "data_collection": "deny", "zdr": True,
        "only": ["azure/us", "azure/eu"], "allow_fallbacks": True,
        "max_price": {"prompt": 2.75, "completion": 11, "request": 0},
    }
    assert result.call["funding_source"] == funding
    assert result.call["actual_microusd"] == 200
    assert store.calls[result.call["call_identity"]]["actual"] == 200


@pytest.mark.parametrize("roles,kind", [
    ([MODEL], "score"),
    ({"company_fit_reverification": MODEL}, "score"),
    ({"industry_evidence_review": MODEL}, "execute"),
])
def test_legacy_roles_and_miner_execution_keep_existing_route(roles, kind):
    broker, _, transport = fixture(roles=roles)
    result = broker.execute(replace(CONTEXT, kind=kind), operation_id="openrouter.chat",
                            parameters={**CHAT, "model": MODEL}, action_sequence=0,
                            timeout_ms=65000)
    assert result.status == 200
    assert json.loads(transport.sent[0]["body"])["provider"] == dict(
        operations.OPENROUTER_STRICT_PROVIDER_POLICY)


def test_worker_cannot_select_a_regional_provider():
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.chat", {
            **CHAT, "model": MODEL, "provider": {"only": ["azure/us"]},
        })


def test_regional_industry_route_rejects_wrong_model_or_weakened_privacy():
    policy = {"data_collection": "deny", "zdr": True,
              "only": ["azure/us", "azure/eu"], "allow_fallbacks": True,
              "max_price": {"prompt": 2.75, "completion": 11, "request": 0}}
    for model, overrides in [("openai/gpt-4o-mini", {}), (MODEL, {"zdr": False}),
                              (MODEL, {"only": ["azure"]})]:
        with pytest.raises(operations.OperationRequestError):
            operations.build_outbound_request("openrouter.chat", {**CHAT, "model": model},
                                              openrouter_provider_policy={**policy, **overrides})


@pytest.mark.parametrize("long_input", [False, True])
def test_industry_route_reservation_covers_published_tiers(long_input):
    broker, _, _ = fixture(roles={"industry_evidence_review": MODEL})
    request = {**CHAT, "model": MODEL}
    if long_input:
        request["messages"] = [{"role": "user", "content": "x" * 272_000}]
    route = br._openrouter_host_route(
        kind="score", operation_id="openrouter.chat", model=MODEL,
        pricing=broker._price_table["models"][MODEL], parameters=request,
        industry_review_model=MODEL,
    )
    prompt = Decimal("0.0000055" if long_input else "0.00000275")
    completion = Decimal("0.0000165" if long_input else "0.000011")
    assert Decimal(route.reservation_pricing["prompt"]) == prompt
    assert Decimal(route.reservation_pricing["completion"]) == completion
    bound = br._max_openrouter_cost_for_pricing(route.reservation_pricing, request,
                                              max_output_tokens=request["max_tokens"])
    expected = (prompt * br.bounded_input_tokens(request)
                + completion * request["max_tokens"]) * br.MICROUSD
    assert Decimal(bound) >= expected


def test_frozen_role_map_is_not_changed_by_caller_mutation():
    roles = {"company_fit_reverification": MODEL}
    broker, _, transport = fixture(roles=roles)
    roles["industry_evidence_review"] = MODEL
    result = broker.execute(replace(CONTEXT, kind="score"), operation_id="openrouter.chat",
                            parameters={**CHAT, "model": MODEL}, action_sequence=0,
                            timeout_ms=65000)
    assert result.status == 200
    assert json.loads(transport.sent[0]["body"])["provider"] == dict(
        operations.OPENROUTER_STRICT_PROVIDER_POLICY)


@pytest.mark.parametrize("funding", ["host", "miner_key"])
@pytest.mark.parametrize("industry_role", [False, True, None])
def test_testnet_factory_uses_frozen_round_roles_not_service_defaults(funding, industry_role):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    # Execute the real nested factory without starting the testnet CLI,
    # loading credentials, or opening network connections.
    source = Path(__file__).resolve().parents[2] / "scripts/validate_miner_testnet.py"
    factories = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.FunctionDef) and node.name == "broker_factory"]
    assert len(factories) == 1
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        factories[0],
    ], type_ignores=[])
    ast.fix_missing_locations(module)
    frozen_role = "industry_evidence_review" if industry_role else "company_fit_reverification"
    service_role = "company_fit_reverification" if industry_role else "industry_evidence_review"
    frozen = {frozen_role: MODEL} if industry_role is not None else {}
    reference, store, transport = fixture(roles=frozen, funding=funding)
    namespace = {
        "broker_module": SimpleNamespace(Broker=br.Broker, HttpxProviderTransport=lambda: transport),
        "store": store,
        "provider_keys": HOST_KEYS,
        "price_table": reference._price_table,
        "submission_keys": SimpleNamespace(
            credential_for=lambda _context, provider: HOST_KEYS[provider],
            funding_source_for=lambda _context: funding,
        ),
    }
    exec(compile(module, str(source), "exec"), namespace)
    instance = namespace["broker_factory"](
        SimpleNamespace(scorer_policy={"judge_models": {service_role: MODEL}}),
        {"configuration_doc": {"scorer_policy": {"judge_models": frozen}}
         if industry_role is not None else {}},
    )
    frozen.clear()  # The constructed broker must retain its frozen role map.
    result = instance.execute(
        replace(CONTEXT, kind="score"), operation_id="openrouter.chat",
        parameters={**CHAT, "model": MODEL}, action_sequence=0, timeout_ms=65000,
    )
    if industry_role is None:
        assert result.status == 400
        assert json.loads(result.body)["error"] == {"code": "model_not_allowed"}
        assert not transport.sent
        return
    assert result.status == 200
    sent = json.loads(transport.sent[0]["body"])
    expected_route = {"data_collection": "deny", "zdr": True,
                      "only": ["azure/us", "azure/eu"], "allow_fallbacks": True,
                      "max_price": {"prompt": 2.75, "completion": 11, "request": 0}}
    assert sent["provider"] == (expected_route if industry_role else dict(
        operations.OPENROUTER_STRICT_PROVIDER_POLICY))
    assert result.call["funding_source"] == funding
    assert result.call["actual_microusd"] == 200
    assert store.calls[result.call["call_identity"]]["actual"] == 200
