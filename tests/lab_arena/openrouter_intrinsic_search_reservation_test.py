"""Published per-request Sonar search fees, with no provider/network calls."""

from dataclasses import replace
from decimal import Decimal
import json

import pytest

from lab_arena import broker as br, operations
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeLedgerStore, FakeTransport, make_broker, price_table,
)


MODELS = ("perplexity/sonar", "perplexity/sonar-pro")


def table_for(model):
    table = price_table()
    table["models"][model] = {
        "prompt": "0.000001", "completion": "0.000001", "request": "0",
        "image": "0", "web_search": "0.005", "internal_reasoning": "0",
    }
    return br.validate_price_table(table)


def parameters(model):
    return {"model": model, "messages": [{"role": "user", "content": "synthetic"}], "max_tokens": 1000}


@pytest.mark.parametrize("model", MODELS)
def test_intrinsic_search_reserves_catalog_fee_without_tools(model):
    params = parameters(model)
    table = table_for(model)
    token_bound = br.bounded_input_tokens(params) + params["max_tokens"]
    assert br.max_openrouter_cost_microusd(table, model, params, max_output_tokens=1000) == token_bound + 5000
    # The public helper's model argument selects pricing even when the body
    # is only a token-estimation envelope, without a repeated model field.
    body = {key: value for key, value in params.items() if key != "model"}
    assert br.max_openrouter_cost_microusd(table, model, body, max_output_tokens=1000) == br.bounded_input_tokens(body) + 6000


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind,funding", [("execute", "host"), ("execute", "miner_key"), ("score", "host")])
def test_native_broker_reserves_fee_and_settles_exact_cost_once(model, kind, funding):
    table = table_for(model)
    payload = {"id": "gen-synthetic-search", "model": model, "choices": [], "usage": {"cost": "0.00582"}}
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, payload)]),
        provider_funding_source_for=lambda _context, _provider: funding,
    )
    broker._price_table = table
    broker._judge_models = (model,)
    params = parameters(model)
    context = replace(CONTEXT, kind=kind)
    first = broker.execute(context, operation_id="openrouter.chat", parameters=params, action_sequence=0, timeout_ms=30000)
    second = broker.execute(context, operation_id="openrouter.chat", parameters=params, action_sequence=0, timeout_ms=30000)
    assert first.status == second.status == 200
    row = store.calls[first.call["call_identity"]]
    assert row["amount"] == br.bounded_input_tokens(params) + 6000
    assert row["actual"] == first.call["actual_microusd"] == 5820
    assert row["funding_source"] == funding
    assert row["reserve_remaining"] is False
    assert store.log.count("settle") == 1
    assert len(transport.sent) == 1
    sent = json.loads(transport.sent[0]["body"])
    assert sent["messages"] == params["messages"]
    assert sent["max_tokens"] == params["max_tokens"]
    assert "tools" not in sent and "web_search_options" not in sent


def test_observed_sonar_underreservation_has_synthetic_equivalent_envelope():
    # Real GET-only receipt gen-1793399973-j9YjLwwSpWdZfED8BZKF:
    # 120 native input + 700 native output at $1/M = $0.00082;
    # total $0.00582 includes the published $0.005 intrinsic search fee.
    # Original request bytes were not retained. This is an equivalent bound,
    # not a claim to reconstruct that request or its hash.
    params = parameters("perplexity/sonar")
    params["messages"][0]["content"] = ""
    padding = 472 - br.bounded_input_tokens(params)
    assert padding > 0
    params["messages"][0]["content"] = "x" * padding
    assert br.bounded_input_tokens(params) + 1000 == 1472
    fixed = br.max_openrouter_cost_microusd(table_for(params["model"]), params["model"], params, max_output_tokens=1000)
    assert fixed == 6472
    assert 1472 < 5820 <= fixed
    assert (Decimal(120 + 700) / Decimal(1000000) + Decimal("0.005")) == Decimal("0.00582")


def test_token_only_models_do_not_pay_an_optional_search_fee():
    model = "openai/gpt-4o-mini"
    table = table_for(model)  # Optional hosted search has a nonzero catalog price.
    params = parameters(model)
    assert br.max_openrouter_cost_microusd(table, model, params, max_output_tokens=1000) == br.bounded_input_tokens(params) + 1000


@pytest.mark.parametrize("model", MODELS)
def test_intrinsic_fee_uses_the_catalog_and_preserves_other_components(model):
    table = table_for(model)
    table["models"][model].update(prompt="0.000003", completion="0.000015", web_search="0.007", request="0.001", internal_reasoning="0.000002")
    params = parameters(model)
    expected = br.bounded_input_tokens(params) * 3 + 1000 * (15 + 2) + 1000 + 7000
    assert br.max_openrouter_cost_microusd(table, model, params, max_output_tokens=1000) == expected


@pytest.mark.parametrize("model", MODELS)
def test_missing_intrinsic_price_refuses_before_any_reservation(model):
    broker, store, transport = make_broker()
    broker._price_table = table_for(model)
    broker._price_table["models"][model]["web_search"] = "0"
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=parameters(model), action_sequence=0, timeout_ms=30000)
    assert result.status == 503
    assert store.log == [] and transport.sent == []


@pytest.mark.parametrize("fee", ["0", "0.005", "0.02"])
def test_explicit_server_search_keeps_all_bounded_calls_and_existing_floor(fee):
    model = "openai/gpt-4o-mini"
    table = table_for(model)
    table["models"][model]["web_search"] = fee
    params = {"model": model, "input": "synthetic", "max_output_tokens": 1000,
              "tools": [{"type": "openrouter:web_search"}],
              "max_tool_calls": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS}
    expected = Decimal(br.bounded_input_tokens(params) + 1000) + max(Decimal(fee), br.OPENROUTER_NATIVE_WEB_SEARCH_RESERVATION_USD_PER_CALL) * br.MICROUSD * params["max_tool_calls"]
    assert br.max_openrouter_cost_microusd(table, model, params, max_output_tokens=1000) == int(expected)


def test_sonar_refuses_before_dispatch_if_fixed_fee_does_not_fit():
    broker, store, transport = make_broker(store=FakeLedgerStore(openrouter_capacity=5000))
    broker._price_table = table_for("perplexity/sonar")
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=parameters("perplexity/sonar"), action_sequence=0, timeout_ms=30000)
    assert result.status == 402
    assert transport.sent == []
    assert store.log == ["reserve"]


def test_old_lower_nonterminal_reservation_cannot_dispatch_after_fix():
    class OldReservationStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            if result["status"] == "reserved":
                row = self.calls[kwargs["call_identity"]]
                row["amount"] -= 5000
                result.update(idempotent=True, amount_microusd=row["amount"])
            return result

    broker, store, transport = make_broker(store=OldReservationStore())
    broker._price_table = table_for("perplexity/sonar")
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=parameters("perplexity/sonar"), action_sequence=0, timeout_ms=30000)
    assert result.status == 503
    assert transport.sent == []
    assert "dispatch" not in store.log
