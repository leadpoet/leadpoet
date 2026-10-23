"""The request policy cannot bypass confirmed $4 sourcing admission."""
from dataclasses import replace

import pytest

from lab_arena import broker as br, contracts, reasoning_capture as rc
from tests.lab_arena.flat_per_icp_cost500_postgres_test import database, _capped_broker
from tests.lab_arena.test_lab_arena_broker import CHAT, GPT6_LUNA_RESPONSES


@pytest.mark.parametrize("operation,parameters", [
    ("openrouter.chat", CHAT), ("openrouter.responses", GPT6_LUNA_RESPONSES),
])
def test_frozen_reasoning_request_is_refused_at_exact_four_dollar_cap(database, operation, parameters):
    instance, context, store, transport, recorded = _capped_broker(database)
    profiles = {parameters["model"]: {"mandatory": False, "default_enabled": False,
                                      "supported_efforts": ["low", "medium", "high"]}}
    context = replace(context, reasoning_capture_policy=rc.POLICY, reasoning_model_profiles=profiles)
    try:
        result = instance.execute(context, operation_id=operation, parameters=parameters,
                                  action_sequence=3, timeout_ms=5000)
        assert result.status == 402 and result.call["error_code"] == "budget_refused"
        assert transport.sent == [] and len(recorded) == 1
        expected = br.normalized_request(operation, parameters,
            reasoning_capture_policy=rc.POLICY, reasoning_model_profiles=profiles)
        assert recorded[0]["call_doc"]["request_hash"] == contracts.document_hash(expected)
        assert recorded[0]["call_doc"]["reasoning_capture"]["requested"] is True
        assert recorded[0]["amount_microusd"] > 0
    finally:
        store.close()
