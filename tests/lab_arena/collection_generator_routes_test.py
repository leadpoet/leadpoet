"""The launch roster and hosting terms are identical for both collection arms."""

import pytest

from lab_arena.collection_intake import MODELS
from scripts.run_locked_collection import load_batch
from tests.lab_arena.collection_judge_test import PROMPT


EXPECTED = {
    "moonshotai/kimi-k3.1-pro": "deepinfra/bf16",
    "deepseek/deepseek-v4-pro-0813": "deepinfra/fp8",
    "mistralai/mistral-large-4-0": "mistral/zdr",
    "qwen/qwen3.8-27b": "deepinfra/fp8",
    "nvidia/nemotron-3.5-super-vl": "deepinfra/fp4",
    "deepseek/deepseek-v4.1-flash-1017": "particle-ai/us/flex",
}


def test_collection_assigns_verified_models_in_order():
    assert MODELS == tuple(EXPECTED)


def test_batch_rejects_unverified_host_before_leasing_a_miner_key():
    routes = {
        **EXPECTED,
        "anthropic/claude-fable-5.2": "test",
        "openai/gpt-6.2-astra": "test",
    }
    batch = dict(
        mode="pilot",
        name="test",
        seed="seed",
        routes=routes,
        scenario_ids=["B1-s2027"],
        budgets_microusd=dict.fromkeys(EXPECTED, 10000),
        control_slots={},
        drift_slots={},
    )
    load_batch(batch, PROMPT)
    routes["deepseek/deepseek-v4-pro-0813"] = "together"
    with pytest.raises(ValueError, match="verified generator route"):
        load_batch(batch, PROMPT)
