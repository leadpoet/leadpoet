"""Prompt routing checks; these tests do not claim model accuracy."""
import pytest
from qualification.scoring import intent_details


def _prompt(targets=None):
    held = {"unit_grounding": [
        {"unit_id": i, "status": "UNPROVEN", "contains_factual_claim": True}
        for i in (0, 1)
    ]}
    return intent_details._semantic_repair_prompt("", {0: [], 1: []}, held, {0, 1}, targets)


@pytest.mark.parametrize("targets", [None, {}])
def test_nonrelative_semantic_repair_omits_inactive_strict_timing_rule(targets):
    prompt = _prompt(targets)
    assert "For relative_time_grounding_review" not in prompt
    assert "support from the same source_index" not in prompt
    assert "For every semantic recheck" in prompt
    assert "Choose the strongest complementary" in prompt
    assert "EVERY factual clause is supported" in prompt
    assert "Do not use outside knowledge" in prompt


def test_mixed_repair_keeps_strict_timing_rule_explicitly_target_scoped():
    prompt = _prompt({1: {"disputed_clause": "recent coverage"}})
    assert "only to units with relative_time_target" in prompt
    assert "including units containing absolute dates" in prompt
    assert "For relative_time_grounding_review" in prompt
    assert "support from the same source_index; otherwise return UNPROVEN." in prompt
