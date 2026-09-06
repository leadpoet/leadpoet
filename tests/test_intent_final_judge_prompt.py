"""Regression tests for final intent-judge status taxonomy."""

from qualification.scoring.prompts._common import FINAL_JUDGE_RULES_BLOCK


def test_semantic_mismatch_is_contradicted_not_wrong_entity():
    semantic_rule = FINAL_JUDGE_RULES_BLOCK.split("- Use only", 1)[0]

    assert "return\n  contradicted regardless" in semantic_rule
    assert "wrong_entity" not in semantic_rule
