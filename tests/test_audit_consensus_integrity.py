"""Regression tests for two audit-CONFIRMED consensus defects.

SEC1: consensus summed v_trust*stake over every evidence row with no dedup by
      validator, so a concurrent double-submit let one validator's vote count
      twice and bias approve/deny. Fix dedups to one row per validator.
C2:   when the total consensus weight was 0 (all covering validators had
      v_trust*stake==0, e.g. brand-new validators), a unanimous approve was
      silently flipped to deny. Fix falls back to an unweighted majority + a loud
      log instead of a silent deny.
"""

import asyncio

from gateway.utils.consensus import compute_weighted_consensus


def _ev(hk, decision, v_trust, stake, rep=0.8):
    return {
        "validator_hotkey": hk,
        "decision": decision,
        "rep_score": rep,
        "rejection_reason": None,
        "v_trust": v_trust,
        "stake": stake,
    }


def _run(ev):
    return asyncio.run(compute_weighted_consensus("lead1234abcd", 1, ev))


def test_sec1_duplicate_validator_rows_do_not_double_weight():
    # A approves (weight 100) but its row appears TWICE (double-submit); B denies
    # (weight 150). Deduped: single A(100 approve) vs B(150 deny) -> deny.
    # Un-deduped, A would count 200 and wrongly flip the outcome to approve.
    ev = [
        _ev("A", "approve", 1.0, 100.0),
        _ev("A", "approve", 1.0, 100.0),
        _ev("B", "deny", 1.0, 150.0),
    ]
    result = _run(ev)
    assert result["final_decision"] == "deny"
    assert result["validator_count"] == 2  # A and B, not 3


def test_c2_zero_weight_falls_back_to_majority_not_silent_deny():
    # All validators have v_trust 0 -> total_weight 0. 2 approve, 1 deny.
    ev = [
        _ev("A", "approve", 0.0, 100.0),
        _ev("B", "approve", 0.0, 100.0),
        _ev("C", "deny", 0.0, 100.0),
    ]
    result = _run(ev)
    assert result["final_decision"] == "approve"  # majority, not silent deny


def test_c2_zero_weight_majority_deny_still_denies():
    ev = [
        _ev("A", "deny", 0.0, 100.0),
        _ev("B", "deny", 0.0, 100.0),
        _ev("C", "approve", 0.0, 100.0),
    ]
    assert _run(ev)["final_decision"] == "deny"


def test_normal_weighted_path_unchanged():
    # 200 weighted approve vs 50 weighted deny -> approve, as before.
    ev = [
        _ev("A", "approve", 1.0, 100.0),
        _ev("B", "approve", 1.0, 100.0),
        _ev("C", "deny", 1.0, 50.0),
    ]
    assert _run(ev)["final_decision"] == "approve"
