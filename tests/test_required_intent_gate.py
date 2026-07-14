"""The ICP's primary intent is a hard requirement in scoring.

A verified bonus intent (matched_icp_signal >= 1) must never carry a company
whose required intent (index 0) failed — bonus evidence adds on top of a
verified primary, never substitutes for it.
"""

from qualification.scoring.lead_scorer import required_intent_satisfied


def _row(idx, after_decay):
    return {"matched_icp_signal": idx, "after_decay": after_decay, "raw": after_decay}


def test_verified_primary_satisfies():
    assert required_intent_satisfied([_row(0, 45.0)])
    assert required_intent_satisfied([_row(1, 30.0), _row(0, 12.0)])


def test_bonus_only_does_not_satisfy():
    # the exact hole: primary failed (0 score), bonus verified
    assert not required_intent_satisfied([_row(0, 0.0), _row(1, 40.0)])
    assert not required_intent_satisfied([_row(1, 40.0), _row(2, 25.0)])


def test_unmatched_or_empty_does_not_satisfy():
    assert not required_intent_satisfied([])
    assert not required_intent_satisfied([_row(-1, 50.0)])
    assert not required_intent_satisfied([{"matched_icp_signal": None, "after_decay": 50.0, "raw": 50.0}])
