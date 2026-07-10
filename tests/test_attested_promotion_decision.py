from research_lab.eval.promotion_metric import promotion_gate_decision


def _bundle(delta=2.0, *, rejected=False):
    if rejected:
        return {
            "private_holdout_gate": {
                "decision": "private_holdout_rejected",
                "private_holdout_evaluated": True,
                "baseline_aggregate_score": 10.0,
                "candidate_total_score": 12.0,
            },
            "aggregates": {},
        }
    return {"aggregates": {"mean_delta": delta}}


def _decision(bundle=None, **overrides):
    values = {
        "candidate_kind": "image_build",
        "candidate_parent": "sha256:parent",
        "active_parent": "sha256:parent",
        "threshold_points": 1.0,
        "auto_promotion_enabled": True,
    }
    values.update(overrides)
    return promotion_gate_decision(bundle or _bundle(), **values)


def test_promotion_gate_decision_preserves_every_existing_branch():
    assert _decision(auto_promotion_enabled=False).status == "disabled"
    assert _decision(candidate_kind="patch").status == "rejected_legacy_patch_candidate"
    assert _decision(_bundle(rejected=True)).status == "rejected_basis_unavailable"
    assert _decision(_bundle(delta=0.5)).status == "rejected_below_threshold"
    assert _decision(active_parent="sha256:new-parent").status == "stale_parent_needs_rescore"
    assert _decision().status == "promotion_passed"


def test_promotion_gate_decision_is_a_pure_canonical_projection():
    decision = _decision()
    assert decision.to_dict() == {
        "status": "promotion_passed",
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "candidate_kind": "image_build",
        "auto_promotion_enabled": True,
        "active_parent_matches": True,
        "metric_rejection_status": None,
    }
