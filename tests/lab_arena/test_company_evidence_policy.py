"""The new evidence path must not enable a different qualification policy."""

from lab_arena import scoring


def test_contact_round_without_company_quality_uses_evidence_investigator():
    # September 18 uses contact/integrity/intent-details requirements without
    # the later company_quality marker. Evidence recovery must work on this
    # policy without adding that marker or changing its output requirements.
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
        intent_details=True,
        company_quality=False,
    )
    frozen_policy = {**policy, "judge_models": {
        key: value for key, value in policy["judge_models"].items()
        if key != "company_evidence_investigator"
    }}
    scorer = scoring.lab_scorer(frozen_policy)
    assert scorer.evidence_investigator is True
    assert scorer.contacts_required is True
    assert scorer.integrity_policy is True
    assert scorer.company_quality is False
    assert "company_quality_policy" not in frozen_policy
    assert set(frozen_policy["judge_models"].values()) == set(
        policy["judge_models"].values()
    )


def test_evidence_role_does_not_change_score_or_qualification_settings():
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
        intent_details=True,
    )
    assert policy["fp_penalty_points"] == 10.0
    assert policy["fp_unverified_primary_penalty_points"] == 10.0
    assert policy["fp_penalty_icp_floor"] == 0.0
    assert policy["company_cap_rule"] == "icp_max_companies"
    assert policy["max_scored_companies"] == 0
    assert policy["pre_slice_rule"] == "first_n_model_order"
    assert policy["employee_bucket_rule"] == "lab_relaxed_buckets"
    assert policy["intent_details_policy"] == "intent_details_v1"
    assert "company_quality_policy" not in policy
