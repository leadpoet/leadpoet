import json

import pytest

from lab_arena import contracts, scoring


def policy(**kwargs):
    return scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2",
                                       company_judgment_cache=True, **kwargs)


def test_evidence_policy_roundtrips_and_requires_existing_authorities():
    value = policy(evidence_judgment_cache=True)
    assert contracts.validate_scorer_policy(json.loads(json.dumps(value))) == value
    assert value["signal_judgment_policy"] == "signal_judgment_v1"
    assert value["evidence_judgment_policy"] == "evidence_judgment_v2"
    del value["signal_judgment_policy"]
    with pytest.raises(contracts.ArenaContractError, match="evidence judgments require"):
        contracts.validate_scorer_policy(value)


@pytest.mark.parametrize("active", [False, True])
def test_frozen_policy_controls_reuse_and_rejects_conflicting_environment(active):
    value = policy(evidence_judgment_cache=active)
    credentials = {key: "test-only-value" for key in scoring.CREDENTIAL_ENV_NAMES}
    environment = {}
    scoring.apply_policy_to_environment(value, credentials=credentials, environ=environment)
    assert environment["ARENA_EVIDENCE_JUDGMENT_POLICY"] == ("evidence_judgment_v2" if active else "")
    environment["ARENA_EVIDENCE_JUDGMENT_POLICY"] = "" if active else "evidence_judgment_v2"
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(value, credentials=credentials, environ=environment)


def test_binding_cannot_enable_reuse_without_frozen_marker():
    value = policy()
    value["env_bindings"]["ARENA_EVIDENCE_JUDGMENT_POLICY"] = "evidence_judgment_v2"
    with pytest.raises(contracts.ArenaContractError, match="binding differs"):
        contracts.validate_scorer_policy(value)
