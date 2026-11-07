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


def test_activation_changes_only_new_rounds_with_existing_signal_authority():
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from lab_arena.service import ArenaService, RoundDefaults
    from tests.lab_arena.signal_cache_control_test import _RoundStore
    defaults = RoundDefaults(
        baseline_hotkey="5" + "B" * 47, baseline_source_url="https://example.com/baseline.tar.gz",
        scorer_image_digest="sha256:" + "1" * 64,
        scorer_image_reference="registry.example/scorer@sha256:" + "1" * 64,
        integrity_from="2026-10-01T00:00:00Z", company_judgment_from="2026-10-01T00:00:00Z",
        signal_judgment_from="2026-10-01T00:00:00Z")
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(defaults=defaults, mode="shadow", network_name="test", netuid=401, pinned_round_id=None)
    service._store = _RoundStore()
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (["5" + "C" * 47], [])
    service._require_integrity_schema = lambda: None
    service._require_company_quality_schema = lambda: None
    cutoff = datetime(2026, 11, 5, tzinfo=timezone.utc)
    old = service.create_round(cutoff, round_id="arena-2026-11-05-test")
    service._config.defaults = replace(defaults, evidence_judgment_cache=True)
    assert service.create_round(cutoff, round_id="arena-2026-11-05-test") == old
    future = service.create_round(cutoff + timedelta(days=1), round_id="arena-2026-11-06-test")
    assert "evidence_judgment_policy" not in old["scorer_policy"]
    assert future["scorer_policy"]["evidence_judgment_policy"] == "evidence_judgment_v2"
    assert future["scorer_policy"]["env_bindings"]["ARENA_EVIDENCE_JUDGMENT_POLICY"] == "evidence_judgment_v2"
    assert contracts.validate_scorer_policy(future["scorer_policy"]) == future["scorer_policy"]
