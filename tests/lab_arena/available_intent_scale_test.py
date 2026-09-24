"""The frozen scale changes percentages, never evidence or qualification."""

from copy import deepcopy
from fractions import Fraction

import pytest

from lab_arena import contracts, scoring, verify
from lab_arena.service import RoundDefaults
from qualification.scoring import competition
from qualification.scoring.lead_scorer import (
    COMPETITION_INTENT_CAP_BY_SIGNAL_COUNT,
    aggregate_competition_intent_scores,
)
from tests.lab_arena.test_lab_arena_verify import _breakdown, _icp, _junk, _signal


def policy(*, normalized=True, quality=False):
    return scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        normalize_intent_scale=normalized,
        company_quality=quality,
    )


def icp(count=1):
    return dict(_icp(), icp_id="scale-test", intent_signals=[
        f"Requested criterion {index}" for index in range(count)
    ])


@pytest.mark.parametrize("count", range(1, 9))
def test_perfect_results_reach_100_for_every_available_intent_count(count):
    # The raw ceiling comes from the actual intent aggregation, not a fake
    # normalized receipt. Additional criteria share the existing six-item cap.
    cap = aggregate_competition_intent_scores([60.0] * count)
    assert cap == COMPETITION_INTENT_CAP_BY_SIGNAL_COUNT[min(count, 6)]
    rows = [_breakdown(cap, details=[_signal(i, 60) for i in range(count)]) for _ in range(5)]
    original = deepcopy(rows)
    assert verify.per_icp_score(icp(count), rows, policy())["per_icp_score"] == 100
    assert verify.per_icp_score(icp(count), rows, policy(normalized=False))["per_icp_score"] == cap
    assert rows == original


@pytest.mark.parametrize("count", range(1, 7))
def test_partial_and_weaker_results_preserve_proportions_and_order(count):
    buyer = icp(count)
    cap = competition.available_intent_score_cap(buyer)
    results = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        rows = [_breakdown(cap * fraction) for _ in range(5)]
        result = verify.per_icp_score(buyer, rows, policy())
        assert result["per_icp_score"] == fraction * 100
        results.append(result["per_icp_score"])
    assert results == sorted(results)
    assert verify.per_icp_score(buyer, [_breakdown(cap)], policy())["per_icp_score"] == 20
    assert verify.per_icp_score(buyer, [], policy())["per_icp_score"] == 0
    assert verify.per_icp_score(buyer, [_junk()] * 5, policy())["per_icp_score"] == 0


@pytest.mark.parametrize("count", range(1, 7))
@pytest.mark.parametrize("quality", [False, True])
def test_existing_penalties_and_optional_coverage_scale_together(count, quality):
    buyer = icp(count)
    cap = competition.available_intent_score_cap(buyer)
    rows = [dict(_breakdown(cap), company_qualified=True) for _ in range(3)] + [_junk()]
    before = verify.per_icp_score(buyer, rows, policy(normalized=False, quality=quality))
    after = verify.per_icp_score(buyer, rows, policy(quality=quality))
    assert after["per_icp_score"] == float(Fraction(str(before["per_icp_score"])) * 100 / Fraction(str(cap)))
    assert after["fp_gate_count"] == before["fp_gate_count"] == 1
    assert after["fp_unverified_primary_count"] == before["fp_unverified_primary_count"] == 0
    assert after["company_scores"] == before["company_scores"]
    assert after["company_goal"] == before["company_goal"] == 5


def test_denominator_uses_full_requested_criteria_with_shared_dedup_and_bonus_rules():
    buyer = dict(icp(), intent_signals=["First", {"intent_signal": "First"}, "Second"],
                 bonus_intents=[{"intent_signal": "Second"}, {"intent_signal": "Third"}])
    assert competition.available_intent_score_cap(buyer) == 88
    # Returning only one piece of evidence cannot shrink the denominator.
    assert verify.per_icp_score(buyer, [_breakdown(60)] * 5, policy())["per_icp_score"] == float(Fraction(60 * 100, 88))
    for field in ("First", {"intent_signal": "First"}):
        assert competition.available_intent_score_cap(dict(icp(), intent_signals=field)) == 60
    assert competition.available_intent_score_cap(dict(icp(), intent_signals=[], intent_signal="First")) == 60
    with pytest.raises(competition.CompetitionScorerInputError, match="no intent signal"):
        competition.available_intent_score_cap(icp(0))


def test_frozen_policy_controls_scale_not_process_environment(monkeypatch):
    rows = [_breakdown(60)] * 5
    legacy = policy(normalized=False)
    monkeypatch.setenv(contracts.SCORE_NORMALIZATION_BINDING, contracts.AVAILABLE_INTENT_CAP_NORMALIZATION)
    before = competition.competition_score_from_breakdowns(icp(), rows)
    assert verify.per_icp_score(icp(), rows, legacy) == before
    monkeypatch.setenv(contracts.SCORE_NORMALIZATION_BINDING, "unknown-process-value")
    assert verify.per_icp_score(icp(), rows, policy())["per_icp_score"] == 100
    assert RoundDefaults().normalize_intent_scale is True


def test_unknown_versions_fail_closed_and_policy_roundtrips_existing_transport():
    import json

    settings = policy()
    assert contracts.validate_scorer_policy(json.loads(json.dumps(settings))) == settings
    assert set(settings) == set(policy(normalized=False))
    env = {}
    scoring.apply_policy_to_environment(settings, environ=env, credentials={name: "test-placeholder" for name in scoring.CREDENTIAL_ENV_NAMES})
    assert env[contracts.SCORE_NORMALIZATION_BINDING] == contracts.AVAILABLE_INTENT_CAP_NORMALIZATION
    settings["env_bindings"][contracts.SCORE_NORMALIZATION_BINDING] = "unknown"
    with pytest.raises(contracts.ArenaContractError, match="unsupported score normalization"):
        contracts.validate_scorer_policy(settings)
    with pytest.raises(contracts.ArenaContractError, match="requires integrity"):
        scoring.build_scorer_policy(normalize_intent_scale=True)


def test_benchmark_mean_has_no_new_count_or_coverage_penalty():
    scores = [verify.per_icp_score(icp(i), [_breakdown(competition.available_intent_score_cap(icp(i)))] * 5, policy())["per_icp_score"] for i in range(1, 7)]
    for count in (6, 10, 15, 30):
        perfect = (scores * count)[:count]
        assert verify.stage_score(perfect, count) == 100
        assert verify.stage_score([value / 2 for value in perfect], count) == 50
