"""Tests for the fableanalysis.md config fixes (bug #32, §6.4, §8.3 clamp).

These tests construct the Research Lab gateway config from a clean environment
so they exercise the same defaults prod gets when an env var is unset.
"""

import dataclasses
import logging
import os

import pytest

from gateway.research_lab.config import (
    MIN_IMPROVEMENT_THRESHOLD_POINTS,
    ResearchLabGatewayConfig,
)
from leadpoet_verifier.economics import DEFAULT_RESEARCH_LAB_EMISSION_PERCENT


# Every env var the config reads is either RESEARCH_LAB_*-prefixed, a
# QUALIFICATION_WEBSHARE_PROXY_<n> proxy slot, or one of the subnet selectors.
_ENV_PREFIXES = ("RESEARCH_LAB_", "QUALIFICATION_WEBSHARE_PROXY")
_ENV_EXACT = {"BITTENSOR_NETWORK", "SUBTENSOR_NETWORK", "BITTENSOR_NETUID", "NETUID"}


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES) or key in _ENV_EXACT:
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_dataclass_defaults_match_env_parsing_defaults(clean_env):
    """Bug #32: every dataclass field default must equal its env-parsing default.

    Prod behavior for an unset env var is the from_env default; direct
    dataclass constructions (tests, verify scripts) must exercise the same
    values. This comparison is programmatic so any future divergence on any
    field fails loudly.
    """
    from_env = ResearchLabGatewayConfig.from_env()
    declared = ResearchLabGatewayConfig()
    mismatches = {}
    for field in dataclasses.fields(ResearchLabGatewayConfig):
        env_value = getattr(from_env, field.name)
        default_value = getattr(declared, field.name)
        if env_value != default_value:
            mismatches[field.name] = {"dataclass": default_value, "from_env": env_value}
    assert not mismatches, f"dataclass defaults diverge from env-parsing defaults: {mismatches}"


def test_bug_32_reconciled_contradictions(clean_env):
    """The specific contradictions called out in fableanalysis bug #32."""
    from_env = ResearchLabGatewayConfig.from_env()
    declared = ResearchLabGatewayConfig()
    # icps_per_day: was 6 (dataclass) vs 2 (env default); env default wins —
    # the §6.4 raise to 4-6 is deliberately held until parity is verified.
    assert declared.lab_champion_icps_per_day == 2
    assert from_env.lab_champion_icps_per_day == 2
    # planner tokens: was 2400 (dataclass) vs 12000 (env default).
    assert declared.loop_planner_max_tokens == 12000
    assert from_env.loop_planner_max_tokens == 12000
    # Hybrid benchmark defaults: 10 fresh + 10 retained, public 7 weak / 3 strong.
    assert declared.lab_champion_window_mode == from_env.lab_champion_window_mode == "hybrid_fresh_retained"
    assert declared.lab_champion_fresh_icp_count == from_env.lab_champion_fresh_icp_count == 10
    assert declared.lab_champion_retained_icp_count == from_env.lab_champion_retained_icp_count == 10
    assert declared.public_benchmark_public_total_icps == from_env.public_benchmark_public_total_icps
    assert declared.public_benchmark_public_weak_total == from_env.public_benchmark_public_weak_total


def test_section_6_4_recommended_defaults(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    # Fail-closed scoring (2026-07-10): the health gate defaults ON —
    # measurements with critical provider/runtime failures quarantine instead
    # of standing as authoritative results. Zero-company thresholds default to
    # 1.0 because empty results are legitimate model outcomes.
    assert config.scoring_health_gate_enabled is True
    assert config.baseline_health_gate_enforced is True
    assert config.scoring_health_max_provider_error_rate == pytest.approx(0.10)
    assert config.scoring_health_max_reference_zero_company_rate == pytest.approx(1.0)
    assert config.scoring_health_max_candidate_zero_company_rate == pytest.approx(1.0)
    # Draft timeout: 12k-token diffs exceed 90s.
    assert config.auto_research_draft_timeout_seconds == 180
    # Build timeout includes the cold docker pull.
    assert config.code_edit_build_timeout_seconds == 1800
    # Source inspection budget.
    assert config.code_edit_source_inspection_max_files == 12
    # Daily rebenchmark starts predictably at 00:15 UTC; new candidate scoring
    # stops starting at 23:30 UTC until the next baseline completes.
    assert config.baseline_start_utc_offset_seconds == 900
    assert config.candidate_scoring_quiet_start_utc_seconds == 84600
    assert config.private_baseline_rebenchmark_enabled is True


def test_source_intelligence_defaults_and_independent_rollbacks(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    status = config.public_status()
    assert config.code_edit_source_access_v2 is True
    assert config.planner_symbol_index_enabled is True
    assert config.planner_reference_repair_enabled is True
    assert config.planner_reference_repair_max_attempts == 1
    assert status["hosted_worker"]["code_edit_source_inspection"]["source_access_v2"] is True
    assert status["hosted_worker"]["loop_planner"]["symbol_index_enabled"] is True
    assert status["hosted_worker"]["loop_planner"]["reference_repair_enabled"] is True

    clean_env.setenv("RESEARCH_LAB_SOURCE_ACCESS_V2", "false")
    clean_env.setenv("RESEARCH_LAB_PLANNER_SYMBOL_INDEX_ENABLED", "false")
    clean_env.setenv("RESEARCH_LAB_PLANNER_REFERENCE_REPAIR_ENABLED", "false")
    clean_env.setenv("RESEARCH_LAB_PLANNER_REFERENCE_REPAIR_MAX_ATTEMPTS", "9")
    config = ResearchLabGatewayConfig.from_env()
    assert config.code_edit_source_access_v2 is False
    assert config.planner_symbol_index_enabled is False
    assert config.planner_reference_repair_enabled is False
    assert config.planner_reference_repair_max_attempts == 1


def test_daily_scoring_schedule_env_overrides(clean_env):
    clean_env.setenv("RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS", "600")
    clean_env.setenv("RESEARCH_LAB_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS", "82800")
    config = ResearchLabGatewayConfig.from_env()
    assert config.baseline_start_utc_offset_seconds == 600
    assert config.candidate_scoring_quiet_start_utc_seconds == 82800

    clean_env.delenv("RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS", raising=False)
    clean_env.setenv("RESEARCH_LAB_BASELINE_MIN_UTC_DAY_DELAY_SECONDS", "120")
    assert ResearchLabGatewayConfig.from_env().baseline_start_utc_offset_seconds == 120


def test_source_add_status_defaults_enabled_and_env_gated(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    status = config.public_status()
    assert config.source_add_enabled is True
    assert config.source_add_rewards_enabled is True
    assert config.source_add_max_per_day_per_hotkey == 5
    assert not hasattr(config, "source_add_leg2_expiry_months")
    assert not hasattr(config, "source_add_ablation_required")
    assert not hasattr(config, "source_add_shadow_window_days")
    assert not hasattr(config, "source_add_ablation_threshold_points")
    assert status["source_add_enabled"] is True
    assert status["source_add"]["enabled"] is True
    assert status["source_add"]["rewards_enabled"] is True
    assert status["source_add"]["max_per_day_per_hotkey"] == 5

    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_ENABLED", "false")
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_REWARDS_ENABLED", "false")
    config = ResearchLabGatewayConfig.from_env()
    status = config.public_status()
    assert config.source_add_enabled is False
    assert config.source_add_rewards_enabled is False
    assert status["source_add_enabled"] is False
    assert status["source_add"]["enabled"] is False
    assert status["source_add"]["rewards_enabled"] is False


def test_source_add_work_lease_covers_three_probe_deadline(clean_env):
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_PROBE_TIMEOUT_SECONDS", "120")
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_WORK_LEASE_SECONDS", "30")

    config = ResearchLabGatewayConfig.from_env()

    assert config.source_add_probe_timeout_seconds == 120
    assert config.source_add_work_lease_seconds == 480


def test_research_lab_reward_allocation_defaults(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    assert config.lab_emission_percent == pytest.approx(
        float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)
    )
    assert config.fulfillment_emission_percent == pytest.approx(60.5)
    assert config.fulfillment_leaderboard_emission_percent == pytest.approx(9.5)
    assert config.lab_champion_min_alpha_percent == pytest.approx(7.0)
    assert config.lab_champion_extra_alpha_percent_per_point == pytest.approx(0.3)
    assert config.lab_champion_max_alpha_percent == pytest.approx(15.0)
    assert config.provider_cost_cap_usd_per_icp == pytest.approx(1.0)


def test_hybrid_window_and_public_split_defaults(clean_env):
    """Hybrid benchmark defaults to 10 fresh / 10 retained and 7/3 public."""
    config = ResearchLabGatewayConfig.from_env()
    window = config.lab_champion_eval_days * config.lab_champion_icps_per_day
    assert window == 20
    assert config.lab_champion_window_mode == "hybrid_fresh_retained"
    assert config.lab_champion_fresh_icp_count == 10
    assert config.lab_champion_retained_icp_count == 10
    assert config.lab_champion_fresh_icp_count + config.lab_champion_retained_icp_count == window
    assert config.public_benchmark_public_total_icps == 10
    assert config.public_benchmark_public_weak_total == 7
    assert config.public_benchmark_public_weak_total <= config.public_benchmark_public_total_icps
    # The default configuration must satisfy its own split validation.
    config.validate_public_benchmark_split()


def test_hybrid_window_total_validation(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    bad = dataclasses.replace(config, lab_champion_retained_icp_count=9)
    with pytest.raises(ValueError, match="FRESH_ICP_COUNT"):
        bad.validate_public_benchmark_split()


def test_public_split_env_overrides_still_respected(clean_env):
    clean_env.setenv("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_TOTAL_ICPS", "8")
    clean_env.setenv("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_TOTAL", "5")
    config = ResearchLabGatewayConfig.from_env()
    assert config.public_benchmark_public_total_icps == 8
    assert config.public_benchmark_public_weak_total == 5


def test_deliberately_unchanged_defaults(clean_env):
    """Knobs the audit says NOT to change in this pass (§6.4 / §8.3)."""
    config = ResearchLabGatewayConfig.from_env()
    # Never 0; default stays 1.0 (§8.3).
    assert config.improvement_threshold_points == pytest.approx(1.0)
    # Serial benchmark default; prod sets concurrency explicitly (§8.3: no >1
    # without dedicated Exa key + parity check).
    assert config.private_baseline_concurrency == 1
    # Auto-commit stays opt-in at the code level (§8.3; prod sets it itself).
    assert config.auto_commit_enabled is False
    assert config.crowning_enabled is True
    assert config.auto_promotion_enabled is True


def test_scoring_active_claim_cap_defaults_and_override(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    assert config.scoring_worker_max_active_claims == 8
    assert config.scoring_worker_min_available_memory_mb == 4096
    assert config.scoring_worker_max_load_per_cpu == pytest.approx(4.0)

    clean_env.setenv("RESEARCH_LAB_SCORING_WORKER_MAX_ACTIVE_CLAIMS", "4")
    clean_env.setenv("RESEARCH_LAB_SCORING_WORKER_MIN_AVAILABLE_MEMORY_MB", "8192")
    clean_env.setenv("RESEARCH_LAB_SCORING_WORKER_MAX_LOAD_PER_CPU", "2.5")
    config = ResearchLabGatewayConfig.from_env()
    assert config.scoring_worker_max_active_claims == 4
    assert config.scoring_worker_min_available_memory_mb == 8192
    assert config.scoring_worker_max_load_per_cpu == pytest.approx(2.5)

    clean_env.setenv("RESEARCH_LAB_SCORING_WORKER_MAX_ACTIVE_CLAIMS", "0")
    clean_env.setenv("RESEARCH_LAB_SCORING_WORKER_MIN_AVAILABLE_MEMORY_MB", "0")
    clean_env.setenv("RESEARCH_LAB_SCORING_WORKER_MAX_LOAD_PER_CPU", "0")
    config = ResearchLabGatewayConfig.from_env()
    assert config.scoring_worker_max_active_claims == 0
    assert config.scoring_worker_min_available_memory_mb == 0
    assert config.scoring_worker_max_load_per_cpu == pytest.approx(0.0)


@pytest.mark.parametrize("raw", ["0", "0.0", "-2.5", "0.05"])
def test_improvement_threshold_clamps_low_values_with_warning(clean_env, caplog, raw):
    clean_env.setenv("RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS", raw)
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.config"):
        config = ResearchLabGatewayConfig.from_env()
    assert config.improvement_threshold_points == pytest.approx(MIN_IMPROVEMENT_THRESHOLD_POINTS)
    assert config.improvement_threshold_points > 0
    warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS" in record.getMessage()
    ]
    assert warnings, "expected a warning log when clamping the improvement threshold"


def test_improvement_threshold_accepts_values_at_or_above_minimum(clean_env, caplog):
    clean_env.setenv("RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS", "0.5")
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.config"):
        config = ResearchLabGatewayConfig.from_env()
    assert config.improvement_threshold_points == pytest.approx(0.5)
    warnings = [
        record
        for record in caplog.records
        if "RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS" in record.getMessage()
    ]
    assert not warnings
