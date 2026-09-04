"""Current Research Lab configuration boundaries."""

import dataclasses
import os

import pytest

from gateway.research_lab.config import ResearchLabGatewayConfig
from leadpoet_verifier.economics import DEFAULT_RESEARCH_LAB_EMISSION_PERCENT


_ENV_PREFIXES = ("RESEARCH_LAB_", "QUALIFICATION_WEBSHARE_PROXY")
_ENV_EXACT = {
    "BITTENSOR_NETWORK",
    "SUBTENSOR_NETWORK",
    "BITTENSOR_NETUID",
    "NETUID",
}


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES) or key in _ENV_EXACT:
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_dataclass_defaults_match_env_parsing_defaults(clean_env):
    parsed = ResearchLabGatewayConfig.from_env()
    declared = ResearchLabGatewayConfig()
    assert {
        field.name: getattr(parsed, field.name)
        for field in dataclasses.fields(ResearchLabGatewayConfig)
    } == {
        field.name: getattr(declared, field.name)
        for field in dataclasses.fields(ResearchLabGatewayConfig)
    }


def test_retired_model_and_loop_environment_has_no_effect(clean_env):
    baseline = ResearchLabGatewayConfig.from_env()
    retired = {
        "RESEARCH_LAB_PRIVATE_REPO_BRANCH": "retired-branch",
        "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI": "s3://retired/model.json",
        "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID": "alias/retired",
        "RESEARCH_LAB_AUTORESEARCH_ENABLED": "true",
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "true",
        "RESEARCH_LAB_CONDITIONAL_VALIDATION_MODE": "enforce",
        "RESEARCH_LAB_CODE_EDIT_BUILD_TIMEOUT_SECONDS": "9999",
        "RESEARCH_LAB_LOOP_PLANNER_MAX_TOKENS": "9999",
    }
    for name, value in retired.items():
        clean_env.setenv(name, value)

    assert ResearchLabGatewayConfig.from_env() == baseline
    assert not hasattr(baseline, "private_repo_branch")
    assert not hasattr(baseline, "private_model_manifest_uri")
    assert not hasattr(baseline, "miner_submissions_enabled")
    assert not hasattr(baseline, "conditional_validation_mode")


def test_source_add_status_is_current_and_old_loop_intake_stays_closed(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    status = config.public_status()

    assert status["miner_submissions_enabled"] is False
    assert config.source_add_enabled is True
    assert config.source_add_rewards_enabled is True
    assert config.source_add_dispatcher_enabled is True
    assert config.source_add_functional_probes_enabled is True
    assert config.source_add_functional_rewards_enabled is True
    assert config.source_add_leg1_alpha_percent == pytest.approx(0.2)
    assert config.source_add_leg2_alpha_percent == pytest.approx(0.0)
    assert status["source_add"]["enabled"] is True
    assert status["source_add"]["max_per_day_per_hotkey"] == 5

    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_ENABLED", "false")
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_REWARDS_ENABLED", "false")
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED", "false")
    config = ResearchLabGatewayConfig.from_env()
    assert config.source_add_enabled is False
    assert config.source_add_rewards_enabled is False
    assert config.source_add_dispatcher_enabled is False
    assert config.public_status()["miner_submissions_enabled"] is False


def test_source_add_work_lease_covers_three_probe_deadlines(clean_env):
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_PROBE_TIMEOUT_SECONDS", "120")
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_WORK_LEASE_SECONDS", "30")

    config = ResearchLabGatewayConfig.from_env()

    assert config.source_add_probe_timeout_seconds == 120
    assert config.source_add_work_lease_seconds == 480


def test_retained_reward_allocation_defaults(clean_env):
    config = ResearchLabGatewayConfig.from_env()
    assert config.lab_emission_percent == pytest.approx(
        float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)
    )
    assert config.fulfillment_emission_percent == pytest.approx(60.5)
    assert config.fulfillment_leaderboard_emission_percent == pytest.approx(9.5)
    assert config.lab_champion_min_alpha_percent == pytest.approx(7.0)
    assert config.lab_champion_extra_alpha_percent_per_point == pytest.approx(0.3)
    assert config.lab_champion_max_alpha_percent == pytest.approx(15.0)


def test_current_source_add_and_reward_overrides(clean_env):
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_PROBE_MAX_ATTEMPTS", "8")
    clean_env.setenv("RESEARCH_LAB_SOURCE_ADD_LEG1_ALPHA_PERCENT", "0.35")
    clean_env.setenv("RESEARCH_LAB_EMISSION_PERCENT", "8.5")

    config = ResearchLabGatewayConfig.from_env()

    assert config.source_add_probe_max_attempts == 5
    assert config.source_add_leg1_alpha_percent == pytest.approx(0.35)
    assert config.lab_emission_percent == pytest.approx(8.5)
