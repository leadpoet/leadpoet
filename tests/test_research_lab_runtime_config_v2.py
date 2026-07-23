from dataclasses import replace

import pytest

from gateway.research_lab.config import (
    DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI,
    DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG,
    ResearchLabGatewayConfig,
)
from gateway.research_lab.dev_eval_runner import dev_eval_runner_enabled
from gateway.tee.research_lab_runtime_config_v2 import (
    BEHAVIOR_ENV_NAMES,
    HOST_ONLY_SECRET_FIELDS,
    ResearchLabRuntimeConfigV2Error,
    apply_behavior_environment,
    build_research_lab_execution_config,
    measured_git_tree_config,
    research_lab_config_from_document,
    validate_research_lab_execution_config,
)
from tests.v2_epoch_test_utils import epoch_test_environment


def test_execution_config_round_trips_every_non_secret_behavior_field():
    source = replace(
        ResearchLabGatewayConfig(),
        improvement_threshold_points=3.25,
        loop_planner_model="openai/test-model",
        loop_planner_fallback_models=("model/a", "model/b"),
        reimbursement_epochs=37,
        source_add_leg2_alpha_percent=6.5,
        internal_api_key="must-not-cross-boundary",
        hosted_worker_proxy_url="https://user:password@proxy.invalid",
        scoring_worker_proxy_url="https://user:password@proxy.invalid",
        miner_openrouter_key_ref_env_map_json='{"ref":"SECRET_ENV"}',
    )
    document = build_research_lab_execution_config(
        config=source,
        environment=epoch_test_environment(
            RESEARCH_LAB_TREE_MODE="active",
            RESEARCH_LAB_TREE_MAX_NODES="6",
        ),
        network="finney",
        netuid=71,
    )
    restored = research_lab_config_from_document(document)
    for name in document["fields"]:
        assert getattr(restored, name) == getattr(source, name)
    for name in HOST_ONLY_SECRET_FIELDS:
        assert getattr(restored, name) == getattr(ResearchLabGatewayConfig(), name)
    assert "must-not-cross-boundary" not in str(document)
    assert "https://user:password@proxy.invalid" not in str(document)
    assert document["deployment"] == {"network": "finney", "netuid": 71}
    assert document["behavior_environment"]["RESEARCH_LAB_TREE_MODE"] == "active"
    assert measured_git_tree_config(document).live_max_icps_per_node == (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    )


def test_execution_config_rejects_tampering_and_secret_material():
    document = build_research_lab_execution_config(
        environment=epoch_test_environment()
    )
    missing = {**document, "fields": dict(document["fields"])}
    missing["fields"].pop("improvement_threshold_points")
    with pytest.raises(ResearchLabRuntimeConfigV2Error, match="reviewed schema"):
        validate_research_lab_execution_config(missing)

    secret = {**document, "fields": dict(document["fields"])}
    secret["fields"]["private_repo_url"] = "https://user:pass@example.invalid/repo"
    with pytest.raises(ResearchLabRuntimeConfigV2Error, match="URI credentials"):
        validate_research_lab_execution_config(secret)


def test_execution_config_commits_production_defaults_when_env_omits_them():
    document = build_research_lab_execution_config(
        environment=epoch_test_environment(), network="finney", netuid=71
    )

    behavior = document["behavior_environment"]
    assert behavior["RESEARCH_LAB_TREE_MODE"] == "off"
    assert behavior["RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED"] == "true"
    assert (
        behavior["RESEARCH_LAB_DEV_SNAPSHOT_URI"]
        == DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI
    )


def test_dev_eval_runner_is_enabled_when_override_is_absent(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", raising=False)
    assert dev_eval_runner_enabled() is True


def test_behavior_environment_is_exact_and_applied(monkeypatch):
    values = epoch_test_environment(
        **{name: None for name in BEHAVIOR_ENV_NAMES}
    )
    values["RESEARCH_LAB_TREE_MAX_NODES"] = "5"
    document = build_research_lab_execution_config(environment=values)
    monkeypatch.setenv("RESEARCH_LAB_TREE_MAX_NODES", "2")
    monkeypatch.setenv("RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT", "false")
    apply_behavior_environment(document)
    assert document["behavior_environment"][
        "RESEARCH_LAB_TREE_MAX_NODES"
    ] == "5"
    import os

    assert os.environ["RESEARCH_LAB_TREE_MAX_NODES"] == "5"
    assert "RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT" not in os.environ


def test_measured_tree_icp_count_uses_the_committed_override():
    document = build_research_lab_execution_config(
        environment=epoch_test_environment(
            RESEARCH_LAB_TREE_LIVE_MAX_ICPS_PER_NODE="7"
        )
    )
    assert measured_git_tree_config(document).live_max_icps_per_node == 7
