from dataclasses import replace

import pytest

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee.research_lab_runtime_config_v2 import (
    BEHAVIOR_ENV_NAMES,
    HOST_ONLY_SECRET_FIELDS,
    ResearchLabRuntimeConfigV2Error,
    apply_behavior_environment,
    build_research_lab_execution_config,
    research_lab_config_from_document,
    validate_research_lab_execution_config,
)


def test_execution_config_round_trips_every_non_secret_behavior_field():
    source = replace(
        ResearchLabGatewayConfig(),
        improvement_threshold_points=3.25,
        inner_loop_mode="rank",
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
        environment={"RESEARCH_LAB_LOOP_DRAFTS_PER_CALL": "4"},
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
    assert document["fields"]["inner_loop_mode"] == "rank"


def test_execution_config_rejects_tampering_and_secret_material():
    document = build_research_lab_execution_config(environment={})
    missing = {**document, "fields": dict(document["fields"])}
    missing["fields"].pop("improvement_threshold_points")
    with pytest.raises(ResearchLabRuntimeConfigV2Error, match="reviewed schema"):
        validate_research_lab_execution_config(missing)

    secret = {**document, "fields": dict(document["fields"])}
    secret["fields"]["private_repo_url"] = "https://user:pass@example.invalid/repo"
    with pytest.raises(ResearchLabRuntimeConfigV2Error, match="URI credentials"):
        validate_research_lab_execution_config(secret)


def test_behavior_environment_is_exact_and_applied(monkeypatch):
    values = {name: None for name in BEHAVIOR_ENV_NAMES}
    values["RESEARCH_LAB_LOOP_DRAFTS_PER_CALL"] = "5"
    document = build_research_lab_execution_config(environment=values)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DRAFTS_PER_CALL", "2")
    monkeypatch.setenv("RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT", "false")
    apply_behavior_environment(document)
    assert document["behavior_environment"][
        "RESEARCH_LAB_LOOP_DRAFTS_PER_CALL"
    ] == "5"
    import os

    assert os.environ["RESEARCH_LAB_LOOP_DRAFTS_PER_CALL"] == "5"
    assert "RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT" not in os.environ
