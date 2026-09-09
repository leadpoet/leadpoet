"""The retired intake cannot be reached through public or measured surfaces."""

import inspect
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.research_lab import api
from gateway.research_lab.admin import build_parser
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee.coordinator_executor_v2 import COORDINATOR_OPERATIONS_V2
from gateway.tee.reward_executor_v2 import execute_reward_decision_v2
from gateway.tee.supabase_source_v2 import QUERY_POLICIES
from leadpoet_verifier.economics import allocate_research_lab_epoch


@pytest.mark.parametrize("enabled", [False, True])
def test_live_status_exposes_shared_restart_latch_without_source_controls(monkeypatch, enabled):
    from gateway.tee import gateway_miner_maintenance_restart_v1 as maintenance

    monkeypatch.setenv("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED", str(enabled).lower())
    app = FastAPI()
    app.include_router(api.router)
    response = TestClient(app).get("/research-lab/status")
    assert response.status_code == 200
    status = response.json()
    assert status["miner_submissions_enabled"] is enabled
    assert not any("source_add" in key for key in status)
    monkeypatch.setattr(maintenance, "_production_parity_clone_authority", lambda *a, **k: {"verified": True})
    kwargs = {
        "deploy_commit": "a" * 40,
        "candidate_tree_hash": "b" * 40,
        "runtime_environment": {"RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false"},
        "runtime_status": status,
    }
    if enabled:
        with pytest.raises(maintenance.GatewayMinerMaintenanceRestartError, match="not disabled"):
            maintenance.verify_gateway_miner_maintenance_runtime_state(**kwargs)
    else:
        assert maintenance.verify_gateway_miner_maintenance_runtime_state(**kwargs)["runtime_status"] == "disabled"


def test_source_intake_routes_operations_and_configuration_are_absent():
    assert not any("source-adapter" in route.path for route in api.router.routes)
    assert not any("source_add" in name for name in COORDINATOR_OPERATIONS_V2)
    assert not any("source_add" in name for name in QUERY_POLICIES)
    assert not any("source_add" in name for name in inspect.signature(ResearchLabGatewayConfig).parameters)
    root = Path(__file__).resolve().parents[1]
    for path in (
        "research_lab/source_add.py",
        "research_lab/source_add_miner.py",
        "research_lab/source_add_rewards.py",
        "gateway/research_lab/source_add_workflow.py",
        "gateway/tee/coordinator_source_add_v2.py",
        "gateway/tee/source_add_runtime_v2.py",
    ):
        assert not (root / path).exists()


@pytest.mark.parametrize("command", (
    "pause-source-add", "resume-source-add", "reconcile-source-add-reward-statuses",
))
def test_source_intake_admin_commands_are_not_available(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


@pytest.mark.parametrize("kind", ("source_add_leg1", "source_add_leg2", "source_add_migration"))
def test_source_reward_decisions_cannot_execute(kind):
    with pytest.raises(ValueError, match="unsupported"):
        execute_reward_decision_v2({"decision_kind": kind, "decision_payload": {}})


def test_current_allocator_has_no_source_obligations_or_reward_section():
    assert "active_source_add_obligations" not in inspect.signature(allocate_research_lab_epoch).parameters
    allocation = allocate_research_lab_epoch(1, {}, [], [])
    assert not any("source_add" in name for name in allocation)
    assert allocation["unallocated_percent"] == allocation["lab_cap_percent"]
