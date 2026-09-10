"""Restart wiring for the optional Arena service and validator runner."""

import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_gateway_restart_replaces_and_checks_the_arena_sidecar() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    destructive = script.index("GATEWAY_DESTRUCTIVE_PHASE_STARTED=1")
    stop_helper = script.index(
        'GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER="$GATEWAY_CONTROLLER_PROCESS_HELPER"'
    )
    stop = script.index("stop_lab_arena_service", destructive)
    gateway_health = script.index(
        'record_gateway_restart_timing "gateway_v2_health_ready"'
    )
    start = script.index("start_lab_arena_service", gateway_health)
    completed = script.index('GATEWAY_DEPLOY_STAGE="completed"', start)

    assert stop_helper < destructive < stop < gateway_health < start < completed
    assert 'stop_lab_arena_service "$GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER"' in script
    assert (
        'GATEWAY_CONTROLLER_PROCESS_HELPER="$GATEWAY_RESTART_AUTHORITY_ROOT/'
        'scripts/manage_owned_process_group.py"'
        in script
    )
    assert "scripts/run_lab_arena_service.py" in script
    function_start = script.index("start_lab_arena_service() {")
    function_end = script.index("\n}\n", function_start)
    function = script[function_start:function_end]
    assert '"$GATEWAY_CONTROLLER_PROCESS_HELPER" record' in function
    assert '"$GATEWAY_CONTROLLER_PROCESS_STATE_FILE"' in function
    sidecar_health = function.index("http://127.0.0.1:8792/arena/v1/current")
    public_health = function.index("http://127.0.0.1:8000/arena/v1/current")
    assert sidecar_health < public_health
    assert 'case "$mode" in' in script
    assert "shadow|live" in script
    assert '--environment-file "$GATEWAY_ENV_FILE"' in function
    assert "manage_owned_process_group.py" in script
    assert 'pkill -TERM -f "scripts/run_lab_arena_service' not in script


def test_arena_service_loads_only_scoped_values(tmp_path, monkeypatch) -> None:
    from scripts import run_lab_arena_service

    environment = tmp_path / "gateway.env"
    environment.write_text(
        "UNRELATED_BARE_VALUE=production value with spaces # retained cache form\n"
        "LAB_ARENA_MODE=shadow\n"
        "LAB_ARENA_OPENROUTER_API_KEY='scoped secret'\n"
        "OPENROUTER_API_KEY=shared-secret\n"
        "SUPABASE_SERVICE_ROLE_KEY=unrelated-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LAB_ARENA_MODE", raising=False)
    monkeypatch.delenv("LAB_ARENA_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    run_lab_arena_service.load_scoped_environment(environment)

    assert os.environ["LAB_ARENA_MODE"] == "shadow"
    assert os.environ["LAB_ARENA_OPENROUTER_API_KEY"] == "scoped secret"
    assert "UNRELATED_BARE_VALUE" not in os.environ
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "SUPABASE_SERVICE_ROLE_KEY" not in os.environ


def test_arena_service_keeps_explicit_scoped_override(tmp_path, monkeypatch) -> None:
    from scripts import run_lab_arena_service

    environment = tmp_path / "gateway.env"
    environment.write_text("LAB_ARENA_MODE=live\n", encoding="utf-8")
    monkeypatch.setenv("LAB_ARENA_MODE", "off")

    run_lab_arena_service.load_scoped_environment(environment)

    assert os.environ["LAB_ARENA_MODE"] == "off"


def test_arena_service_loads_only_scoped_json_values(tmp_path, monkeypatch) -> None:
    from scripts import run_lab_arena_service

    environment = tmp_path / "gateway.json"
    environment.write_text(
        json.dumps({"LAB_ARENA_MODE": "shadow", "SHARED_SECRET": "ignored"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("LAB_ARENA_MODE", raising=False)
    monkeypatch.delenv("SHARED_SECRET", raising=False)

    run_lab_arena_service.load_scoped_environment(environment)

    assert os.environ["LAB_ARENA_MODE"] == "shadow"
    assert "SHARED_SECRET" not in os.environ


@pytest.mark.parametrize(
    "document",
    [
        "LAB_ARENA_MODE='unterminated\n",
        "LAB_ARENA_MODE=shadow\nLAB_ARENA_MODE=live\n",
    ],
)
def test_arena_service_rejects_invalid_scoped_values_before_restore(
    tmp_path, monkeypatch, document
) -> None:
    from gateway.tee.prepare_gateway_envelopes_v2 import (
        GatewayEnvelopePreparationV2Error,
    )
    from scripts import run_lab_arena_service

    environment = tmp_path / "gateway.env"
    environment.write_text(document, encoding="utf-8")
    monkeypatch.delenv("LAB_ARENA_MODE", raising=False)

    with pytest.raises(GatewayEnvelopePreparationV2Error):
        run_lab_arena_service.load_scoped_environment(environment)

    assert "LAB_ARENA_MODE" not in os.environ


def test_normal_validator_service_checks_readiness_before_start() -> None:
    service = (ROOT / "deploy/leadpoet-arena-validator.service").read_text(encoding="utf-8")
    launcher = "scripts/run_arena_validator.py --environment-file"
    assert "ExecStartPre=/usr/bin/python3 " + launcher in service
    assert "--check-only" in service
    assert "ExecStart=/usr/bin/python3 " + launcher in service
    assert service.index("ExecStartPre=") < service.index("ExecStart=")
    assert "TimeoutStopSec=9300" in service


def test_gateway_arena_sidecar_remains_explicitly_configured() -> None:
    gateway = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert 'mode="${LAB_ARENA_MODE:-off}"' in gateway
    assert "Lab Arena service is disabled" in gateway
