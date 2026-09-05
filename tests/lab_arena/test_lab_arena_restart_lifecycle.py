"""Restart wiring for the optional Arena service and validator runner."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gateway_restart_replaces_and_checks_the_arena_sidecar() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    destructive = script.index("GATEWAY_DESTRUCTIVE_PHASE_STARTED=1")
    stop = script.index("stop_lab_arena_service", destructive)
    gateway_health = script.index('record_gateway_restart_timing "gateway_v2_health_ready"')
    start = script.index("start_lab_arena_service", gateway_health)
    handoff = script.index("gateway.tee.verify_weight_submission_ready_v2", start)

    assert stop < gateway_health < start < handoff
    assert "scripts/run_lab_arena_service.py" in script
    function_start = script.index("start_lab_arena_service() {")
    function_end = script.index("\n}\n", function_start)
    function = script[function_start:function_end]
    sidecar_health = function.index("http://127.0.0.1:8792/arena/v1/current")
    public_health = function.index("http://127.0.0.1:8000/arena/v1/current")
    assert sidecar_health < public_health
    assert 'case "$mode" in' in script
    assert "shadow|live" in script
    assert '--environment-file "$GATEWAY_ENV_FILE"' in function


def test_arena_service_loads_only_scoped_values(tmp_path, monkeypatch) -> None:
    from scripts import run_lab_arena_service

    environment = tmp_path / "gateway.env"
    environment.write_text(
        "LAB_ARENA_MODE=shadow\n"
        "LAB_ARENA_OPENROUTER_API_KEY=scoped-secret\n"
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
    assert os.environ["LAB_ARENA_OPENROUTER_API_KEY"] == "scoped-secret"
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "SUPABASE_SERVICE_ROLE_KEY" not in os.environ


def test_arena_service_keeps_explicit_scoped_override(tmp_path, monkeypatch) -> None:
    from scripts import run_lab_arena_service

    environment = tmp_path / "gateway.env"
    environment.write_text("LAB_ARENA_MODE=live\n", encoding="utf-8")
    monkeypatch.setenv("LAB_ARENA_MODE", "off")

    run_lab_arena_service.load_scoped_environment(environment)

    assert os.environ["LAB_ARENA_MODE"] == "off"


def test_validator_restart_replaces_runner_after_gateway_alignment() -> None:
    script = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")

    destructive = script.index(
        'record_validator_restart_timing "destructive_phase_started"'
    )
    stop = script.index("stop_lab_arena_runner", destructive)
    application = script.index('record_validator_restart_timing "validator_application_ready"')
    alignment = script.index("verify_pinned_gateway_release", application)
    start = script.index("start_lab_arena_runner", alignment)
    complete = script.index('VALIDATOR_DEPLOY_STAGE="completed"', start)

    assert stop < application < alignment < start < complete
    assert "scripts/run_lab_arena_runner.py" in script
    assert 'LAB_ARENA_WALLET_PATH="${LAB_ARENA_WALLET_PATH:-$VALIDATOR_WALLET_ROOT}"' in script
    assert 'LAB_ARENA_API_BASE_URL:-$VALIDATOR_V2_GATEWAY_URL' in script
    assert "gateway/tee/runsc-runtime.lock.json" in script


def test_restart_scripts_leave_arena_disabled_by_default() -> None:
    gateway = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    validator = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")

    assert 'mode="${LAB_ARENA_MODE:-off}"' in gateway
    assert 'mode="${LAB_ARENA_MODE:-off}"' in validator
    assert "Lab Arena service is disabled" in gateway
    assert "Lab Arena runner is disabled" in validator
