from __future__ import annotations

from pathlib import Path

import pytest

from Leadpoet.utils import restart_epoch_gate
from Leadpoet.utils.restart_epoch_gate import (
    MAXIMUM_RESTART_EPOCH_BLOCK,
    RestartEpochGateError,
    verify_captured_restart_epoch_start,
    verify_restart_epoch_window,
    write_restart_epoch_start,
)
from Leadpoet.utils.subnet_epoch import SubnetEpochSnapshot


ROOT = Path(__file__).resolve().parents[1]


def _snapshot(epoch_block: int) -> SubnetEpochSnapshot:
    return SubnetEpochSnapshot(
        network_genesis_hash="1" * 64,
        netuid=71,
        head_kind="best",
        block_hash="2" * 64,
        current_block=10_000 + epoch_block,
        last_epoch_block=10_000,
        pending_epoch_at=0,
        subnet_epoch_index=123,
        tempo=360,
        blocks_since_last_step=epoch_block,
        observed_at="2026-07-18T00:00:00Z",
    )


@pytest.mark.parametrize("epoch_block", [0, 299, 300])
def test_restart_gate_accepts_official_epoch_block_at_or_before_300(
    monkeypatch: pytest.MonkeyPatch,
    epoch_block: int,
) -> None:
    monkeypatch.setattr(
        restart_epoch_gate,
        "read_subnet_epoch_snapshot",
        lambda subtensor, *, netuid: _snapshot(epoch_block),
    )

    result = verify_restart_epoch_window(object(), netuid=71)

    assert MAXIMUM_RESTART_EPOCH_BLOCK == 300
    assert result["epoch_block"] == epoch_block
    assert result["restart_allowed"] is True


def test_restart_gate_rejects_official_epoch_block_after_300(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        restart_epoch_gate,
        "read_subnet_epoch_snapshot",
        lambda subtensor, *, netuid: _snapshot(301),
    )

    with pytest.raises(RestartEpochGateError, match="observed 301"):
        verify_restart_epoch_window(object(), netuid=71)


def test_captured_restart_start_is_not_rechecked_after_block_300(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _snapshot(250)
    current = _snapshot(330)
    report = {
        "schema_version": "leadpoet.restart_epoch_start.v1",
        "maximum_restart_epoch_block": 300,
        "restart_allowed": True,
        "snapshot": captured.to_dict(),
    }
    path = tmp_path / "restart-start.json"
    write_restart_epoch_start(path, report)

    def read_snapshot(_subtensor, *, netuid, block_hash=None):
        assert netuid == 71
        return captured if block_hash is not None else current

    monkeypatch.setattr(restart_epoch_gate, "read_subnet_epoch_snapshot", read_snapshot)

    result = verify_captured_restart_epoch_start(
        object(),
        path=path,
        netuid=71,
    )

    assert result["captured_epoch_block"] == 250
    assert result["current_epoch_block"] == 330
    assert result["deadline_reapplied"] is False


def test_gateway_captures_start_gate_and_validator_gates_before_shutdown() -> None:
    gateway = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    validator = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")

    gateway_gate = gateway.index("Leadpoet.utils.restart_epoch_gate")
    gateway_shutdown = gateway.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    gateway_release = gateway.index("gateway.tee.release_channel_v2")
    validator_gate = validator.index("Leadpoet.utils.restart_epoch_gate")
    validator_shutdown = validator.index(
        'echo "Stopping validator processes and containers"'
    )

    assert gateway_gate < gateway_release < gateway_shutdown
    assert validator_gate < validator_shutdown
    assert '--captured-report "$GATEWAY_RESTART_START_PATH"' in gateway
    assert '--captured-report "$VALIDATOR_RESTART_START_PATH"' in validator
    assert "MAXIMUM_RESTART_EPOCH_BLOCK = 300" in (
        ROOT / "Leadpoet" / "utils" / "restart_epoch_gate.py"
    ).read_text(encoding="utf-8")
    assert "--maximum" not in gateway
    assert "--maximum" not in validator


def test_validator_restart_is_fail_closed_and_postflight_verified() -> None:
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    validator = (ROOT / "neurons" / "validator.py").read_text(encoding="utf-8")
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    assert "VALIDATOR_FORCE_CONTAINER_DEPLOY=1" in restart
    assert "VALIDATOR_AUTO_CONTAINER_FOLLOW_LOGS=0" in restart
    assert "fail_closed_container_deploy" in validator
    assert "containerized validator deployment failed" in validator
    assert "Authoritative validator coordinator runtime verified" in deploy
    assert "VALIDATOR_V2_DEPLOY_COMMIT" in deploy
    assert "VALIDATOR_WEIGHT_PROTOCOL" in deploy
    assert "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON" in deploy
    assert "LEADPOET_EPOCH_MODE" not in deploy
    assert "get_hotkey_state_v2" in deploy
    assert "/health/v2-authority" in deploy
    assert "read_subnet_epoch_snapshot" in deploy
    assert "RestartCount" in deploy


def test_validator_restart_does_not_require_a_live_gateway() -> None:
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    assert 'gateway_authority_status = "deferred"' in deploy
    assert 'gateway_authority_status = "not_aligned"' in deploy
    assert '"gateway_authority_status": gateway_authority_status' in deploy
    assert 'raise SystemExit("VALIDATOR_V2_GATEWAY_URL is missing")' not in deploy
    assert "gateway V2 authority is not ready" not in deploy


def test_validator_secret_environment_overrides_local_fallback_files() -> None:
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    capture = deploy.index('INHERITED_ENV_FILE="$(mktemp')
    main_env = deploy.index('source "$MAIN_ENV_PATH"')
    docker_env = deploy.index("source .env.docker")
    restore = deploy.index('source "$INHERITED_ENV_FILE"')
    first_container = deploy.index("docker run -d")

    assert capture < main_env < docker_env < restore < first_container
    assert "destination.chmod(0o600)" in deploy
    assert 'rm -f "$INHERITED_ENV_FILE"' in deploy
