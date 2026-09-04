from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from Leadpoet.utils import restart_epoch_gate
from Leadpoet.utils.restart_epoch_gate import (
    MAXIMUM_RESTART_EPOCH_BLOCK,
    RestartEpochGateError,
    verify_captured_restart_epoch_start,
    verify_restart_epoch_window,
    write_restart_epoch_start,
)
from Leadpoet.utils.subnet_epoch import SubnetEpochError, SubnetEpochSnapshot


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


def test_gateway_and_validator_use_protocol_compatibility_not_history_floor() -> None:
    gateway = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    validator = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")

    for source in (gateway, validator):
        assert "exact_commit_restart_v2.py" in source
        assert "--branch-ref origin/main" in source
        assert "V2_DEPLOYMENT_COMPATIBILITY_FLOOR_SHA" not in source
        assert "predates the supported stateful V2 rollback floor" not in source
        assert "--compatibility-floor" not in source


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


def test_restart_gate_retries_transient_epoch_read_with_fresh_connection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    subtensors = []
    sleeps = []

    class Subtensor:
        def __init__(self, *, network):
            self.network = network
            self.closed = False
            subtensors.append(self)

        def close(self):
            self.closed = True

    def read_snapshot(subtensor, *, netuid):
        assert netuid == 71
        if subtensor is subtensors[0]:
            raise SubnetEpochError(
                "block_hash must be a 32-byte lowercase hex hash"
            )
        return _snapshot(200)

    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(Subtensor=Subtensor),
    )
    monkeypatch.setattr(
        restart_epoch_gate,
        "read_subnet_epoch_snapshot",
        read_snapshot,
    )
    monkeypatch.setattr(
        restart_epoch_gate.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    assert restart_epoch_gate.main(["--network", "finney"]) == 0

    assert len(subtensors) == 2
    assert all(subtensor.closed for subtensor in subtensors)
    assert sleeps == [restart_epoch_gate.RESTART_EPOCH_RETRY_DELAY_SECONDS]
    assert '"restart_allowed": true' in capsys.readouterr().out


def test_restart_gate_retries_connection_timeout_via_pinned_archive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    networks = []
    subtensors = []
    sleeps = []

    class Subtensor:
        def __init__(self, *, network):
            networks.append(network)
            if len(networks) == 1:
                raise TimeoutError("timed out")
            self.closed = False
            subtensors.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(Subtensor=Subtensor),
    )
    monkeypatch.setattr(
        restart_epoch_gate,
        "read_subnet_epoch_snapshot",
        lambda _subtensor, *, netuid: _snapshot(200),
    )
    monkeypatch.setattr(
        restart_epoch_gate.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    assert restart_epoch_gate.main(["--network", "finney"]) == 0

    assert networks == [
        "finney",
        restart_epoch_gate.OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    ]
    assert len(subtensors) == 1
    assert subtensors[0].closed is True
    assert sleeps == [restart_epoch_gate.RESTART_EPOCH_RETRY_DELAY_SECONDS]
    captured = capsys.readouterr()
    assert "Transient official subnet epoch connection failed" in captured.err
    assert '"restart_allowed": true' in captured.out


def test_restart_gate_fails_closed_after_bounded_transient_read_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensors = []

    class Subtensor:
        def __init__(self, *, network):
            self.network = network
            self.closed = False
            subtensors.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(Subtensor=Subtensor),
    )

    def fail_read(_subtensor, *, netuid):
        assert netuid == 71
        raise SubnetEpochError("temporary malformed head")

    monkeypatch.setattr(
        restart_epoch_gate,
        "read_subnet_epoch_snapshot",
        fail_read,
    )
    monkeypatch.setattr(restart_epoch_gate.time, "sleep", lambda _seconds: None)

    with pytest.raises(SubnetEpochError, match="temporary malformed head"):
        restart_epoch_gate.main(["--network", "finney"])

    assert len(subtensors) == restart_epoch_gate.RESTART_EPOCH_READ_ATTEMPTS
    assert all(subtensor.closed for subtensor in subtensors)


def test_restart_gate_does_not_retry_policy_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtensors = []

    class Subtensor:
        def __init__(self, *, network):
            self.network = network
            self.closed = False
            subtensors.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(Subtensor=Subtensor),
    )
    monkeypatch.setattr(
        restart_epoch_gate,
        "read_subnet_epoch_snapshot",
        lambda subtensor, *, netuid: _snapshot(301),
    )

    with pytest.raises(RestartEpochGateError, match="observed 301"):
        restart_epoch_gate.main(["--network", "finney"])

    assert len(subtensors) == 1
    assert subtensors[0].closed is True


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


def test_captured_restart_start_survives_epoch_transition_and_pruned_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _snapshot(300)
    current = replace(
        _snapshot(40),
        current_block=10_400,
        last_epoch_block=10_360,
        subnet_epoch_index=124,
    )
    report = {
        "schema_version": "leadpoet.restart_epoch_start.v1",
        "maximum_restart_epoch_block": 300,
        "restart_allowed": True,
        "snapshot": captured.to_dict(),
    }
    path = tmp_path / "restart-start.json"
    write_restart_epoch_start(path, report)

    primary = object()

    class Archive:
        closed = False

        def close(self):
            self.closed = True

    archive = Archive()

    def read_snapshot(subtensor, *, netuid, block_hash=None):
        assert netuid == 71
        if block_hash is not None:
            if subtensor is primary:
                raise SubnetEpochError("primary state was pruned")
            assert subtensor is archive
            return captured
        assert subtensor is primary
        return current

    monkeypatch.setattr(restart_epoch_gate, "read_subnet_epoch_snapshot", read_snapshot)
    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(
            Subtensor=lambda *, network: (
                archive
                if network
                == restart_epoch_gate.OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
                else None
            )
        ),
    )

    result = verify_captured_restart_epoch_start(
        primary,
        path=path,
        netuid=71,
    )

    assert result["captured_epoch_block"] == 300
    assert result["current_snapshot"]["subnet_epoch_index"] == 124
    assert result["deadline_reapplied"] is False
    assert archive.closed is True


def test_gateway_captures_start_gate_and_validator_gates_before_shutdown() -> None:
    gateway = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    validator = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")

    gateway_gate = gateway.index("Leadpoet.utils.restart_epoch_gate")
    gateway_shutdown = gateway.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    gateway_release = gateway.index(
        "gateway/tee/build_local_release_v2.sh",
        gateway_gate,
    )
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
    assert (
        'PUBLICATION_JOURNAL_ARG="-e VALIDATOR_V2_PUBLICATION_JOURNAL_PATH='
        '/app/validator_weights/authoritative_weight_publication_v2.json"'
        in deploy
    )
    assert "$PUBLICATION_JOURNAL_ARG" in deploy
    assert "get_hotkey_state_v2" in deploy
    assert "/health/v2-authority" in deploy
    assert "read_subnet_epoch_snapshot" in deploy
    assert "RestartCount" in deploy
    assert "VALIDATOR_DESTRUCTIVE_PHASE_STARTED=1" in restart
    assert "VALIDATOR_RESTART_COMPLETED=1" in restart
    assert "Cleaning incomplete validator activation" in restart
    assert "stop_pinned_validator_after_alignment_failure" in restart
    assert "Rechecking same-SHA gateway alignment after validator startup" in restart


def test_validator_cutover_preparation_keeps_normal_activation_guard() -> None:
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    epoch_runtime = (ROOT / "gateway" / "utils" / "epoch.py").read_text(
        encoding="utf-8"
    )

    prepare = restart.index(
        'if [ "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" = "1" ]; then'
    )
    bootstrap = restart.index("python3 -m validator_tee.host.runtime_v2_bootstrap")
    hotkey = restart.index("python3 -m validator_tee.host.hotkey_bootstrap_v2")
    start_validator = restart.index('echo "Starting validator"')

    assert bootstrap < hotkey < prepare < start_validator
    assert "configured cutover has not been explicitly activated" in epoch_runtime


def test_validator_prepares_without_live_gateway_but_requires_it_for_activation() -> None:
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    build = deploy.index("if docker_build_validator_image; then")
    commit_label = deploy.index('IMAGE_COMMIT="$(', build)
    prepared_image = deploy.index('PREPARED_IMAGE_ID="$(', commit_label)
    activation_barrier = deploy.index(
        'if [ "$VALIDATOR_EXACT_RELEASE_PINNED" = "1" ]; then',
        prepared_image,
    )
    gateway_verify = deploy.index(
        'bash "$ACTIVATION_VERIFIER"',
        activation_barrier,
    )
    image_recheck = deploy.index('ACTIVE_IMAGE_ID="$(', gateway_verify)
    coordinator = deploy.index(
        '\nstart_container "leadpoet-validator-main"',
        image_recheck,
    )

    assert (
        build
        < commit_label
        < prepared_image
        < activation_barrier
        < gateway_verify
        < image_recheck
        < coordinator
    )
    assert "VALIDATOR_GATEWAY_ACTIVATION_BARRIER_V2=1" in deploy
    assert "VALIDATOR_V2_GATEWAY_URL is required for pinned activation" in deploy
    assert 'if [ "$ACTIVE_IMAGE_ID" != "$PREPARED_IMAGE_ID" ]; then' in deploy
    assert 'gateway_authority_status = "deferred"' in deploy
    assert 'gateway_authority_status = "not_aligned"' in deploy
    assert '"gateway_authority_status": gateway_authority_status' in deploy
    assert (
        'os.environ.get("VALIDATOR_EXACT_RELEASE_PINNED") == "1"'
        in deploy
    )
    assert "pinned gateway V2 authority is not ready" in deploy


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


def test_validator_runtime_never_logs_proxy_userinfo() -> None:
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    validator = (ROOT / "neurons" / "validator.py").read_text(encoding="utf-8")
    checks = (
        ROOT / "validator_models" / "checks_utils.py"
    ).read_text(encoding="utf-8")

    assert "Proxy: ${PROXY_URL" not in deploy
    assert "Proxy: ${QUAL_PROXY_VALUE" not in deploy
    assert "Proxy: ${FF_PROXY_VALUE" not in deploy
    assert "Proxy: {proxy_url" not in validator
    assert "Proxy enabled: {HTTP_PROXY_URL" not in checks
    assert deploy.count("Proxy: configured (credentials redacted)") == 2
