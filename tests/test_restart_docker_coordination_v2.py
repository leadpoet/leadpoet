from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_holds_shared_docker_lock_only_across_docker_restart_work() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    acquire = script.index("leadpoet_acquire_docker_operation_lock_v2")
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    enclave = script.index("bash ./start_enclave.sh")
    release = script.index("leadpoet_release_docker_operation_lock_v2")
    launch = script.index(
        'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main'
    )

    assert acquire < shutdown < enclave < release < launch
    assert "wait_for_gateway_build_memory" in script
    assert "--watch-parent" not in script
    assert "PYTHONSAFEPATH=1 LEADPOET_REPO_ROOT=" in script


def test_validator_waits_for_shared_host_before_shutdown_and_retries_only_build_transport() -> None:
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    gate = restart.index("Validating the official restart start")
    acquire = restart.index("leadpoet_acquire_docker_operation_lock_v2")
    shutdown = restart.index('echo "Stopping validator processes and containers"')
    final_check = restart.index(
        "validator coordinator failed its final restart-wrapper check"
    )
    release = restart.index("leadpoet_release_docker_operation_lock_v2")

    assert gate < acquire < shutdown < final_check < release
    assert "docker_operation_guard_v2" in restart
    assert "7>&- &" in restart
    assert "leadpoet_run_docker_build_with_retry_v2" in deploy
    assert "pkill -TERM" not in deploy
    assert "pkill -KILL" not in deploy


def test_attestation_builds_share_the_same_host_docker_lock() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "attested-v2-release.yml"
    ).read_text(encoding="utf-8")
    reclaim = (
        ROOT / "validator_tee" / "scripts" / "reclaim_docker_storage_v2.sh"
    ).read_text(encoding="utf-8")
    validator_build = (
        ROOT / "validator_tee" / "scripts" / "build_enclave.sh"
    ).read_text(encoding="utf-8")
    gateway_build = (
        ROOT / "gateway" / "tee" / "build_role_enclaves.sh"
    ).read_text(encoding="utf-8")

    assert workflow.count(
        ". validator_tee/scripts/docker_operation_lock_v2.sh"
    ) == 2
    assert workflow.count("leadpoet_acquire_docker_operation_lock_v2") == 2
    assert "--watch-parent" not in workflow
    assert reclaim.index("leadpoet_acquire_docker_operation_lock_v2") < reclaim.index(
        "docker image prune --all --force"
    )
    assert "docker_operation_guard_v2" in reclaim
    assert "leadpoet_acquire_docker_operation_lock_v2" in validator_build
    assert "leadpoet_acquire_docker_operation_lock_v2" in gateway_build
