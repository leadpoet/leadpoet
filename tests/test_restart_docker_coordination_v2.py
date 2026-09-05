from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_holds_shared_docker_lock_through_authority_repair() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    acquire = script.index(
        "leadpoet_acquire_docker_operation_lock_v2",
        script.index('DOCKER_LOCK_HELPER="$GATEWAY_PREFLIGHT_TREE'),
    )
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    enclave = script.index("bash ./start_enclave.sh")
    repair = script.index(
        "\nrepair_chain_settlements_and_prepare_current_weight_input\n",
        enclave,
    )
    release = script.index("leadpoet_release_docker_operation_lock_v2", repair)
    cleanup_stage = script.index('GATEWAY_DEPLOY_STAGE="docker_disk_cleanup"')
    cleanup = script.index("run_bounded_restart_artifact_cleanup", cleanup_stage)
    prune = script.index("sudo docker system prune -af --volumes", cleanup)
    launch = script.index(
        'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main'
    )

    assert acquire < shutdown < enclave < repair < release < launch
    assert shutdown < cleanup < prune < enclave
    emergency = script.index("emergency_disk_preflight()")
    emergency_end = script.index(
        "acquire_gateway_restart_lock()", emergency
    )
    emergency_source = script[emergency:emergency_end]
    assert "validator_tee/scripts/reclaim_docker_storage_v2.sh" in emergency_source
    assert "sudo docker system prune" not in emergency_source
    assert (
        emergency_source.index("leadpoet_acquire_docker_operation_lock_v2")
        < emergency_source.index("run_bounded_restart_artifact_cleanup")
        < emergency_source.index("reclaim_docker_storage_v2.sh")
        < emergency_source.index("leadpoet_release_docker_operation_lock_v2")
    )
    reset = script.index("reset_orphaned_docker_storage_if_needed()")
    reset_end = script.index("ensure_docker_ready()", reset)
    reset_source = script[reset:reset_end]
    assert "validator_tee/scripts/reclaim_docker_storage_v2.sh" in reset_source
    assert "systemctl stop docker" not in reset_source
    assert "sudo rm -rf /tmp/research-lab-*" not in script
    assert (
        '--docker-lock-owner-pid "${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-$$}"'
        in script
    )
    assert '--gateway-eif-root "$GATEWAY_TEE_EIF_ROOT"' in script
    assert "wait_for_gateway_build_memory" in script
    assert "--watch-parent" not in script
    assert "PYTHONSAFEPATH=1 LEADPOET_REPO_ROOT=" in script
    assert script.count(
        "7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- &"
    ) == 2


def test_validator_holds_shared_host_lock_through_late_activation_barrier() -> None:
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    gate = restart.index("Validating the official restart start")
    acquire = restart.index("leadpoet_acquire_docker_operation_lock_v2")
    guard = restart.index("docker_operation_guard_v2", acquire)
    cleanup = restart.index("run_bounded_validator_restart_artifact_cleanup", guard)
    shutdown = restart.index('echo "Stopping validator processes and containers"')
    final_check = restart.index(
        "validator coordinator failed its final restart-wrapper check"
    )
    release = restart.index(
        "leadpoet_release_docker_operation_lock_v2",
        final_check,
    )
    app_lock = deploy.index("leadpoet_acquire_docker_operation_lock_v2")
    app_build = deploy.index("if docker_build_validator_image; then")
    image_id = deploy.index('PREPARED_IMAGE_ID="$(', app_build)
    activation_barrier = deploy.index(
        'if [ "$VALIDATOR_EXACT_RELEASE_PINNED" = "1" ]; then',
        image_id,
    )
    active_image_id = deploy.index('ACTIVE_IMAGE_ID="$(', activation_barrier)
    coordinator = deploy.index(
        '\nstart_container "leadpoet-validator-main"',
        active_image_id,
    )

    assert gate < acquire < guard < cleanup < shutdown < final_check < release
    assert (
        app_lock
        < app_build
        < image_id
        < activation_barrier
        < active_image_id
        < coordinator
    )
    assert "VALIDATOR_DOCKER_LOCK_ACQUIRED=1" in restart
    assert "VALIDATOR_DOCKER_LOCK_ACQUIRED=0" in restart
    assert "Cleaning incomplete validator activation" in restart
    assert "leadpoet_release_docker_operation_lock_v2 || true" in restart
    assert "docker_operation_guard_v2" in restart
    assert (
        '--docker-lock-owner-pid "${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-$$}"'
        in restart
    )
    assert "7>&- 8>&- &" in restart
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
    pcr0_builder = (
        ROOT / "gateway" / "utils" / "pcr0_builder.py"
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
    assert (
        "/home/ec2-user/.config/leadpoet/docker-operation-v2.lock"
        in pcr0_builder
    )
    assert pcr0_builder.count(
        "async with _docker_operation_lock_scope()"
    ) == 1
    assert (
        "async with _docker_operation_lock_scope(\n"
        "                opportunistic=commit_hash not in required_commit_hashes,\n"
        "            ):"
        in pcr0_builder
    )
