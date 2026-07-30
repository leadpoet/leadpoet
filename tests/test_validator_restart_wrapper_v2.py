import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_restart_preserves_all_tracked_diffs_before_pull():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    preserve = script.index("preserving tracked local validator checkout changes")
    stash = script.index('git stash push -m "$restart_stash_message" -- .')
    fetch = script.index("git fetch origin")

    assert preserve < stash < fetch
    assert "--include-untracked" not in script[preserve:fetch]


def test_restart_allows_only_one_invocation_pinned_ancestor_commit():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert 'REQUESTED_VALIDATOR_DEPLOY_COMMIT="${VALIDATOR_DEPLOY_COMMIT:-}"' in script
    assert "unset VALIDATOR_DEPLOY_COMMIT" in script
    assert '[[ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]' in script
    assert (
        'git merge-base --is-ancestor "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" origin/main'
        in script
    )
    assert 'git checkout --detach "$REQUESTED_VALIDATOR_DEPLOY_COMMIT"' in script
    assert '"VALIDATOR_DEPLOY_COMMIT",' in script
    assert (
        'if [ -z "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \\\n'
        '    && [ "$restart_script_differs" = "1" ]; then'
        in script
    )


def test_restart_accepts_exact_commit_argument_and_rejects_conflicts():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert 'requested_commit="${1#--commit=}"' in script
    assert "unsupported validator restart argument" in script
    assert (
        '[[ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]'
        in script
    )

    invalid = subprocess.run(
        ["bash", "validator_restart.sh", "--commit", "abc123"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert invalid.returncode == 2
    assert "lowercase full 40-character SHA" in invalid.stderr
    assert "Pulling latest GitHub main" not in invalid.stdout

    conflict = subprocess.run(
        ["bash", "validator_restart.sh", "--commit", "2" * 40],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "VALIDATOR_DEPLOY_COMMIT": "1" * 40},
    )
    assert conflict.returncode == 2
    assert "--commit conflicts with VALIDATOR_DEPLOY_COMMIT" in conflict.stderr

    invalid_forward = subprocess.run(
        ["bash", "validator_restart.sh"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "VALIDATOR_COORDINATED_EXPECTED_COMMIT": "abc123",
        },
    )
    assert invalid_forward.returncode == 2
    assert "VALIDATOR_COORDINATED_EXPECTED_COMMIT" in invalid_forward.stderr
    assert "Pulling latest GitHub main" not in invalid_forward.stdout


def test_secret_hydration_cannot_replace_operator_gateway_barrier(
    tmp_path: Path,
) -> None:
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start_marker = (
        'python3 - "$SECRET_TMP" "$VALIDATOR_ENV_FILE" '
        '"$VALIDATOR_ENV_EXPORT" <<\'PY\'\n'
    )
    start = script.index(start_marker) + len(start_marker)
    hydration = script[start : script.index("\nPY\n", start)]

    secret = tmp_path / "secret"
    cache = tmp_path / "validator.env"
    export_file = tmp_path / "validator.exports"
    secret.write_text(
        "\n".join(
            (
                "VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE=/tmp/stale-marker",
                "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS=1",
                "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS=1",
                "VALIDATOR_NETUID=71",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-", str(secret), str(cache), str(export_file)],
        input=hydration,
        check=True,
        text=True,
        capture_output=True,
    )

    exports = export_file.read_text(encoding="utf-8")
    assert "VALIDATOR_NETUID" in exports
    assert "VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE" not in exports
    assert "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS" not in exports
    assert "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS" not in exports

    preserved = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -a; . \"$1\"; set +a; "
                "printf '%s\\n%s\\n%s\\n' "
                "\"$VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE\" "
                "\"$VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS\" "
                "\"$VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS\""
            ),
            "bash",
            str(export_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE": "/tmp/operator-marker",
            "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS": "600",
            "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS": "2100",
        },
    )
    assert preserved.stdout.splitlines() == [
        "/tmp/operator-marker",
        "600",
        "2100",
    ]


def _make_forward_restart_fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    repo = tmp_path / "repo"
    origin = tmp_path / "origin.git"
    repo.mkdir()
    subprocess.run(["git", "init", "--bare", str(origin)], check=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Restart Test"],
        check=True,
    )
    for relative in (
        Path("validator_restart.sh"),
        Path("Leadpoet/utils/exact_commit_restart_v2.py"),
        Path("validator_tee/scripts/verify_pinned_gateway_release_v2.sh"),
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, target)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "candidate"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", str(origin)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "push", "-u", "origin", "main"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws = bin_dir / "aws"
    aws.write_text("#!/bin/bash\nexit 88\n", encoding="utf-8")
    aws.chmod(0o755)
    return repo, commit, bin_dir


def _run_forward_restart_fixture(
    tmp_path: Path,
    *,
    expected_commit: str,
    launcher: Path,
) -> subprocess.CompletedProcess[str]:
    repo = tmp_path / "repo"
    return subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
            "VALIDATOR_ROOT": str(repo),
            "VALIDATOR_RESTART_CONTROLLER_ROOT": str(tmp_path / "controller"),
            "VALIDATOR_HOST_RESTART_SCRIPT": str(tmp_path / "installed-host.sh"),
            "VALIDATOR_ENV_FILE": str(tmp_path / "validator.env"),
            "VALIDATOR_ENV_BACKUP_DIR": str(tmp_path / "env-backups"),
            "VALIDATOR_COORDINATED_EXPECTED_COMMIT": expected_commit,
        },
    )


def test_coordinated_forward_retry_reexecs_changed_launcher_once(
    tmp_path: Path,
) -> None:
    repo, commit, _ = _make_forward_restart_fixture(tmp_path)
    installed = tmp_path / "installed-validator_restart.sh"
    installed.write_text(
        (repo / "validator_restart.sh").read_text(encoding="utf-8")
        + "\n# stale installed launcher fixture\n",
        encoding="utf-8",
    )
    installed.chmod(0o755)

    result = _run_forward_restart_fixture(
        tmp_path,
        expected_commit=commit,
        launcher=installed,
    )

    assert result.returncode == 88
    assert (
        result.stdout.count(
            "Restart wrapper updated from GitHub; re-executing latest validator_restart.sh"
        )
        == 1
    )
    assert "Preparing validator runtime env from Secrets Manager" in result.stdout


def test_coordinated_forward_rejects_moved_candidate_before_preparation(
    tmp_path: Path,
) -> None:
    repo, _, _ = _make_forward_restart_fixture(tmp_path)

    result = _run_forward_restart_fixture(
        tmp_path,
        expected_commit="f" * 40,
        launcher=repo / "validator_restart.sh",
    )

    assert result.returncode == 1
    assert "coordinated validator candidate moved" in result.stderr
    assert "Preparing validator runtime env" not in result.stdout
    assert "Stopping validator processes and containers" not in result.stdout


def test_exact_restart_preserves_newer_validator_restart_controller():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    capture = script.index("capture_validator_restart_controller")
    install = script.index("install_validator_restart_controller")
    checkout = script.index(
        'git checkout --detach "$REQUESTED_VALIDATOR_DEPLOY_COMMIT"'
    )
    assert capture < checkout
    assert install < checkout
    assert "VALIDATOR_RESTART_CONTROLLER_CURRENT" in script
    assert (
        'VALIDATOR_HOST_RESTART_SCRIPT="${VALIDATOR_HOST_RESTART_SCRIPT:-'
        '/home/ec2-user/validator_restart.sh}"'
    ) in script
    assert "VALIDATOR_EXACT_COMMIT_HELPER_SOURCE" in script


def test_exact_restart_prepares_before_same_gateway_release_activation():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    deploy = Path(
        "validator_models/containerizing/deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    verifier = Path(
        "validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
    ).read_text(encoding="utf-8")

    release_ready = script.index(
        'if [ "$VALIDATOR_V2_RELEASE_READY" != "1" ]'
    )
    shutdown = script.index('echo "Stopping validator processes and containers"')
    enclave_build = script.index("bash validator_tee/scripts/build_enclave.sh")
    hotkey = script.index("python3 -m validator_tee.host.hotkey_bootstrap_v2")
    defer_alignment = script.index(
        'echo "Deferring same-SHA gateway alignment until the exact validator '
        'application image is verified"'
    )
    legacy_alignment = script.index(
        'echo "Checking same-SHA gateway alignment before invoking the legacy '
        'deployer"'
    )
    legacy_verify = script.index(
        "if ! verify_pinned_gateway_release",
        legacy_alignment,
    )
    start = script.index('echo "Starting validator"')
    poststart_verify = script.index(
        'echo "Rechecking same-SHA gateway alignment after validator startup"'
    )
    assert (
        release_ready
        < shutdown
        < enclave_build
        < hotkey
        < defer_alignment
        < legacy_alignment
        < legacy_verify
        < start
        < poststart_verify
    )
    assert "Checking same-SHA gateway alignment before stopping validator" not in script
    assert (
        "\nif ! verify_pinned_gateway_release \\\n"
        '    "${VALIDATOR_PINNED_GATEWAY_POSTSTART_MAX_ATTEMPTS:-12}"; then\n'
    ) in script
    for endpoint in (
        "/health/v2-authority",
        "/build-info",
        "/weights/v2/release-evidence/",
    ):
        assert endpoint in verifier
    assert 'export VALIDATOR_EXACT_RELEASE_PINNED=1' in script
    assert "VALIDATOR_EXACT_RELEASE_PINNED=0" not in script
    assert "VALIDATOR_PINNED_GATEWAY_PRESTART_MAX_ATTEMPTS:-600" in script
    assert "VALIDATOR_PINNED_GATEWAY_POSTSTART_MAX_ATTEMPTS:-12" in script
    assert "verify_pinned_gateway_release_v2.sh" in script
    assert "stop_pinned_validator_after_alignment_failure" in script
    expected_check = script.index(
        "coordinated validator candidate moved before restart preparation"
    )
    runtime_env = script.index(
        'echo "Preparing validator runtime env from Secrets Manager"'
    )
    assert expected_check < runtime_env < shutdown
    assert '"VALIDATOR_COORDINATED_EXPECTED_COMMIT",' in script
    assert "VALIDATOR_GATEWAY_ACTIVATION_BARRIER_V2=1" in deploy
    image_build = deploy.index("if docker_build_validator_image; then")
    image_commit = deploy.index('IMAGE_COMMIT="$(')
    prepared_image = deploy.index('PREPARED_IMAGE_ID="$(')
    activation_barrier = deploy.index(
        'if [ "$VALIDATOR_EXACT_RELEASE_PINNED" = "1" ]; then',
        prepared_image,
    )
    active_image = deploy.index('ACTIVE_IMAGE_ID="$(', activation_barrier)
    container_cleanup = deploy.index(
        'echo "🛑 Stopping existing containers (if any)..."',
        active_image,
    )
    coordinator_start = deploy.index(
        '\nstart_container "leadpoet-validator-main"',
        container_cleanup,
    )
    assert (
        image_build
        < image_commit
        < prepared_image
        < activation_barrier
        < active_image
        < container_cleanup
        < coordinator_start
    )
    assert 'if [ "$ACTIVE_IMAGE_ID" != "$PREPARED_IMAGE_ID" ]; then' in deploy
    assert "VALIDATOR_GATEWAY_ACTIVATION_VERIFIER" in script
    assert "VALIDATOR_GATEWAY_ACTIVATION_VERIFIER" in deploy
    assert (
        '-e VALIDATOR_EXACT_RELEASE_PINNED="${VALIDATOR_EXACT_RELEASE_PINNED:-0}"'
        in deploy
    )
    assert 'git -C "$REPO_DIR" symbolic-ref -q HEAD' in deploy
    assert "pinned gateway V2 authority is not ready" in deploy
    assert "gateway_attempts = 12 if gateway_exact_release_pinned else 1" in deploy
    assert "for attempt in range(1, gateway_attempts + 1):" in deploy


def test_incomplete_late_activation_cleans_every_prepared_resource():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    destructive = script.index("VALIDATOR_DESTRUCTIVE_PHASE_STARTED=1")
    shutdown = script.index('echo "Stopping validator processes and containers"')
    activation = script.index('echo "Starting validator"')
    complete = script.index("VALIDATOR_RESTART_COMPLETED=1", activation)
    cleanup = script.index("cleanup() {")
    cleanup_end = script.index("trap cleanup EXIT", cleanup)
    cleanup_body = script[cleanup:cleanup_end]
    stop = script.index("stop_pinned_validator_after_alignment_failure()")
    stop_end = script.index("\n}\n", stop)
    stop_body = script[stop:stop_end]

    assert destructive < shutdown < activation < complete
    assert 'VALIDATOR_DESTRUCTIVE_PHASE_STARTED" = "1"' in cleanup_body
    assert 'VALIDATOR_RESTART_COMPLETED" != "1"' in cleanup_body
    assert "stop_pinned_validator_after_alignment_failure" in cleanup_body
    assert "leadpoet_release_docker_operation_lock_v2" in cleanup_body
    assert 'filter "name=leadpoet-validator"' in stop_body
    assert 'filter "name=leadpoet-qual-worker"' in stop_body
    assert 'filter "name=leadpoet-ff-worker"' in stop_body
    assert 'pkill -TERM -f ".auto_update_wrapper.sh"' in stop_body
    assert 'pkill -KILL -f ".auto_update_wrapper.sh"' in stop_body
    assert 'pkill -TERM -f "neurons/validator.py"' in stop_body
    assert 'pkill -KILL -f "neurons/validator.py"' in stop_body
    assert 'pkill -TERM -f "docker logs -f leadpoet-validator-main"' in stop_body
    assert 'pkill -KILL -f "docker logs -f leadpoet-validator-main"' in stop_body
    assert 'pkill -TERM -f "validator_tee.host.chain_relay_v2"' in stop_body
    assert 'pkill -KILL -f "validator_tee.host.chain_relay_v2"' in stop_body
    assert "validator_tee.host.chain_relay_v2" in stop_body
    assert "nitro-cli terminate-enclave --all" in stop_body
    for signal, status in (("HUP", 129), ("INT", 130), ("TERM", 143)):
        assert f"trap 'exit {status}' {signal}" in script


def test_exact_restart_checks_current_auditor_contract_before_checkout():
    gateway = Path("gw_restart.sh").read_text(encoding="utf-8")
    validator = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert "Leadpoet/utils/exact_commit_restart_v2.py" in gateway
    assert "Leadpoet/utils/exact_commit_restart_v2.py" in validator
    gateway_compatibility = gateway.index(
        "Validating exact-commit V2 rollback compatibility"
    )
    gateway_shutdown = gateway.index(
        "Stopping existing gateway and Research Lab worker processes"
    )
    validator_compatibility = validator.index(
        "Validating exact-commit V2 rollback compatibility"
    )
    validator_checkout = validator.index(
        'git checkout --detach "$REQUESTED_VALIDATOR_DEPLOY_COMMIT"'
    )
    assert gateway_compatibility < gateway_shutdown
    assert validator_compatibility < validator_checkout


def test_restart_loads_one_canonical_cutover_manifest():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert (
        'VALIDATOR_STATEFUL_CUTOVER_MANIFEST="/home/ec2-user/.config/'
        'leadpoet/stateful-epoch-cutover.json"'
    ) in script
    assert 'if [ ! -s "$VALIDATOR_STATEFUL_CUTOVER_MANIFEST" ]; then' in script
    assert 'json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))' in script
    assert 'export LEADPOET_SUBNET_EPOCH_CUTOVER_JSON' in script
    assert 'unset LEADPOET_SUBNET_EPOCH_CUTOVER_PATH' in script


def test_restart_passes_the_canonical_cutover_manifest_into_validator_container():
    restart = Path("validator_restart.sh").read_text(encoding="utf-8")
    deploy = Path(
        "validator_models/containerizing/deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    assert 'export LEADPOET_SUBNET_EPOCH_CUTOVER_JSON' in restart
    assert (
        '-e LEADPOET_SUBNET_EPOCH_CUTOVER_JSON='
        '"${LEADPOET_SUBNET_EPOCH_CUTOVER_JSON:-}"'
    ) in deploy


def test_cutover_preparation_stops_before_full_validator_and_preserves_start():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert (
        'REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY="${VALIDATOR_STATEFUL_'
        'CUTOVER_PREPARE_ONLY:-0}"'
    ) in script
    assert "stateful cutover enclave preparation requires a captured restart start" in script
    preserve = script.index(
        'if [ "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" != "1" ]; then'
    )
    delete_start = script.index('rm -f "$VALIDATOR_RESTART_START_PATH"', preserve)
    hotkey = script.index("python3 -m validator_tee.host.hotkey_bootstrap_v2")
    prepared = script.index(
        "SUCCESS: exact attested validator enclave is prepared for stateful cutover "
        "boundary capture"
    )
    exit_prepare = script.index("exit 0", prepared)
    activation_barrier = script.index(
        "VALIDATOR_GATEWAY_ACTIVATION_BARRIER_V2=1"
    )
    start_validator = script.index('echo "Starting validator"')
    final_check = script.index(
        'docker inspect -f \'{{.State.Running}}\' leadpoet-validator-main'
    )

    assert (
        preserve
        < hotkey
        < prepared
        < exit_prepare
        < activation_barrier
        < start_validator
        < final_check
        < delete_start
    )
