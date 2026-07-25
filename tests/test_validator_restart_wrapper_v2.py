import os
from pathlib import Path
import subprocess


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


def test_pinned_restart_requires_same_gateway_release_before_shutdown():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    deploy = Path(
        "validator_models/containerizing/deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    verifier = Path(
        "validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
    ).read_text(encoding="utf-8")

    prestart_verify = script.index(
        'echo "Checking same-SHA gateway alignment before stopping validator"'
    )
    release_ready = script.index(
        'if [ "$VALIDATOR_V2_RELEASE_READY" != "1" ]'
    )
    shutdown = script.index('echo "Stopping validator processes and containers"')
    start = script.index('echo "Starting validator"')
    poststart_verify = script.index(
        'echo "Rechecking same-SHA gateway alignment after validator startup"'
    )
    assert release_ready < prestart_verify < shutdown < start < poststart_verify
    assert "\n  verify_pinned_gateway_release\n" in script
    assert "\n  if ! verify_pinned_gateway_release; then\n" in script
    for endpoint in (
        "/health/v2-authority",
        "/build-info",
        "/weights/v2/release-evidence/",
    ):
        assert endpoint in verifier
    assert 'export VALIDATOR_EXACT_RELEASE_PINNED=1' in script
    assert "verify_pinned_gateway_release_v2.sh" in script
    assert "stop_pinned_validator_after_alignment_failure" in script
    assert "if ! verify_pinned_gateway_release; then" in script
    assert (
        '-e VALIDATOR_EXACT_RELEASE_PINNED="${VALIDATOR_EXACT_RELEASE_PINNED:-0}"'
        in deploy
    )
    assert 'git -C "$REPO_DIR" symbolic-ref -q HEAD' in deploy
    assert "pinned gateway V2 authority is not ready" in deploy
    assert "gateway_attempts = 12 if gateway_exact_release_pinned else 1" in deploy
    assert "for attempt in range(1, gateway_attempts + 1):" in deploy


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
    start_validator = script.index('echo "Starting validator"')
    final_check = script.index(
        'docker inspect -f \'{{.State.Running}}\' leadpoet-validator-main'
    )

    assert (
        preserve
        < hotkey
        < prepared
        < exit_prepare
        < start_validator
        < final_check
        < delete_start
    )
