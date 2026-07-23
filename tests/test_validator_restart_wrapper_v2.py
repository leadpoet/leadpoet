from pathlib import Path


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
