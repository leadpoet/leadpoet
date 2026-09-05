import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_restart_preserves_all_tracked_diffs_before_pull():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    preserve = script.index("preserving tracked local validator checkout changes")
    stash = script.index('git stash push -m "$restart_stash_message" -- .')
    fetch = script.index("fetch_validator_origin_with_retry", stash)

    assert preserve < stash < fetch
    assert "--include-untracked" not in script[preserve:fetch]


def _validator_fetch_functions() -> str:
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    functions = []
    for name in (
        "git_fetch_failure_is_transient",
        "fetch_validator_origin_with_retry",
    ):
        start = script.index(f"{name}() {{")
        end = script.index("\n}\n", start) + 3
        functions.append(script[start:end])
    return "\n".join(functions)


def _run_validator_fetch_fixture(
    tmp_path: Path,
    *,
    mode: str,
) -> tuple[subprocess.CompletedProcess[str], int, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count_file = tmp_path / "git-count"
    sleep_file = tmp_path / "sleeps"
    git = bin_dir / "git"
    git.write_text(
        """#!/bin/bash
set -euo pipefail
count=0
[ ! -s "$GIT_COUNT_FILE" ] || count="$(cat "$GIT_COUNT_FILE")"
count="$((count + 1))"
printf '%s\\n' "$count" > "$GIT_COUNT_FILE"
case "$GIT_FETCH_MODE" in
  transient_then_ok)
    if [ "$count" -eq 1 ]; then
      echo "error: RPC failed; HTTP 503 curl 22" >&2
      echo "fatal: expected 'acknowledgments'" >&2
      exit 1
    fi
    echo "fetch succeeded"
    ;;
  exhausted)
    echo "error: RPC failed; HTTP 503 curl 22" >&2
    exit 1
    ;;
  permanent)
    echo "remote: Repository not found" >&2
    exit 1
    ;;
esac
""",
        encoding="utf-8",
    )
    git.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$1" >> "$SLEEP_FILE"\n',
        encoding="utf-8",
    )
    sleep.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + _validator_fetch_functions() + "\nfetch_validator_origin_with_retry"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "GIT_COUNT_FILE": str(count_file),
            "GIT_FETCH_MODE": mode,
            "SLEEP_FILE": str(sleep_file),
        },
    )
    count = int(count_file.read_text(encoding="utf-8"))
    sleeps = (
        sleep_file.read_text(encoding="utf-8").splitlines()
        if sleep_file.exists()
        else []
    )
    return result, count, sleeps


def test_validator_fetch_retries_transient_transport_failure(tmp_path: Path) -> None:
    result, count, sleeps = _run_validator_fetch_fixture(
        tmp_path,
        mode="transient_then_ok",
    )

    assert result.returncode == 0
    assert count == 2
    assert sleeps == ["1"]
    assert "fetch succeeded" in result.stdout


def test_validator_fetch_does_not_retry_permanent_failure(tmp_path: Path) -> None:
    result, count, sleeps = _run_validator_fetch_fixture(
        tmp_path,
        mode="permanent",
    )

    assert result.returncode == 1
    assert count == 1
    assert sleeps == []


def test_validator_fetch_exhausts_bounded_transient_retries(tmp_path: Path) -> None:
    result, count, sleeps = _run_validator_fetch_fixture(
        tmp_path,
        mode="exhausted",
    )

    assert result.returncode == 1
    assert count == 4
    assert sleeps == ["1", "2", "4"]


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


def test_restart_uses_the_arena_runner_capacity_default():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert 'LAB_ARENA_MAX_PARALLEL_RUNS="${LAB_ARENA_MAX_PARALLEL_RUNS:-8}"' in script


def test_validator_restart_passes_bytecode_suppression_to_arena_runner():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start = script.index("start_lab_arena_runner() {")
    end = script.index("\n}\n", start) + 3
    runner = script[start:end]
    privileged_launch_start = runner.index("setsid sudo env \\")
    privileged_launch = runner[privileged_launch_start:]
    entrypoint = '"$VALIDATOR_PYTHON_BIN" -u scripts/run_lab_arena_runner.py'

    assert "PYTHONDONTWRITEBYTECODE=1" in privileged_launch
    assert privileged_launch.index("PYTHONDONTWRITEBYTECODE=1") < privileged_launch.index(
        entrypoint
    )


def test_python_bytecode_suppression_prevents_import_cache(tmp_path: Path):
    module = tmp_path / "isolated_runner_module.py"
    module.write_text("VALUE = 7\n", encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import isolated_runner_module; print(isolated_runner_module.VALUE)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "7"
    assert not list(tmp_path.glob("__pycache__/*.pyc"))


def test_unpinned_validator_local_build_follows_new_main_before_shutdown():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start = script.index("follow_superseding_validator_release() {")
    end = script.index("\n}\n", start) + 3
    follow = script[start:end]

    assert follow.index('if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ]') < follow.index(
        "restart_release_supersession_v2.py"
    )
    assert '|| [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ]' in follow
    assert follow.index("cleanup_validator_restart_preparation") < follow.index(
        'git merge --ff-only "$latest_sha"'
    )
    assert follow.index('git merge --ff-only "$latest_sha"') < follow.index(
        'bash "$VALIDATOR_ROOT/validator_restart.sh"'
    )
    assert 'LEADPOET_USE_CAPTURED_RESTART_START=1' in follow
    assert 'VALIDATOR_RELEASE_SUPERSESSION_COUNT="$next_count"' in follow

    release_build = script[
        script.index(
            'echo "Preparing exact local build inputs before production shutdown"'
        ) : script.index('record_validator_restart_timing "local_release_ready"')
    ]
    assert release_build.count("follow_superseding_validator_release") == 1
    assert release_build.index("follow_superseding_validator_release") < (
        release_build.index("gateway/tee/build_local_release_v2.sh")
    )
    acquisition_start = script.index(
        'if ! follow_superseding_validator_release; then',
        script.index('echo "Capturing the official subnet restart start before release acquisition"'),
    )
    acquisition_end = script.index("VALIDATOR_V2_MISSING_INPUTS=()", acquisition_start)
    acquisition = script[acquisition_start:acquisition_end]
    assert acquisition.count("follow_superseding_validator_release") == 4
    assert "Acquiring the exact historical attested V2 release channel" in acquisition
    assert '--expected-commit "$VALIDATOR_DEPLOY_SHA"' in acquisition
    assert '--gateway-output "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST"' in acquisition
    assert '--validator-output "$VALIDATOR_V2_RELEASE_MANIFEST"' in acquisition
    assert acquisition.index('[ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ]') < (
        acquisition.index("--ensure")
    )
    assert script.index("follow_superseding_validator_release") < script.index(
        'echo "Stopping validator processes and containers"'
    )

    alignment_start = script.index(
        "verify_forward_gateway_release_before_shutdown() {"
    )
    alignment_end = script.index("\n}\n", alignment_start) + 3
    alignment = script[alignment_start:alignment_end]
    assert 'batch_attempts=4' in alignment
    assert alignment.index("verify_pinned_gateway_release") < alignment.index(
        "follow_superseding_validator_release"
    )
    assert (
        'if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \\\n'
        '      || [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ]; then'
        in alignment
    )
    pre_shutdown = script.index(
        'VALIDATOR_DEPLOY_STAGE="pre_shutdown_gateway_alignment"'
    )
    destructive = script.index('VALIDATOR_DEPLOY_STAGE="runtime_rebuild"')
    assert "verify_forward_gateway_release_before_shutdown" in script[
        pre_shutdown:destructive
    ]


def test_secret_hydration_cannot_replace_restart_controller_state(
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
                "VALIDATOR_RESTART_INVOCATION_ID=validator-stale",
                "LEADPOET_RESTART_INVOCATION_ID=validator-stale",
                "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT=/tmp/leadpoet-stale-initial.json",
                "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT=/tmp/leadpoet-stale-final-requirements.json",
                "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT=/tmp/leadpoet-stale-final-lineage.json",
                "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS=/tmp/leadpoet-stale-recovery-requirements.json",
                "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE=/tmp/leadpoet-stale-recovery-lineage.json",
                "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS=/tmp/leadpoet-stale-stable.json",
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
    assert "VALIDATOR_RESTART_INVOCATION_ID" not in exports
    assert "LEADPOET_RESTART_INVOCATION_ID" not in exports
    assert "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" not in exports
    assert "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" not in exports
    assert "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" not in exports
    assert "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" not in exports
    assert "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE" not in exports
    assert "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" not in exports
    cached = cache.read_text(encoding="utf-8")
    assert "VALIDATOR_RESTART_INVOCATION_ID" not in cached
    assert "LEADPOET_RESTART_INVOCATION_ID" not in cached
    assert "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" not in cached
    assert "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" not in cached
    assert "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" not in cached
    assert "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" not in cached
    assert "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE" not in cached
    assert "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" not in cached

    preserved = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -a; . \"$1\"; set +a; "
                "printf '%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n' "
                "\"$VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE\" "
                "\"$VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS\" "
                "\"$VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS\" "
                "\"$VALIDATOR_RESTART_INVOCATION_ID\" "
                "\"$LEADPOET_RESTART_INVOCATION_ID\" "
                "\"$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT\" "
                "\"$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT\" "
                "\"$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT\" "
                "\"$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS\""
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
            "VALIDATOR_RESTART_INVOCATION_ID": "validator-active",
            "LEADPOET_RESTART_INVOCATION_ID": "validator-active",
            "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT": "/tmp/leadpoet-active-initial.json",
            "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT": "/tmp/leadpoet-active-final-requirements.json",
            "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT": "/tmp/leadpoet-active-final-lineage.json",
            "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS": "/tmp/leadpoet-active-stable.json",
        },
    )
    assert preserved.stdout.splitlines() == [
        "/tmp/operator-marker",
        "600",
        "2100",
        "validator-active",
        "validator-active",
        "/tmp/leadpoet-active-initial.json",
        "/tmp/leadpoet-active-final-requirements.json",
        "/tmp/leadpoet-active-final-lineage.json",
        "/tmp/leadpoet-active-stable.json",
    ]


def _handoff_environment(prefix: str) -> dict[str, str]:
    return {
        "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT": (
            f"/tmp/leadpoet-{prefix}-initial.json"
        ),
        "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT": (
            f"/tmp/leadpoet-{prefix}-final-requirements.json"
        ),
        "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT": (
            f"/tmp/leadpoet-{prefix}-final-lineage.json"
        ),
    }


def test_validator_restart_requires_safe_paired_handoff_paths_before_fetch(
    tmp_path: Path,
) -> None:
    names = set(_handoff_environment("fixture"))
    names.update(
        {
            "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS",
            "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE",
        }
    )
    clean_env = {key: value for key, value in os.environ.items() if key not in names}
    clean_env["VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED"] = "1"
    missing = subprocess.run(
        ["bash", "validator_restart.sh"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=clean_env,
    )
    assert missing.returncode == 2
    assert "paired validator active release handoff is incomplete" in missing.stderr
    assert "Pulling latest GitHub main" not in missing.stdout

    unsafe_env = _handoff_environment("fixture")
    unsafe_env["VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT"] = str(
        tmp_path / "outside-controller-root.json"
    )
    unsafe = subprocess.run(
        ["bash", "validator_restart.sh"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**clean_env, **unsafe_env},
    )
    assert unsafe.returncode == 2
    assert "one exact controller-owned /tmp/leadpoet-*.json path" in unsafe.stderr
    assert "Pulling latest GitHub main" not in unsafe.stdout

    partial_recovery_env = {
        **_handoff_environment("fixture"),
        "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS": (
            "/tmp/leadpoet-recovery-requirements.json"
        ),
    }
    partial = subprocess.run(
        ["bash", "validator_restart.sh"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**clean_env, **partial_recovery_env},
    )
    assert partial.returncode == 2
    assert "missing-runtime recovery authority is incomplete" in partial.stderr
    assert "Pulling latest GitHub main" not in partial.stdout


def test_validator_restart_uses_bounded_active_release_handoff_before_shutdown():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert (
        'VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS="${VALIDATOR_V2_GATEWAY_'
        'RELEASE_REQUIREMENTS:-/home/ec2-user/.config/leadpoet/'
        'gateway-v2-release-requirements.json}"'
    ) in script
    assert (
        'VALIDATOR_ACTIVE_PUBLICATION_JOURNAL="$VALIDATOR_ROOT/'
        'validator_weights/authoritative_weight_publication_v2.json"'
    ) in script
    module_gate = script.index(
        'VALIDATOR_ACTIVE_RELEASE_PREPARER="$VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT/gateway/tee/'
        'prepare_active_release_lineage_v2.py"'
    )
    runtime_env = script.index(
        'echo "Preparing validator runtime env from Secrets Manager"'
    )
    shutdown = script.index('echo "Stopping validator processes and containers"')
    assert module_gate < runtime_env < shutdown
    assert "selected validator release lacks bounded active release lineage support" in script[
        module_gate:runtime_env
    ]
    assert "Validator remains running; production shutdown has not started." in script[
        module_gate:runtime_env
    ]

    release_start = script.index(
        'echo "Building the exact local gateway and validator runtime identities"'
    )
    release_ready = script.index(
        'record_validator_restart_timing "local_release_ready"', release_start
    )
    release = script[release_start:release_ready]
    assert "gateway/tee/build_local_release_v2.sh" in release
    assert '--gateway-output "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST"' in release
    assert '--validator-output "$VALIDATOR_V2_RELEASE_MANIFEST"' in release
    assert "--lineage-output" not in release
    assert "--lineage-repository" not in release
    assert "--lineage-authority-commit" not in release

    running = script.index(
        "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}'",
        release_ready,
    )
    initial = script.index(
        "--phase validator-initial",
        running,
    )
    marker = script.index(
        'echo "Prepared validator active release requirements sidecar"', initial
    )
    wait = script.index(
        'while [ "$SECONDS" -lt "$VALIDATOR_ACTIVE_RELEASE_HANDOFF_DEADLINE" ]; do',
        marker,
    )
    final = script.index(
        "--phase validator-final",
        wait,
    )
    preflight = script.index(
        "python3 -m validator_tee.host.restart_preflight_v2", final
    )
    docker_guard = script.index(
        "-m validator_tee.host.docker_operation_guard_v2",
        preflight,
    )
    artifact_cleanup = script.index(
        "run_bounded_validator_restart_artifact_cleanup",
        docker_guard,
    )
    final_recheck = script.index(
        'echo "Rechecking publication journal and compact lineage immediately before validator shutdown"',
        artifact_cleanup,
    )
    second_final = script.index(
        "prepare_validator_final_active_release_lineage",
        final_recheck,
    )
    unchanged_hash_gate = script.index(
        'if [ "$(\n    sha256sum',
        second_final,
    )
    destructive = script.index("VALIDATOR_DESTRUCTIVE_PHASE_STARTED=1", preflight)
    assert release_ready < running < initial < marker < wait < final < preflight
    assert (
        preflight
        < docker_guard
        < artifact_cleanup
        < final_recheck
        < second_final
        < unchanged_hash_gate
        < destructive
    )
    assert "validator active release authority changed before shutdown" in script[
        unchanged_hash_gate:destructive
    ]

    initial_call = script[initial:marker]
    for argument in (
            '--candidate-commit "$VALIDATOR_DEPLOY_SHA"',
            '--authority-commit "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT"',
            '--restart-invocation-id "$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID"',
            '"${VALIDATOR_INITIAL_TRANSITION_ARGS[@]}"',
        '--journal "$VALIDATOR_ACTIVE_PUBLICATION_JOURNAL"',
        '--validator-hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG"',
        '--chain-signing-profile "$VALIDATOR_CHAIN_SIGNING_PROFILE"',
        '--repository "$VALIDATOR_ROOT"',
        '--lineage-id "$VALIDATOR_ANCESTRY_LINEAGE_ID"',
        '--bucket "$VALIDATOR_V2_RELEASE_BUCKET"',
        '--prefix "$VALIDATOR_V2_RELEASE_PREFIX"',
        '--requirements-output "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT"',
    ):
        assert argument in initial_call
    live_transition = script[running:initial]
    assert (
        '--running-validator-commit "$VALIDATOR_RUNNING_DEPLOY_SHA"'
        in live_transition
    )
    assert "validator missing-runtime recovery was supplied while the validator container exists" in live_transition
    assert "Error: No such object: leadpoet-validator-main" in live_transition
    assert "validator runtime state could not be established safely" in live_transition
    assert "--recovery-requirements" in live_transition
    assert "--recovery-lineage" in live_transition
    assert 'sudo chown -- "$(id -u):$(id -g)"' in initial_call

    final_call = script[final:preflight]
    for argument in (
        '--candidate-commit "$VALIDATOR_DEPLOY_SHA"',
        '--authority-commit "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT"',
        '--restart-invocation-id "$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID"',
        '--initial-requirements "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT"',
        '--final-requirements-input "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT"',
        '--lineage-input "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT"',
        '--journal "$VALIDATOR_ACTIVE_PUBLICATION_JOURNAL"',
        '--validator-hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG"',
        '--chain-signing-profile "$VALIDATOR_CHAIN_SIGNING_PROFILE"',
        '--repository "$VALIDATOR_ROOT"',
        '--lineage-id "$VALIDATOR_ANCESTRY_LINEAGE_ID"',
        '--bucket "$VALIDATOR_V2_RELEASE_BUCKET"',
        '--prefix "$VALIDATOR_V2_RELEASE_PREFIX"',
        '--requirements-output "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS"',
        '--lineage-output "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"',
    ):
        assert argument in final_call
    assert '"$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS"' in final_call
    assert '"$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"' in final_call
    assert "Validator remains running; production shutdown has not started." in script[
        marker:destructive
    ]


@pytest.mark.parametrize(
    ("inspect_message", "expected_status"),
    (
        ("\nError: No such object: leadpoet-validator-main\n", 0),
        ("\npermission denied\n", 1),
        ("\nCannot connect to the Docker daemon\n", 1),
        ("\nError: No such object: leadpoet-validator-main\nextra", 1),
    ),
)
def test_validator_restart_trims_only_outer_inspect_whitespace(
    inspect_message: str,
    expected_status: int,
) -> None:
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start = script.index(
        'VALIDATOR_RUNNING_INSPECT_ERROR="$VALIDATOR_RUNNING_INSPECT_OUTPUT"'
    )
    end = script.index(
        '  if [ "$VALIDATOR_RUNNING_INSPECT_ERROR" != ', start
    )
    normalization = script[start:end]
    result = subprocess.run(
        [
            "bash",
            "-c",
            normalization
            + "\n[ \"$VALIDATOR_RUNNING_INSPECT_ERROR\" = "
            + "\"Error: No such object: leadpoet-validator-main\" ]",
        ],
        check=False,
        env={**os.environ, "VALIDATOR_RUNNING_INSPECT_OUTPUT": inspect_message},
    )
    assert result.returncode == expected_status


def test_standalone_active_release_sidecars_use_bounded_nofollow_reads() -> None:
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start = script.index('if [ "$active_release_handoff_count" -eq 0 ]; then')
    end = script.index("\nfi\n\nfollow_superseding_validator_release()", start)
    standalone = script[start:end]

    assert "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW" in standalone
    assert "stat.S_ISREG(metadata.st_mode)" in standalone
    assert "max_document_bytes = 4 * 1024 * 1024" in standalone
    assert "os.read(fd, max_document_bytes + 1)" in standalone
    assert standalone.count("load_bounded_json(") == 4
    assert "validate_historical_compact_release_lineage_v2" in standalone
    assert ".read_text(" not in standalone
    assert ".read_bytes(" not in standalone


def test_validator_active_release_handoff_survives_reexec_and_cleans_exact_files():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    handoff_names = (
        "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT",
        "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT",
        "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT",
        "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS",
        "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE",
        "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS",
    )

    candidate_reexec_start = script.index("exec env", script.index("restart_script_differs"))
    candidate_reexec_end = script.index(
        'bash "$VALIDATOR_ROOT/validator_restart.sh" "$@"', candidate_reexec_start
    )
    candidate_reexec = script[candidate_reexec_start:candidate_reexec_end]
    supersession_start = script.index("follow_superseding_validator_release() {")
    supersession_end = script.index("\n}\n", supersession_start)
    supersession = script[supersession_start:supersession_end]
    for name in handoff_names:
        assert f'{name}="${name}"' in candidate_reexec
        assert f'{name}="${name}"' in supersession
        assert f'"{name}"' in script[script.index("skip_keys = {") : script.index("exports = []")]
    assert script.count("reset_standalone_active_release_handoff_for_reexec") == 3

    cleanup_start = script.index("cleanup_validator_restart_preparation() {")
    cleanup_end = script.index("\n}\n", cleanup_start)
    cleanup = script[cleanup_start:cleanup_end]
    assert 'rm -f -- "$handoff_path"' in cleanup
    assert "rm -rf" not in cleanup[cleanup.index("for handoff_path") :]


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
        Path("gateway/tee/prepare_active_release_lineage_v2.py"),
        Path("gateway/tee/topology.json"),
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
    include_paired_handoff: bool = True,
) -> subprocess.CompletedProcess[str]:
    repo = tmp_path / "repo"
    handoff_prefix = f"/tmp/leadpoet-{tmp_path.name}"
    environment = {
        **os.environ,
        "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
        "VALIDATOR_ROOT": str(repo),
        "VALIDATOR_RESTART_CONTROLLER_ROOT": str(tmp_path / "controller"),
        "VALIDATOR_HOST_RESTART_SCRIPT": str(tmp_path / "installed-host.sh"),
        "VALIDATOR_ENV_FILE": str(tmp_path / "validator.env"),
        "VALIDATOR_ENV_BACKUP_DIR": str(tmp_path / "env-backups"),
        "VALIDATOR_COORDINATED_EXPECTED_COMMIT": expected_commit,
    }
    if include_paired_handoff:
        environment.update(
            {
                "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT": (
                    handoff_prefix + "-initial.json"
                ),
                "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT": (
                    handoff_prefix + "-final-requirements.json"
                ),
                "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT": (
                    handoff_prefix + "-final-lineage.json"
                ),
            }
        )
    return subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=environment,
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


def test_standalone_forward_retry_rederives_internal_handoff_after_reexec(
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
        include_paired_handoff=False,
    )

    assert result.returncode == 1
    assert (
        result.stdout.count(
            "Restart wrapper updated from GitHub; re-executing latest validator_restart.sh"
        )
        == 1
    )
    assert "standalone validator compact-lineage fallback is unavailable" in result.stderr
    assert "must be one exact controller-owned" not in result.stderr


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
    assert (
        'VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT="${VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT:-$VALIDATOR_ROOT}"'
        in script
    )
    assert (
        'VALIDATOR_ACTIVE_RELEASE_PREPARER="$VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT/gateway/tee/'
        'prepare_active_release_lineage_v2.py"'
        in script
    )
    initial = script.index("--phase validator-initial")
    active_release_runner = script[
        script.index("run_validator_active_release_phase() {"):initial
    ]
    assert (
        'PYTHONPATH="$VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT"'
        in active_release_runner
    )
    assert '"$VALIDATOR_ACTIVE_RELEASE_PREPARER"' in active_release_runner
    assert "-m gateway.tee.prepare_active_release_lineage_v2" not in active_release_runner


def test_active_release_phase_cannot_load_helper_from_historical_checkout(
    tmp_path: Path,
):
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start = script.index("run_validator_active_release_phase() {")
    end = script.index("\n}\n", start) + 3
    function = script[start:end]

    controller = tmp_path / "controller"
    target = tmp_path / "target"
    for root in (controller, target):
        helper = root / "gateway" / "tee" / "prepare_active_release_lineage_v2.py"
        helper.parent.mkdir(parents=True)
        (helper.parent.parent / "__init__.py").write_text("", encoding="utf-8")
        (helper.parent / "__init__.py").write_text("", encoding="utf-8")
    controller_helper = (
        controller / "gateway" / "tee" / "prepare_active_release_lineage_v2.py"
    )
    controller_helper.write_text(
        "import sys\n"
        "print('controller-helper', *sys.argv[1:])\n",
        encoding="utf-8",
    )
    target_helper = target / "gateway" / "tee" / "prepare_active_release_lineage_v2.py"
    target_helper.write_text(
        "raise SystemExit('historical checkout helper was selected')\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text('#!/bin/bash\nexec "$@"\n', encoding="utf-8")
    sudo.chmod(0o755)
    historical_hash = "sha256:" + "a" * 64
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + function
            + '\ncd "$TARGET_ROOT"\n'
            + "run_validator_active_release_phase --phase validator-initial\n",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "TARGET_ROOT": str(target),
            "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT": str(controller),
            "VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT": str(controller),
            "VALIDATOR_ACTIVE_RELEASE_PREPARER": str(controller_helper),
            "VALIDATOR_HISTORICAL_TOPOLOGY_HASH": historical_hash,
            "VALIDATOR_PYTHON_BIN": sys.executable,
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert "controller-helper" in result.stdout
    assert f"--historical-topology-hash {historical_hash}" in result.stdout
    assert "historical checkout helper was selected" not in result.stderr


def test_active_release_phase_preserves_candidate_local_release_authority_across_sudo(
    tmp_path: Path,
):
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    start = script.index("run_validator_active_release_phase() {")
    end = script.index("\n}\n", start) + 3
    function = script[start:end]

    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    controller = tmp_path / "controller"
    helper = controller / "gateway" / "tee" / "prepare_active_release_lineage_v2.py"
    helper.parent.mkdir(parents=True)
    (helper.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    (helper.parent / "__init__.py").write_text("", encoding="utf-8")
    helper.write_text(
        "import json, os\n"
        "candidate = os.environ['LEADPOET_LOCAL_RELEASE_COMMIT_SHA']\n"
        "for name in ('LEADPOET_LOCAL_GATEWAY_RELEASE', 'LEADPOET_LOCAL_VALIDATOR_RELEASE'):\n"
        "    with open(os.environ[name], encoding='utf-8') as handle:\n"
        "        assert json.load(handle)['commit_sha'] == candidate\n"
        "print(candidate)\n",
        encoding="utf-8",
    )
    gateway_release = tmp_path / "gateway-release.json"
    validator_release = tmp_path / "validator-release.json"
    for path in (gateway_release, validator_release):
        path.write_text('{"commit_sha":"%s"}\n' % candidate, encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text('#!/bin/bash\nenv -i PATH="$PATH" "$@"\n', encoding="utf-8")
    sudo.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + function
            + "\nrun_validator_active_release_phase --phase validator-initial\n",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT": str(controller),
            "VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT": str(controller),
            "VALIDATOR_ACTIVE_RELEASE_PREPARER": str(helper),
            "VALIDATOR_HISTORICAL_TOPOLOGY_HASH": "sha256:" + "a" * 64,
            "VALIDATOR_PYTHON_BIN": sys.executable,
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
            "LEADPOET_LOCAL_RELEASE_COMMIT_SHA": candidate,
            "LEADPOET_LOCAL_GATEWAY_RELEASE": str(gateway_release),
            "LEADPOET_LOCAL_VALIDATOR_RELEASE": str(validator_release),
        },
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == candidate


def test_exact_restart_requires_gateway_before_shutdown_and_rechecks_activation():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    deploy = Path(
        "validator_models/containerizing/deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    verifier = Path(
        "validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
    ).read_text(encoding="utf-8")

    release_ready = script.index(
        'record_validator_restart_timing "local_release_ready"'
    )
    pre_shutdown_alignment = script.index(
        'echo "Checking same-SHA gateway readiness before stopping the running validator"'
    )
    pre_shutdown_verify = script.index(
        "if ! verify_forward_gateway_release_before_shutdown",
        pre_shutdown_alignment,
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
        < pre_shutdown_alignment
        < pre_shutdown_verify
        < shutdown
        < enclave_build
        < hotkey
        < defer_alignment
        < legacy_alignment
        < legacy_verify
        < start
        < poststart_verify
    )
    assert (
        'echo "Validator remains running; production shutdown has not started." >&2'
        in script[pre_shutdown_verify:shutdown]
    )
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
    assert "VALIDATOR_PINNED_GATEWAY_PRESTART_MAX_ATTEMPTS:-3000" in script
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


def test_validator_restart_records_nonblocking_commit_bound_stage_timings():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert "leadpoet.validator_restart_timing.v1" in script
    assert 'record_validator_restart_timing "invoked"' in script
    assert 'record_validator_restart_timing "local_release_ready"' in script
    assert (
        'record_validator_restart_timing "pre_shutdown_checks_complete"'
        in script
    )
    assert (
        'record_validator_restart_timing "destructive_phase_started"'
        in script
    )
    assert (
        'record_validator_restart_timing "attested_enclave_ready"' in script
    )
    assert (
        'record_validator_restart_timing "validator_application_ready"'
        in script
    )
    assert 'record_validator_restart_timing "completed" "passed"' in script
    assert (
        "WARNING: validator restart timing event could not be recorded"
        in script
    )
    assert (
        'VALIDATOR_RESTART_TIMING_INITIALIZED="'
        '$VALIDATOR_RESTART_TIMING_INITIALIZED" \\'
    ) in script


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
