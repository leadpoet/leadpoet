from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import pytest

from gateway.tee import build_identity


ROOT = Path(__file__).resolve().parents[1]


def _ordered_offsets(text: str, markers: tuple[str, ...]) -> list[int]:
    offsets = [text.index(marker) for marker in markers]
    assert offsets == sorted(offsets)
    return offsets


def _shell_function_source(script: str, name: str) -> str:
    lines = script.splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated shell function: {name}")


def test_gateway_restart_drains_arena_claims_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    drain = script.index('drain_lab_arena_for_restart "$GATEWAY_PREFLIGHT_TREE"')
    authorize = script.index("--phase gateway_destructive", drain)
    destructive = script.index("GATEWAY_DESTRUCTIVE_PHASE_STARTED=1", authorize)
    ready = script.index("--phase gateway_ready", destructive)

    assert drain < authorize < destructive < ready
    assert "abort_lab_arena_restart_guard_before_destructive" in _shell_function_source(
        script, "on_gateway_restart_exit"
    )
    assert "-u GATEWAY_ACTIVE_RELEASE_COMPONENT" in script


def _run_post_activate_guard_reexec(
    tmp_path: Path, *, candidate: str, generation: str,
    plan_candidate: str | None = None, head_candidate: str | None = None,
    invocation: str = "restart-fixture",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    start = script.index('GATEWAY_DEPLOY_STAGE="restart_reexec"')
    end = script.index('\nfi\n\nbind_activated_gateway_guard_candidate', start)
    reexec = script[start:end]
    validate = _shell_function_source(
        script, "validate_post_activate_arena_guard_authority"
    )
    bind_candidate = _shell_function_source(
        script, "bind_activated_gateway_guard_candidate"
    )
    run_guard = _shell_function_source(script, "run_lab_arena_restart_guard")
    initialization = "\n".join((
        'PREPARED_GATEWAY_SHA="${PREPARED_GATEWAY_SHA:-}"',
        'LAB_ARENA_RESTART_GUARD_GENERATION="${LAB_ARENA_RESTART_GUARD_GENERATION:-}"',
    ))
    assert initialization in script

    authority = tmp_path / "authority"
    helper = authority / "scripts" / "lab_arena_restart_claim_guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# exact test helper\n", encoding="utf-8")
    helper_output = tmp_path / "helper-argv"
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_HELPER_OUTPUT\"\n"
        "printf '%s\\n' '{}'\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    canonical_env = tmp_path / "gateway.env"
    canonical_env.write_text(
        "LAB_ARENA_SUPABASE_URL=https://arena.invalid\n"
        "LAB_ARENA_SUPABASE_ANON_KEY=test-anon\n"
        "LAB_ARENA_SERVICE_KEY=sb_secret_test\n",
        encoding="utf-8",
    )
    target = tmp_path / "post-activate-target.sh"
    target.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + f"ENV_CLONE={shlex.quote(str(tmp_path / 'missing-env-clone'))}\n"
        + f"GATEWAY_ENV_FILE={shlex.quote(str(canonical_env))}\n"
        + initialization
        + "\n"
        + validate
        + "\n"
        + bind_candidate
        + "\n"
        + run_guard
        + "\n"
        + "deployment_field() {\n"
        + '  case "$1" in target_sha) printf \'%s\\n\' "$FAKE_PLAN_SHA" ;; '
        + "branch) printf '%s\\n' main ;; remote_url) printf '%s\\n' test ;; esac\n"
        + "}\n"
        + "git() { printf '%s\\n' \"$FAKE_HEAD_SHA\"; }\n"
        + "validate_post_activate_arena_guard_authority\n"
        + "bind_activated_gateway_guard_candidate\n"
        + 'test "$GATEWAY_ACTIVE_RELEASE_COMPONENT" = all\n'
        + 'run_lab_arena_restart_guard "$GATEWAY_RESTART_AUTHORITY_ROOT" ready '
        + '--generation "$LAB_ARENA_RESTART_GUARD_GENERATION" '
        + '--phase gateway_ready >/dev/null\n'
        + "printf '%s\\n' \"$PREPARED_GATEWAY_SHA\" "
        + "\"$LAB_ARENA_RESTART_GUARD_GENERATION\" "
        + "\"$GATEWAY_ACTIVE_RELEASE_COMPONENT\"\n",
        encoding="utf-8",
    )
    target.chmod(0o755)

    referenced = sorted(set(re.findall(r"\$([A-Z][A-Z0-9_]*)", reexec)))
    values = {name: "fixture" for name in referenced}
    values.update({
        "GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT": str(target),
        "GATEWAY_RESTART_AUTHORITY_ROOT": str(authority),
        "GATEWAY_PYTHON_BIN": str(python),
        "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID": invocation,
        "GATEWAY_ACTIVE_RELEASE_COMPONENT": "all",
        "PREPARED_GATEWAY_SHA": candidate,
        "LAB_ARENA_RESTART_GUARD_GENERATION": generation,
    })
    assignments = "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in values.items()
    )
    driver = tmp_path / "reexec-driver.sh"
    driver.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + assignments
        + "\nexport LAB_ARENA_SUPABASE_URL=https://arena.invalid\n"
        + "export LAB_ARENA_SUPABASE_ANON_KEY=test-anon\n"
        + "export LAB_ARENA_SERVICE_KEY=test-service\n"
        + f"export FAKE_HELPER_OUTPUT={shlex.quote(str(helper_output))}\n"
        + f"export FAKE_PLAN_SHA={shlex.quote(plan_candidate if plan_candidate is not None else candidate)}\n"
        + f"export FAKE_HEAD_SHA={shlex.quote(head_candidate if head_candidate is not None else candidate)}\n"
        + reexec
        + "\n",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(driver)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed, helper_output


def test_gateway_post_activate_reexec_preserves_exact_arena_guard_authority(
    tmp_path: Path,
) -> None:
    candidate = "a" * 40
    completed, helper_output = _run_post_activate_guard_reexec(
        tmp_path, candidate=candidate, generation="7"
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [candidate, "7", "all"]
    helper_argv = helper_output.read_text(encoding="utf-8")
    assert f"--candidate\n{candidate}\n" in helper_argv
    assert "--invocation\nrestart-fixture\n" in helper_argv
    assert "--generation\n7\n" in helper_argv
    assert "--phase\ngateway_ready\n" in helper_argv


@pytest.mark.parametrize(
    ("candidate", "generation", "diagnostic"),
    [
        ("", "7", "guard candidate is invalid"),
        ("not-a-commit", "7", "guard candidate is invalid"),
        ("a" * 40, "", "guard generation is invalid"),
        ("a" * 40, "0", "guard generation is invalid"),
        ("a" * 40, "9223372036854775808", "guard generation is invalid"),
    ],
)
def test_gateway_post_activate_reexec_rejects_invalid_arena_guard_authority(
    tmp_path: Path, candidate: str, generation: str, diagnostic: str,
) -> None:
    completed, helper_output = _run_post_activate_guard_reexec(
        tmp_path, candidate=candidate, generation=generation
    )

    assert completed.returncode != 0
    assert diagnostic in completed.stderr
    assert not helper_output.exists()


@pytest.mark.parametrize("mismatch", ["plan", "head"])
def test_gateway_post_activate_reexec_rejects_candidate_binding_mismatch(
    tmp_path: Path, mismatch: str,
) -> None:
    candidate = "a" * 40
    kwargs = {
        "plan_candidate": "b" * 40 if mismatch == "plan" else candidate,
        "head_candidate": "b" * 40 if mismatch == "head" else candidate,
    }
    completed, helper_output = _run_post_activate_guard_reexec(
        tmp_path, candidate=candidate, generation="7", **kwargs
    )

    assert completed.returncode != 0
    assert "prepared guard candidate, and activated deployment differ" in completed.stderr
    assert not helper_output.exists()


def test_gateway_prepare_discards_caller_arena_guard_authority() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    prepare = script.index('if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then')
    clear_candidate = script.index('PREPARED_GATEWAY_SHA=""', prepare)
    clear_generation = script.index('LAB_ARENA_RESTART_GUARD_GENERATION=""', prepare)
    acquire_lock = script.index("acquire_gateway_restart_lock", prepare)
    drain = script.index('drain_lab_arena_for_restart "$GATEWAY_PREFLIGHT_TREE"')

    assert prepare < clear_candidate < clear_generation < acquire_lock < drain
    initialized = script.index(
        'LAB_ARENA_RESTART_GUARD_GENERATION="${LAB_ARENA_RESTART_GUARD_GENERATION:-}"'
    )
    first_guard_call = script.index("run_lab_arena_restart_guard()", initialized)
    assert 'LAB_ARENA_RESTART_GUARD_GENERATION=""' not in script[
        initialized:first_guard_call
    ]
    reexec = script[
        script.index('GATEWAY_DEPLOY_STAGE="restart_reexec"'):
        script.index('\nfi\n\nbind_activated_gateway_guard_candidate')
    ]
    assert 'PREPARED_GATEWAY_SHA="$PREPARED_GATEWAY_SHA"' in reexec
    assert (
        'LAB_ARENA_RESTART_GUARD_GENERATION='
        '"$LAB_ARENA_RESTART_GUARD_GENERATION"'
    ) in reexec
    assert 'GATEWAY_ACTIVE_RELEASE_COMPONENT="$GATEWAY_ACTIVE_RELEASE_COMPONENT"' in reexec


def test_gateway_restart_accepts_only_one_exact_commit_argument() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert 'REQUESTED_GATEWAY_DEPLOY_COMMIT="${GATEWAY_DEPLOY_COMMIT:-}"' in script
    assert 'requested_commit="${1#--commit=}"' in script
    assert "unsupported gateway restart argument" in script
    assert (
        '[[ "$REQUESTED_GATEWAY_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]'
        in script
    )
    assert '--deploy-commit "$REQUESTED_GATEWAY_DEPLOY_COMMIT"' in script

    invalid = subprocess.run(
        ["bash", str(ROOT / "gw_restart.sh"), "--commit", "abc123"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert invalid.returncode == 2
    assert "lowercase full 40-character SHA" in invalid.stderr
    assert "Hydrating gateway env" not in invalid.stdout

    conflict = subprocess.run(
        [
            "bash",
            str(ROOT / "gw_restart.sh"),
            "--commit",
            "2" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "GATEWAY_DEPLOY_COMMIT": "1" * 40},
    )
    assert conflict.returncode == 2
    assert "--commit conflicts with GATEWAY_DEPLOY_COMMIT" in conflict.stderr


def test_unpinned_gateway_local_build_follows_new_main_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    start = script.index("follow_superseding_gateway_release() {")
    follow = script[start : script.index("install_gateway_python_dependencies() {", start)]

    assert follow.index('if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ]') < follow.index(
        "restart_release_supersession_v2.py"
    )
    assert 'git -C "$LEADPOET_REPO_ROOT" archive "$latest_sha"' in follow
    assert 'GATEWAY_RESTART_LOCK_HELD=1' in follow
    assert 'GATEWAY_RELEASE_SUPERSESSION_COUNT="$next_count"' in follow
    assert follow.index("cancel_gateway_offline_artifact_prepare") < follow.index(
        'bash "$superseding_tree/gw_restart.sh"'
    )

    acquisition_start = script.index(
        'if ! follow_superseding_gateway_release; then',
        script.index('if ! wait_for_gateway_offline_artifact_prepare; then'),
    )
    acquisition_end = script.index(
        'echo "Preparing commit-bound KMS credential envelopes"',
        acquisition_start,
    )
    release_build = script[acquisition_start:acquisition_end]
    assert release_build.count("follow_superseding_gateway_release") == 3
    assert release_build.index("follow_superseding_gateway_release") < (
        release_build.index("gateway/tee/build_local_release_v2.sh")
    )
    assert "Acquiring the exact historical attested V2 release channel" in release_build
    assert '--expected-commit "$PREPARED_GATEWAY_SHA"' in release_build
    assert '--gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"' in release_build
    assert release_build.index('[ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ]') < (
        release_build.index("--ensure")
    )
    assert script.index("follow_superseding_gateway_release") < script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )


def test_gateway_release_follow_reexec_preserves_existing_restart_lock() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    lock_section = script[
        script.index('if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then') :
        script.index('elif [ "$GATEWAY_RESTART_PHASE" = "post_activate" ]; then')
    ]

    inherited = 'if [ "${GATEWAY_RESTART_LOCK_HELD:-0}" = "1" ]; then'
    assert inherited in lock_section
    assert lock_section.index(inherited) < lock_section.index(
        "acquire_gateway_restart_lock"
    )
    assert 'readlink "/proc/$$/fd/9"' in lock_section
    assert "re-executed gateway restart lost the deployment lock" in lock_section


def test_pinned_gateway_rollback_preserves_newer_restart_controller() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert "GATEWAY_RESTART_CONTROLLER_CURRENT" in script
    assert (
        'if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \\\n'
        '    && [ "$PREPARED_GATEWAY_SHA" != "$ORIGIN_MAIN_GATEWAY_SHA" ] \\\n'
        '    && [ -z "$GATEWAY_RESTART_AUTHORITY_ROOT" ]; then'
        in script
    )
    assert (
        'POST_ACTIVATE_GATEWAY_HOST_RESTART_SCRIPT="$LEADPOET_REPO_ROOT/gw_restart.sh"'
        in script
    )
    assert (
        'GATEWAY_HOST_RESTART_SCRIPT="$POST_ACTIVATE_GATEWAY_HOST_RESTART_SCRIPT"'
        in script
    )
    assert (
        'GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT="$GATEWAY_RESTART_AUTHORITY_ROOT/gw_restart.sh"'
        in script
    )
    installer = _shell_function_source(script, "install_successful_restart_script")
    assert (
        'controller_sha="${GATEWAY_RESTART_AUTHORITY_COMMIT:-$GATEWAY_DEPLOY_SHA}"'
        in installer
    )
    assert (
        'controller_source_root="${GATEWAY_RESTART_AUTHORITY_ROOT:-$LEADPOET_REPO_ROOT}"'
        in installer
    )
    materialize = script.index(
        'echo "Materializing the prepared commit for pre-shutdown V2 tooling"'
    )
    preserved_tree_check = script.index(
        'echo "Verifying the prepared gateway tree with the preserved restart controller"'
    )
    candidate_preflight = script.index(
        'echo "Validating the prepared V2 release before production shutdown"'
    )
    tree_section = script[preserved_tree_check:candidate_preflight]
    assert materialize < preserved_tree_check < candidate_preflight
    assert '"$GATEWAY_PYTHON_BIN" "$GATEWAY_GIT_HELPER"' in tree_section
    assert "--phase prepared_archive" in tree_section
    assert "--strict-extras" in tree_section


def test_gateway_restart_activates_git_between_shutdown_and_existing_workflow() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    main_flow = script[
        script.index(
            'echo "Preparing exact gateway commit from configured GitHub branch"'
        ) :
    ]
    _ordered_offsets(
        main_flow,
        (
            'echo "Preparing exact gateway commit from configured GitHub branch"',
            'echo "Capturing the official subnet restart window before release acquisition"',
            'echo "Preparing exact hash-locked V2 build artifacts during release acquisition"',
            'echo "Validating the prepared V2 release before production shutdown"',
            'echo "Stopping existing gateway and Research Lab worker processes"',
            'echo "Waiting for :8000 to free"',
            'echo "Activating prepared gateway Git commit after process shutdown"',
            'GATEWAY_RESTART_PHASE=post_activate',
            'echo "Clearing Python caches"',
            'echo "Preflight disk cleanup for Docker/PCR0/Research Lab builds"',
            'echo "Loading gateway runtime env for AWS/ECR checks"',
            'echo "Building/restarting TEE enclave"',
            'bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"',
            'echo "Installing Python dependencies"',
            'echo "Relaunching gateway with cloned runtime env"',
            'unset RESEARCH_LAB_EVIDENCE_PROXY_URL',
            'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main',
            'for attempt in $(seq 1 120)',
            'curl -fsS http://localhost:8000/health',
            'if ! wait_for_gateway_v2_authority; then',
            'GATEWAY_DEPLOY_STAGE="host_restart_script_install"',
            'finalize_deployment_record succeeded',
        ),
    )


def test_gateway_restart_preserves_release_lineage_path_across_reexec() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    reexec_start = script.index("exec env ", script.index("GATEWAY_DEPLOY_STAGE=\"restart_reexec\""))
    reexec = script[reexec_start : script.index("\nfi", reexec_start)]

    assert (
        'GATEWAY_V2_RELEASE_LINEAGE="$GATEWAY_V2_RELEASE_LINEAGE"' in reexec
    )
    assert 'GATEWAY_V2_RELEASE_BUCKET="$GATEWAY_V2_RELEASE_BUCKET"' in reexec
    assert 'GATEWAY_V2_RELEASE_PREFIX="$GATEWAY_V2_RELEASE_PREFIX"' in reexec




def test_gateway_restart_fails_closed_on_all_authoritative_readiness_routes() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert "http://localhost:8000/health/v2-authority >/dev/null" in script
    assert "wait_for_gateway_v2_authority" in script
    assert "http://localhost:8000/research-lab/status >/dev/null" in script
    assert "http://localhost:8000/attest >/dev/null" in script
    assert "http://localhost:8000/research-lab/status || true" not in script
    assert "http://localhost:8000/attest || true" not in script


def test_gateway_restart_retries_v2_authority_after_base_health(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper = _shell_function_source(script, "wait_for_gateway_v2_authority")
    counter = tmp_path / "v2-attempts"
    log_path = tmp_path / "gateway.log"
    log_path.write_text("startup\n", encoding="utf-8")
    probe = f'''{helper}
mode="$1"
counter={shlex.quote(str(counter))}
printf '0\n' > "$counter"
pgrep() {{ printf '12345\n'; }}
tail() {{ :; }}
sleep() {{ :; }}
timeout() {{ shift; "$@"; }}
curl() {{
  case "$*" in
    *health/v2-authority*)
      count="$(cat "$counter")"
      count=$((count + 1))
      printf '%s\n' "$count" > "$counter"
      if [ "$mode" = deadline ]; then
        SECONDS=61
        return 22
      fi
      if [ "$mode" = success ] && [ "$count" -ge 3 ]; then
        return 0
      fi
      return 22
      ;;
    *health*) return 0 ;;
    *) return 22 ;;
  esac
}}
GATEWAY_PYTHON_BIN=/candidate/python3
GATEWAY_LOG_FILE={shlex.quote(str(log_path))}
GATEWAY_V2_HEALTH_MAX_ATTEMPTS=3
GATEWAY_V2_HEALTH_RETRY_SECONDS=0
GATEWAY_V2_HEALTH_DEADLINE_SECONDS=60
if ! timeout 5 curl -fsS http://localhost:8000/health >/dev/null; then
  exit 90
fi
wait_for_gateway_v2_authority
'''

    recovered = subprocess.run(
        ["bash", "-c", probe, "gateway-v2-health-probe", "success"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert counter.read_text(encoding="utf-8").strip() == "3"
    assert "ready after attempt 3" in recovered.stdout

    exhausted = subprocess.run(
        ["bash", "-c", probe, "gateway-v2-health-probe", "exhaust"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert exhausted.returncode != 0
    assert counter.read_text(encoding="utf-8").strip() == "3"
    assert "did not become ready before the bounded deadline" in exhausted.stderr

    deadline = subprocess.run(
        ["bash", "-c", probe, "gateway-v2-health-probe", "deadline"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert deadline.returncode != 0
    assert counter.read_text(encoding="utf-8").strip() == "1"
    assert "did not become ready before the bounded deadline" in deadline.stderr


def test_gateway_restart_forces_instance_role_for_runtime_aws_calls() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    runtime_launch = script.index('echo "Relaunching gateway with cloned runtime env"')
    unset_credentials = script.index(
        "unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE "
        "AWS_SESSION_TOKEN AWS_SECURITY_TOKEN",
        runtime_launch,
    )
    instance_role = script.index(
        "export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true",
        unset_credentials,
    )
    gateway_launch = script.index(
        'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main',
        instance_role,
    )
    assert runtime_launch < unset_credentials < instance_role < gateway_launch


def test_gateway_restart_installs_commit_bound_admin_wrapper_after_handoff() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    wrapper = (
        ROOT / "scripts" / "research_lab_admin_wrapper_runtime.sh"
    ).read_text(encoding="utf-8")

    status_handoff = script.index(
        "timeout 30 curl -fsS http://localhost:8000/research-lab/status"
    )
    install_stage = script.index(
        'GATEWAY_DEPLOY_STAGE="host_restart_script_install"'
    )
    wrapper_install = script.index(
        "install_research_lab_admin_wrapper",
        install_stage,
    )
    restart_install = script.index(
        "install_successful_restart_script",
        wrapper_install,
    )
    completed = script.index(
        'GATEWAY_DEPLOY_STAGE="completed"',
        restart_install,
    )

    assert (
        status_handoff
        < install_stage
        < wrapper_install
        < restart_install
        < completed
    )
    assert (
        'source_script="$LEADPOET_REPO_ROOT/scripts/'
        'research_lab_admin_wrapper_runtime.sh"'
    ) in script
    assert "RESEARCH_LAB_PRIVATE_REPO_BRANCH" not in wrapper
    assert "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI" not in wrapper
    assert "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID" not in wrapper
    assert "LEADPOET_AWS_INSTANCE_ROLE_ONLY=true" in wrapper










def test_gateway_restart_does_not_kill_colocated_runner_builds() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert "wait_for_foreign_docker_builds" in script
    assert "stop_local_stale_build_processes TERM" in script
    assert "stop_local_stale_build_processes KILL" in script
    assert 'pkill -TERM -f "docker build' not in script
    assert 'pkill -KILL -f "docker build' not in script
    assert "ensure_docker_ready" in script
    assert "validator_tee/scripts/reclaim_docker_storage_v2.sh" in script
    assert 'bash "$reclaim_script"' in script
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in script
    assert "sudo nsenter -t 1 -m --" not in script




def test_gateway_restart_exports_attested_artifact_bucket_to_runtime() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert (
        'RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="${'
        'RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET:-$GATEWAY_V2_RELEASE_BUCKET}"'
        in script
    )
    assert "export RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET" in script
    assert (
        'RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="'
        '$RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET" \\'
        in script
    )










def test_gateway_restart_v2_preflight_runs_target_commit_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    materialize = script.index("Materializing the prepared commit for pre-shutdown V2 tooling")
    restart_window = script.index("Capturing the official subnet restart window before release acquisition")
    artifact_prepare = script.index("Preparing exact hash-locked V2 build artifacts during release acquisition")
    preflight = script.index("Validating the prepared V2 release before production shutdown")
    shutdown = script.index("Stopping existing gateway and Research Lab worker processes")
    assert materialize < restart_window < artifact_prepare < preflight < shutdown
    assert "verify_weight_submission_ready_v2" not in script
    assert "bootstrap_active_ancestry_checkpoints_v2" not in script



def test_gateway_restart_isolates_candidate_release_until_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    prepare = script.index("Materializing the prepared commit for pre-shutdown V2 tooling")
    shutdown = script.index("Stopping existing gateway and Research Lab worker processes")
    pre_shutdown = script[prepare:shutdown]
    assert '--gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"' in pre_shutdown
    assert '--gateway-output "$GATEWAY_V2_RELEASE_MANIFEST"' not in pre_shutdown
    assert "prepare_active_release_lineage_v2" not in script



def test_gateway_restart_counts_only_exact_live_gateway_as_reclaimable(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = _shell_function_source(
        script,
        "gateway_memory_ready_after_running_gateway_shutdown",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "gateway").mkdir(parents=True)
    proc_root = tmp_path / "proc"
    process_root = proc_root / "123"
    process_root.mkdir(parents=True)
    (process_root / "status").write_text(
        f"Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n"
        "VmRSS:\t12582912 kB\n",
        encoding="utf-8",
    )
    (process_root / "stat").write_text(
        "123 (python3) S " + " ".join(["1"] * 30) + "\n",
        encoding="utf-8",
    )
    command_path = process_root / "cmdline"
    command_path.write_bytes(
        b"\0".join(
            value.encode("utf-8")
            for value in (sys.executable, "-u", "-m", "gateway.main", "")
        )
    )
    (process_root / "cwd").symlink_to(repo_root)
    report_path = tmp_path / "memory.json"
    report_path.write_text(
        json.dumps(
            {
                "available_memory_mib": 7000,
                "minimum_available_memory_mib": 16384,
                "schema_version": "leadpoet.gateway_host_memory_guard.v2",
                "status": "blocked",
            }
        ),
        encoding="utf-8",
    )
    harness = f"""set -euo pipefail
{helper_source}
GATEWAY_PYTHON_BIN="$1"
LEADPOET_REPO_ROOT="$2"
GATEWAY_RECLAIMABLE_MEMORY_SAFETY_MARGIN_MIB=2048
gateway_memory_ready_after_running_gateway_shutdown "$3" "$4" "$5"
"""
    arguments = [
        "bash",
        "-c",
        harness,
        "gateway-reclaimable-memory-test",
        sys.executable,
        str(repo_root),
        str(report_path),
        "123",
        str(proc_root),
    ]

    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "available_memory_mib": 7000,
        "minimum_available_memory_mib": 16384,
        "reclaimable_gateway_memory_mib": 12288,
        "reclaimable_gateway_parent_memory_mib": 12288,
        "reclaimable_gateway_worker_count": 0,
        "reclaimable_gateway_worker_memory_mib": 0,
        "safety_margin_mib": 2048,
        "schema_version": "leadpoet.gateway_reclaimable_memory.v1",
        "status": "ready_after_gateway_shutdown",
    }

    command_path.write_bytes(
        b"\0".join(
            value.encode("utf-8")
            for value in (sys.executable, "-u", "-m", "gateway.other", "")
        )
    )
    rejected = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert rejected.returncode != 0
    assert "running gateway command differs" in rejected.stderr


def test_gateway_restart_rechecks_real_memory_after_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    pre_shutdown = script.index("wait_for_gateway_build_memory 1")
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    post_shutdown = script.index("wait_for_gateway_build_memory 0 10", shutdown)
    activation = script.index(
        'echo "Activating prepared gateway Git commit after process shutdown"'
    )

    assert pre_shutdown < shutdown < post_shutdown < activation
    assert "gateway_memory_ready_after_running_gateway_shutdown" in script[
        :shutdown
    ]
    assert 'pkill -9 -f "/gateway/research_lab/worker_process[.]py"' in script[
        shutdown:post_shutdown
    ]


def test_gateway_restart_counts_exact_direct_worker_children_as_reclaimable(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = _shell_function_source(
        script,
        "gateway_memory_ready_after_running_gateway_shutdown",
    )
    repo_root = tmp_path / "repo"
    worker_script = repo_root / "gateway" / "research_lab" / "worker_process.py"
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text("# worker\n", encoding="utf-8")
    proc_root = tmp_path / "proc"

    def write_process(pid: str, ppid: str, rss_kib: int, argv: tuple[str, ...]) -> None:
        process_root = proc_root / pid
        process_root.mkdir(parents=True)
        (process_root / "status").write_text(
            f"Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n"
            f"VmRSS:\t{rss_kib} kB\n",
            encoding="utf-8",
        )
        (process_root / "stat").write_text(
            f"{pid} (python3) S {ppid} " + " ".join(["1"] * 29) + "\n",
            encoding="utf-8",
        )
        (process_root / "cmdline").write_bytes(
            b"\0".join(value.encode("utf-8") for value in (*argv, ""))
        )
        (process_root / "cwd").symlink_to(repo_root)

    write_process("123", "1", 524288, (sys.executable, "-u", "-m", "gateway.main"))
    for pid, kind, index, total, prefix in (
        ("456", "hosted", "0", "10", "research-lab-worker"),
        ("789", "scoring", "0", "25", "research-lab-scorer"),
    ):
        write_process(
            pid,
            "123",
            2097152,
            (
                sys.executable,
                str(worker_script),
                "--kind",
                kind,
                "--worker-index",
                index,
                "--total-workers",
                total,
                "--worker-prefix",
                prefix,
                "--log-level",
                "INFO",
            ),
        )

    report_path = tmp_path / "memory.json"
    report_path.write_text(
        json.dumps(
            {
                "available_memory_mib": 14000,
                "minimum_available_memory_mib": 16384,
                "schema_version": "leadpoet.gateway_host_memory_guard.v2",
                "status": "blocked",
            }
        ),
        encoding="utf-8",
    )
    harness = f"""set -euo pipefail
{helper_source}
GATEWAY_PYTHON_BIN="$1"
LEADPOET_REPO_ROOT="$2"
GATEWAY_RECLAIMABLE_MEMORY_SAFETY_MARGIN_MIB=2048
gateway_memory_ready_after_running_gateway_shutdown "$3" "$4" "$5"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "gateway-worker-memory-test",
            sys.executable,
            str(repo_root),
            str(report_path),
            "123",
            str(proc_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["reclaimable_gateway_parent_memory_mib"] == 512
    assert result["reclaimable_gateway_worker_count"] == 2
    assert result["reclaimable_gateway_worker_memory_mib"] == 4096
    assert result["reclaimable_gateway_memory_mib"] == 4608










def test_gateway_offline_artifact_prepare_overlaps_release_and_fails_closed(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = "\n\n".join(
        _shell_function_source(script, name)
        for name in (
            "wait_for_gateway_owned_process_group",
            "start_gateway_offline_artifact_prepare",
            "wait_for_gateway_offline_artifact_prepare",
        )
    )
    preflight_tree = tmp_path / "candidate"
    prepare_script = (
        preflight_tree / "gateway" / "tee" / "prepare_offline_artifacts_v2.sh"
    )
    prepare_script.parent.mkdir(parents=True)
    prepare_script.write_text(
        """#!/bin/bash
set -euo pipefail
echo artifact-start >> "$CONCURRENCY_LOG"
sleep "${FAKE_ARTIFACT_SECONDS}"
echo artifact-end >> "$CONCURRENCY_LOG"
echo artifact-ready
exit "${FAKE_ARTIFACT_STATUS:-0}"
""",
        encoding="utf-8",
    )
    prepare_script.chmod(0o755)
    artifact_root = tmp_path / "artifacts"
    artifact_log = tmp_path / "artifact.log"
    timing_log = tmp_path / "timing.log"
    concurrency_log = tmp_path / "concurrency.log"
    shutdown_marker = tmp_path / "shutdown"
    harness = f"""set -euo pipefail
record_gateway_restart_timing() {{
  printf '%s:%s\\n' "$1" "${{2:-reached}}" >> "$TIMING_LOG"
}}
{helper_source}
GATEWAY_PREFLIGHT_TREE="$1"
GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$2"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG="$3"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
TIMING_LOG="$4"
SHUTDOWN_MARKER="$5"
start_gateway_offline_artifact_prepare
echo release-start >> "$CONCURRENCY_LOG"
sleep "$FAKE_RELEASE_SECONDS"
echo release-end >> "$CONCURRENCY_LOG"
if wait_for_gateway_offline_artifact_prepare; then
  touch "$SHUTDOWN_MARKER"
else
  status=$?
  test ! -e "$SHUTDOWN_MARKER"
  exit "$status"
fi
"""
    command = [
        "bash",
        "-c",
        harness,
        "gateway-offline-overlap-test",
        str(preflight_tree),
        str(artifact_root),
        str(artifact_log),
        str(timing_log),
        str(shutdown_marker),
    ]

    success = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=4,
        env={
            **os.environ,
            "CONCURRENCY_LOG": str(concurrency_log),
            "FAKE_ARTIFACT_SECONDS": "1.0",
            "FAKE_RELEASE_SECONDS": "1.0",
        },
    )
    assert success.returncode == 0, success.stderr
    concurrency_events = concurrency_log.read_text(encoding="utf-8").splitlines()
    assert concurrency_events.index("artifact-start") < concurrency_events.index(
        "release-end"
    )
    assert concurrency_events.index("release-start") < concurrency_events.index(
        "artifact-end"
    )
    assert shutdown_marker.exists()
    assert "artifact-ready" in success.stdout
    assert timing_log.read_text(encoding="utf-8").splitlines() == [
        "offline_artifact_prepare_started:reached",
        "offline_artifact_prepare_complete:passed",
    ]

    shutdown_marker.unlink()
    timing_log.unlink()
    concurrency_log.unlink()
    failed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
        env={
            **os.environ,
            "CONCURRENCY_LOG": str(concurrency_log),
            "FAKE_ARTIFACT_SECONDS": "0.05",
            "FAKE_ARTIFACT_STATUS": "23",
            "FAKE_RELEASE_SECONDS": "0.1",
        },
    )
    assert failed.returncode == 23
    assert not shutdown_marker.exists()
    assert "failed before shutdown" in failed.stderr
    assert timing_log.read_text(encoding="utf-8").splitlines() == [
        "offline_artifact_prepare_started:reached",
        "offline_artifact_prepare_complete:failed",
    ]




def test_gateway_exit_cleanup_terminates_offline_artifact_prepare(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = "\n\n".join(
        _shell_function_source(script, name)
        for name in (
            "wait_for_gateway_owned_process_group",
            "cancel_gateway_owned_process_group",
            "start_gateway_offline_artifact_prepare",
            "cancel_gateway_offline_artifact_prepare",
        )
    )
    preflight_tree = tmp_path / "candidate"
    prepare_script = (
        preflight_tree / "gateway" / "tee" / "prepare_offline_artifacts_v2.sh"
    )
    prepare_script.parent.mkdir(parents=True)
    prepare_script.write_text(
        """#!/bin/bash
set -euo pipefail
echo started > "$FAKE_STARTED_MARKER"
sleep 300 &
child_pid="$!"
printf '%s\n' "$child_pid" > "$FAKE_CHILD_PID_MARKER"
wait "$child_pid"
""",
        encoding="utf-8",
    )
    prepare_script.chmod(0o755)
    started_marker = tmp_path / "started"
    child_pid_marker = tmp_path / "child-pid"
    harness = f"""set -euo pipefail
record_gateway_restart_timing() {{ :; }}
{helper_source}
GATEWAY_PREFLIGHT_TREE="$1"
GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$2"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG="$3"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
start_gateway_offline_artifact_prepare
prepare_pid="$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
for _ in $(seq 1 20); do
  if [ -s "$FAKE_STARTED_MARKER" ] && [ -s "$FAKE_CHILD_PID_MARKER" ]; then
    break
  fi
  sleep 0.05
done
test -s "$FAKE_STARTED_MARKER"
test -s "$FAKE_CHILD_PID_MARKER"
child_pid="$(cat "$FAKE_CHILD_PID_MARKER")"
cancel_gateway_offline_artifact_prepare
test -z "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
for _ in $(seq 1 20); do
  if ! kill -0 "$child_pid" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if kill -0 "$prepare_pid" 2>/dev/null || kill -0 "$child_pid" 2>/dev/null; then
  exit 91
fi
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "gateway-offline-cleanup-test",
            str(preflight_tree),
            str(tmp_path / "artifacts"),
            str(tmp_path / "artifact.log"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "FAKE_STARTED_MARKER": str(started_marker),
            "FAKE_CHILD_PID_MARKER": str(child_pid_marker),
        },
    )
    assert result.returncode == 0, result.stderr
    assert child_pid_marker.exists()

    exit_handler = _shell_function_source(script, "on_gateway_restart_exit")
    assert exit_handler.index("cancel_gateway_offline_artifact_prepare") < (
        exit_handler.index('rm -rf "$GATEWAY_PREFLIGHT_TREE"')
    )


def test_gateway_background_launch_waits_for_owned_process_group_before_return(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = "\n\n".join(
        _shell_function_source(script, name)
        for name in (
            "wait_for_gateway_owned_process_group",
            "cancel_gateway_owned_process_group",
            "start_gateway_offline_artifact_prepare",
            "cancel_gateway_offline_artifact_prepare",
        )
    )
    preflight_tree = tmp_path / "candidate"
    prepare_script = (
        preflight_tree / "gateway" / "tee" / "prepare_offline_artifacts_v2.sh"
    )
    prepare_script.parent.mkdir(parents=True)
    prepare_script.write_text("#!/bin/bash\nsleep 300\n", encoding="utf-8")
    prepare_script.chmod(0o755)
    artifact_log = tmp_path / "artifact.log"
    harness = f"""set -euo pipefail
record_gateway_restart_timing() {{ :; }}
{helper_source}
GATEWAY_PREFLIGHT_TREE="$1"
GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$2"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG="$3"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
start_gateway_offline_artifact_prepare
prepare_pid="$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
marker="${{GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG}}.process-group"
test "$(cat "$marker")" = "$prepare_pid"
kill -0 -- "-$prepare_pid"
cancel_gateway_offline_artifact_prepare
test ! -e "$marker"
if kill -0 "$prepare_pid" 2>/dev/null || kill -0 -- "-$prepare_pid" 2>/dev/null; then
  exit 91
fi
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "gateway-owned-process-group-test",
            str(preflight_tree),
            str(tmp_path / "artifacts"),
            str(artifact_log),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_gateway_restart_has_no_retired_rebenchmark_maintenance() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert script.count('"GATEWAY_V2_DEFER_WORKER_FLEETS"') == 3
    assert "reconcile_gateway_rebenchmark_retry_runtime" not in script
    assert "GATEWAY_REBENCHMARK_RETRY_RECONCILIATION_HELPER" not in script
    assert "stop_research_lab_private_model_containers" not in script


def test_gateway_restart_verifies_prepared_and_activated_candidate_git_blobs() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    preflight = script.index("gateway.tee.restart_preflight_v2")
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    activation = script.index(
        'echo "Activating prepared gateway Git commit after process shutdown"'
    )
    activated_verification = script.index(
        'echo "Verifying prepared and activated gateway trees against exact Git blobs"'
    )

    assert preflight < shutdown < activation < activated_verification
    assert (
        '"$LEADPOET_REPO_ROOT/scripts/gateway_git_deploy.py" \\\n'
        "  verify-tree-pair"
    ) in script
    assert (
        '--prepared-evidence \\\n'
        '    "$GATEWAY_V2_CONFIG_DIR/gateway-candidate-tree-preflight.json"'
    ) in script
    assert '--activated-root "$LEADPOET_REPO_ROOT"' in script

    preflight_source = (
        ROOT / "gateway/tee/restart_preflight_v2.py"
    ).read_text(encoding="utf-8")
    assert "write_tree_verification_evidence" in preflight_source
    assert 'phase="prepared_archive"' in preflight_source
    assert "strict_extras=True" in preflight_source




def test_gateway_restart_installs_declared_host_dependencies_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    dependency_preflight = script.index(
        'echo "Installing gateway host Python dependencies before production shutdown"'
    )
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    post_activate_install = script.index('echo "Installing Python dependencies"')

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert '--requirement "$requirements_file"' in script
    assert 'requirements_file="$GATEWAY_PREFLIGHT_TREE/requirements.txt"' in script
    assert "bittensor==10.5.0" in requirements
    assert "async-substrate-interface==2.2.1" in requirements
    assert "publicsuffix2>=2.20191221" in requirements
    assert "leadpoet-subnet substrate-interface" in script
    assert "py-scale-codec scalecodec" in script
    metadata_cleanup = script.index(
        'rm -rf -- "$legacy_project_metadata"'
    )
    dependency_check = script.index('"$GATEWAY_PYTHON_BIN" -m pip check')
    assert metadata_cleanup < dependency_check
    assert '"$GATEWAY_PYTHON_BIN" -m pip check' in script
    assert script.count("install_gateway_python_dependencies") == 3
    assert "GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT" in script
    assert (
        'if [ "$GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT" '
        '= "$dependency_fingerprint" ]; then'
    ) in script
    assert (
        "Reusing exact candidate dependency installation from "
        "pre-shutdown validation"
    ) in script
    assert (
        'GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT="'
        '$GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT" \\'
    ) in script
    assert dependency_preflight < shutdown < post_activate_install
    assert (
        'echo "Gateway remains running; production shutdown has not started." >&2'
        in script[dependency_preflight:shutdown]
    )


def test_gateway_restart_records_nonblocking_commit_bound_stage_timings() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert "leadpoet.gateway_restart_timing.v1" in script
    for stage in ("offline_artifact_prepare_started", "local_release_ready", "pre_shutdown_checks_complete", "candidate_activated", "attested_runtime_staged", "gateway_enclaves_started", "completed"):
        assert f'record_gateway_restart_timing "{stage}"' in script
    for retired in ("ancestry_precheckpoint", "chain_settlement_repair", "validator_weight_input"):
        assert retired not in script



def test_gateway_runtime_env_cannot_replace_current_restart_controller_state() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    for key in ("GATEWAY_RESTART_AUTHORITY_ROOT", "GATEWAY_RESTART_AUTHORITY_COMMIT"):
        assert script.count(f"-u {key} ") >= 1
        assert f'    "{key}",' in script
    assert 'cat "$ENV_SECRET" >> "$ENV_CLONE"' in script
    assert "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF" not in script
    assert "GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS" not in script



def test_gateway_live_env_clone_removes_both_prepared_release_paths(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    marker = 'python3 - "$PID" "$ENV_CLONE" <<\'PY\'\n'
    start = script.index(marker) + len(marker)
    clone_source = script[start : script.index("\nPY\n", start)]
    clone = tmp_path / "gateway-env-clone.sh"
    inherited = tmp_path / "inherited-environ"
    inherited.write_bytes(
        b"SAFE_RUNTIME_VALUE=retained\0"
        b"GATEWAY_PREPARED_V2_RELEASE_MANIFEST=/stale/f5-gateway.json\0"
    )
    clone_source = clone_source.replace(
        'f"/proc/{pid}/environ"', repr(str(inherited))
    )
    subprocess.run(
        [sys.executable, "-", "unused-pid", str(clone)],
        input=clone_source,
        text=True,
        check=True,
        timeout=5,
    )

    cloned = clone.read_text(encoding="utf-8")
    assert "SAFE_RUNTIME_VALUE=retained" in cloned
    assert "GATEWAY_PREPARED_V2_RELEASE_MANIFEST" not in cloned


def test_gateway_candidate_reexec_rebinds_restart_identity_before_telemetry() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    timing_file = script.index('GATEWAY_RESTART_TIMING_FILE="')
    binding = script.index("bind_gateway_restart_invocation_to_timing_file")
    first_timing = script.index('record_gateway_restart_timing "')
    assert timing_file < binding < first_timing
    assert 'GATEWAY_RESTART_TIMING_INITIALIZED="$GATEWAY_RESTART_TIMING_INITIALIZED"' in script
    assert 'GATEWAY_RESTART_TIMING_FILE="$GATEWAY_RESTART_TIMING_FILE"' in script
    assert '[ ! -f "$GATEWAY_RESTART_TIMING_FILE" ]' in script
    assert "^gateway-([0-9]+)-([0-9]+)\\.jsonl$" in script
    assert '[ -L "$GATEWAY_RESTART_TIMING_FILE" ]' in script
    assert '[ ! -s "$GATEWAY_RESTART_TIMING_FILE" ]' in script
    assert '[ "$ledger_pid" != "$$" ]' in script
    assert (
        'expected_ledger="${GATEWAY_RESTART_TIMING_DIR%/}/$ledger_name"'
        in script
    )


def test_gateway_candidate_reexec_uses_only_current_canonical_ledger(
    tmp_path: Path,
) -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    bootstrap = restart.split('\nwhile [ "$#" -gt 0 ]; do\n', 1)[0]

    def run_probe(mode: str) -> subprocess.CompletedProcess[str]:
        probe = r'''
GATEWAY_RESTART_TIMING_DIR="$1/timings"
GATEWAY_RESTART_STARTED_EPOCH=1700000000
GATEWAY_RESTART_TIMING_INITIALIZED=1
GATEWAY_RESTART_INVOCATION_ID=gateway-stale-n-minus-one
LEADPOET_RESTART_INVOCATION_ID=gateway-stale-n-minus-one
mkdir -p "$GATEWAY_RESTART_TIMING_DIR"
ledger="$GATEWAY_RESTART_TIMING_DIR/gateway-${GATEWAY_RESTART_STARTED_EPOCH}-$$.jsonl"
case "$2" in
  exact)
    printf '%s\n' '{"stage":"invoked","status":"reached"}' > "$ledger"
    ;;
  invalid_name)
    ledger="$GATEWAY_RESTART_TIMING_DIR/gateway.jsonl"
    printf '%s\n' '{"stage":"invoked","status":"reached"}' > "$ledger"
    ;;
  wrong_epoch)
    ledger="$GATEWAY_RESTART_TIMING_DIR/gateway-1700000001-$$.jsonl"
    printf '%s\n' '{"stage":"invoked","status":"reached"}' > "$ledger"
    ;;
  wrong_pid)
    ledger="$GATEWAY_RESTART_TIMING_DIR/gateway-1700000000-$(( $$ + 1 )).jsonl"
    printf '%s\n' '{"stage":"invoked","status":"reached"}' > "$ledger"
    ;;
  empty)
    : > "$ledger"
    ;;
  symlink)
    printf '%s\n' '{"stage":"invoked","status":"reached"}' > "$1/target.jsonl"
    ln -s "$1/target.jsonl" "$ledger"
    ;;
  outside)
    mkdir -p "$1/outside"
    ledger="$1/outside/gateway-1700000000-$$.jsonl"
    printf '%s\n' '{"stage":"invoked","status":"reached"}' > "$ledger"
    ;;
  missing)
    ;;
  *)
    exit 97
    ;;
esac
GATEWAY_RESTART_TIMING_FILE="$ledger"
'''
        return subprocess.run(
            [
                "bash",
                "-c",
                probe
                + bootstrap
                + "\nprintf '%s\\n%s\\n%s\\n' "
                + '"$GATEWAY_RESTART_INVOCATION_ID" '
                + '"$LEADPOET_RESTART_INVOCATION_ID" '
                + '"gateway-${GATEWAY_RESTART_STARTED_EPOCH}-$$"\n',
                "gateway-restart-ledger-probe",
                str(tmp_path / mode),
                mode,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    exact = run_probe("exact")
    assert exact.returncode == 0, exact.stderr
    exact_lines = exact.stdout.splitlines()
    assert len(exact_lines) == 3
    assert exact_lines[0] == exact_lines[1] == exact_lines[2]
    assert re.fullmatch(r"gateway-1700000000-[1-9][0-9]*", exact_lines[0])

    for mode in (
        "invalid_name",
        "wrong_epoch",
        "wrong_pid",
        "empty",
        "symlink",
        "outside",
        "missing",
    ):
        rejected = run_probe(mode)
        assert rejected.returncode != 0, mode
        assert "ERROR: gateway restart timing ledger" in rejected.stderr, mode


def test_miner_bootstrap_exec_preserves_stable_cwd_and_timing_ledger(
    tmp_path: Path,
) -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    branch = restart.index(
        '  GATEWAY_DEPLOY_STAGE="miner_maintenance_pre_hydration"'
    )
    body = restart[branch : restart.index("\nfi\n", branch)]

    repo_root = tmp_path / "repo"
    bootstrap_root = tmp_path / "bootstrap"
    candidate_root = bootstrap_root / "candidate"
    timing_dir = tmp_path / "timings"
    result_file = tmp_path / "result"
    repo_root.mkdir()
    candidate_root.mkdir(parents=True)
    timing_dir.mkdir()

    python_stub = tmp_path / "bootstrap-python"
    python_stub.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'test "$1" = -P\n'
        'test "$2" = -m\n'
        'test "$PWD" = /\n'
        'test "$GATEWAY_RESTART_STARTED_EPOCH" = 1700000000\n'
        'test "$GATEWAY_RESTART_TIMING_DIR" = "$EXPECTED_TIMING_DIR"\n'
        'test "$GATEWAY_RESTART_TIMING_FILE" = "$EXPECTED_TIMING_FILE"\n'
        'test "$GATEWAY_RESTART_TIMING_INITIALIZED" = 1\n'
        'rm -rf -- "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT"\n'
        "/bin/sleep 1.1\n"
        'test "$PWD" = /\n'
        'test -s "$GATEWAY_RESTART_TIMING_FILE"\n'
        "printf '%s\\n%s\\n%s\\n' "
        '"$PWD" "$GATEWAY_RESTART_STARTED_EPOCH" '
        '"$GATEWAY_RESTART_TIMING_FILE" '
        '>"$RESULT_FILE"\n'
    )
    python_stub.chmod(0o700)

    started_epoch = "1700000000"
    timing_file = timing_dir / "gateway-1700000000-probe.jsonl"
    timing_file.write_text(
        '{"stage":"invoked","status":"reached"}\n', encoding="utf-8"
    )
    probe = candidate_root / "gw_restart.sh"
    probe.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"GATEWAY_RESTART_STARTED_EPOCH={started_epoch}\n"
        f"GATEWAY_RESTART_TIMING_DIR={shlex.quote(str(timing_dir))}\n"
        f"GATEWAY_RESTART_TIMING_FILE={shlex.quote(str(timing_file))}\n"
        "GATEWAY_RESTART_TIMING_INITIALIZED=1\n"
        + body
        + "\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)

    probe_environment = dict(os.environ)
    for name in (
        "GATEWAY_RESTART_STARTED_EPOCH",
        "GATEWAY_RESTART_TIMING_DIR",
        "GATEWAY_RESTART_TIMING_FILE",
        "GATEWAY_RESTART_TIMING_INITIALIZED",
    ):
        probe_environment.pop(name, None)
    probe_environment.update(
        {
            "EXPECTED_TIMING_DIR": str(timing_dir),
            "EXPECTED_TIMING_FILE": str(timing_file),
            "GATEWAY_HOST_RESTART_SCRIPT": str(tmp_path / "host-restart"),
            "GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN": str(
                bootstrap_root / "plan.json"
            ),
            "GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT": str(bootstrap_root),
            "GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE": str(
                tmp_path / "handoff"
            ),
            "GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE": "0" * 64,
            "GATEWAY_PYTHON_BIN": str(python_stub),
            "GATEWAY_RESTART_CONTROLLER_CURRENT": str(tmp_path / "current"),
            "LEADPOET_REPO_ROOT": str(repo_root),
            "REQUESTED_GATEWAY_DEPLOY_COMMIT": "1" * 40,
            "RESULT_FILE": str(result_file),
            "bootstrap_candidate_root": str(candidate_root),
        }
    )
    completed = subprocess.run(
        ["/bin/bash", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=probe_environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "getcwd" not in completed.stderr
    assert "timing ledger is unavailable" not in completed.stderr
    assert not bootstrap_root.exists()
    assert result_file.read_text(encoding="utf-8").splitlines() == [
        "/",
        started_epoch,
        str(timing_file),
    ]


def test_gateway_restart_checks_shared_maintenance_without_retired_admin_command() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    v2_health = "if ! wait_for_gateway_v2_authority; then"
    shared_status = (
        "curl -fsS http://localhost:8000/research-lab/status"
    )
    maintenance_runtime = (
        "-m gateway.tee.gateway_miner_maintenance_restart_v1"
    )
    completed = 'GATEWAY_DEPLOY_STAGE="completed"'

    assert "resume-restart-maintenance" not in script
    for command in (
        "pause-autoresearch",
        "resume-autoresearch",
        "pause-scoring",
        "resume-scoring",
    ):
        assert f"-m gateway.research_lab.admin {command}" not in script
    assert (
        script.rindex(shared_status)
        < script.rindex(maintenance_runtime)
        < script.rindex(completed)
    )


def test_gateway_restart_uses_one_canonical_checkout_for_host_processes() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert 'LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"' in script
    assert 'GATEWAY_ROOT="${GATEWAY_ROOT:-$LEADPOET_REPO_ROOT/gateway}"' in script
    assert 'export PYTHONPATH="$LEADPOET_REPO_ROOT"' in script
    assert 'cd "$LEADPOET_REPO_ROOT"' in script
    assert 'PYTHONPATH=/home/ec2-user' not in script
    assert 'export PYTHONPATH="/home/ec2-user"' not in script
    assert 'sys.path.insert(1, "/home/ec2-user")' not in script
    assert 'GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"' in script
    assert 'GATEWAY_TEE_EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"' in script
    assert 'GATEWAY_TEE_FALLBACK_LOG_DIR="$GATEWAY_LOG_ROOT/gateway/logs/tee_fallback"' in script
    assert 'chmod +x "$GATEWAY_ROOT"/tee/*.sh' not in script
    assert 'bash ./start_enclave.sh' in script
    assert 'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main' in script
    assert 'GATEWAY_LAUNCHER_PID="$!"' in script
    assert (
        'pgrep -f "^$GATEWAY_PYTHON_BIN -u -m gateway[.]main$"'
        in script
    )
    assert 'GATEWAY_PID="$!"' not in script
    assert 'pkill -9 -f "python3 -u -m gateway.main"' in script


def test_gateway_restart_disables_the_retired_host_provider_proxy() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert 'pkill -9 -f "gateway.research_lab.provider_evidence_proxy"' in script
    assert '"$GATEWAY_PYTHON_BIN" -m gateway.research_lab.provider_evidence_proxy' not in script
    assert "legacy_v1" not in script
    assert "unset RESEARCH_LAB_EVIDENCE_PROXY_URL" in script


def test_gateway_restart_starts_tee_egress_before_v2_readiness() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    managed_service_cleanup = (
        "sudo systemctl stop leadpoet-tee-egress-forwarder.service"
    )
    cleanup = 'pkill -9 -f "gateway.utils.tee_egress_forwarder"'
    launch = (
        '-m gateway.utils.tee_egress_forwarder \\\n'
        '    >> "$GATEWAY_LOG_ROOT/tee_egress_forwarder.log" '
        '2>&1 < /dev/null \\\n'
        '    7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &'
    )
    readiness = '"$GATEWAY_PYTHON_BIN" -m gateway.tee.verify_v2_runtime_ready'

    assert managed_service_cleanup in script
    assert cleanup in script
    assert launch in script
    assert (
        script.index(managed_service_cleanup)
        < script.index(cleanup)
        < script.index(launch)
        < script.index(readiness)
    )


def test_gateway_restart_has_fail_closed_lock_and_official_epoch_gate() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert 'flock -n 9' in script
    assert 'another gateway restart is already running' in script
    assert "Recovering gateway restart lock inherited by a detached runtime process" in script
    assert "leadpoet_ensure_post_activation_docker_operation_lock_v2" in script
    assert (
        '-m gateway.utils.tee_inter_enclave_relay \\\n'
        '    >> "$GATEWAY_LOG_ROOT/inter_enclave_relay.log" '
        '2>&1 < /dev/null \\\n'
        '    7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &'
    ) in script
    assert 'VALIDATOR_GATEWAY_PCR0_CACHE_FILE' not in script
    assert 'independent_gateway_identity' not in script
    gate = "Leadpoet.utils.restart_epoch_gate"
    release = "gateway/tee/build_local_release_v2.sh"
    shutdown = 'echo "Stopping existing gateway and Research Lab worker processes"'
    assert gate in script
    gate_offset = script.index(gate)
    assert gate_offset < script.index(release, gate_offset) < script.index(shutdown)
    assert "Approved V2 release is not published yet" not in script
    assert "--maximum" not in script


def test_gateway_restart_does_not_clone_restart_control_state_into_runtime() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    restart_only_keys = (
        "GATEWAY_RESTART_PHASE",
        "GATEWAY_RESTART_LOCK_HELD",
        "GATEWAY_DEPLOY_PLAN_FILE",
        "GATEWAY_DEPLOY_STAGE",
        "GATEWAY_DEPLOY_COMPLETED",
        "GATEWAY_DEPLOY_COMMIT",
    )

    # Both the Secrets Manager parser and the live-process environment clone
    # must reject these values. Otherwise the relaunched gateway preserves a
    # stale per-restart /tmp plan path and the next rollout cannot finalize.
    for key in restart_only_keys:
        assert script.count(f'"{key}",') >= 2
    restart_only_block = script.split("restart_only_keys = {", 1)[1].split("}", 1)[0]
    for key in (
        "GATEWAY_DEPLOY_COMMIT",
        "GATEWAY_V2_DEFER_WORKER_FLEETS",
        "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
        "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
        "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
        "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
    ):
        assert f'"{key}",' in restart_only_block
    assert "unset GATEWAY_DEPLOY_COMMIT" in script


def test_concurrent_restart_exits_before_checkout_or_process_changes(tmp_path: Path) -> None:
    lock_file = tmp_path / "gateway-restart.lock"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    flock_stub = bin_dir / "flock"
    flock_stub.write_text(
        """#!/usr/bin/env python3
import fcntl
import sys

try:
    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(1)
""",
        encoding="utf-8",
    )
    flock_stub.chmod(0o755)

    with lock_file.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(ROOT / "gw_restart.sh")],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "GATEWAY_RESTART_LOCK_FILE": str(lock_file),
                "GATEWAY_DEPLOYMENT_DIR": str(tmp_path / "deployments"),
                "GATEWAY_DEPLOY_PLAN_FILE": str(tmp_path / "plan.json"),
            },
        )

    assert result.returncode != 0
    assert "another gateway restart is already running" in result.stderr
    assert "Hydrating gateway env" not in result.stdout
    assert "Stopping existing gateway" not in result.stdout


def test_restart_recovers_lock_inherited_by_detached_relay(tmp_path: Path) -> None:
    if not Path("/proc/self/fd").exists() or shutil.which("flock") is None:
        return

    lock_file = tmp_path / "gateway-restart.lock"
    holder_code = """
import fcntl
import sys
import time

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    print("ready", flush=True)
    time.sleep(30)
"""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder_code,
            "gateway.utils.tee_inter_enclave_relay",
            str(lock_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        result = subprocess.run(
            ["bash", str(ROOT / "gw_restart.sh")],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={
                **os.environ,
                "GATEWAY_ROOT": str(tmp_path / "missing-gateway"),
                "GATEWAY_RESTART_LOCK_FILE": str(lock_file),
                "GATEWAY_RESTART_RECOVERY_LOCK_FILE": str(
                    tmp_path / "gateway-restart.recovery.lock"
                ),
                "GATEWAY_DEPLOYMENT_DIR": str(tmp_path / "deployments"),
                "GATEWAY_DEPLOY_PLAN_FILE": str(tmp_path / "plan.json"),
            },
        )
    finally:
        holder.terminate()
        holder.wait(timeout=5)

    assert result.returncode != 0
    assert (
        "Recovering gateway restart lock inherited by a detached runtime process"
        in result.stdout
    )
    assert "another gateway restart is already running" not in result.stderr


def test_gateway_fallback_logs_stay_outside_canonical_checkout(tmp_path: Path) -> None:
    checkout_cwd = tmp_path / "checkout"
    fallback_dir = tmp_path / "legacy-flat" / "gateway" / "logs" / "tee_fallback"
    checkout_cwd.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gateway.utils.logger import FALLBACK_LOG_DIR; print(FALLBACK_LOG_DIR)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=checkout_cwd,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "GATEWAY_TEE_FALLBACK_LOG_DIR": str(fallback_dir),
        },
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.splitlines()[-1]) == fallback_dir
    assert fallback_dir.is_dir()
    assert not (checkout_cwd / "gateway").exists()


def test_gateway_restart_pins_all_build_provenance_to_selected_sha() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    role_builder = (ROOT / "gateway" / "tee" / "build_role_enclaves.sh").read_text(
        encoding="utf-8"
    )
    for assignment in (
        'export GITHUB_SHA="$GATEWAY_DEPLOY_SHA"',
        'export GITHUB_COMMIT="$GATEWAY_DEPLOY_SHA"',
        'export ATTESTED_RUNTIME_COMMIT_SHA="$GATEWAY_DEPLOY_SHA"',
        'export RESEARCH_LAB_RUNTIME_SOURCE_ROOT="$LEADPOET_REPO_ROOT"',
        'export GATEWAY_BUILD_INFO_GIT_ROOT="$LEADPOET_REPO_ROOT"',
    ):
        assert assignment in script
    assert 'printf \'%s\\n\' "$GATEWAY_DEPLOY_SHA" > "$GATEWAY_ROOT/.source_commit"' in script
    assert 'http://localhost:8000/build-info' in script
    # Preserve the current exact build evidence long enough to prove a
    # content-addressed same-release restore.  A cache miss performs the old
    # cold path, which deletes each output/measurement immediately before it
    # is rebuilt.
    assert 'rm -f "$GATEWAY_TEE_EIF_ROOT"/enclave-build-*.json' not in script
    assert 'rm -f "$output" "$measurements"' in role_builder
    assert "release_archive_v2" in role_builder
    assert (
        'enclave-build-gateway.json' in script
        or 'build_role_enclaves.sh' in script
    )


def test_explicit_deployment_sha_beats_stale_build_info(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stale = "1" * 40
    selected = "2" * 40
    gateway_root = tmp_path / "gateway"
    gateway_root.mkdir()
    (gateway_root / "BUILD_INFO.json").write_text(
        json.dumps({"git_commit": stale}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATTESTED_RUNTIME_COMMIT_SHA", selected)
    assert (
        build_identity.resolve_commit(
            gateway_root=gateway_root,
            source_root=tmp_path,
        )
        == selected
    )


def test_generated_gateway_artifacts_are_ignored_by_deploy_checkout() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for path in (
        "gateway/_attested_runtime/",
        "gateway/_enclave_source/",
        "gateway/_enclave_wheelhouse/",
        "gateway/.source_commit",
        "gateway/BUILD_INFO.json",
    ):
        assert path in ignore


def test_gateway_docker_image_copies_complete_runtime_package_graph() -> None:
    dockerfile = (ROOT / "gateway" / "Dockerfile").read_text(encoding="utf-8")
    for path in (
        "leadpoet_canonical",
        "leadpoet_verifier",
        "research_lab",
        "qualification",
        "validator_models",
        "Leadpoet",
        "schemas",
    ):
        assert f"COPY {path}/ ./{path}/" in dockerfile


def test_gateway_restart_does_not_require_closed_model_identity() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert "RESEARCH_LAB_PRIVATE_REPO_BRANCH" not in script
    assert "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI" not in script
    assert "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID" not in script


def test_gateway_restart_wires_automatic_signed_dev_snapshot_refresh() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert (
        'export RESEARCH_LAB_DEV_SNAPSHOT_AUTO_REFRESH_ENABLED="${'
        'RESEARCH_LAB_DEV_SNAPSHOT_AUTO_REFRESH_ENABLED:-true}"'
    ) in script
    assert (
        'export RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED="${'
        'RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED:-true}"'
    ) in script
    assert (
        'export RESEARCH_LAB_DEV_SNAPSHOT_KMS_KEY_ID="${'
        'RESEARCH_LAB_DEV_SNAPSHOT_KMS_KEY_ID:-alias/'
        'leadpoet-research-lab-artifact-signing}"'
    ) in script


@pytest.mark.parametrize(
    "failure_reason",
    ["status_true", "durable_secret_drift", "locked_channel_drift"],
)
def test_failed_miner_maintenance_runtime_gate_stops_before_terminal_success(
    tmp_path: Path,
    failure_reason: str,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    start = script.index(
        'GATEWAY_DEPLOY_STAGE="miner_maintenance_runtime_verify"'
    )
    terminal = 'finalize_deployment_record succeeded "$GATEWAY_DEPLOY_STAGE" >/dev/null'
    end = script.index(terminal, start) + len(terminal)
    gate = script[start:end]
    stopped = tmp_path / "runtime-stopped"
    succeeded = tmp_path / "terminal-success"
    fake_python = tmp_path / "verify-runtime"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' "
        + shlex.quote(failure_reason)
        + " >&2\nexit 86\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    harness = tmp_path / "runtime-gate.sh"
    harness.write_text(
        "#!/bin/bash\nset -Eeuo pipefail\n"
        f"LEADPOET_REPO_ROOT={shlex.quote(str(tmp_path))}\n"
        f"GATEWAY_PYTHON_BIN={shlex.quote(str(fake_python))}\n"
        "GATEWAY_DEPLOY_SHA='" + "a" * 40 + "'\n"
        f"GATEWAY_V2_RELEASE_MANIFEST={shlex.quote(str(tmp_path / 'release.json'))}\n"
        f"stop_failed_miner_maintenance_runtime() {{ touch {shlex.quote(str(stopped))}; }}\n"
        f"finalize_deployment_record() {{ touch {shlex.quote(str(succeeded))}; }}\n"
        + gate
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert stopped.exists()
    assert not succeeded.exists()


def test_failed_miner_maintenance_cleanup_kills_all_new_runtime_groups(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    cleanup = _shell_function_source(
        script, "stop_failed_miner_maintenance_runtime"
    )
    late_runtime = tmp_path / "late-runtime"
    ready_paths = [tmp_path / f"ready-{index}" for index in range(3)]
    harness = tmp_path / "runtime-cleanup.sh"
    launch_code = (
        "import os,signal,sys,time\n"
        "os.setsid()\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(7)\n"
        "open(sys.argv[2], 'w').write('unsafe')\n"
        "time.sleep(30)\n"
    )
    launch_lines = []
    pid_names = (
        "GATEWAY_LAUNCHER_PID",
        "TEE_EGRESS_FORWARDER_PID",
        "INTER_ENCLAVE_RELAY_PID",
    )
    for name, ready in zip(pid_names, ready_paths):
        launch_lines.extend(
            [
                "python3 -c "
                + shlex.quote(launch_code)
                + " "
                + shlex.quote(str(ready))
                + " "
                + shlex.quote(str(late_runtime))
                + " &",
                f"{name}=$!",
            ]
        )
    harness.write_text(
        "#!/bin/bash\nset -Eeuo pipefail\n"
        + cleanup
        + "\n"
        + "sudo() { return 0; }\n"
        + "GATEWAY_ROOT="
        + shlex.quote(str(tmp_path / "gateway"))
        + "\n"
        + "\n".join(launch_lines)
        + "\n"
        + "for _ in $(seq 1 200); do\n"
        + "  [ -s "
        + shlex.quote(str(ready_paths[0]))
        + " ] && [ -s "
        + shlex.quote(str(ready_paths[1]))
        + " ] && [ -s "
        + shlex.quote(str(ready_paths[2]))
        + " ] && break\n"
        + "  sleep 0.01\ndone\n"
        + "stop_failed_miner_maintenance_runtime\n"
        + "sleep 2.5\n"
        + "test ! -e "
        + shlex.quote(str(late_runtime))
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert not late_runtime.exists()
