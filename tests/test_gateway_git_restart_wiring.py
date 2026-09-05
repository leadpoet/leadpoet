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
    follow = script[start : script.index("start_gateway_ancestry_checkpoint_bootstrap() {", start)]

    assert follow.index('if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ]') < follow.index(
        "restart_release_supersession_v2.py"
    )
    assert 'git -C "$LEADPOET_REPO_ROOT" archive "$latest_sha"' in follow
    assert 'GATEWAY_RESTART_LOCK_HELD=1' in follow
    assert 'GATEWAY_RELEASE_SUPERSESSION_COUNT="$next_count"' in follow
    assert follow.index("cancel_gateway_offline_artifact_prepare") < follow.index(
        'bash "$superseding_tree/gw_restart.sh"'
    )
    assert follow.index("cancel_gateway_ancestry_checkpoint_bootstrap") < follow.index(
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
    assert '--validator-output "$GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST"' in release_build
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
    assert "GATEWAY_EXACT_COMMIT_HELPER" in script
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
            'echo "Resetting gateway PCR0 builder checkout/cache"',
            'echo "Loading gateway runtime env for AWS/ECR checks"',
            'echo "Building/restarting TEE enclave"',
            'bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"',
            'echo "Installing Python dependencies"',
            'echo "Relaunching gateway with cloned runtime env"',
            'unset RESEARCH_LAB_EVIDENCE_PROXY_URL RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH',
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
    assert (
        'GATEWAY_V2_RELEASE_REQUIREMENTS="$GATEWAY_V2_RELEASE_REQUIREMENTS"'
        in reexec
    )
    assert (
        'GATEWAY_PREPARED_V2_RELEASE_REQUIREMENTS='
        '"$GATEWAY_PREPARED_V2_RELEASE_REQUIREMENTS"'
        in reexec
    )
    assert (
        'GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS='
        '"$GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS"'
        in reexec
    )
    assert 'GATEWAY_V2_RELEASE_BUCKET="$GATEWAY_V2_RELEASE_BUCKET"' in reexec
    assert 'GATEWAY_V2_RELEASE_PREFIX="$GATEWAY_V2_RELEASE_PREFIX"' in reexec


def test_gateway_restart_installs_preselected_release_lineage_after_activation() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    reexec = script.index("GATEWAY_RESTART_PHASE=post_activate")
    candidate = script.index('GATEWAY_DEPLOY_SHA="$(deployment_field target_sha)"')
    revalidate = script.index(
        "if ! ensure_activated_gateway_release_lineage;", candidate
    )
    enclave_build = script.index('echo "Building/restarting TEE enclave"')

    assert reexec < candidate < revalidate < enclave_build
    assert "rev-parse --verify 'origin/main^{commit}'" in script
    assert "merge-base --is-ancestor" in script
    assert '"$GATEWAY_DEPLOY_SHA" "$authority_commit"' in script
    installer = _shell_function_source(
        script, "ensure_activated_gateway_release_lineage"
    )
    assert "validate_active_release_requirements_v2" in installer
    assert "validate_compact_release_lineage_v2" in installer
    assert 'expected_current_commit=expected_commit' in installer
    assert (
        'set(lineage["releases"]) != set(requirements["required_commits"])'
        in installer
    )
    assert 'os.replace(temporary, destination)' in installer
    assert 'gateway.tee.release_channel_v2' not in installer
    assert "list_objects" not in installer


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


def test_gateway_restart_repairs_and_proves_automatic_weight_input() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    storage_preflight = script.index("--storage-read-preflight")
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    runtime_ready = script.index(
        '"$GATEWAY_PYTHON_BIN" -m gateway.tee.verify_v2_runtime_ready'
    )
    cutover = script.index(
        'echo "Executing the one-time receipt-backed stateful epoch cutover"'
    )
    repair = script.index(
        "\nrepair_chain_settlements_and_prepare_current_weight_input\n",
        cutover,
    )
    launch = script.index(
        'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main'
    )
    base_health = script.index(
        "curl -fsS http://localhost:8000/health >/dev/null"
    )
    http_handoff = script.index(
        "--gateway-url http://localhost:8000"
    )
    http_timeout = script.index(
        '--http-timeout-seconds "$GATEWAY_WEIGHT_INPUT_HTTP_TIMEOUT_SECONDS"',
        http_handoff,
    )
    install = script.index(
        'GATEWAY_DEPLOY_STAGE="host_restart_script_install"'
    )

    assert "GATEWAY_WEIGHT_INPUT_HTTP_TIMEOUT_SECONDS=360" in script
    assert (
        'GATEWAY_WEIGHT_INPUT_REPAIR_MAX_ATTEMPTS="${'
        'GATEWAY_WEIGHT_INPUT_REPAIR_MAX_ATTEMPTS:-3}"'
    ) in script
    assert (
        'GATEWAY_WEIGHT_INPUT_REPAIR_RETRY_SECONDS="${'
        'GATEWAY_WEIGHT_INPUT_REPAIR_RETRY_SECONDS:-5}"'
    ) in script
    assert "repair_and_verify_gateway_weight_input()" in script
    assert "repair_chain_settlements_and_prepare_current_weight_input()" in script
    repair_function = script[
        script.index(
            "repair_chain_settlements_and_prepare_current_weight_input()"
        ) : script.index("\ninstall_gateway_python_dependencies()")
    ]
    _ordered_offsets(
        repair_function,
        (
            "--repair-chain-settlements",
            "verify_gateway_active_ancestry_checkpoints",
            'repair_and_verify_gateway_weight_input "$requested_epoch"',
        ),
    )
    assert (
        'for attempt in $(seq 1 "$GATEWAY_WEIGHT_INPUT_REPAIR_MAX_ATTEMPTS")'
        in script
    )
    assert (
        'sleep "$GATEWAY_WEIGHT_INPUT_REPAIR_RETRY_SECONDS"'
        in script
    )
    assert 'if [ "$status" -ne 75 ]; then' not in script
    assert (
        storage_preflight
        < shutdown
        < runtime_ready
        < cutover
        < repair
        < launch
        < base_health
        < http_handoff
        < http_timeout
        < install
    )
    assert 'GATEWAY_DEPLOY_STAGE="validator_weight_input_repair"' in script
    assert (
        'GATEWAY_DEPLOY_STAGE="validator_weight_input_http_check"' in script
    )


def test_gateway_weight_storage_preflight_uses_target_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    dependencies = script.index(
        'GATEWAY_DEPLOY_STAGE="dependency_preflight"'
    )
    stage = script.index(
        'GATEWAY_DEPLOY_STAGE="validator_weight_input_storage_preflight"'
    )
    command = script.index("--storage-read-preflight", stage)
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    preflight_block = script[stage:shutdown]

    assert dependencies < stage < command < shutdown
    assert '. "$ENV_CLONE"' in preflight_block
    assert "ast.parse(" in preflight_block
    assert (
        'required_arguments = {"--storage-read-preflight", "--epoch"}'
        in preflight_block
    )
    assert (
        'if [ -z "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \\\n'
        '        || [ "$PREPARED_GATEWAY_SHA" = '
        '"$ORIGIN_MAIN_GATEWAY_SHA" ]; then'
        in preflight_block
    )
    assert (
        "selected current release lacks the required weight storage preflight"
        in preflight_block
    )
    assert (
        "Selected attested rollback release predates the optional weight "
        "storage preflight"
        in preflight_block
    )
    assert (
        "run_prepared_gateway_module \\\n"
        "          gateway.tee.verify_weight_submission_ready_v2"
        in preflight_block
    )
    assert "gateway_weight_preflight_epoch_from_restart_report" in preflight_block
    assert '--epoch "$GATEWAY_WEIGHT_STORAGE_PREFLIGHT_EPOCH"' in preflight_block
    assert "gateway_ancestry_safe_epoch_from_report" in preflight_block
    assert 'report["ancestry_safe_epoch"]' in script
    assert "Pinned active ancestry bootstrap to proven-safe epoch" in preflight_block
    assert "Gateway remains running; production shutdown has not started." in (
        preflight_block
    )


def test_gateway_weight_preflight_reuses_exact_restart_epoch() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = _shell_function_source(
        script,
        "gateway_weight_preflight_epoch_from_restart_report",
    )
    manifest = json.loads(
        (ROOT / "config/stateful-epoch-cutover-sn71.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = {
        "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": manifest["network_genesis_hash"],
        "netuid": 71,
        "head_kind": "exact",
        "block_hash": "0x" + "1" * 64,
        "current_block": 100,
        "last_epoch_block": 95,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 24020,
        "tempo": 360,
        "blocks_since_last_step": 5,
        "observed_at": "2026-09-04T00:00:00+00:00",
    }
    report = {
        "schema_version": "leadpoet.restart_epoch_gate.v1",
        "restart_allowed": True,
        "snapshot": snapshot,
    }
    harness = f"""set -euo pipefail
{helper_source}
GATEWAY_PREFLIGHT_TREE="$1"
GATEWAY_PYTHON_BIN="$2"
GATEWAY_STATEFUL_CUTOVER_MANIFEST="$3"
gateway_weight_preflight_epoch_from_restart_report "$4"
"""
    arguments = [
        "bash",
        "-c",
        harness,
        "gateway-restart-epoch-reuse-test",
        str(ROOT),
        sys.executable,
        str(ROOT / "config/stateful-epoch-cutover-sn71.json"),
        json.dumps(report, sort_keys=True),
    ]

    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "24073"

    report["snapshot"]["network_genesis_hash"] = "0x" + "2" * 64
    arguments[-1] = json.dumps(report, sort_keys=True)
    rejected = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert rejected.returncode != 0
    assert "snapshot and cutover genesis hashes differ" in rejected.stderr


def test_gateway_restart_cutover_hook_is_explicit_and_fail_closed() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert 'GATEWAY_STATEFUL_CUTOVER_CEREMONY="${' in script
    assert 'GATEWAY_STATEFUL_CUTOVER_CEREMONY must be 0 or 1' in script
    preflight = script.index(
        'echo "Validating the one-time receipt-backed cutover before production shutdown"'
    )
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    execution = script.index(
        'echo "Executing the one-time receipt-backed stateful epoch cutover"'
    )
    launch = script.index('setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main')
    assert preflight < shutdown < execution < launch
    assert '"already_stateful_staged"' in script
    assert '"already_stateful_active"' in script
    assert 'report.get("would_write") is not False' in script
    assert '--use-attested-historical-predecessor' in script
    assert (
        'report.get("predecessor_kind") != '
        '"legacy_finalized_chain_migration_v2"'
    ) in script
    assert '"attested_historical_finalization_v2"' not in script
    assert '--confirm-all-writers-stopped' in script
    assert '--confirm-stateful-release-prepared' in script
    assert script.count(
        '--validator-release-manifest '
        '"$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST"'
    ) == 3
    assert (
        '--validator-release-manifest '
        '"$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST"'
        in script[preflight:shutdown]
    )
    assert (
        'load_validator_release_manifest_v2(sys.argv[1])'
        in script[preflight:shutdown]
    )
    assert 'report.get("status") != "stateful_active"' in script
    assert 'json.loads(sys.argv[1])' in script[execution:launch]
    assert 'json.loads(os.environ["CUTOVER_PREFLIGHT_REPORT"])' not in script
    assert 'json.loads(os.environ["CUTOVER_STAGE_REPORT"])' not in script
    assert 'json.loads(os.environ["CUTOVER_ACTIVATION_REPORT"])' not in script
    assert (
        '"$GATEWAY_PYTHON_BIN" - "$CUTOVER_PREFLIGHT_REPORT"'
        in script[preflight:shutdown]
    )
    assert (
        '"$GATEWAY_PYTHON_BIN" - "$CUTOVER_STAGE_REPORT"'
        in script[execution:launch]
    )
    assert (
        '"$GATEWAY_PYTHON_BIN" - "$CUTOVER_ACTIVATION_REPORT"'
        in script[execution:launch]
    )


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
    assert "GATEWAY_STATEFUL_CUTOVER_SUPABASE_TIMEOUT_SECONDS=120" in script
    assert script.count(
        'export SUPABASE_TIMEOUT_SECONDS="'
        '$GATEWAY_STATEFUL_CUTOVER_SUPABASE_TIMEOUT_SECONDS"'
    ) == 3


def test_gateway_restart_loads_one_canonical_cutover_manifest() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    assert (
        'GATEWAY_STATEFUL_CUTOVER_MANIFEST="/home/ec2-user/.config/'
        'leadpoet/stateful-epoch-cutover.json"'
    ) in script
    assert 'unset LEADPOET_SUBNET_EPOCH_CUTOVER_JSON' in script
    assert (
        'export LEADPOET_SUBNET_EPOCH_CUTOVER_PATH="$GATEWAY_STATEFUL_CUTOVER_MANIFEST"'
        in script
    )


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


def test_gateway_weight_input_repair_runs_from_canonical_repo_root() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    repair_stage = script.index(
        'GATEWAY_DEPLOY_STAGE="validator_weight_input_repair"'
    )
    repair_command = script.index(
        "-m gateway.tee.verify_weight_submission_ready_v2 \\\n"
        '        --repair "${epoch_args[@]}"',
    )
    repair_call = script.index(
        "\nrepair_chain_settlements_and_prepare_current_weight_input\n",
        repair_stage,
    )
    repair_function = script.index(
        "repair_and_verify_gateway_weight_input()"
    )
    repair_block = script[repair_function:repair_command]

    assert 'cd "$LEADPOET_REPO_ROOT"\n' in repair_block
    assert repair_command < repair_stage < repair_call


def test_gateway_restart_preserves_safe_ancestry_epoch_across_reexec() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    reexec_start = script.index("exec env ", script.index("GATEWAY_DEPLOY_STAGE=\"restart_reexec\""))
    reexec = script[reexec_start : script.index("\nfi", reexec_start)]

    assert (
        'GATEWAY_ANCESTRY_SAFE_EPOCH="$GATEWAY_ANCESTRY_SAFE_EPOCH"'
        in reexec
    )
    assert (
        'verify_gateway_active_ancestry_checkpoints '
        '"$GATEWAY_ANCESTRY_SAFE_EPOCH"'
        in script
    )
    assert 'epoch_args=(--epoch "$epoch")' in script
    candidate_env = script.index(
        '. "$ENV_CLONE"',
        script.index('GATEWAY_RESTART_PHASE=post_activate'),
    )
    candidate_recovery = script.index(
        "ensure_gateway_ancestry_safe_epoch",
        candidate_env,
    )
    enclave_build = script.index(
        'GATEWAY_DEPLOY_STAGE="attested_runtime_and_enclave_build"',
        candidate_recovery,
    )
    postcheckpoint = script.index(
        'verify_gateway_active_ancestry_checkpoints '
        '"$GATEWAY_ANCESTRY_SAFE_EPOCH"',
        enclave_build,
    )
    assert candidate_env < candidate_recovery < enclave_build < postcheckpoint


def test_gateway_candidate_recovers_frontier_from_n_minus_one_controller(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = "\n\n".join(
        _shell_function_source(script, name)
        for name in (
            "gateway_ancestry_safe_epoch_from_report",
            "ensure_gateway_ancestry_safe_epoch",
            "verify_gateway_active_ancestry_checkpoints",
        )
    )
    fake_python = tmp_path / "gateway-python"
    calls = tmp_path / "calls"
    fake_python.write_text(
        f"""#!/bin/bash
set -euo pipefail
if [ "${{1:-}}" = "-" ]; then
  exec {shlex.quote(sys.executable)} "$@"
fi
printf '%s\\n' "$*" >> "$FAKE_CALLS"
if [ "${{3:-}}" = "--storage-read-preflight" ]; then
  printf '%s\\n' '{{"schema_version":"leadpoet.weight_submission_storage_readiness.v2","status":"readable","epoch":24307,"ancestry_safe_epoch":24303}}'
elif [ "${{2:-}}" = "gateway.tee.bootstrap_active_ancestry_checkpoints_v2" ]; then
  printf '%s\\n' '{{"status":"complete","epoch_id":24303}}'
else
  exit 2
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    harness = f"""set -euo pipefail
record_gateway_restart_timing() {{ :; }}
{helper_source}
GATEWAY_PYTHON_BIN="$1"
LEADPOET_REPO_ROOT="$2"
GATEWAY_V2_RELEASE_MANIFEST="$3"
GATEWAY_ANCESTRY_SAFE_EPOCH=""
ensure_gateway_ancestry_safe_epoch
verify_gateway_active_ancestry_checkpoints "$GATEWAY_ANCESTRY_SAFE_EPOCH"
printf 'safe_epoch=%s\\n' "$GATEWAY_ANCESTRY_SAFE_EPOCH"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "gateway-n-minus-one-frontier-test",
            str(fake_python),
            str(tmp_path),
            str(tmp_path / "release.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "FAKE_CALLS": str(calls)},
    )

    assert completed.returncode == 0, completed.stderr
    assert "safe_epoch=24303" in completed.stdout
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "-m gateway.tee.verify_weight_submission_ready_v2 --storage-read-preflight",
        (
            "-m gateway.tee.bootstrap_active_ancestry_checkpoints_v2 "
            f"--release-manifest {tmp_path / 'release.json'} --epoch 24303"
        ),
    ]


def test_gateway_weight_preparation_repeats_after_epoch_advance(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = "\n\n".join(
        _shell_function_source(script, name)
        for name in (
            "verify_gateway_active_ancestry_checkpoints",
            "repair_and_verify_gateway_weight_input",
            "repair_chain_settlements_and_prepare_current_weight_input",
        )
    )
    fake_python = tmp_path / "gateway-python"
    calls = tmp_path / "calls"
    chain_count = tmp_path / "chain-count"
    repair_count = tmp_path / "repair-count"
    fake_python.write_text(
        f"""#!/bin/bash
set -euo pipefail
if [ "${{1:-}}" = "-" ]; then
  exec {shlex.quote(sys.executable)} "$@"
fi
printf '%s\\n' "$*" >> "$FAKE_CALLS"
if [ "${{2:-}}" = "gateway.tee.bootstrap_active_ancestry_checkpoints_v2" ]; then
  printf '%s\\n' '{{"status":"complete"}}'
elif [ "${{3:-}}" = "--repair-chain-settlements" ]; then
  count=$(( $(cat "$FAKE_CHAIN_COUNT" 2>/dev/null || printf 0) + 1 ))
  printf '%s' "$count" > "$FAKE_CHAIN_COUNT"
  epoch=$((99 + count))
  printf '{{"schema_version":"leadpoet.chain_realized_settlement_repair.v1","status":"ready","epoch":%s,"observed_epoch":%s,"settled_through_epoch":%s}}\\n' "$epoch" "$epoch" "$((epoch - 1))"
elif [ "${{3:-}}" = "--repair" ]; then
  count=$(( $(cat "$FAKE_REPAIR_COUNT" 2>/dev/null || printf 0) + 1 ))
  printf '%s' "$count" > "$FAKE_REPAIR_COUNT"
  epoch="${{5}}"
  observed="$epoch"
  if [ "$count" = "1" ]; then
    observed=$((epoch + 1))
  fi
  printf '{{"schema_version":"leadpoet.weight_submission_readiness.v2","status":"ready","epoch":%s,"observed_epoch":%s}}\\n' "$epoch" "$observed"
else
  exit 2
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    harness = f"""set -euo pipefail
record_gateway_restart_timing() {{ :; }}
{helper_source}
GATEWAY_PYTHON_BIN="$1"
LEADPOET_REPO_ROOT="$2"
GATEWAY_V2_RELEASE_MANIFEST="$3"
GATEWAY_WEIGHT_INPUT_REPAIR_MAX_ATTEMPTS=3
GATEWAY_WEIGHT_INPUT_REPAIR_RETRY_SECONDS=0
GATEWAY_WEIGHT_INPUT_REPAIR_REPORT=""
repair_chain_settlements_and_prepare_current_weight_input
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "gateway-weight-epoch-stability-test",
            str(fake_python),
            str(tmp_path),
            str(tmp_path / "release.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "FAKE_CALLS": str(calls),
            "FAKE_CHAIN_COUNT": str(chain_count),
            "FAKE_REPAIR_COUNT": str(repair_count),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "-m gateway.tee.verify_weight_submission_ready_v2 --repair-chain-settlements",
        (
            "-m gateway.tee.bootstrap_active_ancestry_checkpoints_v2 "
            f"--release-manifest {tmp_path / 'release.json'} --epoch 100"
        ),
        "-m gateway.tee.verify_weight_submission_ready_v2 --repair --epoch 100",
        "-m gateway.tee.verify_weight_submission_ready_v2 --repair-chain-settlements",
        (
            "-m gateway.tee.bootstrap_active_ancestry_checkpoints_v2 "
            f"--release-manifest {tmp_path / 'release.json'} --epoch 101"
        ),
        "-m gateway.tee.verify_weight_submission_ready_v2 --repair --epoch 101",
    ]


def test_gateway_restart_v2_preflight_runs_target_commit_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    materialize = script.index(
        'echo "Materializing the prepared commit for pre-shutdown V2 tooling"'
    )
    local_release = script.index(
        'bash "$GATEWAY_PREFLIGHT_TREE/gateway/tee/build_local_release_v2.sh"'
    )
    restart_window = script.index(
        'echo "Capturing the official subnet restart window before release acquisition"'
    )
    credential_envelopes = script.index(
        'run_prepared_gateway_module gateway.tee.prepare_gateway_envelopes_v2'
    )
    artifact_prepare = script.index(
        'echo "Preparing exact hash-locked V2 build artifacts during release acquisition"'
    )
    checkpoint_prepare = script.index(
        "if ! prepare_gateway_ancestry_checkpoint_bootstrap; then"
    )
    artifact_start = script.index(
        "if ! start_gateway_offline_artifact_prepare; then",
        artifact_prepare,
    )
    artifact_join = script.index(
        "if ! wait_for_gateway_offline_artifact_prepare; then",
        artifact_start,
    )
    release_ready = script.index(
        'record_gateway_restart_timing "local_release_ready"',
        local_release,
    )
    preflight = script.index(
        'echo "Validating the prepared V2 release before production shutdown"'
    )
    checkpoint_join = script.index(
        "if ! wait_for_gateway_ancestry_checkpoint_bootstrap; then"
    )
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    dependency_preflight = script.index(
        'echo "Installing gateway host Python dependencies before production shutdown"'
    )
    safe_frontier = script.index(
        "Pinned active ancestry bootstrap to proven-safe epoch"
    )
    checkpoint_start = script.index(
        "if ! start_gateway_ancestry_checkpoint_bootstrap; then",
        safe_frontier,
    )
    assert (
        materialize
        < restart_window
        < checkpoint_prepare
        < artifact_prepare
        < artifact_start
        < artifact_join
        < local_release
        < release_ready
        < credential_envelopes
        < dependency_preflight
        < safe_frontier
        < checkpoint_start
        < preflight
        < checkpoint_join
    )
    assert preflight < shutdown
    assert dependency_preflight < preflight < shutdown
    assert (
        script.index(
            'git -C "$LEADPOET_REPO_ROOT" archive "$PREPARED_GATEWAY_SHA"'
        )
        < local_release
    )
    assert (
        'PYTHONPATH="$LEADPOET_REPO_ROOT" '
        "python3 -m gateway.tee.release_channel_v2"
    ) not in script
    assert 'cd "$GATEWAY_PREFLIGHT_TREE"' in script
    assert (
        '(\n  cd "$GATEWAY_PREFLIGHT_TREE"\n'
        '  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \\\n'
        '  "$GATEWAY_PYTHON_BIN" - "$ENV_CLONE" '
        '"$GATEWAY_V2_CONFIG_DIR/gateway-v2-env-transition.json"'
    ) in script
    assert "scrub_parent_environment_file_v2" in script
    assert credential_envelopes < script.index("scrub_parent_environment_file_v2")
    assert script.index("gateway.tee.restart_preflight_v2") < shutdown
    assert script.index('--deploy-commit "$PREPARED_GATEWAY_SHA"') < shutdown
    assert (
        script.index(
            '--release-manifest "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"'
        )
        < shutdown
    )
    assert script.index('--parent-env-file "$ENV_CLONE"') < shutdown
    acceptance = script.index("V2_PREFLIGHT_ACCEPTANCE_ARGS=()")
    assert acceptance < shutdown
    assert 'if [ -n "${GATEWAY_HISTORICAL_TOPOLOGY_HASH:-}" ]; then' in script[
        acceptance:shutdown
    ]
    assert "--acceptance-corpus-manifest" in script[acceptance:shutdown]
    assert "--acceptance-corpus-root" in script[acceptance:shutdown]
    assert '"${V2_PREFLIGHT_ACCEPTANCE_ARGS[@]}"' in script[acceptance:shutdown]
    assert script.index('--topology-mode "${GATEWAY_TEE_TOPOLOGY_MODE:-full}"') < shutdown
    assert script.index("prepare_offline_artifacts_v2.sh") < shutdown
    assert script.index("bootstrap_active_ancestry_checkpoints_v2.py") < shutdown
    assert script.count('bash "$prepare_script"') == 1
    assert (
        'echo "Preparing exact hash-locked V2 build artifacts before production shutdown"'
        not in script
    )
    assert script.index('pkill -9 -f "python3 -u -m gateway.main"') > shutdown


def test_gateway_restart_isolates_candidate_release_until_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    prepare = script.index(
        'echo "Materializing the prepared commit for pre-shutdown V2 tooling"'
    )
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    pre_shutdown = script[prepare:shutdown]

    assert (
        '--gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"'
        in pre_shutdown
    )
    release_start = pre_shutdown.index(
        'echo "Building the exact local gateway and validator runtime identities"'
    )
    release_end = pre_shutdown.index(
        'record_gateway_restart_timing "local_release_ready"', release_start
    )
    release_acquisition = pre_shutdown[release_start:release_end]
    assert (
        '--lineage-output "$GATEWAY_PREPARED_V2_RELEASE_LINEAGE"'
        not in release_acquisition
    )
    assert "--lineage-repository" not in release_acquisition
    assert "--lineage-authority-commit" not in release_acquisition
    assert (
        '--release-manifest "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"'
        in pre_shutdown
    )
    assert '--gateway-output "$GATEWAY_V2_RELEASE_MANIFEST"' not in pre_shutdown
    assert '--lineage-output "$GATEWAY_V2_RELEASE_LINEAGE"' not in pre_shutdown
    assert '--release-manifest "$GATEWAY_V2_RELEASE_MANIFEST"' not in pre_shutdown

    selector = _shell_function_source(
        script, "prepare_gateway_active_release_lineage"
    )
    assert "gateway.tee.prepare_active_release_lineage_v2" in selector
    assert "gateway-final" in selector
    for argument in (
        '--candidate-commit "$PREPARED_GATEWAY_SHA"',
        '--authority-commit "$authority_commit"',
        '--restart-invocation-id "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID"',
        '--running-gateway-manifest "$running_gateway_manifest"',
        '--epoch "$GATEWAY_ANCESTRY_SAFE_EPOCH"',
        '--netuid "${BITTENSOR_NETUID:-71}"',
        '--repository "$LEADPOET_REPO_ROOT"',
        '--lineage-id "$lineage_id"',
        '--bucket "$GATEWAY_V2_RELEASE_BUCKET"',
        '--prefix "$GATEWAY_V2_RELEASE_PREFIX"',
    ):
        assert argument in selector
    assert (
        '--validator-requirements "$GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS"'
        in selector
    )
    assert (
        '--requirements-output "$GATEWAY_PREPARED_V2_RELEASE_REQUIREMENTS"'
        in selector
    )
    assert '--lineage-output "$GATEWAY_PREPARED_V2_RELEASE_LINEAGE"' in selector
    assert "--validator-hotkey-config" not in selector
    assert "--chain-signing-profile" not in selector
    assert "if ! prepare_gateway_active_release_lineage; then" in pre_shutdown
    assert "list_objects" not in selector

    revalidator = _shell_function_source(
        script, "ensure_activated_gateway_release_lineage"
    )
    assert '"$GATEWAY_V2_RELEASE_MANIFEST"' in revalidator
    assert '"$GATEWAY_V2_RELEASE_REQUIREMENTS"' in revalidator
    assert '"$GATEWAY_V2_RELEASE_LINEAGE"' in revalidator
    assert (
        '"$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \\\n'
        '    "$GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST" \\\n'
        '    "$GATEWAY_PREPARED_V2_RELEASE_REQUIREMENTS" \\\n'
        '    "$GATEWAY_PREPARED_V2_RELEASE_LINEAGE"'
        in revalidator
    )


def test_gateway_restart_bounds_active_ancestry_before_weight_preparation() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    runtime_ready = script.index(
        'record_gateway_restart_timing "v2_runtime_ready"'
    )
    postcheckpoint = script.index(
        'verify_gateway_active_ancestry_checkpoints '
        '"$GATEWAY_ANCESTRY_SAFE_EPOCH"',
        runtime_ready,
    )
    repair = script.index(
        "repair_chain_settlements_and_prepare_current_weight_input\n",
        postcheckpoint,
    )

    checkpoint_prepare = script.index(
        "prepare_gateway_ancestry_checkpoint_bootstrap"
    )
    safe_frontier = script.index(
        "Pinned active ancestry bootstrap to proven-safe epoch"
    )
    checkpoint_start = script.index(
        "if ! start_gateway_ancestry_checkpoint_bootstrap; then",
        safe_frontier,
    )
    checkpoint_join = script.index(
        "if ! wait_for_gateway_ancestry_checkpoint_bootstrap; then",
        checkpoint_start,
    )
    docker_guard = script.index(
        "-m validator_tee.host.docker_operation_guard_v2",
        checkpoint_join,
    )
    memory_wait = script.index("wait_for_gateway_build_memory", docker_guard)
    paired_liveness_handoff = script.index(
        "if ! wait_for_paired_gateway_destructive_handoff; then",
        memory_wait,
    )
    active_release_selection = script.index(
        "if ! prepare_gateway_active_release_lineage; then",
        paired_liveness_handoff,
    )
    assert (
        checkpoint_prepare
        < safe_frontier
        < checkpoint_start
        < checkpoint_join
        < docker_guard
        < memory_wait
        < paired_liveness_handoff
        < active_release_selection
        < shutdown
    )
    assert runtime_ready < postcheckpoint < repair
    assert '--release-manifest "$GATEWAY_V2_RELEASE_MANIFEST"' in script[
        runtime_ready:repair
    ]
    assert "return 3" in (
        ROOT / "gateway" / "tee" / "bootstrap_active_ancestry_checkpoints_v2.py"
    ).read_text(encoding="utf-8")
    assert '3)\n      # Expected exactly once' in script
    assert "candidate runtime did not durably bound active receipt ancestry" in script


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


def test_gateway_active_release_selection_requires_paired_validator_authority(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    function = _shell_function_source(
        script, "prepare_gateway_active_release_lineage"
    )
    harness = tmp_path / "require-paired-validator.sh"
    harness.write_text(
        "\n".join(
            (
                "#!/bin/bash",
                "set -u",
                    function,
                    'GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED="1"',
                    'GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS=""',
                "if prepare_gateway_active_release_lineage; then",
                "  exit 9",
                "fi",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0
    assert (
        "paired validator active release requirements are unavailable"
        in completed.stderr
    )


def test_gateway_final_release_selection_restores_prepared_local_identity(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    function = _shell_function_source(
        script, "prepare_gateway_active_release_lineage"
    )
    candidate_commit = "2" * 40
    installed_commit = "1" * 40
    candidate_gateway = tmp_path / "candidate-gateway.json"
    candidate_validator = tmp_path / "candidate-validator.json"
    installed_gateway = tmp_path / "installed-gateway.json"
    installed_validator = tmp_path / "installed-validator.json"
    running_gateway = tmp_path / "running-gateway.json"
    installed_lineage = tmp_path / "installed-lineage.json"
    validator_requirements = Path(
        f"/tmp/leadpoet-{os.getpid()}-{time.time_ns()}.json"
    )
    env_clone = tmp_path / "gateway-env-clone.sh"
    controller_log = tmp_path / "controller.log"
    for path in (
        candidate_gateway,
        candidate_validator,
        installed_gateway,
        installed_validator,
        installed_lineage,
        running_gateway,
        validator_requirements,
    ):
        path.write_text("{}\n", encoding="utf-8")
    env_clone.write_text(
        f"export LEADPOET_LOCAL_RELEASE_COMMIT_SHA={installed_commit}\n"
        f"export LEADPOET_LOCAL_GATEWAY_RELEASE={installed_gateway}\n"
        f"export LEADPOET_LOCAL_VALIDATOR_RELEASE={installed_validator}\n",
        encoding="utf-8",
    )
    python_stub = tmp_path / "python"
    python_stub.write_text(
        "#!/bin/bash\nprintf 'sha256:%064d\\n' 0\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    harness = tmp_path / "restore-local-release.sh"
    harness.write_text(
        "\n".join(
            (
                "#!/bin/bash",
                "set -e",
                function,
                "run_gateway_active_release_controller_module() {",
                "  printf '%s\\n%s\\n%s\\n' \"$LEADPOET_LOCAL_RELEASE_COMMIT_SHA\" \"$LEADPOET_LOCAL_GATEWAY_RELEASE\" \"$LEADPOET_LOCAL_VALIDATOR_RELEASE\" >> \"$CONTROLLER_LOG\"",
                "  while [ \"$#\" -gt 0 ]; do",
                "    case \"$1\" in",
                "      --requirements-output) shift; printf '{}\\n' > \"$1\" ;;",
                "      --lineage-output) shift; printf '{}\\n' > \"$1\" ;;",
                "    esac",
                "    shift",
                "  done",
                "}",
                "prepare_gateway_active_release_lineage || exit 8",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    prepared_requirements = tmp_path / "prepared-requirements.json"
    prepared_lineage = tmp_path / "prepared-lineage.json"
    try:
        completed = subprocess.run(
            ["bash", str(harness)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={
                **os.environ,
                "CONTROLLER_LOG": str(controller_log),
                "ENV_CLONE": str(env_clone),
                "GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT": "standalone",
                "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID": "restart-1",
                "GATEWAY_ANCESTRY_CHECKPOINT_RELEASE_SNAPSHOT": str(running_gateway),
                "GATEWAY_ANCESTRY_SAFE_EPOCH": "1",
                "GATEWAY_COUNTERPART_RELEASE_LINEAGE": "",
                "GATEWAY_HISTORICAL_TOPOLOGY_HASH": "",
                "GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED": "1",
                "GATEWAY_PREFLIGHT_TREE": str(tmp_path),
                "GATEWAY_PREPARED_V2_RELEASE_LINEAGE": str(prepared_lineage),
                "GATEWAY_PREPARED_V2_RELEASE_MANIFEST": str(candidate_gateway),
                "GATEWAY_PREPARED_V2_RELEASE_REQUIREMENTS": str(prepared_requirements),
                "GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST": str(candidate_validator),
                "GATEWAY_PYTHON_BIN": str(python_stub),
                "GATEWAY_RESTART_AUTHORITY_COMMIT": candidate_commit,
                "GATEWAY_RESTART_AUTHORITY_ROOT": "",
                "GATEWAY_STATEFUL_CUTOVER_CEREMONY": "0",
                "GATEWAY_V2_RELEASE_BUCKET": "bucket",
                "GATEWAY_V2_RELEASE_LINEAGE": str(installed_lineage),
                "GATEWAY_V2_RELEASE_MANIFEST": str(running_gateway),
                "GATEWAY_V2_RELEASE_PREFIX": "prefix",
                "GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS": str(validator_requirements),
                "LEADPOET_LOCAL_GATEWAY_RELEASE": str(candidate_gateway),
                "LEADPOET_LOCAL_RELEASE_COMMIT_SHA": candidate_commit,
                "LEADPOET_LOCAL_VALIDATOR_RELEASE": str(candidate_validator),
                "LEADPOET_REPO_ROOT": str(tmp_path),
                "PREPARED_GATEWAY_SHA": candidate_commit,
            },
        )
    finally:
        validator_requirements.unlink(missing_ok=True)

    assert completed.returncode == 0, completed.stderr
    assert controller_log.read_text(encoding="utf-8").splitlines() == [
        candidate_commit,
        str(candidate_gateway),
        str(candidate_validator),
    ]


def test_gateway_standalone_active_release_uses_only_explicit_compact_fallback() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    selector = _shell_function_source(
        script, "prepare_gateway_active_release_lineage"
    )

    paired_rejection = selector.index(
        'elif [ "$GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED" = "1" ]; then'
    )
    fallback = selector.index(
        '--fallback-lineage "$GATEWAY_V2_RELEASE_LINEAGE"',
        paired_rejection,
    )
    controller_call = selector.index(
        "gateway.tee.prepare_active_release_lineage_v2",
        fallback,
    )
    assert paired_rejection < fallback < controller_call
    assert '--fallback-context "$fallback_context"' in selector
    assert (
        '--validator-requirements "$GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS"'
        in selector
    )
    assert "standalone gateway compact-lineage fallback is unavailable" in selector
    assert "list_objects" not in selector
    assert "fetch_release_lineage_v2" not in selector


def test_gateway_component_lineage_comparison_uses_bounded_nofollow_reads() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    selector = _shell_function_source(
        script, "prepare_gateway_active_release_lineage"
    )

    assert "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW" in selector
    assert "stat.S_ISREG(metadata.st_mode)" in selector
    assert "max_document_bytes = 4 * 1024 * 1024" in selector
    assert "os.read(descriptor, max_document_bytes + 1)" in selector
    assert "Path(sys.argv[2]).read_text" not in selector
    assert "Path(sys.argv[3]).read_text" not in selector


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


def test_gateway_ancestry_precheckpoint_waits_for_proven_frontier_and_keeps_running_release(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    helper_source = "\n\n".join(
        _shell_function_source(script, name)
        for name in (
            "wait_for_gateway_owned_process_group",
            "prepare_gateway_ancestry_checkpoint_bootstrap",
            "start_gateway_ancestry_checkpoint_bootstrap",
            "wait_for_gateway_ancestry_checkpoint_bootstrap",
        )
    )
    preflight_tree = tmp_path / "candidate"
    helper = (
        preflight_tree
        / "gateway"
        / "tee"
        / "bootstrap_active_ancestry_checkpoints_v2.py"
    )
    helper.parent.mkdir(parents=True)
    helper.write_text("# candidate helper\n", encoding="utf-8")
    environment = tmp_path / "gateway.env"
    environment.write_text("export TEST_GATEWAY_ENV=ready\n", encoding="utf-8")
    commit = "a" * 40
    candidate_commit = "b" * 40
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"commit_sha": commit}), encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        f"""#!/bin/bash
case "$*" in
  *build-info*) printf '%s\\n' '{{"git_commit":"{commit}"}}' ;;
  *health/v2-authority*) printf '%s\\n' '{{"status":"ready"}}' ;;
  *) exit 22 ;;
esac
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    fake_timeout = fake_bin / "timeout"
    fake_timeout.write_text(
        "#!/bin/bash\nshift\nexec \"$@\"\n",
        encoding="utf-8",
    )
    fake_timeout.chmod(0o755)
    fake_python = tmp_path / "gateway-python"
    fake_python.write_text(
        f"""#!/bin/bash
if [ "${{1:-}}" = "-" ]; then
  exec {shlex.quote(sys.executable)} "$@"
fi
test "$1" = "-m"
test "$2" = "gateway.tee.bootstrap_active_ancestry_checkpoints_v2"
test "$3" = "--release-manifest"
grep -Fq '"commit_sha": "{commit}"' "$4"
test "$5" = "--epoch"
test "$6" = "24324"
test "$TEST_GATEWAY_ENV" = "ready"
printf '%s\n' "$6" > "$FAKE_CHECKPOINT_EPOCH"
sleep "${{FAKE_CHECKPOINT_SECONDS:-0}}"
exit "${{FAKE_CHECKPOINT_STATUS:-0}}"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    timing_log = tmp_path / "timing.log"
    harness = f"""set -euo pipefail
record_gateway_restart_timing() {{
  printf '%s:%s\\n' "$1" "${{2:-reached}}" >> "$TIMING_LOG"
}}
{helper_source}
GATEWAY_PREFLIGHT_TREE="$1"
GATEWAY_PYTHON_BIN="$2"
GATEWAY_V2_RELEASE_MANIFEST="$3"
GATEWAY_ANCESTRY_CHECKPOINT_RELEASE_SNAPSHOT="$4"
GATEWAY_ANCESTRY_CHECKPOINT_LOG="$5"
ENV_CLONE="$6"
TIMING_LOG="$7"
GATEWAY_ANCESTRY_CHECKPOINT_PID=""
GATEWAY_ANCESTRY_CHECKPOINT_STATE="not_started"
GATEWAY_ANCESTRY_SAFE_EPOCH=""
GATEWAY_WEIGHT_STORAGE_PREFLIGHT_CAPABILITY="supported"
printf '%s\n' '{{"commit_sha": "{commit}"}}' > "$GATEWAY_V2_RELEASE_MANIFEST"
prepare_gateway_ancestry_checkpoint_bootstrap
test "$GATEWAY_ANCESTRY_CHECKPOINT_STATE" = "prepared"
test -z "$GATEWAY_ANCESTRY_CHECKPOINT_PID"
GATEWAY_ANCESTRY_SAFE_EPOCH="24324"
printf '%s\n' '{{"commit_sha": "{candidate_commit}"}}' > "$GATEWAY_V2_RELEASE_MANIFEST"
start_gateway_ancestry_checkpoint_bootstrap
sleep "$FAKE_RELEASE_SECONDS"
if [ "${{REQUIRE_CHECKPOINT_RUNNING:-0}}" = "1" ]; then
  kill -0 "$GATEWAY_ANCESTRY_CHECKPOINT_PID"
fi
wait_for_gateway_ancestry_checkpoint_bootstrap
printf '%s\\n' "$GATEWAY_ANCESTRY_CHECKPOINT_STATE"
"""
    command = [
        "bash",
        "-c",
        harness,
        "gateway-ancestry-overlap-test",
        str(preflight_tree),
        str(fake_python),
        str(release),
        str(tmp_path / "release-snapshot.json"),
        str(tmp_path / "checkpoint.log"),
        str(environment),
        str(timing_log),
    ]
    base_env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "FAKE_RELEASE_SECONDS": "0.05",
        "FAKE_CHECKPOINT_EPOCH": str(tmp_path / "checkpoint-epoch"),
    }

    passed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
        env={
            **base_env,
            "FAKE_CHECKPOINT_SECONDS": "0.3",
            "REQUIRE_CHECKPOINT_RUNNING": "1",
        },
    )
    assert passed.returncode == 0, passed.stderr
    assert passed.stdout.rstrip().endswith("passed")
    assert (tmp_path / "checkpoint-epoch").read_text(encoding="utf-8").strip() == (
        "24324"
    )
    assert timing_log.read_text(encoding="utf-8").splitlines() == [
        "ancestry_precheckpoint_started:reached",
        "ancestry_precheckpoint_complete:passed",
    ]

    timing_log.unlink()
    unsupported = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
        env={
            **base_env,
            "FAKE_CHECKPOINT_SECONDS": "0.01",
            "FAKE_CHECKPOINT_STATUS": "3",
            "FAKE_RELEASE_SECONDS": "0.01",
        },
    )
    assert unsupported.returncode == 0, unsupported.stderr
    assert unsupported.stdout.rstrip().endswith("unsupported")
    assert timing_log.read_text(encoding="utf-8").splitlines() == [
        "ancestry_precheckpoint_started:reached",
        "ancestry_precheckpoint_complete:unsupported",
    ]

    timing_log.unlink()
    failed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
        env={
            **base_env,
            "FAKE_CHECKPOINT_SECONDS": "0.01",
            "FAKE_CHECKPOINT_STATUS": "23",
            "FAKE_RELEASE_SECONDS": "0.01",
        },
    )
    assert failed.returncode == 23
    assert "failed before shutdown" in failed.stderr
    assert timing_log.read_text(encoding="utf-8").splitlines() == [
        "ancestry_precheckpoint_started:reached",
        "ancestry_precheckpoint_complete:failed",
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
    assert exit_handler.index("cancel_gateway_ancestry_checkpoint_bootstrap") < (
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
    assert 'record_gateway_restart_timing "invoked"' in script
    assert 'record_gateway_restart_timing "offline_artifact_prepare_started"' in script
    assert 'record_gateway_restart_timing "ancestry_precheckpoint_started"' in script
    assert 'record_gateway_restart_timing "local_release_ready"' in script
    assert (
        'record_gateway_restart_timing "offline_artifact_prepare_complete" "passed"'
        in script
    )
    assert (
        'record_gateway_restart_timing "offline_artifact_prepare_complete" "failed"'
        in script
    )
    assert (
        'record_gateway_restart_timing "pre_shutdown_checks_complete"'
        in script
    )
    assert (
        'record_gateway_restart_timing "${timing_stage}_started"'
        in script
    )
    assert (
        'record_gateway_restart_timing "${timing_stage}_complete" "passed"'
        in script
    )
    assert 'record_gateway_restart_timing "chain_settlement_repair_started"' in script
    assert (
        'record_gateway_restart_timing "chain_settlement_repair_complete" "passed"'
        in script
    )
    assert 'record_gateway_restart_timing "candidate_activated"' in script
    assert 'record_gateway_restart_timing "attested_runtime_staged"' in script
    assert 'record_gateway_restart_timing "gateway_role_eifs_built"' in script
    assert 'record_gateway_restart_timing "gateway_enclaves_started"' in script
    assert 'record_gateway_restart_timing "v2_runtime_bootstrapped"' in script
    assert 'record_gateway_restart_timing "v2_kms_provisioned"' in script
    assert 'record_gateway_restart_timing "v2_runtime_ready"' in script
    assert (
        'record_gateway_restart_timing "validator_weight_input_ready"'
        in script
    )
    assert 'record_gateway_restart_timing "completed" "passed"' in script
    assert (
        "WARNING: gateway restart timing event could not be recorded"
        in script
    )
    assert (
        'GATEWAY_RESTART_TIMING_INITIALIZED="'
        '$GATEWAY_RESTART_TIMING_INITIALIZED" \\'
    ) in script


def test_gateway_runtime_env_cannot_replace_current_restart_controller_state(
    tmp_path: Path,
) -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")

    for key in (
        "GATEWAY_RESTART_AUTHORITY_ROOT",
        "GATEWAY_RESTART_AUTHORITY_COMMIT",
    ):
        assert script.count(f"-u {key} \\") == 7

    for key in (
        "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
        "GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED",
        "GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT",
        "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_FILE",
        "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_NONCE",
        "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_TIMEOUT_SECONDS",
        "GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS",
        "GATEWAY_COUNTERPART_RELEASE_LINEAGE",
    ):
        assert script.count(f"-u {key} \\") == 5

    for key in (
        "GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT",
        "GATEWAY_RESTART_STARTED_EPOCH",
        "GATEWAY_RESTART_TIMING_DIR",
        "GATEWAY_RESTART_TIMING_FILE",
        "GATEWAY_RESTART_TIMING_INITIALIZED",
    ):
        assert script.count(f'    "{key}",') == 2

    clone = script.index('echo "Cloning live gateway env before stopping processes"')
    merge = script.index('cat "$ENV_SECRET" >> "$ENV_CLONE"')
    first_reload = script.index('. "$ENV_CLONE"', merge)
    assert clone < merge < first_reload

    invocation_keys = (
        "GATEWAY_RESTART_INVOCATION_ID",
        "LEADPOET_RESTART_INVOCATION_ID",
    )
    for key in invocation_keys:
        # Secrets-cache, prepared-secret, and live-process parsers must all
        # remove stale values before the active controller reasserts them.
        assert script.count(f'    "{key}",') == 3

    # A prior gateway process must not redirect either half of the candidate
    # local release pair to an older restart's retained manifest.
    for key in (
        "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
        "GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST",
    ):
        assert script.count(f'    "{key}",') == 3

    cleanup = _shell_function_source(script, "on_gateway_restart_exit")
    assert '"$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \\' in cleanup
    assert '"$GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST" \\' in cleanup

    merge_end = script.index(
        'if [ -f "$GATEWAY_STATEFUL_CUTOVER_MANIFEST" ]; then', merge
    )
    merge_and_reassert = script[merge:merge_end]
    env_clone = tmp_path / "gateway.env.clone"
    env_secret = tmp_path / "gateway.env.secret"
    for path in (env_clone, env_secret):
        path.write_text(
            "export GATEWAY_RESTART_INVOCATION_ID=stale\n"
            "export LEADPOET_RESTART_INVOCATION_ID=stale\n"
            "export GATEWAY_PRIVATE_KEY_PATH=/stale/private-key.pem\n"
            "export ARWEAVE_KEYFILE_PATH=/stale/arweave-keyfile.json\n"
            "export GATEWAY_RESTART_GIT_SSH_COMMAND=stale-restart-command\n"
            "export GIT_SSH_COMMAND=stale-git-command\n",
            encoding="utf-8",
        )
    active_invocation = "gateway-active-invocation"
    active_private_key = "/run-scoped/gateway-private-key.pem"
    active_arweave_keyfile = "/run-scoped/arweave-keyfile.json"
    active_git_ssh_command = "ssh -i /run-scoped/deploy-key"
    preserved = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -euo pipefail\n"
                + merge_and_reassert
                + '\nset -a\n. "$ENV_CLONE"\nset +a\n'
                + "printf '%s\\n%s\\n%s\\n%s\\n%s\\n' \"$GATEWAY_RESTART_INVOCATION_ID\" "
                + '"$LEADPOET_RESTART_INVOCATION_ID" '
                + '"$GATEWAY_PRIVATE_KEY_PATH" "$ARWEAVE_KEYFILE_PATH" '
                + '"$GIT_SSH_COMMAND"\n'
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ENV_CLONE": str(env_clone),
            "ENV_SECRET": str(env_secret),
            "GATEWAY_RESTART_INVOCATION_ID": active_invocation,
            "LEADPOET_RESTART_INVOCATION_ID": "stale-parent",
            "GATEWAY_PRIVATE_KEY_PATH": active_private_key,
            "ARWEAVE_KEYFILE_PATH": active_arweave_keyfile,
            "GATEWAY_RESTART_GIT_SSH_COMMAND": active_git_ssh_command,
        },
    )
    assert preserved.stdout.splitlines() == [
        active_invocation,
        active_invocation,
        active_private_key,
        active_arweave_keyfile,
        active_git_ssh_command,
    ]


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
        b"GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST=/stale/f5-validator.json\0"
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
    assert "GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST" not in cloned


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


def test_gateway_restart_checks_source_add_without_retired_admin_command() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    v2_health = "if ! wait_for_gateway_v2_authority; then"
    handoff = "-m gateway.tee.verify_weight_submission_ready_v2"
    source_add_status = (
        "curl -fsS http://localhost:8000/research-lab/status"
    )
    source_add_runtime = (
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
    assert script.rindex(v2_health) < script.rindex(handoff)
    assert script.rindex(handoff) < script.rindex(source_add_status)
    assert (
        script.rindex(source_add_status)
        < script.rindex(source_add_runtime)
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
    assert (
        "unset RESEARCH_LAB_EVIDENCE_PROXY_URL "
        "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH"
    ) in script


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
        '    7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- &'
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
        '    7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- &'
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
