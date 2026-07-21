from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from gateway.tee import build_identity


ROOT = Path(__file__).resolve().parents[1]


def _ordered_offsets(text: str, markers: tuple[str, ...]) -> list[int]:
    offsets = [text.index(marker) for marker in markers]
    assert offsets == sorted(offsets)
    return offsets


def test_gateway_restart_activates_git_between_shutdown_and_existing_workflow() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    _ordered_offsets(
        script,
        (
            'echo "Preparing exact gateway commit from configured GitHub branch"',
            'echo "Capturing the official subnet restart window before release acquisition"',
            'echo "Validating the prepared V2 release before production shutdown"',
            'echo "Preparing exact hash-locked V2 build artifacts before production shutdown"',
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
            'curl -fsS http://localhost:8000/health/v2-authority',
            'GATEWAY_DEPLOY_STAGE="host_restart_script_install"',
            'finalize_deployment_record succeeded',
        ),
    )


def test_gateway_restart_fails_closed_on_all_authoritative_readiness_routes() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert "http://localhost:8000/health/v2-authority >/dev/null" in script
    assert "http://localhost:8000/research-lab/status >/dev/null" in script
    assert "http://localhost:8000/attest >/dev/null" in script
    assert "http://localhost:8000/research-lab/status || true" not in script
    assert "http://localhost:8000/attest || true" not in script


def test_gateway_restart_repairs_and_proves_automatic_weight_input() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    runtime_ready = script.index(
        '"$GATEWAY_PYTHON_BIN" -m gateway.tee.verify_v2_runtime_ready'
    )
    cutover = script.index(
        'echo "Executing the one-time receipt-backed stateful epoch cutover"'
    )
    repair = script.index(
        "-m gateway.tee.verify_weight_submission_ready_v2 --repair"
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
    install = script.index(
        'GATEWAY_DEPLOY_STAGE="host_restart_script_install"'
    )

    assert (
        runtime_ready
        < cutover
        < repair
        < launch
        < base_health
        < http_handoff
        < install
    )
    assert 'GATEWAY_DEPLOY_STAGE="validator_weight_input_repair"' in script
    assert (
        'GATEWAY_DEPLOY_STAGE="validator_weight_input_http_check"' in script
    )


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
    assert 'findmnt -rn -R /var/lib/docker -o TARGET' in script
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
        "-m gateway.tee.verify_weight_submission_ready_v2 --repair",
        repair_stage,
    )
    repair_block = script[repair_stage:repair_command]

    assert '(\n  cd "$LEADPOET_REPO_ROOT"\n' in repair_block


def test_gateway_restart_v2_preflight_runs_target_commit_before_shutdown() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    materialize = script.index(
        'echo "Materializing the prepared commit for pre-shutdown V2 tooling"'
    )
    release_channel = script.index(
        'run_prepared_gateway_module gateway.tee.release_channel_v2'
    )
    restart_window = script.index(
        'echo "Capturing the official subnet restart window before release acquisition"'
    )
    credential_envelopes = script.index(
        'run_prepared_gateway_module gateway.tee.prepare_gateway_envelopes_v2'
    )
    preflight = script.index(
        'echo "Validating the prepared V2 release before production shutdown"'
    )
    shutdown = script.index(
        'echo "Stopping existing gateway and Research Lab worker processes"'
    )
    artifact_prepare = script.index(
        'echo "Preparing exact hash-locked V2 build artifacts before production shutdown"'
    )
    dependency_preflight = script.index(
        'echo "Installing gateway host Python dependencies before production shutdown"'
    )
    assert (
        materialize
        < restart_window
        < release_channel
        < credential_envelopes
        < preflight
    )
    assert preflight < shutdown
    assert preflight < artifact_prepare < shutdown
    assert dependency_preflight < preflight < shutdown
    assert (
        script.index(
            'git -C "$LEADPOET_REPO_ROOT" archive "$PREPARED_GATEWAY_SHA"'
        )
        < release_channel
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
    assert script.index('--release-manifest "$GATEWAY_V2_RELEASE_MANIFEST"') < shutdown
    assert script.index('--parent-env-file "$ENV_CLONE"') < shutdown
    assert script.index(
        '--acceptance-corpus-manifest "$GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST"'
    ) < shutdown
    assert script.index(
        '--acceptance-corpus-root "$GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT"'
    ) < shutdown
    assert script.index('--topology-mode "${GATEWAY_TEE_TOPOLOGY_MODE:-full}"') < shutdown
    assert script.index("prepare_offline_artifacts_v2.sh") < shutdown
    assert script.index('pkill -9 -f "python3 -u -m gateway.main"') > shutdown


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
    assert dependency_preflight < shutdown < post_activate_install
    assert (
        'echo "Gateway remains running; production shutdown has not started." >&2'
        in script[dependency_preflight:shutdown]
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
        '2>&1 < /dev/null 9>&- &'
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
    assert (
        '-m gateway.utils.tee_inter_enclave_relay \\\n'
        '    >> "$GATEWAY_LOG_ROOT/inter_enclave_relay.log" 2>&1 < /dev/null 9>&- &'
    ) in script
    assert 'VALIDATOR_GATEWAY_PCR0_CACHE_FILE' not in script
    assert 'independent_gateway_identity' not in script
    gate = "Leadpoet.utils.restart_epoch_gate"
    release = "gateway.tee.release_channel_v2"
    shutdown = 'echo "Stopping existing gateway and Research Lab worker processes"'
    assert gate in script
    assert script.index(gate) < script.index(release) < script.index(shutdown)
    assert "waiting inside the valid restart invocation" in script
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
    assert 'restart_only_keys = {"GATEWAY_DEPLOY_COMMIT"}' in script
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
    assert 'rm -f "$GATEWAY_TEE_EIF_ROOT"/enclave-build-*.json' in script
    assert (
        'enclave-build-gateway.json' in script
        or 'build_role_enclaves.sh' in script
    )


def test_worker_process_prefers_checkout_but_keeps_attested_fallback() -> None:
    source = (ROOT / "gateway" / "research_lab" / "worker_process.py").read_text(
        encoding="utf-8"
    )
    assert "for path in (ATTESTED_RUNTIME, PACKAGE_PARENT):" in source
    assert "while str(path) in sys.path:" in source
    assert source.index("for path in (ATTESTED_RUNTIME, PACKAGE_PARENT):") < source.index(
        "from gateway.research_lab.config"
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
