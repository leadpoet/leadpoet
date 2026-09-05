from __future__ import annotations

import base64
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

from tests.readiness_test_venv import build_dependency_complete_readiness_venv

from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
)
from gateway.tee.active_release_requirements_v2 import (
    build_active_release_requirements_v2,
)
from gateway.tee.topology import ROLE_SPECS
from tests.test_release_channel_v2 import _gateway_manifest, _validator_manifest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restart_attested_release_local.sh"


def _fake_readiness_observations(tmp_path: Path, commit: str) -> tuple[Path, Path]:
    gateway_release = _gateway_manifest(commit)
    validator_release = _validator_manifest(commit)
    channel = build_release_channel_v2(
        gateway_release_manifest=gateway_release,
        validator_release_manifest=validator_release,
    )
    lineage = build_release_lineage_v2([channel], current_commit=commit)
    expectations = lineage["releases"][commit]["roles"]

    def boot(role: str, character: str) -> dict:
        expectation = expectations[role]
        return {
            "role": role,
            "physical_role": role,
            "commit_sha": commit,
            "pcr0": expectation["pcr0"],
            "build_manifest_hash": expectation["build_manifest_hash"],
            "dependency_lock_hash": expectation["dependency_lock_hash"],
            "config_hash": "sha256:" + character * 64,
            "boot_identity_hash": "sha256:" + character * 64,
        }

    gateway_boots = {
        role: boot(role, character)
        for role, character in zip(sorted(ROLE_SPECS), "567")
    }
    validator_boot = boot("validator_weights", "8")
    gateway_observation = {
        "schema_version": "leadpoet.gateway_deploy_readiness_observation.v2",
        "source_commit": commit,
        "build_commit": commit,
        "gateway_release_manifest": gateway_release,
        "validator_release_manifest": validator_release,
        "compact_lineage": lineage,
        "boot_identities": gateway_boots,
        "expected_role_config_hashes": {
            role: value["config_hash"] for role, value in gateway_boots.items()
        },
        "runtime_readiness": {
            "schema_version": "leadpoet.gateway_v2_runtime_readiness.v2",
            "status": "ready",
            "provider_registry_hash": "sha256:" + "9" * 64,
            "roles": [
                {
                    "physical_role": role,
                    "role": ROLE_SPECS[role]["service_role"],
                    "worker_count": 1,
                    "configured_worker_count": 1,
                    "boot_identity_hash": gateway_boots[role][
                        "boot_identity_hash"
                    ],
                }
                for role in sorted(ROLE_SPECS)
            ],
        },
        "coordinator_attestation_pcr0": gateway_boots[
            "gateway_coordinator"
        ]["pcr0"],
    }
    validator_observation = {
        "schema_version": "leadpoet.validator_deploy_readiness_observation.v2",
        "host_commit": commit,
        "gateway_release_manifest": gateway_release,
        "validator_release_manifest": validator_release,
        "compact_lineage": lineage,
        "boot_identity": validator_boot,
        "expected_config_hash": validator_boot["config_hash"],
    }
    gateway_path = tmp_path / "gateway-observation.json"
    validator_path = tmp_path / "validator-observation.json"
    gateway_path.write_text(json.dumps(gateway_observation), encoding="utf-8")
    validator_path.write_text(json.dumps(validator_observation), encoding="utf-8")
    return gateway_path, validator_path


def test_attested_release_restart_operator_is_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    expected_key = "$HOME/Downloads/leadpoet-2026-07-28.pem"
    assert (
        f'GATEWAY_KEY="${{LEADPOET_GATEWAY_SSH_KEY:-{expected_key}}}"'
        in source
    )
    assert (
        f'VALIDATOR_KEY="${{LEADPOET_VALIDATOR_SSH_KEY:-{expected_key}}}"'
        in source
    )
    assert (
        'PRODUCTION_VALIDATOR_PYTHON_BIN="/home/ec2-user/venv311/bin/python3"'
        in source
    )
    assert (
        'VALIDATOR_PYTHON_BIN="${LEADPOET_VALIDATOR_PYTHON_BIN:-'
        '$PRODUCTION_VALIDATOR_PYTHON_BIN}"'
    ) in source
    assert 'component="all"' in source
    assert "exact_commit_restart_v2.py" in source
    assert "--compatibility-floor" not in source
    assert "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS" in source
    assert "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS" in source
    assert "VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE" in source
    assert "VALIDATOR_COORDINATION_ATTEMPTS=3000" in source
    assert "VALIDATOR_COORDINATION_TIMEOUT_SECONDS=9300" in source
    assert "Prepared validator active release requirements sidecar" in source
    assert "gw_restart.sh" in source
    assert "validator_restart.sh" in source
    assert "gateway_exact_release_ready" in source
    assert "validator_exact_release_ready" in source
    assert "readiness-transition" in source
    assert "readiness-final" in source
    assert "leadpoet.deploy_readiness.v2" in source
    assert "--local-python </absolute/venv/bin/python>" in source
    assert '"$local_python_target" -I -S -B -c' in source
    assert 'sys.path.insert(0, str(root))' in source
    assert 'sys.path.append(str(site_packages))' in source
    assert "leadpoet.local_readiness_python.v1" in source
    assert "pure readiness imports loaded the validator wallet dependency" in source
    assert source.count("run_local_readiness_python ") == 9
    assert 'PYTHONPATH="$ROOT" python3' not in source
    assert "/health/v2-authority" in source
    assert "attestation = get('/attest')" in source
    assert "/weights/v2/release-evidence/" in source
    assert "fetch_locked_release_identity_cache" in source
    assert "identity_cache_from_release_channel" in source
    assert "active gateway release differs from auditor release evidence" in source
    assert "verify_v2_runtime_ready(clients)" in source
    assert "processes[0].joinpath('environ')" in source
    assert "build_research_lab_execution_config(" in source
    assert "environment=runtime_environment" in source
    assert "provider_reference_hashes_from_envelopes(envelopes)" in source
    assert "configured_scoring_worker_count(config_dir)" in source
    assert "verify_required_worker_proxy_profiles_v2" in source
    assert "expected_historical_topology_hash" in source
    assert "runtime_configuration_documents(" in source
    assert "build_runtime_configuration(" in source
    assert "client.health_check()" in source
    assert "expected_config_hash': sha256_json(configuration)" in source
    assert "VALIDATOR_V2_DEPLOY_COMMIT" in source
    assert "VALIDATOR_WEIGHT_PROTOCOL" in source
    assert "use --component all" in source
    assert "trap cleanup EXIT" in source
    assert "trap 'exit 130' INT" in source
    assert "trap 'exit 143' TERM" in source
    assert "printf -v bootstrap_command_quoted '%q'" not in source
    assert "bootstrap_command_b64=" in source
    assert (
        "exec bash -c \\\"\\$(printf '%s' '$bootstrap_command_b64' | "
        "base64 --decode)\\\""
        in source
    )
    assert "GATEWAY_RESTART_AUTHORITY_COMMIT='$branch_commit'" in source
    assert (
        "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT="
        "'$selected_active_release_authority_commit'"
    ) in source
    assert (
        'local selected_active_release_authority_commit="'
        '${active_release_authority_commit:-$branch_commit}"'
    ) in source
    assert '"$commit" "$active_release_authority_commit"' in source
    assert '"$active_release_authority_commit" "$branch_commit"' in source
    assert (
        'VALIDATOR_STATEFUL_CUTOVER_MANIFEST="/home/ec2-user/.config/'
        'leadpoet/stateful-epoch-cutover.json"'
    ) in source
    assert "test -f '$VALIDATOR_STATEFUL_CUTOVER_MANIFEST'" in source
    assert "test ! -L '$VALIDATOR_STATEFUL_CUTOVER_MANIFEST'" in source
    assert "test -s '$VALIDATOR_STATEFUL_CUTOVER_MANIFEST'" in source
    assert "test -x '$VALIDATOR_PYTHON_BIN'" in source
    assert (
        "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH="
        "'$VALIDATOR_STATEFUL_CUTOVER_MANIFEST' "
        "PYTHONPATH='$VALIDATOR_REPO_ROOT' '$VALIDATOR_PYTHON_BIN'"
    ) in source
    assert (
        "'$VALIDATOR_PYTHON_BIN' -m "
        "gateway.tee.prepare_active_release_lineage_v2"
    ) in source
    assert "python3 -m gateway.tee.prepare_active_release_lineage_v2" not in source
    assert "VALIDATOR_PYTHON_BIN='$VALIDATOR_PYTHON_BIN'" in source
    assert 'local gateway_restart_entrypoint_root="\\$authority_root"' in source
    assert 'gateway_restart_entrypoint_root="\\$candidate_root"' in source
    assert r'bash \"$gateway_restart_entrypoint_root/gw_restart.sh\"' in source
    assert r'bash \"\$authority_root/validator_restart.sh\"' in source
    assert r"""bash \"\$authority_root/validator_restart.sh\" --commit '$commit'""" in source
    assert "git -C '$VALIDATOR_REPO_ROOT' archive '$branch_commit'" in source
    assert "gateway-restart-controller-bootstrap" in source
    assert "--validator-hotkey-config '$VALIDATOR_V2_HOTKEY_CONFIG_PATH'" in source
    assert "--chain-signing-profile '$VALIDATOR_CHAIN_SIGNING_PROFILE_PATH'" in source
    assert source.index("    fetch_gateway_final_release_authority\n") < source.index(
        '    publish_coordination_value "$commit"\n'
    )
    assert source.index('kill -TERM "$validator_job"') < source.index(
        'coordination_remote_command "failed:$commit"'
    )
    assert source.index('kill -TERM "$validator_job"') < source.index(
        'for _ in $(seq 1 "$VALIDATOR_FAILURE_CLEANUP_ATTEMPTS")'
    )
    assert "VALIDATOR_FAILURE_MARKER_ATTEMPTS" in source
    component_case = source.index('case "$component" in')
    paired_start = source.index("  all)\n", component_case)
    paired = source[paired_start : source.index("\nesac\n", paired_start)]
    assert paired.count(
        "paired_restart_deadline=$((SECONDS + VALIDATOR_COORDINATION_TIMEOUT_SECONDS))"
    ) == 1
    assert "validator_completion_deadline" not in paired
    assert '[ "$SECONDS" -ge "$paired_restart_deadline" ]' in paired


def test_miner_maintenance_bootstrap_command_is_shell_parseable(
    tmp_path: Path,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    function_start = source.index("build_gateway_restart_command() {")
    function_end = source.index(
        "\n}\n\nrun_gateway_restart()", function_start
    ) + 2
    function_source = source[function_start:function_end]
    commit = "1" * 40
    shell_program = f"""
set -Eeuo pipefail
commit={shlex.quote(commit)}
branch_commit={shlex.quote(commit)}
component=all
disable_miner_submissions_before_restart=1
GATEWAY_ENV_SECRET_ID=''
controller_verifier_b64='YQ=='
expected_controller_commit='{'3' * 40}'
GATEWAY_PYTHON_BIN='/usr/bin/python3.11'
GATEWAY_REPO_ROOT='/home/ec2-user/leadpoet_repo'
PRODUCTION_GATEWAY_RESTART_CONTROLLER_CURRENT='/controller/current'
PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT='/controller'
GATEWAY_RESTART='/home/ec2-user/gw_restart.sh'
RELEASE_PREFIX='weights/v2/release-evidence'
gateway_handoff_file='/tmp/handoff'
gateway_handoff_nonce='{'2' * 64}'
gateway_validator_requirements_remote='/tmp/validator-requirements.json'
gateway_counterpart_lineage_remote='/tmp/counterpart-lineage.json'
active_release_restart_invocation_id='restart-fixture'
paired_gateway_handoff_file='/tmp/leadpoet-gateway-paired-restart.fixture.ready'
paired_gateway_handoff_nonce='{'4' * 64}'
VALIDATOR_COORDINATION_TIMEOUT_SECONDS=9300
GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH='/tmp/gateway-requirements.json'
GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH='/tmp/gateway-lineage.json'
GATEWAY_KEY='/tmp/key'
GATEWAY_HOST='gateway.invalid'
ssh_common=('-o' 'BatchMode=yes')
{function_source}
build_gateway_restart_command
printf '%s\\0' "${{gateway_restart_command[@]}}"
"""
    rendered = subprocess.run(
        ["bash", "-c", shell_program],
        check=False,
        capture_output=True,
        timeout=5,
    )

    assert rendered.returncode == 0, rendered.stderr.decode("utf-8", "replace")
    remote_command = rendered.stdout.rstrip(b"\0").split(b"\0")[-1]
    encoded_match = re.search(
        rb"printf '%s' '([A-Za-z0-9+/=]+)' \| base64 --decode",
        remote_command,
    )
    assert encoded_match is not None
    bootstrap_command = base64.b64decode(
        encoded_match.group(1), validate=True
    ).decode("utf-8")
    candidate_mode_lines = [
        line.strip()
        for line in bootstrap_command.splitlines()
        if 'find "$candidate_root" -type ' in line
    ]
    assert candidate_mode_lines == [
        'find "$candidate_root" -type f \\( -perm -100 -o -perm -010 -o -perm -001 \\) -exec chmod 500 {} +',
        'find "$candidate_root" -type f ! \\( -perm -100 -o -perm -010 -o -perm -001 \\) -exec chmod 400 {} +',
        'find "$candidate_root" -type d -exec chmod 500 {} +',
    ]
    authority_mode_lines = [
        line.strip()
        for line in bootstrap_command.splitlines()
        if 'find "$authority_root" -type ' in line
    ]
    assert authority_mode_lines == [
        'find "$authority_root" -type f \\( -perm -100 -o -perm -010 -o -perm -001 \\) -exec chmod 500 {} +',
        'find "$authority_root" -type f ! \\( -perm -100 -o -perm -010 -o -perm -001 \\) -exec chmod 400 {} +',
        'find "$authority_root" -type d -exec chmod 500 {} +',
    ]
    candidate_verify = (
        'run_verified_gateway_git_helper verify-tree --plan-file '
        '"$bootstrap_root/plan.json" --materialized-root "$candidate_root" '
        '--phase prepared_archive --strict-extras >/dev/null'
    )
    authority_verify = (
        'run_verified_gateway_git_helper verify-tree --plan-file '
        '"$bootstrap_root/authority-plan.json" '
        '--materialized-root "$authority_root" --phase prepared_archive '
        '--strict-extras >/dev/null'
    )
    assert bootstrap_command.count(candidate_verify) == 2
    assert bootstrap_command.count(authority_verify) == 2
    candidate_first_verify = bootstrap_command.index(candidate_verify)
    candidate_modes = bootstrap_command.index(candidate_mode_lines[0])
    candidate_second_verify = bootstrap_command.index(
        candidate_verify, candidate_first_verify + 1
    )
    assert candidate_first_verify < candidate_modes < candidate_second_verify
    authority_first_verify = bootstrap_command.index(authority_verify)
    authority_modes = bootstrap_command.index(authority_mode_lines[0])
    authority_second_verify = bootstrap_command.index(
        authority_verify, authority_first_verify + 1
    )
    assert authority_first_verify < authority_modes < authority_second_verify
    mode_root = tmp_path / "candidate"
    nested = mode_root / "nested"
    nested.mkdir(parents=True)
    executable = nested / "entrypoint.sh"
    regular = nested / "contract.json"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    regular.write_text("{}\n", encoding="utf-8")
    executable.chmod(0o755)
    regular.chmod(0o644)
    mode_probe = subprocess.run(
        [
            "bash",
            "-c",
            "set -Eeuo pipefail\n"
            'candidate_root="$1"\n'
            + "\n".join(candidate_mode_lines),
            "mode-probe",
            str(mode_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert mode_probe.returncode == 0, mode_probe.stderr
    try:
        assert (executable.stat().st_mode & 0o777) == 0o500
        assert (regular.stat().st_mode & 0o777) == 0o400
        assert (nested.stat().st_mode & 0o777) == 0o500
    finally:
        nested.chmod(0o700)
    assert (
        'exec(compile(source, "<exact-installed-controller-verifier>", "exec"))'
        in bootstrap_command
    )
    assert "--expected-controller-commit" in bootstrap_command
    assert "3" * 40 in bootstrap_command
    assert "--deploy-commit '" + commit + "'" in bootstrap_command
    assert "GATEWAY_RESTART_AUTHORITY_COMMIT='" + commit + "'" in bootstrap_command
    assert 'authority_root="$bootstrap_root/authority"' in bootstrap_command
    assert 'candidate_root="$bootstrap_root/candidate"' in bootstrap_command
    assert '--plan-file "$bootstrap_root/authority-plan.json"' in bootstrap_command
    assert '--plan-file "$bootstrap_root/plan.json"' in bootstrap_command
    assert 'GATEWAY_RESTART_AUTHORITY_ROOT="$authority_root"' in bootstrap_command
    controller_install = bootstrap_command.index(
        "controller_release=\"$controller_root/releases/" + commit + "\""
    )
    controller_activation = bootstrap_command.index(
        'mv -Tf -- "$controller_link" "$controller_root/current"'
    )
    candidate_exec = bootstrap_command.index('bash "$candidate_root/gw_restart.sh"')
    assert controller_install < controller_activation < candidate_exec
    assert "chmod 700 \"$(dirname \"$controller_root\")\"" in bootstrap_command
    assert "stat -c '%u:%g:%a' \"$controller_release/gw_restart.sh\"" in bootstrap_command
    assert "stat -c '%u:%g:%a' '/home/ec2-user/gw_restart.sh'" in bootstrap_command
    assert "cmp -s '/home/ec2-user/gw_restart.sh' \"$controller_release/gw_restart.sh\"" in bootstrap_command
    assert 'bash "$candidate_root/gw_restart.sh"' in bootstrap_command
    assert "--commit '" + commit + "'" in bootstrap_command
    assert (
        "--commit '"
        + commit
        + "' \\\n"
        + '          --miner-maintenance-bootstrap-plan "$bootstrap_root/plan.json" \\'
        in bootstrap_command
    )
    syntax = subprocess.run(
        ["bash", "-n", "-c", bootstrap_command],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert syntax.returncode == 0, syntax.stderr

    exec_start = bootstrap_command.index("      exec env \\\n")
    argument_probe = bootstrap_command[exec_start:].replace(
        "      exec env \\\n",
        "      printf '%s\\0' \\\n",
        1,
    )
    parsed = subprocess.run(
        [
            "bash",
            "-c",
            "set -Eeuo pipefail\n"
            "bootstrap_root=/tmp/gateway-miner-maintenance-bootstrap.fixture\n"
            'authority_root="$bootstrap_root/authority"\n'
            'candidate_root="$bootstrap_root/candidate"\n'
            + argument_probe,
        ],
        check=False,
        capture_output=True,
        timeout=5,
    )
    assert parsed.returncode == 0, parsed.stderr.decode("utf-8", "replace")
    arguments = parsed.stdout.rstrip(b"\0").split(b"\0")
    entrypoint = arguments.index(b"bash")
    assert arguments[entrypoint:] == [
        b"bash",
        b"/tmp/gateway-miner-maintenance-bootstrap.fixture/candidate/gw_restart.sh",
        b"--commit",
        commit.encode("ascii"),
        b"--miner-maintenance-bootstrap-plan",
        b"/tmp/gateway-miner-maintenance-bootstrap.fixture/plan.json",
        b"--miner-maintenance-bootstrap-root",
        b"/tmp/gateway-miner-maintenance-bootstrap.fixture",
        b"--miner-maintenance-handoff-file",
        b"/tmp/handoff",
        b"--miner-maintenance-handoff-nonce",
        ("2" * 64).encode("ascii"),
    ]


def test_general_rollback_bootstraps_current_authority_before_target_runtime() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    function_start = source.index("build_gateway_restart_command() {")
    function_end = source.index(
        "\n}\n\nrun_gateway_restart()", function_start
    ) + 2
    function_source = source[function_start:function_end]
    target_commit = "1" * 40
    authority_commit = "2" * 40
    shell_program = f"""
set -Eeuo pipefail
commit={shlex.quote(target_commit)}
branch_commit={shlex.quote(authority_commit)}
component=all
disable_miner_submissions_before_restart=0
GATEWAY_ENV_SECRET_ID=''
controller_verifier_b64='YQ=='
expected_controller_commit='{'3' * 40}'
GATEWAY_PYTHON_BIN='/usr/bin/python3.11'
GATEWAY_REPO_ROOT='/home/ec2-user/leadpoet_repo'
PRODUCTION_GATEWAY_RESTART_CONTROLLER_CURRENT='/controller/current'
PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT='/controller'
GATEWAY_RESTART='/home/ec2-user/gw_restart.sh'
RELEASE_PREFIX='weights/v2/release-evidence'
gateway_handoff_file=''
gateway_handoff_nonce=''
gateway_validator_requirements_remote='/tmp/validator-requirements.json'
gateway_counterpart_lineage_remote='/tmp/counterpart-lineage.json'
active_release_restart_invocation_id='restart-fixture'
paired_gateway_handoff_file='/tmp/leadpoet-gateway-paired-restart.fixture.ready'
paired_gateway_handoff_nonce='{'4' * 64}'
VALIDATOR_COORDINATION_TIMEOUT_SECONDS=9300
GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH='/tmp/gateway-requirements.json'
GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH='/tmp/gateway-lineage.json'
GATEWAY_KEY='/tmp/key'
GATEWAY_HOST='gateway.invalid'
ssh_common=('-o' 'BatchMode=yes')
{function_source}
build_gateway_restart_command
printf '%s\\0' "${{gateway_restart_command[@]}}"
"""
    rendered = subprocess.run(
        ["bash", "-c", shell_program],
        check=False,
        capture_output=True,
        timeout=5,
    )

    assert rendered.returncode == 0, rendered.stderr.decode("utf-8", "replace")
    remote_command = rendered.stdout.rstrip(b"\0").split(b"\0")[-1]
    encoded_match = re.search(
        rb"printf '%s' '([A-Za-z0-9+/=]+)' \| base64 --decode",
        remote_command,
    )
    assert encoded_match is not None
    bootstrap_command = base64.b64decode(
        encoded_match.group(1), validate=True
    ).decode("utf-8")

    assert "gateway-restart-controller-bootstrap" in bootstrap_command
    assert "--deploy-commit '" + authority_commit + "'" in bootstrap_command
    assert (
        "test \"$prepared_authority_sha\" = '" + authority_commit + "'"
        in bootstrap_command
    )
    assert (
        "GATEWAY_RESTART_AUTHORITY_COMMIT='" + authority_commit + "'"
        in bootstrap_command
    )
    assert 'bash "$authority_root/gw_restart.sh"' in bootstrap_command
    assert "--commit '" + target_commit + "'" in bootstrap_command
    assert "GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED='1'" in bootstrap_command
    assert "miner-maintenance-bootstrap" not in bootstrap_command
    assert "--miner-maintenance-bootstrap-plan" not in bootstrap_command
    assert 'candidate_root="$bootstrap_root/candidate"' not in bootstrap_command
    syntax = subprocess.run(
        ["bash", "-n", "-c", bootstrap_command],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_controller_authority_sidecars_use_bounded_nofollow_reads() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for function_name in (
        "validate_validator_initial_release_requirements",
        "validate_gateway_final_release_authority",
        "fetch_and_install_gateway_counterpart_lineage",
        "bind_component_validator_to_gateway_release_authority",
    ):
        start = source.index(f"{function_name}() {{")
        end = source.index("\n}\n", start)
        function = source[start:end]

        assert "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW" in function
        assert "stat.S_ISREG(metadata.st_mode)" in function
        assert "max_document_bytes = 4 * 1024 * 1024" in function
        assert re.search(
            r"os\.read\([A-Za-z_][A-Za-z0-9_]*, max_document_bytes \+ 1\)",
            function,
        )
        assert ".read_text(" not in function
        assert ".read_bytes(" not in function


def test_attested_release_restart_operator_rejects_invalid_input() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--commit", "abc123"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 2
    assert "lowercase full 40-character SHA" in result.stderr
    assert "Fetching current public V2 compatibility authority" not in result.stdout


@pytest.mark.parametrize(
    "local_python_args",
    [[], ["--local-python="], ["--local-python", ""]],
)
def test_attested_release_restart_operator_requires_local_python(
    local_python_args: list[str],
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--commit", commit, *local_python_args],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 2
    assert "--local-python requires one absolute venv/bin/python path" in result.stderr
    assert "Fetching current public V2 compatibility authority" not in result.stdout


def test_miner_maintenance_operator_rejects_replace_ref_before_ssh(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "operator-repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    copied_script = scripts / SCRIPT.name
    copied_script.write_bytes(SCRIPT.read_bytes())
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "scripts"], check=True)
    commit_command = [
        "git",
        "-C",
        str(repository),
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
    ]
    subprocess.run([*commit_command, "official"], check=True)
    official = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    (repository / "replacement.txt").write_text("replacement\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "replacement.txt"],
        check=True,
    )
    subprocess.run([*commit_command, "replacement"], check=True)
    replacement = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", official, replacement],
        check=True,
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_sentinel = tmp_path / "ssh-was-called"
    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text(
        f"#!/bin/bash\ntouch {ssh_sentinel}\nexit 99\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    gateway_key = tmp_path / "gateway.pem"
    validator_key = tmp_path / "validator.pem"
    gateway_key.write_text("test\n", encoding="utf-8")
    validator_key.write_text("test\n", encoding="utf-8")
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }
    environment.update(
        {
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "LEADPOET_GATEWAY_SSH_KEY": str(gateway_key),
            "LEADPOET_VALIDATOR_SSH_KEY": str(validator_key),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(copied_script),
            "--commit",
            official,
            "--local-python",
            sys.executable,
            "--component",
            "all",
            "--disable-miner-submissions-before-restart",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )

    assert result.returncode == 1
    assert "replacement refs" in result.stderr
    assert not ssh_sentinel.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LEADPOET_GATEWAY_SSH_HOST", "ec2-user@127.0.0.1"),
        ("LEADPOET_GATEWAY_RESTART_PATH", "/tmp/wrapper'$(touch injected)"),
        ("LEADPOET_GATEWAY_REPO_ROOT", "/tmp/alternate-repo"),
        ("LEADPOET_GATEWAY_PYTHON_BIN", "/usr/bin/python3"),
        (
            "LEADPOET_GATEWAY_DEPLOY_READINESS_PATH",
            "/tmp/alternate-readiness.json",
        ),
        ("GATEWAY_RESTART_CONTROLLER_ROOT", "/tmp/controller"),
        ("GATEWAY_RESTART_CONTROLLER_CURRENT", "/tmp/controller/current"),
    ],
)
def test_miner_maintenance_operator_rejects_topology_override_before_ssh(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_sentinel = tmp_path / "ssh-was-called"
    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text(
        f"#!/bin/bash\ntouch {shlex.quote(str(ssh_sentinel))}\nexit 99\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        name: value,
    }

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            "a" * 40,
            "--local-python",
            "/usr/bin/python3",
            "--component",
            "all",
            "--disable-miner-submissions-before-restart",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )

    assert result.returncode == 2
    assert "fixed production gateway topology" in result.stderr
    assert not ssh_sentinel.exists()


def test_attested_release_restart_operator_documents_one_command_modes() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "--component all|gateway|validator" in result.stdout
    assert "single-component restart is accepted only when the other component" in (
        result.stdout
    )


def test_gateway_cleanup_kills_term_ignoring_owned_process_group(
    tmp_path: Path,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    prefix_end = source.index("trap cleanup EXIT") + len("trap cleanup EXIT")
    late_activation = tmp_path / "late-activation"
    process_group_ready = tmp_path / "process-group-ready"
    cancellation = tmp_path / "cancellation"
    harness = tmp_path / "cleanup-harness.sh"
    harness.write_text(
        source[:prefix_end]
        + "\ntrap - EXIT\n"
        + "publish_gateway_handoff_value() { printf '%s\\n' \"$1\" > "
        + shlex.quote(str(cancellation))
        + "; }\n"
        + "ssh() { return 0; }\nssh_common=(test)\nGATEWAY_KEY=x\nGATEWAY_HOST=x\n"
        + "VALIDATOR_KEY=x\nVALIDATOR_HOST=x\n"
        + "temporary_root=''\ncoordination_file=''\nvalidator_job=''\n"
        + "failure_marker_job=''\ngateway_handoff_file='/tmp/test-handoff'\n"
        + "gateway_handoff_nonce='nonce'\ncommit='"
        + "a" * 40
        + "'\n"
        + "python3 -c "
        + shlex.quote(
            "import os,signal,sys,time\n"
            "os.setsid()\n"
            "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
            "child=os.fork()\n"
            "if child == 0:\n"
            " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            " time.sleep(3)\n"
            " open(sys.argv[2], 'w').write('unsafe')\n"
            " time.sleep(30)\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n"
        )
        + " "
        + shlex.quote(str(process_group_ready))
        + " "
        + shlex.quote(str(late_activation))
        + " &\n"
        + "gateway_job=$!\n"
        + "for _ in $(seq 1 100); do [ -s "
        + shlex.quote(str(process_group_ready))
        + " ] && break; sleep 0.01; done\n"
        + "gateway_job_pgid=$gateway_job\n"
        + "cleanup\n"
        + "trap - EXIT\n"
        + "sleep 1.5\n"
        + "test ! -e "
        + shlex.quote(str(late_activation))
        + "\n"
        + "test \"$(cat "
        + shlex.quote(str(cancellation))
        + ")\" = 'failed:"
        + "a" * 40
        + "'\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 7
    assert not late_activation.exists()


@pytest.fixture(scope="module")
def dependency_complete_readiness_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_dependency_complete_readiness_venv(
        tmp_path_factory.mktemp("restart-readiness-venv") / "venv"
    )


def _fake_operator_commands(
    tmp_path: Path,
    commit: str,
    readiness_python: Path,
) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    barrier = tmp_path / "barrier"
    gateway_handoff = tmp_path / "gateway-handoff"
    gateway_started = tmp_path / "gateway-started"
    gateway_complete = tmp_path / "gateway-complete"
    gateway_observation, validator_observation = _fake_readiness_observations(
        tmp_path, commit
    )
    release_channel = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(commit),
        validator_release_manifest=_validator_manifest(commit),
    )
    active_requirements = build_active_release_requirements_v2(
        candidate_commit_sha=commit,
        authority_commit_sha=commit,
        restart_invocation_id="restart-fixture",
        transition_commit_shas=(commit,),
        active_graphs={},
        expected_lineage_id="sha256:" + "a" * 64,
        boot_verifier=lambda identity: identity,
    )
    initial_requirements = tmp_path / "validator-active-release-requirements.json"
    final_requirements = tmp_path / "gateway-active-release-requirements.json"
    final_lineage = tmp_path / "gateway-active-release-lineage.json"
    initial_requirements.write_text(
        json.dumps(active_requirements, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_requirements.write_text(
        json.dumps(active_requirements, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_lineage.write_text(
        json.dumps(
            build_release_lineage_v2(
                (release_channel,),
                current_commit=commit,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        "from leadpoet_canonical import attested_v2\n"
        "attested_v2.verify_boot_identity_nitro = lambda *args, **kwargs: {}\n",
        encoding="utf-8",
    )

    real_git = shutil.which("git")
    real_python = str(readiness_python)
    assert real_git
    real_venv = Path(real_python).parent.parent
    real_site_packages = (
        real_venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert (real_venv / "pyvenv.cfg").is_file()
    assert real_site_packages.is_dir()
    git = bin_dir / "git"
    git.write_text(
        f"""#!/bin/bash
set -euo pipefail
last_arg="${{!#}}"
for arg in "$@"; do
  if [ "$arg" = "fetch" ]; then
    exit 0
  fi
  if [ "$arg" = "origin/main:Leadpoet/utils/exact_commit_restart_v2.py" ]; then
    cat "$FAKE_OPERATOR_EXACT_HELPER"
    exit 0
  fi
done
if [[ " $* " == *" rev-parse "* ]]; then
  case "$last_arg" in
    "$FAKE_OPERATOR_SELECTED_COMMIT:"gateway/*|\
    "$FAKE_OPERATOR_SELECTED_COMMIT:"leadpoet_canonical/*|\
    "$FAKE_OPERATOR_SELECTED_COMMIT:"leadpoet_observability/*|\
    "$FAKE_OPERATOR_SELECTED_COMMIT:"scripts/restart_attested_release_local.sh|\
    "$FAKE_OPERATOR_SELECTED_COMMIT:"scripts/verify_installed_gateway_controller_v1.py|\
    "$FAKE_OPERATOR_SELECTED_COMMIT:"validator_tee/*)
      printf '%s\\n' "$FAKE_OPERATOR_SOURCE_BLOB"
      exit 0
      ;;
  esac
fi
if [[ " $* " == *" hash-object --no-filters "* ]]; then
  case "$last_arg" in
    "$FAKE_OPERATOR_REPO_ROOT/"gateway/*|\
    "$FAKE_OPERATOR_REPO_ROOT/"leadpoet_canonical/*|\
    "$FAKE_OPERATOR_REPO_ROOT/"leadpoet_observability/*|\
    "$FAKE_OPERATOR_REPO_ROOT/"scripts/restart_attested_release_local.sh|\
    "$FAKE_OPERATOR_REPO_ROOT/"scripts/verify_installed_gateway_controller_v1.py|\
    "$FAKE_OPERATOR_REPO_ROOT/"validator_tee/*)
      if [ -e "$FAKE_OPERATOR_SOURCE_DRIFT_MARKER" ]; then
        printf '%s\\n' "$FAKE_OPERATOR_DRIFTED_SOURCE_BLOB"
      else
        printf '%s\\n' "$FAKE_OPERATOR_SOURCE_BLOB"
      fi
      exit 0
      ;;
  esac
fi
exec {real_git} "$@"
""",
        encoding="utf-8",
    )

    ssh = bin_dir / "ssh"
    ssh.write_text(
        """#!/bin/bash
set -euo pipefail
command="${!#}"
record() {
  printf '%s\\n' "$1" >> "$FAKE_OPERATOR_EVENTS"
}
case "$command" in
  *"readlink --"*restart-controller*)
    printf 'releases/%s\n' "$FAKE_OPERATOR_SELECTED_COMMIT"
    ;;
  *"'readiness-transition' <<'PY'"*)
    record readiness_invalidated
    if [ "${FAKE_RETARGET_LOCAL_PYTHON_AFTER_TRANSITION:-0}" = "1" ]; then
      mv -f -- "$FAKE_OPERATOR_ALT_PYTHON" "$FAKE_OPERATOR_LOCAL_PYTHON"
    fi
    if [ "${FAKE_DRIFT_SOURCE_AFTER_TRANSITION:-0}" = "1" ]; then
      touch "$FAKE_OPERATOR_SOURCE_DRIFT_MARKER"
    fi
    ;;
  *"'readiness-final' <<'PY'"*)
    record readiness_finalized
    ;;
  *prepare_active_release_lineage_v2*validator-initial*)
    record validator_requirements_prepared
    ;;
  *"base64 --decode"*)
    encoded="$(printf '%s\n' "$command" | sed -E -n "s/.*printf '%s' '([A-Za-z0-9+/=]*)'.*/\\1/p")"
    if [ -z "$encoded" ]; then
      record invalid_bootstrap_transport
      exit 72
    fi
    decoded="$(printf '%s' "$encoded" | base64 --decode)"
    bash -n -c "$decoded"
    if [[ "$decoded" == *validator_restart.sh* ]]; then
    record validator_command_syntax
    record validator_start
    if [[ "$decoded" == *"VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS='/tmp/leadpoet-validator-recovery-requirements."* ]] \
        && [[ "$decoded" == *"VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE='/tmp/leadpoet-validator-recovery-lineage."* ]]; then
      record validator_missing_runtime_recovery_enabled
    fi
    trap 'record validator_cancelled; record validator_cleanup; exit 143' HUP INT TERM
    record validator_exact_commit_handoff
    record validator_prepare_started
    printf '%s\\n' "Capturing the official subnet restart start before release acquisition"
    printf '%s\\n' "Acquiring the independently built V2 release channel"
    printf '%s\\n' "Prepared validator active release requirements sidecar"
    record validator_captured
    if [[ "$decoded" == *"VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE=''"* ]]; then
      record validator_image_prepared
      record validator_activation
      record validator_complete
      exit 0
    fi
    for _ in $(seq 1 500); do
      if [ -e "$FAKE_OPERATOR_GATEWAY_STARTED" ]; then
        record validator_image_prepared
        break
      fi
      sleep 0.01
    done
    for _ in $(seq 1 500); do
      if [ "${FAKE_VALIDATOR_EXIT_AFTER_GATEWAY_HANDOFF:-0}" = "1" ] \
          && [ -e "$FAKE_OPERATOR_GATEWAY_HANDOFF" ]; then
        record validator_exit_after_gateway_handoff
        exit 78
      fi
      if [ -e "$FAKE_OPERATOR_BARRIER" ]; then
        marker="$(cat "$FAKE_OPERATOR_BARRIER")"
        if [ "$marker" = "$FAKE_VALIDATOR_COMMIT" ]; then
          record validator_activation
          record validator_complete
          exit 0
        fi
        if [ "$marker" = "failed:$FAKE_VALIDATOR_COMMIT" ]; then
          record validator_alignment_failed
          record validator_cleanup
          exit 76
        fi
        record validator_invalid_barrier
        exit 77
      fi
      sleep 0.01
    done
    record validator_barrier_timeout
    exit 70
    fi
    record gateway_start
    trap 'record gateway_cancelled; exit 143' HUP INT TERM
    touch "$FAKE_OPERATOR_GATEWAY_STARTED"
    if [[ "$decoded" == *"GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED='1'"* ]]; then
      printf '%s\n' "Gateway pre-shutdown checks complete; awaiting paired validator liveness handoff"
      for _ in $(seq 1 500); do
        if [ -e "$FAKE_OPERATOR_GATEWAY_HANDOFF" ]; then
          record gateway_handoff_received
          break
        fi
        sleep 0.01
      done
      if [ ! -e "$FAKE_OPERATOR_GATEWAY_HANDOFF" ]; then
        record gateway_handoff_timeout
        exit 70
      fi
      if [ "${FAKE_VALIDATOR_EXIT_AFTER_GATEWAY_HANDOFF:-0}" = "1" ]; then
        sleep 3
      fi
    fi
    record gateway_destructive_started
    if [ "${FAKE_GATEWAY_RESTART_FAIL:-0}" = "1" ]; then
      record gateway_failed
      exit 73
    fi
    touch "$FAKE_OPERATOR_GATEWAY_COMPLETE"
    record gateway_complete
    ;;
  *leadpoet-gateway-paired-restart*"mv -f --"*)
    if [[ "$command" == *"failed:"* ]]; then
      record paired_gateway_failure_handoff
    else
      record paired_gateway_handoff_released
    fi
    touch "$FAKE_OPERATOR_GATEWAY_HANDOFF"
    ;;
  *leadpoet-validator-active-release-requirements*|*leadpoet-gateway-active-release*|*leadpoet-validator-counterpart-release-lineage*|*leadpoet-validator-recovery-*)
    record active_release_authority_installed
    ;;
  *"mv -f --"*)
    if [[ "$command" == *"failed:"* ]]; then
      record failure_barrier_publish_started
      if [ -n "${FAKE_FAILURE_MARKER_DELAY_SECONDS:-}" ]; then
        sleep "$FAKE_FAILURE_MARKER_DELAY_SECONDS"
      fi
      printf '%s\\n' "failed:$FAKE_VALIDATOR_COMMIT" > "$FAKE_OPERATOR_BARRIER"
      record failure_barrier_released
    elif [ ! -e "$FAKE_OPERATOR_GATEWAY_COMPLETE" ]; then
      record barrier_before_gateway
      exit 71
    else
      printf '%s\\n' "$FAKE_VALIDATOR_COMMIT" > "$FAKE_OPERATOR_BARRIER"
      record barrier_released
    fi
    ;;
  *gateway_exact_release_ready*)
    record gateway_verified
    if [ "${FAKE_GATEWAY_VERIFY_FAIL:-0}" = "1" ]; then
      exit 74
    fi
    cat "$FAKE_GATEWAY_OBSERVATION"
    ;;
  *validator_exact_release_ready*)
    record validator_verified
    if [ "${FAKE_VALIDATOR_VERIFY_FAIL:-0}" = "1" ]; then
      exit 75
    fi
    cat "$FAKE_VALIDATOR_OBSERVATION"
    ;;
  *"docker inspect"*VALIDATOR_V2_DEPLOY_COMMIT*)
    if [ "${FAKE_VALIDATOR_RUNTIME_PROBE_FAIL:-0}" = "1" ]; then
      printf '%s\\n' "permission denied" >&2
      exit 43
    fi
    if [ "${FAKE_VALIDATOR_RUNTIME_MISSING:-0}" = "1" ]; then
      printf '%s\\n' "Error: No such object: leadpoet-validator-main" >&2
      exit 44
    fi
    printf '%s\\n' "$FAKE_VALIDATOR_COMMIT"
    record validator_active_probe
    ;;
  *"/build-info"*git_commit*)
    printf '%s\\n' "$FAKE_GATEWAY_COMMIT"
    record gateway_active_probe
    ;;
  *"rm -f --"*)
    rm -f "$FAKE_OPERATOR_BARRIER" "$FAKE_OPERATOR_GATEWAY_HANDOFF"
    record barrier_cleanup
    ;;
  *)
    record unknown_ssh_command
    printf '%s\\n' "$command" >&2
    exit 72
    ;;
esac
""",
        encoding="utf-8",
    )
    scp = bin_dir / "scp"
    scp.write_text(
        """#!/bin/bash
set -euo pipefail
source_path="${@: -2:1}"
destination_path="${@: -1}"
case "$destination_path" in
  *:*) ;;
  *)
    case "$source_path" in
      *leadpoet-validator-active-release-requirements*)
        cp "$FAKE_VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS" "$destination_path"
        ;;
      *gateway-v2-release-requirements.json*)
        cp "$FAKE_GATEWAY_ACTIVE_RELEASE_REQUIREMENTS" "$destination_path"
        ;;
      *gateway-v2-release-lineage.json*)
        cp "$FAKE_GATEWAY_ACTIVE_RELEASE_LINEAGE" "$destination_path"
        ;;
    esac
    ;;
esac
printf '%s\\n' active_release_authority_transferred >> "$FAKE_OPERATOR_EVENTS"
""",
        encoding="utf-8",
    )
    git.chmod(0o755)
    ssh.chmod(0o755)
    scp.chmod(0o755)
    (tmp_path / "pyvenv.cfg").write_text(
        "home = test-local-readiness-adapter\n",
        encoding="utf-8",
    )
    (
        tmp_path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ).mkdir(parents=True)
    python = bin_dir / "python3"
    python.write_text(
        """#!/bin/bash
set -euo pipefail
if [ "$#" -ge 7 ] \
    && [ "$1" = "-I" ] \
    && [ "$2" = "-S" ] \
    && [ "$3" = "-B" ] \
    && [ "$4" = "-c" ]; then
  bootstrap="$5"
  root="$6"
  shift 7
  readiness_source="$(mktemp)"
  trap 'rm -f -- "$readiness_source"' EXIT
  cat > "$readiness_source"
  if grep -Fq 'leadpoet.local_readiness_python.v1' "$readiness_source"; then
    exec "$FAKE_OPERATOR_REAL_PYTHON" -I -S -B -c "$bootstrap" \
      "$root" "$FAKE_OPERATOR_REAL_SITE_PACKAGES" \
      "$FAKE_OPERATOR_REAL_PYTHON" "$FAKE_OPERATOR_REAL_VENV" \
      "$FAKE_OPERATOR_REAL_PYTHON" "$FAKE_OPERATOR_REAL_SITE_PACKAGES" \
      < "$readiness_source"
  fi
  {
    printf '%s\n' \
      'from leadpoet_canonical import attested_v2; attested_v2.verify_boot_identity_nitro = lambda *args, **kwargs: {}'
    cat "$readiness_source"
  } | "$FAKE_OPERATOR_REAL_PYTHON" -I -S -B -c "$bootstrap" \
    "$root" "$FAKE_OPERATOR_REAL_SITE_PACKAGES" "$@"
  exit "${PIPESTATUS[1]}"
fi
exec "$FAKE_OPERATOR_REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    alternate_python = bin_dir / "python3.alternate"
    alternate_python.write_text(
        "#!/bin/bash\nexit 99\n",
        encoding="utf-8",
    )
    alternate_python.chmod(0o755)

    for name in ("gateway.pem", "validator.pem"):
        path = tmp_path / name
        path.write_text("test-only\n", encoding="utf-8")
        path.chmod(0o600)

    os.environ.pop("FAKE_OPERATOR_EVENTS", None)
    env = tmp_path / "operator-env"
    env.write_text(
        "\n".join(
            (
                f"FAKE_OPERATOR_EVENTS={events}",
                f"FAKE_OPERATOR_BARRIER={barrier}",
                f"FAKE_OPERATOR_GATEWAY_HANDOFF={gateway_handoff}",
                f"FAKE_OPERATOR_GATEWAY_STARTED={gateway_started}",
                f"FAKE_OPERATOR_GATEWAY_COMPLETE={gateway_complete}",
                f"FAKE_GATEWAY_COMMIT={commit}",
                f"FAKE_VALIDATOR_COMMIT={commit}",
                f"FAKE_OPERATOR_SELECTED_COMMIT={commit}",
                f"FAKE_GATEWAY_OBSERVATION={gateway_observation}",
                f"FAKE_VALIDATOR_OBSERVATION={validator_observation}",
                f"FAKE_VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS={initial_requirements}",
                f"FAKE_GATEWAY_ACTIVE_RELEASE_REQUIREMENTS={final_requirements}",
                f"FAKE_GATEWAY_ACTIVE_RELEASE_LINEAGE={final_lineage}",
                f"FAKE_OPERATOR_SITE_ROOT={tmp_path}",
                f"FAKE_OPERATOR_REAL_PYTHON={real_python}",
                f"FAKE_OPERATOR_REAL_VENV={real_venv}",
                f"FAKE_OPERATOR_REAL_SITE_PACKAGES={real_site_packages}",
                f"FAKE_OPERATOR_REPO_ROOT={ROOT}",
                f"FAKE_OPERATOR_SOURCE_BLOB={'f' * 40}",
                f"FAKE_OPERATOR_DRIFTED_SOURCE_BLOB={'e' * 40}",
                f"FAKE_OPERATOR_SOURCE_DRIFT_MARKER={tmp_path / 'source-drift'}",
                f"FAKE_OPERATOR_LOCAL_PYTHON={python}",
                f"FAKE_OPERATOR_ALT_PYTHON={alternate_python}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return bin_dir, events


def _operator_env(tmp_path: Path, bin_dir: Path, commit: str) -> dict[str, str]:
    values = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "LEADPOET_GATEWAY_SSH_KEY": str(tmp_path / "gateway.pem"),
        "LEADPOET_VALIDATOR_SSH_KEY": str(tmp_path / "validator.pem"),
        "FAKE_GATEWAY_COMMIT": commit,
        "FAKE_VALIDATOR_COMMIT": commit,
        "LEADPOET_ACTIVE_RELEASE_RESTART_INVOCATION_ID": "restart-fixture",
        "FAKE_OPERATOR_EXACT_HELPER": str(
            ROOT / "Leadpoet" / "utils" / "exact_commit_restart_v2.py"
        ),
        "PYTHONPATH": os.pathsep.join(
            value
            for value in (
                str(tmp_path),
                str(ROOT),
                os.environ.get("PYTHONPATH", ""),
            )
            if value
        ),
    }
    for line in (tmp_path / "operator-env").read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _operator_argv(
    bin_dir: Path,
    commit: str,
    *extra: str,
) -> list[str]:
    return [
        "bash",
        str(SCRIPT),
        "--commit",
        commit,
        "--local-python",
        str(bin_dir / "python3"),
        *extra,
    ]


def test_paired_operator_overlaps_preparation_and_gates_validator_activation(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "readiness_invalidated",
        "validator_exact_commit_handoff",
        "validator_command_syntax",
        "validator_prepare_started",
        "validator_captured",
        "gateway_start",
        "validator_image_prepared",
        "gateway_complete",
        "barrier_released",
        "validator_activation",
        "validator_complete",
        "gateway_verified",
        "validator_verified",
        "readiness_finalized",
    ]
    positions = {event: observed.index(event) for event in required}
    assert (
        positions["readiness_invalidated"]
        < positions["validator_prepare_started"]
        < positions["validator_captured"]
        < positions["gateway_start"]
        < positions["validator_image_prepared"]
        < positions["gateway_complete"]
    )
    assert (
        positions["gateway_complete"]
        < positions["barrier_released"]
        < positions["validator_activation"]
        < positions["validator_complete"]
        < positions["gateway_verified"]
        < positions["validator_verified"]
        < positions["readiness_finalized"]
    )
    assert "barrier_before_gateway" not in observed
    assert "validator_forward_handoff" not in observed
    assert "SUCCESS: gateway and validator are aligned" in result.stdout


def test_operator_rejects_non_venv_local_python_before_ssh(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    standalone_dir = tmp_path / "standalone"
    standalone_dir.mkdir()
    standalone_python = standalone_dir / "python3"
    standalone_python.symlink_to(sys.executable)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--local-python",
            str(standalone_python),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 2
    assert "must be a venv bin/python executable" in result.stderr
    assert not events.exists()


def test_operator_rejects_local_python_retarget_before_final_readiness(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_RETARGET_LOCAL_PYTHON_AFTER_TRANSITION"] = "1"

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode != 0
    assert "local readiness Python binding changed after preflight" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "readiness_invalidated" in observed
    assert "readiness_finalized" not in observed


def test_operator_rejects_candidate_source_drift_before_final_readiness(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_DRIFT_SOURCE_AFTER_TRANSITION"] = "1"

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode != 0
    assert "local readiness candidate source differs from exact commit" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "readiness_invalidated" in observed
    assert "readiness_finalized" not in observed


def test_operator_rejects_initial_candidate_source_drift_before_ssh(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    Path(environment["FAKE_OPERATOR_SOURCE_DRIFT_MARKER"]).touch()

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode != 0
    assert "local readiness candidate source differs from exact commit" in result.stderr
    assert "local readiness Python preflight failed before production mutation" in result.stderr
    assert "Local readiness Python preflight:" not in result.stdout
    assert not events.exists()


def test_gateway_only_operator_rejects_mismatched_validator(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_COMMIT"] = "b" * 40

    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "gateway"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert "use --component all" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "validator_active_probe" in observed
    assert "readiness_invalidated" not in observed
    assert "gateway_start" not in observed


def test_gateway_only_operator_requires_healthy_matching_validator(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )

    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "gateway"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "validator_active_probe",
        "validator_verified",
        "readiness_invalidated",
        "gateway_start",
        "gateway_complete",
        "gateway_verified",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)
    assert observed.count("validator_verified") == 2
    assert observed.index("readiness_invalidated") < observed.index("gateway_start")
    assert observed.index("validator_verified") < observed.index(
        "readiness_invalidated"
    )
    assert len(observed) - 1 - observed[::-1].index(
        "validator_verified"
    ) < observed.index(
        "readiness_finalized"
    )


def test_gateway_only_operator_rejects_unhealthy_matching_validator(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_VERIFY_FAIL"] = "1"

    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "gateway"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 75
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "validator_verified" in observed
    assert "readiness_invalidated" not in observed
    assert "gateway_start" not in observed


def test_validator_only_operator_rejects_mismatched_gateway(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_COMMIT"] = "b" * 40

    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "validator"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert "use --component all" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "gateway_active_probe" in observed
    assert "readiness_invalidated" not in observed
    assert "validator_start" not in observed


def test_validator_only_operator_requires_healthy_matching_gateway(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )

    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "validator"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "gateway_active_probe",
        "gateway_verified",
        "readiness_invalidated",
        "validator_start",
        "validator_complete",
        "validator_verified",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)
    assert observed.count("gateway_verified") == 2
    assert observed.index("gateway_verified") < observed.index(
        "readiness_invalidated"
    )
    assert observed.index("readiness_invalidated") < observed.index(
        "validator_start"
    )
    assert len(observed) - 1 - observed[::-1].index(
        "gateway_verified"
    ) < observed.index(
        "readiness_finalized"
    )


def test_validator_only_operator_recovers_missing_runtime_from_gateway_pair(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_RUNTIME_MISSING"] = "1"
    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "validator"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "gateway_verified",
        "active_release_authority_installed",
        "readiness_invalidated",
        "validator_missing_runtime_recovery_enabled",
        "validator_complete",
        "validator_verified",
        "readiness_finalized",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)


def test_validator_only_operator_rejects_unknown_runtime_probe_failure(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_RUNTIME_PROBE_FAIL"] = "1"
    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "validator"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 1
    assert "runtime state could not be established safely" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "readiness_invalidated" not in observed
    assert "validator_start" not in observed


def test_validator_only_operator_rejects_unhealthy_matching_gateway(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_VERIFY_FAIL"] = "1"

    result = subprocess.run(
        _operator_argv(bin_dir, commit, "--component", "validator"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 74
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "gateway_verified" in observed
    assert "readiness_invalidated" not in observed
    assert "validator_start" not in observed


def test_paired_operator_failure_marker_cleans_prepared_validator(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_RESTART_FAIL"] = "1"
    environment["FAKE_FAILURE_MARKER_DELAY_SECONDS"] = "0.5"

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert "gateway restart exited before coordinated completion (status 73)" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "readiness_invalidated",
        "validator_exact_commit_handoff",
        "validator_prepare_started",
        "validator_captured",
        "gateway_start",
        "validator_image_prepared",
        "gateway_failed",
        "validator_cancelled",
        "validator_cleanup",
        "failure_barrier_released",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)
    assert observed.index("gateway_failed") < observed.index(
        "failure_barrier_publish_started"
    )
    assert observed.index("validator_cleanup") < observed.index(
        "failure_barrier_released"
    )
    assert "validator_alignment_failed" not in observed
    assert "barrier_released" not in observed
    assert "validator_activation" not in observed
    assert "validator_complete" not in observed
    assert "gateway_verified" not in observed
    assert "validator_verified" not in observed
    assert "readiness_finalized" not in observed
    assert "barrier_cleanup" in observed


def test_paired_operator_cancels_gateway_when_validator_exits_after_liveness_handoff(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_EXIT_AFTER_GATEWAY_HANDOFF"] = "1"

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert "validator restart exited before coordinated completion (status 78)" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    for event in (
        "paired_gateway_handoff_released",
        "gateway_handoff_received",
        "validator_exit_after_gateway_handoff",
        "gateway_cancelled",
    ):
        assert event in observed
    assert observed.index("paired_gateway_handoff_released") < observed.index(
        "validator_exit_after_gateway_handoff"
    )
    assert observed.index("validator_exit_after_gateway_handoff") < observed.index(
        "gateway_cancelled"
    )
    assert "gateway_destructive_started" not in observed
    assert "gateway_complete" not in observed
    assert "readiness_finalized" not in observed


@pytest.mark.parametrize(
    ("failure_flag", "failed_event"),
    [
        ("FAKE_GATEWAY_VERIFY_FAIL", "gateway_verified"),
        ("FAKE_VALIDATOR_VERIFY_FAIL", "validator_verified"),
    ],
)
def test_paired_operator_final_probe_failure_leaves_resume_blocked(
    tmp_path: Path,
    dependency_complete_readiness_python: Path,
    failure_flag: str,
    failed_event: str,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(
        tmp_path, commit, dependency_complete_readiness_python
    )
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment[failure_flag] = "1"

    result = subprocess.run(
        _operator_argv(bin_dir, commit),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode in {74, 75}
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "readiness_invalidated" in observed
    assert failed_event in observed
    assert "readiness_finalized" not in observed
    assert "SUCCESS: gateway and validator are aligned" not in result.stdout
