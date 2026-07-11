from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
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
            'echo "Starting Research Lab provider evidence proxy"',
            'setsid python3 -u -m gateway.main',
            'sleep 240',
            'curl -fsS http://localhost:8000/health',
            'GATEWAY_DEPLOY_STAGE="host_restart_script_install"',
            'finalize_deployment_record succeeded',
        ),
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
    assert 'setsid python3 -u -m gateway.main' in script
    assert 'pkill -9 -f "python3 -u -m gateway.main"' in script


def test_gateway_restart_has_fail_closed_lock_and_no_validator_deploy_gate() -> None:
    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert 'flock -n 9' in script
    assert 'another gateway restart is already running' in script
    assert 'GATEWAY_DEPLOY_COMMIT' not in script  # The helper consumes the optional env safely.
    assert 'VALIDATOR_GATEWAY_PCR0_CACHE_FILE' not in script
    assert 'independent_gateway_identity' not in script


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
