"""Tests for supervising gateway.main under systemd.

The value of the change is that merging it is the whole install, so the things
worth pinning are the ones a future edit could silently break: the unit
template renders to something that always restarts, the launcher replays the
environment snapshot gw_restart.sh writes, it execs the exact command line the
restart script's `pgrep` pattern matches, and gw_restart.sh still has a
non-systemd fallback and stops the unit before the destructive `pkill`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "gateway_supervised_launch.sh"
UNIT_TEMPLATE = REPO_ROOT / "config" / "systemd" / "leadpoet-gateway.service"
RESTART_SCRIPT = REPO_ROOT / "gw_restart.sh"

# Kept identical to the `env -u` list in gw_restart.sh's unsupervised launch.
RESTART_ONLY_VARIABLES = (
    "GATEWAY_MINER_MAINTENANCE_PROOF_FD",
    "GATEWAY_REBENCHMARK_RETRY_RECONCILIATION_HELPER",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_EXACT_COMMIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED",
    "GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT",
    "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_FILE",
    "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_NONCE",
    "GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_TIMEOUT_SECONDS",
    "GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS",
    "GATEWAY_COUNTERPART_RELEASE_LINEAGE",
)


def read_restart_script() -> str:
    return RESTART_SCRIPT.read_text()


@pytest.fixture()
def launch_harness(tmp_path: Path):
    """A fake gateway interpreter plus the env snapshot the launcher replays."""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    log_file = tmp_path / "gateway.log"

    python_bin = tmp_path / "python3"
    python_bin.write_text(
        "#!/bin/bash\n"
        'echo "ARGV: $*"\n'
        'echo "CWD: $PWD"\n'
        'echo "SECRET: ${SUPABASE_SERVICE_ROLE_KEY:-missing}"\n'
        'echo "SPACED: ${GATEWAY_SPACED_VALUE:-missing}"\n'
        + "".join(
            f'echo "{name}: ${{{name}-unset}}"\n' for name in RESTART_ONLY_VARIABLES
        )
    )
    python_bin.chmod(0o755)

    snapshot = tmp_path / "gateway-launch-env.sh"
    lines = [
        f'declare -x GATEWAY_PYTHON_BIN="{python_bin}"',
        f'declare -x LEADPOET_REPO_ROOT="{repo_root}"',
        f'declare -x GATEWAY_LOG_FILE="{log_file}"',
        'declare -x SUPABASE_SERVICE_ROLE_KEY="service-role-value"',
        'declare -x GATEWAY_SPACED_VALUE="one two  three"',
    ]
    lines += [f'declare -x {name}="restart-only"' for name in RESTART_ONLY_VARIABLES]
    snapshot.write_text("\n".join(lines) + "\n")
    snapshot.chmod(0o600)

    class Harness:
        def __init__(self) -> None:
            self.snapshot = snapshot
            self.log_file = log_file
            self.repo_root = repo_root

        def run(self, env_path: Path | None = None) -> subprocess.CompletedProcess:
            env = {
                # A deliberately bare environment: everything the gateway needs
                # has to come from the snapshot, not from the caller.
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "GATEWAY_SUPERVISED_LAUNCH_ENV": str(
                    snapshot if env_path is None else env_path
                ),
            }
            return subprocess.run(
                [str(LAUNCHER)],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

    return Harness()


def test_launcher_execs_the_pgrep_matched_command_line(launch_harness) -> None:
    """gw_restart.sh finds the gateway by an exact cmdline; exec must preserve it."""

    result = launch_harness.run()
    assert result.returncode == 0, result.stderr
    logged = launch_harness.log_file.read_text()
    assert "ARGV: -u -m gateway.main" in logged
    assert f"CWD: {launch_harness.repo_root}" in logged


def test_launcher_replays_the_snapshot_environment(launch_harness) -> None:
    launch_harness.run()
    logged = launch_harness.log_file.read_text()
    assert "SECRET: service-role-value" in logged
    # Values with runs of whitespace must survive the snapshot round trip.
    assert "SPACED: one two  three" in logged


def test_launcher_drops_restart_only_variables(launch_harness) -> None:
    launch_harness.run()
    logged = launch_harness.log_file.read_text()
    for name in RESTART_ONLY_VARIABLES:
        assert f"{name}: unset" in logged, name


def test_launcher_appends_across_restarts(launch_harness) -> None:
    launch_harness.run()
    launch_harness.run()
    assert launch_harness.log_file.read_text().count("ARGV: -u -m gateway.main") == 2


def test_launcher_fails_loudly_without_a_snapshot(launch_harness, tmp_path) -> None:
    result = launch_harness.run(env_path=tmp_path / "absent.sh")
    assert result.returncode != 0
    assert "no launch environment" in result.stderr


def test_unit_template_never_gives_up_restarting() -> None:
    unit = UNIT_TEMPLATE.read_text()
    assert "Restart=always" in unit
    # A start-rate limit would let systemd stop retrying, which is the exact
    # failure this unit exists to remove.
    assert "StartLimitIntervalSec=0" in unit
    assert re.search(r"^RestartSec=\d+$", unit, re.MULTILINE)
    assert "WantedBy=multi-user.target" in unit
    assert "ExecStart=@LEADPOET_REPO_ROOT@/scripts/gateway_supervised_launch.sh" in unit


def test_unit_template_placeholders_are_all_substituted() -> None:
    """Every @PLACEHOLDER@ in the template must be one gw_restart.sh renders."""

    placeholders = set(re.findall(r"@[A-Z0-9_]+@", UNIT_TEMPLATE.read_text()))
    rendered = set(
        re.findall(r'-e "s#(@[A-Z0-9_]+@)#', read_restart_script())
    )
    assert placeholders <= rendered, placeholders - rendered


def test_restart_script_stops_the_unit_before_the_destructive_pkill() -> None:
    script = read_restart_script()
    stop_index = script.index("stop_supervised_gateway\nsudo systemctl stop leadpoet-tee-egress-forwarder.service")
    pkill_index = script.index('pkill -9 -f "python3 -u -m gateway.main"')
    assert stop_index < pkill_index


def test_restart_script_installs_the_unit_before_shutdown() -> None:
    """A systemd problem must stall at a gate, not after the pkill."""

    script = read_restart_script()
    install_index = script.index("\nensure_gateway_supervisor_unit\n")
    destructive_index = script.index("GATEWAY_DESTRUCTIVE_PHASE_STARTED=1")
    assert install_index < destructive_index


def test_restart_script_keeps_an_unsupervised_fallback() -> None:
    script = read_restart_script()
    assert "systemd process supervision is unavailable on this host" in script
    # The fallback stays inline at the launch site, after the weight-input
    # repair and the Docker lock release. Several existing tests locate "the
    # launch" by the offset of this literal, so it must not move into a
    # function defined earlier in the file.
    launch = script.index('setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main')
    assert script.count('setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main') == 1
    for earlier in (
        'echo "Stopping existing gateway and Research Lab worker processes"',
        "\nrepair_chain_settlements_and_prepare_current_weight_input\n",
        "leadpoet_release_docker_operation_lock_v2",
    ):
        assert script.index(earlier) < launch, earlier
    assert '9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &' in script


def test_restart_script_health_poll_is_unchanged() -> None:
    """Several OnePatch sensors key off this exact poll; keep it verbatim."""

    script = read_restart_script()
    assert (
        'timeout 5 curl -fsS http://localhost:8000/health/v2-authority'
        in script
    )
    assert 'GATEWAY_V2_HEALTH_RETRY_SECONDS:-5' in script


def test_restart_script_is_syntactically_valid() -> None:
    subprocess.run(["bash", "-n", str(RESTART_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
