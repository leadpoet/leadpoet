from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from Leadpoet.utils.restart_release_supersession_v2 import (
    RestartReleaseSupersessionV2Error,
    resolve_forward_release_head,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / "value.txt").write_text(name + "\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-m", name, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _repositories(tmp_path: Path) -> tuple[Path, Path, str]:
    origin = tmp_path / "origin.git"
    publisher = tmp_path / "publisher"
    checkout = tmp_path / "checkout"
    _git("init", "--bare", "--initial-branch=main", str(origin))
    _git("init", "-b", "main", str(publisher))
    for repo in (publisher,):
        _git("config", "user.email", "restart@example.invalid", cwd=repo)
        _git("config", "user.name", "Restart Test", cwd=repo)
    first = _commit(publisher, "first")
    _git("remote", "add", "origin", str(origin), cwd=publisher)
    _git("push", "-u", "origin", "main", cwd=publisher)
    _git("clone", str(origin), str(checkout))
    return publisher, checkout, first


def test_forward_authority_returns_current_commit_without_moving_checkout(
    tmp_path: Path,
) -> None:
    _, checkout, first = _repositories(tmp_path)

    resolved = resolve_forward_release_head(
        repo_root=checkout,
        expected_commit=first,
    )

    assert resolved == first
    assert _git("rev-parse", "HEAD", cwd=checkout) == first


def test_forward_authority_fetches_fast_forward_without_activating_it(
    tmp_path: Path,
) -> None:
    publisher, checkout, first = _repositories(tmp_path)
    second = _commit(publisher, "second")
    _git("push", "origin", "main", cwd=publisher)

    resolved = resolve_forward_release_head(
        repo_root=checkout,
        expected_commit=first,
    )

    assert resolved == second
    assert _git("rev-parse", "origin/main", cwd=checkout) == second
    assert _git("rev-parse", "HEAD", cwd=checkout) == first


def test_forward_authority_rejects_non_fast_forward_remote(tmp_path: Path) -> None:
    publisher, checkout, first = _repositories(tmp_path)
    _git("checkout", "--orphan", "replacement", cwd=publisher)
    _git("rm", "-rf", ".", cwd=publisher)
    replacement = _commit(publisher, "replacement")
    _git("push", "--force", "origin", f"{replacement}:main", cwd=publisher)

    with pytest.raises(
        RestartReleaseSupersessionV2Error,
        match="non-fast-forward",
    ):
        resolve_forward_release_head(
            repo_root=checkout,
            expected_commit=first,
        )


def test_forward_authority_rejects_invalid_expected_commit(tmp_path: Path) -> None:
    _, checkout, _ = _repositories(tmp_path)

    with pytest.raises(
        RestartReleaseSupersessionV2Error,
        match="lowercase full SHA",
    ):
        resolve_forward_release_head(
            repo_root=checkout,
            expected_commit="abc123",
        )


def _restart_repositories(tmp_path: Path) -> tuple[Path, Path, str, str]:
    origin = tmp_path / "restart-origin.git"
    publisher = tmp_path / "restart-publisher"
    checkout = tmp_path / "restart-checkout"
    _git("init", "--bare", "--initial-branch=main", str(origin))
    _git("init", "-b", "main", str(publisher))
    _git("config", "user.email", "restart@example.invalid", cwd=publisher)
    _git("config", "user.name", "Restart Test", cwd=publisher)

    helper = publisher / "Leadpoet/utils/restart_release_supersession_v2.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        (ROOT / "Leadpoet/utils/restart_release_supersession_v2.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    for relative in (
        "scripts/gateway_git_deploy.py",
        "Leadpoet/utils/exact_commit_restart_v2.py",
        "gateway/tee/host_memory_guard_v2.py",
    ):
        target = publisher / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# restart fixture\n", encoding="utf-8")
    (publisher / "gw_restart.sh").write_text(
        '#!/bin/bash\nGATEWAY_GIT_DEPLOY_PROTOCOL="1"\necho OLD_GATEWAY\n',
        encoding="utf-8",
    )
    (publisher / "validator_restart.sh").write_text(
        "#!/bin/bash\necho OLD_VALIDATOR\n",
        encoding="utf-8",
    )
    first = _commit(publisher, "first")
    _git("remote", "add", "origin", str(origin), cwd=publisher)
    _git("push", "-u", "origin", "main", cwd=publisher)
    _git("clone", str(origin), str(checkout))

    (publisher / "gw_restart.sh").write_text(
        """#!/bin/bash
set -euo pipefail
GATEWAY_GIT_DEPLOY_PROTOCOL="1"
test "$GATEWAY_RESTART_PHASE" = prepare
test "$GATEWAY_RESTART_LOCK_HELD" = 1
test "$GATEWAY_RELEASE_SUPERSESSION_COUNT" = 1
test -e /dev/fd/9
echo NEW_GATEWAY_EXECUTED
""",
        encoding="utf-8",
    )
    (publisher / "validator_restart.sh").write_text(
        """#!/bin/bash
set -euo pipefail
test "$VALIDATOR_RELEASE_SUPERSESSION_COUNT" = 1
test "$LEADPOET_USE_CAPTURED_RESTART_START" = 1
echo NEW_VALIDATOR_EXECUTED
""",
        encoding="utf-8",
    )
    second = _commit(publisher, "second")
    _git("push", "origin", "main", cwd=publisher)
    return publisher, checkout, first, second


def _extract_shell_function(path: Path, name: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1]) + "\n"
    raise AssertionError(f"unterminated shell function: {name}")


def test_gateway_release_follow_executes_new_candidate_with_same_lock(
    tmp_path: Path,
) -> None:
    _, checkout, first, _ = _restart_repositories(tmp_path)
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    archive = subprocess.Popen(
        ["git", "-C", str(checkout), "archive", first],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    subprocess.run(["tar", "-xf", "-", "-C", str(preflight)], stdin=archive.stdout, check=True)
    assert archive.wait() == 0

    function = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "follow_superseding_gateway_release",
    )
    lock = tmp_path / "gateway.lock"
    driver = tmp_path / "gateway-driver.sh"
    driver.write_text(
        f"""#!/bin/bash
set -euo pipefail
GATEWAY_PREFLIGHT_TREE={preflight!s}
REQUESTED_GATEWAY_DEPLOY_COMMIT=""
GATEWAY_PYTHON_BIN={Path(__import__('sys').executable)!s}
LEADPOET_REPO_ROOT={checkout!s}
PREPARED_GATEWAY_SHA={first}
GATEWAY_RELEASE_SUPERSESSION_COUNT=0
GATEWAY_RELEASE_SUPERSESSION_MAX=2
GATEWAY_RELEASE_FOLLOW_ROOT=""
GATEWAY_RESTART_LOCK_FILE={lock!s}
GATEWAY_RESTART_RECOVERY_LOCK_FILE={tmp_path / 'recovery.lock'!s}
GATEWAY_RESTART_STARTED_EPOCH=1
GATEWAY_RESTART_TIMING_DIR={tmp_path!s}
GATEWAY_RESTART_TIMING_FILE={tmp_path / 'gateway.jsonl'!s}
GATEWAY_RESTART_TIMING_INITIALIZED=1
GATEWAY_DEPLOY_PLAN_FILE={tmp_path / 'plan.json'!s}
GATEWAY_DEPLOYMENT_DIR={tmp_path / 'deployments'!s}
GATEWAY_DEPLOYMENT_MANIFEST={tmp_path / 'current.json'!s}
GATEWAY_LAST_GOOD_MANIFEST={tmp_path / 'last-good.json'!s}
GATEWAY_HOST_RESTART_SCRIPT={tmp_path / 'host-gateway.sh'!s}
GATEWAY_ROOT={tmp_path / 'gateway'!s}
GATEWAY_LOG_ROOT={tmp_path / 'logs'!s}
GATEWAY_LOG_FILE={tmp_path / 'logs/gateway.log'!s}
GATEWAY_ENV_FILE={tmp_path / 'gateway.env'!s}
LEADPOET_GATEWAY_ENV_SECRET_ID=test
GATEWAY_RESTART_CONTROLLER_ROOT={tmp_path / 'controller'!s}
GATEWAY_RESTART_AUTHORITY_ROOT=""
GATEWAY_RESTART_AUTHORITY_COMMIT={first}
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-fixture
GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED=0
GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT=standalone
GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_FILE=""
GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_NONCE=""
GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_TIMEOUT_SECONDS=9300
GATEWAY_TEE_EIF_ROOT={tmp_path / 'tee'!s}
GATEWAY_V2_RELEASE_ARCHIVE_ROOT={tmp_path / 'tee/releases'!s}
GATEWAY_V2_CONFIG_DIR={tmp_path / 'v2'!s}
GATEWAY_V2_RELEASE_MANIFEST={tmp_path / 'release.json'!s}
GATEWAY_V2_VALIDATOR_RELEASE_MANIFEST={tmp_path / 'validator-release.json'!s}
GATEWAY_V2_RELEASE_LINEAGE={tmp_path / 'lineage.json'!s}
GATEWAY_V2_RELEASE_REQUIREMENTS={tmp_path / 'requirements.json'!s}
GATEWAY_PREPARED_V2_RELEASE_MANIFEST={tmp_path / 'prepared-release.json'!s}
GATEWAY_PREPARED_V2_VALIDATOR_RELEASE_MANIFEST={tmp_path / 'prepared-validator-release.json'!s}
GATEWAY_PREPARED_V2_RELEASE_LINEAGE={tmp_path / 'prepared-lineage.json'!s}
GATEWAY_PREPARED_V2_RELEASE_REQUIREMENTS={tmp_path / 'prepared-requirements.json'!s}
GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS={tmp_path / 'validator-requirements.json'!s}
GATEWAY_COUNTERPART_RELEASE_LINEAGE=""
GATEWAY_V2_RELEASE_BUCKET=test
GATEWAY_V2_RELEASE_PREFIX=test
GATEWAY_V2_ARTIFACT_POLICY={tmp_path / 'policy.json'!s}
GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST={tmp_path / 'corpus.json'!s}
GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT={tmp_path / 'corpus'!s}
RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET=test
GATEWAY_V2_OFFLINE_ARTIFACT_ROOT={tmp_path / 'offline'!s}
VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT={tmp_path / 'validator-offline'!s}
GATEWAY_STATEFUL_CUTOVER_CEREMONY=0
GATEWAY_ANCESTRY_CHECKPOINT_RELEASE_SNAPSHOT={tmp_path / 'snapshot.json'!s}
record_gateway_restart_timing() {{ echo TIMING:$1; }}
cancel_gateway_offline_artifact_prepare() {{ echo CANCEL_OFFLINE; }}
cancel_gateway_ancestry_checkpoint_bootstrap() {{ echo CANCEL_ANCESTRY; }}
exec 9>"$GATEWAY_RESTART_LOCK_FILE"
{function}
follow_superseding_gateway_release
echo UNEXPECTED_RETURN
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(driver)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert "CANCEL_OFFLINE" in completed.stdout
    assert "CANCEL_ANCESTRY" in completed.stdout
    assert "NEW_GATEWAY_EXECUTED" in completed.stdout
    assert "UNEXPECTED_RETURN" not in completed.stdout


def test_gateway_candidate_reexec_binds_invocation_to_active_timing_ledger(
    tmp_path: Path,
) -> None:
    derive = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "gateway_restart_invocation_id_from_timing_file",
    )
    bind = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "bind_gateway_restart_invocation_to_timing_file",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    derive,
                    bind,
                    'GATEWAY_RESTART_TIMING_DIR="$1"',
                    "GATEWAY_RESTART_STARTED_EPOCH=1700000000",
                    'GATEWAY_RESTART_TIMING_FILE="$1/gateway-1700000000-$$.jsonl"',
                    "printf '%s\\n' '{\"stage\":\"invoked\",\"status\":\"reached\"}' > \"$GATEWAY_RESTART_TIMING_FILE\"",
                    "GATEWAY_RESTART_TIMING_INITIALIZED=1",
                    "GATEWAY_RESTART_INVOCATION_ID=gateway-stale-parent",
                    "LEADPOET_RESTART_INVOCATION_ID=gateway-stale-parent",
                    "bind_gateway_restart_invocation_to_timing_file",
                    "printf '%s\\n%s\\n%s\\n' \"$GATEWAY_RESTART_INVOCATION_ID\" "
                    '"$LEADPOET_RESTART_INVOCATION_ID" "gateway-1700000000-$$"',
                )
            ),
            "gateway-timing-binding-test",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = completed.stdout.splitlines()
    assert len(lines) == 3
    assert lines[0] == lines[1] == lines[2]


@pytest.mark.parametrize(
    "mode",
    (
        "invalid_name",
        "wrong_epoch",
    ),
)
def test_gateway_candidate_reexec_rejects_invalid_timing_ledger_identity(
    tmp_path: Path,
    mode: str,
) -> None:
    derive = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "gateway_restart_invocation_id_from_timing_file",
    )
    bind = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "bind_gateway_restart_invocation_to_timing_file",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    derive,
                    bind,
                    'GATEWAY_RESTART_TIMING_DIR="$1"',
                    "GATEWAY_RESTART_STARTED_EPOCH=1700000000",
                    'if [ "$2" = invalid_name ]; then',
                    '  GATEWAY_RESTART_TIMING_FILE="$1/gateway.jsonl"',
                    "else",
                    '  GATEWAY_RESTART_TIMING_FILE="$1/gateway-1700000001-$$.jsonl"',
                    "fi",
                    "printf '%s\\n' '{\"stage\":\"invoked\",\"status\":\"reached\"}' > \"$GATEWAY_RESTART_TIMING_FILE\"",
                    "GATEWAY_RESTART_TIMING_INITIALIZED=1",
                    "GATEWAY_RESTART_INVOCATION_ID=gateway-stale-parent",
                    "bind_gateway_restart_invocation_to_timing_file",
                )
            ),
            "gateway-timing-rejection-test",
            str(tmp_path),
            mode,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "ERROR: gateway restart timing ledger" in completed.stderr


def test_gateway_candidate_reexec_rejects_missing_timing_ledger(
    tmp_path: Path,
) -> None:
    derive = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "gateway_restart_invocation_id_from_timing_file",
    )
    bind = _extract_shell_function(
        ROOT / "gw_restart.sh",
        "bind_gateway_restart_invocation_to_timing_file",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    derive,
                    bind,
                    'GATEWAY_RESTART_TIMING_DIR="$1"',
                    "GATEWAY_RESTART_STARTED_EPOCH=1700000000",
                    'GATEWAY_RESTART_TIMING_FILE="$1/gateway-1700000000-$$.jsonl"',
                    "GATEWAY_RESTART_TIMING_INITIALIZED=1",
                    "bind_gateway_restart_invocation_to_timing_file",
                )
            ),
            "gateway-timing-missing-test",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        "ERROR: gateway restart timing ledger is unavailable"
        in completed.stderr
    )


def test_validator_release_follow_executes_new_candidate_before_shutdown(
    tmp_path: Path,
) -> None:
    _, checkout, first, second = _restart_repositories(tmp_path)
    reset_handoff = _extract_shell_function(
        ROOT / "validator_restart.sh",
        "reset_standalone_active_release_handoff_for_reexec",
    )
    function = _extract_shell_function(
        ROOT / "validator_restart.sh",
        "follow_superseding_validator_release",
    )
    driver = tmp_path / "validator-driver.sh"
    driver.write_text(
        f"""#!/bin/bash
set -euo pipefail
REQUESTED_VALIDATOR_DEPLOY_COMMIT=""
REQUESTED_COORDINATED_EXPECTED_COMMIT=""
VALIDATOR_ROOT={checkout!s}
VALIDATOR_PYTHON_BIN={Path(__import__('sys').executable)!s}
VALIDATOR_DEPLOY_SHA={first}
VALIDATOR_RELEASE_SUPERSESSION_COUNT=0
VALIDATOR_RELEASE_SUPERSESSION_MAX=2
VALIDATOR_RESTART_STARTED_EPOCH=1
VALIDATOR_RESTART_TIMING_DIR={tmp_path!s}
VALIDATOR_RESTART_TIMING_FILE={tmp_path / 'validator.jsonl'!s}
VALIDATOR_RESTART_TIMING_INITIALIZED=1
VALIDATOR_ENV_FILE={tmp_path / 'validator.env'!s}
LEADPOET_VALIDATOR_ENV_SECRET_ID=test
VALIDATOR_ENV_BACKUP_DIR={tmp_path / 'env-backups'!s}
VALIDATOR_RESTART_CONTROLLER_ROOT={tmp_path / 'controller'!s}
VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT=""
VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT={second}
VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-fixture
VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED=0
VALIDATOR_HOST_RESTART_SCRIPT={tmp_path / 'host-validator.sh'!s}
VALIDATOR_V2_RELEASE_BUCKET=test
VALIDATOR_V2_RELEASE_PREFIX=test
VALIDATOR_V2_RELEASE_ARCHIVE_ROOT={tmp_path / 'releases'!s}
VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT={tmp_path / 'offline'!s}
VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT={tmp_path / 'initial-requirements.json'!s}
VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT={tmp_path / 'final-requirements.json'!s}
VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT={tmp_path / 'final-lineage.json'!s}
VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS={tmp_path / 'stable-requirements.json'!s}
active_release_handoff_count=0
REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY=0
record_validator_restart_timing() {{ echo TIMING:$1; }}
cleanup_validator_restart_preparation() {{ echo CLEANUP_PREPARATION; }}
cd "$VALIDATOR_ROOT"
{reset_handoff}
{function}
follow_superseding_validator_release
echo UNEXPECTED_RETURN
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(driver)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert "CLEANUP_PREPARATION" in completed.stdout
    assert "NEW_VALIDATOR_EXECUTED" in completed.stdout
    assert "UNEXPECTED_RETURN" not in completed.stdout
    assert _git("rev-parse", "HEAD", cwd=checkout) == second


def test_unpinned_validator_gateway_wait_rechecks_forward_authority(
    tmp_path: Path,
) -> None:
    function = _extract_shell_function(
        ROOT / "validator_restart.sh",
        "verify_forward_gateway_release_before_shutdown",
    )
    driver = tmp_path / "validator-gateway-wait-driver.sh"
    driver.write_text(
        f"""#!/bin/bash
set -euo pipefail
REQUESTED_VALIDATOR_DEPLOY_COMMIT=""
REQUESTED_COORDINATED_EXPECTED_COMMIT=""
TRACE={tmp_path / 'forward-wait.trace'!s}
verify_pinned_gateway_release() {{ echo VERIFY:$1 >>"$TRACE"; return 1; }}
follow_superseding_validator_release() {{ echo FOLLOW >>"$TRACE"; return 0; }}
{function}
if verify_forward_gateway_release_before_shutdown 9; then
  echo UNEXPECTED_SUCCESS
  exit 1
fi
cat "$TRACE"
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(driver)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "VERIFY:4",
        "FOLLOW",
        "VERIFY:4",
        "FOLLOW",
        "VERIFY:1",
    ]


def test_explicit_validator_gateway_wait_remains_pinned(tmp_path: Path) -> None:
    function = _extract_shell_function(
        ROOT / "validator_restart.sh",
        "verify_forward_gateway_release_before_shutdown",
    )
    driver = tmp_path / "validator-pinned-gateway-wait-driver.sh"
    driver.write_text(
        f"""#!/bin/bash
set -euo pipefail
REQUESTED_VALIDATOR_DEPLOY_COMMIT={'1' * 40}
REQUESTED_COORDINATED_EXPECTED_COMMIT=""
TRACE={tmp_path / 'pinned-wait.trace'!s}
verify_pinned_gateway_release() {{ echo VERIFY:$1 >>"$TRACE"; return 0; }}
follow_superseding_validator_release() {{ echo FOLLOW >>"$TRACE"; return 1; }}
{function}
verify_forward_gateway_release_before_shutdown 9
cat "$TRACE"
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(driver)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["VERIFY:9"]
