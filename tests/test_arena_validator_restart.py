from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validator_restart.sh"


def test_restart_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_restart_stages_exact_commit_before_stopping_service():
    text = SCRIPT.read_text()
    assert "git archive --format=tar \"$TARGET_SHA\"" in text
    drain = text.index("# Only now drain")
    assert text.index("--check-only") < text.index('systemctl stop "$SERVICE"', drain)
    assert "git stash" not in text
    assert "git checkout" not in text
    assert "git reset" not in text


def test_restart_uses_dual_enclave_handoff_and_no_manual_weight_call():
    text = SCRIPT.read_text()
    assert 'CANDIDATE_CID' in text
    assert 'terminate-enclave --enclave-id "$old_enclave_id"' in text
    assert "terminate-enclave --all" not in text
    assert "set_weights" not in text
    assert "author_submitExtrinsic" not in text
    assert "pkill" not in text


def test_restart_requires_persistent_state_and_supervised_readiness():
    text = SCRIPT.read_text()
    assert "Arena validator state directory must already exist" in text
    assert 'systemctl start "$SERVICE"' in text
    assert 'systemctl show -p MainPID' in text
    assert "sport = :5002 or sport = :5003" in text
    assert "Arena runsc must be an executable regular file" in text
    assert 'install -m 0600 -o root -g root "$ENV_FILE" "$CANDIDATE_SERVICE_ENV"' in text
    assert 'mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"' in text
    unit = (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
    assert "User=root" in unit
    assert "arena-validator-service.env" in unit
    assert "PYTHONDONTWRITEBYTECODE=1" in unit


def test_restart_privilege_boundaries_keep_active_config_until_activation():
    text = SCRIPT.read_text()
    assert "operator Arena environment owner differs from restart identity" in text
    assert "candidate service environment is not root-owned" in text
    assert 'sudo sed -n \'s/^ENCLAVE_CID=//p\' "$RUNTIME_ENV"' in text
    assert 'sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" -m validator_tee.host.arena_restart_identity' in text
    # Legacy migration retains access to the ec2-user-owned v2 ciphertext.
    migration = text[text.index("arena_hotkey_bootstrap migrate-legacy") - 100:text.index("arena_hotkey_bootstrap migrate-legacy")]
    assert "sudo" not in migration
    assert text.index('mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"') > text.index("# Only now drain")
    assert 'mv -f "$SERVICE_ENV_BACKUP" "$SERVICE_ENV"' in text
    assert 'mv -f "$RUNTIME_ENV_BACKUP" "$RUNTIME_ENV"' in text


def test_restart_supports_first_transition_and_a_second_supervised_restart():
    text = SCRIPT.read_text()
    # N-1 adoption is limited to scripts resolved beneath the installed source.
    assert "validator_tee.host.arena_restart_identity" in text
    # A later restart recognizes only systemd's exact MainPID under current.
    identity = (ROOT / "validator_tee/host/arena_restart_identity.py").read_text()
    assert "pid == service_pid" in identity
    assert "current_root in (path, *path.parents)" in identity
    # The new signer uses a free CID and an old release directory is byte checked.
    assert "range(19,32) if cid not in used" in text
    assert 'diff -qr "$STAGE" "$RELEASE"' in text


def _proc_entry(proc: Path, pid: int, cwd: Path, command: Path):
    entry = proc / str(pid)
    entry.mkdir()
    (entry / "cwd").symlink_to(cwd, target_is_directory=True)
    (entry / "cmdline").write_bytes(b"python3\0" + str(command).encode() + b"\0")
    fields = ["0"] * 22
    fields[21] = str(1000 + pid)
    (entry / "stat").write_text(" ".join(fields))


def test_process_identity_cli_handles_n_minus_one_and_second_restart(tmp_path):
    source = tmp_path / "installed"
    release = tmp_path / "release"
    current = tmp_path / "current"
    proc = tmp_path / "proc"
    for root in (source, release, proc):
        (root / "scripts").mkdir(parents=True, exist_ok=True)
    current.symlink_to(release, target_is_directory=True)
    module = "validator_tee.host.arena_restart_identity"

    _proc_entry(proc, 101, source, source / "scripts/run_arena_validator.py")
    first = subprocess.run(
        [sys.executable, "-m", module, str(source), str(current), "0", "--proc-root", str(proc)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert first.stdout.strip() == "101 1101"

    for child in proc.iterdir():
        import shutil
        shutil.rmtree(child)
    _proc_entry(proc, 202, release, release / "scripts/run_arena_validator.py")
    second = subprocess.run(
        [sys.executable, "-m", module, str(source), str(current), "202", "--proc-root", str(proc)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert second.stdout.strip() == "202 1202"


def test_failed_activation_cleanup_stops_service_before_signer_termination(tmp_path):
    text = SCRIPT.read_text()
    cleanup = text[text.index("cleanup() {"):text.index("trap cleanup EXIT")]
    assert cleanup.index('systemctl stop "$SERVICE"') < cleanup.index("terminate-enclave")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "calls"
    sudo = fake_bin / "sudo"
    sudo.write_text('#!/bin/sh\necho "$*" >> "$CALL_LOG"\n')
    sudo.chmod(0o755)
    program = f"""set -euo pipefail
SERVICE_ATTEMPTED=1
ACTIVATED=0
SERVICE_STARTED=1
SERVICE=test.service
CANDIDATE_ENCLAVE_ID=candidate-id
STAGE=
CANDIDATE_SERVICE_ENV=
SERVICE_ENV_PROMOTED=0
SERVICE_ENV_BACKUP=
RUNTIME_ENV_PROMOTED=0
RUNTIME_ENV_BACKUP=
{cleanup}
trap cleanup EXIT
false
"""
    result = subprocess.run(
        ["bash", "-c", program],
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "CALL_LOG": str(log)},
    )
    assert result.returncode != 0
    calls = log.read_text().splitlines()
    assert calls[:3] == [
        "systemctl stop test.service",
        "systemctl reset-failed test.service",
        "nitro-cli terminate-enclave --enclave-id candidate-id",
    ]
