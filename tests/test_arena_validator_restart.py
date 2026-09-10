from pathlib import Path
import subprocess
import sys
import json
import os
import types


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


def test_restart_uses_single_enclave_handoff_and_no_manual_weight_call():
    text = SCRIPT.read_text()
    assert 'CANDIDATE_CID' in text
    assert 'terminate-enclave --enclave-id "$OLD_ENCLAVE_ID"' in text
    assert "terminate-enclave --all" not in text
    assert "set_weights" not in text
    assert "author_submitExtrinsic" not in text
    assert "pkill" not in text


def test_restart_requires_persistent_state_and_supervised_readiness():
    text = SCRIPT.read_text()
    assert "Arena validator state directory must already exist" in text
    assert 'systemctl start "$SERVICE"' in text
    assert 'systemctl show -p MainPID' in text
    assert "ss --vsock" in text
    assert "Arena runsc must be an executable regular file" in text
    assert 'install -m 0600 -o root -g root "$ENV_FILE" "$CANDIDATE_SERVICE_ENV"' in text
    assert 'mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"' in text
    unit = (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
    assert "User=root" in unit
    assert "arena-validator-service.env" in unit
    assert "PYTHONDONTWRITEBYTECODE=1" in unit


def _run_durable_directory_setup(tmp_path, state_path, runner_path):
    text = SCRIPT.read_text()
    start = text.index('for durable in "$STATE_PATH" "$RUNNER_PATH"; do')
    end = text.index("\ndone", start) + len("\ndone")
    setup = text[start:end]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text(
        """#!/bin/bash
set -euo pipefail
chmod 0700 "$GUARDED_PARENT"
status=0
if [ "$1" = install ]; then
  shift
  filtered=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -o|-g) shift 2 ;;
      *) filtered+=("$1"); shift ;;
    esac
  done
  install "${filtered[@]}" || status=$?
else
  "$@" || status=$?
fi
chmod 0000 "$GUARDED_PARENT"
exit "$status"
"""
    )
    sudo.chmod(0o755)
    program = f"""set -euo pipefail
fail() {{ echo "ERROR: $*" >&2; exit 1; }}
STATE_PATH={state_path}
RUNNER_PATH={runner_path}
{setup}
"""
    guarded_parent = state_path.parent
    guarded_parent.chmod(0)
    try:
        return subprocess.run(
            ["bash", "-c", program],
            env={
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "GUARDED_PARENT": str(guarded_parent),
            },
            text=True,
            capture_output=True,
        )
    finally:
        guarded_parent.chmod(0o700)


def test_durable_directory_setup_checks_root_private_paths_as_root(tmp_path):
    guarded = tmp_path / "durable"
    guarded.mkdir()
    state = guarded / "state"
    runner = guarded / "runner"

    result = _run_durable_directory_setup(tmp_path, state, runner)

    assert result.returncode == 0, result.stderr
    assert state.is_dir() and runner.is_dir()
    assert state.stat().st_mode & 0o777 == 0o700
    assert runner.stat().st_mode & 0o777 == 0o700


def test_durable_directory_setup_rejects_symlink_before_install(tmp_path):
    guarded = tmp_path / "durable"
    guarded.mkdir()
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    state = guarded / "state"
    state.symlink_to(target, target_is_directory=True)
    runner = guarded / "runner"

    result = _run_durable_directory_setup(tmp_path, state, runner)

    assert result.returncode != 0
    assert "Arena durable directory is unsafe" in result.stderr
    assert target.stat().st_mode & 0o777 == 0o755
    assert not runner.exists()


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
    # First transition keeps new RPCs off the old clients' CID.
    assert "CANDIDATE_CID=19" in text
    assert 'diff -qr "$STAGE" "$RELEASE"' in text


def test_candidate_host_module_is_loaded_from_release_not_stale_source(tmp_path):
    source = tmp_path / "source"
    release = tmp_path / "release"
    for root, identity in ((source, "stale"), (release, "candidate")):
        host = root / "validator_tee" / "host"
        host.mkdir(parents=True)
        (root / "validator_tee" / "__init__.py").write_text("")
        (host / "__init__.py").write_text("")
        (host / "arena_restart_identity.py").write_text(
            f'print("{identity} 123")\n'
        )
    stale = subprocess.run(
        [sys.executable, "-m", "validator_tee.host.arena_restart_identity"],
        cwd=source,
        env={**os.environ, "PYTHONPATH": str(release)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert stale.stdout.strip() == "stale 123"

    text = SCRIPT.read_text()
    start = text.index("read -r old_pid old_start < <(")
    end = text.index("\nread -r old_runner_pgid", start)
    identity_call = text[start:end]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text('#!/bin/sh\nexec "$@"\n')
    sudo.chmod(0o755)
    program = f"""set -euo pipefail
fail() {{ echo "ERROR: $*" >&2; exit 1; }}
cd "$SOURCE_ROOT"
{identity_call}
printf '%s %s' "$old_pid" "$old_start"
"""
    selected = subprocess.run(
        ["bash", "-c", program],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SOURCE_ROOT": str(source),
            "RELEASE": str(release),
            "CURRENT_LINK": str(release),
            "PYTHON": sys.executable,
            "service_pid": "0",
            "legacy_container_pid": "0",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert selected.stdout == "candidate 123"


def test_single_enclave_transition_has_bounded_automatic_rollback():
    text = SCRIPT.read_text()
    public_preflight = text.index("Arena public preflight is valid")
    stop_weight = text.index('docker stop --time "$STOP_TIMEOUT" "$legacy_container_id"')
    stop_old_signer = text.index('terminate-enclave --enclave-id "$OLD_ENCLAVE_ID"')
    start_candidate = text.index('run-enclave --eif-path "$EIF_FILE"')
    protected_ready = text.index("--check-only", start_candidate)
    stop_runner = text.index('stop_owned_process "$old_runner_pgid"')
    assert public_preflight < stop_weight < stop_old_signer < start_candidate < protected_ready < stop_runner
    cleanup = text[text.index("cleanup() {"):text.index("trap cleanup EXIT")]
    assert 'run-enclave --eif-path "$LEGACY_EIF_SNAPSHOT"' in cleanup
    assert "runtime_v2_bootstrap" in cleanup
    assert "hotkey_bootstrap_v2" in cleanup
    assert 'docker start "$legacy_container_id"' in cleanup
    assert '"$SIGNER_HANDOFF_COMMITTED" -eq 0' in cleanup
    assert "leadpoet_acquire_docker_operation_lock_v2" in text
    assert 'cp --reflink=auto "$LEGACY_EIF" "$LEGACY_EIF_SNAPSHOT"' in text
    assert "request_timeout_seconds=timeout" in text
    assert 'ENCLAVE_CID="$OLD_ENCLAVE_CID" PYTHONPATH="$SOURCE_ROOT"' in cleanup
    assert "no recoverable legacy signer is running" not in text


def _proc_entry(proc: Path, pid: int, cwd: Path, command: Path, *, ppid=1, pgid=None):
    entry = proc / str(pid)
    entry.mkdir()
    (entry / "cwd").symlink_to(cwd, target_is_directory=True)
    (entry / "cmdline").write_bytes(b"python3\0" + str(command).encode() + b"\0")
    fields = ["0"] * 22
    fields[3] = str(ppid)
    fields[4] = str(pgid if pgid is not None else pid)
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


def test_process_identity_cli_owns_legacy_runner_group_and_relay(tmp_path):
    source = tmp_path / "installed"
    current = tmp_path / "current"
    proc = tmp_path / "proc"
    (source / "scripts").mkdir(parents=True)
    current.mkdir()
    proc.mkdir()
    _proc_entry(proc, 301, source, source / "scripts/run_lab_arena_runner.py", pgid=301)
    _proc_entry(proc, 302, source, source / "scripts/run_lab_arena_runner.py", ppid=301, pgid=301)
    module = "validator_tee.host.arena_restart_identity"
    runner = subprocess.run(
        [sys.executable, "-m", module, str(source), str(current), "0", "--kind", "runner", "--proc-root", str(proc)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert runner.stdout.strip() == "301 1301"
    import shutil
    for child in proc.iterdir(): shutil.rmtree(child)
    relay_command = source / "validator_tee.host.chain_relay_v2"
    _proc_entry(proc, 401, source, relay_command)
    # Auxiliary relay matching uses the exact module argument.
    (proc / "401/cmdline").write_bytes(b"python3\0-m\0validator_tee.host.chain_relay_v2\0")
    relay = subprocess.run(
        [sys.executable, "-m", module, str(source), str(current), "0", "--kind", "relay", "--proc-root", str(proc)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert relay.stdout.strip() == "401 1401"


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
CANDIDATE_CREATED=1
STAGE=
CANDIDATE_SERVICE_ENV=
SERVICE_ENV_PROMOTED=0
SERVICE_ENV_BACKUP=
RUNTIME_ENV_PROMOTED=0
RUNTIME_ENV_BACKUP=
SIGNER_HANDOFF_COMMITTED=0
OLD_ENCLAVE_TERMINATED=0
LEGACY_EIF_SNAPSHOT=
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


def test_reused_candidate_preflight_failure_never_terminates_signer(tmp_path):
    text = SCRIPT.read_text()
    cleanup = text[text.index("cleanup() {"):text.index("trap cleanup EXIT")]
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    log = tmp_path / "calls"
    sudo = fake_bin / "sudo"
    sudo.write_text('#!/bin/sh\necho "$*" >> "$CALL_LOG"\n')
    sudo.chmod(0o755)
    program = """set -euo pipefail
SERVICE_ATTEMPTED=0; ACTIVATED=0; SERVICE_STARTED=0; SERVICE=test.service
CANDIDATE_ENCLAVE_ID=existing-id; CANDIDATE_CREATED=0; STAGE=; CANDIDATE_SERVICE_ENV=
SERVICE_ENV_PROMOTED=0; SERVICE_ENV_BACKUP=; RUNTIME_ENV_PROMOTED=0; RUNTIME_ENV_BACKUP=
SIGNER_HANDOFF_COMMITTED=0; OLD_ENCLAVE_TERMINATED=0; LEGACY_EIF_SNAPSHOT=
""" + cleanup + "\ntrap cleanup EXIT\nfalse\n"
    subprocess.run(["bash", "-c", program], env={"PATH": f"{fake_bin}:/usr/bin:/bin", "CALL_LOG": str(log)})
    assert "terminate-enclave" not in (log.read_text() if log.exists() else "")


def test_created_candidate_failure_reprovisions_captured_old_cid_before_container_start(tmp_path):
    text = SCRIPT.read_text(); cleanup = text[text.index("cleanup() {"):text.index("trap cleanup EXIT")]
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); log = tmp_path / "calls"
    sudo = fake_bin / "sudo"
    sudo.write_text('''#!/bin/sh
echo "sudo:$*" >> "$CALL_LOG"
case "$*" in
  "nitro-cli run-enclave"*) echo '{"EnclaveID":"old"}' ;;
  "nitro-cli describe-enclaves"*) echo '[{"State":"RUNNING","EnclaveCID":81,"Measurements":{"PCR0":"oldpcr"}}]' ;;
esac
'''); sudo.chmod(0o755)
    python = fake_bin / "python"
    python.write_text('''#!/bin/sh
if [ "$1" = "-c" ]; then cat >/dev/null; echo oldpcr; exit 0; fi
echo "bootstrap:$ENCLAVE_CID:$*" >> "$CALL_LOG"
'''); python.chmod(0o755)
    snapshot = tmp_path / "old.eif"; snapshot.write_text("old")
    program = f"""set -euo pipefail
SERVICE_ATTEMPTED=0; ACTIVATED=0; SERVICE_STARTED=0; SERVICE=test.service
CANDIDATE_ENCLAVE_ID=new-id; CANDIDATE_CREATED=1; STAGE=; CANDIDATE_SERVICE_ENV=
SERVICE_ENV_PROMOTED=0; SERVICE_ENV_BACKUP=; RUNTIME_ENV_PROMOTED=0; RUNTIME_ENV_BACKUP=
SIGNER_HANDOFF_COMMITTED=0; OLD_ENCLAVE_TERMINATED=1; LEGACY_EIF_SNAPSHOT={snapshot}
OLD_ENCLAVE_CPUS=2; OLD_ENCLAVE_MEMORY=1024; OLD_ENCLAVE_CID=81; OLD_ENCLAVE_NAME=validator-enclave; OLD_PCR0=oldpcr
PYTHON={python}; SOURCE_ROOT=/source; LEGACY_RELEASE_MANIFEST=/release; LEGACY_GATEWAY_MANIFEST=/gateway
LEGACY_GATEWAY_LINEAGE=/lineage; LEGACY_HOTKEY_CONFIG=/hotkey; LEGACY_ENVELOPE=/envelope
LEGACY_CONTAINER_STOPPED=1; legacy_container_id=container-id
READY_TIMEOUT=2
{cleanup}
trap cleanup EXIT
false
"""
    subprocess.run(["bash", "-c", program], env={"PATH":f"{fake_bin}:/usr/bin:/bin","CALL_LOG":str(log)})
    calls = log.read_text()
    assert "bootstrap:81:-m validator_tee.host.runtime_v2_bootstrap" in calls
    assert "bootstrap:81:-m validator_tee.host.hotkey_bootstrap_v2" in calls
    assert calls.index("bootstrap:81:-m validator_tee.host.hotkey_bootstrap_v2") < calls.index("sudo:docker start container-id")


def test_embedded_public_preflight_supplies_timeout_and_reads_finalized_chain(tmp_path, monkeypatch):
    text = SCRIPT.read_text()
    start = text.index("import json,os,sys", text.index("Arena signer EIF measurement mismatch"))
    end = text.index("\nPY\n)", start)
    code = text[start:end]
    env_file = tmp_path / "arena.env"
    env_file.write_text("LAB_ARENA_CHAIN_ENDPOINT=wss://chain.example\nLAB_ARENA_NETUID=71\nLAB_ARENA_NETWORK=finney\nLAB_ARENA_API_BASE_URL=https://arena.example\n")
    env_file.chmod(0o600)
    policy_file = tmp_path / "policy.json"
    policy = {"network":"finney","netuid":71,"chain_profile":{"chain_endpoint":"wss://chain.example","genesis_hash":"11"*32},"arena_signing_key_hash":"sha256:key"}
    policy_file.write_text(json.dumps(policy)); policy_file.chmod(0o600)
    observed = {}
    class Config:
        def __init__(self, *, endpoint, netuid, network_name, request_timeout_seconds):
            observed["timeout"] = request_timeout_seconds
            self.endpoint, self.netuid, self.network_name = endpoint, netuid, network_name
    class Chain:
        def __init__(self, config, client): self.client = client
        def finalized_head(self): return types.SimpleNamespace(number=123)
        def refresh_metagraph(self): return types.SimpleNamespace(hotkeys=["registered"])
        def close(self): observed["closed"] = True
    client = types.SimpleNamespace(get_block_hash=lambda block_id: "0x" + "11"*32)
    monkeypatch.setitem(sys.modules, "validator_tee.enclave.arena_hotkey", types.SimpleNamespace(validate_policy=lambda value:value))
    monkeypatch.setitem(sys.modules, "lab_arena.chain", types.SimpleNamespace(ArenaChain=Chain,ArenaChainConfig=Config,connect_substrate=lambda config:client))
    monkeypatch.setitem(sys.modules, "lab_arena.validator", types.SimpleNamespace(ArenaPublicApi=lambda url:types.SimpleNamespace(signing_key=lambda:{"public_key_hash":"sha256:key"})))
    monkeypatch.setattr(sys, "argv", ["preflight", str(env_file), str(policy_file)])
    for name in ("LAB_ARENA_CHAIN_ENDPOINT","LAB_ARENA_NETUID","LAB_ARENA_NETWORK","LAB_ARENA_API_BASE_URL","LAB_ARENA_CHAIN_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    exec(compile(code, "embedded-public-preflight", "exec"), {})
    assert observed == {"timeout": 30, "closed": True}


def test_vsock_readiness_parses_live_ss_column_shape():
    fixture = "v_str LISTEN 0      0      *:5002 *:*\nv_str LISTEN 0      0      *:5003 *:*\n"
    result = subprocess.run(
        ["awk", '$5 ~ /:5002$/ {chain=1} $5 ~ /:5003$/ {state=1} END{exit !(chain&&state)}'],
        input=fixture, text=True,
    )
    assert result.returncode == 0


def test_service_uses_the_preflight_interpreter_for_both_start_commands(tmp_path):
    text = SCRIPT.read_text()
    start = text.index('UNIT_STAGE="$(mktemp')
    end = text.index('sudo install -m 0644 "$UNIT_STAGE"', start)
    result = subprocess.run(
        ["bash", "-euc", 'fail() { exit 1; };\n' + text[start:end] + '\nprintf "%s" "$UNIT_STAGE"'],
        env={**os.environ, "RELEASE_ROOT": str(tmp_path), "RELEASE": str(ROOT), "PYTHON": sys.executable},
        text=True, capture_output=True, check=True,
    )
    unit = Path(result.stdout).read_text()
    commands = [line for line in unit.splitlines() if line.startswith(("ExecStart=", "ExecStartPre="))]
    assert len(commands) == 2
    assert all(line.partition("=")[2].startswith(sys.executable + " ") for line in commands)
    assert "--check-only" in next(line for line in commands if line.startswith("ExecStartPre="))
