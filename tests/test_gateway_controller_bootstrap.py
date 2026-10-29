"""Exact-main bootstrap changes controller authority only through canonical startup."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import tarfile
import textwrap
from types import SimpleNamespace

import pytest

from scripts import bootstrap_gateway_restart_controller as bootstrap
from scripts import lab_arena_restart_claim_guard as guard


ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, check=True, timeout=10)
    return result.stdout.strip()


@pytest.fixture
def host(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    repo = tmp_path / "repo"
    repo.mkdir()
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Bootstrap test")
    _git(repo, "config", "user.email", "bootstrap-test@example.invalid")
    _git(repo, "remote", "add", "origin", str(remote))
    monkeypatch.setattr(bootstrap, "ORIGINS", {str(remote)})
    old = {}
    for name in bootstrap.FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ((ROOT / name).read_bytes() if name == "scripts/gateway_git_deploy.py"
                   else b"previous controller bytes\n")
        old[name] = content
        path.write_bytes(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "previous controller")
    previous = _git(repo, "rev-parse", "HEAD")
    controller = repo / "gw_restart.sh"
    controller.write_text(textwrap.dedent('''\
        #!/bin/bash
        set -euo pipefail
        GATEWAY_GIT_DEPLOY_PROTOCOL="1"
        [ "$#" = 2 ] && [ "$1" = --commit ]
        [ "$2" = "$GATEWAY_RESTART_AUTHORITY_COMMIT" ]
        [ "${GATEWAY_RESTART_PHASE:-prepare}" = prepare ]
        [ "${GATEWAY_DESTRUCTIVE_PHASE_STARTED:-0}" = 0 ]
        [ -z "${GATEWAY_RESTART_INVOCATION_ID+x}" ]
        [ -z "${GATEWAY_PREFLIGHT_TREE+x}" ]
        [ -z "${PREPARED_GATEWAY_SHA+x}" ]
        [ -z "${AWS_ACCESS_KEY_ID+x}" ]
        [ -z "${LAB_ARENA_OPENROUTER_API_KEY+x}" ]
        [ -z "${GIT_SSH_COMMAND+x}" ]
        if [ -f canonical-paired ]; then
          [ "${GATEWAY_PAIRED_RESTART:-}" = 1 ]
        else
          [ -z "${GATEWAY_PAIRED_RESTART+x}" ]
        fi
        if [ -f canonical-resume-owner ]; then
          [ "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" = "$(cat canonical-resume-owner)" ]
        else
          [ -z "${GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID+x}" ]
        fi
        [[ "$GATEWAY_RESTART_AUTHORITY_ROOT" == */gateway-restart-controller-bootstrap.*/authority ]]
        [ "$(python3 -c 'import os, sys; print(oct(os.stat(sys.argv[1]).st_mode & 0o777))' "$GATEWAY_RESTART_AUTHORITY_ROOT")" = 0o700 ]
        python3 -B - <<'PY'
        import fcntl, os
        # Bootstrap must release its locks before the full controller enters.
        for key in ("GATEWAY_RESTART_RECOVERY_LOCK_FILE", "GATEWAY_RESTART_LOCK_FILE"):
            with open(os.environ[key], "r+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        PY
        printf 'entry:%s\\n' "$2" >> canonical-events
        for gate in release epoch ownership drain; do printf 'gate:%s\\n' "$gate" >> canonical-events; done
        stage="$(cat canonical-control)"
        if [ "$stage" = early ]; then exit 23; fi
        if [ "$stage" = no-install ]; then exit 0; fi
        printf 'gate:readiness\\n' >> canonical-events
        python3 -B - <<'PY'
        import os, pathlib, shutil
        root = pathlib.Path(os.environ["GATEWAY_RESTART_CONTROLLER_ROOT"])
        source = pathlib.Path(os.environ["GATEWAY_RESTART_AUTHORITY_ROOT"])
        sha = os.environ["GATEWAY_RESTART_AUTHORITY_COMMIT"]
        release = root / "releases" / sha
        release.mkdir(mode=0o700)
        for name in ("gw_restart.sh", "scripts/gateway_git_deploy.py",
                     "gateway/tee/host_memory_guard_v2.py", "scripts/manage_owned_process_group.py"):
            target = release / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            target.chmod(0o700 if name == "gw_restart.sh" else 0o600)
        link = root / ".current-test"
        link.symlink_to("releases/" + sha)
        link.replace(root / "current")
        target = pathlib.Path(os.environ["GATEWAY_HOST_RESTART_SCRIPT"])
        temporary = target.with_suffix(".next")
        shutil.copyfile(source / "gw_restart.sh", temporary)
        temporary.chmod(0o700)
        temporary.replace(target)
        PY
        if [ "$stage" = late ]; then exit 24; fi
        if [ "$stage" = unknown ]; then printf 'unexpected third state' > "$GATEWAY_HOST_RESTART_SCRIPT"; exit 0; fi
        printf 'guard:release\\n' >> canonical-events
    '''))
    controller.chmod(0o755)
    helper = repo / bootstrap.SELF_PATH
    helper.write_bytes((ROOT / bootstrap.SELF_PATH).read_bytes())
    (repo / "scripts/lab_arena_restart_claim_guard.py").write_bytes(
        (ROOT / "scripts/lab_arena_restart_claim_guard.py").read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "candidate canonical controller")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "-u", "origin", "main")
    root = tmp_path / "controller"
    source = tmp_path / "installed-source"
    for name, content in old.items():
        path = source / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(content)
    installed = tmp_path / "gw_restart.sh"
    canonical = (ROOT / "gw_restart.sh").read_text()
    installer = canonical[canonical.index("install_successful_restart_script() {"):]
    installer = installer.split("\n}\n", 1)[0] + "\n}\n"
    # Run the actual canonical installer under production's umask. Its private
    # release contains 0775 descendants; the four files retain exact modes.
    shell = "umask 002\n" + installer + "\n"
    # BSD mv lacks -T. This fresh-install fixture has no prior current link.
    shell += 'mv() { if [ "$1" = -Tf ]; then shift; command mv -f "$@"; else command mv "$@"; fi; }\n'
    for key, value in {
        "GATEWAY_RESTART_AUTHORITY_ROOT": source,
        "GATEWAY_RESTART_AUTHORITY_COMMIT": previous,
        "GATEWAY_DEPLOY_SHA": previous,
        "LEADPOET_REPO_ROOT": repo,
        "GATEWAY_RESTART_CONTROLLER_ROOT": root,
        "GATEWAY_RESTART_CONTROLLER_CURRENT": root / "current",
        "GATEWAY_HOST_RESTART_SCRIPT": installed,
    }.items():
        shell += f"{key}={shlex.quote(str(value))}\n"
    subprocess.run(["bash", "-c", shell + "install_successful_restart_script\n"],
                   check=True, capture_output=True, timeout=10)
    release = root / "releases" / previous
    assert stat.S_IMODE(release.stat().st_mode) == 0o700
    for relative in ("gateway", "gateway/tee", "scripts"):
        assert stat.S_IMODE((release / relative).stat().st_mode) == 0o775
    (repo / "canonical-control").write_text("success")
    (repo / "live-runtime-and-guard").write_text("accepted live work")
    arguments = dict(repo=repo, source_commit=candidate, expected_controller_commit=previous,
                     controller_root=root, host_script=installed,
                     lock_file=tmp_path / "restart.lock",
                     recovery_lock_file=tmp_path / "recovery.lock", helper_path=helper,
                     bootstrap_parent=tmp_path)
    yield arguments
    assert (repo / "live-runtime-and-guard").read_text() == "accepted live work"
    assert not list(tmp_path.glob("gateway-restart-controller-bootstrap.*"))


@pytest.mark.parametrize("stage,code,bootstrap_code,identity", [
    ("success", 0, 0, "candidate"), ("early", 23, 23, "previous"),
    ("late", 24, 24, "candidate"), ("no-install", 0, 1, "previous"),
    ("unknown", 0, 1, "unknown"),
])
def test_n_minus_one_runs_full_candidate_entrypoint_and_reports_observed_identity(
    host, monkeypatch, capsys, stage, code, bootstrap_code, identity,
):
    (host["repo"] / "canonical-control").write_text(stage)
    for key in ("GATEWAY_RESTART_PHASE", "GATEWAY_RESTART_INVOCATION_ID", "GATEWAY_PREFLIGHT_TREE",
                "GATEWAY_DESTRUCTIVE_PHASE_STARTED", "GATEWAY_RESTART_AUTHORITY_ROOT",
                "GATEWAY_RESTART_AUTHORITY_COMMIT", "PREPARED_GATEWAY_SHA",
                "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
                "AWS_ACCESS_KEY_ID", "LAB_ARENA_OPENROUTER_API_KEY", "GIT_SSH_COMMAND"):
        monkeypatch.setenv(key, "poisoned-fixture-value")
    previous_bytes = host["host_script"].read_bytes()
    assert bootstrap.bootstrap(**host) == bootstrap_code
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["canonical_exit_code"] == code
    assert receipt["bootstrap_exit_code"] == bootstrap_code
    assert receipt["installed_controller"] == identity
    assert receipt["runtime_deployment_verified"] is False
    events = (host["repo"] / "canonical-events").read_text().splitlines()
    assert events[:5] == [f"entry:{host['source_commit']}", "gate:release", "gate:epoch",
                          "gate:ownership", "gate:drain"]
    assert ("guard:release" in events) is (stage == "success")
    if stage in ("early", "no-install"):
        assert host["host_script"].read_bytes() == previous_bytes
    assert (host["controller_root"] / "releases" / host["expected_controller_commit"]).exists()


@pytest.mark.parametrize("key", ["lock_file", "recovery_lock_file"])
def test_owned_host_lock_refuses_without_reclaim_or_canonical_entry(host, key):
    with host[key].open("w") as lock:
        host[key].chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns"):
            bootstrap.bootstrap(**host)
    assert not (host["repo"] / "canonical-events").exists()


@pytest.mark.parametrize("failure", ["host-bytes", "release-bytes", "host-mode",
                                     "host-symlink", "current-target", "helper-bytes"])
def test_unknown_or_unsafe_installed_or_source_state_refuses(host, failure):
    release = host["controller_root"] / "releases" / host["expected_controller_commit"]
    if failure == "host-bytes":
        host["host_script"].write_text("concurrent host controller")
    elif failure == "release-bytes":
        (release / "scripts/manage_owned_process_group.py").write_text("modified helper")
    elif failure == "host-mode":
        host["host_script"].chmod(0o777)
    elif failure == "host-symlink":
        host["host_script"].unlink()
        host["host_script"].symlink_to(release / "gw_restart.sh")
    elif failure == "current-target":
        (host["controller_root"] / "current").unlink()
        (host["controller_root"] / "current").symlink_to("/unexpected/controller")
    else:
        host["helper_path"].write_text("uncommitted bootstrap")
    with pytest.raises(RuntimeError):
        bootstrap.bootstrap(**host)
    assert not (host["repo"] / "canonical-events").exists()


@pytest.mark.parametrize("change", ["controller-blob", "extra-archive-file", "main-advance", "installed-change"])
def test_changed_candidate_or_authority_refuses_before_canonical_entry(host, monkeypatch, change):
    archive = bootstrap._archive

    def changed(repo, commit, authority):
        archive(repo, commit, authority)
        if change == "controller-blob":
            (authority / "gw_restart.sh").write_text("poisoned archive")
        elif change == "extra-archive-file":
            (authority / "untracked").write_text("poisoned extra")
        elif change == "main-advance":
            _git(repo, "commit", "--allow-empty", "-m", "concurrent main")
            _git(repo, "push", "origin", "main")
        else:
            host["host_script"].write_text("unexpected third state")

    monkeypatch.setattr(bootstrap, "_archive", changed)
    with pytest.raises(RuntimeError):
        bootstrap.bootstrap(**host)
    assert not (host["repo"] / "canonical-events").exists()
    if change == "installed-change":
        assert host["host_script"].read_text() == "unexpected third state"


def test_non_main_source_and_unknown_origin_refuse(host, monkeypatch):
    with pytest.raises(RuntimeError, match="current fetched origin/main"):
        bootstrap.bootstrap(**(host | {"source_commit": host["expected_controller_commit"]}))
    monkeypatch.setattr(bootstrap, "ORIGINS", set())
    with pytest.raises(RuntimeError, match="unknown repository origin"):
        bootstrap.bootstrap(**host)


def _resume(host, monkeypatch):
    invocation = "gateway-1792039691-1674776"
    candidate = host["expected_controller_commit"]
    timing_dir = host["bootstrap_parent"] / "restart-timings"
    timing_dir.mkdir(mode=0o700)
    ledger = timing_dir / f"{invocation}.jsonl"
    started = datetime.fromtimestamp(1792039691, timezone.utc)
    rows = [{"schema_version": "leadpoet.gateway_restart_timing.v1",
             "stage": stage, "status": status, "commit_sha": candidate,
             "observed_at": (started + timedelta(seconds=elapsed)).isoformat(),
             "elapsed_seconds": elapsed}
            for stage, status, elapsed in [("pre_shutdown_checks_complete", "reached", 1070.0),
                                           ("lab_arena_destructive_authorization", "failed", 1079.729)]]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ledger.chmod(0o600)
    environment = host["bootstrap_parent"] / "gateway.env"
    environment.write_text("LAB_ARENA_MODE=live\n")
    environment.chmod(0o600)
    identity, owner = guard._identity(candidate, invocation)
    commitment, owner_commitment = guard._commitments(identity, owner)
    # Exact guard489 response shape/counts, with opaque commitments redacted.
    state = json.loads((ROOT / "tests/fixtures/arena_restart_guard489_pre_shutdown.json").read_text())
    state.update(candidate_commit=candidate, guard_commitment=commitment, owner_commitment=owner_commitment)
    calls = []
    run = subprocess.run

    def owned(command, **kwargs):
        if not any(str(item).endswith("lab_arena_restart_claim_guard.py") for item in command):
            return run(command, **kwargs)
        assert command[2].endswith("lab_arena_restart_claim_guard.py")
        assert command[3] == "owned"
        assert command[-2:] == ["--environment-file", str(environment)]
        calls.append((command, kwargs["env"]))
        try:
            proof = guard._owned(argparse.Namespace(
                candidate=command[command.index("--candidate") + 1],
                invocation=command[command.index("--invocation") + 1],
            ))
            return subprocess.CompletedProcess(command, 0, json.dumps(guard._safe(proof)).encode(), b"")
        except guard.GuardError:
            return subprocess.CompletedProcess(command, 1, b"", b"owned guard refused")

    def request(function, payload):
        assert function == "lab_arena_restart_guard_state_v1" and payload == {}
        return state

    monkeypatch.setattr(guard, "_request", request)
    monkeypatch.setattr(subprocess, "run", owned)
    args = host | {"resume_invocation": invocation, "timing_dir": timing_dir,
                   "environment_file": environment}
    return args, ledger, rows, state, calls


def test_pre_shutdown_resume_validates_existing_owner_without_inheriting_restart_state(host, monkeypatch, capsys):
    args, _ledger, _rows, _state, calls = _resume(host, monkeypatch)
    (host["repo"] / "canonical-resume-owner").write_text(args["resume_invocation"])
    for key in ("GATEWAY_RESTART_INVOCATION_ID", "GATEWAY_RESTART_PHASE", "GATEWAY_DESTRUCTIVE_PHASE_STARTED",
                "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID", "AWS_ACCESS_KEY_ID", "LAB_ARENA_SERVICE_KEY"):
        monkeypatch.setenv(key, "poisoned-fixture-value")
    assert bootstrap.bootstrap(**args) == 0
    assert len(calls) == 1
    assert not any(key.startswith(("GATEWAY_", "LAB_ARENA_", "AWS_")) for key in calls[0][1])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["installed_controller"] == "candidate" and receipt["canonical_exit_code"] == 0


@pytest.mark.parametrize("paired", [False, True])
def test_explicit_paired_bootstrap_preserves_full_canonical_and_drops_ambient_mode(host, monkeypatch, capsys, paired):
    monkeypatch.setenv("GATEWAY_PAIRED_RESTART", "1" if not paired else "invalid")
    if paired:
        (host["repo"] / "canonical-paired").write_text("all-component guard")
    assert bootstrap.bootstrap(**host, paired_restart=paired) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["installed_controller"] == "candidate"
    events = (host["repo"] / "canonical-events").read_text()
    assert all("gate:" + gate in events for gate in ("release", "epoch", "ownership", "drain", "readiness"))


@pytest.mark.parametrize("paired", [1, "1", None])
def test_bootstrap_rejects_nonboolean_paired_mode_before_mutation(host, paired):
    with pytest.raises(bootstrap.BootstrapError, match="explicit boolean"):
        bootstrap.bootstrap(**host, paired_restart=paired)
    assert not (host["repo"] / "canonical-events").exists()


def test_paired_resume_requires_all_scope_without_weakening_owner(host, monkeypatch):
    args, _ledger, _rows, state, _calls = _resume(host, monkeypatch)
    args["paired_restart"] = True
    (host["repo"] / "canonical-paired").write_text("all-component guard")
    (host["repo"] / "canonical-resume-owner").write_text(args["resume_invocation"])
    state["restart_scope"] = "all"
    assert bootstrap.bootstrap(**args) == 0


@pytest.mark.parametrize("change", ["owner", "guard-source", "generation-zero", "generation-string", "generation-bool",
                                    "scope", "phase", "inactive", "drain-schema", "drain-count", "drain-not-preserved"])
def test_pre_shutdown_resume_refuses_wrong_guard_proof(host, monkeypatch, change):
    args, _ledger, _rows, state, _calls = _resume(host, monkeypatch)
    key, value = {
        "owner": ("owner_commitment", "sha256:" + "f" * 64),
        "guard-source": ("candidate_commit", host["source_commit"]),
        "generation-zero": ("guard_generation", 0),
        "generation-string": ("guard_generation", "489"),
        "generation-bool": ("guard_generation", True),
        "scope": ("restart_scope", "all"),
        "phase": ("restart_phase", "gateway_ready"),
        "inactive": ("guard_active", False),
        "drain-schema": ("schema_version", "unknown"),
        "drain-count": ("current_leased_count", "0"),
        "drain-not-preserved": ("preserved", False),
    }[change]
    (state["drain"] if change.startswith("drain-") else state)[key] = value
    with pytest.raises(RuntimeError, match="resume guard"):
        bootstrap.bootstrap(**args)
    assert not (host["repo"] / "canonical-events").exists()


@pytest.mark.parametrize("change", ["missing", "symlink", "world-writable", "shutdown", "ingress", "malformed", "malformed-row",
                                    "source", "mixed-source", "not-failed", "wrong-schema", "bad-invocation"])
def test_pre_shutdown_resume_refuses_unproved_ledger(host, monkeypatch, change):
    args, ledger, rows, _state, calls = _resume(host, monkeypatch)
    if change == "missing":
        ledger.unlink()
    elif change == "symlink":
        target = ledger.with_suffix(".retained")
        ledger.rename(target)
        ledger.symlink_to(target)
    elif change == "world-writable":
        ledger.chmod(0o666)
    elif change == "malformed":
        ledger.write_text("{broken-json\n")
    elif change == "bad-invocation":
        args["resume_invocation"] = "../another-owner"
    else:
        if change in ("shutdown", "ingress"):
            rows[0]["stage"] = "shutdown_started" if change == "shutdown" else "arena_candidate_ingress_ready"
        elif change == "source":
            for row in rows:
                row["commit_sha"] = "f" * 40
        elif change == "mixed-source":
            rows[0]["commit_sha"] = host["source_commit"]
        elif change == "not-failed":
            rows[-1]["status"] = "reached"
        elif change == "malformed-row":
            rows.append([])
        else:
            rows[-1]["schema_version"] = "unknown"
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((OSError, RuntimeError)):
        bootstrap.bootstrap(**args)
    assert not calls
    assert not (host["repo"] / "canonical-events").exists()


def test_private_install_boundary_does_not_relax_other_parent_checks(host):
    release = host["controller_root"] / "releases" / host["expected_controller_commit"]
    path = release / "scripts/manage_owned_process_group.py"
    bootstrap._parents(path, private_boundary=release)
    with pytest.raises(RuntimeError, match="unsafe parent"):
        bootstrap._parents(path)
    with pytest.raises(RuntimeError, match="not a parent"):
        bootstrap._parents(path, private_boundary=host["repo"])


@pytest.mark.parametrize("change", ["world-writable", "sticky-world-writable", "symlink",
                                    "boundary-group-writable", "exposed-parent"])
def test_unsafe_release_directories_still_refuse_before_canonical_entry(host, change):
    release = host["controller_root"] / "releases" / host["expected_controller_commit"]
    nested = release / "scripts"
    if change == "world-writable":
        nested.chmod(0o777)
    elif change == "sticky-world-writable":
        nested.chmod(0o1777)
    elif change == "symlink":
        nested.rename(release / "retained-scripts")
        nested.symlink_to(release / "retained-scripts", target_is_directory=True)
    elif change == "boundary-group-writable":
        release.chmod(0o775)
    else:
        # The controller root's parent is outside the release's private boundary.
        host["controller_root"].parent.chmod(0o775)
    with pytest.raises(RuntimeError):
        bootstrap.bootstrap(**host)
    assert not (host["repo"] / "canonical-events").exists()


@pytest.mark.parametrize("owner", ["foreign-user", "foreign-group", "root-user"])
def test_private_boundary_does_not_accept_writable_directories_from_other_owners(
    host, monkeypatch, owner,
):
    if owner == "root-user" and os.getuid() == 0:
        pytest.skip("root is the current test owner")
    release = host["controller_root"] / "releases" / host["expected_controller_commit"]
    nested = release / "scripts"
    lstat = Path.lstat

    def changed(path, *args, **kwargs):
        metadata = lstat(path, *args, **kwargs)
        if path == nested:
            return SimpleNamespace(st_mode=metadata.st_mode,
                                   st_uid=(0 if owner == "root-user" else
                                           metadata.st_uid + 10000 if owner == "foreign-user" else metadata.st_uid),
                                   st_gid=metadata.st_gid + (1 if owner == "foreign-group" else 0))
        return metadata

    monkeypatch.setattr(Path, "lstat", changed)
    with pytest.raises(RuntimeError, match="unsafe parent"):
        bootstrap.bootstrap(**host)
    assert not (host["repo"] / "canonical-events").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_unsafe_archive_member_refuses_without_writing_outside_authority(host, monkeypatch, kind):
    run = subprocess.run

    def poisoned(command, **kwargs):
        if "archive" not in command:
            return run(command, **kwargs)
        with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
            member = tarfile.TarInfo("../escaped" if kind == "traversal" else "linked")
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = str(host["host_script"])
                archive.addfile(member)
            else:
                member.size = 3
                archive.addfile(member, io.BytesIO(b"bad"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", poisoned)
    with pytest.raises(RuntimeError, match="archive has"):
        bootstrap.bootstrap(**host)
    assert not (host["repo"] / "canonical-events").exists()
    assert not (host["bootstrap_parent"] / "escaped").exists()


def test_interrupted_observer_retains_authority_for_running_canonical_child(host, monkeypatch, capsys):
    popen = subprocess.Popen

    class RunningChild:
        pid = 987654321

        def wait(self):
            raise KeyboardInterrupt

        def poll(self):
            return None

    def launch(command, **kwargs):
        if command[0] == "/bin/bash":
            return RunningChild()
        return popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt):
        bootstrap.bootstrap(**host)
    receipt = json.loads(capsys.readouterr().out)
    authority = Path(receipt["authority_retained"])
    assert authority.is_dir()
    assert (authority / "gw_restart.sh").read_bytes() == (host["repo"] / "gw_restart.sh").read_bytes()
    assert receipt["canonical_exit_code"] is None
    assert receipt["canonical_pid"] == 987654321
    assert not (host["repo"] / "canonical-events").exists()
    # The child is a test double. Remove only this test's retained authority.
    shutil.rmtree(authority.parent)
