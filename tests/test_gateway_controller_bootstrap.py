"""Exact-main bootstrap changes controller authority only through canonical startup."""

from __future__ import annotations

import fcntl
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import textwrap

import pytest

from scripts import bootstrap_gateway_restart_controller as bootstrap


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
        [[ "$GATEWAY_RESTART_AUTHORITY_ROOT" == */gateway-restart-controller-bootstrap.*/authority ]]
        [ "$(stat -f '%Lp' "$GATEWAY_RESTART_AUTHORITY_ROOT" 2>/dev/null || stat -c '%a' "$GATEWAY_RESTART_AUTHORITY_ROOT")" = 700 ]
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
        printf 'guard:release\\n' >> canonical-events
    '''))
    controller.chmod(0o755)
    helper = repo / bootstrap.SELF_PATH
    helper.write_bytes((ROOT / bootstrap.SELF_PATH).read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "candidate canonical controller")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "-u", "origin", "main")
    root = tmp_path / "controller"
    root.mkdir(mode=0o700)
    (root / "releases").mkdir(mode=0o700)
    release = root / "releases" / previous
    release.mkdir(mode=0o700)
    for name, mode in bootstrap.FILES.items():
        path = release / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(old[name])
        path.chmod(mode)
    (root / "current").symlink_to(f"releases/{previous}")
    installed = tmp_path / "gw_restart.sh"
    installed.write_bytes(old["gw_restart.sh"])
    installed.chmod(0o700)
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


@pytest.mark.parametrize("stage,code,identity", [("success", 0, "candidate"),
                                                ("early", 23, "previous"),
                                                ("late", 24, "candidate")])
def test_n_minus_one_runs_full_candidate_entrypoint_and_reports_observed_identity(
    host, monkeypatch, capsys, stage, code, identity,
):
    (host["repo"] / "canonical-control").write_text(stage)
    for key in ("GATEWAY_RESTART_PHASE", "GATEWAY_RESTART_INVOCATION_ID", "GATEWAY_PREFLIGHT_TREE",
                "GATEWAY_DESTRUCTIVE_PHASE_STARTED", "GATEWAY_RESTART_AUTHORITY_ROOT",
                "GATEWAY_RESTART_AUTHORITY_COMMIT", "PREPARED_GATEWAY_SHA",
                "AWS_ACCESS_KEY_ID", "LAB_ARENA_OPENROUTER_API_KEY", "GIT_SSH_COMMAND"):
        monkeypatch.setenv(key, "poisoned-fixture-value")
    previous_bytes = host["host_script"].read_bytes()
    assert bootstrap.bootstrap(**host) == code
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["canonical_exit_code"] == code
    assert receipt["installed_controller"] == identity
    assert receipt["runtime_deployment_verified"] is False
    events = (host["repo"] / "canonical-events").read_text().splitlines()
    assert events[:5] == [f"entry:{host['source_commit']}", "gate:release", "gate:epoch",
                          "gate:ownership", "gate:drain"]
    assert ("guard:release" in events) is (stage == "success")
    if stage == "early":
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
