from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

import pytest

from validator_tee.host.docker_live_restore_reconciler_v2 import (
    DockerLiveRestoreReconcilerV2Error,
    reconcile_live_docker_daemon_v2,
)


class _DockerRuntime:
    def __init__(
        self,
        *,
        live_restore: bool = False,
        reload_failure: bool = False,
        change_pid_after_restart: bool = False,
        omit_running_container: bool = False,
    ) -> None:
        self.container_id = hashlib.sha256(b"container").hexdigest()
        self.image_id = "sha256:" + hashlib.sha256(b"image").hexdigest()
        self.live_restore = live_restore
        self.reload_failure = reload_failure
        self.change_pid_after_restart = change_pid_after_restart
        self.omit_running_container = omit_running_container
        self.restart_count = 0
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        values = tuple(command)
        self.commands.append(values)
        stdout = ""
        returncode = 0
        stderr = ""
        if values == ("docker", "ps", "-aq", "--no-trunc"):
            stdout = self.container_id + "\n"
        elif values == ("docker", "ps", "-q", "--no-trunc"):
            stdout = "" if self.omit_running_container else self.container_id + "\n"
        elif values == ("docker", "images", "-aq", "--no-trunc"):
            stdout = self.image_id + "\n"
        elif values[:3] == ("docker", "image", "inspect"):
            stdout = json.dumps([{"Id": self.image_id, "RootFS": {"Layers": [self.image_id]},
                                  "RepoTags": ["retained:pg"], "RepoDigests": []}])
        elif values == ("docker", "volume", "ls", "-q"):
            stdout = ""
        elif values[:3] == ("docker", "container", "inspect"):
            pid = 2000 if self.restart_count and self.change_pid_after_restart else 1000
            stdout = json.dumps(
                [
                    {
                        "GraphDriver": {
                            "Data": {
                                "MergedDir": (
                                    "/var/lib/docker/overlay2/active/merged"
                                )
                            },
                            "Name": "overlay2",
                        },
                        "Id": self.container_id,
                        "Image": self.image_id,
                        "State": {
                            "Pid": pid,
                            "Running": True,
                            "StartedAt": "2026-08-02T00:00:00Z",
                        },
                    }
                ]
            )
        elif values == ("ctr", "-n", "moby", "tasks", "list", "-q"):
            stdout = self.container_id + "\n"
        elif values == ("docker", "version", "--format", "{{.Server.Version}}"):
            stdout = "25.0.13\n"
        elif values == (
            "docker",
            "info",
            "--format",
            "{{json .LiveRestoreEnabled}}",
        ):
            stdout = "true\n" if self.live_restore else "false\n"
        elif values[:3] == ("dockerd", "--validate", "--config-file"):
            document = json.loads(Path(values[3]).read_text(encoding="utf-8"))
            assert document["live-restore"] is True
            stdout = "configuration OK\n"
        elif values == ("systemctl", "reload", "docker.service"):
            if self.reload_failure:
                returncode = 1
                stderr = "reload refused"
            else:
                self.live_restore = True
        elif values == ("systemctl", "restart", "docker.service"):
            self.restart_count += 1
        elif values == ("systemctl", "is-active", "containerd.service"):
            stdout = "active\n"
        else:
            raise AssertionError(f"unexpected command: {values}")
        return subprocess.CompletedProcess(
            list(values), returncode, stdout=stdout, stderr=stderr
        )


def test_reconcile_enables_live_restore_and_preserves_exact_runtime(
    tmp_path: Path,
) -> None:
    runtime = _DockerRuntime()
    config = tmp_path / "daemon.json"
    config.write_text('{"log-driver":"json-file"}\n', encoding="utf-8")
    config.chmod(0o600)

    result = reconcile_live_docker_daemon_v2(
        runner=runtime,
        config_path=config,
        sleeper=lambda _seconds: None,
    )

    assert result.config_changed is True
    assert result.container_count == 1
    assert result.image_count == 1
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "live-restore": True,
        "log-driver": "json-file",
    }
    backups = list(tmp_path.glob("daemon.json.leadpoet-v2-backup-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b'{"log-driver":"json-file"}\n'
    assert ("systemctl", "reload", "docker.service") in runtime.commands
    assert ("systemctl", "restart", "docker.service") in runtime.commands


def test_reconcile_is_idempotent_when_live_restore_is_already_active(
    tmp_path: Path,
) -> None:
    runtime = _DockerRuntime(live_restore=True)
    config = tmp_path / "daemon.json"
    config.write_text('{"live-restore":true}\n', encoding="utf-8")
    config.chmod(0o600)

    result = reconcile_live_docker_daemon_v2(
        runner=runtime,
        config_path=config,
        sleeper=lambda _seconds: None,
    )

    assert result.config_changed is False
    assert ("systemctl", "reload", "docker.service") not in runtime.commands
    assert runtime.restart_count == 1


def test_reconcile_refuses_to_restart_with_nonrunning_container(
    tmp_path: Path,
) -> None:
    runtime = _DockerRuntime(omit_running_container=True)

    with pytest.raises(
        DockerLiveRestoreReconcilerV2Error,
        match="every Docker container is running",
    ):
        reconcile_live_docker_daemon_v2(
            runner=runtime,
            config_path=tmp_path / "daemon.json",
            sleeper=lambda _seconds: None,
        )

    assert runtime.restart_count == 0


def test_reconcile_fails_closed_if_container_pid_changes(tmp_path: Path) -> None:
    runtime = _DockerRuntime(change_pid_after_restart=True)

    with pytest.raises(
        DockerLiveRestoreReconcilerV2Error,
        match="identity changed",
    ):
        reconcile_live_docker_daemon_v2(
            runner=runtime,
            config_path=tmp_path / "daemon.json",
            sleeper=lambda _seconds: None,
        )


def test_reload_failure_restores_original_configuration(tmp_path: Path) -> None:
    runtime = _DockerRuntime(reload_failure=True)
    config = tmp_path / "daemon.json"
    original = b'{"log-level":"warn"}\n'
    config.write_bytes(original)
    config.chmod(0o600)

    with pytest.raises(
        DockerLiveRestoreReconcilerV2Error,
        match="configuration reload failed",
    ):
        reconcile_live_docker_daemon_v2(
            runner=runtime,
            config_path=config,
            sleeper=lambda _seconds: None,
        )

    assert config.read_bytes() == original
    assert runtime.restart_count == 0


def test_reconcile_rejects_unsafe_or_malformed_configuration(
    tmp_path: Path,
) -> None:
    runtime = _DockerRuntime()
    config = tmp_path / "daemon.json"
    config.write_text("[]\n", encoding="utf-8")
    config.chmod(0o600)

    with pytest.raises(
        DockerLiveRestoreReconcilerV2Error,
        match="must be a JSON object",
    ):
        reconcile_live_docker_daemon_v2(
            runner=runtime,
            config_path=config,
            sleeper=lambda _seconds: None,
        )

    config.unlink()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    config.symlink_to(target)
    with pytest.raises(
        DockerLiveRestoreReconcilerV2Error,
        match="not a regular file",
    ):
        reconcile_live_docker_daemon_v2(
            runner=runtime,
            config_path=config,
            sleeper=lambda _seconds: None,
        )


@pytest.fixture
def guarded_reclaim(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import validator_tee.host.docker_live_restore_reconciler_v2 as module

    root = tmp_path.resolve() / "docker"
    root.mkdir(mode=0o700)
    config = tmp_path / "daemon.json"
    config.write_text('{"live-restore":true}\n')
    config.chmod(0o600)
    runtime = _DockerRuntime(live_restore=True)
    inputs = {"docker_service": "docker-before", "binary": ("dockerd",),
              "containerd_service": "containerd-before", "host_gateway": (30, 200)}
    state = SimpleNamespace(clean=False, mutates=True, failure=None, cache=(0, 0), locks=True,
                            foreign=False, inputs=inputs, calls=0, after_restart_stale=False)
    audit = {"active_manifest_hash": "active", "active_container_count": 1,
             "active_image_count": 1, "active_layer_count": 2, "active_mount_count": 1,
             "active_overlay_dir_count": 4, "docker_root": str(root), "metadata_layout": "layerdb"}
    def inventory(**kwargs):
        dirty = not state.clean or (runtime.restart_count and state.after_restart_stale)
        return {**audit, "mounted_overlay_count": 1,
                **{key: int(dirty) for key in ("stale_layer_record_count", "stale_mount_record_count",
                                               "stale_overlay_dir_count", "stale_overlay_link_count")}}
    def locked(**kwargs):
        assert kwargs["owner_pid"] == 50
        assert kwargs["lock_file"] == root / "operation.lock"
        assert kwargs["admission_lock_file"] == root / "admission.lock"
        if not state.locks:
            raise module.DockerZeroRuntimeReconcilerV2Error("lost locks")
    def raw(**kwargs):
        state.calls += 1
        kwargs["pre_mutation_check"]()
        if state.mutates:
            kwargs["mutation_started"]()
            assert (root / module._PENDING_RECONCILE).is_file()
            state.clean = True
        if state.failure:
            state.failure()
        return SimpleNamespace(reclaimed_layer_record_count=int(state.mutates), reclaimed_mount_count=0,
            reclaimed_mount_record_count=0, reclaimed_overlay_dir_count=0, reclaimed_overlay_link_count=0,
            as_dict=lambda: {"status": "ready"})
    def run(command):
        if tuple(command) == ("docker", "builder", "prune", "--all", "--force"):
            runtime.commands.append(tuple(command))
            return subprocess.CompletedProcess(command, 0, "", "")
        return runtime(command)
    monkeypatch.setattr(module, "verify_docker_operation_locks_v2", locked)
    monkeypatch.setattr(module, "find_foreign_docker_operations", lambda **kw: ["foreign"] if state.foreign else [])
    monkeypatch.setattr(module, "_restart_inputs", lambda *args: dict(state.inputs))
    monkeypatch.setattr(module, "_build_cache_counts", lambda *args: state.cache)
    monkeypatch.setattr(module, "audit_stale_docker_overlay_state_v2", inventory)
    monkeypatch.setattr(module, "reclaim_stale_docker_overlay_mounts_v2", raw)
    def execute():
        return module.reclaim_live_docker_overlay_state_v2(lock_file=root / "operation.lock",
            admission_lock_file=root / "admission.lock", lock_owner_pid=50, runner=run,
            config_path=config, expected_root=root, sleeper=lambda seconds: None)
    return module, state, runtime, root, execute


def test_guarded_reclaim_refreshes_once_and_clears_durable_flag(guarded_reclaim):
    module, state, runtime, root, execute = guarded_reclaim
    result = execute()
    assert result["daemon_reconciled"] is True
    assert runtime.restart_count == 1
    assert not (root / module._PENDING_RECONCILE).exists()


def test_guarded_reclaim_noop_does_not_restart(guarded_reclaim):
    _, state, runtime, _, execute = guarded_reclaim
    state.mutates = False
    state.clean = True
    assert execute()["daemon_reconciled"] is False
    assert runtime.restart_count == 0


@pytest.mark.parametrize("kind", ["locks", "foreign", "active-cache"])
def test_guarded_reclaim_refuses_before_mutation(guarded_reclaim, kind):
    module, state, runtime, _, execute = guarded_reclaim
    if kind == "locks": state.locks = False
    elif kind == "foreign": state.foreign = True
    else: state.cache = (1, 1)
    with pytest.raises((module.DockerLiveRestoreReconcilerV2Error, module.DockerZeroRuntimeReconcilerV2Error)):
        execute()
    assert state.calls == 0 and runtime.restart_count == 0


def test_guarded_reclaim_prunes_only_idle_builder_cache(guarded_reclaim):
    _, state, runtime, _, execute = guarded_reclaim
    state.cache = (5, 0)
    execute()
    assert ("docker", "builder", "prune", "--all", "--force") in runtime.commands
    assert not any("image" in cmd and "prune" in cmd or "volume" in cmd and "prune" in cmd for cmd in runtime.commands)


@pytest.mark.parametrize("fault", ["first-delete", "last-delete"])
def test_guarded_reclaim_recovers_partial_failure_and_preserves_error(guarded_reclaim, fault):
    module, state, runtime, root, execute = guarded_reclaim
    def fail():
        state.clean = fault == "last-delete"
        raise RuntimeError("original raw failure")
    state.failure = fail
    with pytest.raises(RuntimeError, match="original raw failure"):
        execute()
    assert runtime.restart_count == 1
    assert (root / module._PENDING_RECONCILE).exists() == (fault == "first-delete")


@pytest.mark.parametrize("change", ["locks", "inputs", "container"])
def test_guarded_reclaim_partial_failure_does_not_restart_changed_runtime(guarded_reclaim, change):
    module, state, runtime, root, execute = guarded_reclaim
    def fail():
        if change == "locks": state.locks = False
        elif change == "inputs": state.inputs["host_gateway"] = None
        else: runtime.omit_running_container = True
        raise RuntimeError("original raw failure")
    state.failure = fail
    with pytest.raises(module.DockerLiveRestoreReconcilerV2Error, match="recovery is incomplete") as exc:
        execute()
    assert str(exc.value.__cause__) == "original raw failure"
    assert runtime.restart_count == 0
    assert (root / module._PENDING_RECONCILE).exists()


def test_guarded_reclaim_post_restart_stale_state_keeps_flag(guarded_reclaim):
    module, state, runtime, root, execute = guarded_reclaim
    state.after_restart_stale = True
    with pytest.raises(module.DockerLiveRestoreReconcilerV2Error, match="stale state returned"):
        execute()
    assert runtime.restart_count == 1 and (root / module._PENDING_RECONCILE).exists()


def test_guarded_reclaim_interruption_reentry_refreshes_even_when_disk_is_clean(guarded_reclaim):
    module, state, runtime, root, execute = guarded_reclaim
    # Model SIGKILL: a separate process cannot run Python exception handlers.
    def stopped(**kwargs):
        kwargs["mutation_started"]()
        state.clean = True
        raise SystemExit(137)
    original = module.reclaim_stale_docker_overlay_mounts_v2
    module.reclaim_stale_docker_overlay_mounts_v2 = stopped
    # Prevent catchable-exception recovery to leave the same on-disk condition as SIGKILL.
    original_refresh = module.reconcile_live_docker_daemon_v2
    module.reconcile_live_docker_daemon_v2 = lambda **kw: (_ for _ in ()).throw(SystemExit(137))
    with pytest.raises(module.DockerLiveRestoreReconcilerV2Error): execute()
    assert (root / module._PENDING_RECONCILE).exists()
    module.reclaim_stale_docker_overlay_mounts_v2 = original
    module.reconcile_live_docker_daemon_v2 = original_refresh
    state.mutates = False
    assert execute()["daemon_reconciled"] is True
    assert runtime.restart_count == 1 and not (root / module._PENDING_RECONCILE).exists()


@pytest.mark.parametrize("bad", ["symlink", "mode", "identity"])
def test_pending_flag_fails_closed(guarded_reclaim, bad):
    module, state, runtime, root, execute = guarded_reclaim
    marker = root / module._PENDING_RECONCILE
    if bad == "symlink": marker.symlink_to(root / "outside")
    else:
        marker.write_text('{}')
        marker.chmod(0o644 if bad == "mode" else 0o600)
    with pytest.raises(module.DockerLiveRestoreReconcilerV2Error, match="Pending reconcile"):
        execute()
    assert state.calls == 0 and runtime.restart_count == 0


@pytest.mark.skipif(not hasattr(__import__('os'), 'fork'), reason='POSIX kill rehearsal')
def test_sigkill_after_raw_mutation_requires_refresh_on_reentry(guarded_reclaim):
    import os
    import signal
    module, state, runtime, root, execute = guarded_reclaim
    child = os.fork()
    if child == 0:
        def interrupted(**kwargs):
            kwargs["mutation_started"]()
            os.kill(os.getpid(), signal.SIGKILL)
        module.reclaim_stale_docker_overlay_mounts_v2 = interrupted
        execute()
        os._exit(90)
    _, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
    assert (root / module._PENDING_RECONCILE).is_file()
    assert runtime.restart_count == 0
    state.clean = True
    state.mutates = False
    assert execute()["daemon_reconciled"] is True
    assert runtime.restart_count == 1
    assert not (root / module._PENDING_RECONCILE).exists()


@pytest.fixture
def restart_inputs(tmp_path, monkeypatch):
    import os
    import time
    from types import SimpleNamespace
    import validator_tee.host.docker_live_restore_reconciler_v2 as module
    from validator_tee.host.docker_zero_runtime_reconciler_v2 import _ServiceIdentity
    root = tmp_path.resolve()
    now = time.time_ns()
    started = now - 10_000_000_000
    old = now - 100_000_000_000
    config = root / 'daemon.json'
    unit = root / 'docker.service'
    binary = root / 'dockerd'
    script = root / 'docker-setup-runtimes.sh'
    env = (root / 'docker', root / 'docker-storage')
    generated = root / 'runtimes.env'
    runtime_dir = root / 'docker-runtimes.d'
    runtime_dir.mkdir(mode=0o700)
    for path, content in ((config, '{"live-restore":true}'), (unit, 'unit'), (binary, 'binary'),
                          (script, 'setup'), (env[0], 'OPTIONS='), (env[1], 'DOCKER_STORAGE_OPTIONS='),
                          (generated, 'DOCKER_ADD_RUNTIMES=""\n')):
        path.write_text(content)
        path.chmod(0o600)
        os.utime(path, ns=(old, old))
    proc = root / 'proc'
    (proc / '99').mkdir(parents=True)
    (proc / 'stat').write_text('btime 1000000000\n')
    (proc / '99' / 'exe').symlink_to(binary)
    monkeypatch.setattr(module, '_RUNTIME_DIRECTORY', runtime_dir)
    monkeypatch.setattr(module, '_RUNTIME_SETUP_SCRIPT', script)
    monkeypatch.setattr(module, '_RUNTIME_ENV_FILE', generated)
    monkeypatch.setattr(module, '_DOCKER_ENV_FILES', env)
    monkeypatch.setattr(module, '_service_identity', lambda runner, service: _ServiceIdentity(99 if service == 'docker.service' else 100, 'a' * 32))
    monkeypatch.setattr(module, 'inspect_exact_host_gateway_runtime', lambda **kw: {'gateway_process': None})
    def command(path, argv):
        return f'{{ path={path} ; argv[]={argv} ; ignore_errors=no ; start_time=old ; stop_time=old ; pid=0 ; code=0 ; status=0 }}'
    fields = [
        ('FragmentPath', str(unit)), ('DropInPaths', ''), ('NeedDaemonReload', 'no'),
        ('ExecMainStartTimestampMonotonic', str((started - 1_000_000_000_000_000_000) // 1000)),
        ('EnvironmentFiles', f'{env[0]} (ignore_errors=yes)'),
        ('EnvironmentFiles', f'{env[1]} (ignore_errors=yes)'),
        ('EnvironmentFiles', f'{generated} (ignore_errors=yes)'),
        ('ExecStart', command(binary, f'{binary} -H fd:// --containerd=/run/containerd/containerd.sock $OPTIONS $DOCKER_STORAGE_OPTIONS $DOCKER_ADD_RUNTIMES')),
        ('ExecStartPre', command('/bin/mkdir', '/bin/mkdir -p /run/docker')),
        ('ExecStartPre', command(script, str(script))),
        *[(key, '') for key in ('ExecStartPost', 'ExecStop', 'ExecStopPost', 'ExecCondition', 'PassEnvironment')],
        ('KillMode', 'process'),
    ]
    def runner(args):
        if args[0] == 'systemctl':
            # systemd 252 omits empty command arrays even with --all.
            output = '\n'.join(key + '=' + value for key, value in fields
                               if value or key not in ('ExecStartPost', 'ExecStop', 'ExecStopPost', 'ExecCondition'))
        else:
            assert args[:2] == ['docker', 'info']
            output = 'true'
        return subprocess.CompletedProcess(args, 0, output, '')
    return SimpleNamespace(module=module, fields=fields, config=config, unit=unit, binary=binary,
        script=script, env=env, generated=generated, runtime_dir=runtime_dir, proc=proc,
        now=now, old=old, read=lambda: module._restart_inputs(runner, config, proc))


def test_restart_inputs_bind_all_repeated_environment_paths_and_script(restart_inputs):
    fixture = restart_inputs
    result = fixture.read()
    paths = [row[0] for row in result['files']]
    assert all(str(path) in paths for path in (*fixture.env, fixture.script, fixture.unit, fixture.config))
    assert len(result['prepare']) == 2


@pytest.mark.parametrize('change', ['config', 'script', 'env0', 'env1', 'binary', 'unit'])
def test_restart_inputs_reject_pending_file_changes(restart_inputs, change):
    import os
    fixture = restart_inputs
    path = fixture.env[int(change[-1])] if change.startswith('env') else getattr(fixture, change)
    os.utime(path, ns=(fixture.now, fixture.now))
    with pytest.raises(DockerLiveRestoreReconcilerV2Error, match='pending or unsafe'):
        fixture.read()


@pytest.mark.parametrize('change', ['missing-env', 'extra-env', 'wrapper', 'stop-hook', 'manager-env', 'kill-mode', 'custom-runtime', 'generated-value', 'generated-symlink'])
def test_restart_inputs_reject_unbound_startup_shapes(restart_inputs, change):
    fixture = restart_inputs
    if change == 'missing-env': fixture.fields[:] = [row for row in fixture.fields if not (row[0] == 'EnvironmentFiles' and str(fixture.env[0]) in row[1])]
    elif change == 'extra-env': fixture.fields.append(('EnvironmentFiles', '/unbound (ignore_errors=yes)'))
    elif change == 'custom-runtime': (fixture.runtime_dir / 'runtime').write_text('custom')
    elif change == 'generated-value': fixture.generated.write_text('DOCKER_ADD_RUNTIMES="unbound"\n')
    elif change == 'generated-symlink':
        fixture.generated.unlink()
        fixture.generated.symlink_to(fixture.env[0])
    else:
        key = {'wrapper': 'ExecStart', 'stop-hook': 'ExecStop', 'manager-env': 'PassEnvironment', 'kill-mode': 'KillMode'}[change]
        fixture.fields[:] = [(name, 'unbound' if name == key else value) for name, value in fixture.fields]
    with pytest.raises(DockerLiveRestoreReconcilerV2Error): fixture.read()


def test_regenerated_empty_environment_does_not_change_restart_binding(restart_inputs):
    fixture = restart_inputs
    before = fixture.read()
    fixture.generated.unlink()
    fixture.generated.write_text('DOCKER_ADD_RUNTIMES=""\n')
    fixture.generated.chmod(0o600)
    assert fixture.read() == before


def test_old_file_content_change_is_still_bound(restart_inputs):
    import os
    fixture = restart_inputs
    before = fixture.read()
    fixture.env[0].write_text('OPTIONS=changed')
    os.utime(fixture.env[0], ns=(fixture.old, fixture.old))
    assert fixture.read() != before


def test_pending_flag_allows_new_retained_image_before_retry(guarded_reclaim, monkeypatch):
    module, state, runtime, root, execute = guarded_reclaim
    module._pending_reconcile(root, create=True)
    state.clean = True
    state.mutates = False
    added = 'sha256:' + hashlib.sha256(b'new-retained-image').hexdigest()
    original = runtime.__class__.__call__
    def with_new_image(self, command):
        result = original(self, command)
        if tuple(command) == ('docker', 'images', '-aq', '--no-trunc'):
            result.stdout += added + '\n'
        elif tuple(command[:3]) == ('docker', 'image', 'inspect'):
            rows = json.loads(result.stdout)
            rows.append({'Id': added, 'RootFS': {'Layers': [added]}, 'RepoTags': ['new:retained'], 'RepoDigests': []})
            result.stdout = json.dumps(rows)
        return result
    monkeypatch.setattr(runtime.__class__, '__call__', with_new_image)
    result = execute()
    assert result['daemon_reconciled'] is True and runtime.restart_count == 1
    assert added in module._runtime_snapshot(runtime).image_ids
    assert not (root / module._PENDING_RECONCILE).exists()


@pytest.mark.parametrize('changed', ['rootfs', 'tags', 'digests', 'volume', 'mounts', 'started'])
def test_refresh_detects_all_retained_object_identity_changes(tmp_path, changed):
    runtime = _DockerRuntime(live_restore=True)
    config = tmp_path / 'daemon.json'
    config.write_text('{"live-restore":true}')
    config.chmod(0o600)
    def runner(command):
        if tuple(command) == ('docker', 'volume', 'ls', '-q'):
            return subprocess.CompletedProcess(command, 0, 'retained-volume\n', '')
        if tuple(command[:3]) == ('docker', 'volume', 'inspect'):
            row = {'Name': 'retained-volume', 'Driver': 'local', 'Mountpoint': '/retained',
                   'Options': None, 'Labels': None, 'Scope': 'local', 'CreatedAt': 'old'}
            if runtime.restart_count and changed == 'volume': row['Mountpoint'] = '/changed'
            return subprocess.CompletedProcess(command, 0, json.dumps([row]), '')
        result = runtime(command)
        if runtime.restart_count and tuple(command[:3]) == ('docker', 'image', 'inspect'):
            rows = json.loads(result.stdout)
            if changed == 'rootfs': rows[0]['RootFS']['Layers'] = ['sha256:' + 'b' * 64]
            if changed == 'tags': rows[0]['RepoTags'] = ['changed:tag']
            if changed == 'digests': rows[0]['RepoDigests'] = ['repository@sha256:' + 'a' * 64]
            result.stdout = json.dumps(rows)
        if runtime.restart_count and tuple(command[:3]) == ('docker', 'container', 'inspect'):
            rows = json.loads(result.stdout)
            if changed == 'mounts': rows[0]['Mounts'] = [{'Source': '/changed'}]
            if changed == 'started': rows[0]['State']['StartedAt'] = 'new'
            result.stdout = json.dumps(rows)
        return result
    with pytest.raises(DockerLiveRestoreReconcilerV2Error, match='identity changed'):
        reconcile_live_docker_daemon_v2(runner=runner, config_path=config, sleeper=lambda seconds: None)


def test_live_reclaim_integrates_real_stale_inventory_deletion_and_refresh(tmp_path, monkeypatch):
    from test_docker_stale_mount_reclaimer_v2 import _DockerFixture
    import validator_tee.host.docker_live_restore_reconciler_v2 as module
    fixture = _DockerFixture(tmp_path.resolve() / 'docker')
    fixture.root.chmod(0o700)
    runtime = _DockerRuntime(live_restore=True)
    runtime.container_id = fixture.container_id
    runtime.image_id = fixture.image_id
    config = tmp_path / 'daemon.json'
    config.write_text('{"live-restore":true}')
    config.chmod(0o600)
    monkeypatch.setattr(module, 'verify_docker_operation_locks_v2', lambda **kw: None)
    monkeypatch.setattr(module, 'find_foreign_docker_operations', lambda **kw: [])
    monkeypatch.setattr(module, '_restart_inputs', lambda *args: {'docker_service': 'before', 'binary': ('dockerd',)})
    def runner(command):
        if tuple(command[:3]) == ('docker', 'container', 'inspect'):
            row = fixture.container_doc
            row['State'] = {'Pid': 1000, 'Running': True, 'StartedAt': 'unchanged'}
            row['Mounts'] = []
            return subprocess.CompletedProcess(command, 0, json.dumps([row]), '')
        try:
            return fixture(command)
        except AssertionError:
            return runtime(command)
    arguments = dict(lock_file=tmp_path / 'operation.lock', admission_lock_file=tmp_path / 'admission.lock',
        lock_owner_pid=50, runner=runner, config_path=config, expected_root=fixture.root,
        sleeper=lambda seconds: None)
    result = module.reclaim_live_docker_overlay_state_v2(**arguments)
    assert result['reclaimed_layer_record_count'] == 1
    assert result['reclaimed_overlay_dir_count'] == 4
    assert result['daemon_reconciled'] is True and runtime.restart_count == 1
    assert not (fixture.root / module._PENDING_RECONCILE).exists()
    after = module.audit_stale_docker_overlay_state_v2(runner=runner, expected_root=fixture.root)
    assert after['stale_layer_record_count'] == after['stale_overlay_dir_count'] == 0
    assert after['active_image_count'] == after['active_container_count'] == 1
    assert module.reclaim_live_docker_overlay_state_v2(**arguments)['daemon_reconciled'] is False
    assert runtime.restart_count == 1
