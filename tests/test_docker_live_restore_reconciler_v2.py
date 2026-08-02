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
