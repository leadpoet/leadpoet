"""Restart, concurrency, and partial-failure tests for snapshot refresh."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from gateway.research_lab import snapshot_refresh
from gateway.research_lab.config import RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD
from gateway.research_lab.git_tree_models import TreePolicy
from research_lab.eval.snapshot_store import SNAPSHOT_URI_ENV


IMAGE = "123456789.dkr.ecr.test/model@sha256:" + "a" * 64
COMMIT = "b" * 40
CONFIG_HASH = "sha256:" + "c" * 64
MODEL_MANIFEST_HASH = "sha256:" + "f" * 64


def _active(
    image: str = IMAGE,
    commit: str = COMMIT,
    config_hash: str = CONFIG_HASH,
    manifest_hash: str = MODEL_MANIFEST_HASH,
):
    return SimpleNamespace(
        artifact=SimpleNamespace(
            image_digest=image,
            git_commit_sha=commit,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
        )
    )


def _ready(**overrides: Any) -> dict[str, Any]:
    return {
        "ready": True,
        "reason": "ready",
        "manifest_hash": "sha256:" + "d" * 64,
        "snapshot_age_seconds": 60,
        "champion_image_digest": IMAGE,
        "source_commit": COMMIT,
        "model_config_hash": CONFIG_HASH,
        "private_model_manifest_hash": MODEL_MANIFEST_HASH,
        **overrides,
    }


def _configure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(snapshot_refresh.AUTO_REFRESH_ENABLED_ENV, "true")
    monkeypatch.setenv(snapshot_refresh.RECORD_ENABLED_ENV, "true")
    monkeypatch.setenv(snapshot_refresh.KMS_KEY_ID_ENV, "alias/dev-snapshot")
    monkeypatch.setenv(
        snapshot_refresh.PROVIDER_MODEL_IDS_ENV,
        '["provider/model-a","provider/model-b"]',
    )
    monkeypatch.setenv(snapshot_refresh.RUNTIME_SOURCE_ROOT_ENV, str(Path.cwd()))
    monkeypatch.setenv(
        snapshot_refresh.REFRESH_STATE_PATH_ENV,
        str(tmp_path / "state.json"),
    )
    monkeypatch.setenv(
        snapshot_refresh.REFRESH_WORK_ROOT_ENV,
        str(tmp_path / "work"),
    )
    monkeypatch.setenv(SNAPSHOT_URI_ENV, "s3://private-bucket/dev/current.json")


def test_only_worker_zero_in_active_tree_mode_can_refresh(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    config = SimpleNamespace()
    policy = TreePolicy(mode="active")

    async def _run():
        return await asyncio.gather(
            *(
                snapshot_refresh.maybe_refresh_dev_snapshot(
                    config,
                    worker_index=index,
                    tree_policy=policy,
                    now=1000,
                )
                for index in range(1, 10)
            )
        )

    results = asyncio.run(_run())
    assert {row["reason"] for row in results} == {"not_refresh_worker"}

    disabled = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="off"),
            now=1000,
        )
    )
    assert disabled == {"status": "skipped", "reason": "tree_mode_not_active"}


def test_healthy_snapshot_check_is_persisted_across_restart(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    calls: list[Sequence[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command, _env, _timeout):
        calls.append(command)
        return ""

    first = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
    )
    second = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1100,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
    )
    assert first["status"] == "healthy"
    assert second == {"status": "skipped", "reason": "check_not_due"}
    assert not calls
    assert (tmp_path / "state.json").is_file()


def test_due_refresh_publishes_immutable_target_before_pointer(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    command_envs: list[dict[str, str]] = []
    readiness = iter(
        [
            _ready(ready=False, reason="snapshot_not_ready"),
            _ready(manifest_hash="sha256:" + "e" * 64),
        ]
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        command_envs.append(dict(_env))
        return "ok"

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )
    assert result["status"] == "refreshed"
    assert len(commands) == 4
    assert commands[0][1].endswith("export_research_lab_dev_icp_inputs.py")
    assert commands[1][1].endswith("record_research_lab_dev_snapshots.py")
    assert "--size" not in commands[1]
    assert all(
        env[RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["live_max_icps_per_node"]]
        == "5"
        for env in command_envs
    )
    assert commands[2][1].endswith("publish_research_lab_dev_snapshot.py")
    assert "--skip-current-pointer" in commands[2]
    assert "--skip-current-pointer" not in commands[3]
    assert not any((tmp_path / "work").glob("refresh-*"))


def test_active_model_change_keeps_existing_pointer_untouched(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    active_rows = iter([_active(), _active(image=IMAGE[:-1] + "f")])

    async def active_loader(*_args: Any, **_kwargs: Any):
        return next(active_rows)

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        return "ok"

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="active_model_mismatch",
            ),
            active_loader=active_loader,
        )
    )
    assert result["status"] == "failed"
    assert "active private model changed" in result["last_error"]
    assert len(commands) == 3
    assert "--skip-current-pointer" in commands[-1]
    assert not any(
        command[1].endswith("publish_research_lab_dev_snapshot.py")
        and "--skip-current-pointer" not in command
        for command in commands
    )


def test_recording_failure_is_visible_and_never_promotes_pointer(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        if command[1].endswith("record_research_lab_dev_snapshots.py"):
            raise RuntimeError("recording failed")
        return "ok"

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(ready=False, reason="not_ready"),
            active_loader=active_loader,
        )
    )
    assert result["status"] == "failed"
    assert "recording failed" in result["last_error"]
    assert len(commands) == 2
    assert not any(command[1].endswith("publish_research_lab_dev_snapshot.py") for command in commands)


def test_cross_process_lock_allows_only_one_simultaneous_check(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)

    async def _run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def active_loader(*_args: Any, **_kwargs: Any):
            entered.set()
            await release.wait()
            return _active()

        first = asyncio.create_task(
            snapshot_refresh.maybe_refresh_dev_snapshot(
                SimpleNamespace(),
                worker_index=0,
                tree_policy=TreePolicy(mode="active"),
                now=1000,
                readiness_loader=lambda _uri, **_kwargs: _ready(),
                active_loader=active_loader,
            )
        )
        await entered.wait()
        second = await snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
        release.set()
        return await first, second

    first, second = asyncio.run(_run())
    assert first["status"] == "healthy"
    assert second == {"status": "skipped", "reason": "refresh_lock_held"}
