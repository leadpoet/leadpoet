"""PCR0 cache keys cover every exact validator EIF build input."""

from __future__ import annotations

import asyncio
import builtins
from contextlib import asynccontextmanager
import fcntl
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import gateway.utils.pcr0_builder as pcr0_builder
from gateway.utils.pcr0_builder import (
    MONITORED_DIRS,
    MONITORED_FILES,
    PCR0_COPY_PATHS,
    compute_files_content_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_BUILD_INPUTS = (
    "validator_tee/enclave/protected_workflows_v2.json",
    "validator_tee/enclave/weight_authority_v2.py",
    "validator_tee/enclave/chain_source_v2.py",
    "validator_tee/enclave/Cargo.drand-cabi-v2.lock",
    "validator_tee/enclave/libbittensor_drand_v2.sha256",
)


def test_startup_warm_delay_is_fixed_at_fifteen_seconds(tmp_path):
    env = dict(os.environ)
    env["PCR0_STARTUP_WARM_DELAY_SECONDS"] = "999"
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from gateway.utils import pcr0_builder; "
                "print(hasattr(pcr0_builder, "
                "'PCR0_STARTUP_WARM_DELAY_SECONDS')); "
                "print(pcr0_builder.get_cache_status()"
                "['startup_warm_delay_seconds'])"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "15"]


@pytest.fixture()
def repo_copy(tmp_path):
    destination = tmp_path / "repo"
    entries = sorted(
        set(MONITORED_FILES)
        | {path.rstrip("/") for path in MONITORED_DIRS}
        | {path.rstrip("/") for path in PCR0_COPY_PATHS}
    )
    for entry in entries:
        source = REPO_ROOT / entry
        target = destination / entry
        assert source.exists(), entry
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return destination


def test_content_hash_changes_when_any_sensitive_build_input_changes(repo_copy):
    baseline = compute_files_content_hash(str(repo_copy))
    assert baseline is not None

    for entry in SENSITIVE_BUILD_INPUTS:
        target = repo_copy / entry
        original = target.read_bytes()
        target.write_bytes(original + b"\n# pcr0-key-coverage-probe\n")
        try:
            assert compute_files_content_hash(str(repo_copy)) != baseline, entry
        finally:
            target.write_bytes(original)

    assert compute_files_content_hash(str(repo_copy)) == baseline


def test_content_hash_fails_closed_when_required_input_is_missing(repo_copy):
    (repo_copy / "validator_tee/runtime-artifacts-v2.lock.json").unlink()
    assert compute_files_content_hash(str(repo_copy)) is None


def test_content_hash_fails_closed_when_required_input_is_unreadable(
    repo_copy,
    monkeypatch,
):
    denied_path = os.path.abspath(
        repo_copy / "validator_tee/enclave/weight_authority_v2.py"
    )
    real_open = builtins.open

    def deny_one_input(path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == denied_path:
            raise PermissionError("test-denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_one_input)
    assert compute_files_content_hash(str(repo_copy)) is None


def test_base_image_stamp_survives_sparse_checkout_replacement(
    tmp_path,
    monkeypatch,
):
    repo_dir = tmp_path / "pcr0_builder"
    state_dir = tmp_path / "stable-state"
    repo_dir.mkdir()
    monkeypatch.setenv(pcr0_builder.BASE_IMAGE_STAMP_DIR_ENV, str(state_dir))

    pcr0_builder.write_base_image_stamp(
        str(repo_dir),
        "dockerfile-hash",
        "sha256:image-id",
    )
    stamp_path = Path(pcr0_builder.get_base_image_stamp_path(str(repo_dir)))
    assert os.path.commonpath((stamp_path, repo_dir)) != str(repo_dir)
    assert pcr0_builder.read_base_image_stamp(str(repo_dir)) == (
        "dockerfile-hash",
        "sha256:image-id",
    )

    # Reproduce a sparse-clone refresh that replaces the complete checkout.
    shutil.rmtree(repo_dir)
    repo_dir.mkdir()

    assert stamp_path.exists()
    assert pcr0_builder.read_base_image_stamp(str(repo_dir)) == (
        "dockerfile-hash",
        "sha256:image-id",
    )


def test_historical_cache_warming_always_includes_current_deploy_head():
    head = {
        "hash": "f" * 40,
        "timestamp": "3",
        "message": "gateway-only release",
        "date": "",
    }
    measured_history = [
        {
            "hash": "a" * 40,
            "timestamp": "2",
            "message": "validator change",
            "date": "",
        },
        {
            "hash": "b" * 40,
            "timestamp": "1",
            "message": "older validator change",
            "date": "",
        },
    ]

    selected = pcr0_builder._include_current_head_commit(
        measured_history,
        [head],
        2,
    )

    assert [commit["hash"] for commit in selected] == [
        head["hash"],
        measured_history[0]["hash"],
    ]


def test_historical_cache_warming_prioritizes_deployed_runtime_commit():
    deployed = {
        "hash": "d" * 40,
        "timestamp": "3",
        "message": "deployed release changed no measured input",
        "date": "",
    }
    head = {
        "hash": "f" * 40,
        "timestamp": "4",
        "message": "branch head",
        "date": "",
    }
    measured_history = [
        {
            "hash": "a" * 40,
            "timestamp": "2",
            "message": "last measured change",
            "date": "",
        },
        {
            "hash": "b" * 40,
            "timestamp": "1",
            "message": "older measured change",
            "date": "",
        },
    ]

    selected = pcr0_builder._include_current_head_commit(
        measured_history,
        [head],
        3,
        required_commits=[deployed],
    )

    assert [commit["hash"] for commit in selected] == [
        deployed["hash"],
        head["hash"],
        measured_history[0]["hash"],
    ]


def test_historical_cache_warming_includes_recent_unmeasured_release_aliases():
    recent = [
        {
            "hash": character * 40,
            "timestamp": str(10 - index),
            "message": "gateway-only release",
            "date": "",
        }
        for index, character in enumerate(("f", "e", "d"))
    ]
    measured = [
        {
            "hash": "a" * 40,
            "timestamp": "1",
            "message": "validator measured-input change",
            "date": "",
        }
    ]

    selected = pcr0_builder._historical_commit_candidates(
        measured,
        recent,
        unique_version_limit=2,
    )

    assert [commit["hash"] for commit in selected] == [
        recent[0]["hash"],
        recent[1]["hash"],
        recent[2]["hash"],
        measured[0]["hash"],
    ]


def test_cache_pruning_retains_deployed_runtime_commit(monkeypatch):
    deployed = "d" * 40
    monkeypatch.setattr(pcr0_builder, "PCR0_CACHE_SIZE", 2)
    monkeypatch.setattr(pcr0_builder, "_active_runtime_commit", lambda: deployed)
    monkeypatch.setattr(
        pcr0_builder,
        "_pcr0_cache",
        {
            "deployed": {
                "pcr0": "1" * 96,
                "commit_hash": deployed,
                "commit_hashes": [deployed],
                "commit_timestamp": "1",
            },
            "newest": {
                "pcr0": "2" * 96,
                "commit_hash": "e" * 40,
                "commit_timestamp": "3",
            },
            "middle": {
                "pcr0": "3" * 96,
                "commit_hash": "c" * 40,
                "commit_timestamp": "2",
            },
        },
    )

    pcr0_builder._prune_pcr0_cache()

    assert list(pcr0_builder._pcr0_cache) == ["deployed", "newest"]


@pytest.mark.asyncio
async def test_deployed_runtime_commit_is_fetched_when_shallow_history_lacks_it(
    monkeypatch,
):
    deployed = "d" * 40
    metadata = {
        "hash": deployed,
        "timestamp": "3",
        "message": "deployed",
        "date": "",
    }
    get_metadata = AsyncMock(side_effect=[None, metadata])
    monkeypatch.setattr(pcr0_builder, "_get_commit_metadata", get_metadata)

    class FetchProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    create_process = AsyncMock(return_value=FetchProcess())
    monkeypatch.setattr(
        pcr0_builder.asyncio,
        "create_subprocess_exec",
        create_process,
    )

    assert (
        await pcr0_builder._get_required_commit_metadata("/builder", deployed)
        == metadata
    )
    assert create_process.await_args.args[:6] == (
        "git",
        "fetch",
        "--depth",
        "1",
        "origin",
        deployed,
    )


@pytest.mark.asyncio
async def test_unreadable_input_aborts_builder_without_docker_or_cache_relabel(
    repo_copy,
    monkeypatch,
):
    existing_cache = {
        "old-content:old-base": {
            "pcr0": "old-pcr0",
            "commit_hash": "a" * 40,
        }
    }
    monkeypatch.setattr(pcr0_builder, "BUILD_DIR", str(repo_copy))
    monkeypatch.setattr(
        pcr0_builder,
        "clone_or_update_repo",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(pcr0_builder, "compute_files_content_hash", lambda _repo: None)
    prepare_base = AsyncMock(return_value=True)
    build_enclave = AsyncMock(return_value="new-pcr0")
    monkeypatch.setattr(pcr0_builder, "ensure_base_image_exists", prepare_base)
    monkeypatch.setattr(
        pcr0_builder,
        "build_enclave_and_extract_pcr0",
        build_enclave,
    )
    monkeypatch.setattr(pcr0_builder, "_pcr0_cache", existing_cache)
    monkeypatch.setattr(pcr0_builder, "_last_content_hash", "old-content")
    monkeypatch.setattr(pcr0_builder, "_build_in_progress", False)

    await pcr0_builder.check_and_build_pcr0()

    prepare_base.assert_not_awaited()
    build_enclave.assert_not_awaited()
    assert pcr0_builder._pcr0_cache == existing_cache
    assert pcr0_builder._last_content_hash == "old-content"
    assert pcr0_builder._build_in_progress is False


@pytest.mark.asyncio
async def test_identical_measured_inputs_register_current_commit_without_rebuild(
    repo_copy,
    monkeypatch,
):
    original_commit = "a" * 40
    current_commit = "f" * 40
    cache_key = "same-content:same-base"
    existing_cache = {
        cache_key: {
            "pcr0": "9" * 96,
            "content_hash": "same-content",
            "commit_hash": original_commit,
            "commit_hashes": [original_commit],
            "commit_timestamp": "1",
        }
    }
    monkeypatch.setattr(pcr0_builder, "BUILD_DIR", str(repo_copy))
    monkeypatch.setattr(
        pcr0_builder,
        "clone_or_update_repo",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "compute_files_content_hash",
        lambda _repo: "same-content",
    )
    monkeypatch.setattr(
        pcr0_builder,
        "ensure_base_image_exists",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "build_pcr0_cache_key",
        lambda _content_hash, _repo: cache_key,
    )
    monkeypatch.setattr(
        pcr0_builder,
        "get_latest_commits",
        AsyncMock(
            return_value=[
                {
                    "hash": current_commit,
                    "timestamp": "2",
                    "message": "current",
                    "date": "",
                }
            ]
        ),
    )
    build_enclave = AsyncMock(return_value="unexpected")
    monkeypatch.setattr(
        pcr0_builder,
        "build_enclave_and_extract_pcr0",
        build_enclave,
    )
    monkeypatch.setattr(pcr0_builder, "_pcr0_cache", existing_cache)
    monkeypatch.setattr(pcr0_builder, "_last_content_hash", None)
    monkeypatch.setattr(pcr0_builder, "_build_in_progress", False)
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_FILE",
        str(repo_copy.parent / "docker-operation.lock"),
    )

    await pcr0_builder.check_and_build_pcr0()

    build_enclave.assert_not_awaited()
    assert pcr0_builder._pcr0_cache[cache_key]["commit_hashes"] == [
        original_commit,
        current_commit,
    ]
    assert pcr0_builder.verify_pcr0(
        "9" * 96,
        expected_commit=current_commit,
    )["valid"] is True


@pytest.mark.asyncio
async def test_pcr0_builder_waits_for_shared_docker_operation_lock(
    tmp_path,
    monkeypatch,
):
    lock_path = tmp_path / "docker-operation.lock"
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_FILE",
        str(lock_path),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS",
        2,
    )
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_POLL_SECONDS",
        0.01,
    )

    owner_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    entered = asyncio.Event()

    async def contend():
        async with pcr0_builder._docker_operation_lock_scope():
            entered.set()

    task = asyncio.create_task(contend())
    await asyncio.sleep(0.05)
    assert entered.is_set() is False

    fcntl.flock(owner_fd, fcntl.LOCK_UN)
    os.close(owner_fd)
    await asyncio.wait_for(task, timeout=1)
    assert entered.is_set() is True


@pytest.mark.asyncio
async def test_pcr0_builder_releases_waiting_lock_on_cancellation(
    tmp_path,
    monkeypatch,
):
    lock_path = tmp_path / "docker-operation.lock"
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_FILE",
        str(lock_path),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS",
        2,
    )
    monkeypatch.setattr(
        pcr0_builder,
        "DOCKER_OPERATION_LOCK_POLL_SECONDS",
        0.01,
    )

    entered = asyncio.Event()
    never = asyncio.Event()

    async def hold_lock():
        async with pcr0_builder._docker_operation_lock_scope():
            entered.set()
            await never.wait()

    task = asyncio.create_task(hold_lock())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    probe_fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(probe_fd, fcntl.LOCK_UN)
        os.close(probe_fd)


@pytest.mark.asyncio
async def test_sync_build_step_does_not_block_gateway_event_loop():
    started = threading.Event()
    release = threading.Event()

    def blocking_step():
        started.set()
        assert release.wait(timeout=2)
        return "complete"

    task = asyncio.create_task(
        pcr0_builder._run_sync_build_step_to_completion(blocking_step)
    )
    assert await asyncio.to_thread(started.wait, 1)

    loop_progressed = False

    async def mark_progress():
        nonlocal loop_progressed
        await asyncio.sleep(0)
        loop_progressed = True

    await asyncio.wait_for(mark_progress(), timeout=0.2)
    assert loop_progressed is True
    assert task.done() is False

    release.set()
    assert await asyncio.wait_for(task, timeout=1) == "complete"


@pytest.mark.asyncio
async def test_sync_build_step_finishes_before_cancellation_releases_build_lock():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_step():
        started.set()
        assert release.wait(timeout=2)
        finished.set()

    task = asyncio.create_task(
        pcr0_builder._run_sync_build_step_to_completion(blocking_step)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert finished.is_set() is True


@pytest.mark.asyncio
async def test_build_subprocess_finishes_before_cancellation_releases_build_lock():
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class Process:
        async def communicate(self):
            started.set()
            await release.wait()
            finished.set()
            return b"stdout", b"stderr"

    task = asyncio.create_task(
        pcr0_builder._communicate_build_process_to_completion(Process())
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert finished.is_set() is True


@pytest.mark.asyncio
async def test_current_pcr0_build_holds_docker_operation_lock(
    repo_copy,
    monkeypatch,
):
    lock_held = False
    commit = "f" * 40

    @asynccontextmanager
    async def operation_lock():
        nonlocal lock_held
        assert lock_held is False
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    async def prepare_base(_repo):
        assert lock_held is True
        return True

    async def build_enclave(_repo):
        assert lock_held is True
        return "9" * 96

    monkeypatch.setattr(pcr0_builder, "BUILD_DIR", str(repo_copy))
    monkeypatch.setattr(
        pcr0_builder,
        "clone_or_update_repo",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "compute_files_content_hash",
        lambda _repo: "current-content",
    )
    monkeypatch.setattr(
        pcr0_builder,
        "get_latest_commits",
        AsyncMock(
            return_value=[
                {
                    "hash": commit,
                    "timestamp": "2",
                    "message": "current",
                    "date": "",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        pcr0_builder,
        "_docker_operation_lock_scope",
        operation_lock,
    )
    monkeypatch.setattr(
        pcr0_builder,
        "ensure_base_image_exists",
        prepare_base,
    )
    monkeypatch.setattr(
        pcr0_builder,
        "build_pcr0_cache_key",
        lambda _content_hash, _repo: "current-key",
    )
    monkeypatch.setattr(
        pcr0_builder,
        "build_enclave_and_extract_pcr0",
        build_enclave,
    )
    monkeypatch.setattr(pcr0_builder, "_pcr0_cache", {})
    monkeypatch.setattr(pcr0_builder, "_last_content_hash", None)
    monkeypatch.setattr(pcr0_builder, "_build_in_progress", False)

    await pcr0_builder.check_and_build_pcr0()

    assert lock_held is False
    assert pcr0_builder._pcr0_cache["current-key"]["pcr0"] == "9" * 96
