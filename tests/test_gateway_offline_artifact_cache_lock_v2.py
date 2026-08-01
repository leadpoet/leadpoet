from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "gateway" / "tee" / "prepare_offline_artifacts_v2.sh"
LOCK_SCRIPT = ROOT / "validator_tee" / "scripts" / "docker_operation_lock_v2.sh"


def test_offline_artifact_downloads_remain_unlocked_but_publication_is_locked() -> None:
    script = PREPARE_SCRIPT.read_text(encoding="utf-8")

    download = script.index("python3 -m pip download")
    validator_download = script.index("--allow-download")
    acquire = script.index("leadpoet_acquire_docker_operation_lock_v2")
    publish = script.index('rm -rf "$WHEELHOUSE" "$VALIDATOR_RUNTIME"')
    exact_readback = script.index(
        '--offline-artifact-root "$VALIDATOR_RUNTIME" >/dev/null', publish
    )
    release = script.index("leadpoet_release_docker_operation_lock_v2")

    assert download < acquire
    assert validator_download < acquire
    assert acquire < publish < exact_readback < release


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is Linux-only")
def test_offline_artifact_publication_waits_for_active_cache_consumer(
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "docker-operation-v2.lock"
    consumer_held = tmp_path / "consumer-held"
    release_consumer = tmp_path / "release-consumer"
    publisher_entered = tmp_path / "publisher-entered"
    environment = {
        **os.environ,
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "10",
    }

    consumer = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_SCRIPT}"\n'
                "leadpoet_acquire_docker_operation_lock_v2\n"
                f'touch "{consumer_held}"\n'
                f'while [ ! -e "{release_consumer}" ]; do sleep 0.05; done\n'
                "leadpoet_release_docker_operation_lock_v2\n"
            ),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    publisher: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + 5
        while not consumer_held.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert consumer_held.exists(), consumer.communicate(timeout=1)

        publisher = subprocess.Popen(
            [
                "bash",
                "-c",
                (
                    f'. "{LOCK_SCRIPT}"\n'
                    "leadpoet_acquire_docker_operation_lock_v2\n"
                    f'touch "{publisher_entered}"\n'
                    "leadpoet_release_docker_operation_lock_v2\n"
                ),
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.2)
        assert not publisher_entered.exists()

        release_consumer.touch()
        assert consumer.wait(timeout=5) == 0
        assert publisher.wait(timeout=5) == 0
        assert publisher_entered.exists()
    finally:
        release_consumer.touch()
        for process in (consumer, publisher):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
