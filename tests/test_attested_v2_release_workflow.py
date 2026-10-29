import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "attested-v2-release.yml"
IMAGE_RECLAIMER = (
    ROOT / "validator_tee" / "scripts" / "reclaim_release_build_images_v2.sh"
)


def _run_image_reclaimer(
    *,
    mode: str,
    inventory: Path,
    state: Path,
    fake_bin: Path,
    lock: Path,
) -> subprocess.CompletedProcess[str]:
    command = """
exec 7>>"$TEST_DOCKER_LOCK"
export LEADPOET_DOCKER_OPERATION_LOCK_FILE="$TEST_DOCKER_LOCK"
export LEADPOET_DOCKER_OPERATION_LOCK_HELD=1
export LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD=1
bash "$TEST_IMAGE_RECLAIMER" "$1" "$2"
"""
    return subprocess.run(
        ["bash", "-c", command, "reclaimer", mode, str(inventory)],
        env={
            **os.environ,
            "DOCKER_TEST_STATE": str(state),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "TEST_DOCKER_LOCK": str(lock),
            "TEST_IMAGE_RECLAIMER": str(IMAGE_RECLAIMER),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_image_reclaimer_preserves_preexisting_images(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/bin/bash
set -euo pipefail
state="${DOCKER_TEST_STATE:?}"
if [ "$1" = image ] && [ "$2" = ls ]; then
  cat "$state"
elif [ "$1" = image ] && [ "$2" = rm ] && [ "$3" = --force ] && [ "$4" = -- ]; then
  image_id="$5"
  next="${state}.next"
  grep -Fvx -- "$image_id" "$state" > "$next" || true
  mv "$next" "$state"
else
  echo "unexpected docker command: $*" >&2
  exit 2
fi
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o700)
    fake_readlink = fake_bin / "readlink"
    fake_readlink.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$1" = /proc/*/fd/7 ]]; then
  printf '%s\n' "${TEST_DOCKER_LOCK:?}"
else
  exec /usr/bin/readlink "$@"
fi
""",
        encoding="utf-8",
    )
    fake_readlink.chmod(0o700)
    first = "sha256:" + "1" * 64
    second = "sha256:" + "2" * 64
    created = "sha256:" + "3" * 64
    state = tmp_path / "images.txt"
    state.write_text(f"{first}\n{second}\n", encoding="utf-8")
    inventory = tmp_path / "baseline.txt"
    lock = tmp_path / "docker.lock"

    snapshot = _run_image_reclaimer(
        mode="snapshot",
        inventory=inventory,
        state=state,
        fake_bin=fake_bin,
        lock=lock,
    )
    assert snapshot.returncode == 0, snapshot.stderr
    state.write_text(f"{first}\n{second}\n{created}\n", encoding="utf-8")

    cleanup = _run_image_reclaimer(
        mode="cleanup",
        inventory=inventory,
        state=state,
        fake_bin=fake_bin,
        lock=lock,
    )
    assert cleanup.returncode == 0, cleanup.stderr
    assert state.read_text(encoding="utf-8").splitlines() == [first, second]
    assert inventory.read_text(encoding="utf-8").splitlines() == [first, second]



def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_suspended_release_has_no_production_or_publication_authority():
    workflow = _workflow()
    assert workflow.get("on", workflow.get(True)) == {"workflow_dispatch": None}
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"release-isolation-admission"}
    job = jobs["release-isolation-admission"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["environment"] == "attested-release"
    assert "github.repository == 'leadpoet/leadpoet'" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "permissions" not in job
    for step in job["steps"]:
        if "uses" in step:
            assert step["uses"].split("@", 1)[0] == "actions/checkout"
            assert len(step["uses"].split("@", 1)[1]) == 40
            assert step["with"]["persist-credentials"] is False
        if "run" in step:
            subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
    source = WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "self-hosted", "leadpoet-gateway-v2-builder", "leadpoet-validator-v2-builder",
        "id-token", "secrets.", "release_channel_v2", "docker", "configure-aws-credentials",
    ):
        assert forbidden not in source


def test_release_block_is_unconditional_and_cannot_be_enabled_by_a_variable():
    steps = _workflow()["jobs"]["release-isolation-admission"]["steps"]
    block = steps[-1]
    assert "if" not in block
    assert "continue-on-error" not in block
    assert "vars." not in block["run"]
    result = subprocess.run(
        ["bash", "-e", "-c", block["run"]], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "Attested publication is suspended" in result.stdout
