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


def test_release_authority_is_limited_to_separate_protected_publisher():
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"source", "builds", "verify", "publish"}
    for name, job in workflow["jobs"].items():
        if name == "publish":
            assert job["environment"] == "attested-release"
            assert job["permissions"] == {"contents": "read", "id-token": "write"}
            assert job["runs-on"] == "ubuntu-24.04"
            assert "inputs.reproduce_commit == ''" in job["if"]
            assert "needs.source.outputs.mode == 'publish'" in job["if"]
        else:
            assert "environment" not in job
            assert "permissions" not in job
        for step in job.get("steps", []):
            if "uses" in step:
                action, pin = step["uses"].split("@", 1)
                assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)
                if action == "actions/checkout":
                    assert step["with"]["persist-credentials"] is False
                    assert step["with"]["ref"] == "${{ github.sha }}"
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
    source = WORKFLOW.read_text(encoding="utf-8")
    for forbidden in ("self-hosted", "leadpoet-gateway-v2-builder", "leadpoet-validator-v2-builder", "secrets."):
        assert forbidden not in source
    assert workflow["jobs"]["builds"]["strategy"]["matrix"]["include"] == [
        {"domain": "gateway", "runner": "ubuntu-24.04"},
        {"domain": "validator", "runner": "runs-on"},
    ]
    runner = workflow["jobs"]["builds"]["runs-on"]
    assert "/runner=attested-v2-isolated/spot=false/retry=false" in runner
    assert "|| 'ubuntu" not in runner


def test_release_runner_does_not_inherit_custom_image_hooks():
    config = yaml.safe_load((ROOT / ".github/runs-on.yml").read_text())
    # RunsOn replaces common top-level keys, so even a same-named image in a
    # future inherited configuration cannot add hooks to the release runner.
    assert config["images"] == {}
    assert config["runners"]["attested-v2-isolated"]["image"] == "ubuntu24-full-x64"


def test_artifact_validation_precedes_oidc_and_uses_exact_separate_artifacts():
    for name in ("verify", "publish"):
        steps = _workflow()["jobs"][name]["steps"]
        downloads = [s for s in steps if s.get("uses", "").startswith("actions/download-artifact@")]
        assert len(downloads) == 2
        assert len({s["with"]["path"] for s in downloads}) == 2
        for download in downloads:
            assert "pattern" not in download["with"]
            assert "merge-multiple" not in download["with"]
            artifact = download["with"]["name"]
            assert "${{ needs.source.outputs.build_commit }}" in artifact
            assert "${{ github.run_id }}-${{ github.run_attempt }}" in artifact
        if name == "publish":
            verify = next(i for i, s in enumerate(steps) if s.get("run") == "bash .github/scripts/assemble_attested_release.sh")
            oidc = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials@"))
            assert verify < oidc
            assert steps[oidc]["with"]["role-to-assume"] == "arn:aws:iam::493765492819:role/leadpoet-attested-release-publisher"
