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


def test_release_workflow_reclaims_only_images_created_by_the_build():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "docker image prune --all --force" not in source
    assert source.count("docker builder prune --all --force") == 2
    assert source.count("reclaim_release_build_images_v2.sh") == 4
    assert source.count("preexisting-images.txt") == 2
    assert source.count("trap cleanup_release_build_images_on_exit EXIT") == 2
    assert source.count("validator_tee/scripts/reclaim_docker_storage_v2.sh") == 4
    assert source.count("VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1") == 2
    assert source.count("REQUIRE_ZERO_RUNTIME_RECONCILE=1") == 1
    assert source.count(
        'sudo rm -rf -- \\\n            "$RUNNER_TEMP/offline-artifacts"'
    ) == 4
    assert source.count('"$RUNNER_TEMP/release-evidence" \\') == 4
    assert "Reclaim gateway-parent storage after evidence generation" in source
    assert "Reclaim validator-parent storage after evidence generation" in source
    assert "docker image prune --force" not in source

    workflow = yaml.safe_load(source)
    cleanup_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("name", "")).startswith("Reclaim ")
        and "storage after evidence generation" in str(step.get("name", ""))
    ]
    assert len(cleanup_steps) == 2
    assert all(step.get("if") == "always()" for step in cleanup_steps)

    gateway_steps = workflow["jobs"]["gateway-parent"]["steps"]
    gateway_prebuild = next(
        step
        for step in gateway_steps
        if step.get("name") == "Reclaim unreferenced gateway build space"
    )
    assert "REQUIRE_ZERO_RUNTIME_RECONCILE=1" in gateway_prebuild["run"]
    reclaim_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "validator_tee/scripts/reclaim_docker_storage_v2.sh"
        in str(step.get("run", ""))
    ]
    assert len(reclaim_steps) == 4
    assert all(
        ("REQUIRE_ZERO_RUNTIME_RECONCILE=1" in str(step.get("run", "")))
        == (step is gateway_prebuild)
        for step in reclaim_steps
    )
    for job_name in ("gateway-parent", "validator-parent"):
        build_step = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("name") == "Build gateway and validator evidence"
        )
        subprocess.run(
            ["bash", "-n"],
            input=build_step["run"],
            text=True,
            check=True,
        )


def test_release_workflow_is_valid_yaml():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert document["name"] == "Attested V2 Release"


def test_attested_release_builders_are_not_block_gated():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Wait for safe subnet build window" not in workflow
    assert "subnet71_position" not in workflow
    assert "chain.get_current_block()" not in workflow


def test_release_workflow_resets_exact_runner_workspace_before_checkout():
    source = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)

    step_name = "Reset stale V2 runner workspace before checkout"
    assert source.count(step_name) == 3
    for job_name in ("gateway-parent", "validator-parent", "publish"):
        steps = workflow["jobs"][job_name]["steps"]
        repair_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == step_name
        )
        checkout_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Check out exact release source"
        )
        assert repair_index < checkout_index

        repair = steps[repair_index]
        command = repair["run"]
        assert repair.get("id") == "stale_cleanup"
        assert 'test ! -L "$GITHUB_WORKSPACE"' in command
        assert 'realpath -- "$GITHUB_WORKSPACE"' in command
        assert 'realpath -- "$(dirname -- "$RUNNER_TEMP")"' in command
        assert '$(dirname -- "$(dirname -- "$workspace")")' in command
        assert '$(basename -- "$workspace")' in command
        assert (
            'sudo find -P "$workspace" -xdev -depth -mindepth 1 -delete'
            in command
        )
        assert "sudo rm" not in command
        assert "sudo chown" not in command
        subprocess.run(
            ["bash", "-n"],
            input=command,
            text=True,
            check=True,
        )


def test_release_workspace_reset_is_bounded_and_fail_closed(tmp_path):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    command = next(
        step["run"]
        for step in workflow["jobs"]["gateway-parent"]["steps"]
        if step.get("name") == "Reset stale V2 runner workspace before checkout"
    )

    repository_name = "leadpoet"
    runner_work_root = tmp_path / "runner" / "_work"
    runner_temp = runner_work_root / "_temp"
    workspace = runner_work_root / repository_name / repository_name
    stale_cache = workspace / "Leadpoet" / "__pycache__"
    stale_cache.mkdir(parents=True)
    (stale_cache / "module.cpython-311.pyc").write_bytes(b"stale")
    (workspace / ".drand-cabi-v2.stale").mkdir()
    runner_temp.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_marker = outside / "must-remain"
    outside_marker.write_text("outside", encoding="utf-8")
    (workspace / "outside-link").symlink_to(outside, target_is_directory=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text('#!/bin/sh\nexec "$@"\n', encoding="utf-8")
    fake_sudo.chmod(0o700)

    env = os.environ.copy()
    env.update(
        {
            "GITHUB_REPOSITORY": f"leadpoet/{repository_name}",
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(runner_temp),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(workspace.iterdir()) == []
    assert outside_marker.read_text(encoding="utf-8") == "outside"

    wrong_workspace = runner_work_root / repository_name
    wrong_workspace.mkdir(exist_ok=True)
    wrong_marker = wrong_workspace / "must-remain"
    wrong_marker.write_text("wrong path", encoding="utf-8")
    env["GITHUB_WORKSPACE"] = str(wrong_workspace)
    rejected = subprocess.run(
        ["bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert wrong_marker.read_text(encoding="utf-8") == "wrong path"
