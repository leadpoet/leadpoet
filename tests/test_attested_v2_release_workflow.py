from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "attested-v2-release.yml"


def test_release_workflow_reclaims_all_unreferenced_docker_state():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert source.count("docker image prune --all --force") == 2
    assert source.count("docker builder prune --all --force") == 2
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
    assert "docker builder prune --force" not in source

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


def test_release_workflow_is_valid_yaml():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert document["name"] == "Attested V2 Release"


def test_attested_release_builders_are_not_block_gated():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Wait for safe subnet build window" not in workflow
    assert "subnet71_position" not in workflow
    assert "chain.get_current_block()" not in workflow
