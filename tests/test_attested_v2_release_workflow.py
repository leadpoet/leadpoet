from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "attested-v2-release.yml"


def test_release_workflow_reclaims_all_unreferenced_docker_state():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert source.count("docker image prune --all --force") == 2
    assert source.count("docker builder prune --all --force") == 2
    assert source.count("validator_tee/scripts/reclaim_docker_storage_v2.sh") == 2
    assert source.count(
        'sudo rm -rf -- \\\n            "$RUNNER_TEMP/offline-artifacts"'
    ) == 2
    assert source.count('"$RUNNER_TEMP/release-evidence" \\') == 2
    assert "docker image prune --force" not in source
    assert "docker builder prune --force" not in source


def test_release_workflow_is_valid_yaml():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert document["name"] == "Attested V2 Release"


def test_attested_release_builders_are_not_block_gated():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Wait for safe subnet build window" not in workflow
    assert "subnet71_position" not in workflow
    assert "chain.get_current_block()" not in workflow
