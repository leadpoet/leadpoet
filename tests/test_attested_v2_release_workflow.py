from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "attested-v2-release.yml"


def test_release_workflow_reclaims_all_unreferenced_docker_state():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert source.count("docker image prune --all --force") == 4
    assert source.count("docker builder prune --all --force") == 4
    assert "docker image prune --force" not in source
    assert "docker builder prune --force" not in source


def test_release_workflow_is_valid_yaml():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert document["name"] == "Attested V2 Release"
