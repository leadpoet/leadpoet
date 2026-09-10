import json
from pathlib import Path

import pytest

from gateway.tee.install_gateway_release_state_v2 import install_gateway_release_state
from gateway.tee.release_channel_v2 import build_release_channel_v2, build_release_lineage_v2
from tests.test_release_channel_v2 import COMMIT, _manifest


def test_installs_matching_manifest_and_lineage_and_preserves_previous(tmp_path: Path):
    manifest = _manifest()
    channel = build_release_channel_v2(gateway_release_manifest=manifest)
    lineage = build_release_lineage_v2([channel], current_commit=COMMIT)
    prepared_manifest = tmp_path / "prepared-manifest.json"
    prepared_lineage = tmp_path / "prepared-lineage.json"
    prepared_manifest.write_text(json.dumps(manifest))
    prepared_lineage.write_text(json.dumps(lineage))
    active_manifest = tmp_path / "active-manifest.json"
    active_lineage = tmp_path / "active-lineage.json"
    active_manifest.write_text('{"old":true}')
    active_lineage.write_text('{"old":true}')

    result = install_gateway_release_state(
        prepared_manifest=prepared_manifest, prepared_lineage=prepared_lineage,
        active_manifest=active_manifest, active_lineage=active_lineage,
        expected_commit=COMMIT,
    )
    assert result["commit_sha"] == COMMIT
    assert json.loads(active_manifest.read_text())["commit_sha"] == COMMIT
    assert json.loads(active_lineage.read_text())["current_commit_sha"] == COMMIT
    assert json.loads((tmp_path / "active-manifest.json.previous").read_text()) == {"old": True}
    assert json.loads((tmp_path / "active-lineage.json.previous").read_text()) == {"old": True}


def test_rejects_mismatched_lineage_before_active_state_changes(tmp_path: Path):
    manifest = _manifest()
    channel = build_release_channel_v2(gateway_release_manifest=manifest)
    lineage = build_release_lineage_v2([channel], current_commit=COMMIT)
    lineage["current_gateway_release_hash"] = "sha256:" + "0" * 64
    prepared_manifest = tmp_path / "prepared-manifest.json"
    prepared_lineage = tmp_path / "prepared-lineage.json"
    prepared_manifest.write_text(json.dumps(manifest))
    prepared_lineage.write_text(json.dumps(lineage))
    active_manifest = tmp_path / "active-manifest.json"
    active_lineage = tmp_path / "active-lineage.json"
    active_manifest.write_text('{"old":true}')
    active_lineage.write_text('{"old":true}')
    with pytest.raises(Exception):
        install_gateway_release_state(
            prepared_manifest=prepared_manifest, prepared_lineage=prepared_lineage,
            active_manifest=active_manifest, active_lineage=active_lineage,
            expected_commit=COMMIT,
        )
    assert json.loads(active_manifest.read_text()) == {"old": True}
    assert json.loads(active_lineage.read_text()) == {"old": True}
