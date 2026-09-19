import json
from pathlib import Path

import pytest

from gateway.tee.install_gateway_release_state_v2 import (
    GatewayReleaseStateInstallError,
    install_gateway_release_state,
)
from tests.test_release_channel_v2 import COMMIT, _manifest


def test_installs_current_manifest_and_preserves_previous(tmp_path: Path):
    prepared_manifest = tmp_path / "prepared-manifest.json"
    prepared_manifest.write_text(json.dumps(_manifest()))
    active_manifest = tmp_path / "active-manifest.json"
    active_manifest.write_text('{"old":true}')

    result = install_gateway_release_state(
        prepared_manifest=prepared_manifest,
        active_manifest=active_manifest,
        expected_commit=COMMIT,
    )

    assert result["commit_sha"] == COMMIT
    assert json.loads(active_manifest.read_text())["commit_sha"] == COMMIT
    assert json.loads(
        (tmp_path / "active-manifest.json.previous").read_text()
    ) == {"old": True}


def test_rejects_mismatched_manifest_before_active_state_changes(tmp_path: Path):
    prepared_manifest = tmp_path / "prepared-manifest.json"
    prepared_manifest.write_text(json.dumps(_manifest("2" * 40)))
    active_manifest = tmp_path / "active-manifest.json"
    active_manifest.write_text('{"old":true}')

    with pytest.raises(GatewayReleaseStateInstallError, match="commit differs"):
        install_gateway_release_state(
            prepared_manifest=prepared_manifest,
            active_manifest=active_manifest,
            expected_commit=COMMIT,
        )

    assert json.loads(active_manifest.read_text()) == {"old": True}
