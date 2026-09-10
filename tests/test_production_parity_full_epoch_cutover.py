from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from scripts import run_production_parity_full_host as full_host
from scripts import run_production_parity_fast as fast_parity
from scripts import run_local_restart_rehearsal as restart_rehearsal


def _candidate_cutover() -> dict[str, object]:
    return json.loads(
        (full_host.ROOT / "config/stateful-epoch-cutover-sn71.json").read_text(
            encoding="utf-8"
        )
    )


def _runtime_config(path: Path, cutover: dict[str, object]) -> Path:
    path.write_text(
        json.dumps({"execution_config": {"epoch_authority": {"cutover": cutover}}}),
        encoding="utf-8",
    )
    return path


def _restart_epoch_report(cutover: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "leadpoet.restart_epoch_gate.v1",
        "restart_allowed": True,
        "snapshot": {
            "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
            "epoch_scheme": "bittensor.subnet_epoch_index.v1",
            "network_genesis_hash": cutover["network_genesis_hash"],
            "netuid": 71,
            "head_kind": "exact",
            "block_hash": "0x" + "1" * 64,
            "current_block": 100,
            "last_epoch_block": 95,
            "pending_epoch_at": 0,
            "subnet_epoch_index": 24020,
            "tempo": 360,
            "blocks_since_last_step": 5,
            "observed_at": "2026-09-04T00:00:00+00:00",
        },
    }


def _gateway_mapping_helper() -> str:
    lines = (full_host.ROOT / "gw_restart.sh").read_text(encoding="utf-8").splitlines()
    start = lines.index("gateway_weight_preflight_epoch_from_restart_report() {")
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "}")
    return "\n".join(lines[start : end + 1])


def test_full_materializes_mapping_for_exact_gateway_helper_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    manifest = home / ".config/leadpoet/stateful-epoch-cutover.json"
    monkeypatch.setattr(full_host, "FULL_STATEFUL_CUTOVER_MANIFEST", manifest)
    cutover = _candidate_cutover()
    runtime = _runtime_config(tmp_path / "runtime.json", cutover)

    state = full_host._materialize_full_stateful_cutover_manifest(runtime)

    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    harness = f"""set -euo pipefail
{_gateway_mapping_helper()}
GATEWAY_PREFLIGHT_TREE="$1"
GATEWAY_PYTHON_BIN="$2"
GATEWAY_STATEFUL_CUTOVER_MANIFEST="$3"
gateway_weight_preflight_epoch_from_restart_report "$4"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "full-clean-host-transition",
            str(full_host.ROOT),
            sys.executable,
            str(manifest),
            json.dumps(_restart_epoch_report(cutover), sort_keys=True),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "24073"
    assert full_host._cleanup_full_stateful_cutover_manifest(state) == "removed"
    assert not manifest.exists()


@pytest.mark.parametrize("existing_kind", ["regular", "symlink"])
def test_full_preserves_preexisting_mapping_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing_kind: str,
) -> None:
    manifest = tmp_path / ".config/leadpoet/stateful-epoch-cutover.json"
    manifest.parent.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    foreign.write_text("foreign\n", encoding="utf-8")
    if existing_kind == "regular":
        manifest.write_text("preexisting\n", encoding="utf-8")
    else:
        manifest.symlink_to(foreign)
    monkeypatch.setattr(full_host, "FULL_STATEFUL_CUTOVER_MANIFEST", manifest)
    runtime = _runtime_config(tmp_path / "runtime.json", _candidate_cutover())

    with pytest.raises(full_host.FullParityError, match="already exists"):
        full_host._materialize_full_stateful_cutover_manifest(runtime)

    assert foreign.read_text(encoding="utf-8") == "foreign\n"
    if existing_kind == "regular":
        assert manifest.read_text(encoding="utf-8") == "preexisting\n"
    else:
        assert manifest.is_symlink()


def test_full_rejects_captured_mapping_mismatch_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / ".config/leadpoet/stateful-epoch-cutover.json"
    monkeypatch.setattr(full_host, "FULL_STATEFUL_CUTOVER_MANIFEST", manifest)
    cutover = _candidate_cutover()
    cutover.pop("mapping_hash")
    cutover["first_settlement_epoch_id"] = int(cutover["first_settlement_epoch_id"]) + 1
    cutover["last_legacy_epoch_id"] = int(cutover["last_legacy_epoch_id"]) + 1
    runtime = _runtime_config(tmp_path / "runtime.json", cutover)

    with pytest.raises(full_host.FullParityError, match="authorities differ"):
        full_host._materialize_full_stateful_cutover_manifest(runtime)

    assert not manifest.exists()


def test_full_cleanup_preserves_replacement_mapping_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / ".config/leadpoet/stateful-epoch-cutover.json"
    monkeypatch.setattr(full_host, "FULL_STATEFUL_CUTOVER_MANIFEST", manifest)
    state = full_host._materialize_full_stateful_cutover_manifest(
        _runtime_config(tmp_path / "runtime.json", _candidate_cutover())
    )
    manifest.unlink()
    manifest.write_text("replacement\n", encoding="utf-8")
    manifest.chmod(0o600)

    with pytest.raises(full_host.FullParityError, match="cleanup identity differs"):
        full_host._cleanup_full_stateful_cutover_manifest(state)

    assert manifest.read_text(encoding="utf-8") == "replacement\n"


def test_full_rejects_config_directory_symlink_before_creating_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (home / ".config").symlink_to(foreign, target_is_directory=True)
    manifest = home / ".config/leadpoet/stateful-epoch-cutover.json"
    monkeypatch.setattr(full_host, "FULL_STATEFUL_CUTOVER_MANIFEST", manifest)

    with pytest.raises(full_host.FullParityError, match="directory is unsafe"):
        full_host._materialize_full_stateful_cutover_manifest(
            _runtime_config(tmp_path / "runtime.json", _candidate_cutover())
        )

    assert not (foreign / "leadpoet").exists()


def test_full_weight_rehearsal_parent_uses_fast_cleanup_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_sha = "a" * 40
    candidate_sha = "b" * 40
    allocation_doc = {"epoch": 24100, "weights": [1.0]}
    allocation = {
        "allocation_doc": allocation_doc,
        "allocation_hash": full_host.sha256_json(allocation_doc),
        "handoff_hash": "sha256:" + "1" * 64,
        "source_epoch": 24100,
    }
    production_allocation = tmp_path / "allocation.json"
    production_allocation.write_text(json.dumps(allocation), encoding="utf-8")
    expected_evidence = tmp_path / (
        f"leadpoet-restart-rehearsal-{candidate_sha}-prepush.json"
    )
    real_path = Path
    observed: dict[str, int] = {}

    def isolated_path(value: object) -> Path:
        return tmp_path if value == "/tmp" else real_path(value)

    def fake_run(
        *_args: object, timeout: int, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed["timeout"] = timeout
        if timeout < 661:
            raise subprocess.TimeoutExpired("rehearsal", timeout)
        expected_evidence.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "release_sha": candidate_sha,
                    "from_sha": base_sha,
                    "bundle_hash": "sha256:" + "2" * 64,
                    "canonical_vector": {"hotkeys": ["one"], "weights": [1.0]},
                    "auditor": {"hotkeys": ["one"], "weights": [1.0]},
                    "signed_extrinsic": {"submitted": False},
                    "finalization": {"verified": True},
                    "reveal": {"verified": True},
                    "production_allocation": {
                        "allocation_hash": allocation["allocation_hash"],
                        "handoff_hash": allocation["handoff_hash"],
                        "source_epoch": allocation["source_epoch"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(full_host, "Path", isolated_path)
    monkeypatch.setattr(full_host, "_run", fake_run)
    result = full_host._run_nonforwarding_weight_path(
        base_sha=base_sha,
        candidate_sha=candidate_sha,
        production_allocation=production_allocation,
    )

    assert observed == {"timeout": full_host.FAST_REHEARSAL_TIMEOUT_SECONDS}
    assert restart_rehearsal.PROFILE_LIMITS["prepush"]["target_seconds"] == 900
    assert full_host.FAST_REHEARSAL_TIMEOUT_SECONDS == (
        restart_rehearsal.PROFILE_LIMITS["prepush"]["target_seconds"]
        + fast_parity.FAST_REHEARSAL_PARENT_CLEANUP_HEADROOM_SECONDS
    )
    assert fast_parity.FAST_REHEARSAL_PARENT_CLEANUP_HEADROOM_SECONDS == 120
    assert result["production_allocation_bound"] is True
    assert result["production_allocation_hash"] == allocation["allocation_hash"]
    assert result["production_allocation_document_hash"] == full_host.sha256_json(
        allocation_doc
    )
    assert result["primary_audit_equal"] is True
    assert result["sdk_signed"] is True
    assert result["finalization_verified"] is True
    assert result["readback_verified"] is True
    assert result["chain_boundary"] == "strict-non-forwarding"
