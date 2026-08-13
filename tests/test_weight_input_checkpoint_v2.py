from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway.research_lab.weight_input_authorization_v2 import (
    GatewayWeightInputAuthorizationStoreV2,
    WeightInputAuthorizationV2Error,
)
from gateway.research_lab.weight_input_checkpoint_v2 import (
    GatewayWeightInputCheckpointStoreV2,
    WeightInputCheckpointV2Error,
)
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.hotkey_authority_v2 import build_weight_inputs_request_v2
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
)


def _hash(index: int) -> str:
    return "sha256:" + format(index, "064x")


def _release_identity(commit: str = "a" * 40) -> dict:
    return {
        "physical_role": "gateway_coordinator",
        "service_role": "gateway_coordinator",
        "commit_sha": commit,
        "pcr0": "b" * 96,
        "build_manifest_hash": _hash(1),
        "dependency_lock_hash": _hash(2),
        "build_identity_hash": _hash(3),
        "release_hash": _hash(4),
    }


def _scope(epoch_id: int = 42) -> dict:
    calculation = {
        "netuid": 71,
        "epoch_id": epoch_id,
        "block": 123456,
        "research_lab_allocation_doc": {"allocation_hash": _hash(5)},
    }
    request = build_weight_inputs_request_v2(
        validator_hotkey="5" + "A" * 47,
        netuid=calculation["netuid"],
        epoch_id=calculation["epoch_id"],
        block=calculation["block"],
        calculation_snapshot_hash=sha256_json(calculation),
        allocation_hash=_hash(5),
        leaderboard_window_start="2026-08-01T00:00:00Z",
        leaderboard_window_end="2026-08-08T00:00:00Z",
    )
    return {"calculation": calculation, "request": request}


def _result(*, producer_release: dict | None = None) -> dict:
    result = {
        "input_receipt_hashes": {
            category: _hash(index + 10)
            for index, category in enumerate(
                sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)
            )
        },
        "gateway_authority_event_hash": _hash(30),
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": [],
            "transport_attempts": [],
            "host_operations": [],
        },
        "compact_ancestry": None,
    }
    if producer_release is not None:
        result["upstream_receipt_set"]["boot_identities"].append(
            {
                "role": producer_release["service_role"],
                "physical_role": producer_release["physical_role"],
                "commit_sha": producer_release["commit_sha"],
                "pcr0": producer_release["pcr0"],
                "build_manifest_hash": producer_release[
                    "build_manifest_hash"
                ],
                "dependency_lock_hash": producer_release[
                    "dependency_lock_hash"
                ],
            }
        )
    return result


def test_authorization_survives_process_state_and_is_exact(tmp_path):
    scope = _scope()
    directory = tmp_path / "checkpoint"
    store = GatewayWeightInputAuthorizationStoreV2(directory)
    persisted = store.persist(
        release_identity=_release_identity(),
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        validator_hotkey_signature="11" * 64,
        source_cutoff_block=240,
    )

    reloaded = GatewayWeightInputAuthorizationStoreV2(directory).load(
        release_identity=_release_identity(),
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        source_cutoff_block=240,
    )
    assert reloaded == persisted
    assert next(directory.iterdir()).stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700

    # sr25519 signatures are not deterministic. The endpoint verifies every
    # retry, while this record retains the original valid authorization proof.
    retried = store.persist(
        release_identity=_release_identity(),
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        validator_hotkey_signature="22" * 64,
        source_cutoff_block=240,
    )
    assert retried == persisted
    assert retried["validator_hotkey_signature"] == "11" * 64

    new_release = _release_identity("c" * 40)
    new_release["release_hash"] = _hash(40)
    assert GatewayWeightInputAuthorizationStoreV2(directory).load(
        release_identity=new_release,
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        source_cutoff_block=240,
    ) is None
    replacement = store.persist(
        release_identity=new_release,
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        validator_hotkey_signature="22" * 64,
        source_cutoff_block=240,
    )
    assert replacement["release_identity"] == new_release
    assert len(list(directory.glob("*.authorized.json"))) == 2


def test_gateway_store_capacity_is_shared_and_never_prunes_authority(tmp_path):
    scope = _scope()
    directory = tmp_path / "checkpoint"
    authorization = GatewayWeightInputAuthorizationStoreV2(
        directory,
        max_files=2,
    )
    persisted = authorization.persist(
        release_identity=_release_identity(),
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        validator_hotkey_signature="11" * 64,
        source_cutoff_block=240,
    )

    checkpoint = GatewayWeightInputCheckpointStoreV2(
        directory,
        max_files=2,
    )
    checkpoint.persist(
        release_identity=_release_identity(),
        request_hash=scope["request"]["request_hash"],
        netuid=scope["request"]["netuid"],
        epoch_id=scope["request"]["epoch_id"],
        allocation_hash=scope["request"]["allocation_hash"],
        calculation_snapshot_hash=scope["request"]["calculation_snapshot_hash"],
        leaderboard_window_start=scope["request"]["leaderboard_window_start"],
        leaderboard_window_end=scope["request"]["leaderboard_window_end"],
        result=_result(producer_release=_release_identity()),
    )
    newer = _scope(epoch_id=43)
    with pytest.raises(
        WeightInputAuthorizationV2Error,
        match="storage capacity is insufficient",
    ):
        authorization.persist(
            release_identity=_release_identity(),
            request=newer["request"],
            calculation_snapshot=newer["calculation"],
            validator_hotkey_signature="22" * 64,
            source_cutoff_block=240,
        )
    with pytest.raises(
        WeightInputAuthorizationV2Error,
        match="storage capacity is insufficient",
    ):
        authorization.verify_storage_ready()

    assert len(list(directory.glob("*.json"))) == 2
    assert authorization.load(
        release_identity=_release_identity(),
        request=scope["request"],
        calculation_snapshot=scope["calculation"],
        source_cutoff_block=240,
    ) == persisted


def test_gateway_production_defaults_have_no_deterministic_quota(tmp_path):
    authorization = GatewayWeightInputAuthorizationStoreV2(tmp_path)
    checkpoint = GatewayWeightInputCheckpointStoreV2(tmp_path)
    assert authorization.max_files is None
    assert authorization.max_bytes is None
    assert checkpoint.max_files is None
    assert checkpoint.max_bytes is None


def test_gateway_storage_cli_executes_the_real_readiness_probe(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gateway.research_lab.weight_input_authorization_v2",
            "--verify-storage-ready",
            "--directory",
            str(tmp_path / "checkpoint"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gateway weight-input storage is ready"


def test_gateway_optional_cap_counts_crash_leftovers_without_deleting(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    leftover = directory / ".interrupted-checkpoint"
    leftover.write_bytes(b"incomplete")
    scope = _scope()
    store = GatewayWeightInputAuthorizationStoreV2(directory, max_files=2)
    with pytest.raises(
        WeightInputAuthorizationV2Error,
        match="storage capacity is insufficient",
    ):
        store.persist(
            release_identity=_release_identity(),
            request=scope["request"],
            calculation_snapshot=scope["calculation"],
            validator_hotkey_signature="11" * 64,
            source_cutoff_block=240,
        )
    assert leftover.read_bytes() == b"incomplete"


def test_gateway_optional_byte_cap_remains_test_configurable(tmp_path):
    scope = _scope()
    store = GatewayWeightInputCheckpointStoreV2(tmp_path, max_bytes=1)
    with pytest.raises(
        WeightInputCheckpointV2Error,
        match="storage capacity is insufficient",
    ):
        store.persist(
            release_identity=_release_identity(),
            request_hash=scope["request"]["request_hash"],
            netuid=scope["request"]["netuid"],
            epoch_id=scope["request"]["epoch_id"],
            allocation_hash=scope["request"]["allocation_hash"],
            calculation_snapshot_hash=scope["request"][
                "calculation_snapshot_hash"
            ],
            leaderboard_window_start=scope["request"][
                "leaderboard_window_start"
            ],
            leaderboard_window_end=scope["request"]["leaderboard_window_end"],
            result=_result(producer_release=_release_identity()),
        )
    assert not list(tmp_path.glob("*.json"))


def test_gateway_store_fails_before_write_when_free_space_is_low(
    tmp_path, monkeypatch
):
    scope = _scope()
    directory = tmp_path / "checkpoint"
    monkeypatch.setattr(
        "gateway.research_lab.weight_input_checkpoint_v2.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 0})(),
    )
    store = GatewayWeightInputCheckpointStoreV2(
        directory,
        min_free_bytes=1,
    )
    with pytest.raises(
        WeightInputCheckpointV2Error,
        match="storage capacity is insufficient",
    ):
        store.persist(
            release_identity=_release_identity(),
            request_hash=scope["request"]["request_hash"],
            netuid=scope["request"]["netuid"],
            epoch_id=scope["request"]["epoch_id"],
            allocation_hash=scope["request"]["allocation_hash"],
            calculation_snapshot_hash=scope["request"][
                "calculation_snapshot_hash"
            ],
            leaderboard_window_start=scope["request"][
                "leaderboard_window_start"
            ],
            leaderboard_window_end=scope["request"]["leaderboard_window_end"],
            result=_result(producer_release=_release_identity()),
        )
    assert not list(directory.glob("*.json"))


def test_ready_checkpoint_replays_exact_result_and_rejects_tamper(tmp_path):
    scope = _scope()
    directory = tmp_path / "checkpoint"
    store = GatewayWeightInputCheckpointStoreV2(directory)
    producer_release = _release_identity()
    persisted = store.persist(
        release_identity=producer_release,
        request_hash=scope["request"]["request_hash"],
        netuid=scope["request"]["netuid"],
        epoch_id=scope["request"]["epoch_id"],
        allocation_hash=scope["request"]["allocation_hash"],
        calculation_snapshot_hash=scope["request"]["calculation_snapshot_hash"],
        leaderboard_window_start=scope["request"]["leaderboard_window_start"],
        leaderboard_window_end=scope["request"]["leaderboard_window_end"],
        result=_result(producer_release=producer_release),
    )
    reloaded = GatewayWeightInputCheckpointStoreV2(directory).load(
        release_identity=_release_identity(),
        request_hash=scope["request"]["request_hash"],
        netuid=scope["request"]["netuid"],
        epoch_id=scope["request"]["epoch_id"],
        allocation_hash=scope["request"]["allocation_hash"],
        calculation_snapshot_hash=scope["request"]["calculation_snapshot_hash"],
        leaderboard_window_start=scope["request"]["leaderboard_window_start"],
        leaderboard_window_end=scope["request"]["leaderboard_window_end"],
    )
    assert reloaded == persisted

    replacement_release = _release_identity("c" * 40)
    replacement_release["release_hash"] = _hash(40)
    replayed = GatewayWeightInputCheckpointStoreV2(directory).load(
        release_identity=replacement_release,
        request_hash=scope["request"]["request_hash"],
        netuid=scope["request"]["netuid"],
        epoch_id=scope["request"]["epoch_id"],
        allocation_hash=scope["request"]["allocation_hash"],
        calculation_snapshot_hash=scope["request"]["calculation_snapshot_hash"],
        leaderboard_window_start=scope["request"]["leaderboard_window_start"],
        leaderboard_window_end=scope["request"]["leaderboard_window_end"],
    )
    assert replayed == persisted
    assert replayed["release_identity"] == producer_release

    path = next(directory.glob("*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["result"]["gateway_authority_event_hash"] = _hash(99)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(WeightInputCheckpointV2Error, match="result hash differs"):
        GatewayWeightInputCheckpointStoreV2(directory).load(
            release_identity=_release_identity(),
            request_hash=scope["request"]["request_hash"],
            netuid=scope["request"]["netuid"],
            epoch_id=scope["request"]["epoch_id"],
            allocation_hash=scope["request"]["allocation_hash"],
            calculation_snapshot_hash=scope["request"][
                "calculation_snapshot_hash"
            ],
            leaderboard_window_start=scope["request"][
                "leaderboard_window_start"
            ],
            leaderboard_window_end=scope["request"]["leaderboard_window_end"],
        )


def test_cross_release_checkpoint_requires_producer_receipt_ancestry(tmp_path):
    scope = _scope()
    directory = tmp_path / "checkpoint"
    producer_release = _release_identity()
    GatewayWeightInputCheckpointStoreV2(directory).persist(
        release_identity=producer_release,
        request_hash=scope["request"]["request_hash"],
        netuid=scope["request"]["netuid"],
        epoch_id=scope["request"]["epoch_id"],
        allocation_hash=scope["request"]["allocation_hash"],
        calculation_snapshot_hash=scope["request"]["calculation_snapshot_hash"],
        leaderboard_window_start=scope["request"]["leaderboard_window_start"],
        leaderboard_window_end=scope["request"]["leaderboard_window_end"],
        result=_result(),
    )
    replacement_release = _release_identity("c" * 40)
    replacement_release["release_hash"] = _hash(40)
    with pytest.raises(
        WeightInputCheckpointV2Error,
        match="producer release is absent from receipt ancestry",
    ):
        GatewayWeightInputCheckpointStoreV2(directory).load(
            release_identity=replacement_release,
            request_hash=scope["request"]["request_hash"],
            netuid=scope["request"]["netuid"],
            epoch_id=scope["request"]["epoch_id"],
            allocation_hash=scope["request"]["allocation_hash"],
            calculation_snapshot_hash=scope["request"][
                "calculation_snapshot_hash"
            ],
            leaderboard_window_start=scope["request"][
                "leaderboard_window_start"
            ],
            leaderboard_window_end=scope["request"]["leaderboard_window_end"],
        )
