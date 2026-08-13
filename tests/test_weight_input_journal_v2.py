from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.hotkey_authority_v2 import (
    MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
)
from validator_tee.host.weight_input_journal_v2 import (
    AuthoritativeWeightInputJournalV2,
    MAX_WEIGHT_INPUT_JOURNAL_ENVELOPE_BYTES,
    MAX_WEIGHT_INPUT_PLAN_CANONICAL_BYTES,
    WEIGHT_INPUT_JOURNAL_ATOMIC_WRITE_OVERHEAD_BYTES,
    WeightInputJournalV2Error,
    maximum_weight_input_journal_file_bytes_v2,
    require_weight_input_metagraph_match_v2,
    require_weight_input_plan_metagraph_match_v2,
    validate_weight_input_release_identity_v2,
    weight_input_journal_atomic_write_reserve_bytes_v2,
)


COMMIT = "a" * 40
HOTKEY = "5ValidatorHotkey"
NETUID = 71
EPOCH = 24001
ALLOCATION_HASH = "sha256:" + "9" * 64
HASH_MARKERS = "123456789abcdef"


def _hash(marker: str) -> str:
    return "sha256:" + marker * 64


def _release(**updates):
    value = {
        "commit_sha": COMMIT,
        "pcr0": "b" * 96,
        "build_manifest_hash": _hash("c"),
        "dependency_lock_hash": _hash("d"),
        "config_hash": _hash("e"),
        "boot_identity_hash": _hash("f"),
    }
    value.update(updates)
    return value


def _calculation(**updates):
    value = {
        "netuid": NETUID,
        "epoch_id": EPOCH,
        "block": EPOCH * 360 + 180,
        "commit_sha": COMMIT,
        "metagraph_hotkeys": ["5MinerOne", "5MinerTwo", "5MinerThree"],
    }
    value.update(updates)
    return value


def _gateway_inputs(*, authority_marker: str = "8"):
    categories = sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    receipt_hashes = {
        category: _hash(HASH_MARKERS[index])
        for index, category in enumerate(categories)
    }
    return {
        "input_receipt_hashes": receipt_hashes,
        "gateway_authority_event_hash": _hash(authority_marker),
        "request_authorization": {
            "purpose": "validator.gateway_weight_inputs.v2",
            "signature": "public-signature",
        },
        "upstream_ancestry_proofs": {
            category: {"category": category, "receipt_hash": receipt_hashes[category]}
            for category in categories
        },
        "upstream_transport_attempts": [],
    }


def _record_plan(journal, *, release=None, calculation=None, weights=None):
    return journal.record_plan(
        release_identity=release or _release(),
        validator_hotkey=HOTKEY,
        netuid=NETUID,
        epoch_id=EPOCH,
        calculation_snapshot=calculation or _calculation(),
        host_uids=[0, 1, 2],
        host_weights=weights or [0.1, 0.2, 0.7],
        allocation_hash=ALLOCATION_HASH,
        leaderboard_window_start="2026-08-01T00:00:00Z",
        leaderboard_window_end="2026-08-08T00:00:00Z",
    )


def _record_inputs(journal, planned, *, release=None, inputs=None, **scope):
    return journal.record_gateway_inputs(
        release_identity=release or _release(),
        validator_hotkey=scope.get("validator_hotkey", HOTKEY),
        netuid=scope.get("netuid", NETUID),
        epoch_id=scope.get("epoch_id", EPOCH),
        plan_hash=scope.get("plan_hash", planned["plan_hash"]),
        gateway_inputs=inputs or _gateway_inputs(),
    )


def test_journal_exact_bytes_survive_restart_and_metagraph_guard(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)

    assert planned["state"] == "planned"
    assert planned["revision"] == 0
    assert require_weight_input_plan_metagraph_match_v2(
        planned, _calculation()["metagraph_hotkeys"]
    ) == planned
    assert base64.b64decode(planned["plan_canonical_bytes_b64"]) == (
        canonical_json(planned["plan"]).encode("utf-8")
    )
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    assert os.stat(paths[0]).st_mode & 0o777 == 0o600
    with pytest.raises(WeightInputJournalV2Error, match="no verified"):
        require_weight_input_metagraph_match_v2(
            planned, _calculation()["metagraph_hotkeys"]
        )
    with pytest.raises(WeightInputJournalV2Error, match="current metagraph differs"):
        require_weight_input_plan_metagraph_match_v2(
            planned,
            list(reversed(_calculation()["metagraph_hotkeys"])),
        )

    verified = _record_inputs(journal, planned)
    assert verified["state"] == "inputs_verified"
    assert verified["revision"] == 1
    assert base64.b64decode(
        verified["gateway_inputs_canonical_bytes_b64"]
    ) == canonical_json(verified["gateway_inputs"]).encode("utf-8")
    assert require_weight_input_metagraph_match_v2(
        verified, _calculation()["metagraph_hotkeys"]
    ) == verified

    restarted = AuthoritativeWeightInputJournalV2(tmp_path)
    loaded = restarted.load_epoch(
        release_identity=_release(),
        validator_hotkey=HOTKEY,
        netuid=NETUID,
        epoch_id=EPOCH,
    )
    assert loaded == verified
    assert _record_inputs(restarted, planned) == verified


def test_journal_uses_one_deterministic_scoped_path_without_directory_scan(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    unrelated = tmp_path / ("%d-%d-unrelated.json" % (NETUID, EPOCH))
    unrelated.write_text("not-json", encoding="utf-8")

    assert journal.load_epoch(
        release_identity=_release(),
        validator_hotkey=HOTKEY,
        netuid=NETUID,
        epoch_id=EPOCH,
    ) == planned
    assert journal.load_epoch(
        release_identity=_release(pcr0="c" * 96),
        validator_hotkey=HOTKEY,
        netuid=NETUID,
        epoch_id=EPOCH,
    ) is None


def test_same_measured_release_can_recover_after_boot_identity_rotation(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    rotated = _release(boot_identity_hash=_hash("1"))

    loaded = journal.load_epoch(
        release_identity=rotated,
        validator_hotkey=HOTKEY,
        netuid=NETUID,
        epoch_id=EPOCH,
    )
    assert loaded == planned
    verified = _record_inputs(journal, planned, release=rotated)
    assert verified["state"] == "inputs_verified"
    assert verified["release_identity"] == _release()


def test_journal_capacity_refuses_old_scope_without_pruning_current(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path, max_files=1)
    planned = _record_plan(journal)
    older_epoch = EPOCH - 1
    with pytest.raises(
        WeightInputJournalV2Error,
        match="storage capacity is insufficient",
    ):
        journal.record_plan(
            release_identity=_release(),
            validator_hotkey=HOTKEY,
            netuid=NETUID,
            epoch_id=older_epoch,
            calculation_snapshot=_calculation(
                epoch_id=older_epoch,
                block=older_epoch * 360 + 180,
            ),
            host_uids=[0, 1, 2],
            host_weights=[0.1, 0.2, 0.7],
            allocation_hash=ALLOCATION_HASH,
            leaderboard_window_start="2026-08-01T00:00:00Z",
            leaderboard_window_end="2026-08-08T00:00:00Z",
        )

    assert journal.load_epoch(
        release_identity=_release(),
        validator_hotkey=HOTKEY,
        netuid=NETUID,
        epoch_id=EPOCH,
    ) == planned
    assert len(list(tmp_path.glob("*.json"))) == 1
    with pytest.raises(
        WeightInputJournalV2Error,
        match="storage capacity is insufficient",
    ):
        journal.verify_storage_ready()
    verified = _record_inputs(journal, planned)
    assert verified["state"] == "inputs_verified"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_journal_production_defaults_have_no_deterministic_quota(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    assert journal.max_files is None
    assert journal.max_bytes is None


def test_journal_optional_byte_cap_remains_test_configurable(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path, max_bytes=1)
    with pytest.raises(
        WeightInputJournalV2Error,
        match="storage capacity is insufficient",
    ):
        _record_plan(journal)
    assert not list(tmp_path.glob("*.json"))


def test_journal_storage_cli_executes_the_real_readiness_probe(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validator_tee.host.weight_input_journal_v2",
            "--verify-storage-ready",
            "--directory",
            str(tmp_path / "journal"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "validator weight-input storage is ready"


def test_journal_fails_before_write_when_free_space_is_low(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "validator_tee.host.weight_input_journal_v2.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 0})(),
    )
    journal = AuthoritativeWeightInputJournalV2(
        tmp_path,
        min_free_bytes=1,
    )
    with pytest.raises(
        WeightInputJournalV2Error,
        match="storage capacity is insufficient",
    ):
        _record_plan(journal)
    assert not list(tmp_path.glob("*.json"))


def test_journal_readiness_reserves_maximum_duplicated_atomic_write(
    tmp_path,
    monkeypatch,
):
    base64_input = 4 * ((MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES + 2) // 3)
    base64_plan = 4 * ((MAX_WEIGHT_INPUT_PLAN_CANONICAL_BYTES + 2) // 3)
    maximum_file = (
        MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES
        + base64_input
        + MAX_WEIGHT_INPUT_PLAN_CANONICAL_BYTES
        + base64_plan
        + MAX_WEIGHT_INPUT_JOURNAL_ENVELOPE_BYTES
    )
    atomic_reserve = (
        maximum_file + WEIGHT_INPUT_JOURNAL_ATOMIC_WRITE_OVERHEAD_BYTES
    )
    assert maximum_weight_input_journal_file_bytes_v2() == maximum_file
    assert weight_input_journal_atomic_write_reserve_bytes_v2() == atomic_reserve

    free = {"bytes": atomic_reserve}
    monkeypatch.setattr(
        "validator_tee.host.weight_input_journal_v2.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": free["bytes"]})(),
    )
    journal = AuthoritativeWeightInputJournalV2(tmp_path, min_free_bytes=1)
    with pytest.raises(
        WeightInputJournalV2Error,
        match="storage capacity is insufficient",
    ):
        journal.verify_storage_ready()
    free["bytes"] = atomic_reserve + 1
    journal.verify_storage_ready()


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        ({"validator_hotkey": "5OtherValidator"}, "unavailable"),
        ({"netuid": NETUID + 1}, "unavailable"),
        ({"epoch_id": EPOCH + 1}, "unavailable"),
        ({"plan_hash": _hash("0")}, "plan hash differs"),
    ],
)
def test_verified_transition_requires_the_exact_scope(tmp_path, scope, message):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    with pytest.raises(WeightInputJournalV2Error, match=message):
        _record_inputs(journal, planned, **scope)


def test_verified_transition_requires_the_exact_release_identity(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    with pytest.raises(WeightInputJournalV2Error, match="unavailable"):
        _record_inputs(
            journal,
            planned,
            release=_release(pcr0="c" * 96),
        )


def test_journal_rejects_plan_and_gateway_conflicts(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    with pytest.raises(WeightInputJournalV2Error, match="another weight input plan"):
        _record_plan(journal, weights=[0.2, 0.2, 0.6])

    _record_inputs(journal, planned)
    with pytest.raises(WeightInputJournalV2Error, match="gateway inputs conflict"):
        _record_inputs(
            journal,
            planned,
            inputs=_gateway_inputs(authority_marker="7"),
        )


def test_journal_rejects_canonical_byte_tamper_even_with_new_outer_hash(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    _record_inputs(journal, planned)
    path = next(tmp_path.glob("*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    exact = base64.b64decode(value["gateway_inputs_canonical_bytes_b64"])
    value["gateway_inputs_canonical_bytes_b64"] = base64.b64encode(
        exact + b"\n"
    ).decode("ascii")
    body = {key: item for key, item in value.items() if key != "journal_hash"}
    value["journal_hash"] = sha256_json(body)
    path.write_text(canonical_json(value), encoding="utf-8")

    with pytest.raises(WeightInputJournalV2Error, match="canonical bytes differ"):
        journal.load_epoch(
            release_identity=_release(),
            validator_hotkey=HOTKEY,
            netuid=NETUID,
            epoch_id=EPOCH,
        )


def test_journal_rejects_release_scope_tamper(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    _record_plan(journal)
    path = next(tmp_path.glob("*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["release_identity"]["pcr0"] = "c" * 96
    body = {key: item for key, item in value.items() if key != "journal_hash"}
    value["journal_hash"] = sha256_json(body)
    path.write_text(canonical_json(value), encoding="utf-8")

    with pytest.raises(WeightInputJournalV2Error, match="requested scope"):
        journal.load_epoch(
            release_identity=_release(),
            validator_hotkey=HOTKEY,
            netuid=NETUID,
            epoch_id=EPOCH,
        )


def test_late_metagraph_guard_rejects_changed_order(tmp_path):
    journal = AuthoritativeWeightInputJournalV2(tmp_path)
    planned = _record_plan(journal)
    verified = _record_inputs(journal, planned)

    with pytest.raises(WeightInputJournalV2Error, match="current metagraph differs"):
        require_weight_input_metagraph_match_v2(
            verified,
            ["5MinerTwo", "5MinerOne", "5MinerThree"],
        )


def test_release_identity_rejects_missing_fields_and_zero_pcr0():
    missing = _release()
    missing.pop("boot_identity_hash")
    with pytest.raises(WeightInputJournalV2Error, match="fields"):
        validate_weight_input_release_identity_v2(missing)
    with pytest.raises(WeightInputJournalV2Error, match="PCR0"):
        validate_weight_input_release_identity_v2(
            _release(pcr0="0" * 96)
        )
