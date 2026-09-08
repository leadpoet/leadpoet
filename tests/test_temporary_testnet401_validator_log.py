from __future__ import annotations

import json
import sys

import pytest

from scripts import bootstrap_temporary_testnet_weights_host as native
from scripts import extract_temporary_testnet401_validator_log as reader


CANDIDATE = "a" * 40
EPOCH = 22_060
SECRET = "private-canary-never-return"


def _run(tmp_path, monkeypatch, capsys, *, logged_epoch=EPOCH):
    runtime = tmp_path / "runtime"
    logs = runtime / "logs"
    logs.mkdir(parents=True)
    (logs / "validator_application.log").write_text(
        "\n".join(
            (
                SECRET,
                f"SUBMITTING WEIGHTS FOR EPOCH {logged_epoch}",
                "   Block: 7959790 (block 65/360, 295 remaining)",
                "   Authoritative V2 gateway bundle persisted: sha256:3333333333333...",
                "   Authoritative V2 finalized chain state persisted: sha256:4444444444444...",
                "Successfully submitted weights to Bittensor chain",
                '{"event": "automatic_weight_tick", "submitted_or_already_complete": true}',
                "",
            )
        ),
        encoding="utf-8",
    )
    config = {
        "run_id": "pp-123456-1",
        "candidate_sha": CANDIDATE,
        "expected_instance_id": "i-0123456789abcdef0",
        "runtime_root": str(runtime),
    }
    process = {
        "name": "validator_application",
        "cmdline_hash": "sha256:" + "5" * 64,
        "private": SECRET,
    }
    monkeypatch.setattr(native, "load_config", lambda _path: config)
    monkeypatch.setattr(
        native, "_load_process_state", lambda _config: {"processes": [process]}
    )
    monkeypatch.setattr(native, "_same_process", lambda _process: True)
    monkeypatch.setattr(reader, "CONFIG", tmp_path / "config.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--run-id",
            "pp-123456-1",
            "--candidate-sha",
            CANDIDATE,
            "--instance-id",
            "i-0123456789abcdef0",
            "--epoch-id",
            str(EPOCH),
        ],
    )
    reader.main()
    return capsys.readouterr().out


def test_returns_only_exact_public_markers(tmp_path, monkeypatch, capsys):
    stdout = _run(tmp_path, monkeypatch, capsys)
    value = json.loads(stdout)
    assert value["epoch_id"] == EPOCH
    assert value["marker_line_numbers"] == [2, 3, 4, 5, 6, 7]
    assert value["weight_submission_event_hash_prefix"] == "sha256:" + "3" * 13
    assert value["weight_finalization_event_hash_prefix"] == "sha256:" + "4" * 13
    assert SECRET not in stdout


def test_rejects_wrong_or_missing_epoch(tmp_path, monkeypatch, capsys):
    with pytest.raises(RuntimeError, match="exact epoch markers unavailable"):
        _run(tmp_path, monkeypatch, capsys, logged_epoch=EPOCH - 1)
