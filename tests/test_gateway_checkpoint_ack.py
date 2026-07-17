from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "gateway" / "tee"),
)

from gateway.tee import tee_service


def _reset_checkpoint_state(monkeypatch) -> None:
    tee_service.event_buffer.clear()
    monkeypatch.setattr(tee_service, "pending_checkpoint", None)
    monkeypatch.setattr(tee_service, "prev_checkpoint_root", None)
    monkeypatch.setattr(tee_service, "checkpoint_count", 0)
    monkeypatch.setattr(
        tee_service,
        "compute_merkle_tree",
        lambda events: (b"\x22" * 32, [[b"\x22" * 32]]),
    )
    monkeypatch.setattr(
        tee_service,
        "compute_code_hash",
        lambda: "sha256:" + "3" * 64,
    )
    monkeypatch.setattr(
        tee_service,
        "get_cached_attestation_hash",
        lambda: "sha256:" + "4" * 64,
    )
    monkeypatch.setattr(
        tee_service,
        "sign_data",
        lambda _payload: b"\x55" * 64,
    )


def test_checkpoint_ack_removes_only_the_signed_prefix(monkeypatch):
    _reset_checkpoint_state(monkeypatch)
    tee_service.event_buffer.extend(
        [
            {"sequence": 1, "event_hash": "a"},
            {"sequence": 2, "event_hash": "b"},
        ]
    )

    first = tee_service.build_checkpoint()
    tee_service.event_buffer.append({"sequence": 3, "event_hash": "c"})
    repeated = tee_service.build_checkpoint()

    assert first == repeated
    assert first["header"]["checkpoint_number"] == 0
    assert first["header"]["sequence_range"] == {"first": 1, "last": 2}
    assert tee_service.checkpoint_count == 0

    result = tee_service.acknowledge_checkpoint(
        checkpoint_number=0,
        merkle_root=first["header"]["merkle_root"],
        sequence_range=first["header"]["sequence_range"],
    )

    assert result["status"] == "acknowledged"
    assert result["removed_count"] == 2
    assert result["remaining_count"] == 1
    assert tee_service.event_buffer == [{"sequence": 3, "event_hash": "c"}]
    assert tee_service.checkpoint_count == 1
    assert tee_service.prev_checkpoint_root == b"\x22" * 32


def test_checkpoint_ack_rejects_wrong_root_without_data_loss(monkeypatch):
    _reset_checkpoint_state(monkeypatch)
    tee_service.event_buffer.append({"sequence": 7, "event_hash": "a"})
    checkpoint = tee_service.build_checkpoint()

    with pytest.raises(ValueError, match="differs"):
        tee_service.acknowledge_checkpoint(
            checkpoint_number=0,
            merkle_root="f" * 64,
            sequence_range=checkpoint["header"]["sequence_range"],
        )

    assert tee_service.event_buffer == [{"sequence": 7, "event_hash": "a"}]
    assert tee_service.pending_checkpoint == checkpoint
    assert tee_service.clear_buffer()["status"] == "rejected"
