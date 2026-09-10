from __future__ import annotations

import pytest

from gateway.tasks import hourly_batch


class _FakeTeeClient:
    def __init__(self) -> None:
        self.acknowledged = None

    async def get_buffer_stats(self):
        return {
            "size": 1,
            "age_seconds": 5,
            "sequence_range": {"first": 10, "last": 10},
        }

    async def build_checkpoint(self):
        return {
            "status": "success",
            "header": {
                "checkpoint_number": 42,
                "event_count": 1,
                "merkle_root": "a" * 64,
                "time_range": {
                    "start": "2026-07-19T00:00:00Z",
                    "end": "2026-07-19T00:00:01Z",
                },
                "sequence_range": {"first": 10, "last": 10},
            },
            "signature": "signature",
            "events": [{"event_type": "QUALIFICATION_RESULT"}],
            "tree_levels": [["a" * 64]],
        }

    async def acknowledge_checkpoint(self, **kwargs):
        self.acknowledged = dict(kwargs)
        return {
            "status": "acknowledged",
            "removed_count": 1,
            "remaining_count": 0,
        }


@pytest.mark.asyncio
async def test_immediate_batch_uses_confirmed_production_path(monkeypatch):
    fake_tee = _FakeTeeClient()
    observed = {}

    async def fake_balance():
        return 1.0


    async def fake_upload(**kwargs):
        observed["upload"] = dict(kwargs)
        return "x" * 43

    def fake_payload(**kwargs):
        observed["payload"] = dict(kwargs)
        return b"exact-checkpoint-payload"

    async def fake_confirmation(tx_id, **kwargs):
        observed["confirmation"] = {
            "tx_id": tx_id,
            **kwargs,
        }
        return True


    async def fake_log(event):
        observed["log"] = dict(event)
        return {"sequence": 11}

    async def fail_sleep(seconds):
        raise AssertionError(f"immediate batch slept for {seconds}s")

    monkeypatch.setattr(hourly_batch, "tee_client", fake_tee)
    monkeypatch.setattr(hourly_batch, "get_wallet_balance", fake_balance)
    monkeypatch.setattr(hourly_batch, "upload_checkpoint", fake_upload)
    monkeypatch.setattr(hourly_batch, "checkpoint_payload_bytes", fake_payload)
    monkeypatch.setattr(hourly_batch, "wait_for_confirmation", fake_confirmation)
    monkeypatch.setattr(hourly_batch, "log_event", fake_log)
    monkeypatch.setattr(hourly_batch.asyncio, "sleep", fail_sleep)

    result = await hourly_batch.hourly_batch_task(
        run_immediately=True,
        max_batches=1,
    )

    assert result == {
        "ok": True,
        "status": "checkpointed",
        "batch_count": 1,
        "checkpoint_number": 42,
        "event_count": 1,
        "arweave_tx_id": "x" * 43,
    }
    assert observed["confirmation"] == {
        "tx_id": "x" * 43,
        "expected_payload": b"exact-checkpoint-payload",
        "timeout": hourly_batch.ARWEAVE_CONFIRMATION_TIMEOUT_SECONDS,
    }
    assert fake_tee.acknowledged == {
        "checkpoint_number": 42,
        "merkle_root": "a" * 64,
        "sequence_range": {"first": 10, "last": 10},
    }
