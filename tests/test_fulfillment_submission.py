"""Retained fulfillment adapter checks independent of retired validators."""
from types import SimpleNamespace
import pytest
import requests
from Leadpoet.utils import cloud_db
from gateway.fulfillment.models import FulfillmentScoreResult
from qualification.scoring.fulfillment_scorer import format_scores_for_gateway


def test_score_formatter_rejects_cardinality_mismatch() -> None:
    with pytest.raises(ValueError, match="cardinality mismatch"):
        format_scores_for_gateway(
            "miner-1",
            ["lead-1", "lead-2"],
            [FulfillmentScoreResult()],
            request_id="request-1",
            submission_id="submission-1",
        )


def test_gateway_score_submit_raises_after_exactly_three_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = []
    sleeps = []

    def fail_post(*_args, **_kwargs):
        attempts.append(1)
        raise requests.ConnectionError("gateway unavailable")

    monkeypatch.setattr(cloud_db.requests, "post", fail_post)
    monkeypatch.setattr(cloud_db.time, "sleep", sleeps.append)
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda _message: b"signature",
        )
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        cloud_db.gateway_submit_fulfillment_scores(
            wallet,
            "43c6bc55-80f9-49e3-af0c-7a6ae6e39358",
            [{"lead_id": "lead-1"}],
        )

    assert len(attempts) == 3
    assert sleeps == [2, 2]
