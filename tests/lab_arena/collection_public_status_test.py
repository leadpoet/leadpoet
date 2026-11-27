from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ArenaService


def make_service(work, mode="live", quotes=True):
    config = SimpleNamespace(store=SimpleNamespace(_transport=Mock()), object_store=Mock(),
                             signer=Mock(), clock=lambda: datetime.now(timezone.utc),
                             chain=SimpleNamespace(current_settlement_epoch=lambda: 200))
    svc = ArenaService(config)
    svc.collection_mode = lambda: mode
    svc.collection_terms = lambda: {"offer_id": "dv-" + "a" * 64, "fixed_quotes": [1] if quotes else []}
    svc._collection_reward_active = lambda epoch: epoch == 200
    config.store._transport.rpc.return_value = {"work": work, "billing": "private", "enrollments": "private"}
    return svc


def test_dashboard_reads_only_live_aggregates():
    work = [
        {"mode": "pilot", "arm": "comparison", "accepted": 999},
        {"mode": "live", "arm": "comparison", "assigned": 12, "accepted": 5,
         "failed": 1, "rejected": 1, "ready": 1, "expired": 2},
        {"mode": "live", "arm": "correction", "assigned": 1, "accepted": 1,
         "failed": 0, "rejected": 0, "ready": 0, "expired": 0},
    ]
    svc = make_service(work)
    with TestClient(create_app(svc)) as client:
        reply = client.get("/arena/v1/collection/status")
        assert reply.status_code == 200
        data = reply.json()
        assert datetime.fromisoformat(data.pop("as_of")).utcoffset().total_seconds() == 0
        assert data == {"paid_enrollment_open": True, "champion_rewards_off": True,
                        "running_runs": 2, "verified_runs": 6}
        client.get("/arena/v1/collection/status").raise_for_status()
    svc.store._transport.rpc.assert_called_once_with("collection_participation_v1", {})


def test_empty_closed_and_malformed_data_are_distinct():
    for mode in ("pilot", "live"):
        svc = make_service([], mode, quotes=False)
        with TestClient(create_app(svc)) as client:
            status = client.get("/arena/v1/collection/status").json()
            assert status["paid_enrollment_open"] is False
            assert status["running_runs"] == status["verified_runs"] == 0
    for broken in (None, [{"mode": "live"}], [{"mode": "live", "assigned": 0, "accepted": 1,
         "expired": 0, "failed": 0, "ready": 0, "rejected": 0}]):
        svc = make_service(broken)
        with TestClient(create_app(svc)) as client:
            assert client.get("/arena/v1/collection/status").status_code == 503


def test_status_rechecks_retirement_and_never_serves_freshness_after_expiry(monkeypatch):
    svc = make_service([])
    tick = [0]
    monkeypatch.setattr("lab_arena.service.time.monotonic", lambda: tick[0])
    svc._collection_reward_active = lambda epoch: False
    assert svc.public_collection_status()["champion_rewards_off"] is False
    tick[0] = 31
    svc._collection_reward_active = lambda epoch: True
    assert svc.public_collection_status()["champion_rewards_off"] is True
    svc.collection_mode = lambda: "off"
    with TestClient(create_app(svc)) as client:
        assert client.get("/arena/v1/collection/status").status_code == 404
