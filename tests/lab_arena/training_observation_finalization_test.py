"""Late capture never grants execution authority or changes run accounting."""

import copy
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from lab_arena import training_observations
from lab_arena.api import create_app
from lab_arena.service import ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for
from tests.lab_arena.training_observations_test import observation, payload


ROUTE = "/arena/v1/runs/r1/training-observations"
HEADERS = {"x-lab-arena-lease": TOKEN}


@pytest.mark.parametrize("status,cause", [("leased", None), ("failed", "lease_expired")])
@pytest.mark.parametrize("seconds", [0, 0.2, 179.999])
def test_final_snapshot_survives_expiry_without_changing_run(tmp_path, status, cause, seconds):
    service, objects = service_for(
        tmp_path, status=status, terminal_cause=cause,
        lease_expires_at=(NOW - timedelta(seconds=seconds)).isoformat(),
    )
    original = copy.deepcopy(service._store.get_run("r1"))
    client = TestClient(create_app(service))
    document = payload(observation())
    assert client.head(ROUTE, headers=HEADERS).status_code == 204
    result = client.post(ROUTE, headers=HEADERS, json=document)
    assert result.status_code == 200
    assert client.post(ROUTE, headers=HEADERS, json=document).json() == result.json()
    assert client.post(ROUTE, headers=HEADERS, json=payload()).status_code == 409
    assert training_observations.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"] == document
    assert service._store.get_run("r1") == original
    # The input endpoint and default authorization remain active-lease-only.
    with pytest.raises(ServiceError):
        service.handle_training_input("r1", TOKEN, {"late": "input"})
    with pytest.raises(ServiceError):
        service._training_capture_run("r1", TOKEN)


@pytest.mark.parametrize("changes,token,status", [
    ({"lease_expires_at": (NOW - timedelta(seconds=180)).isoformat()}, TOKEN, 409),
    ({"lease_expires_at": (NOW - timedelta(seconds=181)).isoformat()}, TOKEN, 409),
    ({"status": "accepted"}, TOKEN, 409),
    ({"status": "pending"}, TOKEN, 409),
    ({"status": "failed", "terminal_cause": "model_error"}, TOKEN, 409),
    ({"status": "failed", "terminal_cause": "lease_expired",
      "lease_expires_at": (NOW + timedelta(seconds=1)).isoformat()}, TOKEN, 409),
    ({}, "b" * 64, 401),
    ({"lease_generation": 0}, TOKEN, 401),
    ({"claim_request_hash": "invalid"}, TOKEN, 401),
    ({"lease_generation": 2, "claim_request_hash": "sha256:" + "d" * 64,
      "lease_token_hash": hash_lease_token("b" * 64)}, TOKEN, 401),
])
def test_grace_rejects_wrong_authority_before_reading_body(tmp_path, changes, token, status):
    values = {"lease_expires_at": (NOW - timedelta(seconds=1)).isoformat(), **changes}
    service, objects = service_for(tmp_path, **values)
    client = TestClient(create_app(service))
    headers = {"x-lab-arena-lease": token}
    assert client.head(ROUTE, headers=headers).status_code == status
    # Invalid JSON would return 400 if the unauthorized body were parsed.
    assert client.post(ROUTE, headers=headers, content=b"not JSON").status_code == status
    assert not list(tmp_path.rglob("*.gz"))


def test_grace_still_requires_round_ownership(tmp_path):
    service, _ = service_for(tmp_path, lease_expires_at=NOW.isoformat())
    def reject(_):
        raise ServiceError("round_not_owned", 403)
    service._require_round_ownership = reject
    client = TestClient(create_app(service))
    assert client.head(ROUTE, headers=HEADERS).status_code == 403
    assert client.post(ROUTE, headers=HEADERS, content=b"not JSON").status_code == 403
    assert not list(tmp_path.rglob("*.gz"))


@pytest.mark.parametrize("change", ["token", "status", "time"])
def test_handler_rechecks_after_body_authorization(tmp_path, change):
    service, _ = service_for(tmp_path, lease_expires_at=NOW.isoformat())
    original = service._training_capture_run
    count = 0
    def authorize(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == 1:
            if change == "token":
                result["lease_token_hash"] = hash_lease_token("b" * 64)
                result["lease_generation"] = 2
            elif change == "status":
                result["status"] = "accepted"
            else:
                service._clock = lambda: NOW + timedelta(seconds=180)
        return result
    service._training_capture_run = authorize
    result = TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json=payload())
    assert result.status_code == (401 if change == "token" else 409)
    assert not list(tmp_path.rglob("*.gz"))


@pytest.mark.parametrize("current_delta,claim_delta,expected", [
    (-1, -3600, 200),  # Provider activity extended the effective expiry.
    (-181, 3600, 409),  # Early recovery shortened it; the old claim cannot extend grace.
])
def test_finalization_uses_current_effective_expiry(tmp_path, current_delta, claim_delta, expected):
    service, _ = service_for(
        tmp_path, status="failed", terminal_cause="lease_expired",
        lease_expires_at=(NOW + timedelta(seconds=current_delta)).isoformat(),
        claim_response={"lease_expires_at": (NOW + timedelta(seconds=claim_delta)).isoformat()},
    )
    result = TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json=payload())
    assert result.status_code == expected
