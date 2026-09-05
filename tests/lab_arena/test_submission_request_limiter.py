from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gateway.middleware.priority import classify_path
from lab_arena import submission_rate_limit
from lab_arena.service import ArenaService, ServiceError


def test_submission_routes_use_the_miner_priority_pool_only():
    assert classify_path("/arena/v1/submissions/presign") == "miner"
    assert classify_path("/arena/v1/submissions/sub-1/finalize") == "miner"
    assert classify_path("/arena/v1/submissions/sub-1") == "miner"
    assert classify_path("/arena/v1/runs/claim") == "other"
    assert classify_path("/arena/v1/rounds/arena-1") == "other"


def test_limiter_is_shared_per_hotkey_and_recovers_after_the_window():
    now = [100.0]
    limiter = submission_rate_limit.SubmissionRequestLimiter(clock=lambda: now[0])

    for _ in range(6):
        assert limiter.check("miner-a").allowed is True
    denied = limiter.check("miner-a")
    assert denied.allowed is False
    assert denied.retry_after_seconds == 60
    assert limiter.check("miner-b").allowed is True

    now[0] += 60.0
    assert limiter.check("miner-a").allowed is True


def test_limiter_keeps_a_bounded_identity_map():
    limiter = submission_rate_limit.SubmissionRequestLimiter(max_identities=2)
    assert limiter.check("miner-a").allowed is True
    assert limiter.check("miner-b").allowed is True
    assert limiter.check("miner-c").allowed is True
    assert list(limiter._entries) == ["miner-b", "miner-c"]


def test_presign_stops_before_storage_work_when_hotkey_limit_is_full():
    hotkey = "5" + "A" * 47
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 9, 2, 0, 30, tzinfo=timezone.utc)
    service._config = SimpleNamespace(
        chain=SimpleNamespace(uid_for_hotkey=lambda value: 1 if value == hotkey else None)
    )
    service._request_round = lambda *_args, **_kwargs: (
        {
            "hotkey": hotkey,
            "body": {
                "source_size_bytes": 10,
                "consent": {"public_rerun": True},
            },
        },
        {
            "round_id": "arena-2026-09-02",
            "status": "open",
            "configuration_doc": {
                "schedule": {
                    "submission_open": "2026-09-02T00:00:00Z",
                    "submission_cutoff": "2026-09-02T01:00:00Z",
                },
                "baseline_hotkey": "5" + "Z" * 47,
            },
        },
    )
    service._submission_request_limiter = submission_rate_limit.SubmissionRequestLimiter(
        limit=1
    )
    service._submission_request_limiter.check(hotkey)
    service._store = SimpleNamespace(
        register_submission=lambda *_args: pytest.fail("storage must not run")
    )

    with pytest.raises(ServiceError) as exc:
        service.handle_submission_presign({})
    assert exc.value.status == 429
    assert exc.value.code == "submission_rate_limited"


def test_accepted_finalize_idempotency_does_not_consume_the_limit():
    hotkey = "5" + "A" * 47
    submission_id = "sub-1"
    body = {
        "submission_id": submission_id,
        "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
        "source_size_bytes": 10,
        "credentials": {
            "openrouter_api_key": "o" * 16,
            "openrouter_management_key": "m" * 16,
            "deepline_api_key": "d" * 16,
        },
    }
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 9, 2, 0, 30, tzinfo=timezone.utc)
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": hotkey, "body": body},
        {
            "round_id": "arena-2026-09-02",
            "status": "open",
            "configuration_doc": {
                "schedule": {
                    "submission_open": "2026-09-02T00:00:00Z",
                    "submission_cutoff": "2026-09-02T01:00:00Z",
                }
            },
        },
    )
    service._store = SimpleNamespace(
        get_submission=lambda _submission_id: {
            **body,
            "round_id": "arena-2026-09-02",
            "miner_hotkey": hotkey,
            "status": "accepted",
        },
        get_submission_credential=lambda *_args: {"ciphertext_b64": "encrypted"},
    )
    service._submission_request_limiter = submission_rate_limit.SubmissionRequestLimiter(
        limit=1
    )
    service._submission_request_limiter.check(hotkey)

    assert service.handle_submission_finalize(submission_id, {}) == {
        "status": "accepted",
        "submission_id": submission_id,
    }


def test_finalize_stops_before_source_and_credential_work_when_limit_is_full():
    hotkey = "5" + "A" * 47
    submission_id = "sub-1"
    body = {
        "submission_id": submission_id,
        "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
        "source_size_bytes": 10,
        "credentials": {
            "openrouter_api_key": "o" * 16,
            "openrouter_management_key": "m" * 16,
            "deepline_api_key": "d" * 16,
        },
    }
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 9, 2, 0, 30, tzinfo=timezone.utc)
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": hotkey, "body": body},
        {
            "round_id": "arena-2026-09-02",
            "status": "open",
            "configuration_doc": {
                "schedule": {
                    "submission_open": "2026-09-02T00:00:00Z",
                    "submission_cutoff": "2026-09-02T01:00:00Z",
                }
            },
        },
    )
    service._store = SimpleNamespace(
        get_submission=lambda _submission_id: {
            **body,
            "round_id": "arena-2026-09-02",
            "miner_hotkey": hotkey,
            "status": "uploading",
        }
    )
    service._objects = SimpleNamespace(
        get_bounded=lambda *_args: pytest.fail("source read must not run")
    )
    service._submission_request_limiter = submission_rate_limit.SubmissionRequestLimiter(
        limit=1
    )
    service._submission_request_limiter.check(hotkey)

    with pytest.raises(ServiceError) as exc:
        service.handle_submission_finalize(submission_id, {})
    assert exc.value.status == 429
    assert exc.value.code == "submission_rate_limited"
