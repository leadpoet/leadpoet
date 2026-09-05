"""A cutoff crossed during live key validation is a conflict, not a server error."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena.service import ArenaService, ServiceError


def test_finalize_reports_a_database_cutoff_race_without_accepting():
    row = {
        "submission_id": "sub-cutoff-test",
        "round_id": "arena-2026-09-05",
        "miner_hotkey": "5" + "A" * 47,
        "source_ref": "arena/arena-2026-09-05/sources/sub-cutoff-test.tar.gz",
        "source_size_bytes": 100,
        "status": "uploading",
    }
    calls = []
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 9, 5, 5, 59, 59, tzinfo=timezone.utc)
    service._store = SimpleNamespace(
        get_submission=lambda _id: row,
        accept_submission_with_credentials=lambda *args: calls.append(args) or {"status": "window_closed"},
    )
    service._validate_uploaded_source = lambda *args, **kwargs: None
    service._config = SimpleNamespace(credential_manager=SimpleNamespace(
        validate_and_encrypt=lambda *args, **kwargs: {"openrouter": "encrypted-or", "deepline": "encrypted-dl"},
    ))
    service._request_round = lambda *args, **kwargs: (
        {"hotkey": row["miner_hotkey"], "body": {
            "submission_id": row["submission_id"], "source_ref": row["source_ref"],
            "source_size_bytes": row["source_size_bytes"], "credentials": {
                "openrouter_api_key": "sk-or-v1-" + "a" * 32,
                "openrouter_management_key": "sk-or-v1-" + "b" * 32,
                "deepline_api_key": "deepline-" + "c" * 32,
            },
        }},
        {"round_id": row["round_id"], "status": "open", "configuration_doc": {
            "schedule": {"submission_open": "2026-09-04T06:00:00Z", "submission_cutoff": "2026-09-05T06:00:00Z"},
        }},
    )
    with pytest.raises(ServiceError, match="submission_window_closed") as error:
        service.handle_submission_finalize(row["submission_id"], {})
    assert error.value.status == 409
    assert len(calls) == 1
    assert row["status"] == "uploading"
