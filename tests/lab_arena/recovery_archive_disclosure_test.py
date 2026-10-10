"""Operational evidence archives never disclose another round's bank."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import icp_disclosure
from lab_arena.service import ArenaService, ServiceError


ROUND_ID = "arena-2026-09-13"
PUBLIC_AT = datetime(2026, 9, 13, tzinfo=timezone.utc)


def _service(*, status="cancelled", mode="live", reason="capacity:stage1:20",
             source_round_id=None, embedded_round_id=ROUND_ID):
    configuration = {
        "mode": mode,
        "benchmark_disclosure_policy": icp_disclosure.CUTOFF_PUBLIC_POLICY,
        "schedule": {
            "submission_open": "2026-09-12T00:00:00Z",
            "submission_cutoff": "2026-09-13T00:00:00Z",
        },
    }
    if source_round_id is not None:
        configuration["recovery_source_round_id"] = source_round_id
    row = {
        "round_id": ROUND_ID,
        "status": status,
        "cancel_reason": reason,
        "configuration_doc": configuration,
        "benchmark_ref": "arena/immutable-bank.json",
        "icp_set_date": "2026-09-12",
        "evaluation_date": "2026-09-13",
        "participants": [],
    }
    reads = []

    def get(ref):
        reads.append(ref)
        return json.dumps({
            "schema_version": "leadpoet.lab_arena.benchmark.v1",
            "round_id": embedded_round_id,
            "icps": [{"icp_id": "icp-%02d" % n} for n in range(20)],
        }).encode()

    service = object.__new__(ArenaService)
    service._round = lambda _round_id: row
    service._store = SimpleNamespace()
    service._objects = SimpleNamespace(get=get)
    service._clock = lambda: PUBLIC_AT
    return service, row, reads


@pytest.mark.parametrize("mode", ["live", "shadow"])
@pytest.mark.parametrize("status", ["published", "cancelled", "stage2_closed"])
def test_real_round_benchmark_disclosure_remains_available(mode, status):
    service, row, reads = _service(mode=mode, status=status)

    result = service.public_benchmark(ROUND_ID)

    assert len(result["icps"]) == 20
    assert reads == [row["benchmark_ref"]]


@pytest.mark.parametrize("mode", ["live", "shadow"])
@pytest.mark.parametrize("reason,source_round_id", [
    ("authorized_baseline_evidence_archive", None),
    ("authorized_invalid_reward_archive340", None),
    ("authorized_baseline_recovery440", "arena-original"),
    ("authorized_uniform_rejudge445_recovery_archive", "arena-original"),
])
def test_archive_is_not_public_and_never_reads_original_bank(mode, reason, source_round_id):
    service, row, reads = _service(
        mode=mode, reason=reason, source_round_id=source_round_id,
        embedded_round_id="arena-original",
    )

    assert icp_disclosure.disclosure_metadata(row) is None
    assert icp_disclosure.source_public_at(row) is None
    with pytest.raises(ServiceError, match="benchmark_not_public") as error:
        service.public_benchmark(ROUND_ID)

    assert error.value.status == 403
    assert reads == []


@pytest.mark.parametrize("reason,source_round_id,status", [
    ("capacity:stage1:20", "arena-original", "cancelled"),
    ("authorized_recovery", ROUND_ID, "cancelled"),
    ("authorized_recovery", "", "cancelled"),
    ("authorized_recovery", "  ", "cancelled"),
    ("authorized_recovery", 123, "cancelled"),
    ("authorized_recovery", None, "cancelled"),
    ("authorized_recovery", "arena-original", "published"),
    ("authorized_recovery_archive", None, "published"),
])
def test_archive_filter_requires_server_authored_cancelled_archive(reason, source_round_id, status):
    service, row, _ = _service(
        reason=reason, source_round_id=source_round_id, status=status,
    )

    assert not icp_disclosure.is_administrative_archive(row)
    assert len(service.public_benchmark(ROUND_ID)["icps"]) == 20


@pytest.mark.parametrize("status", ["published", "cancelled", "stage2_closed"])
def test_real_round_with_wrong_bank_identity_still_fails_closed(status):
    service, _, reads = _service(status=status, embedded_round_id="arena-unrelated")

    with pytest.raises(ServiceError, match="benchmark_data_invalid") as error:
        service.public_benchmark(ROUND_ID)

    assert error.value.status == 500
    assert len(reads) == 1
