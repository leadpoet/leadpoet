from __future__ import annotations

from leadpoet_canonical.attested_v2 import sha256_json

import pytest

from gateway.research_lab import v2_authority
from gateway.research_lab.attested_scoring_v2 import (
    derive_execution_job_id_v2,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


@pytest.mark.asyncio
async def test_provider_preflight_uses_unique_measured_jobs_and_dedicated_profile(
    monkeypatch,
):
    calls = []
    job_ids = []
    measurement_ids = iter(("1" * 32, "2" * 32))

    async def execute(**kwargs):
        calls.append(kwargs)
        job_ids.append(
            derive_execution_job_id_v2(
                operation=kwargs["operation"],
                purpose=kwargs["purpose"],
                epoch_id=kwargs["epoch_id"],
                sequence=kwargs["sequence"],
                payload_sha256=sha256_json(kwargs["payload"]),
                parent_receipt_hashes=(),
                input_artifact_hashes=(),
                release_hash=HASH_B,
                physical_role="gateway_scoring",
            )
        )
        return {"result": {"healthy": True, "pause_worthy": False, "verdicts": []}}

    monkeypatch.setattr(
        v2_authority.uuid,
        "uuid4",
        lambda: type("MeasurementId", (), {"hex": next(measurement_ids)})(),
    )
    for _ in range(2):
        result = await v2_authority.execute_provider_preflight_v2(
            scope_key="scoring:worker-4",
            worker_index=4,
            settings={
                "enabled": True,
                "ttl_seconds": 600.0,
                "timeout_seconds": 12.0,
                "failure_streak_threshold": 3,
            },
            execute=execute,
        )
        assert result["healthy"] is True

    assert calls[0]["purpose"] == "research_lab.provider_preflight.v2"
    assert [call["sequence"] for call in calls] == [0, 0]
    assert [call["payload"]["measurement_id"] for call in calls] == [
        "1" * 32,
        "2" * 32,
    ]
    assert len(set(job_ids)) == 2
    assert all(call["worker_index"] == 4 for call in calls)
    assert all(
        call["provider_credential_profile"] == "provider_preflight"
        for call in calls
    )
