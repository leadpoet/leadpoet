import pytest

from gateway.research_lab import attested_coordinator_v2
from gateway.tee.coordinator_executor_v2 import COORDINATOR_OPERATIONS_V2


@pytest.mark.asyncio
async def test_coordinator_bridge_uses_strict_coordinator_role(monkeypatch):
    observed = {}

    async def execute(**kwargs):
        observed.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(attested_coordinator_v2, "execute_scoring_v2", execute)
    result = await attested_coordinator_v2.execute_coordinator_v2(
        operation="promotion_improvement",
        purpose="research_lab.ranking.v2",
        epoch_id=9,
        sequence=1,
        payload={"score_bundle": {}},
        client=object(),
    )
    assert result == {"status": "succeeded"}
    assert observed["operation_registry"] == COORDINATOR_OPERATIONS_V2
    assert observed["physical_role_override"] == "gateway_coordinator"
    assert observed["expected_service_role"] == "gateway_coordinator"
    assert observed["rpc_namespace"] == "coordinator_v2"
