from __future__ import annotations

import pytest

from gateway.tee.provider_broker_v2 import (
    expected_job_credential_slot_ref_hashes,
    expected_provider_credential_slots,
    provider_registry_hash,
)
from gateway.tee.topology import ROLE_SPECS
from gateway.tee.verify_v2_runtime_ready import (
    V2RuntimeReadinessError,
    verify_v2_runtime_ready,
)


class _Client:
    def __init__(self, role: str):
        self.role = role

    async def v2_provider_broker_health(self):
        return {
            "status": "ready",
            "credential_slots": list(expected_provider_credential_slots()),
            "missing_credential_slots": [],
            "registry_hash": provider_registry_hash(),
            "job_credential_slot_ref_hashes": (
                expected_job_credential_slot_ref_hashes()
            ),
        }

    async def v2_provider_semantics_health(self):
        return {
            "status": "ready",
            "broker_registry_hash": provider_registry_hash(),
            "memory_cache_entry_count": 0,
            "inflight_count": 0,
            "cost_scope_count": 0,
        }

    def _health(self, workers):
        configured_workers = {
            "gateway_coordinator": 0,
            "gateway_scoring": 25,
            "gateway_autoresearch": 10,
        }[self.role]
        return {
            "authority": "v2_only",
            "physical_role": self.role,
            "role": ROLE_SPECS[self.role]["service_role"],
            "worker_count": workers,
            "configured_worker_count": configured_workers,
            "workers_alive": True,
            "boot_identity_hash": "sha256:" + "a" * 64,
        }

    async def coordinator_v2_health(self):
        return self._health(1)

    async def scoring_v2_health(self):
        return self._health(10)

    async def autoresearch_v2_health(self):
        return self._health(10)


@pytest.mark.asyncio
async def test_runtime_ready_requires_every_manager_and_provider_slot():
    clients = {role: _Client(role) for role in ROLE_SPECS}
    result = await verify_v2_runtime_ready(clients)
    assert result["status"] == "ready"
    assert len(result["roles"]) == 3


@pytest.mark.asyncio
async def test_runtime_ready_fails_when_shared_scoring_runner_is_dead():
    clients = {role: _Client(role) for role in ROLE_SPECS}

    async def dead():
        value = clients["gateway_scoring"]._health(10)
        value["workers_alive"] = False
        return value

    clients["gateway_scoring"].scoring_v2_health = dead
    with pytest.raises(V2RuntimeReadinessError, match="gateway_scoring"):
        await verify_v2_runtime_ready(clients)


@pytest.mark.asyncio
async def test_runtime_ready_fails_when_provider_semantics_is_not_ready():
    clients = {role: _Client(role) for role in ROLE_SPECS}

    async def unavailable():
        return {
            "status": "provisioning",
            "broker_registry_hash": provider_registry_hash(),
            "memory_cache_entry_count": 0,
            "inflight_count": 0,
            "cost_scope_count": 0,
        }

    clients["gateway_coordinator"].v2_provider_semantics_health = unavailable
    with pytest.raises(V2RuntimeReadinessError, match="semantics"):
        await verify_v2_runtime_ready(clients)
