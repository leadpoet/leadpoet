from __future__ import annotations

import json
from pathlib import Path

import pytest

from Leadpoet.utils.subnet_epoch import CUTOVER_JSON_ENV, SubnetEpochCutover
from gateway.tee.provider_broker_v2 import (
    expected_job_credential_slot_ref_hashes,
    expected_provider_credential_slots,
    provider_registry_hash,
)
from gateway.tee.research_lab_runtime_config_v2 import (
    ResearchLabRuntimeConfigV2Error,
    build_research_lab_execution_config,
)
from gateway.tee.topology import ROLE_SPECS
from gateway.tee.verify_v2_runtime_ready import (
    V2RuntimeReadinessError,
    verify_v2_runtime_ready,
)
from tests.v2_epoch_test_utils import epoch_test_environment


class _Client:
    def __init__(self, role: str, registry_hash=None):
        self.role = role
        self.registry_hash = registry_hash or provider_registry_hash()

    async def v2_provider_broker_health(self):
        return {
            "status": "ready",
            "credential_slots": list(expected_provider_credential_slots()),
            "missing_credential_slots": [],
            "registry_hash": self.registry_hash,
            "job_credential_slot_ref_hashes": (
                expected_job_credential_slot_ref_hashes()
            ),
        }

    async def v2_provider_semantics_health(self):
        return {
            "status": "ready",
            "broker_registry_hash": self.registry_hash,
            "memory_cache_entry_count": 0,
            "inflight_count": 0,
            "cost_scope_count": 0,
        }

    def _health(self, workers):
        configured_workers = {
            "gateway_coordinator": 0,
            "gateway_scoring": 25,
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

@pytest.mark.asyncio
async def test_runtime_ready_requires_every_manager_and_provider_slot():
    clients = {role: _Client(role) for role in ROLE_SPECS}
    result = await verify_v2_runtime_ready(clients)
    assert result["status"] == "ready"
    assert len(result["roles"]) == len(ROLE_SPECS) == 2


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


@pytest.mark.asyncio
async def test_runtime_ready_rejects_tampered_provider_registry_hash():
    clients = {role: _Client(role) for role in ROLE_SPECS}

    async def tampered_provider_policy():
        value = await _Client("gateway_coordinator").v2_provider_broker_health()
        value["registry_hash"] = "sha256:" + "f" * 64
        return value

    clients[
        "gateway_coordinator"
    ].v2_provider_broker_health = tampered_provider_policy
    with pytest.raises(V2RuntimeReadinessError, match="provider broker"):
        await verify_v2_runtime_ready(clients)


def _testnet401_execution_config():
    cutover = SubnetEpochCutover(
        network_genesis_hash=(
            "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        ),
        netuid=401,
        cutover_block=7_955_391,
        cutover_block_hash="0x" + "4" * 64,
        first_subnet_epoch_index=22_041,
        first_settlement_epoch_id=22_042,
        last_legacy_epoch_id=22_041,
    )
    return build_research_lab_execution_config(environment={
        "BITTENSOR_NETWORK": "test",
        "BITTENSOR_NETUID": "401",
        CUTOVER_JSON_ENV: json.dumps(cutover.to_dict()),
    })


@pytest.mark.asyncio
async def test_runtime_ready_uses_validated_testnet401_provider_registry():
    execution_config = _testnet401_execution_config()
    expected = provider_registry_hash(execution_config=execution_config)
    assert expected != provider_registry_hash()
    clients = {role: _Client(role, expected) for role in ROLE_SPECS}

    result = await verify_v2_runtime_ready(
        clients, execution_config=execution_config
    )

    assert result["provider_registry_hash"] == expected


@pytest.mark.asyncio
async def test_explicit_finney_execution_config_preserves_default_registry():
    execution_config = build_research_lab_execution_config(
        environment=epoch_test_environment()
    )
    assert provider_registry_hash(
        execution_config=execution_config
    ) == provider_registry_hash()
    clients = {role: _Client(role) for role in ROLE_SPECS}

    result = await verify_v2_runtime_ready(
        clients, execution_config=execution_config
    )

    assert result["provider_registry_hash"] == provider_registry_hash()


@pytest.mark.asyncio
async def test_runtime_ready_rejects_finney_registry_for_testnet401():
    clients = {role: _Client(role) for role in ROLE_SPECS}
    with pytest.raises(V2RuntimeReadinessError, match="provider broker"):
        await verify_v2_runtime_ready(
            clients, execution_config=_testnet401_execution_config()
        )


@pytest.mark.asyncio
async def test_runtime_ready_rejects_malformed_execution_config():
    clients = {role: _Client(role) for role in ROLE_SPECS}
    with pytest.raises(ResearchLabRuntimeConfigV2Error):
        await verify_v2_runtime_ready(
            clients, execution_config={"schema_version": "invalid"}
        )


def test_gateway_v2_health_uses_its_validated_execution_config():
    source = (Path(__file__).parents[1] / "gateway" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "execution_config=build_research_lab_execution_config()" in source
