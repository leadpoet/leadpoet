"""Verify all V2 managers and provider credentials before gateway launch."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Mapping, Optional

from gateway.tee.provider_broker_v2 import (
    expected_job_credential_slot_ref_hashes,
    expected_provider_credential_slots,
    provider_registry_hash,
)
from gateway.tee.topology import ROLE_SPECS
from gateway.utils.tee_client import TEEClient


class V2RuntimeReadinessError(RuntimeError):
    """A V2 execution role or credential authority is not ready."""


def _clients() -> Dict[str, Any]:
    return {
        role: TEEClient(cid=int(spec["cid"]))
        for role, spec in ROLE_SPECS.items()
    }


async def verify_v2_runtime_ready(
    clients: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    role_clients = dict(clients or _clients())
    if set(role_clients) != set(ROLE_SPECS):
        raise V2RuntimeReadinessError("runtime clients do not cover every role")
    provider = await role_clients[
        "gateway_coordinator"
    ].v2_provider_broker_health()
    if (
        provider.get("status") != "ready"
        or set(provider.get("credential_slots") or ())
        != set(expected_provider_credential_slots())
        or provider.get("missing_credential_slots")
        or provider.get("registry_hash") != provider_registry_hash()
        or provider.get("job_credential_slot_ref_hashes")
        != expected_job_credential_slot_ref_hashes()
    ):
        raise V2RuntimeReadinessError("coordinator provider broker is not ready")
    semantics = await role_clients[
        "gateway_coordinator"
    ].v2_provider_semantics_health()
    if (
        semantics.get("status") != "ready"
        or semantics.get("broker_registry_hash") != provider_registry_hash()
        or not isinstance(semantics.get("memory_cache_entry_count"), int)
        or not isinstance(semantics.get("inflight_count"), int)
        or not isinstance(semantics.get("cost_scope_count"), int)
    ):
        raise V2RuntimeReadinessError(
            "coordinator provider semantics authority is not ready"
        )

    calls = {
        "gateway_coordinator": "coordinator_v2_health",
        "gateway_scoring": "scoring_v2_health",
        "gateway_autoresearch": "autoresearch_v2_health",
    }
    expected_workers = {
        "gateway_coordinator": 1,
        "gateway_scoring": 10,
    }
    health_rows = []
    for role in sorted(ROLE_SPECS):
        health = await getattr(role_clients[role], calls[role])()
        configured_worker_count = health.get("configured_worker_count")
        expected_worker_count = expected_workers.get(role)
        if (
            health.get("authority") != "v2_only"
            or health.get("physical_role") != role
            or health.get("role") != ROLE_SPECS[role]["service_role"]
            or (
                expected_worker_count is not None
                and health.get("worker_count") != expected_worker_count
            )
            or not isinstance(configured_worker_count, int)
            or configured_worker_count < 0
            or (
                role == "gateway_coordinator"
                and configured_worker_count != 0
            )
            or (
                role == "gateway_scoring"
                and configured_worker_count <= 0
            )
            or (
                role == "gateway_autoresearch"
                and (
                    configured_worker_count <= 0
                    or health.get("worker_count") != configured_worker_count
                )
            )
            or health.get("workers_alive") is not True
        ):
            raise V2RuntimeReadinessError("%s execution manager is not ready" % role)
        health_rows.append(
            {
                "physical_role": role,
                "role": health["role"],
                "worker_count": health["worker_count"],
                "configured_worker_count": configured_worker_count,
                "boot_identity_hash": health["boot_identity_hash"],
            }
        )
    return {
        "schema_version": "leadpoet.gateway_v2_runtime_readiness.v2",
        "status": "ready",
        "provider_registry_hash": provider_registry_hash(),
        "roles": health_rows,
    }


def main() -> int:
    print(
        json.dumps(
            asyncio.run(verify_v2_runtime_ready()),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
