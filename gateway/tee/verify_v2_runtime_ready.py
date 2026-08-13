"""Verify all V2 managers and provider credentials before gateway launch."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from gateway.research_lab.weight_input_authorization_v2 import (
    GatewayWeightInputAuthorizationStoreV2,
)
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
    *,
    storage_probe: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    if storage_probe is None:
        checkpoint_directory = os.environ.get(
            "GATEWAY_WEIGHT_INPUT_CHECKPOINT_DIR", ""
        ).strip()
        if not checkpoint_directory:
            raise V2RuntimeReadinessError(
                "gateway weight input checkpoint storage is not configured"
            )
        checkpoint_path = Path(checkpoint_directory).expanduser()
        if not checkpoint_path.is_absolute():
            raise V2RuntimeReadinessError(
                "gateway weight input checkpoint storage must be absolute"
            )
        storage_probe = GatewayWeightInputAuthorizationStoreV2(
            checkpoint_path
        ).verify_storage_ready
    try:
        storage_probe()
    except V2RuntimeReadinessError:
        raise
    except Exception as exc:
        raise V2RuntimeReadinessError(
            "gateway weight input checkpoint storage is not ready"
        ) from exc
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
    provider_lane = (provider.get("reserved_lanes") or {}).get(
        "weight_submission"
    )
    if (
        not isinstance(provider_lane, Mapping)
        or provider_lane.get("dedicated_transport") is not True
        or provider_lane.get("reserved_capacity") is not True
        or not isinstance(provider_lane.get("capacity"), int)
        or provider_lane["capacity"] < 1
    ):
        raise V2RuntimeReadinessError(
            "reserved weight provider transport is not ready"
        )
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
        if role == "gateway_coordinator":
            execution_lane = (health.get("reserved_lanes") or {}).get(
                "weight_submission"
            )
            if (
                not isinstance(execution_lane, Mapping)
                or execution_lane.get("dedicated_worker") is not True
                or execution_lane.get("reserved_capacity") is not True
                or not isinstance(execution_lane.get("worker_count"), int)
                or execution_lane["worker_count"] < 1
            ):
                raise V2RuntimeReadinessError(
                    "reserved weight execution lane is not ready"
                )
        health_rows.append(
            {
                "physical_role": role,
                "role": health["role"],
                "worker_count": health["worker_count"],
                "configured_worker_count": configured_worker_count,
                "boot_identity_hash": health["boot_identity_hash"],
                **(
                    {"reserved_lanes": health.get("reserved_lanes")}
                    if role == "gateway_coordinator"
                    else {}
                ),
            }
        )
    return {
        "schema_version": "leadpoet.gateway_v2_runtime_readiness.v2",
        "status": "ready",
        "weight_input_storage": "ready",
        "provider_registry_hash": provider_registry_hash(),
        "reserved_lanes": provider.get("reserved_lanes"),
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
