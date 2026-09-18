"""Verify measured gateway enclave identities before gateway launch."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Mapping, Optional

from gateway.tee.topology import ROLE_SPECS
from gateway.utils.tee_client import TEEClient


class V2RuntimeReadinessError(RuntimeError):
    """A measured gateway enclave role is not ready."""


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

    health_rows = []
    for role in sorted(ROLE_SPECS):
        health = await role_clients[role].role_health()
        runtime = health.get("v2_runtime")
        if (
            health.get("status") != "healthy"
            or health.get("role") != role
            or health.get("service_role") != ROLE_SPECS[role]["service_role"]
            or not isinstance(runtime, Mapping)
            or runtime.get("status") != "ready"
            or runtime.get("physical_role") != role
            or runtime.get("service_role") != ROLE_SPECS[role]["service_role"]
            or not runtime.get("boot_identity_hash")
        ):
            raise V2RuntimeReadinessError(
                "%s measured runtime identity is not ready" % role
            )
        health_rows.append(
            {
                "physical_role": role,
                "service_role": health["service_role"],
                "commit_sha": health["commit_sha"],
                "build_identity_hash": health["build_identity_hash"],
                "boot_identity_hash": runtime["boot_identity_hash"],
                "pcr0": runtime["pcr0"],
            }
        )
    return {
        "schema_version": "leadpoet.gateway_v2_runtime_readiness.v3",
        "status": "ready",
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
