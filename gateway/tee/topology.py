"""Canonical coordinator-enclave topology for the gateway deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


TOPOLOGY_SCHEMA_VERSION = "leadpoet.gateway_enclave_topology.v3"
PRODUCTION_INSTANCE_TYPE = "r7i.4xlarge"
PRODUCTION_PARENT_VCPUS = 16
PRODUCTION_PARENT_MEMORY_MIB = 128 * 1024

COORDINATOR_ROLE = "gateway_coordinator"

ROLE_SPECS = {
    COORDINATOR_ROLE: {
        "cid": 16,
        "vcpus": 2,
        "memory_mib": 8 * 1024,
        "service_role": COORDINATOR_ROLE,
    },
}

HOST_RESERVED_VCPUS = PRODUCTION_PARENT_VCPUS - 2
HOST_RESERVED_MEMORY_MIB = PRODUCTION_PARENT_MEMORY_MIB - 8 * 1024


class TopologyError(ValueError):
    """The selected parent cannot safely host the measured coordinator."""


def topology_document() -> Dict[str, Any]:
    return {
        "schema_version": TOPOLOGY_SCHEMA_VERSION,
        "production_instance_type": PRODUCTION_INSTANCE_TYPE,
        "production_parent_vcpus": PRODUCTION_PARENT_VCPUS,
        "production_parent_memory_mib": PRODUCTION_PARENT_MEMORY_MIB,
        "host_reserved_vcpus": HOST_RESERVED_VCPUS,
        "host_reserved_memory_mib": HOST_RESERVED_MEMORY_MIB,
        "roles": {
            role: dict(spec) for role, spec in sorted(ROLE_SPECS.items())
        },
    }


def topology_hash() -> str:
    encoded = json.dumps(
        topology_document(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def manifest_document() -> Dict[str, Any]:
    return {**topology_document(), "topology_hash": topology_hash()}


def validate_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    expected = manifest_document()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise TopologyError("gateway enclave topology manifest is not canonical")
    return expected


def role_spec(role: str) -> Dict[str, Any]:
    if role not in ROLE_SPECS:
        raise TopologyError("unknown gateway enclave role")
    return dict(ROLE_SPECS[role])


def validate_production_capacity(
    *, parent_vcpus: int, parent_memory_mib: int
) -> Dict[str, int]:
    if int(parent_vcpus) < PRODUCTION_PARENT_VCPUS:
        raise TopologyError(
            "full gateway topology requires %s with at least %s vCPUs"
            % (PRODUCTION_INSTANCE_TYPE, PRODUCTION_PARENT_VCPUS)
        )
    if int(parent_memory_mib) < PRODUCTION_PARENT_MEMORY_MIB:
        raise TopologyError(
            "full gateway topology requires %s with at least %s MiB"
            % (PRODUCTION_INSTANCE_TYPE, PRODUCTION_PARENT_MEMORY_MIB)
        )
    return {
        "parent_vcpus": int(parent_vcpus),
        "parent_memory_mib": int(parent_memory_mib),
        "enclave_vcpus": 2,
        "enclave_memory_mib": 8 * 1024,
        "host_vcpus": int(parent_vcpus) - 2,
        "host_memory_mib": int(parent_memory_mib) - 8 * 1024,
    }


def validate_topology() -> None:
    spec = ROLE_SPECS.get(COORDINATOR_ROLE)
    if (
        set(ROLE_SPECS) != {COORDINATOR_ROLE}
        or spec is None
        or spec.get("cid") != 16
        or spec.get("service_role") != COORDINATOR_ROLE
        or HOST_RESERVED_VCPUS != 14
        or HOST_RESERVED_MEMORY_MIB != 120 * 1024
    ):
        raise TopologyError("gateway coordinator topology differs from policy")


def validate_worker_partition() -> None:
    validate_topology()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    validate_topology()
    if args.verify:
        validate_manifest(json.loads(args.verify.read_text(encoding="utf-8")))
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(manifest_document(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print("gateway_enclave_topology_hash=%s" % topology_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
