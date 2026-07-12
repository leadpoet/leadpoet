"""Canonical four-enclave topology for the two-host V2 gateway deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


TOPOLOGY_SCHEMA_VERSION = "leadpoet.gateway_enclave_topology.v2"
PRODUCTION_INSTANCE_TYPE = "r7i.8xlarge"
PRODUCTION_PARENT_VCPUS = 32
PRODUCTION_PARENT_MEMORY_MIB = 256 * 1024

COORDINATOR_ROLE = "gateway_coordinator"
SCORING_A_ROLE = "gateway_scoring_a"
SCORING_B_ROLE = "gateway_scoring_b"
AUTORESEARCH_ROLE = "gateway_autoresearch"

ROLE_SPECS = {
    COORDINATOR_ROLE: {
        "cid": 16,
        "vcpus": 4,
        "memory_mib": 16 * 1024,
        "worker_ids": [],
        "service_role": "gateway_coordinator",
    },
    SCORING_A_ROLE: {
        "cid": 17,
        "vcpus": 8,
        "memory_mib": 40 * 1024,
        "worker_ids": list(range(0, 13)),
        "service_role": "gateway_scoring",
    },
    SCORING_B_ROLE: {
        "cid": 18,
        "vcpus": 8,
        "memory_mib": 40 * 1024,
        "worker_ids": list(range(13, 25)),
        "service_role": "gateway_scoring",
    },
    AUTORESEARCH_ROLE: {
        "cid": 19,
        "vcpus": 6,
        "memory_mib": 32 * 1024,
        "worker_ids": list(range(0, 10)),
        "service_role": "gateway_autoresearch",
    },
}

HOST_RESERVED_VCPUS = PRODUCTION_PARENT_VCPUS - sum(
    int(spec["vcpus"]) for spec in ROLE_SPECS.values()
)
HOST_RESERVED_MEMORY_MIB = PRODUCTION_PARENT_MEMORY_MIB - sum(
    int(spec["memory_mib"]) for spec in ROLE_SPECS.values()
)


class TopologyError(ValueError):
    """The selected parent cannot safely host the authoritative topology."""


def topology_document() -> Dict[str, Any]:
    return {
        "schema_version": TOPOLOGY_SCHEMA_VERSION,
        "production_instance_type": PRODUCTION_INSTANCE_TYPE,
        "production_parent_vcpus": PRODUCTION_PARENT_VCPUS,
        "production_parent_memory_mib": PRODUCTION_PARENT_MEMORY_MIB,
        "host_reserved_vcpus": HOST_RESERVED_VCPUS,
        "host_reserved_memory_mib": HOST_RESERVED_MEMORY_MIB,
        "global_scoring_pool_size": 10,
        "candidate_concurrency": 5,
        "benchmark_concurrency": 5,
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


def validate_production_capacity(*, parent_vcpus: int, parent_memory_mib: int) -> Dict[str, int]:
    if int(parent_vcpus) < PRODUCTION_PARENT_VCPUS:
        raise TopologyError(
            "full V2 topology requires %s with at least %s vCPUs"
            % (PRODUCTION_INSTANCE_TYPE, PRODUCTION_PARENT_VCPUS)
        )
    if int(parent_memory_mib) < PRODUCTION_PARENT_MEMORY_MIB:
        raise TopologyError(
            "full V2 topology requires %s with at least %s MiB"
            % (PRODUCTION_INSTANCE_TYPE, PRODUCTION_PARENT_MEMORY_MIB)
        )
    return {
        "parent_vcpus": int(parent_vcpus),
        "parent_memory_mib": int(parent_memory_mib),
        "enclave_vcpus": sum(int(spec["vcpus"]) for spec in ROLE_SPECS.values()),
        "enclave_memory_mib": sum(
            int(spec["memory_mib"]) for spec in ROLE_SPECS.values()
        ),
        "host_vcpus": int(parent_vcpus)
        - sum(int(spec["vcpus"]) for spec in ROLE_SPECS.values()),
        "host_memory_mib": int(parent_memory_mib)
        - sum(int(spec["memory_mib"]) for spec in ROLE_SPECS.values()),
    }


def validate_worker_partition() -> None:
    scoring_workers = list(ROLE_SPECS[SCORING_A_ROLE]["worker_ids"]) + list(
        ROLE_SPECS[SCORING_B_ROLE]["worker_ids"]
    )
    if scoring_workers != list(range(25)) or len(set(scoring_workers)) != 25:
        raise TopologyError("scoring worker partition must cover exactly IDs 0-24")
    if list(ROLE_SPECS[AUTORESEARCH_ROLE]["worker_ids"]) != list(range(10)):
        raise TopologyError("autoresearch worker partition must cover exactly IDs 0-9")
    cids = [int(spec["cid"]) for spec in ROLE_SPECS.values()]
    if len(cids) != 4 or len(set(cids)) != 4:
        raise TopologyError("gateway topology must use four unique enclave CIDs")
    if HOST_RESERVED_VCPUS != 6 or HOST_RESERVED_MEMORY_MIB != 128 * 1024:
        raise TopologyError("gateway host reservation differs from approved topology")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    validate_worker_partition()
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
