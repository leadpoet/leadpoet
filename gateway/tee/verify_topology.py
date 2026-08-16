"""Verify every running gateway enclave reports its assigned V2 role."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Sequence

from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.utils.tee_client import TEEClient


class TopologyHealthError(RuntimeError):
    """A role is unavailable or reports an unexpected measured identity."""


_REQUIRED_TRANSPORT_HEALTH_SCHEMAS_BY_ROLE = {
    role: {
        "parent_rpc_transport": (
            "leadpoet.gateway_vsock_rpc_transport_health.v2"
        ),
        "inter_enclave_transport": (
            "leadpoet.inter_enclave_role_transport_health.v2"
        ),
    }
    for role in ROLE_SPECS
}
_INTER_ENCLAVE_CHILD_TRANSPORT_HEALTH_SCHEMA = (
    "leadpoet.inter_enclave_transport_health.v2"
)
_V2_RUNTIME_CONFIG_SCHEMA = "leadpoet.enclave_runtime_config.v2"


async def verify_roles(
    roles: Sequence[str],
    *,
    release_manifest: Optional[dict] = None,
    prebootstrap_launch_readiness: bool = False,
) -> list[dict]:
    release = (
        validate_release_manifest(release_manifest)
        if release_manifest is not None
        else None
    )
    results = []
    for role in roles:
        if role not in ROLE_SPECS:
            raise TopologyHealthError("unknown topology role %s" % role)
        spec = ROLE_SPECS[role]
        required_transport_health = (
            _REQUIRED_TRANSPORT_HEALTH_SCHEMAS_BY_ROLE.get(role)
        )
        if not required_transport_health:
            raise TopologyHealthError(
                "%s transport health applicability is undefined" % role
            )
        health = await TEEClient(cid=int(spec["cid"])).role_health()
        if not isinstance(health, dict) or health.get("status") != "healthy":
            raise TopologyHealthError("%s role health failed" % role)
        if prebootstrap_launch_readiness:
            expected_runtime = {
                "schema_version": _V2_RUNTIME_CONFIG_SCHEMA,
                "status": "not_configured",
                "physical_role": role,
                "service_role": spec["service_role"],
            }
            if health.get("v2_runtime") != expected_runtime:
                raise TopologyHealthError(
                    "%s pre-bootstrap V2 runtime state is not pristine" % role
                )
        for transport_name, expected_schema in (
            required_transport_health.items()
        ):
            transport_health = health.get(transport_name)
            if (
                prebootstrap_launch_readiness
                and transport_name == "inter_enclave_transport"
            ):
                # Before the parent relay and V2 bootstrap exist, the enclave
                # projects the two absent TLS endpoints through the aggregate
                # error status.  Accept only that exact pristine shape: any
                # initialized, failed, or retained-cleanup state remains fatal.
                expected_transport = {
                    "schema_version": expected_schema,
                    "status": "error",
                    "server": {"status": "unavailable"},
                    "client": {"status": "unavailable"},
                }
                if transport_health != expected_transport:
                    raise TopologyHealthError(
                        "%s pre-bootstrap inter_enclave_transport state is not pristine"
                        % role
                    )
                continue
            if (
                not isinstance(transport_health, dict)
                or transport_health.get("schema_version") != expected_schema
                or transport_health.get("status") != "healthy"
            ):
                raise TopologyHealthError(
                    "%s %s health failed" % (role, transport_name)
                )
            if transport_name == "inter_enclave_transport":
                for child_name in ("server", "client"):
                    child_health = transport_health.get(child_name)
                    if (
                        not isinstance(child_health, dict)
                        or child_health.get("schema_version")
                        != _INTER_ENCLAVE_CHILD_TRANSPORT_HEALTH_SCHEMA
                        or child_health.get("status") != "healthy"
                    ):
                        raise TopologyHealthError(
                            "%s inter_enclave_transport %s health failed"
                            % (role, child_name)
                        )
        if health.get("role") != role:
            raise TopologyHealthError("%s reported role %s" % (role, health.get("role")))
        if health.get("service_role") != spec["service_role"]:
            raise TopologyHealthError("%s service role mismatch" % role)
        if health.get("topology_hash") != topology_hash():
            raise TopologyHealthError("%s topology hash mismatch" % role)
        pcr0 = str(health.get("pcr0") or "").lower()
        if len(pcr0) != 96 or pcr0 == "0" * 96:
            raise TopologyHealthError("%s has no hardware PCR0" % role)
        if release is not None:
            expected = release["roles"][role]
            for field, observed, approved in (
                ("commit_sha", str(health.get("commit_sha") or ""), expected["commit_sha"]),
                ("pcr0", pcr0, expected["pcr0"]),
                (
                    "build_identity_hash",
                    str(health.get("build_identity_hash") or ""),
                    expected["build_identity_hash"],
                ),
            ):
                if observed != approved:
                    raise TopologyHealthError(
                        "%s differs from approved release at %s" % (role, field)
                    )
        results.append(
            {
                "role": role,
                "cid": int(spec["cid"]),
                "commit_sha": str(health.get("commit_sha") or ""),
                "build_identity_hash": str(health.get("build_identity_hash") or ""),
                "pcr0": pcr0,
            }
        )
    commits = {item["commit_sha"] for item in results}
    if len(commits) != 1:
        raise TopologyHealthError("gateway enclave roles run different commits")
    if len({item["build_identity_hash"] for item in results}) != len(results):
        raise TopologyHealthError("physical role build identities are not unique")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roles", nargs="*")
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--prebootstrap-launch-readiness",
        action="store_true",
        help=(
            "verify measured role identity before the V2 runtime and "
            "inter-enclave transport are configured"
        ),
    )
    args = parser.parse_args(argv)
    roles = args.roles or list(ROLE_SPECS)
    release = None
    if args.release_manifest:
        try:
            release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TopologyHealthError(
                "approved gateway release manifest is unavailable"
            ) from exc
    print(
        json.dumps(
            asyncio.run(
                verify_roles(
                    roles,
                    release_manifest=release,
                    prebootstrap_launch_readiness=(
                        args.prebootstrap_launch_readiness
                    ),
                )
            ),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
