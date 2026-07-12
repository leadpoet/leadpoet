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


async def verify_roles(
    roles: Sequence[str],
    *,
    release_manifest: Optional[dict] = None,
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
        health = await TEEClient(cid=int(spec["cid"])).role_health()
        if not isinstance(health, dict) or health.get("status") != "healthy":
            raise TopologyHealthError("%s role health failed" % role)
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
            asyncio.run(verify_roles(roles, release_manifest=release)),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
