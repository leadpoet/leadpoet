"""Host-side bootstrap for the measured gateway coordinator enclave."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional

from gateway.tee.release_manifest_v2 import (
    role_expectation,
    validate_release_manifest,
)
from gateway.tee.runtime_identity_v2 import RUNTIME_CONFIG_SCHEMA_VERSION
from gateway.tee.topology import ROLE_SPECS
from gateway.utils.tee_client import TEEClient
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    verify_boot_identity_nitro,
)


BOOTSTRAP_SCHEMA_VERSION = "leadpoet.gateway_v2_bootstrap.v2"
COORDINATOR_ROLE = "gateway_coordinator"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TEEV2BootstrapError(RuntimeError):
    """The coordinator enclave release cannot be configured or attested."""


def load_release_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TEEV2BootstrapError("V2 release manifest is unavailable") from exc
    return validate_release_manifest(value)


def runtime_configuration_documents(
    *,
    release_manifest: Mapping[str, Any],
    protected_workflow_manifest_hash: str,
) -> Dict[str, Dict[str, Any]]:
    release = validate_release_manifest(release_manifest)
    protected_hash = str(protected_workflow_manifest_hash or "").lower()
    if not _HASH_RE.fullmatch(protected_hash):
        raise TEEV2BootstrapError("protected workflow manifest hash is invalid")

    expectation = role_expectation(release, COORDINATOR_ROLE)
    release_role = {
        "commit_sha": expectation["commit_sha"],
        "pcr0": expectation["pcr0"],
        "build_manifest_hash": expectation["build_manifest_hash"],
        "dependency_lock_hash": expectation["dependency_lock_hash"],
    }
    configuration = {
        "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "release_hash": release["release_hash"],
        "release_commit_sha": release["commit_sha"],
        "own_build_identity_hash": expectation["build_identity_hash"],
        "release_roles": {COORDINATOR_ROLE: release_role},
        "peer_releases": {},
        "protected_workflow_manifest_hash": protected_hash,
    }
    config_document = {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        "physical_role": COORDINATOR_ROLE,
        "service_role": ROLE_SPECS[COORDINATOR_ROLE]["service_role"],
        "configuration": json.loads(canonical_json(configuration)),
    }
    return {
        COORDINATOR_ROLE: {
            "configuration": config_document["configuration"],
            "configuration_hash": sha256_json(config_document),
        }
    }


def _default_clients() -> Dict[str, TEEClient]:
    return {
        COORDINATOR_ROLE: TEEClient(
            cid=int(ROLE_SPECS[COORDINATOR_ROLE]["cid"])
        )
    }


async def bootstrap_gateway_enclaves_v2(
    *,
    release_manifest: Mapping[str, Any],
    runtime_documents: Mapping[str, Mapping[str, Any]],
    clients: Optional[Mapping[str, Any]] = None,
    boot_verifier: Callable[..., Mapping[str, Any]] = verify_boot_identity_nitro,
) -> Dict[str, Any]:
    """Configure and independently verify the coordinator enclave."""

    release = validate_release_manifest(release_manifest)
    expected_roles = {COORDINATOR_ROLE}
    if set(runtime_documents) != expected_roles:
        raise TEEV2BootstrapError(
            "runtime documents must cover only the coordinator role"
        )
    role_clients = dict(clients or _default_clients())
    if set(role_clients) != expected_roles:
        raise TEEV2BootstrapError(
            "enclave clients must cover only the coordinator role"
        )

    document = runtime_documents[COORDINATOR_ROLE]
    if not isinstance(document, Mapping) or set(document) != {
        "configuration",
        "configuration_hash",
    }:
        raise TEEV2BootstrapError("runtime document fields are invalid")
    client = role_clients[COORDINATOR_ROLE]
    status = await client.v2_configure_runtime(
        configuration=dict(document["configuration"]),
        configuration_hash=str(document["configuration_hash"]),
    )
    if (
        status.get("status") != "ready"
        or status.get("physical_role") != COORDINATOR_ROLE
    ):
        raise TEEV2BootstrapError("coordinator runtime configuration failed")

    boot = await client.v2_get_boot_identity()
    expectation = role_expectation(release, COORDINATOR_ROLE)
    if boot.get("physical_role") != COORDINATOR_ROLE:
        raise TEEV2BootstrapError("coordinator returned another physical role")
    if boot.get("commit_sha") != expectation["commit_sha"]:
        raise TEEV2BootstrapError("coordinator boot commit differs from release")
    if boot.get("build_manifest_hash") != expectation["build_manifest_hash"]:
        raise TEEV2BootstrapError("coordinator boot manifest differs from release")
    if boot.get("dependency_lock_hash") != expectation["dependency_lock_hash"]:
        raise TEEV2BootstrapError("coordinator dependency lock differs from release")
    boot_verifier(boot, expected_pcr0=expectation["pcr0"])

    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "status": "ready",
        "release_hash": release["release_hash"],
        "commit_sha": release["commit_sha"],
        "boot_identity_hashes": {
            COORDINATOR_ROLE: boot["boot_identity_hash"]
        },
    }


async def _main_async(args) -> Dict[str, Any]:
    release = load_release_manifest(args.release_manifest)
    try:
        protected_manifest = json.loads(
            args.protected_workflow_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TEEV2BootstrapError(
            "protected workflow manifest is unavailable"
        ) from exc
    protected_hash = str(protected_manifest.get("manifest_hash") or "").lower()
    documents = runtime_configuration_documents(
        release_manifest=release,
        protected_workflow_manifest_hash=protected_hash,
    )
    return await bootstrap_gateway_enclaves_v2(
        release_manifest=release,
        runtime_documents=documents,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument(
        "--protected-workflow-manifest",
        required=True,
        type=Path,
    )
    args = parser.parse_args()
    result = asyncio.run(_main_async(args))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
