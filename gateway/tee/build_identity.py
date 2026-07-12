"""Create and verify deterministic gateway-enclave build identity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "leadpoet.gateway_enclave_build_identity.v4"
IDENTITY_RELATIVE_PATH = "_attested_runtime/gateway_enclave_build_identity.json"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BuildIdentityError(ValueError):
    """Raised when a deterministic enclave identity cannot be established."""


def _canonical_body(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _normalized_commit(value: Any) -> str:
    commit = str(value or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise BuildIdentityError("gateway enclave build requires a full Git commit")
    return commit


def sha256_file(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BuildIdentityError("cannot read enclave dependency lock") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip().lower()
    except Exception:
        return ""


def resolve_commit(*, gateway_root: Path, source_root: Path, explicit_commit: str = "") -> str:
    candidates = [explicit_commit, os.getenv("ATTESTED_RUNTIME_COMMIT_SHA", "")]
    build_info = gateway_root / "BUILD_INFO.json"
    if build_info.is_file():
        try:
            candidates.append(str(json.loads(build_info.read_text(encoding="utf-8")).get("git_commit") or ""))
        except Exception:
            pass
    candidates.extend((_git_commit(source_root), _git_commit(gateway_root)))
    for candidate in candidates:
        try:
            return _normalized_commit(candidate)
        except BuildIdentityError:
            continue
    raise BuildIdentityError("no full Git commit is available for the gateway enclave build")


def build_identity(
    *,
    role: str,
    service_role: str,
    commit_sha: str,
    execution_manifest_hash: str,
    dependency_lock_hash: str,
    protected_manifest_hash: str,
    topology_hash: str,
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "role": str(role or "").strip(),
        "service_role": str(service_role or "").strip(),
        "commit_sha": _normalized_commit(commit_sha),
        "execution_manifest_hash": str(execution_manifest_hash or "").strip().lower(),
        "dependency_lock_hash": str(dependency_lock_hash or "").strip().lower(),
        "protected_manifest_hash": str(protected_manifest_hash or "").strip().lower(),
        "topology_hash": str(topology_hash or "").strip().lower(),
    }
    if not body["role"] or not body["service_role"]:
        raise BuildIdentityError("gateway enclave role identity is invalid")
    if not _HASH_RE.fullmatch(body["execution_manifest_hash"]):
        raise BuildIdentityError("execution manifest hash is invalid")
    if not _HASH_RE.fullmatch(body["dependency_lock_hash"]):
        raise BuildIdentityError("dependency lock hash is invalid")
    if not _HASH_RE.fullmatch(body["protected_manifest_hash"]):
        raise BuildIdentityError("protected workflow manifest hash is invalid")
    if not _HASH_RE.fullmatch(body["topology_hash"]):
        raise BuildIdentityError("topology manifest hash is invalid")
    return {
        **body,
        "identity_hash": "sha256:" + hashlib.sha256(_canonical_body(body)).hexdigest(),
    }


def validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "role",
        "service_role",
        "commit_sha",
        "execution_manifest_hash",
        "dependency_lock_hash",
        "protected_manifest_hash",
        "topology_hash",
        "identity_hash",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise BuildIdentityError("gateway enclave build identity fields are invalid")
    expected = build_identity(
        role=value.get("role"),
        service_role=value.get("service_role"),
        commit_sha=value.get("commit_sha"),
        execution_manifest_hash=value.get("execution_manifest_hash"),
        dependency_lock_hash=value.get("dependency_lock_hash"),
        protected_manifest_hash=value.get("protected_manifest_hash"),
        topology_hash=value.get("topology_hash"),
    )
    if dict(value) != expected:
        raise BuildIdentityError("gateway enclave build identity hash mismatch")
    return expected


def write_identity(value: Mapping[str, Any], output: Path) -> None:
    identity = validate_identity(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_identity(*, gateway_root: Path, expected_role: str = "") -> dict[str, Any]:
    path = gateway_root / IDENTITY_RELATIVE_PATH
    try:
        identity = validate_identity(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildIdentityError("cannot load gateway enclave build identity") from exc
    if expected_role and identity["role"] != expected_role:
        raise BuildIdentityError("gateway enclave build identity role mismatch")
    return identity


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--gateway-root", required=True, type=Path)
    build.add_argument("--source-root", required=True, type=Path)
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--dependency-lock", required=True, type=Path)
    build.add_argument("--protected-manifest", required=True, type=Path)
    build.add_argument("--topology-manifest", required=True, type=Path)
    build.add_argument("--role", required=True)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--commit", default="")
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--gateway-root", required=True, type=Path)
    resolve.add_argument("--source-root", required=True, type=Path)
    resolve.add_argument("--commit", default="")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--gateway-root", required=True, type=Path)
    verify.add_argument("--role", default="")
    args = parser.parse_args(argv)

    if args.command == "resolve":
        print(
            resolve_commit(
                gateway_root=args.gateway_root,
                source_root=args.source_root,
                explicit_commit=args.commit,
            )
        )
        return 0
    if args.command == "verify":
        identity = load_identity(
            gateway_root=args.gateway_root,
            expected_role=args.role,
        )
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        protected_manifest = json.loads(
            args.protected_manifest.read_text(encoding="utf-8")
        )
        topology_manifest = json.loads(
            args.topology_manifest.read_text(encoding="utf-8")
        )
        roles = topology_manifest.get("roles")
        if not isinstance(roles, Mapping) or args.role not in roles:
            raise BuildIdentityError("role is absent from topology manifest")
        role_spec = roles[args.role]
        if not isinstance(role_spec, Mapping):
            raise BuildIdentityError("role topology is invalid")
        identity = build_identity(
            role=args.role,
            service_role=role_spec.get("service_role"),
            commit_sha=resolve_commit(
                gateway_root=args.gateway_root,
                source_root=args.source_root,
                explicit_commit=args.commit,
            ),
            execution_manifest_hash=(
                manifest.get("role_manifests", {})
                .get(args.role, {})
                .get("manifest_hash")
            ),
            dependency_lock_hash=sha256_file(args.dependency_lock),
            protected_manifest_hash=protected_manifest.get("manifest_hash"),
            topology_hash=topology_manifest.get("topology_hash"),
        )
        write_identity(identity, args.output)
    print("gateway_enclave_commit=%s" % identity["commit_sha"])
    print("gateway_enclave_identity_hash=%s" % identity["identity_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
