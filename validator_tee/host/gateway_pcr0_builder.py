"""Independently rebuild the gateway EIF from a clean Git commit.

This verifier is intentionally operator-run. It never restarts services or
changes validator weights. It prunes BuildKit cache to establish the same clean
boundary for every build, and writes a cache entry only after repeated clean
builds produce the same image ID and PCR0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import BUILD_EVIDENCE_SCHEMA_VERSION
from gateway.tee.topology import ROLE_SPECS
from leadpoet_observability import (
    capture_failure,
    configure_sentry_context,
    failure_code_for_exception,
    init_sentry,
    record_stage,
)
from validator_tee.host.docker_image_normalizer_v2 import normalize_docker_image


CACHE_SCHEMA_VERSION = "leadpoet.gateway_pcr0_cache.v2"
DEFAULT_CACHE_SIZE = 20
GATEWAY_ROLES = (
    "gateway_coordinator",
    "gateway_scoring",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class GatewayPCR0BuildError(RuntimeError):
    """Raised when an independent gateway build is incomplete or divergent."""


def _run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: int = 3600,
    check: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            value.strip()
            for value in (exc.stdout or "", exc.stderr or "")
            if value.strip()
        )
        if len(output) > 12000:
            output = output[-12000:]
        raise GatewayPCR0BuildError(
            "%s failed with exit code %s%s"
            % (
                Path(str(command[0])).name,
                exc.returncode,
                ":\n" + output if output else "",
            )
        ) from exc


def resolve_commit(repo_root: Path, revision: str) -> str:
    result = _run(["git", "rev-parse", "%s^{commit}" % revision], cwd=repo_root, timeout=30)
    commit = result.stdout.strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayPCR0BuildError("git did not resolve a full commit")
    return commit


def _safe_extract_git_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive_path), "r:") as archive:
        members = archive.getmembers()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise GatewayPCR0BuildError("git archive contains an unsafe path")
            if member.issym() or member.islnk() or member.isdev():
                raise GatewayPCR0BuildError("git archive contains a non-regular entry")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                os.chmod(str(target), member.mode & 0o777)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise GatewayPCR0BuildError("git archive file content is missing")
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(str(target), member.mode & 0o777)


def extract_clean_commit(*, repo_root: Path, commit: str, destination: Path) -> None:
    archive_path = destination.parent / ("%s.tar" % commit[:12])
    if destination.exists():
        shutil.rmtree(destination)
    if archive_path.exists():
        archive_path.unlink()
    _run(
        ["git", "archive", "--format=tar", "--output", str(archive_path), commit],
        cwd=repo_root,
        timeout=120,
    )
    try:
        _safe_extract_git_archive(archive_path, destination)
    finally:
        archive_path.unlink(missing_ok=True)


def source_manifest_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        kind = "d" if path.is_dir() else "f"
        mode = stat.S_IMODE(path.stat().st_mode)
        digest.update(("%s %04o %s\n" % (kind, mode, relative)).encode("utf-8"))
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _parse_measurement(output: str) -> str:
    candidates = []
    try:
        candidates.append(json.loads(output))
    except json.JSONDecodeError:
        pass
    for start in (index for index, character in enumerate(output) if character == "{"):
        try:
            candidates.append(json.loads(output[start:]))
            break
        except json.JSONDecodeError:
            continue
    for value in candidates:
        if not isinstance(value, Mapping):
            continue
        measurements = value.get("Measurements")
        if isinstance(measurements, Mapping):
            pcr0 = str(measurements.get("PCR0") or "").lower()
            if _PCR0_RE.fullmatch(pcr0) and pcr0 != "0" * 96:
                return pcr0
    raise GatewayPCR0BuildError("nitro-cli output did not contain a valid PCR0")


def _deterministic_docker_build_command(
    *, gateway_root: Path, image: str, role: str
) -> list[str]:
    return [
        "docker",
        "build",
        "--pull",
        "--no-cache",
        "--build-arg",
        "SOURCE_DATE_EPOCH=0",
        "--build-arg",
        "LEADPOET_ENCLAVE_ROLE=%s" % role,
        "-f",
        str(gateway_root / "tee" / "Dockerfile.enclave"),
        "-t",
        image,
        str(gateway_root),
    ]


def _prune_builder_cache() -> None:
    """Require a clean BuildKit state before the repeated-build boundary."""

    _run(["docker", "builder", "prune", "-af"], timeout=600)


def _normalized_rootfs_layer_hashes(image: str) -> list[str]:
    result = _run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{json .RootFS.Layers}}",
            image,
        ],
        timeout=120,
    )
    try:
        layers = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise GatewayPCR0BuildError(
            "normalized gateway image layer identity is invalid"
        ) from exc
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            not isinstance(layer, str) or not _SHA256_RE.fullmatch(layer)
            for layer in layers
        )
    ):
        raise GatewayPCR0BuildError(
            "normalized gateway image layer identity is invalid"
        )
    return layers


def _build_once(
    *,
    source_root: Path,
    commit: str,
    work_root: Path,
    index: int,
    role: str,
) -> dict[str, Any]:
    if role not in GATEWAY_ROLES:
        raise GatewayPCR0BuildError("gateway build role is invalid")
    gateway_root = source_root / "gateway"
    env = dict(os.environ)
    env.update(
        {
            "GATEWAY_ROOT": str(gateway_root),
            "RESEARCH_LAB_RUNTIME_SOURCE_ROOT": str(source_root),
            "ATTESTED_RUNTIME_COMMIT_SHA": commit,
            "ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE": "1",
        }
    )
    _run(["bash", str(gateway_root / "tee" / "stage_attested_runtime.sh")], env=env, timeout=900)
    context_root = gateway_root / "_enclave_source"
    context_hash = source_manifest_hash(context_root)
    identity_path = (
        gateway_root
        / "_attested_runtime"
        / "gateway_enclave_build_identities"
        / (role + ".json")
    )
    try:
        build_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GatewayPCR0BuildError("gateway role build identity is unavailable") from exc
    if build_identity.get("role") != role:
        raise GatewayPCR0BuildError("gateway role build identity mismatch")

    image = "leadpoet-gateway-verify:%s-%s-%s" % (role, commit[:12], index)
    raw_image = image + "-raw"
    _run(["docker", "rmi", "-f", image, raw_image], check=False, timeout=120)
    build_env = dict(os.environ)
    build_env["DOCKER_BUILDKIT"] = "1"
    build_env["BUILDX_NO_DEFAULT_ATTESTATIONS"] = "1"
    _run(
        _deterministic_docker_build_command(
            gateway_root=gateway_root,
            image=raw_image,
            role=role,
        ),
        env=build_env,
        timeout=3600,
    )
    # The tagged image retains every layer needed for normalization. Drop the
    # duplicate BuildKit cache before docker save so constrained independent
    # builders do not need space for the image, cache, and export at once.
    _run(["docker", "builder", "prune", "-af"], timeout=600)
    image_id = normalize_docker_image(
        source_image=raw_image,
        normalized_image=image,
    )
    if not image_id.startswith("sha256:"):
        raise GatewayPCR0BuildError("gateway image ID is invalid")
    rootfs_layer_hashes = _normalized_rootfs_layer_hashes(image)

    eif_path = work_root / (
        "gateway-%s-%s-%s.eif" % (role, commit[:12], index)
    )
    eif_path.unlink(missing_ok=True)
    result = _run(
        [
            "nitro-cli",
            "build-enclave",
            "--docker-uri",
            image,
            "--output-file",
            str(eif_path),
        ],
        timeout=1800,
    )
    pcr0 = _parse_measurement(result.stdout)
    eif_hash = "sha256:" + hashlib.sha256(eif_path.read_bytes()).hexdigest()
    eif_path.unlink()
    dockerfile_hash = "sha256:" + hashlib.sha256(
        (gateway_root / "tee" / "Dockerfile.enclave").read_bytes()
    ).hexdigest()
    _run(["docker", "rmi", "-f", image, raw_image], check=False, timeout=120)
    _run(["docker", "builder", "prune", "-af"], timeout=600)
    return {
        "commit_sha": commit,
        "role": role,
        "pcr0": pcr0,
        "image_id": image_id,
        "rootfs_layer_hashes": rootfs_layer_hashes,
        "eif_sha256": eif_hash,
        "source_manifest_hash": context_hash,
        "build_identity_hash": str(build_identity.get("identity_hash") or ""),
        "execution_manifest_hash": str(
            build_identity.get("execution_manifest_hash") or ""
        ),
        "dependency_lock_hash": str(
            build_identity.get("dependency_lock_hash") or ""
        ),
        "dockerfile_hash": dockerfile_hash,
        "topology_hash": str(build_identity.get("topology_hash") or ""),
    }


def build_reproducible_gateway_pcr0(
    *,
    repo_root: Path,
    revision: str,
    work_root: Path,
    repetitions: int = 3,
    role: str = "gateway_coordinator",
    builder_domain: str = "validator",
    builder_id: str = "validator-parent",
) -> dict[str, Any]:
    local_single_build = repetitions == 1 and builder_domain == "local"
    if repetitions < 3 and not local_single_build:
        raise GatewayPCR0BuildError("at least three independent builds are required")
    if role not in GATEWAY_ROLES:
        raise GatewayPCR0BuildError("gateway build role is invalid")
    if builder_domain not in {"gateway", "validator", "local"}:
        raise GatewayPCR0BuildError("gateway build evidence domain is invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", builder_id):
        raise GatewayPCR0BuildError("gateway build evidence builder id is invalid")
    commit = resolve_commit(repo_root, revision)
    work_root.mkdir(parents=True, exist_ok=True)
    source_root = work_root / "source"
    extract_clean_commit(repo_root=repo_root, commit=commit, destination=source_root)
    # A previous failed attestation can leave BuildKit state behind before its
    # normal post-build prune.  The repeated-build authority must therefore
    # establish the same clean cache boundary before ordinal one that later
    # ordinals receive from their predecessor.
    _prune_builder_cache()
    results = [
        _build_once(
            source_root=source_root,
            commit=commit,
            work_root=work_root,
            index=index + 1,
            role=role,
        )
        for index in range(repetitions)
    ]
    reproducibility_fields = (
        "pcr0",
        "image_id",
        "rootfs_layer_hashes",
        "source_manifest_hash",
        "build_identity_hash",
        "execution_manifest_hash",
        "dependency_lock_hash",
        "dockerfile_hash",
        "topology_hash",
    )
    divergences = {
        field: [result[field] for result in results]
        for field in reproducibility_fields
        if len(
            {
                json.dumps(result[field], sort_keys=True, separators=(",", ":"))
                for result in results
            }
        )
        != 1
    }
    if divergences:
        first_field = next(
            field for field in reproducibility_fields if field in divergences
        )
        raise GatewayPCR0BuildError(
            "gateway builds diverged at %s: %s; divergent_build_diagnostics=%s"
            % (
                first_field,
                json.dumps(divergences[first_field], separators=(",", ":")),
                json.dumps(divergences, sort_keys=True, separators=(",", ":")),
            )
        )
    evidence = [
        {
            "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
            "builder_domain": builder_domain,
            "builder_id": builder_id,
            "build_ordinal": index + 1,
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            "commit_sha": result["commit_sha"],
            "pcr0": result["pcr0"],
            "normalized_image_hash": result["image_id"],
            "eif_hash": result["eif_sha256"],
            "source_manifest_hash": result["source_manifest_hash"],
            "build_identity_hash": result["build_identity_hash"],
            "execution_manifest_hash": result["execution_manifest_hash"],
            "dependency_lock_hash": result["dependency_lock_hash"],
            "dockerfile_hash": result["dockerfile_hash"],
            "topology_hash": result["topology_hash"],
        }
        for index, result in enumerate(results)
    ]
    return {
        "role": role,
        "commit_sha": commit,
        "pcr0": results[0]["pcr0"],
        "image_id": results[0]["image_id"],
        "eif_sha256": results[0]["eif_sha256"],
        "source_manifest_hash": results[0]["source_manifest_hash"],
        "build_identity_hash": results[0]["build_identity_hash"],
        "execution_manifest_hash": results[0]["execution_manifest_hash"],
        "dependency_lock_hash": results[0]["dependency_lock_hash"],
        "dockerfile_hash": results[0]["dockerfile_hash"],
        "topology_hash": results[0]["topology_hash"],
        "verified_build_count": repetitions,
        "build_evidence": evidence,
    }


def write_cache_entry(
    *,
    cache_path: Path,
    entry: Mapping[str, Any],
    cache_size: int = DEFAULT_CACHE_SIZE,
    pin: bool = False,
) -> None:
    try:
        current = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "entries": [],
            "pinned_deployments": [],
        }
    if current.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(current.get("entries"), list):
        raise GatewayPCR0BuildError("gateway PCR0 cache schema is invalid")
    commit = str(entry.get("commit_sha") or "")
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayPCR0BuildError("gateway PCR0 cache entry commit is invalid")
    role = str(entry.get("role") or "")
    if role not in GATEWAY_ROLES:
        raise GatewayPCR0BuildError("gateway PCR0 cache entry role is invalid")
    limit = max(1, int(cache_size))
    pinned_deployments = [
        {
            "role": str(value.get("role") or ""),
            "commit_sha": str(value.get("commit_sha") or ""),
        }
        for value in current.get("pinned_deployments", [])
        if isinstance(value, Mapping)
        and str(value.get("role") or "") in GATEWAY_ROLES
        and _COMMIT_RE.fullmatch(str(value.get("commit_sha") or ""))
    ]
    deployment = {"role": role, "commit_sha": commit}
    if pin and deployment not in pinned_deployments:
        pinned_deployments.insert(0, deployment)
    entries = [dict(entry)] + [
        row
        for row in current["entries"]
        if not (row.get("commit_sha") == commit and row.get("role") == role)
    ]
    retained_entries = []
    for cache_role in GATEWAY_ROLES:
        role_entries = [row for row in entries if row.get("role") == cache_role]
        role_pins = [
            value for value in pinned_deployments if value["role"] == cache_role
        ]
        pinned_entries = [
            row
            for value in role_pins
            for row in role_entries
            if row.get("commit_sha") == value["commit_sha"]
        ]
        if len(pinned_entries) > limit:
            raise GatewayPCR0BuildError(
                "pinned gateway deployments exceed per-role cache capacity"
            )
        pinned_keys = {
            (row.get("role"), row.get("commit_sha")) for row in pinned_entries
        }
        unpinned_entries = [
            row
            for row in role_entries
            if (row.get("role"), row.get("commit_sha")) not in pinned_keys
        ]
        retained_entries.extend(
            pinned_entries + unpinned_entries[: limit - len(pinned_entries)]
        )
    retained_keys = {
        (str(row.get("role") or ""), str(row.get("commit_sha") or ""))
        for row in retained_entries
    }
    document = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "entries": retained_entries,
        "pinned_deployments": [
            value
            for value in pinned_deployments
            if (value["role"], value["commit_sha"]) in retained_keys
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(cache_path))


def load_cached_gateway_identity(
    cache_path: Path,
    commit_sha: str,
    *,
    role: str = "",
) -> Optional[dict[str, Any]]:
    try:
        document = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    matches = [
        entry
        for entry in document.get("entries", [])
        if entry.get("commit_sha") == commit_sha
        and (not role or entry.get("role") == role)
        and int(entry.get("verified_build_count") or 0) >= 3
    ]
    if len(matches) != 1:
        return None
    for entry in matches:
        pcr0 = str(entry.get("pcr0") or "")
        if _PCR0_RE.fullmatch(pcr0) and pcr0 != "0" * 96:
            return dict(entry)
    return None


def _write_json_result(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".%s." % path.name,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path.home() / ".cache" / "leadpoet" / "gateway-pcr0-work",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path.home() / ".cache" / "leadpoet" / "gateway-pcr0-cache.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--builder-domain",
        choices=("gateway", "validator", "local"),
        default="validator",
    )
    parser.add_argument("--builder-id", default="validator-parent")
    parser.add_argument("--role", choices=GATEWAY_ROLES)
    parser.add_argument("--all-roles", action="store_true")
    parser.add_argument(
        "--output-file",
        type=Path,
        help="write the machine-readable result to this file instead of stdout",
    )
    parser.add_argument(
        "--pin",
        action="store_true",
        help="retain this deployed commit when newer cache entries are added",
    )
    args = parser.parse_args(argv)
    if bool(args.role) == bool(args.all_roles):
        raise GatewayPCR0BuildError("select exactly one --role or --all-roles")
    roles = GATEWAY_ROLES if args.all_roles else (args.role,)
    entries = []
    for role in roles:
        entry = build_reproducible_gateway_pcr0(
            repo_root=args.repo_root.resolve(),
            revision=args.revision,
            work_root=args.work_root.resolve() / str(role),
            repetitions=args.repetitions,
            role=str(role),
            builder_domain=args.builder_domain,
            builder_id=args.builder_id,
        )
        write_cache_entry(cache_path=args.cache_file, entry=entry, pin=args.pin)
        entries.append(entry)
        record_stage(
            component="release-gateway-pcr0-builder",
            stage="gateway_reproducible_pcr0_build",
            status="passed",
            physical_role=str(role),
            candidate_sha=entry.get("commit_sha"),
            runtime_sha=entry.get("commit_sha"),
            observed_pcr0_hash=entry.get("pcr0"),
            build_manifest_hash=entry.get("build_manifest_hash"),
        )
    result = entries if args.all_roles else entries[0]
    if args.output_file is None:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        _write_json_result(args.output_file.resolve(), result)
    return 0


def _run_cli() -> int:
    init_sentry(component="release-gateway-pcr0-builder")
    candidate_sha = str(os.environ.get("GITHUB_SHA") or "").lower()
    configure_sentry_context(
        component="release-gateway-pcr0-builder",
        role="release-builder",
        candidate_sha=candidate_sha,
        runtime_sha=candidate_sha,
    )
    try:
        return main()
    except Exception as exc:
        capture_failure(
            failure_code_for_exception(
                exc,
                default="restart.terminal_failure",
            ),
            component="release-gateway-pcr0-builder",
            stage="gateway_reproducible_pcr0_build",
            exception=exc,
            terminal=True,
            retryable=False,
            fail_closed=True,
            candidate_sha=candidate_sha,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(_run_cli())
