"""Fail closed unless locally built gateway EIFs match an approved V2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS


class ReleaseArtifactVerificationError(RuntimeError):
    """A local role artifact differs from independently reproduced evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_manifest_hash(root: Path) -> str:
    """Match the validator's independent clean-context manifest algorithm."""

    if not root.is_dir():
        raise ReleaseArtifactVerificationError(
            "normalized enclave source context is unavailable"
        )
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
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


def _pcr0_from_build_output(path: Path) -> str:
    try:
        output = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseArtifactVerificationError(
            "Nitro build measurement is unavailable"
        ) from exc
    candidates = []
    try:
        candidates.append(json.loads(output))
    except json.JSONDecodeError:
        pass
    for start, character in enumerate(output):
        if character != "{":
            continue
        try:
            candidates.append(json.loads(output[start:]))
            break
        except json.JSONDecodeError:
            continue
    for value in candidates:
        measurements = value.get("Measurements") if isinstance(value, Mapping) else None
        pcr0 = str(measurements.get("PCR0") or "").lower() if isinstance(measurements, Mapping) else ""
        if len(pcr0) == 96 and all(character in "0123456789abcdef" for character in pcr0):
            return pcr0
    raise ReleaseArtifactVerificationError(
        "Nitro build measurement has no valid PCR0"
    )


def _pcr0_from_eif(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["nitro-cli", "describe-eif", "--eif-path", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactVerificationError(
            "Nitro EIF description is unavailable or invalid"
        ) from exc
    measurements = value.get("Measurements") if isinstance(value, Mapping) else None
    pcr0 = (
        str(measurements.get("PCR0") or "").lower()
        if isinstance(measurements, Mapping)
        else ""
    )
    if len(pcr0) != 96 or any(character not in "0123456789abcdef" for character in pcr0):
        raise ReleaseArtifactVerificationError(
            "Nitro EIF description has no valid PCR0"
        )
    return pcr0


def verify_release_artifacts(
    *,
    release_manifest: Mapping[str, Any],
    gateway_root: Path,
    eif_root: Path,
) -> Dict[str, Any]:
    release = validate_release_manifest(release_manifest)
    gateway_root = gateway_root.resolve()
    eif_root = eif_root.resolve()
    context_hash = source_manifest_hash(gateway_root / "_enclave_source")
    dockerfile_hash = _sha256_file(gateway_root / "tee" / "Dockerfile.enclave")
    verified = []
    for role in sorted(ROLE_SPECS):
        expected = release["roles"][role]
        identity_path = (
            gateway_root
            / "_attested_runtime"
            / "gateway_enclave_build_identities"
            / (role + ".json")
        )
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseArtifactVerificationError(
                "%s build identity is unavailable" % role
            ) from exc
        eif_path = eif_root / ("tee-enclave-%s.eif" % role)
        image_id_path = eif_root / ("enclave-image-%s.txt" % role)
        measurement_path = eif_root / ("enclave-build-%s.json" % role)
        try:
            image_id = image_id_path.read_text(encoding="utf-8").strip().lower()
        except OSError as exc:
            raise ReleaseArtifactVerificationError(
                "%s image identity is unavailable" % role
            ) from exc
        build_pcr0 = _pcr0_from_build_output(measurement_path)
        eif_pcr0 = _pcr0_from_eif(eif_path)
        if build_pcr0 != eif_pcr0:
            raise ReleaseArtifactVerificationError(
                "%s EIF PCR0 differs from its build output" % role
            )
        observed = {
            "commit_sha": str(identity.get("commit_sha") or "").lower(),
            "pcr0": eif_pcr0,
            "normalized_image_hash": image_id,
            "eif_hash": _sha256_file(eif_path),
            "source_manifest_hash": context_hash,
            "build_identity_hash": str(identity.get("identity_hash") or "").lower(),
            "execution_manifest_hash": str(
                identity.get("execution_manifest_hash") or ""
            ).lower(),
            "dependency_lock_hash": str(
                identity.get("dependency_lock_hash") or ""
            ).lower(),
            "dockerfile_hash": dockerfile_hash,
            "topology_hash": str(identity.get("topology_hash") or "").lower(),
        }
        for field, actual in observed.items():
            if field == "eif_hash":
                continue
            if actual != str(expected[field]).lower():
                raise ReleaseArtifactVerificationError(
                    "%s local artifact differs from release at %s" % (role, field)
                )
        verified.append(
            {
                "physical_role": role,
                "pcr0": observed["pcr0"],
                "eif_hash": observed["eif_hash"],
                "build_identity_hash": observed["build_identity_hash"],
            }
        )
    return {
        "schema_version": "leadpoet.gateway_release_artifact_verification.v2",
        "release_hash": release["release_hash"],
        "commit_sha": release["commit_sha"],
        "roles": verified,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--gateway-root", required=True, type=Path)
    parser.add_argument("--eif-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactVerificationError(
            "approved V2 release manifest is unavailable"
        ) from exc
    result = verify_release_artifacts(
        release_manifest=release,
        gateway_root=args.gateway_root,
        eif_root=args.eif_root,
    )
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
