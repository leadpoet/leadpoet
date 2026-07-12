"""Stage exact validator V2 binary artifacts into the Docker build context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.request import Request, urlopen


SCHEMA_VERSION = "leadpoet.validator_runtime_artifacts.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeArtifactV2Error(RuntimeError):
    """A required runtime artifact is unpinned, unavailable, or corrupted."""


def load_lock(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactV2Error("runtime artifact lock is unavailable") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "artifacts"}
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("artifacts"), Mapping)
        or not value["artifacts"]
    ):
        raise RuntimeArtifactV2Error("runtime artifact lock is invalid")
    normalized = {}
    filenames = set()
    for name, artifact in sorted(value["artifacts"].items()):
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", str(name))
            or not isinstance(artifact, Mapping)
            or set(artifact) != {"filename", "sha256", "url"}
        ):
            raise RuntimeArtifactV2Error("runtime artifact entry is invalid")
        filename = str(artifact.get("filename") or "")
        digest = str(artifact.get("sha256") or "").lower()
        url = str(artifact.get("url") or "")
        if (
            Path(filename).name != filename
            or not filename
            or filename in filenames
            or not _SHA256_RE.fullmatch(digest)
            or not url.startswith("https://")
        ):
            raise RuntimeArtifactV2Error("runtime artifact pin is invalid")
        filenames.add(filename)
        normalized[str(name)] = {
            "filename": filename,
            "sha256": digest,
            "url": url,
        }
    return {"schema_version": SCHEMA_VERSION, "artifacts": normalized}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified_artifact(
    *,
    source: Path,
    destination: Path,
    expected_sha256: str,
    artifact_name: str,
) -> None:
    if not source.is_file():
        raise RuntimeArtifactV2Error(
            "offline runtime artifact is unavailable: %s" % artifact_name
        )
    if _sha256(source) != expected_sha256:
        raise RuntimeArtifactV2Error(
            "offline runtime artifact hash mismatch: %s" % artifact_name
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(str(source), str(temporary))
        if _sha256(temporary) != expected_sha256:
            raise RuntimeArtifactV2Error(
                "copied runtime artifact hash mismatch: %s" % artifact_name
            )
        os.replace(str(temporary), str(destination))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _download_verified_artifact(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    artifact_name: str,
) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        request = Request(
            url,
            headers={"User-Agent": "leadpoet-validator-artifact-v2/1"},
        )
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if _sha256(temporary) != expected_sha256:
            raise RuntimeArtifactV2Error(
                "runtime artifact hash mismatch: %s" % artifact_name
            )
        os.replace(str(temporary), str(destination))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def stage_artifacts(
    *,
    lock_path: Path,
    output_dir: Path,
    offline_artifact_root: Optional[Path] = None,
    allow_download: bool = False,
) -> Dict[str, Any]:
    lock = load_lock(lock_path)
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    staged = []
    for name, artifact in sorted(lock["artifacts"].items()):
        destination = destination_root / artifact["filename"]
        if not destination.is_file() or _sha256(destination) != artifact["sha256"]:
            offline_source = (
                Path(offline_artifact_root) / artifact["filename"]
                if offline_artifact_root is not None
                else None
            )
            if offline_source is not None and offline_source.is_file():
                _copy_verified_artifact(
                    source=offline_source,
                    destination=destination,
                    expected_sha256=artifact["sha256"],
                    artifact_name=name,
                )
            elif allow_download:
                _download_verified_artifact(
                    url=artifact["url"],
                    destination=destination,
                    expected_sha256=artifact["sha256"],
                    artifact_name=name,
                )
            else:
                raise RuntimeArtifactV2Error(
                    "runtime artifact is unavailable without network access: %s" % name
                )
        if _sha256(destination) != artifact["sha256"]:
            raise RuntimeArtifactV2Error(
                "staged runtime artifact hash mismatch: %s" % name
            )
        os.chmod(str(destination), 0o644)
        os.utime(str(destination), (0, 0))
        staged.append(
            {
                "name": name,
                "filename": artifact["filename"],
                "sha256": artifact["sha256"],
                "size_bytes": destination.stat().st_size,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": staged,
    }
    manifest_path = destination_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(str(manifest_path), 0o644)
    os.utime(str(manifest_path), (0, 0))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--offline-artifact-root", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit HTTPS downloads during explicit artifact preparation only",
    )
    args = parser.parse_args(argv)
    manifest = stage_artifacts(
        lock_path=args.lock,
        output_dir=args.output_dir,
        offline_artifact_root=args.offline_artifact_root,
        allow_download=args.allow_download,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
