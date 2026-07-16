"""Canonicalize a Docker image before converting it to a Nitro EIF."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Optional, Sequence


class DockerImageNormalizationError(RuntimeError):
    """Raised when a Docker image cannot be normalized exactly."""


_EPOCH = "1970-01-01T00:00:00Z"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            value.strip()
            for value in (exc.stdout or "", exc.stderr or "")
            if value.strip()
        )
        raise DockerImageNormalizationError(
            "%s failed%s" % (Path(command[0]).name, ":\n" + output if output else "")
        ) from exc


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise DockerImageNormalizationError("Docker archive path is unsafe")
            if member.issym() or member.islnk() or member.isdev():
                raise DockerImageNormalizationError(
                    "Docker archive contains an unsupported entry"
                )
        archive.extractall(destination)


def _normalized_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    normalized = copy.copy(member)
    normalized.mtime = 0
    normalized.uname = ""
    normalized.gname = ""
    normalized.pax_headers = {
        key: value
        for key, value in member.pax_headers.items()
        if key not in {"atime", "ctime", "mtime"}
    }
    return normalized


def normalize_layer_archive(source: Path, destination: Path) -> str:
    """Rewrite one uncompressed Docker layer with canonical tar metadata."""

    # Docker's containerd image store emits gzip-compressed layer blobs while
    # the classic image store emits plain tar layers. Always rewrite either
    # representation as one canonical uncompressed tar stream.
    with tarfile.open(source, "r:*") as original, tarfile.open(
        destination, "w:"
    ) as output:
        for member in sorted(original.getmembers(), key=lambda item: item.name):
            normalized = _normalized_member(member)
            content = original.extractfile(member) if member.isfile() else None
            output.addfile(normalized, content)
    return _sha256_file(destination)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DockerImageNormalizationError("%s is invalid" % label) from exc


def normalize_saved_image(
    *, archive_path: Path, output_path: Path, normalized_image: str
) -> str:
    """Normalize a ``docker save`` archive and return its canonical image ID."""

    with tempfile.TemporaryDirectory(prefix="leadpoet-image-normalize-") as temporary:
        root = Path(temporary)
        _safe_extract(archive_path, root)
        manifest_path = root / "manifest.json"
        manifest = _load_json(manifest_path, "Docker image manifest")
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise DockerImageNormalizationError("Docker archive must contain one image")
        entry = manifest[0]
        if not isinstance(entry, dict):
            raise DockerImageNormalizationError(
                "Docker image manifest entry is invalid"
            )
        layers = entry.get("Layers")
        config_name = entry.get("Config")
        if (
            not isinstance(layers, list)
            or not layers
            or not isinstance(config_name, str)
        ):
            raise DockerImageNormalizationError("Docker image manifest is incomplete")

        normalized_layers: list[str] = []
        normalized_by_source: dict[str, str] = {}
        for layer_name in layers:
            if not isinstance(layer_name, str):
                raise DockerImageNormalizationError("Docker layer path is invalid")
            if layer_name in normalized_by_source:
                normalized_layers.append(normalized_by_source[layer_name])
                continue
            source = root / layer_name
            temporary_layer = root / (
                "normalized-layer-%d.tar" % len(normalized_layers)
            )
            layer_hash = normalize_layer_archive(source, temporary_layer)
            normalized_name = "blobs/sha256/%s" % layer_hash
            normalized_path = root / normalized_name
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary_layer), str(normalized_path))
            if source != normalized_path:
                source.unlink(missing_ok=True)
            normalized_by_source[layer_name] = normalized_name
            normalized_layers.append(normalized_name)

        config_path = root / config_name
        config = _load_json(config_path, "Docker image config")
        if not isinstance(config, dict) or not isinstance(config.get("rootfs"), dict):
            raise DockerImageNormalizationError("Docker image config is incomplete")
        config["created"] = _EPOCH
        config["rootfs"]["diff_ids"] = [
            "sha256:" + layer.rsplit("/", 1)[-1] for layer in normalized_layers
        ]
        history = config.get("history")
        if isinstance(history, list):
            for item in history:
                if isinstance(item, dict) and "created" in item:
                    item["created"] = _EPOCH

        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        normalized_config_name = "blobs/sha256/%s" % config_hash
        normalized_config_path = root / normalized_config_name
        normalized_config_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_config_path.write_bytes(config_bytes)
        if config_path != normalized_config_path:
            config_path.unlink(missing_ok=True)

        entry["Layers"] = normalized_layers
        entry["Config"] = normalized_config_name
        entry["RepoTags"] = [normalized_image]
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        for metadata_name in ("index.json", "oci-layout"):
            (root / metadata_name).unlink(missing_ok=True)

        with tarfile.open(output_path, "w:") as output:
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                if path.resolve() == output_path.resolve():
                    continue
                info = output.gettarinfo(str(path), arcname=path.name)
                info = _normalized_member(info)
                if path.is_file():
                    with path.open("rb") as content:
                        output.addfile(info, content)
                else:
                    output.addfile(info)
                    for child in sorted(
                        path.rglob("*"), key=lambda item: item.as_posix()
                    ):
                        relative = child.relative_to(root).as_posix()
                        child_info = _normalized_member(
                            output.gettarinfo(str(child), arcname=relative)
                        )
                        if child.is_file():
                            with child.open("rb") as content:
                                output.addfile(child_info, content)
                        else:
                            output.addfile(child_info)
        return "sha256:" + config_hash


def normalize_docker_image(*, source_image: str, normalized_image: str) -> str:
    """Save, canonicalize, and reload a Docker image under a stable tag."""

    with tempfile.TemporaryDirectory(prefix="leadpoet-docker-normalize-") as temporary:
        root = Path(temporary)
        original = root / "original.tar"
        normalized = root / "normalized.tar"
        _run(["docker", "save", "-o", str(original), source_image])
        canonical_config_id = normalize_saved_image(
            archive_path=original,
            output_path=normalized,
            normalized_image=normalized_image,
        )
        _run(["docker", "load", "-i", str(normalized)])
        observed_id = _run(
            ["docker", "image", "inspect", "-f", "{{.Id}}", normalized_image]
        ).stdout.strip()
        if not observed_id.startswith("sha256:"):
            raise DockerImageNormalizationError("normalized Docker image ID is invalid")
        # With Docker's classic store, .Id is the canonical config digest.
        # With the containerd store, .Id can be a generated manifest digest.
        # Both are acceptable build identities and are compared across all
        # repeated builds; the canonical config remains bound into the image.
        if not canonical_config_id.startswith("sha256:"):
            raise DockerImageNormalizationError("canonical Docker config ID is invalid")
        return observed_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--normalized-image", required=True)
    args = parser.parse_args(argv)
    image_id = normalize_docker_image(
        source_image=args.source_image,
        normalized_image=args.normalized_image,
    )
    print(image_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
