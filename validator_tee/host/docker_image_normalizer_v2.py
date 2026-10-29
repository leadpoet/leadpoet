"""Canonicalize a Docker image before converting it to a Nitro EIF."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Optional, Sequence


class DockerImageNormalizationError(RuntimeError):
    """Raised when a Docker image cannot be normalized exactly."""


_EPOCH = "1970-01-01T00:00:00Z"


def _emit_canonical_config_diagnostic(config: dict[str, Any], payload: bytes) -> None:
    # Temporary reproduction-only evidence. Remove after identifying the
    # cross-host delta and proving the regression fix on both real builders.
    if os.environ.get("GATEWAY_V2_CANONICAL_CONFIG_DIAGNOSTICS") != "1":
        return
    allowed_env = {
        "PATH", "LANG", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256",
        "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "SOURCE_DATE_EPOCH",
        "LEADPOET_ENCLAVE_ROLE", "PYTHONPATH",
    }
    env = config.get("config", {}).get("Env", [])
    text = payload.decode("utf-8")
    sensitive = re.search(
        r"(?i)(?:password|secret|token|authorization|(?:https?|all)_proxy)[\"\s]*[:=]"
        r"|(?:AKIA|ASIA)[A-Z0-9]{16}|(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]+"
        r"|://[^/\s:\"\\]+:[^/\s@\"\\]+@|BEGIN [A-Z ]*PRIVATE KEY",
        text,
    )
    if (
        len(payload) > 65536
        or not isinstance(env, list)
        or any(not isinstance(item, str) or item.split("=", 1)[0] not in allowed_env for item in env)
        or sensitive
    ):
        print("Canonical Docker config diagnostic withheld by bounds or credential check", file=sys.stderr)
        return
    print("Canonical Docker config diagnostic: " + text, file=sys.stderr)


def _normalization_temp_parent() -> Optional[Path]:
    """Return the root-backed builder workspace when one is configured."""

    for variable in ("GATEWAY_V2_BUILD_WORK_ROOT", "RUNNER_TEMP"):
        raw_value = os.environ.get(variable, "").strip()
        if not raw_value:
            continue
        parent = Path(raw_value)
        if not parent.is_absolute():
            raise DockerImageNormalizationError(
                "%s must be an absolute path" % variable
            )
        parent = parent / ".docker-image-normalizer-v2"
        parent.mkdir(parents=True, exist_ok=True)
        return parent
    return None


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


def normalize_layer_archive(
    source: Path,
    destination: Path,
    *,
    inherited_directories: Optional[dict[str, tuple[Any, ...]]] = None,
) -> str:
    """Rewrite one uncompressed Docker layer with canonical tar metadata."""

    # Docker's containerd image store emits gzip-compressed layer blobs while
    # the classic image store emits plain tar layers. Always rewrite either
    # representation as one canonical uncompressed tar stream.
    with tarfile.open(source, "r:*") as original, tarfile.open(
        destination, "w:"
    ) as output:
        members = sorted(original.getmembers(), key=lambda item: item.name)
        directory_counts: dict[str, int] = {}
        for member in members:
            if member.isdir():
                name = PurePosixPath(member.name).as_posix()
                directory_counts[name] = directory_counts.get(name, 0) + 1
        if inherited_directories is not None:
            # OCI whiteouts affect lower layers only and precede same-layer
            # creations, even when aliases sort before the whiteout record.
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    inherited_directories.clear()
                    continue
                if path.name == ".wh..wh..opq":
                    parent = path.parent.as_posix()
                    prefix = "" if parent == "." else parent + "/"
                    for old in list(inherited_directories):
                        if old != parent and old.startswith(prefix):
                            del inherited_directories[old]
                elif path.name.startswith(".wh."):
                    target = (path.parent / path.name[4:]).as_posix()
                    for old in list(inherited_directories):
                        if old == target or old.startswith(target + "/"):
                            del inherited_directories[old]
        for member in members:
            normalized = _normalized_member(member)
            if inherited_directories is not None:
                path = PurePosixPath(normalized.name)
                if path.is_absolute() or ".." in path.parts:
                    # Keep unfamiliar paths and discard comparison state.
                    inherited_directories.clear()
                else:
                    name = path.as_posix()
                    basename = path.name
                    if basename.startswith(".wh."):
                        pass  # Already applied to inherited state, not this layer.
                    elif normalized.isdir():
                        metadata = tuple(getattr(normalized, field) for field in (
                            "mode", "uid", "gid", "size", "mtime", "type",
                            "linkname", "uname", "gname", "devmajor", "devminor",
                        )) + (tuple(sorted(normalized.pax_headers.items())),)
                        redundant = (
                            directory_counts[name] == 1
                            and inherited_directories.get(name) == metadata
                        )
                        if redundant:
                            continue
                        inherited_directories[name] = metadata
                    else:
                        # A file, symlink, or device replaces any old directory.
                        for old in list(inherited_directories):
                            if old == name or old.startswith(name + "/"):
                                del inherited_directories[old]
            content = original.extractfile(member) if member.isfile() else None
            output.addfile(normalized, content)
    return _sha256_file(destination)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DockerImageNormalizationError("%s is invalid" % label) from exc


def normalize_saved_image(
    *,
    archive_path: Path,
    output_path: Path,
    normalized_image: str,
    temporary_parent: Optional[Path] = None,
) -> str:
    """Normalize a ``docker save`` archive and return its canonical image ID."""

    parent = temporary_parent or _normalization_temp_parent()
    with tempfile.TemporaryDirectory(
        prefix="leadpoet-image-normalize-",
        dir=None if parent is None else str(parent),
    ) as temporary:
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
        inherited_directories: dict[str, tuple[Any, ...]] = {}
        source_layers: set[Path] = set()
        for layer_name in layers:
            if not isinstance(layer_name, str):
                raise DockerImageNormalizationError("Docker layer path is invalid")
            source = root / layer_name
            source_layers.add(source)
            temporary_layer = root / (
                "normalized-layer-%d.tar" % len(normalized_layers)
            )
            layer_hash = normalize_layer_archive(
                source, temporary_layer, inherited_directories=inherited_directories
            )
            normalized_name = "blobs/sha256/%s" % layer_hash
            normalized_path = root / normalized_name
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary_layer), str(normalized_path))
            normalized_layers.append(normalized_name)
        # A repeated source layer can normalize differently after earlier
        # layers change inherited state. Keep every source until all uses end.
        for source in source_layers - {root / name for name in normalized_layers}:
            source.unlink(missing_ok=True)

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
        _emit_canonical_config_diagnostic(config, config_bytes)
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


def _saved_image_config_hash(archive_path: Path) -> str:
    """Read the exact config bytes exported by Docker after loading an image."""

    try:
        with tarfile.open(archive_path, "r:*") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            if (
                not isinstance(manifest, list)
                or len(manifest) != 1
                or not isinstance(manifest[0], dict)
            ):
                raise DockerImageNormalizationError("loaded Docker manifest is invalid")
            config_name = manifest[0].get("Config")
            if not isinstance(config_name, str) or not config_name:
                raise DockerImageNormalizationError("loaded Docker config path is invalid")
            config = archive.extractfile(config_name)
            if config is None:
                raise DockerImageNormalizationError("loaded Docker config is unavailable")
            digest = hashlib.sha256()
            with config:
                for chunk in iter(lambda: config.read(1024 * 1024), b""):
                    digest.update(chunk)
            return "sha256:" + digest.hexdigest()
    except (OSError, KeyError, ValueError, TypeError, tarfile.TarError) as exc:
        raise DockerImageNormalizationError("loaded Docker config is invalid") from exc


def normalize_docker_image(*, source_image: str, normalized_image: str) -> str:
    """Save, canonicalize, and reload a Docker image under a stable tag."""

    parent = _normalization_temp_parent()
    with tempfile.TemporaryDirectory(
        prefix="leadpoet-docker-normalize-",
        dir=None if parent is None else str(parent),
    ) as temporary:
        root = Path(temporary)
        original = root / "original.tar"
        normalized = root / "normalized.tar"
        reloaded = root / "reloaded.tar"
        _run(["docker", "save", "-o", str(original), source_image])
        canonical_config_id = normalize_saved_image(
            archive_path=original,
            output_path=normalized,
            normalized_image=normalized_image,
            temporary_parent=root,
        )
        _run(["docker", "load", "-i", str(normalized)])
        observed_id = _run(
            ["docker", "image", "inspect", "-f", "{{.Id}}", normalized_image]
        ).stdout.strip()
        if not observed_id.startswith("sha256:"):
            raise DockerImageNormalizationError("normalized Docker image ID is invalid")
        # .Id can be a config digest in the classic store or a generated
        # manifest digest in the containerd store. Compare the exact loaded
        # config instead: it binds all config fields and ordered layer hashes.
        if not canonical_config_id.startswith("sha256:"):
            raise DockerImageNormalizationError("canonical Docker config ID is invalid")
        original.unlink()
        normalized.unlink()
        _run(["docker", "save", "-o", str(reloaded), normalized_image])
        if _saved_image_config_hash(reloaded) != canonical_config_id:
            raise DockerImageNormalizationError("loaded Docker config differs from canonical config")
        print(
            "Verified canonical Docker config %s (store image %s)"
            % (canonical_config_id, observed_id),
            file=sys.stderr,
        )
        return canonical_config_id


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
