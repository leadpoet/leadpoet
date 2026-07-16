import hashlib
import io
import json
from pathlib import Path
import tarfile

from validator_tee.host.docker_image_normalizer_v2 import normalize_saved_image


def _layer(path: Path, *, mtime: int, reverse: bool, compressed: bool) -> None:
    entries = [("app/a.txt", b"a"), ("app/b.txt", b"b")]
    if reverse:
        entries.reverse()
    with tarfile.open(path, "w:gz" if compressed else "w:") as archive:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = mtime
            info.uname = "host-user"
            archive.addfile(info, io.BytesIO(content))


def _docker_archive(path: Path, *, mtime: int, reverse: bool, compressed: bool) -> None:
    root = path.parent / (path.stem + "-root")
    layer = root / "blobs/sha256/original-layer"
    layer.parent.mkdir(parents=True)
    _layer(layer, mtime=mtime, reverse=reverse, compressed=compressed)
    config = {
        "created": "2026-07-16T00:00:00Z",
        "history": [{"created": "2026-07-16T00:00:00Z"}],
        "rootfs": {"type": "layers", "diff_ids": ["sha256:old"]},
        "config": {"Entrypoint": ["python3"]},
    }
    config_bytes = json.dumps(config).encode()
    config_name = "blobs/sha256/" + hashlib.sha256(config_bytes).hexdigest()
    (root / config_name).write_bytes(config_bytes)
    (root / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "Config": config_name,
                    "RepoTags": ["raw:test"],
                    "Layers": ["blobs/sha256/original-layer"],
                }
            ]
        )
    )
    with tarfile.open(path, "w:") as archive:
        for item in root.rglob("*"):
            archive.add(item, arcname=item.relative_to(root))


def test_saved_images_with_different_order_and_times_normalize_identically(tmp_path):
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_output = tmp_path / "first-normalized.tar"
    second_output = tmp_path / "second-normalized.tar"
    _docker_archive(first, mtime=100, reverse=False, compressed=False)
    _docker_archive(second, mtime=200, reverse=True, compressed=True)

    first_id = normalize_saved_image(
        archive_path=first,
        output_path=first_output,
        normalized_image="normalized:test",
    )
    second_id = normalize_saved_image(
        archive_path=second,
        output_path=second_output,
        normalized_image="normalized:test",
    )

    assert first_id == second_id
