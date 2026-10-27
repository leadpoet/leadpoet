"""OCI members replace destination symlinks without following host targets."""
import os
from pathlib import Path

import pytest

from lab_arena import images
from tests.lab_arena.test_lab_arena_images import layer_tar


def apply(tmp_path, rootfs, entries, name):
    archive = tmp_path / (name + ".tar.gz")
    archive.write_bytes(layer_tar(entries))
    return images.apply_layer(archive, rootfs, compressed=True, budget_bytes=1024 * 1024)


@pytest.mark.parametrize("replacement", ["symlink", "file", "dir", "hardlink"])
def test_later_layer_replaces_only_a_destination_symlink_with_a_symlink(tmp_path, replacement):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    member = "usr/share/zoneinfo/localtime"
    apply(tmp_path, rootfs, [(member, "symlink", "/etc/localtime")], "base")
    entries = {
        "symlink": [(member, "symlink", "/etc/localtime")],
        "file": [(member, "file", b"new timezone")],
        "dir": [(member, "dir")],
        "hardlink": [("timezone", "file", b"new timezone"), (member, "hardlink", "timezone")],
    }[replacement]
    if replacement != "symlink":
        with pytest.raises(images.ImageError) as failure:
            apply(tmp_path, rootfs, entries, "next")
        assert failure.value.rule_id == images.RULE_LAYER_INVALID
        assert os.readlink(rootfs / member) == "/etc/localtime"
        return
    apply(tmp_path, rootfs, entries, "next")
    target = rootfs / member
    assert target.is_symlink() and os.readlink(target) == "/etc/localtime"


def test_same_layer_replacement_does_not_write_the_outside_symlink_target(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"host sentinel")
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, [("timezone", "symlink", str(outside)),
                                ("timezone", "file", b"image timezone")], "same")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert outside.read_bytes() == b"host sentinel"
    assert (rootfs / "timezone").is_symlink()


@pytest.mark.parametrize("kind", ["file", "dir", "symlink", "hardlink", "whiteout", "opaque"])
def test_symlink_ancestor_cannot_write_or_remove_outside_members(tmp_path, kind):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_bytes(b"host sentinel")
    apply(tmp_path, rootfs, [("parent", "symlink", str(outside)),
                            ("inside", "file", b"image")], "base")
    entries = {
        "file": [("parent/victim", "file", b"overwrite")],
        "dir": [("parent/new", "dir")],
        "symlink": [("parent/victim", "symlink", "/etc/localtime")],
        "hardlink": [("parent/new", "hardlink", "inside")],
        "whiteout": [("parent/.wh.victim", "file", b"")],
        "opaque": [("parent/.wh..wh..opq", "file", b"")],
    }[kind]
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, entries, "escape")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert victim.read_bytes() == b"host sentinel"
    assert sorted(p.name for p in outside.iterdir()) == ["victim"]
    assert (rootfs / "parent").is_symlink()


def test_invalid_hardlink_is_rejected_before_unlinking_destination(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"host sentinel")
    apply(tmp_path, rootfs, [("destination", "symlink", "/etc/localtime")], "base")
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, [("destination", "hardlink", "../outside")], "escape")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert os.readlink(rootfs / "destination") == "/etc/localtime"
    assert outside.read_bytes() == b"host sentinel"


def test_internal_symlink_ancestor_remains_usable(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    apply(tmp_path, rootfs, [("real", "dir"), ("alias", "symlink", "real"),
                            ("alias/file", "file", b"image")], "internal")
    assert (rootfs / "real/file").read_bytes() == b"image"
    assert (rootfs / "alias").is_symlink()


def test_absolute_hardlink_cannot_import_a_host_file(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "host-secret"
    outside.write_bytes(b"host sentinel")
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, [("imported", "hardlink", str(outside))], "absolute")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert not (rootfs / "imported").exists()
    assert outside.read_bytes() == b"host sentinel"


def test_relative_hardlink_through_outside_symlink_cannot_import_host_file(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "host-secret"
    outside.write_bytes(b"host sentinel")
    apply(tmp_path, rootfs, [("source", "symlink", str(outside))], "base")
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, [("imported", "hardlink", "source")], "hardlink")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert not (rootfs / "imported").exists()
    assert outside.read_bytes() == b"host sentinel"


def test_symlink_unlink_failure_stops_before_tarfile_fallback(tmp_path, monkeypatch):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "host-secret"
    outside.write_bytes(b"host sentinel")
    apply(tmp_path, rootfs, [("destination", "symlink", str(outside))], "base")
    original_unlink = Path.unlink
    extracts = []

    def refuse_destination_unlink(path, *args, **kwargs):
        if path == rootfs / "destination":
            raise PermissionError("test refuses destination unlink")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_destination_unlink)
    monkeypatch.setattr(images.tarfile.TarFile, "extract", lambda *_args, **_kwargs: extracts.append(True))
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, [("destination", "symlink", "/etc/localtime")], "replacement")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert isinstance(failure.value.__cause__, PermissionError)
    assert extracts == []
    assert os.readlink(rootfs / "destination") == str(outside)
    assert outside.read_bytes() == b"host sentinel"


def test_late_child_write_through_replaced_outside_symlink_is_rejected(tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim").write_bytes(b"host sentinel")
    apply(tmp_path, rootfs, [("destination", "symlink", "/etc/localtime")], "base")
    with pytest.raises(images.ImageError) as failure:
        apply(tmp_path, rootfs, [("destination", "symlink", str(outside)),
                                ("destination/victim", "file", b"overwrite")], "replacement")
    assert failure.value.rule_id == images.RULE_LAYER_INVALID
    assert (outside / "victim").read_bytes() == b"host sentinel"
