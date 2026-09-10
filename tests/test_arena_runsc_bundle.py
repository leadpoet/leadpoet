import hashlib
import io
import tarfile

import pytest

from scripts import _lab_arena_runsc_probe_ci as probe


def _bundle(extra="gvisor-bin/gvisor_sentry"):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name in ("runsc", extra):
            member = tarfile.TarInfo(name)
            member.mode = 0o755
            member.size = 4
            archive.addfile(member, io.BytesIO(b"test"))
    return output.getvalue()


def _downloads(monkeypatch, data, checksum=None):
    checksum = checksum or hashlib.sha512(data).hexdigest()
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(
        (checksum + "  gvisor.tar.bz2\n").encode() if request.full_url.endswith(".sha512") else data
    ))


def test_installs_runtime_and_required_companion_together(tmp_path, monkeypatch):
    _downloads(monkeypatch, _bundle())
    assert probe.probe_runsc(tmp_path / "runsc") == tmp_path / "runsc"
    assert (tmp_path / "gvisor-bin/gvisor_sentry").read_bytes() == b"test"


def test_rejects_unverified_bundle_before_extracting(tmp_path, monkeypatch):
    _downloads(monkeypatch, _bundle(), "0" * 128)
    with pytest.raises(RuntimeError, match="checksum"):
        probe.probe_runsc(tmp_path / "runsc")
    assert not list(tmp_path.iterdir())


def test_rejects_archive_escape_before_extracting(tmp_path, monkeypatch):
    _downloads(monkeypatch, _bundle("../outside"))
    with pytest.raises(RuntimeError, match="unsafe path"):
        probe.probe_runsc(tmp_path / "runsc")
    assert not list(tmp_path.iterdir())


def test_probe_root_is_independent_and_contains_closed_shim(tmp_path):
    rootfs = tmp_path / "rootfs"
    for relative in ("model", "input", "output", "run/lab_arena"):
        (rootfs / relative).mkdir(parents=True, exist_ok=True)
    spec = probe.make_spec(
        tmp_path, "ok", probe.MODEL_OK, wall_clock=1, rootfs_path=rootfs
    )
    assert spec.rootfs_path == rootfs
    assert spec.argv[-1] == "/model/ok/main.py"
    assert (rootfs / "model/lab_arena/contracts.py").is_file()
    assert (rootfs / "model/lab_arena/operations.py").is_file()
    assert (rootfs / "model/lab_arena/shim.py").is_file()


def test_probe_cleanup_keeps_directory_when_bind_unmount_failed(tmp_path, monkeypatch):
    work = tmp_path / "probe work"
    (work / "host-rootfs/usr").mkdir(parents=True)
    canary = work / "host-rootfs/usr/keep"
    canary.write_text("host content")
    mountinfo = tmp_path / "mountinfo"
    target = str(work.resolve() / "host-rootfs/usr").replace(" ", "\\040")
    mountinfo.write_text("1 0 8:1 /usr %s rw - ext4 /dev/root rw\n" % target)
    monkeypatch.setattr(probe, "MOUNTINFO", mountinfo)
    with pytest.raises(RuntimeError, match="mount remains"):
        probe.cleanup_probe_work_dir(work, inspect_mounts=True)
    assert canary.read_text() == "host content"


def test_probe_cleanup_removes_only_unmounted_work(tmp_path, monkeypatch):
    work = tmp_path / "probe"
    work.mkdir()
    (work / "fixture").write_text("test")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("1 0 8:1 / / rw - ext4 /dev/root rw\n")
    monkeypatch.setattr(probe, "MOUNTINFO", mountinfo)
    probe.cleanup_probe_work_dir(work, inspect_mounts=True)
    assert not work.exists()
    assert mountinfo.is_file()
