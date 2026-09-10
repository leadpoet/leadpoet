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
