from __future__ import annotations

import gzip
import io
import tarfile

import pytest

from lab_arena import source_bundle


def test_source_archive_is_deterministic_and_needs_no_dockerfile(tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "harness.py").write_text("def run_icp(icp):\n    return []\n")
    (source / "logic.py").write_text("VALUE = 1\n")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    one = source_bundle.write_source_archive(source, first)
    (source / "logic.py").touch()
    two = source_bundle.write_source_archive(source, second)
    assert first.read_bytes() == second.read_bytes()
    facts = source_bundle.validate_source_archive(first.read_bytes())
    assert one == two
    assert facts["source_sha256"] == one["source_sha256"]
    assert facts["source_size_bytes"] == one["source_size_bytes"]
    assert facts["source_root"] == ""


def test_archive_validation_accepts_one_github_wrapper_directory():
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            data = b"def run_icp(icp):\n    return []\n"
            info = tarfile.TarInfo("pydantic-harness-main/harness.py")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    facts = source_bundle.validate_source_archive(raw.getvalue())
    assert facts["source_root"] == "pydantic-harness-main"


def test_archive_validation_rejects_links_traversal_and_missing_harness():
    for name, kind in (("../harness.py", "file"), ("harness.py", "link"), ("logic.py", "file")):
        raw = io.BytesIO()
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                info = tarfile.TarInfo(name)
                if kind == "link":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/etc/passwd"
                    archive.addfile(info)
                else:
                    data = b"def run_icp(icp):\n    return []\n"
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        with pytest.raises(source_bundle.SourceBundleError):
            source_bundle.validate_source_archive(raw.getvalue())


@pytest.mark.parametrize(
    "definition",
    [
        "async def run_icp(icp):\n    return []\n",
        "def run_icp():\n    return []\n",
        "def run_icp(icp, other):\n    return []\n",
        "def run_icp(icp, *args):\n    return []\n",
        "def run_icp(icp, **kwargs):\n    return []\n",
        "def run_icp(icp, *, option=None):\n    return []\n",
    ],
)
def test_harness_requires_one_synchronous_positional_parameter(definition):
    with pytest.raises(source_bundle.SourceBundleError):
        source_bundle.validate_harness_source(definition)


def test_archive_member_count_is_bounded_while_streaming():
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for index in range(source_bundle.MAX_SOURCE_FILES + 1):
                info = tarfile.TarInfo("empty-%04d" % index)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
    with pytest.raises(
        source_bundle.SourceBundleError, match="source_file_count_exceeded"
    ):
        source_bundle.validate_source_archive(raw.getvalue())
