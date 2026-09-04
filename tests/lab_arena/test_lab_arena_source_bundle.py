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
    assert set(one) == {"source_size_bytes"}
    assert set(facts) == {"source_size_bytes", "source_root"}
    assert facts["source_size_bytes"] == one["source_size_bytes"]
    assert facts["source_root"] == ""


def test_archive_validation_accepts_one_github_wrapper_directory():
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            data = (
                b'"""Stable public entrypoint for the PydanticAI lead-sourcing harness."""\n'
                b"from experiments.harness_bakeoff.adapters.pydantic_ai import (\n"
                b"    get_last_usage,\n"
                b"    run_icp,\n"
                b")\n"
                b'__all__ = ["get_last_usage", "run_icp"]\n'
            )
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


def test_harness_syntax_accepts_the_public_pydantic_harness_reexport():
    source_bundle.validate_harness_source(
        '"""Stable public entrypoint for the PydanticAI lead-sourcing harness."""\n'
        "from experiments.harness_bakeoff.adapters.pydantic_ai import (\n"
        "    get_last_usage,\n"
        "    run_icp,\n"
        ")\n"
        '__all__ = ["get_last_usage", "run_icp"]\n'
    )


def test_harness_syntax_rejects_invalid_python():
    with pytest.raises(source_bundle.SourceBundleError, match="harness_invalid"):
        source_bundle.validate_harness_source("def run_icp(:\n")


def test_safe_extraction_removes_the_github_wrapper_and_writes_no_links(tmp_path):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in (
                (
                    "pydantic-harness-main/harness.py",
                    b"from package import run_icp\n",
                ),
                ("pydantic-harness-main/package.py", b"def run_icp(icp): return []\n"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    target = tmp_path / "source"
    target.mkdir()
    facts = source_bundle.extract_source_archive(raw.getvalue(), target)
    assert facts["source_root"] == "pydantic-harness-main"
    assert (target / "harness.py").read_text() == "from package import run_icp\n"
    assert not (target / "pydantic-harness-main").exists()


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
