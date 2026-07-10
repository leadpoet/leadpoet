import io
import json
from pathlib import Path
import tarfile

import pytest

from validator_tee.host import gateway_pcr0_builder


def _result(commit, *, pcr0="a" * 96, image="sha256:image", eif="sha256:eif"):
    return {
        "commit_sha": commit,
        "pcr0": pcr0,
        "image_id": image,
        "eif_sha256": eif,
        "source_manifest_hash": "sha256:source",
    }


def test_repeated_builds_must_match_every_identity_field(tmp_path, monkeypatch):
    commit = "1" * 40
    monkeypatch.setattr(gateway_pcr0_builder, "resolve_commit", lambda *_args: commit)
    monkeypatch.setattr(gateway_pcr0_builder, "extract_clean_commit", lambda **kwargs: kwargs["destination"].mkdir())
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "_build_once",
        lambda **kwargs: _result(commit),
    )

    result = gateway_pcr0_builder.build_reproducible_gateway_pcr0(
        repo_root=tmp_path,
        revision="HEAD",
        work_root=tmp_path / "work",
        repetitions=3,
    )
    assert result["verified_build_count"] == 3
    assert result["pcr0"] == "a" * 96


def test_repeated_build_divergence_fails_closed(tmp_path, monkeypatch):
    commit = "2" * 40
    calls = []
    monkeypatch.setattr(gateway_pcr0_builder, "resolve_commit", lambda *_args: commit)
    monkeypatch.setattr(gateway_pcr0_builder, "extract_clean_commit", lambda **kwargs: kwargs["destination"].mkdir())

    def _build(**kwargs):
        calls.append(kwargs["index"])
        return _result(commit, pcr0=("a" if kwargs["index"] == 1 else "b") * 96)

    monkeypatch.setattr(gateway_pcr0_builder, "_build_once", _build)
    with pytest.raises(gateway_pcr0_builder.GatewayPCR0BuildError, match="pcr0"):
        gateway_pcr0_builder.build_reproducible_gateway_pcr0(
            repo_root=tmp_path,
            revision="HEAD",
            work_root=tmp_path / "work",
            repetitions=3,
        )


def test_repeated_builds_require_three_runs(tmp_path):
    with pytest.raises(gateway_pcr0_builder.GatewayPCR0BuildError, match="three"):
        gateway_pcr0_builder.build_reproducible_gateway_pcr0(
            repo_root=tmp_path,
            revision="HEAD",
            work_root=tmp_path / "work",
            repetitions=2,
        )


def test_independent_builder_marks_extracted_git_archive_clean():
    source = Path(gateway_pcr0_builder.__file__).read_text(encoding="utf-8")
    assert '"ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE": "1"' in source


def test_cache_keeps_latest_twenty_verified_commits(tmp_path):
    cache = tmp_path / "cache.json"
    for index in range(25):
        commit = ("%040x" % index)
        gateway_pcr0_builder.write_cache_entry(
            cache_path=cache,
            entry={**_result(commit), "verified_build_count": 3, "role": "gateway_scoring"},
        )
    document = json.loads(cache.read_text())
    assert len(document["entries"]) == 20
    assert document["entries"][0]["commit_sha"] == "%040x" % 24
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, "%040x" % 24)
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, "%040x" % 0) is None


def test_cache_never_evicts_explicitly_pinned_deployed_commit(tmp_path):
    cache = tmp_path / "cache.json"
    deployed_commit = "1" * 40
    gateway_pcr0_builder.write_cache_entry(
        cache_path=cache,
        entry={**_result(deployed_commit), "verified_build_count": 3, "role": "gateway_scoring"},
        pin=True,
    )

    for index in range(25):
        commit = "%040x" % (index + 100)
        gateway_pcr0_builder.write_cache_entry(
            cache_path=cache,
            entry={**_result(commit), "verified_build_count": 3, "role": "gateway_scoring"},
        )

    document = json.loads(cache.read_text())
    assert len(document["entries"]) == 20
    assert document["pinned_commit_shas"] == [deployed_commit]
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, deployed_commit)


def test_cache_rejects_two_build_identity(tmp_path):
    cache = tmp_path / "cache.json"
    commit = "2" * 40
    gateway_pcr0_builder.write_cache_entry(
        cache_path=cache,
        entry={**_result(commit), "verified_build_count": 2, "role": "gateway_scoring"},
    )
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, commit) is None


def test_git_archive_rejects_symlinks(tmp_path):
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("unsafe-link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(gateway_pcr0_builder.GatewayPCR0BuildError, match="non-regular"):
        gateway_pcr0_builder._safe_extract_git_archive(archive_path, tmp_path / "out")


def test_measurement_parser_rejects_debug_pcr0():
    with pytest.raises(gateway_pcr0_builder.GatewayPCR0BuildError):
        gateway_pcr0_builder._parse_measurement(
            json.dumps({"Measurements": {"PCR0": "0" * 96}})
        )
