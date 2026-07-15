import io
import json
from pathlib import Path
import tarfile

import pytest

from validator_tee.host import gateway_pcr0_builder


def _result(
    commit,
    *,
    role="gateway_coordinator",
    pcr0="a" * 96,
    image="sha256:image",
    eif="sha256:eif",
):
    return {
        "commit_sha": commit,
        "role": role,
        "pcr0": pcr0,
        "image_id": image,
        "eif_sha256": eif,
        "source_manifest_hash": "sha256:source",
        "build_identity_hash": "sha256:identity",
        "execution_manifest_hash": "sha256:execution",
        "dependency_lock_hash": "sha256:dependencies",
        "dockerfile_hash": "sha256:dockerfile",
        "topology_hash": "sha256:topology",
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
    assert [row["build_ordinal"] for row in result["build_evidence"]] == [1, 2, 3]
    assert {row["builder_domain"] for row in result["build_evidence"]} == {
        "validator"
    }
    assert {row["physical_role"] for row in result["build_evidence"]} == {
        "gateway_coordinator"
    }


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
            entry={**_result(commit), "verified_build_count": 3},
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
        entry={**_result(deployed_commit), "verified_build_count": 3},
        pin=True,
    )

    for index in range(25):
        commit = "%040x" % (index + 100)
        gateway_pcr0_builder.write_cache_entry(
            cache_path=cache,
            entry={**_result(commit), "verified_build_count": 3},
        )

    document = json.loads(cache.read_text())
    assert len(document["entries"]) == 20
    assert document["pinned_deployments"] == [
        {"role": "gateway_coordinator", "commit_sha": deployed_commit}
    ]
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, deployed_commit)


def test_cache_rejects_two_build_identity(tmp_path):
    cache = tmp_path / "cache.json"
    commit = "2" * 40
    gateway_pcr0_builder.write_cache_entry(
        cache_path=cache,
        entry={**_result(commit), "verified_build_count": 2},
    )
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, commit) is None


def test_cache_retains_twenty_commits_per_physical_role(tmp_path):
    cache = tmp_path / "cache.json"
    for role_index, role in enumerate(gateway_pcr0_builder.GATEWAY_ROLES):
        for index in range(25):
            commit = "%040x" % (role_index * 1000 + index)
            gateway_pcr0_builder.write_cache_entry(
                cache_path=cache,
                entry={
                    **_result(commit, role=role),
                    "verified_build_count": 3,
                },
            )
    document = json.loads(cache.read_text())
    assert len(document["entries"]) == 60
    for role in gateway_pcr0_builder.GATEWAY_ROLES:
        assert len([row for row in document["entries"] if row["role"] == role]) == 20


def test_same_commit_requires_explicit_role_when_cache_has_multiple_eifs(tmp_path):
    cache = tmp_path / "cache.json"
    commit = "9" * 40
    for role in ("gateway_coordinator", "gateway_scoring"):
        gateway_pcr0_builder.write_cache_entry(
            cache_path=cache,
            entry={
                **_result(commit, role=role, pcr0=("a" if role.endswith("a") else "b") * 96),
                "verified_build_count": 3,
            },
        )
    assert gateway_pcr0_builder.load_cached_gateway_identity(cache, commit) is None
    assert gateway_pcr0_builder.load_cached_gateway_identity(
        cache,
        commit,
        role="gateway_scoring",
    )["role"] == "gateway_scoring"


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
