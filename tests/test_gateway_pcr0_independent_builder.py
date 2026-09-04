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
        "rootfs_layer_hashes": ["sha256:" + "1" * 64],
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
    monkeypatch.setattr(gateway_pcr0_builder, "_prune_builder_cache", lambda: None)
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
    monkeypatch.setattr(gateway_pcr0_builder, "_prune_builder_cache", lambda: None)

    def _build(**kwargs):
        calls.append(kwargs["index"])
        return _result(commit, pcr0=("a" if kwargs["index"] == 1 else "b") * 96)

    monkeypatch.setattr(gateway_pcr0_builder, "_build_once", _build)
    with pytest.raises(
        gateway_pcr0_builder.GatewayPCR0BuildError,
        match=r'pcr0: \["a{96}","b{96}","b{96}"\]',
    ):
        gateway_pcr0_builder.build_reproducible_gateway_pcr0(
            repo_root=tmp_path,
            revision="HEAD",
            work_root=tmp_path / "work",
            repetitions=3,
        )


def test_repeated_build_eif_metadata_may_differ_when_identity_matches(
    tmp_path, monkeypatch
):
    commit = "3" * 40
    monkeypatch.setattr(gateway_pcr0_builder, "resolve_commit", lambda *_args: commit)
    monkeypatch.setattr(gateway_pcr0_builder, "extract_clean_commit", lambda **kwargs: kwargs["destination"].mkdir())
    monkeypatch.setattr(gateway_pcr0_builder, "_prune_builder_cache", lambda: None)

    def _build(**kwargs):
        result = _result(commit)
        result["eif_sha256"] = "sha256:" + str(kwargs["index"]) * 64
        return result

    monkeypatch.setattr(gateway_pcr0_builder, "_build_once", _build)
    result = gateway_pcr0_builder.build_reproducible_gateway_pcr0(
        repo_root=tmp_path,
        revision="HEAD",
        work_root=tmp_path / "work",
    )

    assert [row["eif_hash"] for row in result["build_evidence"]] == [
        "sha256:" + str(index) * 64 for index in (1, 2, 3)
    ]


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


def test_independent_builder_does_not_invent_resource_exhaustion(monkeypatch):
    failures = []
    monkeypatch.setattr(gateway_pcr0_builder, "init_sentry", lambda **_kwargs: None)
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "configure_sentry_context",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "main",
        lambda: (_ for _ in ()).throw(RuntimeError("unclassified build failure")),
    )
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "capture_failure",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="unclassified build failure"):
        gateway_pcr0_builder._run_cli()

    assert failures[0][0] == ("restart.terminal_failure",)


def test_gateway_build_command_binds_reproducible_epoch(tmp_path):
    command = gateway_pcr0_builder._deterministic_docker_build_command(
        gateway_root=tmp_path / "gateway",
        image="leadpoet-gateway-verify:test",
        role="gateway_coordinator",
    )

    assert command.count("--build-arg") == 2
    assert "SOURCE_DATE_EPOCH=0" in command
    assert "LEADPOET_ENCLAVE_ROLE=gateway_coordinator" in command


def test_normalized_rootfs_layer_hashes_are_strict(monkeypatch):
    layer = "sha256:" + "a" * 64
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "_run",
        lambda *_args, **_kwargs: type("Result", (), {"stdout": json.dumps([layer])})(),
    )

    assert gateway_pcr0_builder._normalized_rootfs_layer_hashes("image") == [layer]


def test_normalized_rootfs_layer_hashes_reject_invalid_identity(monkeypatch):
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "_run",
        lambda *_args, **_kwargs: type("Result", (), {"stdout": '["latest"]'})(),
    )

    with pytest.raises(
        gateway_pcr0_builder.GatewayPCR0BuildError,
        match="layer identity is invalid",
    ):
        gateway_pcr0_builder._normalized_rootfs_layer_hashes("image")


def test_builder_cache_prune_is_fail_closed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "_run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    gateway_pcr0_builder._prune_builder_cache()

    assert calls == [
        (["docker", "builder", "prune", "-af"], {"timeout": 600})
    ]


def test_repeated_builds_prune_before_ordinal_one(tmp_path, monkeypatch):
    commit = "5" * 40
    events = []
    monkeypatch.setattr(gateway_pcr0_builder, "resolve_commit", lambda *_args: commit)
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "extract_clean_commit",
        lambda **kwargs: kwargs["destination"].mkdir(),
    )
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "_prune_builder_cache",
        lambda: events.append("prune"),
    )

    def _build(**kwargs):
        events.append("build-%s" % kwargs["index"])
        return _result(commit)

    monkeypatch.setattr(gateway_pcr0_builder, "_build_once", _build)
    gateway_pcr0_builder.build_reproducible_gateway_pcr0(
        repo_root=tmp_path,
        revision="HEAD",
        work_root=tmp_path / "work",
    )

    assert events == ["prune", "build-1", "build-2", "build-3"]


def test_required_builder_prune_failure_aborts_before_build(tmp_path, monkeypatch):
    commit = "6" * 40
    builds = []
    monkeypatch.setattr(gateway_pcr0_builder, "resolve_commit", lambda *_args: commit)
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "extract_clean_commit",
        lambda **kwargs: kwargs["destination"].mkdir(),
    )

    def _fail_prune():
        raise gateway_pcr0_builder.GatewayPCR0BuildError("builder prune failed")

    monkeypatch.setattr(gateway_pcr0_builder, "_prune_builder_cache", _fail_prune)
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "_build_once",
        lambda **kwargs: builds.append(kwargs["index"]),
    )

    with pytest.raises(
        gateway_pcr0_builder.GatewayPCR0BuildError,
        match="builder prune failed",
    ):
        gateway_pcr0_builder.build_reproducible_gateway_pcr0(
            repo_root=tmp_path,
            revision="HEAD",
            work_root=tmp_path / "work",
        )

    assert builds == []


def test_divergence_reports_every_safe_build_identity(tmp_path, monkeypatch):
    commit = "7" * 40
    monkeypatch.setattr(gateway_pcr0_builder, "resolve_commit", lambda *_args: commit)
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "extract_clean_commit",
        lambda **kwargs: kwargs["destination"].mkdir(),
    )
    monkeypatch.setattr(gateway_pcr0_builder, "_prune_builder_cache", lambda: None)

    def _build(**kwargs):
        index = kwargs["index"]
        result = _result(
            commit,
            pcr0=("a" if index == 1 else "b") * 96,
            image="sha256:" + ("1" if index == 1 else "2") * 64,
        )
        result["rootfs_layer_hashes"] = ["sha256:" + ("3" if index == 1 else "4") * 64]
        result["source_manifest_hash"] = "sha256:" + ("5" if index == 1 else "6") * 64
        return result

    monkeypatch.setattr(gateway_pcr0_builder, "_build_once", _build)
    with pytest.raises(gateway_pcr0_builder.GatewayPCR0BuildError) as raised:
        gateway_pcr0_builder.build_reproducible_gateway_pcr0(
            repo_root=tmp_path,
            revision="HEAD",
            work_root=tmp_path / "work",
        )

    message = str(raised.value)
    assert '"pcr0"' in message
    assert '"image_id"' in message
    assert '"rootfs_layer_hashes"' in message
    assert '"source_manifest_hash"' in message


def test_gateway_builder_normalizes_image_before_eif():
    source = Path(gateway_pcr0_builder.__file__).read_text(encoding="utf-8")

    assert "normalize_docker_image(" in source
    assert "source_image=raw_image" in source
    assert "normalized_image=image" in source


def test_gateway_builder_discards_large_intermediate_artifacts():
    source = Path(gateway_pcr0_builder.__file__).read_text(encoding="utf-8")

    assert "eif_path.unlink()" in source
    assert '["docker", "builder", "prune", "-af"]' in source
    build = source.index("_deterministic_docker_build_command(")
    pre_normalization_prune = source.index(
        '["docker", "builder", "prune", "-af"]', build
    )
    normalization = source.index("image_id = normalize_docker_image(", build)
    assert build < pre_normalization_prune < normalization
    assert 'builder", "prune", "-af"], check=False' not in source


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
    assert len(document["entries"]) == 20 * len(
        gateway_pcr0_builder.GATEWAY_ROLES
    )
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


def test_machine_result_file_is_isolated_from_stdout_diagnostics(
    tmp_path, monkeypatch, capsys
):
    commit = "4" * 40
    output = tmp_path / "result.json"

    def build(**kwargs):
        print("optional dependency diagnostic")
        role = kwargs["role"]
        return {
            **_result(commit, role=role),
            "verified_build_count": 3,
            "build_evidence": [],
        }

    monkeypatch.setattr(
        gateway_pcr0_builder,
        "build_reproducible_gateway_pcr0",
        build,
    )
    monkeypatch.setattr(
        gateway_pcr0_builder,
        "write_cache_entry",
        lambda **_kwargs: None,
    )

    assert gateway_pcr0_builder.main(
        [
            "--repo-root",
            str(tmp_path),
            "--revision",
            commit,
            "--work-root",
            str(tmp_path / "work"),
            "--cache-file",
            str(tmp_path / "cache.json"),
            "--builder-domain",
            "gateway",
            "--builder-id",
            "gateway-parent-test",
            "--all-roles",
            "--output-file",
            str(output),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert captured.out.count("optional dependency diagnostic") == len(
        gateway_pcr0_builder.GATEWAY_ROLES
    )
    records = json.loads(output.read_text(encoding="utf-8"))
    assert [record["role"] for record in records] == list(
        gateway_pcr0_builder.GATEWAY_ROLES
    )
    assert output.stat().st_mode & 0o777 == 0o600
