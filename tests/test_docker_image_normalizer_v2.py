import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from validator_tee.host import docker_image_normalizer_v2 as normalizer


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


def _docker_archive(
    path: Path,
    *,
    mtime: int,
    reverse: bool,
    compressed: bool,
) -> None:
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


def test_saved_images_with_different_order_and_times_normalize_identically(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_output = tmp_path / "first-normalized.tar"
    second_output = tmp_path / "second-normalized.tar"
    _docker_archive(first, mtime=100, reverse=False, compressed=False)
    _docker_archive(second, mtime=200, reverse=True, compressed=True)

    first_id = normalizer.normalize_saved_image(
        archive_path=first,
        output_path=first_output,
        normalized_image="normalized:test",
    )
    second_id = normalizer.normalize_saved_image(
        archive_path=second,
        output_path=second_output,
        normalized_image="normalized:test",
    )

    assert first_id == second_id


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def test_normalizer_uses_root_backed_build_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_root = tmp_path / "gateway-build-work"
    runner_temp = tmp_path / "runner-temp"
    monkeypatch.setenv("GATEWAY_V2_BUILD_WORK_ROOT", str(build_root))
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    saved_archive: list[Path] = []
    nested_parents: list[Path] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "save", "-o"]:
            archive = Path(command[3])
            archive.write_bytes(b"docker-save-placeholder")
            saved_archive.append(archive)
            return _completed()
        if command[:4] == ["docker", "image", "inspect", "-f"]:
            return _completed("sha256:" + "a" * 64 + "\n")
        return _completed()

    def fake_normalize_saved_image(
        *,
        archive_path: Path,
        output_path: Path,
        normalized_image: str,
        temporary_parent: Path,
    ) -> str:
        assert archive_path == saved_archive[0]
        assert normalized_image == "normalized:test"
        nested_parents.append(temporary_parent)
        output_path.write_bytes(b"normalized-placeholder")
        return "sha256:" + "b" * 64

    monkeypatch.setattr(normalizer, "_run", fake_run)
    monkeypatch.setattr(
        normalizer,
        "normalize_saved_image",
        fake_normalize_saved_image,
    )
    monkeypatch.setattr(
        normalizer, "_saved_image_config_hash", lambda path: "sha256:" + "b" * 64
    )

    observed = normalizer.normalize_docker_image(
        source_image="source:test",
        normalized_image="normalized:test",
    )

    expected_parent = build_root / ".docker-image-normalizer-v2"
    assert observed == "sha256:" + "b" * 64
    assert saved_archive[0].parent.parent == expected_parent
    assert nested_parents == [saved_archive[0].parent]
    assert not runner_temp.exists()
    assert list(expected_parent.iterdir()) == []


def test_normalizer_uses_runner_temp_when_build_root_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner_temp = tmp_path / "runner-temp"
    monkeypatch.delenv("GATEWAY_V2_BUILD_WORK_ROOT", raising=False)
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    assert normalizer._normalization_temp_parent() == (
        runner_temp / ".docker-image-normalizer-v2"
    )


def test_normalizer_rejects_relative_build_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_V2_BUILD_WORK_ROOT", "relative/build-root")

    with pytest.raises(
        normalizer.DockerImageNormalizationError,
        match="GATEWAY_V2_BUILD_WORK_ROOT must be an absolute path",
    ):
        normalizer._normalization_temp_parent()


@pytest.mark.parametrize("store", ["classic", "containerd"])
def test_normalizer_returns_verified_config_digest_for_each_image_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store: str,
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.delenv("GATEWAY_V2_BUILD_WORK_ROOT", raising=False)
    loaded = tmp_path / "loaded.tar"
    config_ids = []
    saves = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "save", "-o"]:
            saves.append(command[-1])
            if command[-1] == "source:test":
                _docker_archive(
                    Path(command[3]), mtime=100, reverse=False, compressed=False
                )
            else:
                shutil.copyfile(loaded, command[3])
        elif command[:3] == ["docker", "load", "-i"]:
            shutil.copyfile(command[3], loaded)
            with tarfile.open(loaded) as archive:
                entry = json.load(archive.extractfile("manifest.json"))[0]
                config_ids.append(
                    "sha256:" + hashlib.sha256(
                        archive.extractfile(entry["Config"]).read()
                    ).hexdigest()
                )
        elif command[:4] == ["docker", "image", "inspect", "-f"]:
            store_id = config_ids[0] if store == "classic" else "sha256:" + "a" * 64
            return _completed(store_id + "\n")
        return _completed()

    monkeypatch.setattr(normalizer, "_run", fake_run)
    observed = normalizer.normalize_docker_image(
        source_image="source:test", normalized_image="normalized:test"
    )

    assert observed == config_ids[0]
    assert saves == ["source:test", "normalized:test"]


def test_normalizer_rejects_loaded_config_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.delenv("GATEWAY_V2_BUILD_WORK_ROOT", raising=False)
    loaded = tmp_path / "loaded.tar"

    def export_changed_entrypoint(destination: Path) -> None:
        with tarfile.open(loaded) as source, tarfile.open(destination, "w:") as output:
            manifest = json.load(source.extractfile("manifest.json"))
            old_config_name = manifest[0]["Config"]
            config = json.load(source.extractfile(old_config_name))
            assert config["config"]["Entrypoint"] == ["python3"]
            config["config"]["Entrypoint"] = ["sh"]
            config_bytes = json.dumps(
                config, sort_keys=True, separators=(",", ":")
            ).encode()
            new_config_name = "blobs/sha256/" + hashlib.sha256(config_bytes).hexdigest()
            manifest[0]["Config"] = new_config_name
            for original in source.getmembers():
                member = copy.copy(original)
                data = source.extractfile(original).read() if original.isfile() else None
                if original.name == old_config_name:
                    member.name = new_config_name
                    data = config_bytes
                elif original.name == "manifest.json":
                    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                if data is not None:
                    member.size = len(data)
                output.addfile(member, None if data is None else io.BytesIO(data))

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "save", "-o"]:
            if command[-1] == "source:test":
                _docker_archive(
                    Path(command[3]), mtime=100, reverse=False, compressed=False
                )
            else:
                export_changed_entrypoint(Path(command[3]))
        elif command[:3] == ["docker", "load", "-i"]:
            shutil.copyfile(command[3], loaded)
        elif command[:4] == ["docker", "image", "inspect", "-f"]:
            return _completed("sha256:" + "a" * 64 + "\n")
        return _completed()

    monkeypatch.setattr(normalizer, "_run", fake_run)
    with pytest.raises(
        normalizer.DockerImageNormalizationError, match="loaded Docker config differs"
    ):
        normalizer.normalize_docker_image(
            source_image="source:test", normalized_image="normalized:test"
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_temporary_config_diagnostics_require_explicit_gate(monkeypatch, capsys, enabled):
    monkeypatch.setenv("GATEWAY_V2_CANONICAL_CONFIG_DIAGNOSTICS", "1" if enabled else "")
    config = {"config": {"Env": ["PYTHON_VERSION=3.12.15"]}, "rootfs": {"diff_ids": ["sha256:example"]}}
    payload = json.dumps(config).encode()
    normalizer._emit_canonical_config_diagnostic(config, payload)
    output = capsys.readouterr()
    assert not output.out
    assert (payload.decode() in output.err) is enabled


@pytest.mark.parametrize("value", ["UNKNOWN_ENV=private-value", "PATH=https://user:private-value@example.com"])
def test_temporary_config_diagnostics_withhold_sensitive_material(monkeypatch, capsys, value):
    monkeypatch.setenv("GATEWAY_V2_CANONICAL_CONFIG_DIAGNOSTICS", "1")
    config = {"config": {"Env": [value]}}
    normalizer._emit_canonical_config_diagnostic(config, json.dumps(config).encode())
    output = capsys.readouterr()
    assert "private-value" not in output.err
    assert "withheld" in output.err


def test_workflow_config_diagnostics_are_reproduction_only():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/attested-v2-release.yml").read_text()
    assert "GATEWAY_V2_CANONICAL_CONFIG_DIAGNOSTICS: ${{ needs.source.outputs.mode == 'reproduce' && '1' || '' }}" in workflow


def _entry(name, content=None, **metadata):
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE if content is None else tarfile.REGTYPE
    member.mode = 0o755 if content is None else 0o644
    member.size = 0 if content is None else len(content)
    for key, value in metadata.items():
        setattr(member, key, value)
    return member, content


def _saved_layers(path, layers, order=None):
    order = list(range(len(layers))) if order is None else order
    config = {"rootfs": {"type": "layers", "diff_ids": ["sha256:old"] * len(order)},
              "config": {"Entrypoint": ["python3"]}}
    manifest = [{"Config": "config.json", "RepoTags": ["raw:test"],
                 "Layers": [f"layers/{index}.tar" for index in order]}]
    with tarfile.open(path, "w:") as image:
        for name, data in [("config.json", json.dumps(config).encode()),
                           ("manifest.json", json.dumps(manifest).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            image.addfile(member, io.BytesIO(data))
        for index, entries in enumerate(layers):
            layer = io.BytesIO()
            with tarfile.open(fileobj=layer, mode="w:") as archive:
                for member, content in entries:
                    archive.addfile(member, None if content is None else io.BytesIO(content))
            member = tarfile.TarInfo(f"layers/{index}.tar")
            member.size = len(layer.getvalue())
            image.addfile(member, io.BytesIO(layer.getvalue()))


def _normalize_layers(tmp_path, label, layers, order=None):
    source = tmp_path / f"{label}.tar"
    output = tmp_path / f"{label}-normalized.tar"
    _saved_layers(source, layers, order)
    identity = normalizer.normalize_saved_image(
        archive_path=source, output_path=output, normalized_image="normalized:test"
    )
    with tarfile.open(output) as image:
        manifest = json.load(image.extractfile("manifest.json"))[0]
        members = []
        for name in manifest["Layers"]:
            with tarfile.open(fileobj=image.extractfile(name)) as layer:
                members.append([member.name for member in layer.getmembers()])
    return identity, manifest["Layers"], members


def test_redundant_inherited_directory_matches_empty_layer(tmp_path):
    base = [_entry("etc")]
    empty = _normalize_layers(tmp_path, "empty", [base, []])
    redundant = _normalize_layers(tmp_path, "redundant", [base, [_entry("etc")]])
    assert empty[0] == redundant[0]
    assert len(redundant[1]) == 2
    assert redundant[2] == [["etc"], []]


@pytest.mark.parametrize("change", [
    {"mode": 0o700}, {"uid": 123}, {"gid": 456},
    {"type": tarfile.SYMTYPE, "linkname": "elsewhere"},
    {"pax_headers": {"SCHILY.xattr.user.attestation": "changed"}},
])
def test_real_directory_metadata_changes_remain_bound(tmp_path, change):
    base = [_entry("etc")]
    redundant = _normalize_layers(tmp_path, "redundant", [base, [_entry("etc")]])
    changed = _normalize_layers(tmp_path, "changed", [base, [_entry("etc", **change)]])
    assert changed[0] != redundant[0]
    assert changed[2][1] == ["etc"]


@pytest.mark.parametrize("removal,recreated", [
    (".wh.etc", "etc"), (".wh.etc", "etc/sub"),
    ("etc/.wh..wh..opq", "etc/sub"), (".wh..wh..opq", "etc"),
])
def test_directory_recreation_after_whiteout_remains_bound(tmp_path, removal, recreated):
    base = [_entry("etc"), _entry("etc/sub")]
    deleted = [_entry(removal, b"")]
    rebuilt = _normalize_layers(tmp_path, "rebuilt", [base, deleted, [_entry(recreated)]])
    absent = _normalize_layers(tmp_path, "absent", [base, deleted, []])
    assert rebuilt[0] != absent[0]
    assert rebuilt[2][-1] == [recreated]


def test_directory_recreation_after_type_change_remains_bound(tmp_path):
    result = _normalize_layers(tmp_path, "recreated", [
        [_entry("etc"), _entry("etc/sub")],
        [_entry("etc", type=tarfile.SYMTYPE, linkname="elsewhere")],
        [_entry("etc"), _entry("etc/sub")],
    ])
    assert result[2][-1] == ["etc", "etc/sub"]


def test_repeated_source_layer_is_normalized_again_after_whiteout(tmp_path):
    result = _normalize_layers(tmp_path, "repeated", [
        [_entry("etc")], [_entry("etc")], [_entry(".wh.etc", b"")],
    ], order=[0, 1, 2, 1])
    assert result[1][1] != result[1][3]
    assert result[2] == [["etc"], [], [".wh.etc"], ["etc"]]


def test_directory_canonicalization_preserves_file_content_and_layer_order(tmp_path):
    base = [_entry("etc")]
    first = _normalize_layers(tmp_path, "first", [base, [_entry("etc/file", b"a")]])
    changed = _normalize_layers(tmp_path, "changed", [base, [_entry("etc/file", b"b")]])
    reordered = _normalize_layers(tmp_path, "reordered", [base, [_entry("etc/file", b"a")]], order=[1, 0])
    assert len({first[0], changed[0], reordered[0]}) == 3


def test_same_layer_duplicate_directory_records_are_preserved(tmp_path):
    result = _normalize_layers(tmp_path, "duplicates", [
        [_entry("etc")], [_entry("etc"), _entry("etc")],
    ])
    assert result[2][-1] == ["etc", "etc"]


@pytest.mark.parametrize("whiteout,recreated", [
    (".wh.etc", "etc"), ("etc/.wh..wh..opq", "./etc/sub"),
])
def test_whiteout_precedes_same_layer_recreation_regardless_alias_order(tmp_path, whiteout, recreated):
    result = _normalize_layers(tmp_path, "same-layer-recreated", [
        [_entry("etc"), _entry("etc/sub")],
        [_entry(recreated), _entry(whiteout, b"")],
        [_entry(recreated)],
    ])
    assert recreated in result[2][1]
    assert whiteout in result[2][1]
    assert result[2][2] == []


@pytest.mark.parametrize("unfamiliar", ["/absolute-directory", "../traversal-directory"])
def test_unfamiliar_layer_paths_preserve_records_and_clear_comparison_state(tmp_path, unfamiliar):
    result = _normalize_layers(tmp_path, "unfamiliar", [
        [_entry("etc")], [_entry(unfamiliar), _entry("etc")],
    ])
    assert unfamiliar in result[2][1]
    assert "etc" in result[2][1]


@pytest.mark.parametrize("directory,child", [
    ("etc", "etc/config"), ("etc", "./etc/config"),
    ("etc", "etc/.wh.config"), ("etc", "etc/.wh..wh..opq"),
    ("etc", "etc/sub"), ("etc", "etc/link"), (".", "config"),
])
def test_parent_directory_headers_remain_when_any_child_changes(tmp_path, directory, child):
    if child.endswith("sub"):
        descendant = _entry(child)
    elif child.endswith("link"):
        descendant = _entry(child, type=tarfile.SYMTYPE, linkname="elsewhere")
    else:
        descendant = _entry(child, b"" if ".wh." in child else b"data")
    result = _normalize_layers(tmp_path, "parent", [
        [_entry(directory)], [_entry(directory), descendant],
    ])
    assert directory in result[2][1]


def test_layer_replay_preserves_parent_directory_mtime(tmp_path):
    _normalize_layers(tmp_path, "mtime", [
        [_entry("etc")], [_entry("etc"), _entry("etc/config", b"data")],
    ])
    filesystem = tmp_path / "filesystem"
    filesystem.mkdir()
    with tarfile.open(tmp_path / "mtime-normalized.tar") as image:
        manifest = json.load(image.extractfile("manifest.json"))[0]
        for name in manifest["Layers"]:
            with tarfile.open(fileobj=image.extractfile(name)) as layer:
                layer.extractall(filesystem)
    assert (filesystem / "etc/config").read_bytes() == b"data"
    assert (filesystem / "etc").stat().st_mtime_ns == 0


def test_later_directory_header_restores_time_after_child_only_layer(tmp_path):
    result = _normalize_layers(tmp_path, "later-mtime", [
        [_entry("etc")], [_entry("etc/config", b"data")], [_entry("etc")],
    ])
    filesystem = tmp_path / "filesystem"
    filesystem.mkdir()
    with tarfile.open(tmp_path / "later-mtime-normalized.tar") as image:
        manifest = json.load(image.extractfile("manifest.json"))[0]
        for name in manifest["Layers"]:
            with tarfile.open(fileobj=image.extractfile(name)) as layer:
                layer.extractall(filesystem)
    assert (filesystem / "etc").stat().st_mtime_ns == 0
    assert result[2][-1] == ["etc"]


def test_elided_child_noop_does_not_invalidate_parent_metadata(tmp_path):
    result = _normalize_layers(tmp_path, "elided-child", [
        [_entry("etc"), _entry("etc/sub")], [_entry("etc/sub")], [_entry("etc")],
    ])
    assert result[2][1:] == [[], []]
