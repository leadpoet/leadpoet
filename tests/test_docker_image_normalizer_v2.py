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


@pytest.mark.parametrize("enabled", [False, True])
def test_temporary_layer_diagnostic_is_gated_and_excludes_contents(monkeypatch, capsys, tmp_path, enabled):
    monkeypatch.setenv("GATEWAY_V2_CANONICAL_CONFIG_DIAGNOSTICS", "1" if enabled else "")
    source = tmp_path / "original.tar"
    _layer(source, mtime=100, reverse=False, compressed=False)
    normalizer.normalize_layer_archive(source, tmp_path / "normalized-layer-11.tar")
    output = capsys.readouterr()
    assert not output.out
    if enabled:
        record = json.loads(output.err.split("diagnostic: ", 1)[1])
        assert record["member_count"] == 2
        assert record["members"][0]["content_sha256"] == hashlib.sha256(b"a").hexdigest()
        assert len(record["first_header_hex"]) == 1024
        assert "content" not in record["members"][0]
    else:
        assert not output.err
