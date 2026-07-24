import hashlib
import io
import json
from pathlib import Path
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

    observed = normalizer.normalize_docker_image(
        source_image="source:test",
        normalized_image="normalized:test",
    )

    expected_parent = build_root / ".docker-image-normalizer-v2"
    assert observed == "sha256:" + "a" * 64
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


def test_validator_enclave_normalizer_uses_root_backed_workspace() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (
        root / "validator_tee" / "scripts" / "build_enclave.sh"
    ).read_text(encoding="utf-8")
    workflow = (
        root / ".github" / "workflows" / "attested-v2-release.yml"
    ).read_text(encoding="utf-8")

    assert '"VALIDATOR_V2_BUILD_WORK_ROOT", "RUNNER_TEMP"' in build_script
    assert 'dir=str(normalization_work_root())' in build_script
    assert 'tempfile.mkdtemp(prefix="pcr0_normalize_")' not in build_script
    assert workflow.count(
        'export VALIDATOR_V2_BUILD_WORK_ROOT="$RUNNER_TEMP/validator-build-work"'
    ) == 2
    assert workflow.count('"$RUNNER_TEMP/validator-build-work"') == 4
