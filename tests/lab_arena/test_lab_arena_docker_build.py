"""Real Docker proof of the offline miner image build and the runner's rootfs export.

Skips when no Docker daemon is reachable or the pinned base image cannot be
pulled. Everything else is the production code path: ``build.build_image``
with the real ``docker`` binary, an offline ``--network=none`` build that
installs a wheel from the wheelhouse, a ``--network=none`` run of the
entrypoint, and ``runner.docker_image_exporter`` extracting the rootfs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lab_arena import build
from lab_arena import runner as rn

BASE_REPOSITORY = "python"
BASE_TAG = "python:3.12-slim"
DOCKER = shutil.which("docker")


def _docker(*args: str, timeout: int = 120, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([DOCKER, *args], capture_output=True, text=True, timeout=timeout, check=False, **kwargs)


APPROVED_PIN = "h11==0.16.0"  # an approved pure-Python pin with no further requirements


def fetch_approved_wheel(directory: Path) -> Path:
    """Download the approved wheel into the wheelhouse; the build itself stays offline."""

    assert APPROVED_PIN in build.APPROVED_DEPENDENCIES and not build.APPROVED_DEPENDENCY_REQUIRES.get(APPROVED_PIN.split("==")[0])
    downloaded = subprocess.run(
        [sys.executable, "-m", "pip", "download", APPROVED_PIN, "--only-binary=:all:", "--no-deps", "--dest", str(directory), "--quiet"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    wheels = sorted(directory.glob("h11-0.16.0-*.whl"))
    if downloaded.returncode != 0 or len(wheels) != 1:
        pytest.skip("approved wheel could not be downloaded: %s" % downloaded.stderr[-200:])
    return wheels[0]


@pytest.fixture(scope="module")
def base_image_digest() -> str:
    if DOCKER is None or _docker("info", timeout=30).returncode != 0:
        pytest.skip("no Docker daemon is reachable")
    inspected = _docker("image", "inspect", BASE_TAG, "--format", "{{index .RepoDigests 0}}")
    if inspected.returncode != 0:
        pulled = _docker("pull", BASE_TAG, timeout=600)
        if pulled.returncode != 0:
            pytest.skip("base image could not be pulled: %s" % pulled.stderr[-200:])
        inspected = _docker("image", "inspect", BASE_TAG, "--format", "{{index .RepoDigests 0}}")
    reference = inspected.stdout.strip()
    assert reference.startswith(BASE_REPOSITORY + "@sha256:"), reference
    return reference.split("@", 1)[1]


def test_offline_build_installs_only_wheelhouse_wheels_and_the_runner_exports_the_rootfs(base_image_digest, tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    fetch_approved_wheel(wheelhouse)
    source_files = {
        "main.py": b"import json, sys\nimport h11\nprint(json.dumps({'h11': h11.__version__, 'argv': sys.argv[1:]}))\n",
        "model/__init__.py": b"",
    }
    spec = build.BuildSpec(
        base_image=BASE_REPOSITORY, base_image_digest=base_image_digest, wheelhouse_dir=wheelhouse, entry_point="main.py",
        source_files=source_files, dependency_lock=(APPROVED_PIN,),
    )

    def docker_runner(argv, timeout_seconds):
        return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_seconds, check=False)

    result = build.build_image(spec, docker_runner=docker_runner, context_dir=tmp_path / "context", environment={"PATH": "/usr/bin"})
    image_id = result.image_id
    try:
        assert build.IMAGE_DIGEST_RE.match(image_id) and result.source_tree_hash == build.source_tree_hash(source_files)
        # The image is never pulled during the build, so a local-only build has no registry digest.
        assert result.image_digest == image_id
        ran = _docker("run", "--rm", "--network=none", image_id, "icp-argument", timeout=120)
        assert ran.returncode == 0, ran.stderr[-400:]
        assert '"h11": "0.16.0"' in ran.stdout and '"argv": ["icp-argument"]' in ran.stdout
        # The entrypoint runs as nobody with no write access to the model tree.
        whoami = _docker("run", "--rm", "--network=none", "--entrypoint", "id", image_id, "-u", timeout=60)
        assert whoami.stdout.strip() == "65534"
        target = tmp_path / "export"
        target.mkdir()
        rn.docker_image_exporter(image_id, target)
        rootfs = target / "rootfs"
        assert (rootfs / "model" / "main.py").read_bytes() == source_files["main.py"]
        assert (rootfs / "model" / "requirements.lock").read_text().strip() == APPROVED_PIN
        assert not (target / "rootfs.tar").exists()
        installed = list(rootfs.glob("usr/local/lib/python3*/site-packages/h11/__init__.py"))
        assert len(installed) == 1 and (rootfs / "wheelhouse").is_dir()
        assert (rootfs / "usr" / "local" / "bin" / "python3").exists() or (rootfs / "usr" / "local" / "bin" / "python3").is_symlink()
    finally:
        _docker("image", "rm", "--force", image_id, timeout=60)
