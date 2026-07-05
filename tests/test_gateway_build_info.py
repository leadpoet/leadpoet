from __future__ import annotations

import json
from pathlib import Path

from gateway import build_info


_BUILD_ENV_KEYS = [
    "BUILD_ID",
    "GITHUB_SHA",
    "GITHUB_COMMIT",
    "GIT_COMMIT_HASH",
    "GIT_COMMIT",
    "GITHUB_REF_NAME",
    "GIT_BRANCH",
    "BRANCH_NAME",
    "GITHUB_TAG",
    "GIT_TAG",
    "BUILD_TIME_UTC",
    "BUILD_TIMESTAMP",
    "GIT_DIRTY",
    "GATEWAY_BUILD_INFO_FILE",
    "GATEWAY_BUILD_INFO_GIT_ROOT",
]


def _clear_build_env(monkeypatch) -> None:
    for key in _BUILD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    build_info.get_build_info.cache_clear()


def test_load_build_info_reads_explicit_file(monkeypatch, tmp_path: Path) -> None:
    _clear_build_env(monkeypatch)
    info_path = tmp_path / "BUILD_INFO.json"
    commit = "a" * 40
    info_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "service": "leadpoet-gateway",
                "build_id": "gateway-test-build",
                "git_commit": commit,
                "git_branch": "main",
                "git_dirty": False,
                "build_time_utc": "2026-07-02T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    info = build_info.load_build_info(discover_git=False, build_info_path=info_path)

    assert info["build_id"] == "gateway-test-build"
    assert info["git_commit"] == commit
    assert info["git_commit_short"] == "a" * 12
    assert info["git_branch"] == "main"
    assert info["git_dirty"] is False
    assert info["is_commit_known"] is True
    assert info["commit_source"] == f"file:{info_path}"
    assert info["build_info_path"] == str(info_path)


def test_environment_commit_overrides_build_info_file(monkeypatch, tmp_path: Path) -> None:
    _clear_build_env(monkeypatch)
    info_path = tmp_path / "BUILD_INFO.json"
    file_commit = "a" * 40
    env_commit = "b" * 40
    info_path.write_text(
        json.dumps({"build_id": "file-build", "git_commit": file_commit}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GATEWAY_BUILD_INFO_FILE", str(info_path))
    monkeypatch.setenv("GITHUB_SHA", env_commit)
    monkeypatch.setenv("BUILD_ID", "env-build")

    info = build_info.load_build_info(discover_git=False)

    assert info["git_commit"] == env_commit
    assert info["commit_source"] == "env:GITHUB_SHA"
    assert info["build_id"] == "env-build"
    assert info["build_id_source"] == "env:BUILD_ID"


def test_branch_placeholder_commit_env_does_not_override_file(monkeypatch, tmp_path: Path) -> None:
    _clear_build_env(monkeypatch)
    info_path = tmp_path / "BUILD_INFO.json"
    file_commit = "a" * 40
    info_path.write_text(
        json.dumps({"build_id": "file-build", "git_commit": file_commit}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GATEWAY_BUILD_INFO_FILE", str(info_path))
    monkeypatch.setenv("GITHUB_COMMIT", "main")

    info = build_info.load_build_info(discover_git=False)

    assert info["git_commit"] == file_commit
    assert info["commit_source"] == f"file:{info_path}"


def test_create_and_write_build_info_document_from_env(monkeypatch, tmp_path: Path) -> None:
    _clear_build_env(monkeypatch)
    commit = "c" * 40
    monkeypatch.setenv("GITHUB_SHA", commit)
    monkeypatch.setenv("GITHUB_REF_NAME", "codex/build-info")

    document = build_info.create_build_info_document(
        repo_root=tmp_path,
        build_id="manual-build-id",
    )
    output = build_info.write_build_info_file(tmp_path / "BUILD_INFO.json", document)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["build_id"] == "manual-build-id"
    assert saved["git_commit"] == commit
    assert saved["git_commit_short"] == "c" * 12
    assert saved["git_branch"] == "codex/build-info"
    assert saved["service"] == "leadpoet-gateway"
    assert saved["generated_by"] == "scripts/write_gateway_build_info.py"
