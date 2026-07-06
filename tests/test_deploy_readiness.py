from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway import deploy_readiness


def _status(role: str, pcr0: str, *, allowed: bool, commits: list[str]) -> dict:
    return {
        "role": role,
        "pcr0": pcr0,
        "allowed": allowed,
        "allowed_count": 1 if allowed else 0,
        "allowed_source": "test",
        "allowed_error": None,
        "local_allowlist_path": "test",
        "local_match_count": 1 if commits else 0,
        "matched_entry_commits": commits,
        "matched_entries": [],
    }


def test_deploy_readiness_blocks_pcr0_commit_drift(monkeypatch) -> None:
    gateway_commit = "a" * 40
    stale_commit = "b" * 40
    gateway_pcr0 = "1" * 96
    validator_pcr0 = "2" * 96

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {
            "git_commit": gateway_commit,
            "build_time_utc": "2026-07-06T12:00:00Z",
        },
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (gateway_commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(
            role,
            pcr0,
            allowed=True,
            commits=[stale_commit if role == "gateway" else gateway_commit],
        ),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0: {"available": True, "valid": False, "verification": {}, "cache_status": {}},
    )

    result = deploy_readiness.build_deploy_readiness(
        validator_commit=gateway_commit,
        gateway_pcr0=gateway_pcr0,
        validator_pcr0=validator_pcr0,
        require_same_commit=True,
        require_pcr0=True,
        require_pcr0_commit_match=True,
    )

    assert result["ok"] is False
    failed = {check["name"] for check in result["checks"] if not check["ok"]}
    assert failed == {"gateway_pcr0_commit_matches_gateway_commit"}
    assert result["validator"]["pcr0_accepted"] is True


def test_deploy_readiness_accepts_matching_commits(monkeypatch) -> None:
    commit = "a" * 40
    gateway_pcr0 = "1" * 96
    validator_pcr0 = "2" * 96

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {
            "git_commit": commit,
            "build_time_utc": "2026-07-06T12:00:00Z",
        },
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(role, pcr0, allowed=True, commits=[commit]),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0: {"available": True, "valid": False, "verification": {}, "cache_status": {}},
    )

    result = deploy_readiness.build_deploy_readiness(
        validator_commit=commit,
        gateway_pcr0=gateway_pcr0,
        validator_pcr0=validator_pcr0,
        expected_gateway_commit=commit[:12],
        expected_validator_commit=commit,
        require_same_commit=True,
        require_pcr0=True,
        require_pcr0_commit_match=True,
    )

    assert result["ok"] is True
    assert all(check["ok"] for check in result["checks"])


def test_resume_guard_blocks_failed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "deploy_readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "ok": False,
                "enforce_resume_block": True,
                "checks": [
                    {
                        "name": "validator_pcr0_accepted",
                        "ok": False,
                        "severity": "error",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="validator_pcr0_accepted"):
        deploy_readiness.assert_resume_allowed(manifest)


def test_resume_guard_allows_ok_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "deploy_readiness.json"
    manifest.write_text(
        json.dumps({"ok": True, "enforce_resume_block": True, "checks": []}),
        encoding="utf-8",
    )

    result = deploy_readiness.assert_resume_allowed(manifest)

    assert result is not None
    assert result["ok"] is True


def test_optional_docker_health_is_warning_only(monkeypatch) -> None:
    commit = "a" * 40

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": commit, "build_time_utc": "2026-07-06T12:00:00Z"},
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(role, pcr0 or "", allowed=False, commits=[]),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0: {"available": True, "valid": False, "verification": {}, "cache_status": {}},
    )
    monkeypatch.setattr(
        deploy_readiness,
        "docker_build_health",
        lambda *, smoke_build=False: {
            "ok": False,
            "docker_info": {"docker_root": "/var/lib/docker"},
            "disk": {"ok": False, "free_gb": 0.1},
            "smoke_build_requested": smoke_build,
            "smoke_build": None,
        },
    )

    result = deploy_readiness.build_deploy_readiness(include_docker_health=True)

    assert result["ok"] is True
    docker_checks = [check for check in result["checks"] if check["name"] == "docker_build_health"]
    assert docker_checks == [
        {
            "name": "docker_build_health",
            "ok": False,
            "severity": "warning",
            "detail": (
                "Docker host/build health; require flag runs a tiny scratch-image smoke build "
                "and blocks resume on failure"
            ),
            "expected": None,
            "actual": {
                "docker_root": "/var/lib/docker",
                "disk": {"ok": False, "free_gb": 0.1},
                "smoke_build_requested": False,
                "smoke_build_ok": None,
            },
        }
    ]


def test_required_docker_build_health_blocks_readiness(monkeypatch) -> None:
    commit = "a" * 40

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": commit, "build_time_utc": "2026-07-06T12:00:00Z"},
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(role, pcr0 or "", allowed=False, commits=[]),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0: {"available": True, "valid": False, "verification": {}, "cache_status": {}},
    )
    monkeypatch.setattr(
        deploy_readiness,
        "docker_build_health",
        lambda *, smoke_build=False: {
            "ok": False,
            "docker_info": {"docker_root": "/var/lib/docker"},
            "disk": {"ok": True, "free_gb": 92.0},
            "smoke_build_requested": smoke_build,
            "smoke_build": {"ok": False},
        },
    )

    result = deploy_readiness.build_deploy_readiness(require_docker_build_health=True)

    assert result["ok"] is False
    failed = [check for check in result["checks"] if check["name"] == "docker_build_health"]
    assert len(failed) == 1
    assert failed[0]["severity"] == "error"
    assert failed[0]["actual"]["smoke_build_requested"] is True
