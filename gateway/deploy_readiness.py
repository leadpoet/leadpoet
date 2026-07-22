"""Gateway/validator deploy readiness checks.

These helpers keep the production resume decision tied to explicit source and
PCR0 evidence instead of ad hoc operator inference.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gateway.build_info import UNKNOWN, get_build_info


DEFAULT_DEPLOY_READINESS_MANIFEST = "/home/ec2-user/gateway/deploy_readiness.json"
DEPLOY_READINESS_MANIFEST_ENV = "RESEARCH_LAB_DEPLOY_READINESS_MANIFEST"
DEFAULT_DOCKER_MIN_FREE_GB = 5.0
DEFAULT_DOCKER_HEALTH_TIMEOUT_SECONDS = 60

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null", "undefined"}:
        return None
    return text


def normalize_commit(value: Any) -> str | None:
    text = clean_string(value)
    return text.lower() if text else None


def normalize_pcr0(value: Any) -> str | None:
    text = clean_string(value)
    if not text:
        return None
    return text.lower()


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _parse_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def default_manifest_path() -> Path:
    return Path(os.getenv(DEPLOY_READINESS_MANIFEST_ENV, DEFAULT_DEPLOY_READINESS_MANIFEST)).expanduser()


def _source_commit_candidates() -> list[Path]:
    paths: list[Path] = []
    explicit = clean_string(os.getenv("GATEWAY_SOURCE_COMMIT_FILE"))
    if explicit:
        paths.append(Path(explicit).expanduser())
    module_dir = Path(__file__).resolve().parent
    paths.extend([Path.cwd() / ".source_commit", module_dir / ".source_commit", module_dir.parent / ".source_commit"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def read_source_commit() -> tuple[str | None, str | None]:
    for path in _source_commit_candidates():
        try:
            value = normalize_commit(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if value:
            return value, str(path)
    return None, None


def _allowlist_file_candidates() -> list[Path]:
    explicit = clean_string(os.getenv("PCR0_ALLOWLIST_FILE"))
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    module_dir = Path(__file__).resolve().parent
    paths.extend(
        [
            Path.cwd() / "pcr0_allowlist.json",
            module_dir.parent / "pcr0_allowlist.json",
            module_dir / "pcr0_allowlist.json",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_local_allowlist_entries(role: str) -> tuple[list[dict[str, Any]], str | None]:
    key = f"{role.strip().lower()}_pcr0"
    for path in _allowlist_file_candidates():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        except json.JSONDecodeError:
            continue
        entries = doc.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)], str(path)
    return [], None


def extract_allowlist_entry_commit(entry: Mapping[str, Any]) -> str | None:
    for key in ("commit_hash", "git_commit_sha", "git_commit", "commit"):
        commit = normalize_commit(entry.get(key))
        if commit:
            return commit
    notes = entry.get("notes")
    if isinstance(notes, str):
        marker = "commit "
        idx = notes.lower().find(marker)
        if idx >= 0:
            candidate = notes[idx + len(marker) :].split()[0].strip(".,;:)(")
            return normalize_commit(candidate)
    return None


def _static_allowlist_status(pcr0: str | None, *, role: str) -> dict[str, Any]:
    normalized = normalize_pcr0(pcr0)
    entries, local_path = load_local_allowlist_entries(role)
    matching_entries = [
        entry for entry in entries if normalize_pcr0(entry.get("pcr0")) == normalized
    ] if normalized else []
    allowed_values: list[str] = []
    allowed_source = "unavailable"
    allowed_error: str | None = None
    try:
        from leadpoet_canonical.nitro import get_allowed_pcr0_values

        allowed_values = [normalize_pcr0(value) or "" for value in get_allowed_pcr0_values(role)]
        allowed_source = "leadpoet_canonical.nitro"
    except Exception as exc:  # noqa: BLE001 - status report should explain, not raise.
        allowed_error = str(exc)[:500]

    allowed = bool(normalized and normalized in set(allowed_values))
    entry_commits = [commit for entry in matching_entries if (commit := extract_allowlist_entry_commit(entry))]
    return {
        "role": role,
        "pcr0": normalized,
        "allowed": allowed,
        "allowed_count": len([value for value in allowed_values if value]),
        "allowed_source": allowed_source,
        "allowed_error": allowed_error,
        "local_allowlist_path": local_path,
        "local_match_count": len(matching_entries),
        "matched_entry_commits": entry_commits,
        "matched_entries": matching_entries,
    }


def _dynamic_validator_status(
    pcr0: str | None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_pcr0(pcr0)
    try:
        from gateway.utils.pcr0_builder import get_cache_status, verify_pcr0
    except Exception as exc:  # noqa: BLE001 - gateway imports can be unavailable in tests.
        return {
            "available": False,
            "valid": False,
            "error": str(exc)[:500],
            "cache_status": None,
        }
    verification = verify_pcr0(
        normalized or "",
        expected_commit=normalize_commit(expected_commit) or "",
    )
    return {
        "available": True,
        "valid": bool(verification.get("valid")),
        "verification": verification,
        "cache_status": get_cache_status(),
    }


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    *,
    severity: str = "error",
    detail: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "severity": severity,
            "detail": detail,
            "expected": expected,
            "actual": actual,
        }
    )


def _commit_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    return actual.startswith(expected) or expected.startswith(actual)


def _pcr0_matches(expected: str | None, actual: str | None) -> bool:
    return bool(expected and actual and expected == actual)


def _allowlist_commit_matches_runtime(status: Mapping[str, Any], runtime_commit: str | None) -> bool:
    commits = [normalize_commit(value) for value in status.get("matched_entry_commits") or []]
    commits = [value for value in commits if value]
    if not commits or not runtime_commit:
        return False
    return any(_commit_matches(commit, runtime_commit) for commit in commits)


def _truncate(value: str, limit: int = 700) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_command(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": _truncate(exc.stdout or ""),
            "stderr": f"timeout after {timeout_seconds}s",
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _docker_base_command() -> list[str] | None:
    docker_path = shutil.which("docker")
    if docker_path:
        return [docker_path]
    sudo_path = shutil.which("sudo")
    if sudo_path and Path("/usr/bin/docker").exists():
        return [sudo_path, "-n", "/usr/bin/docker"]
    return None


def _run_docker(args: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    base = _docker_base_command()
    if base is None:
        return {
            "ok": False,
            "command": ["docker", *args],
            "returncode": None,
            "stdout": "",
            "stderr": "docker CLI not found",
            "duration_seconds": 0,
        }
    result = _run_command([*base, *args], timeout_seconds=timeout_seconds)
    if result["ok"] or base[0].endswith("sudo"):
        return result
    stderr = str(result.get("stderr") or "").lower()
    if not any(marker in stderr for marker in ("permission denied", "cannot connect", "got permission denied")):
        return result
    sudo_path = shutil.which("sudo")
    if not sudo_path:
        return result
    return _run_command([sudo_path, "-n", base[0], *args], timeout_seconds=timeout_seconds)


def _parse_docker_info(stdout: str) -> dict[str, str | None]:
    parts = str(stdout or "").strip().split("|")
    return {
        "driver": parts[0] if len(parts) > 0 and parts[0] else None,
        "docker_root": parts[1] if len(parts) > 1 and parts[1] else None,
        "server_version": parts[2] if len(parts) > 2 and parts[2] else None,
    }


def _disk_status(path: str | None, *, min_free_gb: float) -> dict[str, Any]:
    probe = Path(path or "/var/lib/docker")
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {
            "ok": False,
            "path": str(probe),
            "error": str(exc)[:500],
            "min_free_gb": min_free_gb,
        }
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)
    return {
        "ok": free_gb >= min_free_gb,
        "path": str(probe),
        "free_gb": round(free_gb, 3),
        "used_gb": round(used_gb, 3),
        "total_gb": round(total_gb, 3),
        "used_percent": round((usage.used / usage.total) * 100, 2) if usage.total else None,
        "min_free_gb": min_free_gb,
    }


def _docker_smoke_build(*, timeout_seconds: int) -> dict[str, Any]:
    tag = f"leadpoet-deploy-readiness-smoke:{os.getpid()}-{int(time.time())}"
    with tempfile.TemporaryDirectory(prefix="leadpoet_docker_smoke_") as tmpdir:
        dockerfile = Path(tmpdir) / "Dockerfile"
        dockerfile.write_text(
            "FROM scratch\nLABEL leadpoet.deploy_readiness=1\n",
            encoding="utf-8",
        )
        build = _run_docker(
            ["build", "--quiet", "--no-cache", "-t", tag, tmpdir],
            timeout_seconds=timeout_seconds,
        )
    cleanup = _run_docker(["rmi", "-f", tag], timeout_seconds=15)
    return {
        "ok": bool(build.get("ok")),
        "tag": tag,
        "build": build,
        "cleanup": {
            "ok": bool(cleanup.get("ok")),
            "returncode": cleanup.get("returncode"),
            "stderr": cleanup.get("stderr"),
        },
    }


def docker_build_health(
    *,
    smoke_build: bool = False,
    timeout_seconds: int | None = None,
    min_free_gb: float | None = None,
) -> dict[str, Any]:
    """Report Docker availability and, optionally, exercise a tiny local build."""
    timeout = int(
        timeout_seconds
        or os.getenv("DEPLOY_READINESS_DOCKER_HEALTH_TIMEOUT_SECONDS")
        or DEFAULT_DOCKER_HEALTH_TIMEOUT_SECONDS
    )
    min_free = float(
        min_free_gb
        if min_free_gb is not None
        else _parse_float(os.getenv("DEPLOY_READINESS_DOCKER_MIN_FREE_GB"), default=DEFAULT_DOCKER_MIN_FREE_GB)
    )
    docker_cli = _docker_base_command()
    info = _run_docker(
        ["info", "--format", "{{.Driver}}|{{.DockerRootDir}}|{{.ServerVersion}}"],
        timeout_seconds=min(timeout, 15),
    )
    parsed_info = _parse_docker_info(str(info.get("stdout") or "")) if info.get("ok") else {}
    disk = _disk_status(str(parsed_info.get("docker_root") or "/var/lib/docker"), min_free_gb=min_free)
    smoke = _docker_smoke_build(timeout_seconds=timeout) if smoke_build and info.get("ok") else None
    ok = bool(docker_cli and info.get("ok") and disk.get("ok") and (not smoke_build or (smoke and smoke.get("ok"))))
    return {
        "ok": ok,
        "docker_cli": docker_cli,
        "docker_info": {
            "ok": bool(info.get("ok")),
            "driver": parsed_info.get("driver"),
            "docker_root": parsed_info.get("docker_root"),
            "server_version": parsed_info.get("server_version"),
            "returncode": info.get("returncode"),
            "stderr": info.get("stderr"),
            "duration_seconds": info.get("duration_seconds"),
        },
        "disk": disk,
        "smoke_build_requested": bool(smoke_build),
        "smoke_build": smoke,
    }


def build_deploy_readiness(
    *,
    gateway_commit: str | None = None,
    validator_commit: str | None = None,
    gateway_pcr0: str | None = None,
    validator_pcr0: str | None = None,
    expected_gateway_commit: str | None = None,
    expected_validator_commit: str | None = None,
    expected_gateway_pcr0: str | None = None,
    expected_validator_pcr0: str | None = None,
    require_same_commit: bool = False,
    require_pcr0: bool = False,
    require_pcr0_commit_match: bool = False,
    include_docker_health: bool = False,
    require_docker_build_health: bool = False,
) -> dict[str, Any]:
    build_info = get_build_info()
    source_commit, source_commit_path = read_source_commit()
    resolved_gateway_commit = (
        normalize_commit(gateway_commit)
        or source_commit
        or normalize_commit(build_info.get("git_commit"))
    )
    resolved_validator_commit = normalize_commit(validator_commit)
    resolved_gateway_pcr0 = normalize_pcr0(gateway_pcr0)
    resolved_validator_pcr0 = normalize_pcr0(validator_pcr0)
    expected_gateway_commit_norm = normalize_commit(expected_gateway_commit)
    expected_validator_commit_norm = normalize_commit(expected_validator_commit)
    expected_gateway_pcr0_norm = normalize_pcr0(expected_gateway_pcr0)
    expected_validator_pcr0_norm = normalize_pcr0(expected_validator_pcr0)

    gateway_static = _static_allowlist_status(resolved_gateway_pcr0, role="gateway")
    validator_static = _static_allowlist_status(resolved_validator_pcr0, role="validator")
    validator_dynamic = _dynamic_validator_status(
        resolved_validator_pcr0,
        resolved_validator_commit,
    )
    validator_pcr0_accepted = bool(validator_static.get("allowed") or validator_dynamic.get("valid"))
    docker_health = (
        docker_build_health(smoke_build=require_docker_build_health)
        if include_docker_health or require_docker_build_health
        else None
    )

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "gateway_commit_known",
        bool(resolved_gateway_commit),
        detail="gateway commit comes from explicit arg, .source_commit, BUILD_INFO, or git",
        actual=resolved_gateway_commit,
    )
    if resolved_validator_commit or expected_validator_commit_norm or require_same_commit:
        _add_check(
            checks,
            "validator_commit_known",
            bool(resolved_validator_commit),
            detail="validator commit must be supplied by the caller or manifest",
            actual=resolved_validator_commit,
        )
    if expected_gateway_commit_norm:
        _add_check(
            checks,
            "gateway_commit_matches_expected",
            _commit_matches(expected_gateway_commit_norm, resolved_gateway_commit),
            expected=expected_gateway_commit_norm,
            actual=resolved_gateway_commit,
        )
    if expected_validator_commit_norm:
        _add_check(
            checks,
            "validator_commit_matches_expected",
            _commit_matches(expected_validator_commit_norm, resolved_validator_commit),
            expected=expected_validator_commit_norm,
            actual=resolved_validator_commit,
        )
    if require_same_commit:
        _add_check(
            checks,
            "gateway_validator_commits_match",
            _commit_matches(resolved_gateway_commit, resolved_validator_commit),
            expected=resolved_gateway_commit,
            actual=resolved_validator_commit,
        )

    if require_pcr0 or resolved_gateway_pcr0 or expected_gateway_pcr0_norm:
        _add_check(
            checks,
            "gateway_pcr0_present",
            bool(resolved_gateway_pcr0),
            actual=resolved_gateway_pcr0,
        )
    if require_pcr0 or resolved_validator_pcr0 or expected_validator_pcr0_norm:
        _add_check(
            checks,
            "validator_pcr0_present",
            bool(resolved_validator_pcr0),
            actual=resolved_validator_pcr0,
        )
    if resolved_gateway_pcr0:
        _add_check(
            checks,
            "gateway_pcr0_static_allowlisted",
            bool(gateway_static.get("allowed")),
            actual=resolved_gateway_pcr0,
            detail="gateway PCR0s are verified by the static allowlist",
        )
    if resolved_validator_pcr0:
        _add_check(
            checks,
            "validator_pcr0_accepted",
            validator_pcr0_accepted,
            actual=resolved_validator_pcr0,
            detail="validator PCR0 is accepted by dynamic cache or static allowlist",
        )
    if expected_gateway_pcr0_norm:
        _add_check(
            checks,
            "gateway_pcr0_matches_expected",
            _pcr0_matches(expected_gateway_pcr0_norm, resolved_gateway_pcr0),
            expected=expected_gateway_pcr0_norm,
            actual=resolved_gateway_pcr0,
        )
    if expected_validator_pcr0_norm:
        _add_check(
            checks,
            "validator_pcr0_matches_expected",
            _pcr0_matches(expected_validator_pcr0_norm, resolved_validator_pcr0),
            expected=expected_validator_pcr0_norm,
            actual=resolved_validator_pcr0,
        )
    if require_pcr0_commit_match and resolved_gateway_pcr0:
        _add_check(
            checks,
            "gateway_pcr0_commit_matches_gateway_commit",
            _allowlist_commit_matches_runtime(gateway_static, resolved_gateway_commit),
            expected=resolved_gateway_commit,
            actual=gateway_static.get("matched_entry_commits"),
        )
    if require_pcr0_commit_match and resolved_validator_pcr0:
        dynamic_commit_matches = bool(
            resolved_validator_commit and validator_dynamic.get("valid")
        )
        static_commit_matches = _allowlist_commit_matches_runtime(
            validator_static,
            resolved_validator_commit,
        )
        _add_check(
            checks,
            "validator_pcr0_commit_matches_validator_commit",
            dynamic_commit_matches or static_commit_matches,
            expected=resolved_validator_commit,
            actual={
                "dynamic": (validator_dynamic.get("verification") or {}).get(
                    "commit_hash"
                ),
                "static": validator_static.get("matched_entry_commits"),
            },
        )
    if docker_health is not None:
        _add_check(
            checks,
            "docker_build_health",
            bool(docker_health.get("ok")),
            severity="error" if require_docker_build_health else "warning",
            detail=(
                "Docker host/build health; require flag runs a tiny scratch-image smoke build "
                "and blocks resume on failure"
            ),
            actual={
                "docker_root": (docker_health.get("docker_info") or {}).get("docker_root"),
                "disk": docker_health.get("disk"),
                "smoke_build_requested": docker_health.get("smoke_build_requested"),
                "smoke_build_ok": (
                    (docker_health.get("smoke_build") or {}).get("ok")
                    if docker_health.get("smoke_build") is not None
                    else None
                ),
            },
        )

    ok = all(check["ok"] for check in checks if check.get("severity") == "error")
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "ok": ok,
        "build_time_utc": build_info.get("build_time_utc", UNKNOWN),
        "source_commit_path": source_commit_path,
        "gateway": {
            "commit": resolved_gateway_commit,
            "build_info": build_info,
            "pcr0": resolved_gateway_pcr0,
            "pcr0_static_allowlist": gateway_static,
        },
        "validator": {
            "commit": resolved_validator_commit,
            "pcr0": resolved_validator_pcr0,
            "pcr0_static_allowlist": validator_static,
            "pcr0_dynamic_cache": validator_dynamic,
            "pcr0_accepted": validator_pcr0_accepted,
        },
        "expected": {
            "gateway_commit": expected_gateway_commit_norm,
            "validator_commit": expected_validator_commit_norm,
            "gateway_pcr0": expected_gateway_pcr0_norm,
            "validator_pcr0": expected_validator_pcr0_norm,
            "require_same_commit": require_same_commit,
            "require_pcr0": require_pcr0,
            "require_pcr0_commit_match": require_pcr0_commit_match,
            "include_docker_health": include_docker_health,
            "require_docker_build_health": require_docker_build_health,
        },
        "host_health": {
            "docker": docker_health,
        },
        "checks": checks,
    }


def write_deploy_readiness_manifest(
    document: Mapping[str, Any],
    path: str | Path | None = None,
    *,
    enforce_resume_block: bool = True,
) -> Path:
    target = Path(path).expanduser() if path else default_manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload["enforce_resume_block"] = bool(enforce_resume_block)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def load_deploy_readiness_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    target = Path(path).expanduser() if path else default_manifest_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise RuntimeError(f"deploy readiness manifest is not an object: {target}")
    payload.setdefault("manifest_path", str(target))
    return payload


def assert_resume_allowed(path: str | Path | None = None) -> dict[str, Any] | None:
    manifest = load_deploy_readiness_manifest(path)
    if manifest is None:
        return None
    if not parse_bool(manifest.get("enforce_resume_block"), default=True):
        return manifest
    if manifest.get("ok") is True:
        return manifest
    failed = [
        check.get("name")
        for check in manifest.get("checks", [])
        if isinstance(check, Mapping) and check.get("severity") == "error" and not check.get("ok")
    ]
    raise RuntimeError(
        "deploy readiness guard blocked resume"
        + (f"; failing checks: {', '.join(str(item) for item in failed)}" if failed else "")
    )
