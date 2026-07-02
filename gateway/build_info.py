"""Gateway build provenance helpers.

The production gateway may run from a copied source tree with no ``.git``
directory.  This module gives deployments a durable, non-secret build metadata
file while still supporting env/git fallbacks for local development.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


UNKNOWN = "unknown"
SERVICE_NAME = "leadpoet-gateway"
BUILD_INFO_FILENAME = "BUILD_INFO.json"

_UNKNOWN_VALUES = {"", UNKNOWN, "none", "null", "undefined"}
_ENV_KEYS = {
    "build_id": ("BUILD_ID",),
    "git_commit": ("GITHUB_SHA", "GITHUB_COMMIT", "GIT_COMMIT_HASH", "GIT_COMMIT"),
    "git_branch": ("GITHUB_REF_NAME", "GIT_BRANCH", "BRANCH_NAME"),
    "git_tag": ("GITHUB_TAG", "GIT_TAG"),
    "build_time_utc": ("BUILD_TIME_UTC", "BUILD_TIMESTAMP"),
}
_DOC_KEYS = {
    "build_id": ("build_id",),
    "git_commit": ("git_commit", "github_commit", "commit", "sha"),
    "git_branch": ("git_branch", "branch"),
    "git_tag": ("git_tag", "tag"),
    "git_dirty": ("git_dirty", "dirty"),
    "git_remote": ("git_remote", "remote"),
    "build_time_utc": ("build_time_utc", "build_time", "built_at"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _UNKNOWN_VALUES:
        return None
    return text


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _short_commit(commit: Any) -> str:
    cleaned = _clean_string(commit)
    return cleaned[:12] if cleaned else UNKNOWN


def _sanitize_remote(remote: str | None) -> str | None:
    cleaned = _clean_string(remote)
    if not cleaned:
        return None
    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return cleaned
    if parsed.scheme and parsed.netloc and "@" in parsed.netloc:
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    return cleaned


def _run_git(args: list[str], cwd: Path, timeout_seconds: float = 2.0) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    return _clean_string(result.stdout)


def _resolve_git_root(start: Path | None = None) -> Path | None:
    root = _run_git(["rev-parse", "--show-toplevel"], (start or Path.cwd()).resolve())
    return Path(root) if root else None


def collect_git_metadata(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Collect git metadata when a checkout is available."""
    candidates: list[Path] = []
    if repo_root:
        candidates.append(Path(repo_root).expanduser())
    env_root = _clean_string(os.getenv("GATEWAY_BUILD_INFO_GIT_ROOT"))
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parent])

    git_root: Path | None = None
    for candidate in candidates:
        git_root = _resolve_git_root(candidate)
        if git_root:
            break
    if not git_root:
        return {}

    commit = _run_git(["rev-parse", "HEAD"], git_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], git_root)
    if branch == "HEAD":
        branch = None
    tag = _run_git(["describe", "--tags", "--exact-match"], git_root)
    status = _run_git(["status", "--porcelain"], git_root)
    remote = _sanitize_remote(_run_git(["config", "--get", "remote.origin.url"], git_root))

    metadata: dict[str, Any] = {"git_root": str(git_root)}
    if commit:
        metadata["git_commit"] = commit
    if branch:
        metadata["git_branch"] = branch
    if tag:
        metadata["git_tag"] = tag
    metadata["git_dirty"] = bool(status)
    if remote:
        metadata["git_remote"] = remote
    return metadata


def _candidate_build_info_paths(explicit_path: Path | str | None = None) -> list[Path]:
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path).expanduser())
    env_path = _clean_string(os.getenv("GATEWAY_BUILD_INFO_FILE"))
    if env_path:
        paths.append(Path(env_path).expanduser())

    module_dir = Path(__file__).resolve().parent
    paths.extend(
        [
            Path.cwd() / BUILD_INFO_FILENAME,
            module_dir / BUILD_INFO_FILENAME,
            module_dir.parent / BUILD_INFO_FILENAME,
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


def _load_build_info_document(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "BUILD_INFO.json must contain a JSON object"
    return payload, None


def _set_string(
    info: dict[str, Any],
    field_sources: dict[str, str],
    key: str,
    value: Any,
    source: str,
) -> None:
    cleaned = _clean_string(value)
    if cleaned:
        info[key] = cleaned
        field_sources[key] = source


def _set_bool(
    info: dict[str, Any],
    field_sources: dict[str, str],
    key: str,
    value: Any,
    source: str,
) -> None:
    parsed = _parse_bool(value)
    if parsed is not None:
        info[key] = parsed
        field_sources[key] = source


def _overlay_mapping(
    info: dict[str, Any],
    field_sources: dict[str, str],
    payload: Mapping[str, Any],
    source: str,
) -> None:
    service = _clean_string(payload.get("service"))
    if service:
        info["service"] = service
        field_sources["service"] = source
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, int):
        info["schema_version"] = schema_version
        field_sources["schema_version"] = source

    for canonical_key, aliases in _DOC_KEYS.items():
        for alias in aliases:
            if alias not in payload:
                continue
            value = payload.get(alias)
            if canonical_key == "git_dirty":
                _set_bool(info, field_sources, canonical_key, value, source)
            elif canonical_key == "git_remote":
                _set_string(info, field_sources, canonical_key, _sanitize_remote(value), source)
            else:
                _set_string(info, field_sources, canonical_key, value, source)
            break


def _overlay_environment(info: dict[str, Any], field_sources: dict[str, str]) -> None:
    for canonical_key, env_names in _ENV_KEYS.items():
        for env_name in env_names:
            value = os.getenv(env_name)
            if _clean_string(value):
                _set_string(info, field_sources, canonical_key, value, f"env:{env_name}")
                break
    dirty = _parse_bool(os.getenv("GIT_DIRTY"))
    if dirty is not None:
        _set_bool(info, field_sources, "git_dirty", dirty, "env:GIT_DIRTY")


def _default_info(service: str = SERVICE_NAME) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "service": service,
        "build_id": UNKNOWN,
        "git_commit": UNKNOWN,
        "git_commit_short": UNKNOWN,
        "git_branch": UNKNOWN,
        "git_tag": UNKNOWN,
        "git_dirty": None,
        "git_remote": UNKNOWN,
        "build_time_utc": UNKNOWN,
    }


def load_build_info(
    *,
    discover_git: bool = True,
    build_info_path: Path | str | None = None,
    service: str = SERVICE_NAME,
) -> dict[str, Any]:
    """Return runtime build info with env > file > git precedence."""
    info = _default_info(service=service)
    field_sources: dict[str, str] = {}

    if discover_git:
        git_metadata = collect_git_metadata()
        if git_metadata:
            _overlay_mapping(info, field_sources, git_metadata, "git")

    paths_checked = _candidate_build_info_paths(build_info_path)
    selected_path: Path | None = None
    load_error: str | None = None
    for path in paths_checked:
        if not path.is_file():
            continue
        selected_path = path
        payload, load_error = _load_build_info_document(path)
        if payload is not None:
            _overlay_mapping(info, field_sources, payload, f"file:{path}")
            load_error = None
        break

    _overlay_environment(info, field_sources)

    info["git_commit_short"] = _short_commit(info.get("git_commit"))
    if info["build_id"] == UNKNOWN and info["git_commit_short"] != UNKNOWN:
        info["build_id"] = f"{service}:{info['git_commit_short']}"
        field_sources["build_id"] = "generated"
    info["is_commit_known"] = info["git_commit_short"] != UNKNOWN
    info["commit_source"] = field_sources.get("git_commit", UNKNOWN)
    info["build_id_source"] = field_sources.get("build_id", UNKNOWN)
    info["build_info_path"] = str(selected_path) if selected_path else None
    info["build_info_paths_checked"] = [str(path) for path in paths_checked]
    info["build_info_error"] = load_error
    info["loaded_at_utc"] = utc_now()
    info["runtime_cwd"] = str(Path.cwd())
    return info


@lru_cache(maxsize=1)
def get_build_info() -> dict[str, Any]:
    return load_build_info()


def create_build_info_document(
    *,
    repo_root: Path | str | None = None,
    service: str = SERVICE_NAME,
    build_id: str | None = None,
) -> dict[str, Any]:
    """Create the JSON document that should be shipped with a gateway deploy."""
    info = _default_info(service=service)
    field_sources: dict[str, str] = {}
    git_metadata = collect_git_metadata(repo_root)
    if git_metadata:
        _overlay_mapping(info, field_sources, git_metadata, "git")
    _overlay_environment(info, field_sources)
    if build_id:
        _set_string(info, field_sources, "build_id", build_id, "arg:build_id")

    build_time = utc_now()
    info["build_time_utc"] = build_time
    field_sources["build_time_utc"] = "generated"
    info["git_commit_short"] = _short_commit(info.get("git_commit"))
    if info["build_id"] == UNKNOWN:
        stamp = build_time.replace("-", "").replace(":", "")
        info["build_id"] = f"{service}:{info['git_commit_short']}:{stamp}"
        field_sources["build_id"] = "generated"

    return {
        "schema_version": 1,
        "service": info["service"],
        "build_id": info["build_id"],
        "git_commit": info["git_commit"],
        "git_commit_short": info["git_commit_short"],
        "git_branch": info["git_branch"],
        "git_tag": info["git_tag"],
        "git_dirty": info["git_dirty"],
        "git_remote": info["git_remote"],
        "build_time_utc": info["build_time_utc"],
        "generated_by": "scripts/write_gateway_build_info.py",
    }


def write_build_info_file(output_path: Path | str, document: Mapping[str, Any]) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path
