#!/usr/bin/env python3
"""Prepare, activate, and record one exact gateway Git deployment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = "leadpoet.gateway_git_deployment.v1"
TREE_VERIFICATION_SCHEMA_VERSION = (
    "leadpoet.gateway_candidate_tree_verification.v1"
)
DEFAULT_REPO_URL = "https://github.com/leadpoet/leadpoet.git"
DEFAULT_BRANCH = "main"
RESTART_PROTOCOL_MARKER = 'GATEWAY_GIT_DEPLOY_PROTOCOL="1"'
GIT_FETCH_MAX_ATTEMPTS = 4
BENCHMARK_DISCLOSURE_SCHEMA_CAPABILITY = (
    "lab_arena_benchmark_disclosure_schema_v1",
    {
        "schema_version": "leadpoet.lab_arena.benchmark_disclosure.v1",
        "version": 210,
        "policy": "commit_reveal_day2_v1",
    },
)
_BENCHMARK_DISCLOSURE_COMPATIBILITY_PATH = (
    "gateway/tee/supabase_schema_preflight_v2.py"
)
_MAX_SCHEMA_RESPONSE_BYTES = 64 * 1024

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_TRANSIENT_GIT_FETCH_MARKERS = (
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "curl 18",
    "curl 28",
    "curl 35",
    "curl 52",
    "curl 55",
    "curl 56",
    "curl 92",
    "connection reset",
    "could not resolve host",
    "early eof",
    "expected 'acknowledgments'",
    "failed to connect",
    "remote end hung up unexpectedly",
    "temporary failure in name resolution",
    "timeoutexpired",
)


class GatewayGitDeployError(RuntimeError):
    """Raised when a gateway Git deployment cannot proceed safely."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _run_git(repo_root: Path, *args: str, timeout: float = 120.0) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayGitDeployError(f"git {args[0]} could not run: {type(exc).__name__}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown git error").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise GatewayGitDeployError(f"git {args[0]} failed: {detail}")
    return result.stdout.strip()


def _run_git_bytes(
    repo_root: Path,
    *args: str,
    timeout: float = 120.0,
) -> bytes:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayGitDeployError(
            f"git {args[0]} could not run: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"unknown git error")
        rendered = detail.decode("utf-8", errors="replace").strip()
        if len(rendered) > 500:
            rendered = rendered[:500] + "..."
        raise GatewayGitDeployError(f"git {args[0]} failed: {rendered}")
    return result.stdout


def _fetch_branch_with_retry(repo_root: Path, branch: str) -> None:
    """Retry only transport-class fetch failures before production shutdown."""

    for attempt in range(1, GIT_FETCH_MAX_ATTEMPTS + 1):
        try:
            _run_git(
                repo_root,
                "fetch",
                "--prune",
                "origin",
                f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
            )
            return
        except GatewayGitDeployError as exc:
            detail = str(exc).lower()
            is_transient = any(
                marker in detail for marker in _TRANSIENT_GIT_FETCH_MARKERS
            )
            if not is_transient or attempt == GIT_FETCH_MAX_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))


def _candidate_blob_entries(
    repo_root: Path,
    target_sha: str,
) -> dict[str, tuple[str, str]]:
    raw = _run_git_bytes(
        repo_root,
        "ls-tree",
        "-rz",
        "--full-tree",
        target_sha,
        timeout=120,
    )
    entries: dict[str, tuple[str, str]] = {}
    for encoded in raw.split(b"\x00"):
        if not encoded:
            continue
        try:
            metadata, encoded_path = encoded.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise GatewayGitDeployError(
                "candidate Git tree contains an unsupported path or entry"
            ) from exc
        if object_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise GatewayGitDeployError(
                f"candidate Git tree contains unsupported entry: {path}"
            )
        if (
            not path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise GatewayGitDeployError("candidate Git tree path is unsafe")
        entries[path] = (mode, object_id.lower())
    if not entries:
        raise GatewayGitDeployError("candidate Git tree has no blobs")
    return entries


def _git_blob_id(payload: bytes, object_format: str) -> str:
    try:
        digest = hashlib.new(object_format)
    except ValueError as exc:
        raise GatewayGitDeployError(
            f"unsupported Git object format: {object_format}"
        ) from exc
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def verify_materialized_tree(
    *,
    repo_root: Path,
    materialized_root: Path,
    target_sha: str,
    strict_extras: bool,
) -> dict[str, Any]:
    """Verify candidate files and modes against the commit's exact Git blobs."""

    repo_root = repo_root.expanduser().resolve()
    materialized_root = materialized_root.expanduser().resolve()
    if not materialized_root.is_dir():
        raise GatewayGitDeployError(
            f"materialized candidate tree is missing: {materialized_root}"
        )
    resolved_sha = _run_git(
        repo_root,
        "rev-parse",
        f"{target_sha}^{{commit}}",
        timeout=30,
    ).lower()
    if resolved_sha != target_sha:
        raise GatewayGitDeployError("candidate tree target SHA is not exact")
    object_format = _run_git(
        repo_root,
        "rev-parse",
        "--show-object-format",
        timeout=30,
    ).lower()
    expected = _candidate_blob_entries(repo_root, target_sha)

    verified_entries: list[dict[str, str]] = []
    for relative_path, (expected_mode, expected_oid) in sorted(expected.items()):
        candidate = materialized_root / relative_path
        try:
            file_stat = candidate.lstat()
        except OSError as exc:
            raise GatewayGitDeployError(
                f"candidate tree is missing Git blob path: {relative_path}"
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode):
            actual_mode = "120000"
            payload = os.fsencode(os.readlink(candidate))
        elif stat.S_ISREG(file_stat.st_mode):
            actual_mode = (
                "100755" if file_stat.st_mode & 0o111 else "100644"
            )
            try:
                payload = candidate.read_bytes()
            except OSError as exc:
                raise GatewayGitDeployError(
                    f"candidate Git blob is unreadable: {relative_path}"
                ) from exc
        else:
            raise GatewayGitDeployError(
                f"candidate Git blob path has invalid type: {relative_path}"
            )
        if actual_mode != expected_mode:
            raise GatewayGitDeployError(
                f"candidate Git blob mode mismatch: {relative_path}"
            )
        actual_oid = _git_blob_id(payload, object_format)
        if actual_oid != expected_oid:
            raise GatewayGitDeployError(
                f"candidate Git blob content mismatch: {relative_path}"
            )
        verified_entries.append(
            {
                "mode": expected_mode,
                "object_id": expected_oid,
                "path": relative_path,
            }
        )

    ignored_runtime_cache_count = 0
    if strict_extras:
        actual_paths: set[str] = set()
        for path in materialized_root.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(materialized_root)
            if ".git" in relative.parts:
                continue
            if "__pycache__" in relative.parts or path.suffix in {
                ".pyc",
                ".pyo",
            }:
                ignored_runtime_cache_count += 1
                continue
            actual_paths.add(relative.as_posix())
        extra_paths = sorted(actual_paths.difference(expected))
        if extra_paths:
            raise GatewayGitDeployError(
                f"candidate tree contains non-Git path: {extra_paths[0]}"
            )

    manifest_payload = json.dumps(
        verified_entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": TREE_VERIFICATION_SCHEMA_VERSION,
        "target_sha": target_sha,
        "tree_hash": _run_git(
            repo_root,
            "rev-parse",
            f"{target_sha}^{{tree}}",
            timeout=30,
        ).lower(),
        "object_format": object_format,
        "blob_count": len(verified_entries),
        "blob_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "strict_extras": bool(strict_extras),
        "ignored_runtime_cache_count": ignored_runtime_cache_count,
    }


def _tree_verification_evidence(
    *,
    repo_root: Path,
    materialized_root: Path,
    target_sha: str,
    phase: str,
    strict_extras: bool,
) -> dict[str, Any]:
    if phase not in {"prepared_archive", "activated_checkout"}:
        raise GatewayGitDeployError("candidate tree verification phase is invalid")
    evidence = verify_materialized_tree(
        repo_root=repo_root,
        materialized_root=materialized_root,
        target_sha=target_sha,
        strict_extras=strict_extras,
    )
    evidence.update(
        {
            "phase": phase,
            "verified_at": _utc_now(),
        }
    )
    return evidence


def write_tree_verification_evidence(
    *,
    repo_root: Path,
    materialized_root: Path,
    target_sha: str,
    phase: str,
    strict_extras: bool,
    output_path: Path,
) -> dict[str, Any]:
    evidence = _tree_verification_evidence(
        repo_root=repo_root,
        materialized_root=materialized_root,
        target_sha=target_sha,
        phase=phase,
        strict_extras=strict_extras,
    )
    _atomic_write_json(output_path, evidence)
    return evidence


def record_tree_verification(
    *,
    plan_file: Path,
    materialized_root: Path,
    phase: str,
    strict_extras: bool,
) -> dict[str, Any]:
    document = _read_json(plan_file)
    repo_root = Path(str(document["repo_root"])).resolve()
    target_sha = str(document.get("target_sha") or "").lower()
    evidence = _tree_verification_evidence(
        repo_root=repo_root,
        materialized_root=materialized_root,
        target_sha=target_sha,
        phase=phase,
        strict_extras=strict_extras,
    )
    if evidence["tree_hash"] != document.get("tree_hash"):
        raise GatewayGitDeployError(
            "candidate tree verification differs from prepared tree hash"
        )
    verifications = dict(document.get("tree_verifications") or {})
    verifications[phase] = evidence
    document["tree_verifications"] = verifications
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(str(document["manifest_file"])), document)
    return evidence


def _read_prepared_tree_evidence(
    path: Path,
    *,
    target_sha: str,
    tree_hash: str,
) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayGitDeployError(
            "prepared candidate tree evidence is unavailable"
        ) from exc
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("schema_version") != TREE_VERIFICATION_SCHEMA_VERSION
        or evidence.get("phase") != "prepared_archive"
        or evidence.get("target_sha") != target_sha
        or evidence.get("tree_hash") != tree_hash
        or evidence.get("strict_extras") is not True
        or not isinstance(evidence.get("blob_count"), int)
        or int(evidence["blob_count"]) < 1
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(evidence.get("blob_manifest_sha256") or ""),
        )
    ):
        raise GatewayGitDeployError(
            "prepared candidate tree evidence is invalid"
        )
    return dict(evidence)


def record_tree_verification_pair(
    *,
    plan_file: Path,
    prepared_evidence_path: Path,
    activated_root: Path,
) -> dict[str, Any]:
    document = _read_json(plan_file)
    repo_root = Path(str(document["repo_root"])).resolve()
    target_sha = str(document.get("target_sha") or "").lower()
    tree_hash = str(document.get("tree_hash") or "").lower()
    prepared = _read_prepared_tree_evidence(
        prepared_evidence_path,
        target_sha=target_sha,
        tree_hash=tree_hash,
    )
    activated = _tree_verification_evidence(
        repo_root=repo_root,
        materialized_root=activated_root,
        target_sha=target_sha,
        phase="activated_checkout",
        strict_extras=False,
    )
    for field in (
        "tree_hash",
        "object_format",
        "blob_count",
        "blob_manifest_sha256",
    ):
        if prepared.get(field) != activated.get(field):
            raise GatewayGitDeployError(
                f"prepared and activated candidate trees differ: {field}"
            )
    document["tree_verifications"] = {
        "prepared_archive": prepared,
        "activated_checkout": activated,
    }
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(str(document["manifest_file"])), document)
    return {
        "prepared_archive": prepared,
        "activated_checkout": activated,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayGitDeployError(f"deployment plan is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise GatewayGitDeployError("deployment plan schema is invalid")
    return value


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): "" if value is None else str(value) for key, value in parsed.items()}

    values: dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = [line]
        candidate = parts[0] if len(parts) == 1 else line
        if "=" not in candidate:
            continue
        key, value = candidate.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values[key] = value
    return values


def _configured_value(
    name: str,
    explicit: str | None,
    env_file_values: Mapping[str, str],
    default: str,
) -> str:
    return str(explicit or os.getenv(name) or env_file_values.get(name) or default).strip()


def _operator_only_value(name: str, explicit: str | None, default: str) -> str:
    """Resolve one-invocation controls without consulting persistent runtime config."""

    return str(explicit or os.getenv(name) or default).strip()


def _sanitize_remote(remote: str) -> str:
    value = str(remote or "").strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        scp_style = re.fullmatch(r"[^@\s]+@([^:\s]+):(.+)", value)
        if scp_style:
            return f"{scp_style.group(1).lower()}:{scp_style.group(2)}"
        return value
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), host.lower(), parsed.path, "", ""))


def _canonical_remote(remote: str) -> str:
    value = _sanitize_remote(remote).rstrip("/")
    return value[:-4] if value.endswith(".git") else value


def _validate_branch(repo_root: Path, branch: str) -> None:
    if not branch or branch.startswith("-"):
        raise GatewayGitDeployError("configured Git branch is invalid")
    _run_git(repo_root, "check-ref-format", "--branch", branch, timeout=10)


def _validate_checkout(repo_root: Path, expected_remote: str) -> None:
    if not repo_root.is_dir():
        raise GatewayGitDeployError(f"gateway Git checkout is missing: {repo_root}")
    actual_root = Path(_run_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != repo_root.resolve():
        raise GatewayGitDeployError("configured gateway repository root is not the Git toplevel")
    actual_remote = _run_git(repo_root, "remote", "get-url", "origin")
    if _canonical_remote(actual_remote) != _canonical_remote(expected_remote):
        raise GatewayGitDeployError("gateway Git origin does not match GITHUB_REPO_URL")


def _require_clean_checkout(repo_root: Path) -> None:
    status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        first_path = status.splitlines()[0][3:]
        raise GatewayGitDeployError(f"gateway Git checkout is dirty: {first_path}")


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode not in (0, 1):
        raise GatewayGitDeployError("git merge-base failed while validating deployment history")
    return result.returncode == 0


def _validate_target_restart_protocol(repo_root: Path, target_sha: str) -> None:
    restart_script = _run_git(repo_root, "show", f"{target_sha}:gw_restart.sh", timeout=30)
    if RESTART_PROTOCOL_MARKER not in restart_script:
        raise GatewayGitDeployError(
            "selected commit does not support the gateway Git restart handoff protocol"
        )
    _run_git(
        repo_root,
        "cat-file",
        "-e",
        f"{target_sha}:scripts/gateway_git_deploy.py",
        timeout=30,
    )


def _arena_schema_authority(
    environment: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    url = str(
        environment.get("LAB_ARENA_SUPABASE_URL")
        or environment.get("SUPABASE_URL")
        or ""
    ).rstrip("/")
    service_key = str(environment.get("LAB_ARENA_SERVICE_KEY") or "").strip()
    service_jwt = str(environment.get("LAB_ARENA_SERVICE_JWT") or "").strip()
    anon_key = str(environment.get("LAB_ARENA_SUPABASE_ANON_KEY") or "").strip()
    if not url or bool(service_key) == bool(service_jwt):
        raise GatewayGitDeployError(
            "Arena benchmark disclosure compatibility authority is unavailable"
        )
    if service_key:
        if not service_key.startswith("sb_secret_"):
            raise GatewayGitDeployError(
                "Arena benchmark disclosure compatibility authority is invalid"
            )
        return url, {"Accept": "application/json", "apikey": service_key}
    if not anon_key or service_jwt.count(".") != 2:
        raise GatewayGitDeployError(
            "Arena benchmark disclosure compatibility authority is invalid"
        )
    return url, {
        "Accept": "application/json",
        "Authorization": f"Bearer {service_jwt}",
        "apikey": anon_key,
    }


def _read_schema_response(response: Any) -> Any:
    encoded = response.read(_MAX_SCHEMA_RESPONSE_BYTES + 1)
    if len(encoded) > _MAX_SCHEMA_RESPONSE_BYTES:
        raise GatewayGitDeployError(
            "Arena benchmark disclosure compatibility response is oversized"
        )
    try:
        return json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise GatewayGitDeployError(
            "Arena benchmark disclosure compatibility response is invalid"
        ) from exc


def _benchmark_disclosure_rows_exist(
    url: str,
    headers: Mapping[str, str],
    *,
    opener: Any,
    timeout_seconds: float,
) -> bool:
    query = urlencode(
        {
            "select": "round_id",
            # JSON null must still count as an explicit, fail-closed policy marker.
            # The JSON operator keeps it distinct from an absent key.
            "configuration_doc->benchmark_disclosure_policy": "not.is.null",
            "limit": "1",
        }
    )
    request = Request(
        f"{url}/rest/v1/lab_arena_rounds?{query}",
        headers=dict(headers),
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            if not 200 <= status < 300:
                raise GatewayGitDeployError(
                    "Arena benchmark disclosure round probe failed"
                )
            value = _read_schema_response(response)
    except GatewayGitDeployError:
        raise
    except Exception as exc:
        raise GatewayGitDeployError(
            "Arena benchmark disclosure round probe failed"
        ) from exc
    if not isinstance(value, list) or len(value) > 1 or any(
        not isinstance(row, Mapping)
        or set(row) != {"round_id"}
        or not isinstance(row["round_id"], str)
        or not row["round_id"]
        for row in value
    ):
        raise GatewayGitDeployError(
            "Arena benchmark disclosure round probe response is invalid"
        )
    return bool(value)


def _probe_benchmark_disclosure_capability(
    environment: Mapping[str, str],
    *,
    opener: Any = urlopen,
    timeout_seconds: float = 10.0,
) -> bool:
    """Return false only for the exact pre-migration missing-RPC response."""

    url, headers = _arena_schema_authority(environment)
    function_name, expected = BENCHMARK_DISCLOSURE_SCHEMA_CAPABILITY
    request = Request(
        f"{url}/rest/v1/rpc/{function_name}",
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            if not 200 <= status < 300:
                raise GatewayGitDeployError(
                    "Arena benchmark disclosure compatibility probe failed"
                )
            value = _read_schema_response(response)
    except HTTPError as exc:
        try:
            missing = _read_schema_response(exc)
        except GatewayGitDeployError:
            missing = None
        if (
            exc.code == 404
            and isinstance(missing, Mapping)
            and missing.get("code") == "PGRST202"
        ):
            if _benchmark_disclosure_rows_exist(
                url,
                headers,
                opener=opener,
                timeout_seconds=timeout_seconds,
            ):
                raise GatewayGitDeployError(
                    "Arena benchmark disclosure rows exist without the required schema capability"
                )
            return False
        raise GatewayGitDeployError(
            "Arena benchmark disclosure compatibility probe failed"
        ) from exc
    except GatewayGitDeployError:
        raise
    except Exception as exc:
        raise GatewayGitDeployError(
            "Arena benchmark disclosure compatibility probe failed"
        ) from exc
    if value != expected:
        raise GatewayGitDeployError(
            "Arena benchmark disclosure schema capability differs"
        )
    return True


def _target_benchmark_disclosure_capability(
    repo_root: Path, target_sha: str
) -> Any:
    try:
        source = _run_git_bytes(
            repo_root,
            "show",
            f"{target_sha}:{_BENCHMARK_DISCLOSURE_COMPATIBILITY_PATH}",
            timeout=30,
        ).decode("utf-8")
        tree = ast.parse(source, filename=_BENCHMARK_DISCLOSURE_COMPATIBILITY_PATH)
    except (GatewayGitDeployError, UnicodeDecodeError, SyntaxError):
        return None
    declarations = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(
            isinstance(target, ast.Name)
            and target.id == "BENCHMARK_DISCLOSURE_SCHEMA_CAPABILITY"
            for target in targets
        ):
            try:
                declarations.append(ast.literal_eval(statement.value))
            except (TypeError, ValueError):
                declarations.append(None)
    return declarations[0] if len(declarations) == 1 else None


def _verify_benchmark_disclosure_release_compatibility(
    *,
    repo_root: Path,
    target_sha: str,
    environment: Mapping[str, str],
    opener: Any = urlopen,
) -> str:
    if not _probe_benchmark_disclosure_capability(environment, opener=opener):
        return "legacy_schema"
    if (
        _target_benchmark_disclosure_capability(repo_root, target_sha)
        != BENCHMARK_DISCLOSURE_SCHEMA_CAPABILITY
    ):
        raise GatewayGitDeployError(
            "selected gateway release predates the installed benchmark disclosure reader floor"
        )
    return "required_and_supported"


def prepare_deployment(
    *,
    repo_root: Path,
    repo_url: str,
    branch: str,
    plan_file: Path,
    manifest_file: Path,
    last_good_file: Path,
    environment: Mapping[str, str],
    deploy_commit: str = "",
    compatibility_opener: Any = None,
) -> dict[str, Any]:
    """Fetch one branch and persist an immutable deployment decision."""

    repo_root = repo_root.expanduser().resolve()
    _validate_checkout(repo_root, repo_url)
    _validate_branch(repo_root, branch)
    _require_clean_checkout(repo_root)

    previous_sha = _run_git(repo_root, "rev-parse", "HEAD").lower()
    _fetch_branch_with_retry(repo_root, branch)
    branch_head_sha = _run_git(
        repo_root,
        "rev-parse",
        f"refs/remotes/origin/{branch}^{{commit}}",
    ).lower()
    if not _FULL_SHA_RE.fullmatch(branch_head_sha):
        raise GatewayGitDeployError("fetched branch did not resolve to a full Git commit")

    requested_sha = str(deploy_commit or "").strip().lower()
    if requested_sha:
        if not _FULL_SHA_RE.fullmatch(requested_sha):
            raise GatewayGitDeployError("GATEWAY_DEPLOY_COMMIT must be a full 40-character SHA")
        _run_git(repo_root, "cat-file", "-e", f"{requested_sha}^{{commit}}", timeout=30)
        if not _is_ancestor(repo_root, requested_sha, branch_head_sha):
            raise GatewayGitDeployError(
                "GATEWAY_DEPLOY_COMMIT is not reachable from the configured branch"
            )
        target_sha = requested_sha
        mode = "pinned"
    else:
        target_sha = branch_head_sha
        mode = "fast_forward"
        if not _is_ancestor(repo_root, previous_sha, target_sha):
            raise GatewayGitDeployError("configured branch is not a fast-forward from deployed HEAD")

    benchmark_disclosure_compatibility = (
        _verify_benchmark_disclosure_release_compatibility(
            repo_root=repo_root,
            target_sha=target_sha,
            environment=environment,
            opener=compatibility_opener or urlopen,
        )
    )
    _validate_target_restart_protocol(repo_root, target_sha)
    tree_hash = _run_git(repo_root, "rev-parse", f"{target_sha}^{{tree}}").lower()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "github",
        "status": "prepared",
        "stage": "git_prepare",
        "mode": mode,
        "repo_root": str(repo_root),
        "remote_url": _sanitize_remote(repo_url),
        "branch": branch,
        "previous_sha": previous_sha,
        "branch_head_sha": branch_head_sha,
        "target_sha": target_sha,
        "tree_hash": tree_hash,
        "benchmark_disclosure_compatibility": benchmark_disclosure_compatibility,
        "prepared_at": _utc_now(),
        "manifest_file": str(manifest_file.expanduser().resolve()),
        "last_good_file": str(last_good_file.expanduser().resolve()),
    }
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(document["manifest_file"]), document)
    return document


def activate_deployment(*, plan_file: Path) -> dict[str, Any]:
    """Move the stopped gateway checkout to the commit selected by prepare."""

    document = _read_json(plan_file)
    repo_root = Path(str(document["repo_root"])).resolve()
    _validate_checkout(repo_root, str(document["remote_url"]))
    _require_clean_checkout(repo_root)

    current_sha = _run_git(repo_root, "rev-parse", "HEAD").lower()
    if current_sha != document.get("previous_sha"):
        raise GatewayGitDeployError("gateway checkout HEAD changed after deployment prepare")

    target_sha = str(document.get("target_sha") or "").lower()
    branch = str(document.get("branch") or "")
    if not _FULL_SHA_RE.fullmatch(target_sha):
        raise GatewayGitDeployError("deployment target SHA is invalid")

    if document.get("mode") == "pinned":
        _run_git(repo_root, "checkout", "--detach", target_sha)
    elif document.get("mode") == "fast_forward":
        remote_head = _run_git(
            repo_root,
            "rev-parse",
            f"refs/remotes/origin/{branch}^{{commit}}",
        ).lower()
        if remote_head != target_sha:
            raise GatewayGitDeployError("prepared remote branch changed before activation")
        local_ref = f"refs/heads/{branch}"
        local_exists = subprocess.run(
            ["git", "-C", str(repo_root), "show-ref", "--verify", "--quiet", local_ref],
            check=False,
            timeout=10,
        ).returncode == 0
        if local_exists:
            local_sha = _run_git(repo_root, "rev-parse", local_ref).lower()
            if not _is_ancestor(repo_root, local_sha, target_sha):
                raise GatewayGitDeployError("local deployment branch cannot fast-forward to target")
            _run_git(repo_root, "checkout", branch)
            _run_git(repo_root, "merge", "--ff-only", target_sha)
        else:
            _run_git(repo_root, "checkout", "-b", branch, "--track", f"origin/{branch}")
    else:
        raise GatewayGitDeployError("deployment activation mode is invalid")

    activated_sha = _run_git(repo_root, "rev-parse", "HEAD").lower()
    if activated_sha != target_sha:
        raise GatewayGitDeployError("activated gateway commit does not match prepared target")
    _require_clean_checkout(repo_root)

    document.update(
        {
            "status": "activated",
            "stage": "git_activate",
            "activated_at": _utc_now(),
        }
    )
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(str(document["manifest_file"])), document)
    return document


def _find_pcr0(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).upper() == "PCR0":
                candidate = str(nested or "").strip().lower()
                if _PCR0_RE.fullmatch(candidate):
                    return candidate
            found = _find_pcr0(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_pcr0(nested)
            if found:
                return found
    return None


def _installed_release_role_pcr0s(eif_root: Path, target_sha: str) -> dict[str, str]:
    from gateway.tee.release_manifest_v2 import validate_prior_release_manifest

    path = eif_root / "gateway-v2-release-manifest.json"
    if not path.is_file() or path.is_symlink():
        raise GatewayGitDeployError("installed gateway release manifest is unavailable")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayGitDeployError(
            "installed gateway release manifest is unreadable"
        ) from exc
    release = validate_prior_release_manifest(document)
    if release["commit_sha"] != target_sha:
        raise GatewayGitDeployError("installed gateway release commit differs")
    return {role: str(summary["pcr0"]) for role, summary in release["roles"].items()}


def collect_role_pcr0s(
    eif_root: Path, expected: Mapping[str, str]
) -> dict[str, str]:
    role_pcr0s: dict[str, str] = {}
    for role, expected_pcr0 in sorted(expected.items()):
        path = eif_root / ("enclave-build-%s.json" % role)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise GatewayGitDeployError(
                "gateway role measurement is unavailable: %s" % role
            ) from None
        pcr0 = _find_pcr0(payload)
        if pcr0 != expected_pcr0:
            raise GatewayGitDeployError(
                "gateway role measurement differs from release: %s" % role
            )
        role_pcr0s[role] = pcr0
    return role_pcr0s


def finalize_deployment(
    *,
    plan_file: Path,
    status: str,
    stage: str,
    eif_root: Path,
) -> dict[str, Any]:
    if status not in {"succeeded", "failed"}:
        raise GatewayGitDeployError("deployment final status is invalid")
    document = _read_json(plan_file)
    expected_role_pcr0s = (
        _installed_release_role_pcr0s(eif_root, str(document.get("target_sha") or ""))
        if status == "succeeded"
        else {}
    )
    document.update(
        {
            "status": status,
            "stage": str(stage or "unknown"),
            "completed_at": _utc_now(),
            "role_pcr0s": collect_role_pcr0s(eif_root, expected_role_pcr0s)
            if status == "succeeded"
            else {},
        }
    )
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(str(document["manifest_file"])), document)
    if status == "succeeded":
        _atomic_write_json(Path(str(document["last_good_file"])), document)
    return document


def repair_last_good_role_pcr0s(
    *, last_good_file: Path, archive_root: Path
) -> dict[str, Any]:
    from gateway.tee.release_archive_v2 import (
        DEFAULT_RETAIN_RELEASES,
        ReleaseArchiveV2Error,
        _archived_role_pcr0s,
        load_last_good_release,
        verify_archive_index,
    )

    document = _read_json(last_good_file)
    try:
        validated = load_last_good_release(last_good_file)
        target = validated["commit_sha"]
        existing = validated["role_pcr0s"]
        index = verify_archive_index(
            archive_root=archive_root,
            minimum_releases=1,
            maximum_releases=DEFAULT_RETAIN_RELEASES,
        )
        matches = [
            item for item in index["releases"] if item["commit_sha"] == target
        ]
        if len(matches) != 1:
            raise GatewayGitDeployError("last-good archive is not unique")
        archived = _archived_role_pcr0s(archive_root, matches[0])
    except ReleaseArchiveV2Error as exc:
        raise GatewayGitDeployError(str(exc)) from exc
    if archived == existing:
        return document
    if set(existing) - set(archived) != {"gateway_autoresearch"} or any(
        existing.get(role) != pcr0 for role, pcr0 in archived.items()
    ):
        raise GatewayGitDeployError("last-good retained role PCR0s differ from archive")
    repaired = {**document, "role_pcr0s": archived}
    if _read_json(last_good_file) != document:
        raise GatewayGitDeployError("last-good deployment changed during repair")
    _atomic_write_json(last_good_file, repaired)
    return repaired


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True, type=Path)
    prepare.add_argument("--env-file", type=Path)
    prepare.add_argument("--repo-url")
    prepare.add_argument("--branch")
    prepare.add_argument("--deploy-commit")
    prepare.add_argument("--plan-file", required=True, type=Path)
    prepare.add_argument("--manifest-file", required=True, type=Path)
    prepare.add_argument("--last-good-file", required=True, type=Path)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--plan-file", required=True, type=Path)

    verify_tree = subparsers.add_parser("verify-tree")
    verify_tree.add_argument("--plan-file", required=True, type=Path)
    verify_tree.add_argument("--materialized-root", required=True, type=Path)
    verify_tree.add_argument(
        "--phase",
        required=True,
        choices=("prepared_archive", "activated_checkout"),
    )
    verify_tree.add_argument("--strict-extras", action="store_true")

    verify_tree_pair = subparsers.add_parser("verify-tree-pair")
    verify_tree_pair.add_argument("--plan-file", required=True, type=Path)
    verify_tree_pair.add_argument(
        "--prepared-evidence",
        required=True,
        type=Path,
    )
    verify_tree_pair.add_argument(
        "--activated-root",
        required=True,
        type=Path,
    )

    field = subparsers.add_parser("field")
    field.add_argument("--plan-file", required=True, type=Path)
    field.add_argument(
        "--name",
        required=True,
        choices=("repo_root", "remote_url", "branch", "previous_sha", "target_sha", "tree_hash"),
    )

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--plan-file", required=True, type=Path)
    finalize.add_argument("--status", required=True, choices=("succeeded", "failed"))
    finalize.add_argument("--stage", required=True)
    finalize.add_argument("--eif-root", type=Path, default=Path("/home/ec2-user/tee"))
    repair = subparsers.add_parser("repair-last-good-role-pcr0s")
    repair.add_argument("--last-good-file", required=True, type=Path)
    repair.add_argument("--archive-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            env_values = _read_env_file(args.env_file)
            document = prepare_deployment(
                repo_root=args.repo_root,
                repo_url=_configured_value(
                    "GITHUB_REPO_URL", args.repo_url, env_values, DEFAULT_REPO_URL
                ),
                branch=_configured_value("GITHUB_BRANCH", args.branch, env_values, DEFAULT_BRANCH),
                plan_file=args.plan_file,
                manifest_file=args.manifest_file,
                last_good_file=args.last_good_file,
                environment=env_values,
                deploy_commit=_operator_only_value(
                    "GATEWAY_DEPLOY_COMMIT", args.deploy_commit, ""
                ),
            )
            print(document["target_sha"])
            return 0
        if args.command == "activate":
            document = activate_deployment(plan_file=args.plan_file)
            print(document["target_sha"])
            return 0
        if args.command == "repair-last-good-role-pcr0s":
            before = _read_json(args.last_good_file)
            document = repair_last_good_role_pcr0s(
                last_good_file=args.last_good_file,
                archive_root=args.archive_root,
            )
            print(
                json.dumps(
                    {
                        "status": "verified"
                        if before == document
                        else "repaired",
                        "target_sha": document["target_sha"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "verify-tree":
            evidence = record_tree_verification(
                plan_file=args.plan_file,
                materialized_root=args.materialized_root,
                phase=args.phase,
                strict_extras=args.strict_extras,
            )
            print(
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "verify-tree-pair":
            evidence = record_tree_verification_pair(
                plan_file=args.plan_file,
                prepared_evidence_path=args.prepared_evidence,
                activated_root=args.activated_root,
            )
            print(
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "field":
            document = _read_json(args.plan_file)
            print(document[args.name])
            return 0
        if args.command == "finalize":
            document = finalize_deployment(
                plan_file=args.plan_file,
                status=args.status,
                stage=args.stage,
                eif_root=args.eif_root,
            )
            print(document["status"])
            return 0
    except (GatewayGitDeployError, KeyError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
