"""Independently rebuild the gateway EIF from a clean Git commit.

This verifier is intentionally operator-run. It never restarts services, prunes
shared Docker state, or changes validator weights. A cache entry is written only
after repeated clean builds produce the same image ID and PCR0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping, Optional, Sequence


CACHE_SCHEMA_VERSION = "leadpoet.gateway_pcr0_cache.v1"
DEFAULT_CACHE_SIZE = 20
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")


class GatewayPCR0BuildError(RuntimeError):
    """Raised when an independent gateway build is incomplete or divergent."""


def _run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: int = 3600,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def resolve_commit(repo_root: Path, revision: str) -> str:
    result = _run(["git", "rev-parse", "%s^{commit}" % revision], cwd=repo_root, timeout=30)
    commit = result.stdout.strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayPCR0BuildError("git did not resolve a full commit")
    return commit


def _safe_extract_git_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive_path), "r:") as archive:
        members = archive.getmembers()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise GatewayPCR0BuildError("git archive contains an unsafe path")
            if member.issym() or member.islnk() or member.isdev():
                raise GatewayPCR0BuildError("git archive contains a non-regular entry")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                os.chmod(str(target), member.mode & 0o777)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise GatewayPCR0BuildError("git archive file content is missing")
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(str(target), member.mode & 0o777)


def extract_clean_commit(*, repo_root: Path, commit: str, destination: Path) -> None:
    archive_path = destination.parent / ("%s.tar" % commit[:12])
    if destination.exists():
        shutil.rmtree(destination)
    if archive_path.exists():
        archive_path.unlink()
    _run(
        ["git", "archive", "--format=tar", "--output", str(archive_path), commit],
        cwd=repo_root,
        timeout=120,
    )
    try:
        _safe_extract_git_archive(archive_path, destination)
    finally:
        archive_path.unlink(missing_ok=True)


def source_manifest_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        kind = "d" if path.is_dir() else "f"
        mode = stat.S_IMODE(path.stat().st_mode)
        digest.update(("%s %04o %s\n" % (kind, mode, relative)).encode("utf-8"))
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _parse_measurement(output: str) -> str:
    candidates = []
    try:
        candidates.append(json.loads(output))
    except json.JSONDecodeError:
        pass
    for start in (index for index, character in enumerate(output) if character == "{"):
        try:
            candidates.append(json.loads(output[start:]))
            break
        except json.JSONDecodeError:
            continue
    for value in candidates:
        if not isinstance(value, Mapping):
            continue
        measurements = value.get("Measurements")
        if isinstance(measurements, Mapping):
            pcr0 = str(measurements.get("PCR0") or "").lower()
            if _PCR0_RE.fullmatch(pcr0) and pcr0 != "0" * 96:
                return pcr0
    raise GatewayPCR0BuildError("nitro-cli output did not contain a valid PCR0")


def _build_once(*, source_root: Path, commit: str, work_root: Path, index: int) -> dict[str, Any]:
    gateway_root = source_root / "gateway"
    env = dict(os.environ)
    env.update(
        {
            "GATEWAY_ROOT": str(gateway_root),
            "RESEARCH_LAB_RUNTIME_SOURCE_ROOT": str(source_root),
            "ATTESTED_RUNTIME_COMMIT_SHA": commit,
            "ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE": "1",
        }
    )
    _run(["bash", str(gateway_root / "tee" / "stage_attested_runtime.sh")], env=env, timeout=900)
    context_root = gateway_root / "_enclave_source"
    context_hash = source_manifest_hash(context_root)

    image = "leadpoet-gateway-verify:%s-%s" % (commit[:12], index)
    _run(["docker", "rmi", "-f", image], check=False, timeout=120)
    _run(
        [
            "docker",
            "build",
            "--pull",
            "--no-cache",
            "-f",
            str(gateway_root / "tee" / "Dockerfile.enclave"),
            "-t",
            image,
            str(gateway_root),
        ],
        timeout=3600,
    )
    image_id = _run(
        ["docker", "image", "inspect", "-f", "{{.Id}}", image],
        timeout=120,
    ).stdout.strip()
    if not image_id.startswith("sha256:"):
        raise GatewayPCR0BuildError("gateway image ID is invalid")

    eif_path = work_root / ("gateway-%s-%s.eif" % (commit[:12], index))
    eif_path.unlink(missing_ok=True)
    result = _run(
        [
            "nitro-cli",
            "build-enclave",
            "--docker-uri",
            image,
            "--output-file",
            str(eif_path),
        ],
        timeout=1800,
    )
    pcr0 = _parse_measurement(result.stdout)
    eif_hash = "sha256:" + hashlib.sha256(eif_path.read_bytes()).hexdigest()
    _run(["docker", "rmi", "-f", image], check=False, timeout=120)
    return {
        "commit_sha": commit,
        "pcr0": pcr0,
        "image_id": image_id,
        "eif_sha256": eif_hash,
        "source_manifest_hash": context_hash,
    }


def build_reproducible_gateway_pcr0(
    *,
    repo_root: Path,
    revision: str,
    work_root: Path,
    repetitions: int = 3,
) -> dict[str, Any]:
    if repetitions < 3:
        raise GatewayPCR0BuildError("at least three independent builds are required")
    commit = resolve_commit(repo_root, revision)
    work_root.mkdir(parents=True, exist_ok=True)
    source_root = work_root / "source"
    extract_clean_commit(repo_root=repo_root, commit=commit, destination=source_root)
    results = [
        _build_once(
            source_root=source_root,
            commit=commit,
            work_root=work_root,
            index=index + 1,
        )
        for index in range(repetitions)
    ]
    for field in ("pcr0", "image_id", "eif_sha256", "source_manifest_hash"):
        values = {result[field] for result in results}
        if len(values) != 1:
            raise GatewayPCR0BuildError("gateway builds diverged at %s" % field)
    return {
        "role": "gateway_scoring",
        "commit_sha": commit,
        "pcr0": results[0]["pcr0"],
        "image_id": results[0]["image_id"],
        "eif_sha256": results[0]["eif_sha256"],
        "source_manifest_hash": results[0]["source_manifest_hash"],
        "verified_build_count": repetitions,
    }


def write_cache_entry(
    *,
    cache_path: Path,
    entry: Mapping[str, Any],
    cache_size: int = DEFAULT_CACHE_SIZE,
    pin: bool = False,
) -> None:
    try:
        current = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "entries": [],
            "pinned_commit_shas": [],
        }
    if current.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(current.get("entries"), list):
        raise GatewayPCR0BuildError("gateway PCR0 cache schema is invalid")
    commit = str(entry.get("commit_sha") or "")
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayPCR0BuildError("gateway PCR0 cache entry commit is invalid")
    limit = max(1, int(cache_size))
    pinned_commits = [
        str(value)
        for value in current.get("pinned_commit_shas", [])
        if _COMMIT_RE.fullmatch(str(value))
    ]
    if pin and commit not in pinned_commits:
        pinned_commits.insert(0, commit)
    entries = [dict(entry)] + [row for row in current["entries"] if row.get("commit_sha") != commit]
    pinned_entries = [row for row in entries if row.get("commit_sha") in pinned_commits]
    if len(pinned_entries) > limit:
        raise GatewayPCR0BuildError("pinned gateway commits exceed cache capacity")
    unpinned_entries = [row for row in entries if row.get("commit_sha") not in pinned_commits]
    retained_entries = pinned_entries + unpinned_entries[: limit - len(pinned_entries)]
    retained_commits = {str(row.get("commit_sha") or "") for row in retained_entries}
    document = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "entries": retained_entries,
        "pinned_commit_shas": [
            commit_sha for commit_sha in pinned_commits if commit_sha in retained_commits
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(cache_path))


def load_cached_gateway_identity(cache_path: Path, commit_sha: str) -> Optional[dict[str, Any]]:
    try:
        document = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    for entry in document.get("entries", []):
        if entry.get("commit_sha") == commit_sha and int(entry.get("verified_build_count") or 0) >= 3:
            pcr0 = str(entry.get("pcr0") or "")
            if _PCR0_RE.fullmatch(pcr0) and pcr0 != "0" * 96:
                return dict(entry)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path.home() / ".cache" / "leadpoet" / "gateway-pcr0-work",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path.home() / ".cache" / "leadpoet" / "gateway-pcr0-cache.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--pin",
        action="store_true",
        help="retain this deployed commit when newer cache entries are added",
    )
    args = parser.parse_args(argv)
    entry = build_reproducible_gateway_pcr0(
        repo_root=args.repo_root.resolve(),
        revision=args.revision,
        work_root=args.work_root.resolve(),
        repetitions=args.repetitions,
    )
    write_cache_entry(cache_path=args.cache_file, entry=entry, pin=args.pin)
    print(json.dumps(entry, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
