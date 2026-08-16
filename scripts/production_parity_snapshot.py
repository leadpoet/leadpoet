#!/usr/bin/env python3
"""Capture, verify, and restore encrypted production-parity snapshots safely."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity import (  # noqa: E402
    MIGRATION_RE,
    SNAPSHOT_SCHEMA_VERSION,
    ProductionParityError,
    file_sha256,
    migration_sequence,
    migration_delta,
    production_database_host_hash,
    safe_database_target,
    sha256_bytes,
    sha256_json,
    validate_archive,
    validate_contract,
    validate_snapshot_manifest,
)


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionParityError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise ProductionParityError(f"{description} must be an object")
    return value


def _postgres_env(dsn: str, *, read_only: bool) -> tuple[dict[str, str], str]:
    parsed = urlparse(str(dsn or ""))
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ProductionParityError("PostgreSQL DSN is invalid")
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ProductionParityError("PostgreSQL DSN has no database")
    env = os.environ.copy()
    env.update(
        {
            "PGHOST": parsed.hostname,
            "PGPORT": str(parsed.port or 5432),
            "PGDATABASE": database,
            "PGUSER": unquote(parsed.username or ""),
            "PGPASSWORD": unquote(parsed.password or ""),
            "PGSSLMODE": "require",
        }
    )
    if read_only:
        env["PGOPTIONS"] = (
            "-c default_transaction_read_only=on "
            "-c statement_timeout=300000 "
            "-c lock_timeout=5000"
        )
    return env, parsed.hostname.lower()


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: int,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env),
        input=stdin,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _require_success(result: subprocess.CompletedProcess[bytes], *, stage: str) -> bytes:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()[-800:]
        raise ProductionParityError(f"{stage} failed: {detail}")
    return result.stdout


def _database_stats(env: Mapping[str, str]) -> dict[str, Any]:
    stats_sql = """
SELECT json_build_object(
  'server_version_num', current_setting('server_version_num'),
  'relation_count', COUNT(*),
  'total_relation_bytes', COALESCE(SUM(pg_total_relation_size(c.oid)), 0),
  'largest_relation_bytes', COALESCE(MAX(pg_total_relation_size(c.oid)), 0),
  'capture_utc_timestamp', (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::text || '+00:00',
  'capture_utc_date', (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date::text,
  'latest_completed_benchmark_date', (
    SELECT MAX(benchmark_date)::text
    FROM public.research_lab_private_model_benchmark_bundles
  ),
  'current_day_rebenchmark_run_count', (
    SELECT COUNT(*)
    FROM public.research_lab_scoring_runs
    WHERE run_type = 'private_baseline_rebenchmark'
      AND benchmark_date = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date
  ),
  'current_day_benchmark_bundle_count', (
    SELECT COUNT(*)
    FROM public.research_lab_private_model_benchmark_bundles
    WHERE benchmark_date = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date
  ),
  'weight_history_scope', (
    SELECT json_build_object(
      'netuid', netuid,
      'start_epoch', MIN(epoch_id),
      'end_epoch', MAX(epoch_id),
      'expected_rows', COUNT(*)
    )
    FROM public.research_lab_finalized_allocation_epochs_v2
    GROUP BY netuid
    ORDER BY COUNT(*) DESC, netuid
    LIMIT 1
  ),
  'source_role', (
    SELECT json_build_object(
      'role_name', rolname,
      'transaction_read_only', current_setting('transaction_read_only') = 'on',
      'superuser', rolsuper,
      'bypass_rls', rolbypassrls,
      'replication', rolreplication,
      'table_write_capable', EXISTS (
        SELECT 1
        FROM pg_class AS writable_class
        JOIN pg_namespace AS writable_namespace
          ON writable_namespace.oid = writable_class.relnamespace
        WHERE writable_namespace.nspname = 'public'
          AND writable_class.relkind IN ('r', 'p')
          AND has_table_privilege(
            current_user,
            writable_class.oid,
            'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER'
          )
      )
    )
    FROM pg_roles
    WHERE rolname = current_user
  )
)::text
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'm')
  AND n.nspname = 'public';
"""
    raw = _require_success(
        _run(
            ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", stats_sql],
            env=env,
            timeout=60,
        ),
        stage="production database shape read",
    ).decode("utf-8", "replace").strip()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ProductionParityError(
            "production database shape response is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ProductionParityError(
            "production database shape response is not an object"
        )
    source_role = value.get("source_role")
    if isinstance(source_role, Mapping):
        role_name = str(source_role.pop("role_name", ""))
        source_role["role_hash"] = sha256_json({"role": role_name})
    return value


def _target_rebenchmark_date(stats: Mapping[str, Any]) -> date:
    try:
        captured = datetime.fromisoformat(
            str(stats.get("capture_utc_timestamp") or "")
        ).astimezone(timezone.utc)
    except ValueError as exc:
        raise ProductionParityError("production snapshot clock is invalid") from exc
    # The clone executes tomorrow's normal production workflow. Candidate
    # code creates that date's ICP set itself, so the test never deletes,
    # rewrites, or reuses a consumed production daily slot.
    return captured.date() + timedelta(days=1)


def _git(
    root: Path,
    *args: str,
    timeout: int = 60,
) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return _require_success(result, stage="production snapshot Git identity")


def _source_migrations(
    *, root: Path, source_sha: str, candidate_sha: str
) -> list[dict[str, Any]]:
    source = str(source_sha or "").strip().lower()
    if not _SHA_RE.fullmatch(source):
        raise ProductionParityError("snapshot producer runtime SHA is invalid")
    resolved = _git(root, "rev-parse", f"{source}^{{commit}}").decode().strip()
    if resolved != source:
        raise ProductionParityError("snapshot producer runtime SHA is unavailable")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, candidate_sha],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise ProductionParityError(
            "snapshot producer runtime is not an ancestor of capture code"
        )
    tracked = _git(root, "ls-tree", "-r", "--name-only", source).decode(
        "utf-8", "strict"
    )
    migrations: list[dict[str, Any]] = []
    for path in sorted(line.strip() for line in tracked.splitlines() if line.strip()):
        if MIGRATION_RE.fullmatch(path) is None:
            continue
        sequence, _name = migration_sequence(path)
        payload = _git(root, "show", f"{source}:{path}")
        migrations.append(
            {
                "path": path,
                "sequence": sequence,
                "sha256": sha256_bytes(payload),
                "transaction_mode": (
                    "autocommit"
                    if path.endswith(".concurrent.sql")
                    else "candidate-file"
                ),
            }
        )
    if not migrations:
        raise ProductionParityError(
            "snapshot producer runtime migration inventory is empty"
        )
    return sorted(migrations, key=lambda item: (item["sequence"], item["path"]))


def capture_snapshot(
    *,
    contract_path: Path,
    archive_path: Path,
    manifest_path: Path,
    dsn: str,
    expected_production_host: str,
    ttl_hours: int,
    source_sha: str,
    capture_mode: str = "full",
) -> dict[str, Any]:
    contract = validate_contract(_load_json(contract_path, description="parity contract"))
    env, observed_host = _postgres_env(dsn, read_only=True)
    if observed_host != str(expected_production_host or "").strip().lower():
        raise ProductionParityError("snapshot source is not the expected production database")
    if ttl_hours < 1 or ttl_hours > 48:
        raise ProductionParityError("snapshot TTL must be between 1 and 48 hours")
    if capture_mode not in {"full", "schema-only"}:
        raise ProductionParityError("production snapshot capture mode is invalid")

    read_only = _require_success(
        _run(
            ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", "SHOW transaction_read_only"],
            env=env,
            timeout=30,
        ),
        stage="production read-only transaction check",
    ).decode("utf-8", "replace").strip()
    if read_only != "on":
        raise ProductionParityError("production snapshot session is not read-only")

    stats = _database_stats(env)
    source_role = stats.get("source_role")
    if (
        not isinstance(source_role, Mapping)
        or source_role.get("transaction_read_only") is not True
        or source_role.get("superuser") is not False
        or not isinstance(source_role.get("bypass_rls"), bool)
        or source_role.get("replication") is not False
        or source_role.get("table_write_capable") is not False
    ):
        raise ProductionParityError(
            "production snapshot credential is not a dedicated read-only role"
        )
    target_rebenchmark_date = _target_rebenchmark_date(stats)
    source_migrations = _source_migrations(
        root=ROOT,
        source_sha=source_sha,
        candidate_sha=contract["candidate_sha"],
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    dump_command = [
            "pg_dump",
            "--format=custom",
            "--compress=6",
            "--schema=public",
            "--no-owner",
            "--no-acl",
            "--serializable-deferrable",
            "--file",
            str(archive_path),
        ]
    if capture_mode == "schema-only":
        dump_command.insert(4, "--schema-only")
    result = _run(
        dump_command,
        env=env,
        timeout=900,
    )
    _require_success(result, stage="read-only production snapshot capture")
    post_stats = _database_stats(env)
    if (
        str(post_stats.get("capture_utc_date") or "")
        != str(stats.get("capture_utc_date") or "")
        or _target_rebenchmark_date(post_stats) != target_rebenchmark_date
    ):
        archive_path.unlink(missing_ok=True)
        raise ProductionParityError(
            "production snapshot crossed its target-day consistency boundary"
        )
    captured_at = datetime.fromisoformat(
        str(stats.get("capture_utc_timestamp") or "")
    ).astimezone(timezone.utc)
    body = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_environment": "production-read-only",
        "source_host_hash": production_database_host_hash(observed_host),
        "capture_sha": contract["candidate_sha"],
        "capture_contract_hash": contract["contract_hash"],
        "source_sha": str(source_sha).lower(),
        "captured_at": captured_at.isoformat(),
        "expires_at": (captured_at + timedelta(hours=ttl_hours)).isoformat(),
        "capture_transaction_read_only": True,
        "capture_mode": capture_mode,
        "archive": {
            "format": (
                "postgres-custom"
                if capture_mode == "full"
                else "postgres-schema-custom"
            ),
            "storage": "ephemeral-encrypted-volume",
            "persisted": False,
            "sha256": file_sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
        "database": {
            "server_version_num": str(stats.get("server_version_num") or ""),
            "relation_count": int(stats.get("relation_count") or 0),
            "total_relation_bytes": int(stats.get("total_relation_bytes") or 0),
            "largest_relation_bytes": int(stats.get("largest_relation_bytes") or 0),
            "capture_utc_date": str(stats.get("capture_utc_date") or ""),
            "target_rebenchmark_date": target_rebenchmark_date.isoformat(),
            "latest_completed_benchmark_date": stats.get(
                "latest_completed_benchmark_date"
            ),
            "current_day_rebenchmark_run_count": int(
                stats.get("current_day_rebenchmark_run_count") or 0
            ),
            "current_day_benchmark_bundle_count": int(
                stats.get("current_day_benchmark_bundle_count") or 0
            ),
            "source_role": dict(source_role),
            "weight_history_scope": dict(stats.get("weight_history_scope") or {}),
        },
        "migrations": source_migrations,
        "data_classification": "production-confidential-ephemeral",
    }
    manifest = validate_snapshot_manifest(
        {**body, "manifest_hash": sha256_json(body)}, now=captured_at
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o600)
    archive_path.chmod(0o600)
    return manifest


def verify_snapshot(
    *,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    expected_production_host: str | None = None,
) -> dict[str, Any]:
    contract = validate_contract(_load_json(contract_path, description="parity contract"))
    manifest = validate_snapshot_manifest(
        _load_json(manifest_path, description="snapshot manifest")
    )
    if (
        expected_production_host is not None
        and manifest["source_host_hash"]
        != production_database_host_hash(expected_production_host)
    ):
        raise ProductionParityError(
            "snapshot source host differs from the configured production database"
        )
    validate_archive(archive_path, manifest)
    delta = migration_delta(
        snapshot_migrations=manifest["migrations"],
        candidate_migrations=contract["migrations"],
    )
    listing = subprocess.run(
        ["pg_restore", "--list", str(archive_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if listing.returncode != 0 or not listing.stdout.strip():
        raise ProductionParityError("snapshot archive is not a readable PostgreSQL custom dump")
    return {
        "manifest_hash": manifest["manifest_hash"],
        "archive_hash": manifest["archive"]["sha256"],
        "source_sha": manifest["source_sha"],
        "source_host_hash": manifest["source_host_hash"],
        "candidate_sha": contract["candidate_sha"],
        "migration_delta": delta,
        "archive_entries": len(listing.stdout.splitlines()),
    }


def restore_snapshot(
    *,
    root: Path,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    target_dsn: str,
    production_host: str,
) -> dict[str, Any]:
    evidence = verify_snapshot(
        contract_path=contract_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        expected_production_host=production_host,
    )
    safe_database_target(target_dsn, production_host=production_host)
    env, _ = _postgres_env(target_dsn, read_only=False)
    _require_success(
        _run(
            [
                "pg_restore",
                "--dbname",
                env["PGDATABASE"],
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-acl",
                "--exit-on-error",
                "--jobs=4",
                str(archive_path),
            ],
            env=env,
            timeout=900,
        ),
        stage="isolated production snapshot restore",
    )
    for migration in evidence["migration_delta"]:
        path = root / str(migration["path"])
        if not path.is_file() or file_sha256(path) != migration["sha256"]:
            raise ProductionParityError(
                f"candidate migration bytes differ: {migration['path']}"
            )
        _require_success(
            _run(
                ["psql", "-X", "-v", "ON_ERROR_STOP=1", "-f", str(path)],
                env=env,
                timeout=300,
            ),
            stage=f"candidate migration {migration['path']}",
        )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--contract", type=Path, required=True)
    capture.add_argument("--archive", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--dsn-env", default="LEADPOET_PARITY_PRODUCTION_READONLY_DSN")
    capture.add_argument("--expected-host-env", default="LEADPOET_PARITY_PRODUCTION_DB_HOST")
    capture.add_argument("--ttl-hours", type=int, default=24)
    capture.add_argument("--source-sha", required=True)
    capture.add_argument(
        "--mode", choices=("full", "schema-only"), default="full"
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--root", type=Path, default=ROOT)
    restore.add_argument("--contract", type=Path, required=True)
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--target-dsn-env", default="LEADPOET_PARITY_TARGET_DSN")
    restore.add_argument("--production-host-env", default="LEADPOET_PARITY_PRODUCTION_DB_HOST")

    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            dsn = os.environ.get(args.dsn_env, "")
            host = os.environ.get(args.expected_host_env, "")
            if not dsn or not host:
                raise ProductionParityError("production snapshot source environment is incomplete")
            result = capture_snapshot(
                contract_path=args.contract,
                archive_path=args.archive,
                manifest_path=args.manifest,
                dsn=dsn,
                expected_production_host=host,
                ttl_hours=args.ttl_hours,
                source_sha=args.source_sha,
                capture_mode=args.mode,
            )
        elif args.command == "verify":
            result = verify_snapshot(
                contract_path=args.contract,
                manifest_path=args.manifest,
                archive_path=args.archive,
            )
        else:
            target_dsn = os.environ.get(args.target_dsn_env, "")
            production_host = os.environ.get(args.production_host_env, "")
            if not target_dsn or not production_host:
                raise ProductionParityError("snapshot restore environment is incomplete")
            result = restore_snapshot(
                root=args.root,
                contract_path=args.contract,
                manifest_path=args.manifest,
                archive_path=args.archive,
                target_dsn=target_dsn,
                production_host=production_host,
            )
    except (OSError, ValueError, ProductionParityError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
